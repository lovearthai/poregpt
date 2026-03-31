# -*- coding: utf-8 -*-
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import argparse
from accelerate import Accelerator

# 假设你的模型和数据集类在以下路径，请根据实际情况修改
# from cnn_model import NanoporeCNNModel 
# from dataset import NanoporeSignalDataset

def load_trained_cnn(checkpoint_path: str, cnn_type: int):
    """加载预训练模型权重"""
    # 这里需要确保 NanoporeCNNModel 已经定义
    from cnn_model import NanoporeCNNModel 
    model = NanoporeCNNModel(cnn_type=cnn_type)

    # weights_only=False 是为了兼容旧版本保存的完整 state_dict
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    # 兼容 DDP 训练保存的权重 (去除 'module.' 前缀)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model

def cnn_eval_distributed(
    input_shards_dir: str,
    output_shard_dir: str,
    checkpoint_path: str,
    shard_size: int = 1_000_000,
    feature_dim: int = 512, # 注意：如果你要统计512维，这里填512
    batch_size: int = 512,  # 提高 Batch 以榨干多卡性能
    num_workers: int = 8,
    cnn_type: int = 1,
):
    # === 1. 初始化 Accelerator ===
    # 使用 fp16 可以大幅减少 .npy 写入磁盘的 IO 压力和显存占用
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    rank = accelerator.process_index
    
    if accelerator.is_main_process:
        print(f"🌟 开始多卡推理任务 | 总卡数: {accelerator.num_processes}")
        os.makedirs(output_shard_dir, exist_ok=True)

    # === 2. 创建各进程私有目录 ===
    process_output_dir = os.path.join(output_shard_dir, f"rank_{rank:03d}")
    os.makedirs(process_output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # === 3. 数据加载 ===
    from dataset import NanoporeSignalDataset
    dataset = NanoporeSignalDataset(shards_dir=input_shards_dir)
    
    # DistributedSampler 确保每张卡分到的数据不重复
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        prefetch_factor=2
    )

    # === 4. 模型准备 ===
    model = load_trained_cnn(checkpoint_path, cnn_type)
    model, dataloader = accelerator.prepare(model, dataloader)

    # === 5. 探测 T (时间步长) ===
    model.eval()
    with torch.no_grad():
        # 拿到第一个 batch 探测输出形状
        first_batch = next(iter(dataloader))
        # 使用 unwrap 确保能访问到你自定义的 .encoder
        sample_feat = accelerator.unwrap_model(model).encoder(first_batch[:1])
        T = sample_feat.shape[2]
        current_c = sample_feat.shape[1]
        
        if accelerator.is_main_process:
            print(f"📊 探测到特征维度: {current_c}, 时间步长 T: {T}")

    # === 6. 主推理循环 ===
    buffer = []
    local_shard_idx = 0
    local_shards_meta = []
    
    # 进度条仅在主进程显示
    pbar = tqdm(dataloader, desc=f"GPU {rank} 推理", disable=not accelerator.is_local_main_process)

    with torch.no_grad():
        for batch in pbar:
            # 推理输出 [B, 512, T]
            feats = accelerator.unwrap_model(model).encoder(batch) 
            
            # 转换为 [B*T, C]，并转为 float32 (npy 存储标准)
            # 使用 .half() 或 .float() 取决于你对精度的要求
            feats = feats.permute(0, 2, 1).reshape(-1, current_c).cpu().numpy()
            
            buffer.append(feats)
            # 检查 buffer 中的总 token 数
            if sum(len(b) for b in buffer) >= shard_size:
                shard_data = np.concatenate(buffer, axis=0)
                shard_file_name = f"shard_{local_shard_idx:05d}.npy"
                shard_path = os.path.join(process_output_dir, shard_file_name)
                
                np.save(shard_path, shard_data)
                
                # 记录相对于 output_shard_dir 的相对路径
                local_shards_meta.append({
                    "shard_file": os.path.join(f"rank_{rank:03d}", shard_file_name),
                    "num_tokens": int(shard_data.shape[0])
                })
                
                buffer = []
                local_shard_idx += 1

        # 处理最后不足一个 shard 的数据
        if buffer:
            shard_data = np.concatenate(buffer, axis=0)
            shard_file_name = "shard_last.npy"
            shard_path = os.path.join(process_output_dir, shard_file_name)
            np.save(shard_path, shard_data)
            local_shards_meta.append({
                "shard_file": os.path.join(f"rank_{rank:03d}", shard_file_name),
                "num_tokens": int(shard_data.shape[0])
            })

    # === 7. 汇总元数据 ===
    # gather_for_metrics 可以收集非 Tensor 对象，但通常建议收集 Tensor 以防万一
    # 这里我们直接 gather list
    all_ranks_meta = accelerator.gather_for_metrics(local_shards_meta)
    
    if accelerator.is_main_process:
        final_shards = []
        # 处理 gather 回来的嵌套列表
        for item in all_ranks_meta:
            if isinstance(item, list):
                final_shards.extend(item)
            else:
                final_shards.append(item)

        # 写入总索引文件
        summary = {
            "total_tokens": sum(m['num_tokens'] for m in final_shards),
            "feature_dim": current_c,
            "tokens_per_sample": T,
            "num_shards": len(final_shards),
            "shards": final_shards
        }
        
        with open(os.path.join(output_shard_dir, "shards.json"), 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"🚀 特征提取成功！总 Token 数: {summary['total_tokens']:,}")

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_shards_dir", type=str, required=True)
    parser.add_argument("--output_shard_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--shard_size", type=int, default=2_000_000)
    parser.add_argument("--cnn_type", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=16) # 多卡建议增加线程数
    
    args = parser.parse_args()
    cnn_eval_distributed(**vars(args))
