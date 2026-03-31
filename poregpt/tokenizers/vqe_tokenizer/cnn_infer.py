# -*- coding: utf-8 -*-
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import argparse
from accelerate import Accelerator

# 导入本地模块
# from .cnn_model import NanoporeCNNModel  # 确保路径正确
# from .dataset import NanoporeSignalDataset

def cnn_eval_distributed(
    input_shards_dir: str,
    output_shard_dir: str,
    checkpoint_path: str,
    shard_size: int = 1_000_000,
    feature_dim: int = 64,
    batch_size: int = 128,
    num_workers: int = 8,
    cnn_type: int = 1,
):
    # === 1. 初始化 Accelerator ===
    # 推理建议使用 fp16/bf16 加速，显存占用减半，速度翻倍
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    rank = accelerator.process_index
    num_processes = accelerator.num_processes

    # === 2. 创建各进程私有目录 (并行写入的关键) ===
    # 每个进程写到自己的文件夹里，互不干扰，完全避免 gather 导致的通信延迟
    process_output_dir = os.path.join(output_shard_dir, f"rank_{rank:03d}")
    os.makedirs(process_output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # === 3. 数据加载 (DistributedSampler 自动切分数据) ===
    dataset = NanoporeSignalDataset(shards_dir=input_shards_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        # 重要：DistributedSampler 保证每个 GPU 拿到的数据不重叠
        sampler=DistributedSampler(dataset, shuffle=False) 
    )

    # === 4. 模型准备 ===
    # 注意：这里调用你之前的 load_trained_cnn 函数
    model = load_trained_cnn(checkpoint_path, cnn_type)
    model, dataloader = accelerator.prepare(model, dataloader)

    # === 5. 探测 T (仅为了元数据统计) ===
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        # 根据你的模型结构，可能是 .module.encoder(sample_batch)
        sample_feat = accelerator.unwrap_model(model).encoder(sample_batch[:1])
        T = sample_feat.shape[2]

    # === 6. 推理与本地写入循环 ===
    buffer = []
    local_shard_idx = 0
    local_token_count = 0
    local_shards_meta = []

    pbar = tqdm(dataloader, desc=f"Rank {rank} 推理中", disable=not accelerator.is_local_main_process)

    with torch.no_grad():
        for batch in pbar:
            # 经过 accelerator.prepare 后，模型会自动处理 .to(device)
            # 直接调用 encoder 得到 512 维特征
            feats = accelerator.unwrap_model(model).encoder(batch) # [B, 512, T]
            
            # 转换为 [B*T, Feature_Dim]
            feats = feats.permute(0, 2, 1).reshape(-1, feature_dim).cpu().numpy()
            
            buffer.append(feats)
            current_buffer_size = sum(f.shape[0] for f in buffer)

            # 当本地 buffer 达到分片阈值时写入磁盘
            if current_buffer_size >= shard_size:
                shard_data = np.concatenate(buffer, axis=0)
                shard_file = f"rank_{rank:03d}_shard_{local_shard_idx:05d}.npy"
                shard_path = os.path.join(process_output_dir, shard_file)
                
                np.save(shard_path, shard_data)
                
                local_shards_meta.append({
                    "shard_file": os.path.join(f"rank_{rank:03d}", shard_file),
                    "num_tokens": shard_data.shape[0]
                })
                
                local_token_count += shard_data.shape[0]
                buffer = []
                local_shard_idx += 1

        # 处理剩余 buffer
        if buffer:
            shard_data = np.concatenate(buffer, axis=0)
            shard_file = f"rank_{rank:03d}_shard_last.npy"
            shard_path = os.path.join(process_output_dir, shard_file)
            np.save(shard_path, shard_data)
            local_shards_meta.append({
                "shard_file": os.path.join(f"rank_{rank:03d}", shard_file),
                "num_tokens": shard_data.shape[0]
            })
            local_token_count += shard_data.shape[0]

    # === 7. 汇总元数据 (使用 gather 收集各进程的 meta) ===
    all_ranks_meta = accelerator.gather_for_metrics(local_shards_meta)
    
    if accelerator.is_main_process:
        # 展平收集到的列表
        final_shards = []
        # 注意：gather 后的结构可能需要根据具体版本微调
        for meta in all_ranks_meta:
            if isinstance(meta, list): final_shards.extend(meta)
            else: final_shards.append(meta)

        # 保存最终的汇总 json
        meta_path = os.path.join(output_shard_dir, "shards.json")
        with open(meta_path, 'w') as f:
            json.dump({
                "total_tokens": sum(m['num_tokens'] for m in final_shards),
                "feature_dim": feature_dim,
                "tokens_per_sample": T,
                "shards": final_shards
            }, f, indent=2)
        print(f"✅ 所有特征提取完成，汇总文件已保存至: {meta_path}")

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    # 使用分布式启动：accelerate launch cnn_eval.py --params...
    # 或者直接 python cnn_eval.py (由内部逻辑检测)
    cnn_eval_distributed(...)
