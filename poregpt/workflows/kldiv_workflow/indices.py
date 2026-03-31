import os
import sys
import json
import glob
import torch
import numpy as np
import pickle
import gc
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from accelerate import Accelerator

# --- 屏蔽 Triton (根据原代码需求) ---
sys.modules["triton"] = None 

def parse_args():
    parser = argparse.ArgumentParser(description="PoreGPT Codebook Indices Extraction Workflow")
    
    # 路径配置
    parser.add_argument("--ckpt_root", type=str, required=True, help="模型 Checkpoint 根目录 (.pth 文件夹所在位置)")
    parser.add_argument("--val_data_dir", type=str, required=True, help="验证集数据目录")
    parser.add_argument("--save_dir", type=str, required=True, help="输出保存目录")

    
    # 模型与数据参数
    parser.add_argument("--codebook_size", type=int, default=65536, help="Codebook 大小")
    parser.add_argument("--cnn_type", type=int, default=7, help="CNN 模型类型")
    parser.add_argument("--batch_size", type=int, default=8, help="推断 Batch Size")
    parser.add_argument("--sample_ratio", type=float, default=0.01, help="数据采样比例 (0.0 到 1.0)")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    
    # 逻辑开关
    parser.add_argument("--use_local_subset", type=bool , default=False, help="是否加载本地已存在的索引文件")
    
    return parser.parse_args()

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        pass

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def safe_load_checkpoint(checkpoint_dir, accelerator: Accelerator):
    meta_path = os.path.join(checkpoint_dir, "metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    try:
        accelerator.load_state(checkpoint_dir)
    except Exception as e:
        print(f"⚠️ 加载失败: {e}")
        raise e
    return metadata

def main():
    args = parse_args()
    accelerator = Accelerator()
    
    # --- 1. 环境与目录初始化 ---
    save_dir = args.save_dir
    log_dir = os.path.join(save_dir, "log")
    data_dir = os.path.join(save_dir, "data")
    indices_save_dir = os.path.join(save_dir, "indices")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_indices_dir = os.path.join(indices_save_dir, timestamp)
    log_path = os.path.join(log_dir, f"{timestamp}.log")

    if accelerator.is_main_process:
        for d in [log_dir, data_dir, indices_save_dir, current_run_indices_dir]:
            ensure_dir(d)
        sys.stdout = Logger(log_path)
        sys.stderr = Logger(log_path)

    # 导入自定义模块路径
    abs_project_root = "/mnt/"
    if abs_project_root not in sys.path:
        sys.path.insert(0, abs_project_root)

    from si003067jezr.default.poregpt.poregpt.poregpt.tokenizers.vqe_tokenizer.vqe_model_v3 import NanoporeVQEModel_V3
    from si003067jezr.default.poregpt.poregpt.poregpt.tokenizers.vqe_tokenizer.dataset import NanoporeSignalDataset

    # --- 2. 数据准备 ---
    full_dataset = NanoporeSignalDataset(shards_dir=args.val_data_dir, logic_chunk_size=6000)
    
    if not args.use_local_subset:
        existing_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not existing_files:
            raise FileNotFoundError("❌ 未找到本地索引文件！")
        load_path = existing_files[-1]
        indices = np.load(load_path)
        if accelerator.is_main_process:
            print(f"📂 加载本地索引: {load_path}")
    else:
        np.random.seed(args.random_seed)
        num_samples = len(full_dataset)
        subset_size = int(num_samples * args.sample_ratio)
        indices = np.random.choice(num_samples, subset_size, replace=False)
        if accelerator.is_main_process:
            subset_indices_path = os.path.join(data_dir, f"subset_{timestamp}.npy")
            np.save(subset_indices_path, indices)
            print(f"📦 抽取样本数: {subset_size} | 比例: {args.sample_ratio}")

    val_dataset = Subset(full_dataset, indices)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = accelerator.prepare(val_loader)

    # 报表初始化
    results_path = os.path.join(save_dir, "usage_ratio_report.csv")
    if accelerator.is_main_process and not os.path.exists(results_path):
        with open(results_path, "w") as f:
            f.write("checkpoint,epoch,step,usage_ratio\n")

    # --- 3. 循环处理 Checkpoints ---
    ckpt_dirs = sorted([d for d in glob.glob(os.path.join(args.ckpt_root, "*.pth")) if os.path.isdir(d)])
    if not ckpt_dirs:
        print(f"❌ 在 {args.ckpt_root} 未找到 checkpoint 目录")
        return

    for ckpt_dir in ckpt_dirs:
        ckpt_name = os.path.basename(ckpt_dir)
        save_pkl_path = os.path.join(current_run_indices_dir, f"{ckpt_name}.pkl")
        
        try:
            # 实例化模型
            model = NanoporeVQEModel_V3(codebook_size=args.codebook_size, cnn_type=args.cnn_type)
            model = accelerator.prepare(model)
            metadata = safe_load_checkpoint(ckpt_dir, accelerator)
            model.eval()

            used_codes = set()
            f_out = open(save_pkl_path, "wb") if accelerator.is_main_process else None

            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"Eval {ckpt_name[:15]}", disable=not accelerator.is_local_main_process)
                for batch_id, batch in enumerate(pbar):
                    x = batch if torch.is_tensor(batch) else batch[0]
                    outputs = model(x)
                    indices_out = outputs[1]  # [B, L]
                    
                    indices_np = indices_out.cpu().numpy()
                    
                    if accelerator.is_main_process:
                        pickle.dump({"batch_id": batch_id, "indices": indices_np}, f_out)
                        used_codes.update(np.unique(indices_np))
                    
                    del outputs, indices_out, indices_np

            if f_out: f_out.close()

            # 保存报表
            if accelerator.is_main_process:
                usage_ratio = len(used_codes) / args.codebook_size
                epoch = metadata.get('epoch', 'N/A')
                step = metadata.get('global_step', 'N/A')
                print(f"✅ {ckpt_name} | Usage: {usage_ratio:.4%}")
                with open(results_path, "a") as f:
                    f.write(f"{ckpt_name},{epoch},{step},{usage_ratio}\n")

            # --- 4. 显存清理 ---
            del model
            accelerator._models = [] 
            torch.cuda.empty_cache()
            gc.collect() 

        except Exception as e:
            print(f"❌ {ckpt_name} 运行出错: {e}")
            if 'f_out' in locals() and f_out: f_out.close()

if __name__ == "__main__":
    main()
