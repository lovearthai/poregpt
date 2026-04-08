import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import csv
import time
from typing import Optional
import yaml
import argparse
from datetime import timedelta
# 引入 accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

# 相对导入模块
from .dataset import NanoporeSignalDataset
from .cnn_model import NanoporeCNNModel

# ======================================================================================
# 辅助函数
# ======================================================================================
import os
import shutil

def save_accelerate_checkpoint(accelerator, output_dir, step, max_keep=100):
    """
    output_dir: 检查点根目录 (来自 config)
    step: 当前 global_step
    max_keep: 最多保留多少个最近的检查点
    """
    # 🚀 格式化 Step 为 8 位补 0: 例如 00000010
    step_str = f"{step:08d}"
    checkpoint_dir = os.path.join(output_dir, f"step-{step_str}")
    
    # 确保父目录存在
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    
    # 这里的 wait_for_everyone 确保所有卡都完成了计算再开始保存
    accelerator.wait_for_everyone()
    accelerator.save_state(checkpoint_dir)
    
    # 管理磁盘空间：只在主进程执行删除操作
    if accelerator.is_main_process:
        accelerator.print(f"✅ Checkpoint saved to {checkpoint_dir}")
        # 获取所有已存在的 checkpoint 目录
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("step-")]
        # 补 0 后，直接按字母顺序排序就是按数字大小排序
        checkpoints = sorted(checkpoints)
        # 如果超过数量限制，删除最早的
        if len(checkpoints) > max_keep:
            for i in range(len(checkpoints) - max_keep):
                old_checkpoint = os.path.join(output_dir, checkpoints[i])
                shutil.rmtree(old_checkpoint)
                accelerator.print(f"🗑️ Removed old checkpoint: {old_checkpoint}")


# ======================================================================================
# 主训练函数
# ======================================================================================
def cnn_train(
    npy_dir: str,
    output_model_path: str,
    device_micro_batch_size: int = 16,
    global_batch_size: int = 256,
    lr: float = 1e-4,
    num_epochs: int = 10,
    num_workers: int = 8,
    prefetch_factor: int = 128,
    val_ratio: float = 0.1,
    val_dataset_path: Optional[str] = None,
    do_evaluate: bool = True,
    loss_log_interval: int = 1,
    use_wandb: bool = True,
    wandb_project: str = "nanopore_cnn",
    wandb_name: str = "accelerate_cnn_run",
    lr_scheduler_type: str = "cosine",
    warmup_steps: int = 1000,
    checkpoint_path: Optional[str] = None,
    cnn_type: int = 1,
    mixed_precision: str = "no", # "no", "fp16", "bf16"
):
    # 1. 初始化 Accelerator
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    gradient_accumulation_steps = global_batch_size // (device_micro_batch_size * world_size)
    if gradient_accumulation_steps < 1:
        gradient_accumulation_steps = 1

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision, # 🚀 开启混合精度
        log_with="wandb" if use_wandb else None,
    )

    if accelerator.is_main_process:
        # 直接确保实验根目录存在
        os.makedirs(output_model_path, exist_ok=True)
        print(f"📂 Created output directory: {output_model_path}")

    if use_wandb:
        accelerator.init_trackers(
            project_name=wandb_project,
            config={
                "device_micro_batch_size": device_micro_batch_size,
                "global_batch_size": global_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "lr": lr,
                "num_epochs": num_epochs,
                "cnn_type": cnn_type,
                "mixed_precision": mixed_precision,
            },
            init_kwargs={"wandb": {"name": wandb_name}}
        )

    set_seed(42)

    # 2. 准备模型和优化器
    model = NanoporeCNNModel(cnn_type=cnn_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 3. 准备数据集
    dataset = NanoporeSignalDataset(shards_dir=npy_dir)
    train_dataloader = DataLoader(
        dataset,
        batch_size=device_micro_batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True
    )

    val_loader = None
    if do_evaluate:
        val_source_dir = val_dataset_path if (val_dataset_path and os.path.isdir(val_dataset_path)) else npy_dir
        full_val_dataset = NanoporeSignalDataset(shards_dir=val_source_dir)
        actual_val_size = max(1, int(val_ratio * len(full_val_dataset)))
        indices = np.random.choice(len(full_val_dataset), size=actual_val_size, replace=False)
        val_dataset = torch.utils.data.Subset(full_val_dataset, indices)
        val_loader = DataLoader(val_dataset, batch_size=device_micro_batch_size, shuffle=False, num_workers=4)

    # 4. 学习率调度器
    total_training_steps = (len(train_dataloader) // gradient_accumulation_steps) * num_epochs
    if lr_scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps)
    else:
        scheduler = None

    # 5. 使用 Accelerate 准备所有对象
    model, optimizer, train_dataloader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_loader, scheduler
    )

    # ======================================================================================
    # 🚀 新增：打印所有训练参数
    # ======================================================================================
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print(" 🚀 Nanopore CNN Training Configuration")
        print("="*60)
        config_summary = {
            "CNN Type": f"v{cnn_type}",
            "NPY Directory": npy_dir,
            "Mixed Precision": accelerator.mixed_precision,
            "Device Micro-Batch": device_micro_batch_size,
            "Global Batch Size": global_batch_size,
            "Grad Accum Steps": gradient_accumulation_steps,
            "World Size (GPUs)": accelerator.num_processes,
            "Learning Rate": lr,
            "Epochs": num_epochs,
            "LR Scheduler": lr_scheduler_type,
            "Warmup Steps": warmup_steps,
            "Output Path": output_model_path,
        }
        for k, v in config_summary.items():
            print(f"  {k:.<30} {v}")
        print("="*60 + "\n")

    # 6. 加载检查点
    if checkpoint_path and os.path.exists(checkpoint_path):
        accelerator.load_state(checkpoint_path)
        accelerator.print(f"📥 Resumed from checkpoint: {checkpoint_path}")

    # 7. 训练循环准备
    global_step = 0
    total_steps = (len(train_dataloader) // gradient_accumulation_steps) * num_epochs
    start_time = time.time() # 🚀 记录开始时间
    last_loss = None  # 🚀 用于记录上一次的 avg_loss
    # 8. 训练循环
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                x = batch # [B, 1, T]
                recon = model(x)
                loss = F.mse_loss(recon, x)

                accelerator.backward(loss)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % loss_log_interval == 0:
                    avg_loss = accelerator.gather_for_metrics(loss).mean().item()
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    # 🚀 计算 Loss Gain
                    # 如果是第一次记录，gain 为 0；否则为 上次 - 当前
                    loss_gain = (last_loss - avg_loss) if last_loss is not None else 0.0
                    last_loss = avg_loss # 更新 last_loss 供下次使用
                    # 🚀 计算时间逻辑
                    elapsed_time = time.time() - start_time
                    steps_per_sec = global_step / elapsed_time if elapsed_time > 0 else 0
                    remaining_steps = total_steps - global_step
                    eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                    
                    # 格式化时间字符串 HH:MM:SS
                    elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                    
                    # 🚀 组合打印信息
                    # 格式：[Epoch 1/10 | Step 10/1000 | 0:00:05<0:08:20] Loss: 0.1234 | Gain: 0.0012 | LR: 1.00e-04
                    progress_msg = (
                        f"[Epoch {epoch+1:3d}/{num_epochs} | "
                        f"Step {global_step:6d}/{total_steps} | "
                        f"{elapsed_str}<{eta_str}] "
                        f"Loss: {avg_loss:.6f} | "
                        f"Gain: {loss_gain:+.6f} | " # 使用 + 显示符号
                        f"LR: {current_lr:.2e}"
                    )
                    accelerator.print(progress_msg)
                    
                    if accelerator.is_main_process:
                        accelerator.log({"train/loss": avg_loss,"train/loss_gain": loss_gain, "lr": current_lr}, step=global_step)
                    # 在训练循环中定期清理
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        # 8. 验证
        if do_evaluate and val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    recon = model(batch)
                    loss = F.mse_loss(recon, batch)
                    # 🔧 修复：先gather再转为标量，避免tensor累积
                    gathered_loss = accelerator.gather_for_metrics(loss)
                    val_losses.append(gathered_loss.item())  # ✅ 只存储数值
            val_loss = torch.cat(val_losses).mean().item()
            accelerator.print(f"✅ Epoch {epoch+1} Val Loss: {val_loss:.6f}")
            if accelerator.is_main_process:
                accelerator.log({"val/loss": val_loss}, step=global_step)
        # 9. 保存状态
        accelerator.wait_for_everyone()
        save_accelerate_checkpoint(accelerator, output_model_path, global_step)

    # ---------------------------------------------------------
    # 9. 训练结束：保存最终模型 (纯权重，用于推理)
    # ---------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 第一步：解包模型（去掉分布式外壳）
        unwrapped_model = accelerator.unwrap_model(model)
        # 第二步：定义最终权重保存的【文件名】
        # 注意：由于 output_model_path 是目录，这里必须拼接文件名
        final_weights_path = os.path.join(output_model_path, "final_model_weights.pth")
        # 第三步：只保存 state_dict
        torch.save(unwrapped_model.state_dict(), final_weights_path)
        # 可选：也存一个最后的【完整状态】，方便以后想再多练几个 epoch
        save_accelerate_checkpoint(accelerator, output_model_path, global_step)
        print(f"🎉 Training Finished!")
        print(f"💾 Inference Weights: {final_weights_path}")
        print(f"📦 Full Training State: {os.path.join(output_model_path, f'step-{global_step:08d}')}")
    accelerator.end_training()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train_cfg = config.get('training', {})

    cnn_train(
        npy_dir=config['data']['npy_dir'],
        output_model_path=train_cfg.get('output_model_path', 'models'),
        # 🚀 补全缺失的性能与路径参数
        num_workers=train_cfg.get('num_workers', 8),
        prefetch_factor=train_cfg.get('prefetch_factor', 128),
        checkpoint_path=train_cfg.get('checkpoint_path', None),
        lr_scheduler_type=train_cfg.get('lr_scheduler_type', 'cosine'),
        device_micro_batch_size=train_cfg.get('device_micro_batch_size', 16),
        global_batch_size=train_cfg.get('global_batch_size', 256),
        loss_log_interval=train_cfg.get('loss_log_interval', 1),
        lr=train_cfg.get('lr', 1e-4),
        num_epochs=train_cfg.get('num_epochs', 10),
        val_ratio=train_cfg.get('val_ratio', 0.1),
        cnn_type=train_cfg.get('cnn_type', 1),
        mixed_precision=train_cfg.get('mixed_precision', 'no'), # 🚀 从 YAML 读取
        use_wandb=config.get('logging', {}).get('use_wandb', True),
        wandb_project=config.get('logging', {}).get('wandb_project',"default_wandb_project"),
        wandb_name=config.get('logging', {}).get('wandb_name',"default_wandb_name"),

    )

if __name__ == "__main__":
    main()
