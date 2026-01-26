# vqe_train.py
# Nanopore Signal Tokenizer Training Script with VQ-VAE
# Industrial-grade training pipeline for nanopore raw signal tokenization using Vector Quantization.
# Supports distributed training (DDP), dynamic logging, checkpointing, and independent evaluation dataset.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import csv
import time
import json
from pprint import pformat
from scipy.stats import entropy
import argparse
from typing import Dict, List, Optional
import yaml
# Relative imports from the same package
from .dataset import NanoporeSignalDataset
from .vq_model import NanoporeVQModel
from .dwa import DynamicWeightAverager


# =============================================================================
# Utility Functions
# =============================================================================

def print_training_args(**kwargs):
    """
    Pretty-print all training hyperparameters at startup for reproducibility and debugging.
    """
    print("\n" + "="*60)
    print(" 🚀 Starting VQE Training with the following configuration:")
    print("="*60)
    print(pformat(kwargs, width=100, sort_dicts=False))
    print("="*60 + "\n")


def save_full_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    spoch: int,
    global_step: int,
    cnn_type: int,
    rank: int
):
    """
    Save a full training checkpoint (model, optimizer, RNG states) for resuming.
    Only rank 0 performs the actual save to avoid file conflicts in DDP.
    """
    if rank != 0:
        return

    checkpoint = {
        'epoch': epoch,
        'spoch': spoch,
        'global_step': global_step,
        'cnn_type': cnn_type,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'numpy_rng_state': np.random.get_state(),
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, path)
    print(f"✅ Full checkpoint saved to {path}")


def log_and_save(
    epoch: int,
    step: int,
    total_epochs: int,
    total_steps: int,
    epoch_start_time: float,
    epoch_total_steps: int,
    avg_recon_loss: float,
    avg_total_loss: float,
    avg_comit_loss: float,
    avg_diver_loss: float,
    avg_ortho_loss: float,
    codebook_usage: float,
    loss_csv_path: str,
    dynamic_recon_weight: float,
    dynamic_comit_weight: float,
    dynamic_ortho_weight: float,
    dynamic_diver_weight: float,
    lr: float,
):
    """
    Log training metrics to console and append to CSV for offline analysis.
    Time estimation is based on current epoch progress.
    """
    import time

    current_time = time.time()
    elapsed_seconds = current_time - epoch_start_time
    steps_done = step % epoch_total_steps or 1
    avg_time_per_step = elapsed_seconds / steps_done
    remaining_seconds = avg_time_per_step * max(0, epoch_total_steps - steps_done)

    def format_hms(seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    elapsed_str = format_hms(elapsed_seconds)
    remaining_str = format_hms(remaining_seconds)

    epoch_width = len(str(total_epochs))
    step_width = len(str(total_steps))

    print(
        f"[Epoch {epoch+1:>{epoch_width}}/{total_epochs} | "
        f"Step {step:>{step_width}}/{total_steps} | "
        f"{elapsed_str}<{remaining_str}] "
        f"Total: {avg_total_loss:>8.6f} | "
        f"Recon: {avg_recon_loss:>8.6f} | "
        f"Comit: {avg_comit_loss:>8.6f} | "
        f"Ortho: {avg_ortho_loss:>8.6f} | "
        f"Diver: {avg_diver_loss:>3.2f} | "
        f"Usage: {codebook_usage*100:>3.1f}% | "
        f"LR: {lr:>7.2e} |"
    )

    row_data = [
        epoch + 1,
        step,
        avg_recon_loss,
        avg_total_loss,
        avg_comit_loss,
        avg_diver_loss,
        avg_ortho_loss,
        codebook_usage * 100,
        dynamic_recon_weight,
        dynamic_comit_weight,
        dynamic_ortho_weight,
        dynamic_diver_weight,
        lr
    ]
    with open(loss_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)




# =============================================================================
# Main Training Function
# =============================================================================

def vqe_train(
    train_npy_dir: str,
    evaluation_npy_dir: Optional[str] = None,
    output_model_path: str = "nanopore_vq_tokenizer.pth",
    batch_size: int = 16,
    lr: float = 1e-4,
    num_epochs: int = 10,
    codebook_size: int = 8192,
    chunk_size: int = 12000,
    num_workers: int = 8,
    update_loss_weight_every: int = 10,
    prefetch_factor: int = 128,
    val_ratio: float = 0.001,
    do_evaluate: bool = True,
    commitment_weight: float = 1.0,
    codebook_diversity_loss_weight: float = 1.0,
    orthogonal_reg_weight: float = 1.0,
    loss_log_interval: int = 10,
    loss_csv_path: str = "train_loss.csv",
    use_wandb: bool = True,
    wandb_project: str = "nanopore_vq",
    wandb_name: str = "default_run",
    lr_scheduler_type: str = "cosine",
    warmup_steps: int = 500,
    warmup_start_factor: float = 1e-6,
    warmup_end_factor: float = 1.0,
    main_scheduler_end_factor: float = 1e-6,
    save_checkpoint_every_spoch: int = 500,
    evaluate_every_spoch: int = 100,
    checkpoint_path: Optional[str] = None,
    cnn_type: int = 0,
    init_codebook_path: Optional[str] = None,
    cnn_checkpoint_path: Optional[str] = None,
    freeze_cnn: int = 0,
    learnable_codebook: bool = False,
    global_batch_size: int = 256,
    device_micro_batch_size: int = 16
):
    """
    Distributed training of Nanopore VQ tokenizer using DDP.
    
    Key features:
      - Independent evaluation dataset via `evaluation_npy_dir`
      - Checkpoint resume support
      - Pre-trained CNN weight loading & freezing
      - Initial codebook initialization (e.g., from Faiss)
      - WandB & CSV logging
      - Learning rate scheduling with warmup

    ⚠️ NOTE ON DWA (Dynamic Weight Averager):
        The DWA module is used SOLELY for monitoring and logging purposes.
        It does NOT influence the actual loss computation or gradient updates.
        The training loss remains:
            total_loss = recon_loss + comit_loss * commitment_weight
        DWA weights are only recorded in logs/CSV/W&B for analysis.
    """
    print_training_args(
        train_npy_dir=train_npy_dir,
        evaluation_npy_dir=evaluation_npy_dir,
        output_model_path=output_model_path,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        codebook_size=codebook_size,
        chunk_size=chunk_size,
        num_workers=num_workers,
        update_loss_weight_every=update_loss_weight_every,
        prefetch_factor=prefetch_factor,
        val_ratio=val_ratio,
        do_evaluate=do_evaluate,
        commitment_weight=commitment_weight,
        codebook_diversity_loss_weight=codebook_diversity_loss_weight,
        orthogonal_reg_weight=orthogonal_reg_weight,
        loss_csv_path=loss_csv_path,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        warmup_start_factor=warmup_start_factor,
        warmup_end_factor=warmup_end_factor,
        main_scheduler_end_factor=main_scheduler_end_factor,
        save_checkpoint_every_spoch=save_checkpoint_every_spoch,
        evaluate_every_spoch=evaluate_every_spoch,
        checkpoint_path=checkpoint_path,
        init_codebook_path=init_codebook_path,
        cnn_type=cnn_type,
        freeze_cnn= freeze_cnn,
        learnable_codebook = learnable_codebook,
        global_batch_size=global_batch_size,
        device_micro_batch_size=device_micro_batch_size,
    )

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    if checkpoint_path and not os.path.isfile(checkpoint_path):
        print(f"Required checkpoint not found: {checkpoint_path}")
        checkpoint_path = None

    # Initialize distributed environment
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_device_id)
    device = f"cuda:{local_device_id}"

    # Initialize WandB (only rank 0)
    if rank == 0 and use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                "batch_size": batch_size,
                "lr": lr,
                "num_epochs": num_epochs,
                "codebook_size": codebook_size,
                "chunk_size": chunk_size,
                "update_loss_weight_every": update_loss_weight_every,
                "commitment_weight": commitment_weight,
                "codebook_diversity_loss_weight": codebook_diversity_loss_weight,
                "orthogonal_reg_weight": orthogonal_reg_weight,
                "world_size": world_size,
            }
        )
    else:
        wandb = None

    if rank == 0:
        print(f"🚀 Using {world_size} GPUs for training.")
        print(f"📂 Training data: {train_npy_dir}")
        if evaluation_npy_dir:
            print(f"🔍 Evaluation data: {evaluation_npy_dir}")
        print(f"💾 Model will be saved to: {output_model_path}")

        with open(loss_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = [
                'epoch', 'step',
                'recon_loss', 'total_loss', 'comit_loss', 'diver_loss', 'ortho_loss', 'codebook_usage',
                'wv_recon', 'wv_comit', 'wv_ortho', 'wv_diver',
                'lr'
            ]
            writer.writerow(header)

    # ========================
    # Data Loading
    # ========================
    # 当你引入了 global_batch_size 和 device_micro_batch_size（或 device_batch_size）的概念后，DataLoader 的 batch_size 参数就应该设置为 device_micro_batch_size。因为 DataLoader 的 batch_size 指的是每个进程（每张卡）每次加载的数据量，也就是我们所说的“微批次”（micro-batch）。
    train_dataset = NanoporeSignalDataset(shards_dir=train_npy_dir)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        train_dataset,
        batch_size=device_micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        drop_last=True
    )

    # ========================
    # Evaluation Setup
    # ========================
    val_loader = None

    def evaluate_codebook_metrics():
        """Evaluate codebook usage, top-k concentration, and entropy on validation set."""
        if val_loader is None:
            return 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        model.eval()
        used_codes = set()
        token_counts = np.zeros(codebook_size, dtype=np.int64)
        total_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                _, indices, _, _ = model.module(x)
                indices = indices.cpu().numpy().flatten()
                used_codes.update(indices.tolist())
                total_tokens += indices.size
                for idx in indices:
                    token_counts[idx] += 1

        usage_ratio = len(used_codes) / codebook_size
        top1_ratio = top3_ratio = top5_ratio = top7_ratio = top9_ratio = top10_ratio = 0.0
        entropy_val = 0.0
        max_entropy = np.log2(codebook_size)

        if total_tokens > 0:
            # 获取排序后的计数值（降序）
            sorted_counts = np.sort(token_counts)[::-1]
            
            base_ratio = 1/codebook_size
            # 修改top-k ratio的计算方式：出现次数/total_tokens
            # 这样可以直接比较不同codebook_size下的token利用率
            top1_ratio = sorted_counts[0] / total_tokens if len(sorted_counts) > 0 else 0.0
            top3_ratio = sorted_counts[2] / total_tokens if len(sorted_counts) > 2 else 0.0
            top5_ratio = sorted_counts[4] / total_tokens if len(sorted_counts) > 4 else 0.0
            top7_ratio = sorted_counts[6] / total_tokens if len(sorted_counts) > 6 else 0.0
            top9_ratio = sorted_counts[8] / total_tokens if len(sorted_counts) > 8 else 0.0


            top1_ratio = top1_ratio/base_ratio
            top3_ratio = top3_ratio/base_ratio
            top5_ratio = top5_ratio/base_ratio
            top7_ratio = top7_ratio/base_ratio
            top9_ratio = top9_ratio/base_ratio

            top10_ratio = float(sorted_counts[:min(9, codebook_size)].sum()) / total_tokens if len(sorted_counts) > 9 else 0.0

            top1_ratio = top1_ratio/base_ratio
            top3_ratio = top3_ratio/base_ratio
            top5_ratio = top5_ratio/base_ratio
            top7_ratio = top7_ratio/base_ratio
            top9_ratio = top9_ratio/base_ratio
            top10_ratio = top10_ratio/base_ratio

            prob = token_counts / total_tokens
            nonzero_prob = prob[prob > 0]
            if nonzero_prob.size > 0:
                entropy_val = -np.sum(nonzero_prob * np.log2(nonzero_prob))

        model.train()
        return (
            usage_ratio, total_tokens,
            top1_ratio, top3_ratio, top5_ratio, top7_ratio, top9_ratio, top10_ratio,
            entropy_val, max_entropy
        )
    if do_evaluate and rank == 0:
        if evaluation_npy_dir and os.path.isdir(evaluation_npy_dir):
            print(f"✅ Using independent evaluation dataset: {evaluation_npy_dir}")
            val_dataset = NanoporeSignalDataset(shards_dir=evaluation_npy_dir)
        else:
            print(f"⚠️ No evaluation_npy_dir. Using {val_ratio:.1%} of training data for eval.")
            val_dataset = train_dataset
        if val_ratio > 0:
            actual_val_size = max(1, int(val_ratio * len(val_dataset)))
            np.random.seed(42)
            indices = np.random.choice(len(val_dataset), size=actual_val_size, replace=False)
            val_subset = torch.utils.data.Subset(val_dataset, indices)
            val_loader = DataLoader(
                val_subset,
                batch_size=device_micro_batch_size,
                shuffle=False,
                num_workers=max(2, num_workers // 2),
                pin_memory=True,
                drop_last=True
            )
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=max(2, num_workers // 2),
                pin_memory=True
            )


    # ========================
    # Model & Optimizer
    # ========================
    # 如果 learnable_codebook=True，码本 e_k 是 nn.Parameter，可以通过梯度更新（此时不支持EMA)
    # 如果 learnable_codebook=False，码本是固定的 buffer，不会更新。注意： 根据代码逻辑 assert not (ema_update and learnable_codebook)，ema_update 和 learnable_codebook 通常不能同时为 True。这意味着标准的 EMA 更新是用于 learnable_codebook=False 的（但这与原始 VQ-VAE 论文不符，可能是此库的特定实现或文档有误，更常见的是 learnable_codebook=True 配合 ema_update=True）。
    model = NanoporeVQModel(
        codebook_size=codebook_size,
        commitment_weight=commitment_weight,
        codebook_diversity_loss_weight=codebook_diversity_loss_weight,
        orthogonal_reg_weight=orthogonal_reg_weight,
        cnn_type=cnn_type,
        init_codebook_path=init_codebook_path,
        cnn_checkpoint_path = cnn_checkpoint_path,
        freeze_cnn = freeze_cnn,
        learnable_codebook=learnable_codebook #default
        #learnable_codebook=False
    ).to(device)




    model = DDP(model, device_ids=[local_device_id])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ========================
    # DWA for Logging Only
    # ========================
    dwa = None
    if rank == 0:
        init_w = {"recon_loss": 0.25, "comit_loss": 0.25, "ortho_loss": 0.25, "diver_loss": 0.25}
        bounds = {k: (0.01, 0.99) for k in init_w}
        dwa = DynamicWeightAverager(
            loss_names=["recon_loss", "comit_loss", "ortho_loss", "diver_loss", "total_loss"],
            weighted_loss_names=["recon_loss", "comit_loss", "ortho_loss", "diver_loss"],
            initial_weights=init_w,
            weight_bounds=bounds,
            warmup_steps=10,
            temperature=1.0,
            window_size=50,
            slow_window=45,
            fast_window=5,
            device=device
        )
        # 📌 CRITICAL: DWA is ONLY for logging. Loss uses fixed hyperparameters below.

    # ========================
    # Learning Rate Scheduler
    # ========================
    scheduler = None
    total_training_steps = len(dataloader) * num_epochs

    if rank == 0 and lr_scheduler_type != "constant":
        print(f"📈 Using LR scheduler: {lr_scheduler_type}, warmup_steps={warmup_steps}")

    if lr_scheduler_type != "constant":
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        warmup_scheduler = LinearLR(optimizer, start_factor=warmup_start_factor, end_factor=warmup_end_factor, total_iters=warmup_steps)
        main_steps = max(1, total_training_steps - warmup_steps)

        if lr_scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(optimizer, T_max=main_steps)
        elif lr_scheduler_type == "linear":
            relative_end_factor = max(1e-8, min(1.0, main_scheduler_end_factor / warmup_end_factor))
            main_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=relative_end_factor, total_iters=main_steps)
        else:
            raise ValueError(f"Unsupported lr_scheduler_type: {lr_scheduler_type}")

        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])

    # ========================
    # Resume from Checkpoint
    # ========================
    start_epoch = start_spoch = start_global_step = 0
    if checkpoint_path and os.path.isfile(checkpoint_path):
        if rank == 0:
            print(f"📥 Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if rank == 0:
            torch.set_rng_state(ckpt['rng_state'])
            if ckpt.get('cuda_rng_state') is not None:
                torch.cuda.set_rng_state(ckpt['cuda_rng_state'])
            if 'numpy_rng_state' in ckpt:
                np.random.set_state(ckpt['numpy_rng_state'])
            start_epoch = ckpt.get('epoch', -1) + 1
            start_spoch = ckpt.get('spoch', -1) + 1
            start_global_step = ckpt.get('global_step', 0)
            print(f"✅ Resuming from epoch {start_epoch}, spoch {start_spoch}")

    # ========================
    # Training Loop
    # ========================

    model.train()
    global_step = start_global_step
    spoch = start_spoch
    total_steps = len(dataloader) * num_epochs
    total_spochs = total_steps // update_loss_weight_every
    cached_wvalue = torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)
    loss_buffer = {"recon": [], "comit": [], "ortho": [], "diver": []}

    # Evaluation metrics (initialized)
    (codebook_usage, total_tokens,
     codebook_top1_ratio, codebook_top3_ratio, codebook_top5_ratio,
     codebook_top7_ratio, codebook_top9_ratio, codebook_top10_ratio,
     codebook_entropy, codebook_max_entropy) = (0.0,) * 10
    # --- 修改点 1: 计算累积步数 ---
    # 假设你已经定义了 global_batch_size 和 device_micro_batch_size
    # world_size 是 DDP 的进程数量 (可以通过 dist.get_world_size() 获取)
    world_size = dist.get_world_size() 
    effective_micro_batch = device_micro_batch_size * world_size
    accumulation_steps = global_batch_size // effective_micro_batch

    if accumulation_steps == 0:
        raise ValueError(f"global_batch_size ({global_batch_size}) 太小，或者 device_micro_batch_size ({device_micro_batch_size}) * world_size ({world_size}) 太大，无法进行累积。")

    print(f"使用梯度累积: Global Batch={global_batch_size}, Micro Batch={device_micro_batch_size}, 累积步数={accumulation_steps}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        sampler.set_epoch(epoch)
        # --- 修改点 2: 在 epoch 开始时清零梯度 ---
        # 但如果第一个 micro-step 不是累积周期的第一个 step（即 global_step % accumulation_steps != 0），
        # 那么第一次 zero_grad 会被提前覆盖。更好的方式是，
        # 让 Dataloader 的起始 global_step 对齐到某个 accumulation 周期的起点，
        # 或者在每个 should_update 之后 zero_grad（如果循环不是严格按 accumulation_steps 划分的，这可能不对）。
        # 通常，如果 global_step 在加载 checkpoint 时正确恢复，
        # 那么 optimizer.zero_grad() 放在循环外部或第一次 should_update 时执行更安全。
        # 但根据你的原始逻辑，它在 epoch 开始时，这没问题，只要保证第一次 step 会 zero_grad。
        # 如果 global_step % accumulation_steps == 0 在循环开始时为 True，则这里 OK。
        # 否则，可能需要在第一个 micro-step 时检查并 zero_grad。
        # 让我们假设初始状态是正确的。
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):
            global_step += 1
            x = batch.to(device)

            recon, indices, break_loss, loss_breakdown = model(x)
            recon_loss = F.mse_loss(recon, x)
            comit_loss = loss_breakdown.commitment
            diver_loss = loss_breakdown.codebook_diversity
            ortho_loss = loss_breakdown.orthogonal_reg
            # 💡 ACTUAL LOSS: Fixed weights. DWA is NOT applied here.
            # 💡 计算当前 step 的 Loss (不立即缩放)
            total_loss = recon_loss + comit_loss * commitment_weight
            # ✅ 🔥 在这里缩放损失（除以 accumulation_steps）
            scaled_total_loss = total_loss / accumulation_steps
            scaled_total_loss.backward() 
            # --- 修改点 4: 判断是否执行优化器更新 ---
            #is_last_step = (step == len(dataloader) - 1)
            should_update = (global_step % accumulation_steps == 0) or (step == len(dataloader) - 1)
            if should_update:
                # 2. 执行梯度更新
                optimizer.step()

                # 3. 执行梯度裁剪 (可选，通常在小 batch 下很有用)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 4. 清空梯度，准备下一个累积周期
                optimizer.zero_grad()

                # --- 保持原有的 Scheduler 更新逻辑 ---
                if scheduler is not None:
                    scheduler.step()

                # --- 以下所有内容都应在此块内 ---
                # 这些操作现在都基于一次完整的参数更新

                # Buffer losses for DWA logging (需要在每个 micro-step 都做，但汇总在 should_update 时)
                # 注意：我们需要一个机制来累积 micro-step 的 loss，然后在 should_update 时取平均
                # 我们可以利用现有的 loss_buffer 逻辑，但需要确保它在每个 accumulation 周期开始时清空，
                # 并且在 should_update 时处理。
                # 原来的 loss_buffer 逻辑似乎在 update_loss_weight_every 时清空，
                # 但我们希望在每个 accumulation 周期结束时清空或处理。

                # 为了让 loss_buffer 适配 accumulation，我们可以将其计数与 accumulation 关联。
                # 但原代码是按 global_step 的倍数 (update_loss_weight_every) 来处理的。
                # 为了保持兼容性，我们可以让 loss_buffer 在每个 should_update 时（即每个有效步）累加，
                # 然后在特定数量的 *有效步* 后（而不是 micro-step）进行 DWA 更新。
                # 这意味着我们需要一个新的计数器，比如 effective_step_count。

                # ... (需要引入 effective_step_count) ...
                # effective_step_count = (global_step - 1) // accumulation_steps # 计算从 0 开始的有效步数

                # 为了最小化改动，我们可以假设 update_loss_weight_every 指的是 *有效步* 的间隔。
                # 那么 effective_step_count = (global_step - 1) // accumulation_steps + 1 (从 1 开始)
                effective_step_count = global_step // accumulation_steps # 整除正好给出有效步数 (从 1 开始 if global_step starts from 1 after first update)

                # Buffer losses for DWA logging (not used in optimization)
                loss_buffer["recon"].append(recon_loss.item())
                loss_buffer["comit"].append(comit_loss.item())
                loss_buffer["ortho"].append(ortho_loss.item())
                loss_buffer["diver"].append(diver_loss.item())

                # --- 将原 should_update_weights 逻辑移入 here ---
                should_update_weights = (effective_step_count % update_loss_weight_every == 0) # 或者根据你的需求调整
                
                if should_update_weights:
                    # spoch 在这里更新更有意义，因为它代表了有效的训练步
                    # spoch = effective_step_count # 或 spoch += 1; 取决于你想如何定义 spoch
                    spoch += 1
                    def safe_mean(lst):
                        return sum(lst) / len(lst) if lst else 0.0
                    local_avg_losses = torch.tensor([
                        safe_mean(loss_buffer["recon"]),
                        safe_mean(loss_buffer["comit"]),
                        safe_mean(loss_buffer["ortho"]),
                        safe_mean(loss_buffer["diver"])
                    ], device=device)

                    dist.all_reduce(local_avg_losses, op=dist.ReduceOp.AVG)
                    g_recon, g_comit, g_ortho, g_diver = local_avg_losses.tolist()
                    g_total = g_recon + g_comit * commitment_weight + g_ortho * orthogonal_reg_weight + g_diver * codebook_diversity_loss_weight

                    if rank == 0:
                        current_losses = {
                            "recon_loss": g_recon,
                            "comit_loss": g_comit,
                            "ortho_loss": g_ortho,
                            "diver_loss": g_diver,
                            "total_loss": g_total
                        }
                        wvalue = dwa.update_and_get_weights(current_losses)
                        wvalue_tensor = torch.tensor([
                            wvalue["recon_loss"],
                            wvalue["comit_loss"],
                            wvalue["ortho_loss"],
                            wvalue["diver_loss"],
                        ], device=device)
                    else:
                        wvalue_tensor = torch.empty(4, device=device)

                    dist.broadcast(wvalue_tensor, src=0)
                    cached_wvalue = wvalue_tensor
                    loss_buffer = {k: [] for k in loss_buffer}

                    if rank == 0:
                        wv_recon, wv_comit, wv_ortho, wv_diver = cached_wvalue.tolist()
                        current_lr = optimizer.param_groups[0]['lr']
                        log_and_save(
                            epoch=epoch,
                            step=global_step,
                            total_epochs=num_epochs,
                            total_steps=total_steps,
                            epoch_start_time=epoch_start_time,
                            epoch_total_steps=len(dataloader),
                            avg_recon_loss=g_recon,
                            avg_total_loss=g_total,
                            avg_comit_loss=g_comit,
                            avg_diver_loss=g_diver,
                            avg_ortho_loss=g_ortho,
                            codebook_usage=codebook_usage,
                            loss_csv_path=loss_csv_path,
                            dynamic_recon_weight=wv_recon,
                            dynamic_comit_weight=wv_comit,
                            dynamic_ortho_weight=wv_ortho,
                            dynamic_diver_weight=wv_diver,
                            lr=current_lr
                        )

                        log_dict = {
                            "train/recon_loss": g_recon,
                            "train/comit_loss": g_comit,
                            "train/ortho_loss": g_ortho,
                            "train/diver_loss": g_diver,
                            "train/total_loss": g_total,
                            "codebook/usage": codebook_usage,
                            "codebook/entropy": codebook_entropy,
                            "codebook/max_entropy": codebook_max_entropy,
                            "topcode/top1_ratio": codebook_top1_ratio,
                            "topcode/top3_ratio": codebook_top3_ratio,
                            "topcode/top5_ratio": codebook_top5_ratio,
                            "topcode/top7_ratio": codebook_top7_ratio,
                            "topcode/top9_ratio": codebook_top9_ratio,
                            "topcode/topx_ratio": codebook_top10_ratio,
                            "weights/recon": wv_recon,
                            "weights/comit": wv_comit,
                            "learning_rate": current_lr,
                            "epoch": epoch + 1,
                        }
                        if use_wandb:
                            wandb.log(log_dict, step=global_step)

                    # --- 评估 ---
                    # 评估的频率现在应该基于 effective_step_count (即 spoch)
                    # 注意：这里需要小心处理 spoch 的定义，确保它与 evaluate_every_spoch 对齐
                    # 假设 spoch 现在等于 effective_step_count
                    # 注意：spoch 在这里被赋值，所以用 spoch 而不是 effective_step_count
                    if (spoch % evaluate_every_spoch == 0): # 使用 spoch
                        dist.barrier()  # 所有 ranks 同步到此
                        if rank == 0:
                            # 这个函数运行时间过长，会导致超时退出
                            (codebook_usage, total_tokens,
                             codebook_top1_ratio, codebook_top3_ratio, codebook_top5_ratio,
                             codebook_top7_ratio, codebook_top9_ratio, codebook_top10_ratio,
                             codebook_entropy, codebook_max_entropy) = evaluate_codebook_metrics()
                            print(f"Effective Step {spoch} - Codebook Usage: {codebook_usage:.2%}") # 更新打印信息
                        else:
                            # 其他 ranks 不做任何事，但必须等待 rank 0 完成
                            pass
                        dist.barrier()  # eval 完再同步 # 所有 ranks 继续

                    # Periodic checkpointing
                    if rank == 0 and (spoch + 1) % save_checkpoint_every_spoch == 0:
                        ckpt_path = f"{output_model_path}.spoch{spoch+1}.pth"
                        save_full_checkpoint(
                            path=ckpt_path,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            spoch=spoch,
                            global_step=global_step,
                            cnn_type=cnn_type,
                            rank=rank
                        )
                # --- End of if should_update_weights block ---
                else: # 如果 not should_update
                    # 在累积周期内，只需添加损失到 buffer
                    # (这已经在 if should_update 之外做了)
                    loss_buffer["recon"].append(recon_loss.item())
                    loss_buffer["comit"].append(comit_loss.item())
                    loss_buffer["ortho"].append(ortho_loss.item())
                    loss_buffer["diver"].append(diver_loss.item())

    # Final save
    if rank == 0:
        save_full_checkpoint(
            path=output_model_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=num_epochs - 1,
            spoch=spoch,
            global_step=global_step,
            cnn_type=cnn_type,
            rank=rank
        )
        print(f"✅ Final model saved to {output_model_path}")
        if use_wandb:
            wandb.finish()

    dist.barrier()
    dist.destroy_process_group()


# =============================================================================
# CLI Entry Point
# =============================================================================




def main():
    parser = argparse.ArgumentParser(description="Train Nanopore VQ Tokenizer")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    # Load configuration from YAML file
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # Pass the configuration dictionary values to vqe_train
    vqe_train(
        train_npy_dir=config.get("train_npy_dir"),
        evaluation_npy_dir=config.get("evaluation_npy_dir"),
        output_model_path=config.get("output_model_path", "nanopore_vq_tokenizer.pth"),
        batch_size=config.get("batch_size", 16),
        lr=config.get("lr", 3e-4),
        num_epochs=config.get("num_epochs", 10),
        codebook_size=config.get("codebook_size", 8192),
        chunk_size=config.get("chunk_size", 12000),
        num_workers=config.get("num_workers", 8),
        val_ratio=config.get("val_ratio", 0.1),
        do_evaluate=config.get("do_evaluate", False),
        commitment_weight=config.get("commitment_weight", 0.25),
        codebook_diversity_loss_weight=config.get("codebook_diversity_loss_weight", 0.0),
        orthogonal_reg_weight=config.get("orthogonal_reg_weight", 0.0),
        loss_csv_path=config.get("loss_csv_path", "train_loss.csv"),
        save_checkpoint_every_spoch=config.get("save_checkpoint_every_spoch", 100),
        loss_log_interval=config.get("loss_log_interval", 10),
        checkpoint_path=config.get("checkpoint_path"),
        cnn_type=config.get("cnn_type", 1),
        init_codebook_path=config.get("init_codebook_path", ""),
        cnn_checkpoint_path=config.get("cnn_checkpoint_path", ""),
        freeze_cnn=config.get("freeze_cnn", 0),
        learnable_codebook=config.get("learnable_codebook", True),
        global_batch_size=config.get("global_batch_size", 256),
        device_micro_batch_size=config.get("device_micro_batch_size", 8),
        # Add other parameters that might be in the YAML but not explicitly listed above
        update_loss_weight_every=config.get("update_loss_weight_every", 10),
        prefetch_factor=config.get("prefetch_factor", 128),
        use_wandb=config.get("use_wandb", True),
        wandb_project=config.get("wandb_project", "nanopore_vq"),
        wandb_name=config.get("wandb_name", "default_run"),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_steps=config.get("warmup_steps", 500),
        warmup_start_factor=config.get("warmup_start_factor", 1e-6),
        warmup_end_factor=config.get("warmup_end_factor", 1.0),
        main_scheduler_end_factor=config.get("main_scheduler_end_factor", 1e-6),
        evaluate_every_spoch=config.get("evaluate_every_spoch", 100),
    )
if __name__ == "__main__":
    main()
