# vqe_train_accelerate.py
# Nanopore Signal Tokenizer Training Script with VQ-VAE
# Industrial-grade training pipeline for nanopore raw signal tokenization using Vector Quantization.
# Supports distributed training (DDP), dynamic logging, checkpointing, and independent evaluation dataset.
# This version uses Hugging Face Accelerate for simplified multi-GPU/mixed precision training.

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
# Import Accelerate components
from accelerate import Accelerator, DistributedType
# Relative imports from the same package
# Note: The relative imports might need adjustment depending on your package structure.
# If this script is run directly, you might need to add the parent directory to sys.path.
try:
    from .dataset import NanoporeSignalDataset
    from .vq_model import NanoporeVQModel
    from .dwa import DynamicWeightAverager
except ImportError:
    # Fallback for direct execution
    from dataset import NanoporeSignalDataset
    from vq_model import NanoporeVQModel
    from dwa import DynamicWeightAverager


# =============================================================================
# Utility Functions
# =============================================================================

def print_training_args(**kwargs):
    """
    Pretty-print all training hyperparameters at startup for reproducibility and debugging.
    """
    print("\n" + "="*60)
    print(" 🚀 Starting VQE Training with Accelerate. Configuration:")
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
    accelerator: Accelerator
):
    """
    Save a full training checkpoint (model, optimizer, RNG states) for resuming.
    Uses Accelerate's save_state method which handles DDP/DDP sharding automatically.
    Only the main process saves the checkpoint.
    """
    # Accelerate's save_state handles distributed saving internally
    accelerator.save_state(path)
    if accelerator.is_main_process:
        print(f"✅ Full checkpoint saved to {path}")
        # Optionally save additional metadata separately
        metadata = {
            'epoch': epoch,
            'spoch': spoch,
            'global_step': global_step,
            'cnn_type': cnn_type,
        }
        meta_path = path.replace('.pth', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)


def load_checkpoint_metadata(path: str):
    """
    Load metadata associated with a checkpoint.
    """
    meta_path = path.replace('.pth', '_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None


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
    accelerator: Accelerator
):
    """
    Log training metrics to console and append to CSV for offline analysis.
    Time estimation is based on current epoch progress.
    Only the main process writes to console and CSV.
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

    # Only main process prints and writes to CSV
    if accelerator.is_main_process:
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
    batch_size: int = 16, # Note: This now refers to the device_micro_batch_size
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
    device_micro_batch_size: int = 16, # Renamed for clarity, was 'batch_size' before
    mixed_precision: str = "bf16", # Options: "no", "fp16", "bf16", "fp8"
    gradient_clipping: float = 1.0, # Set to None to disable
    cpu: bool = False # Set to True to force CPU training
):
    """
    Training of Nanopore VQ tokenizer using Hugging Face Accelerate.

    Key features:
      - Distributed training handled by Accelerate (multi-GPU, multi-node).
      - Mixed precision training support (FP16, BF16).
      - Automatic gradient scaling for FP16.
      - Independent evaluation dataset via `evaluation_npy_dir`.
      - Checkpoint resume support (using Accelerate's state management).
      - Pre-trained CNN weight loading & freezing.
      - Initial codebook initialization (e.g., from Faiss).
      - WandB & CSV logging (main process only).
      - Learning rate scheduling with warmup.
      - Gradient accumulation integrated with Accelerate.

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
        mixed_precision=mixed_precision,
        gradient_clipping=gradient_clipping,
        cpu=cpu
    )

    # Initialize Accelerator
    # This handles device placement, distributed setup, and mixed precision automatically.
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        cpu=cpu,
        # gradient_accumulation_steps is handled manually within the loop
    )
    # Log accelerator info
    if accelerator.is_main_process:
        print(f"🚀 Accelerator initialized. Device: {accelerator.device}, Type: {accelerator.distributed_type}")
        print(f"   Number of processes: {accelerator.num_processes}")
        print(f"   Mixed Precision: {accelerator.mixed_precision}")
        print(f"   Global Batch Size: {global_batch_size}, Device Micro-Batch Size: {device_micro_batch_size}")

    # Initialize WandB (only main process)
    if accelerator.is_main_process and use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                "device_micro_batch_size": device_micro_batch_size, # Changed from 'batch_size'
                "lr": lr,
                "num_epochs": num_epochs,
                "codebook_size": codebook_size,
                "chunk_size": chunk_size,
                "update_loss_weight_every": update_loss_weight_every,
                "commitment_weight": commitment_weight,
                "codebook_diversity_loss_weight": codebook_diversity_loss_weight,
                "orthogonal_reg_weight": orthogonal_reg_weight,
                "world_size": accelerator.num_processes, # Changed from 'world_size'
                "mixed_precision": mixed_precision,
                "global_batch_size": global_batch_size,
            }
        )
    else:
        wandb = None

    if accelerator.is_main_process:
        print(f"🚀 Using {accelerator.num_processes} processes for training.")
        print(f"📂 Training data: {train_npy_dir}")
        if evaluation_npy_dir:
            print(f"🔍 Evaluation data: {evaluation_npy_dir}")
        print(f"💾 Model will be saved to: {output_model_path}")

        # Initialize CSV log file (main process only)
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
    # DataLoader's batch_size is now the micro-batch size per device/process.
    train_dataset = NanoporeSignalDataset(shards_dir=train_npy_dir)
    # Accelerate provides a convenient way to create distributed samplers
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=device_micro_batch_size, # Micro-batch size per process
        shuffle=False, # Shuffling is handled by the sampler
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        drop_last=True
    )
    # Prepare dataloader with Accelerate (handles DistributedSampler internally)
    train_dataloader = accelerator.prepare(train_dataloader)

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
                x = batch # No need to call .to(device), Accelerate handles it
                # Ensure model input is correctly shaped if needed (e.g., adding batch dim)
                # Example: if x.shape is [seq_len, feat_dim] and model expects [batch, seq_len, feat_dim]
                # x = x.unsqueeze(0) # This adds the batch dimension
                # However, since we are using batches from DataLoader, x should already have a batch dimension
                # Assuming x shape is [batch_size, seq_len, feat_dim]
                
                # The model expects the correct shape; ensure your dataset/dataloader provides it.
                # If your raw data from dataset has shape [seq_len, feat_dim], DataLoader makes it [device_micro_batch_size, seq_len, feat_dim]
                # which should be fine for a model expecting [batch, seq_len, feat_dim].
                
                _, indices, _, _ = model(x) # Model input is handled by Accelerate
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
            sorted_counts = np.sort(token_counts)[::-1]

            base_ratio = 1/codebook_size
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

    if do_evaluate and accelerator.is_main_process: # Only setup eval on main process initially
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
            # Prepare val_loader with Accelerate if needed for consistency
            # val_loader = accelerator.prepare(val_loader) # Often not necessary for eval loader, but can be done
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=device_micro_batch_size, # Use same micro-batch size
                shuffle=False,
                num_workers=max(2, num_workers // 2),
                pin_memory=True
            )
            # val_loader = accelerator.prepare(val_loader) # Often not necessary for eval loader


    # ========================
    # Model & Optimizer & Scheduler Preparation
    # ========================
    model = NanoporeVQModel(
        codebook_size=codebook_size,
        commitment_weight=commitment_weight,
        codebook_diversity_loss_weight=codebook_diversity_loss_weight,
        orthogonal_reg_weight=orthogonal_reg_weight,
        cnn_type=cnn_type,
        init_codebook_path=init_codebook_path,
        cnn_checkpoint_path = cnn_checkpoint_path,
        freeze_cnn = freeze_cnn,
        learnable_codebook=learnable_codebook
    )
    # No need to manually call .to(device) or wrap with DDP, Accelerate handles it
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepare model, optimizer, and scheduler with Accelerate
    model, optimizer = accelerator.prepare(model, optimizer)

    # Scheduler setup (Accelerate requires it to be prepared after prepare(optimizer))
    scheduler = None
    total_training_steps = len(train_dataloader) * num_epochs

    if accelerator.is_main_process and lr_scheduler_type != "constant":
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
        # Prepare scheduler with Accelerate
        scheduler = accelerator.prepare(scheduler)

    # ========================
    # DWA for Logging Only (Main Process)
    # ========================
    dwa = None
    if accelerator.is_main_process:
        init_w = {"recon_loss": 0.25, "comit_loss": 0.25, "ortho_loss": 0.25, "diver_loss": 0.25}
        bounds = {k: (0.01, 0.99) for k in init_w}
        # Use the accelerator's device for DWA if it's on the main process
        # Note: DWA itself doesn't need to be prepared by Accelerate as it's only for logging
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
            device=accelerator.device # Use the accelerator's device
        )
        # 📌 CRITICAL: DWA is ONLY for logging. Loss uses fixed hyperparameters below.

    # ========================
    # Resume from Checkpoint
    # ========================
    start_epoch = start_spoch = start_global_step = 0
    if checkpoint_path and os.path.isfile(checkpoint_path):
        if accelerator.is_main_process:
            print(f"📥 Loading checkpoint from: {checkpoint_path}")
        # Load state using Accelerate. This loads model, optimizer, scheduler, and RNG states.
        accelerator.load_state(checkpoint_path)
        # Load metadata if available
        metadata = load_checkpoint_metadata(checkpoint_path)
        if metadata:
            start_epoch = metadata.get('epoch', -1) + 1
            start_spoch = metadata.get('spoch', -1) + 1
            start_global_step = metadata.get('global_step', 0)
            if accelerator.is_main_process:
                print(f"✅ Resuming from epoch {start_epoch}, spoch {start_spoch}, global_step {start_global_step}")
        else:
            if accelerator.is_main_process:
                print("⚠️ Checkpoint loaded, but no metadata found. Starting from scratch.")


    # ========================
    # Gradient Accumulation Setup
    # ========================
    # Calculate how many micro-steps make up one effective step
    # effective_micro_batch = device_micro_batch_size * num_processes
    # accumulation_steps = global_batch_size // effective_micro_batch
    # Or more directly, if global_batch_size is desired effective batch size:
    effective_micro_batch = device_micro_batch_size * accelerator.num_processes
    accumulation_steps = global_batch_size // effective_micro_batch

    if accumulation_steps == 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) is too small for the current "
            f"device_micro_batch_size ({device_micro_batch_size}) and num_processes ({accelerator.num_processes}). "
            f"Minimum global_batch_size required is {effective_micro_batch}."
        )

    if accelerator.is_main_process:
        print(f"🔄 Gradient Accumulation: Global Batch={global_batch_size}, "
              f"Micro Batch (per proc)={device_micro_batch_size}, Num Procs={accelerator.num_processes}, "
              f"Cumulative Steps={accumulation_steps}")


    # ========================
    # Training Loop
    # ========================
    model.train()
    global_step = start_global_step
    spoch = start_spoch
    total_steps = len(train_dataloader) * num_epochs
    total_spochs = total_steps // update_loss_weight_every
    # cached_wvalue is now local to main process or broadcasted
    cached_wvalue = torch.tensor([0.25, 0.25, 0.25, 0.25], device=accelerator.device) # Initialize on accelerator's device
    loss_buffer = {"recon": [], "comit": [], "ortho": [], "diver": []}


    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        # Set epoch for the sampler if applicable (handled by Accelerate)
        train_dataloader.sampler.set_epoch(epoch) if hasattr(train_dataloader.sampler, 'set_epoch') else None

        # Zero grad at the beginning of an epoch (or more precisely, before the first accumulation cycle)
        # Accelerate's optimizer handles the zero_grad call internally when needed,
        # but for explicit control and clarity, especially with manual accumulation, it's good practice.
        # However, Accelerate often calls zero_grad automatically *before* the first backward() in a new step cycle.
        # The safest place is right before the first backward() call in a new accumulation cycle.
        # We'll handle this inside the loop based on the accumulation logic.
        for step, batch in enumerate(train_dataloader):
            global_step += 1
            # No need to manually move batch to device
            x = batch

            # Forward pass
            recon, indices, break_loss, loss_breakdown = model(x)
            recon_loss = F.mse_loss(recon, x)
            comit_loss = loss_breakdown.commitment
            diver_loss = loss_breakdown.codebook_diversity
            ortho_loss = loss_breakdown.orthogonal_reg
            # 💡 ACTUAL LOSS: Fixed weights. DWA is NOT applied here.
            total_loss = recon_loss + comit_loss * commitment_weight

            # Scale the loss by the number of accumulation steps for averaging
            # Accelerate's backward function handles gradient scaling for mixed precision automatically
            accelerator.backward(total_loss / accumulation_steps)

            # Determine if we should perform an optimizer step
            should_update = (global_step % accumulation_steps == 0) or (step == len(train_dataloader) - 1)

            if should_update:
                # Clip gradients if specified
                if gradient_clipping is not None:
                    # Use Accelerate's clip_grad_norm_ method
                    accelerator.clip_grad_norm_(model.parameters(), gradient_clipping)

                # Perform the optimizer step
                optimizer.step()

                # Update scheduler if present
                if scheduler is not None:
                    scheduler.step()

                # Clear gradients for the next accumulation cycle
                optimizer.zero_grad()

                # --- Post-update operations (logging, evaluation, checkpointing) ---
                # These happen after a full parameter update

                # Buffer losses for DWA logging (averaged over the accumulation cycle)
                loss_buffer["recon"].append(recon_loss.item())
                loss_buffer["comit"].append(comit_loss.item())
                loss_buffer["ortho"].append(ortho_loss.item())
                loss_buffer["diver"].append(diver_loss.item())

                # Determine if it's time to update weights (based on effective steps)
                effective_step_count = global_step // accumulation_steps
                should_update_weights = (effective_step_count % update_loss_weight_every == 0)

                if should_update_weights:
                    spoch += 1 # Increment spoch counter

                    # Aggregate losses across all processes
                    def safe_mean(lst):
                        return sum(lst) / len(lst) if lst else 0.0
                    local_avg_losses = torch.tensor([
                        safe_mean(loss_buffer["recon"]),
                        safe_mean(loss_buffer["comit"]),
                        safe_mean(loss_buffer["ortho"]),
                        safe_mean(loss_buffer["diver"])
                    ], device=accelerator.device)

                    # Use Accelerate's gather method to collect from all processes
                    # gather_for_metrics averages the tensor across all processes
                    gathered_avg_losses = accelerator.gather_for_metrics(local_avg_losses)
                    # 在 g_recon, g_comit, g_ortho, g_diver = avg_gathered.tolist() 之前添加
                    # Debug: Check the shape of gathered_avg_losses
                    #print(f"[Rank {accelerator.process_index}] Shape of gathered_avg_losses: {gathered_avg_losses.shape}")
                    #print(f"[Rank {accelerator.process_index}] Content of gathered_avg_losses: {gathered_avg_losses.tolist()}")
                    # Regardless of the exact behavior of gather_for_metrics, reshape and average correctly
                    # We expect num_processes copies of the 4 losses
                    num_processes = accelerator.num_processes # Get the number of processes
                    if gathered_avg_losses.numel() == 4 * num_processes and gathered_avg_losses.numel() > 4:
                        # It seems gather_for_metrics concatenated the tensors directly into [num_proc * 4]
                        # Reshape it back to [num_proc, 4] and then average over the first dimension
                        reshaped_losses = gathered_avg_losses.view(num_processes, -1) # Shape: [num_processes, 4]
                        avg_gathered = reshaped_losses.mean(dim=0) # Shape: [4]
                    elif gathered_avg_losses.shape == torch.Size([4]):
                        # Only one process or gather_for_metrics worked as expected initially
                        avg_gathered = gathered_avg_losses
                    else:
                        # Unexpected shape, raise an error or handle accordingly
                        print(f"Unexpected gathered_avg_losses shape: {gathered_avg_losses.shape} on Rank {accelerator.process_index}")
                        # You might want to raise an exception here depending on your needs
                        # raise RuntimeError(f"Unexpected gathered_avg_losses shape: {gathered_avg_losses.shape}")
                    #print(f"[Rank {accelerator.process_index}] Shape of avg_gathered after fix: {avg_gathered.shape}") # Optional debug print
                    #print(f"[Rank {accelerator.process_index}] Content of avg_gathered after fix: {avg_gathered.tolist()}") # Optional debug print
                    g_recon, g_comit, g_ortho, g_diver = avg_gathered.tolist()
                    g_total = g_recon + g_comit * commitment_weight + g_ortho * orthogonal_reg_weight + g_diver * codebook_diversity_loss_weight

                    if accelerator.is_main_process:
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
                        ], device=accelerator.device)
                    else:
                        # Create an empty tensor on non-main processes
                        wvalue_tensor = torch.empty(4, device=accelerator.device)


                    
                    # --- Fix: Use torch.distributed.broadcast ---
                    if torch.distributed.is_initialized():
                        # Broadcast the tensor FROM the main process (rank 0) TO all other processes.
                        # The wvalue_tensor on ALL processes (including main) will be filled with
                        # the values from the main process's wvalue_tensor.
                        torch.distributed.broadcast(wvalue_tensor, src=0) # src=0 is the main process rank
                    # If not distributed (single GPU/CPU), wvalue_tensor already contains the correct values on the main process
                    # and the empty tensor on others, but since there's only one process, it doesn't matter much.
                    # However, you'd typically not reach this part in single-process mode if the logic above ensures
                    # calculation only happens on main.
                    # In single process mode, wvalue_tensor already holds the value if is_main_process was true,
                    # or remains empty if it wasn't (which shouldn't happen if this code path is only taken when needed).

                    # After broadcast, wvalue_tensor on ALL processes now contains the values calculated on the main process.
                    wvalue_tensor_sync = wvalue_tensor
                    # --- End Fix ---

                    cached_wvalue = wvalue_tensor_sync

                    # Clear the loss buffer for the next cycle
                    loss_buffer = {k: [] for k in loss_buffer}

                    # Log metrics (main process only)
                    if accelerator.is_main_process:
                        wv_recon, wv_comit, wv_ortho, wv_diver = cached_wvalue.tolist()
                        current_lr = optimizer.param_groups[0]['lr']
                        log_and_save(
                            epoch=epoch,
                            step=global_step,
                            total_epochs=num_epochs,
                            total_steps=total_steps,
                            epoch_start_time=epoch_start_time,
                            epoch_total_steps=len(train_dataloader),
                            avg_recon_loss=g_recon,
                            avg_total_loss=g_total,
                            avg_comit_loss=g_comit,
                            avg_diver_loss=g_diver,
                            avg_ortho_loss=g_ortho,
                            codebook_usage=0.0, # Placeholder, updated later if eval runs
                            loss_csv_path=loss_csv_path,
                            dynamic_recon_weight=wv_recon,
                            dynamic_comit_weight=wv_comit,
                            dynamic_ortho_weight=wv_ortho,
                            dynamic_diver_weight=wv_diver,
                            lr=current_lr,
                            accelerator=accelerator # Pass accelerator instance
                        )

                        # Prepare log dict for WandB
                        log_dict = {
                            "train/recon_loss": g_recon,
                            "train/comit_loss": g_comit,
                            "train/ortho_loss": g_ortho,
                            "train/diver_loss": g_diver,
                            "train/total_loss": g_total,
                            "codebook/usage": 0.0, # Placeholder
                            "codebook/entropy": 0.0, # Placeholder
                            "codebook/max_entropy": 0.0, # Placeholder
                            "topcode/top1_ratio": 0.0, # Placeholder
                            "topcode/top3_ratio": 0.0, # Placeholder
                            "topcode/top5_ratio": 0.0, # Placeholder
                            "topcode/top7_ratio": 0.0, # Placeholder
                            "topcode/top9_ratio": 0.0, # Placeholder
                            "topcode/topx_ratio": 0.0, # Placeholder
                            "weights/recon": wv_recon,
                            "weights/comit": wv_comit,
                            "learning_rate": current_lr,
                            "epoch": epoch + 1,
                            "global_step": global_step, # Useful for plotting against global steps
                        }
                        if use_wandb and wandb is not None:
                            wandb.log(log_dict,step=global_step)

                    # --- Evaluation ---
                    # Run evaluation based on spoch
                    if (spoch % evaluate_every_spoch == 0):
                        accelerator.wait_for_everyone() # Sync all processes before eval
                        if accelerator.is_main_process and val_loader is not None:
                            try:
                                (codebook_usage, total_tokens,
                                 codebook_top1_ratio, codebook_top3_ratio, codebook_top5_ratio,
                                 codebook_top7_ratio, codebook_top9_ratio, codebook_top10_ratio,
                                 codebook_entropy, codebook_max_entropy) = evaluate_codebook_metrics()
                                print(f"Effective Step {spoch} - Codebook Usage: {codebook_usage:.2%}")
                                # Update the placeholders in log_dict for WandB
                                log_dict.update({
                                    "codebook/usage": codebook_usage,
                                    "codebook/entropy": codebook_entropy,
                                    "codebook/max_entropy": codebook_max_entropy,
                                    "topcode/top1_ratio": codebook_top1_ratio,
                                    "topcode/top3_ratio": codebook_top3_ratio,
                                    "topcode/top5_ratio": codebook_top5_ratio,
                                    "topcode/top7_ratio": codebook_top7_ratio,
                                    "topcode/top9_ratio": codebook_top9_ratio,
                                    "topcode/topx_ratio": codebook_top10_ratio,
                                })
                                if use_wandb and wandb is not None:
                                     wandb.log(log_dict) # Log updated metrics including eval ones
                            except Exception as e:
                                print(f"Error during evaluation: {e}")
                        else:
                            # Other processes wait for main process to finish eval
                            codebook_usage = 0.0 # Default value if eval doesn't run
                        accelerator.wait_for_everyone() # Sync all processes after eval

                    # Periodic checkpointing (main process only)
                    if accelerator.is_main_process and (spoch + 1) % save_checkpoint_every_spoch == 0:
                        ckpt_path = f"{output_model_path}.spoch{spoch+1}.pth"
                        save_full_checkpoint(
                            path=ckpt_path,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler, # Pass scheduler state
                            epoch=epoch,
                            spoch=spoch,
                            global_step=global_step,
                            cnn_type=cnn_type,
                            accelerator=accelerator # Pass accelerator instance
                        )
                # --- End of if should_update_weights block ---
            else: # If not should_update (still accumulating)
                 # Still buffer losses for logging purposes during accumulation
                 loss_buffer["recon"].append(recon_loss.item())
                 loss_buffer["comit"].append(comit_loss.item())
                 loss_buffer["ortho"].append(ortho_loss.item())
                 loss_buffer["diver"].append(diver_loss.item())


    # Final save (main process only)
    if accelerator.is_main_process:
        save_full_checkpoint(
            path=output_model_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=num_epochs - 1,
            spoch=spoch,
            global_step=global_step,
            cnn_type=cnn_type,
            accelerator=accelerator
        )
        print(f"✅ Final model saved to {output_model_path}")
        if use_wandb and wandb is not None:
            wandb.finish()

    # Clean up Accelerator resources
    accelerator.end_training()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Nanopore VQ Tokenizer with Accelerate")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Pass the configuration dictionary values to vqe_train
    # Added new Accelerate-specific arguments
    vqe_train(
        train_npy_dir=config.get("train_npy_dir"),
        evaluation_npy_dir=config.get("evaluation_npy_dir"),
        output_model_path=config.get("output_model_path", "nanopore_vq_tokenizer.pth"),
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
        loss_log_interval=config.get("loss_log_interval", 10), # Note: This logic was replaced by update_loss_weight_every
        checkpoint_path=config.get("checkpoint_path"),
        cnn_type=config.get("cnn_type", 1),
        init_codebook_path=config.get("init_codebook_path", ""),
        cnn_checkpoint_path=config.get("cnn_checkpoint_path", ""),
        freeze_cnn=config.get("freeze_cnn", 0),
        learnable_codebook=config.get("learnable_codebook", True),
        global_batch_size=config.get("global_batch_size", 256),
        device_micro_batch_size=config.get("device_micro_batch_size", 16), # Passed again for clarity
        # Accelerate specific arguments
        mixed_precision=config.get("mixed_precision", "bf16"), # Options: "no", "fp16", "bf16"
        gradient_clipping=config.get("gradient_clipping", 1.0), # Set to None to disable
        cpu=config.get("cpu", False), # Force CPU training
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
