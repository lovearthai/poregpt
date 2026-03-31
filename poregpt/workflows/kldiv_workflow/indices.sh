#!/bin/bash

# --- 基础配置 ---
PYTHON_EXEC="python" 
SCRIPT_PATH="/mnt/zzbnew/rnamodel/dengyiting/workflow/indeces.py"

# --- 路径参数 ---
CKPT_ROOT="/mnt/si003067jezr/default/poregpt/poregpt/poregpt/workflows/vqe_workflow/step02_train_vqe_model/pass126_w64_c64k_cnn07_dcw01_dna595g_lr4e5_mongoq30_m4_scratch_freeze10k/models"
VAL_DATA="/mnt/si003067jezr/default/poregpt/dataset/human_dna_595g/memap_lemon5/validation"
SAVE_DIR="/mnt/zzbnew/rnamodel/dengyiting/workflow"

# --- 超参数 ---
BATCH_SIZE=8
RATIO=0.01
GPU_ID=0  # 指定使用的显卡 ID
CODEBOOK_SIZE=65536
RANDOM_SEED=42
USE_LOCAL_SUBSET=False


echo "------------------------------------------------"
echo "🚀 Starting Workflow: Codebook Index Extraction (Python Mode)"
echo "Time: $(date)"
echo "GPU ID: $GPU_ID"
echo "------------------------------------------------"

# 直接使用 python 调用
# 环境变量 CUDA_VISIBLE_DEVICES 确保程序只看到指定的显卡
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_EXEC $SCRIPT_PATH \
    --ckpt_root "$CKPT_ROOT" \
    --val_data_dir "$VAL_DATA" \
    --batch_size $BATCH_SIZE \
    --sample_ratio $RATIO \
    --cnn_type 7 \
    --codebook_size $CODEBOOK_SIZE \
    --random_seed $RANDOM_SEED \
    --use_local_subset $USE_LOCAL_SUBSET \
    --save_dir $SAVE_DIR

echo "------------------------------------------------"
echo "✅ Workflow Finished at $(date)"
echo "------------------------------------------------"