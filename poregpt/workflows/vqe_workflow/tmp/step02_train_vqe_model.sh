#!/bin/bash

# ============================================================================
# 脚本名称: train_vq.sh
# 功能:     调用 test/test_vq_tokenizer/test_train_vq_model.py，
#          训练一个 Vector Quantization (VQ) tokenizer 模型，
#          用于将 Nanopore 信号片段（.npy）映射到离散 token。
#
# 依赖:
#   - Python 3.8+ 环境已激活（含 PyTorch、nanopore_signal_tokenizer 等）
#   - 项目根目录下存在 test/test_vq_tokenizer/test_train_vq_model.py
#   - 已通过 pip install -e . 安装 nanopore_signal_tokenizer 包
#
# 作者:     Your Name
# 日期:     2025-06-10
# ============================================================================

export WANDB_API_KEY=748830e9b9acdf804bb0baad0eb82e6ca2592354
# ----------------------------------------------------------------------------
# 配置区：所有训练参数集中在此处，便于修改、版本控制和实验管理。
# 每个变量对应 Python argparse 中的一个参数。
# ----------------------------------------------------------------------------

# 输入目录：包含预处理好的 .npy 信号片段文件（由 step01_fast5_to_chunks.py 生成）
TRAIN_NPTY_DIR="/mnt/nas_syy/default/huada_signal_llm/dataset/dna/human_min0_max2_read96655/memap/train"
EVAL_NPTY_DIR="/mnt/nas_syy/default/huada_signal_llm/dataset/dna/human_min0_max2_read96655/memap/validation"
# 输出模型保存路径（.pth 文件）
mkdir models
OUTPUT_MODEL_PATH="models/porepgt_vqe_tokenizer"


CHECKPOINT_PATH="models_old/nanopore_signal_tokenizer.pth.spoch26000.pth"

# 训练批大小（影响显存占用）

# 当CODEBOOK_SIZE==65536时
BATCH_SIZE=8

GLOBAL_BATCH_SIZE=256
DEVICE_MICRO_BATCH_SIZE=8

# 学习率（Adam 优化器）
# scratch
#LR=0.0002  # 即 2e-4

# init all
LR=0.00005  # 即 5e-5

# init book
#LR=0.0002  # 即 2e-4


# 训练轮数（epochs）
NUM_EPOCHS=50

# VQ 码本大小（即词汇表大小，如 8192 = 2^13）
CODEBOOK_SIZE=16384
CODEBOOK_SIZE=65536

# VQ 码本大小（即词汇表大小，如 8192 = 2^13）
CODEBOOK_DIM=64

# 每个 .npy 文件的信号长度（必须与预处理时的 window_size 一致）
CHUNK_SIZE=12000

# 数据加载线程数（加速 I/O）
NUM_WORKERS=16

# 验证集大小（从数据中随机抽取多少样本用于评估）
VAL_RATIO=0.01

# 验证集大小（从数据中随机抽取多少样本用于评估）
SAVE_CHECKPOINT_INTERVAL=500

LOSS_LOG_INTERVAL=10

# 是否启用 codebook 使用率评估（会额外计算并打印每个码字的使用频率）
DO_EVALUATE="true"  # 设为 "true" 启用 --do_evaluate，"false" 则不启用

CNN_TYPE=1

INIT_CODEBOOK_PATH="/mnt/nas_syy/default/poregpt/shared/kmc_models_apple_c64k_redo1/memap_train_clustered_10p_centroids_k65536.npy"
#INIT_CODEBOOK_PATH="no_codebook"
INIT_CNN_PATH="/mnt/nas_syy/default/poregpt/shared/cnn_models_apple_type1/checkpoints/nanopore_signal_tokenizer.pth.epoch28.pth"
#INIT_CNN_PATH="no_cnn"

VQ_TRAIN_COMMITMENT_WEIGHT=0.250
VQ_TRAIN_ORTHOGONAL_WEIGHT=0.001
VQ_TRAIN_DIVERSITY_WEIGHT=0.00001
VQ_TRAIN_ORTHOGONAL_WEIGHT=0.000
VQ_TRAIN_DIVERSITY_WEIGHT=0.00000


# ----------------------------------------------------------------------------
# 构建 --do_evaluate 参数
# 因为这是一个开关（store_true），所以只在需要时添加该 flag
# ----------------------------------------------------------------------------
EVALUATE_ARG=""
if [ "$DO_EVALUATE" = "true" ]; then
    EVALUATE_ARG="--do_evaluate"
fi

# ----------------------------------------------------------------------------
# 构造完整的 Python 命令
# 使用数组形式避免空格/引号问题，确保命令安全可靠
# ----------------------------------------------------------------------------
CMD=(
    torchrun --nproc_per_node=4  
    -m poregpt.tokenizers.vqe_tokenizer.vqe_train
    --train_npy_dir "$TRAIN_NPTY_DIR"
    --evaluation_npy_dir "$EVAL_NPTY_DIR"
    --output_model_path "$OUTPUT_MODEL_PATH"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --num_epochs "$NUM_EPOCHS"
    --codebook_size "$CODEBOOK_SIZE"
    --chunk_size "$CHUNK_SIZE"
    --num_workers "$NUM_WORKERS"
    --val_ratio "$VAL_RATIO"
    --save_checkpoint_every_spoch $SAVE_CHECKPOINT_INTERVAL
    --loss_log_interval $LOSS_LOG_INTERVAL
    --commitment_weight $VQ_TRAIN_COMMITMENT_WEIGHT
    --codebook_diversity_loss_weight $VQ_TRAIN_DIVERSITY_WEIGHT
    --orthogonal_reg_weight $VQ_TRAIN_ORTHOGONAL_WEIGHT
    --checkpoint_path $CHECKPOINT_PATH
    --cnn_type $CNN_TYPE
    --init_codebook_path $INIT_CODEBOOK_PATH
    --cnn_checkpoint_path $INIT_CNN_PATH
    --freeze_cnn 0
    --global_batch_size $GLOBAL_BATCH_SIZE 
    --device_micro_batch_size $DEVICE_MICRO_BATCH_SIZE
    $EVALUATE_ARG 
)


# ----------------------------------------------------------------------------
# 打印完整命令（用于调试和复现）
# printf "%q" 会对参数进行 shell 转义，确保输出的命令可直接复制运行
# ----------------------------------------------------------------------------
echo ">>> Running VQ training command:"
printf "%q " "${CMD[@]}"
echo  # 换行
echo "--------------------------------------------------"

# ----------------------------------------------------------------------------
# 执行训练命令
# 如果训练失败（非零退出码），脚本将停止（可配合 set -e 更严格）
# ----------------------------------------------------------------------------
"${CMD[@]}"


# ----------------------------------------------------------------------------
# 成功提示
# ----------------------------------------------------------------------------
echo "VQE tokenizer training completed. Model saved to: $OUTPUT_MODEL_PATH"
