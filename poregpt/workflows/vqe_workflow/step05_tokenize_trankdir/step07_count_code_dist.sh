#!/bin/bash

# ================= 配置区域 (在此处修改) =================
DATA_DIR="/mnt/nas_syy/default/poregpt/dataset/human_dna_595g/memap_mongoq30/jsonlgz_vqe311s35000/validation"       # 你的数据目录路径
WORKERS=32                     # 并行线程数
OUTPUT_FILE="token_freq_triple.png" # 输出图片文件名
# =======================================================

echo "开始执行统计..."
echo "数据目录: $DATA_DIR"
echo "线程数: $WORKERS"
echo "输出文件: $OUTPUT_FILE"


# 运行 Python 脚本 (硬编码文件名 step07_count_code_dist.py)
python3 step07_count_code_dist.py "$DATA_DIR" --workers "$WORKERS" --output "$OUTPUT_FILE"

# 退出环境

echo "任务完成。"
