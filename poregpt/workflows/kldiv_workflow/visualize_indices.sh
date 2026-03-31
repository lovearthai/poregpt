#!/bin/bash

# --- 基础配置 ---
PYTHON_EXEC=""
SCRIPT_PATH=""

# --- 输入路径 ---
# 请确保该目录下包含多个 .pkl 文件
SOURCE_DIR="/mnt/zzbnew/rnamodel/dengyiting/indices/20260313_111934"
CSV_REPORT="/mnt/zzbnew/rnamodel/dengyiting/usage_ratio_report.csv"

# --- 输出配置 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 我们将热图子目录名包含时间戳，方便区分多次运行的结果
HEATMAP_SUBDIR="heatmaps_$TIMESTAMP"
USE_LOG_SCALE=True

echo "------------------------------------------------"
echo "📊 Starting Visualization Task (Python Mode)"
echo "Time: $(date)"
echo "Source: $SOURCE_DIR"
echo "------------------------------------------------"

# 执行绘图逻辑
# 注意：--use_log_scale 在 python 中是 action="store_true"，直接加上即可启用
python visualize_indices.py \
    --source_dir "$SOURCE_DIR" \
    --csv_path "$CSV_REPORT" \
    --output_subdir "$HEATMAP_SUBDIR" \
    --grid_size 256 \
    --use_log_scale $USE_LOG_SCALE

echo "------------------------------------------------"
echo "✨ All Visualizations Finished!"
echo "1. KL Divergence Trend: codebook_kl_divergence_trend.png"
echo "2. Heatmaps Directory: $HEATMAP_SUBDIR"
echo "------------------------------------------------"