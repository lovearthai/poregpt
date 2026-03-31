#!/bin/bash

# ========================================
# Simple Shell Script to Run Recursive TSV/NPY Merger
# All parameters are configured within this script.
# Assumes the Python script is in the same directory named 'merge_tsv_and_npy_recursive.py'
# ========================================

set -euo pipefail # Exit on error, undefined var, pipe failure

# --- Configuration (Modify these values directly) ---
ROOT_DIRECTORY="/mnt/nas_syy/default/poregpt/dataset/human_dna_595g/basecall" # 修改为你的数据根目录
NUM_PROCESSES=32              # 手动设置一个数字，或使用 $(nproc) 来使用所有 CPU 核心
PYTHON_SCRIPT_NAME="merge_tsv_and_npy_to_bccsv.py" # 修改为你的 Python 脚本名称

# --- Main Execution ---
echo "========================================="
echo "Starting Recursive TSV/NPY Merge Process"
echo "========================================="
echo "Root Directory: $ROOT_DIRECTORY"
echo "Number of Processes: $NUM_PROCESSES"
echo "Python Script: $PYTHON_SCRIPT_NAME"
echo "========================================="

# Check if the Python script exists in the current directory
if [[ ! -f "./$PYTHON_SCRIPT_NAME" ]]; then
    echo "❌ Error: Python script '$PYTHON_SCRIPT_NAME' not found in the current directory." >&2
    exit 1
fi

# Check if root directory exists
if [[ ! -d "$ROOT_DIRECTORY" ]]; then
    echo "❌ Error: Root directory '$ROOT_DIRECTORY' does not exist." >&2
    exit 1
fi

# Run the Python script with arguments
echo "🚀 Launching Python script..."
python3 "$PYTHON_SCRIPT_NAME" "$ROOT_DIRECTORY" --num_processes "$NUM_PROCESSES"

echo "🎉 Done!"
