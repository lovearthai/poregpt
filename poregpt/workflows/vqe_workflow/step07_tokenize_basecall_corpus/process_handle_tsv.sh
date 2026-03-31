#!/bin/bash

# ========================================
# Shell Script to Run Recursive TSV Processor
# ========================================

set -euo pipefail # Exit on error, undefined var, pipe failure

# --- Configuration (Modify these values directly) ---
ROOT_DIRECTORY="/mnt/nas_syy/default/poregpt/dataset/human_dna_595g/basecall" # 修改为你的数据根目录
PYTHON_SCRIPT_NAME="process_handle_tsv.py" # 修改为你的 Python 脚本名称

# --- Main Execution ---
echo "========================================="
echo "Starting Recursive TSV Processing"
echo "========================================="
echo "Root Directory: $ROOT_DIRECTORY"
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

# Run the Python script
echo "🚀 Launching Python script..."
python3 "$PYTHON_SCRIPT_NAME" "$ROOT_DIRECTORY"

echo "🎉 Done!"
