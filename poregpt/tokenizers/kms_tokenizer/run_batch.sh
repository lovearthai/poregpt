#!/bin/bash

# ==============================
# Nanopore RVQ Tokenizer for Basecall Corpus - Parallel with Skip Existing & Direct Output
# ==============================

# --- 配置区域 ---
FAST5_DIR="/mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/fast5/validation" 
OUTPUT_ROOT="/mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/fast5/validation/1" 
MODEL_CKPT="/mnt/nas_syy/default/olmo_pt_bioseq/olmo_pt_020m_bwavkms8k_rna_80g/HF_20m_DNA_KMS16K_W16S3_V20260121/encoder/centroids_meta.npz"
PYTHON_SCRIPT_PATH="/path/to/your/vqe_tokenize_single_fast5.py" # 修改为你的 Python 脚本的实际路径
NUM_GPUS=4
MAX_CONCURRENT=32  # 总并发数
SIGNAL_STRATEGY="apple"
TOKEN_BATCH_SIZE=8000
# --- 配置区域结束 ---

mkdir -p "$OUTPUT_ROOT"

# 获取所有 .fast5 文件（递归）
mapfile -d '' all_files < <(find "$FAST5_DIR" -name "*.fast5" -print0)

if [ ${#all_files[@]} -eq 0 ]; then
    echo "❌ No .fast5 files found in $FAST5_DIR." >&2
    exit 1
fi

echo "🔍 Found ${#all_files[@]} .fast5 files in $FAST5_DIR. Running up to $MAX_CONCURRENT tasks concurrently..."

task_count=0
total=${#all_files[@]}
processed=0
skipped=0

for ((i=0; i<total; i++)); do
    fast5_path="${all_files[i]}"

    # 推断对应的 .bc.csv 文件路径
    csv_path="${fast5_path%.fast5}.bc.csv"

    # 检查 .bc.csv 文件是否存在
    if [ ! -f "$csv_path" ]; then
        echo "⚠️  Skipping $fast5_path: corresponding .bc.csv file ($csv_path) not found." >&2
        ((skipped++)) # 认为找不到 CSV 的也算跳过
        continue
    fi

    # 构造输出路径 (基于 FAST5 文件路径)
    output_file="${fast5_path%.fast5}.jsonl.gz"
    output_dir="$(dirname "$output_file")"

    # ✅ 如果目标文件已存在，跳过
    if [ -f "$output_file" ]; then
        echo "⏭️  Skipping $output_file due to already existing." >&2
        ((skipped++))
        continue
    fi

    mkdir -p "$output_dir"

    # 分配 GPU
    gpu_id=$(( task_count % NUM_GPUS ))

    # 控制并发
    if (( task_count >= MAX_CONCURRENT )); then
        wait -n # 等待任意一个后台任务结束
    fi

    # 启动任务：✅ 不重定向日志，直接输出
    echo "➡️  Submitting $(basename "$fast5_path") (CSV: $(basename "$csv_path")) to GPU $gpu_id (output: $output_file)" >&2
    
    # 启动 Python 脚本
    poregpt-kms-tokenize-basecall-corpus \
        --fast5_path "$fast5_path" \
        --csv_path "$csv_path" \
        --model_path "$MODEL_CKPT" \
        --device "$gpu_id" \
        --signal_strategy "$SIGNAL_STRATEGY" & # 注意：token_batch_size 是在 Python 脚本内部硬编码的，或者也需要通过命令行传递

    ((task_count++))
    ((processed++)) # 只有提交了任务才算处理
done

# 等待所有后台任务完成
wait

echo "🎉 Done. Processed: $processed, Skipped (missing CSV or already exist): $skipped, Total scanned .fast5: $total" >&2
