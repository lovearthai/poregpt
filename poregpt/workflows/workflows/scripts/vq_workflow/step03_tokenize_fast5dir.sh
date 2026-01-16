#!/bin/bash

# ==============================
# Nanopore RVQ Tokenizer - Continuous Parallel (Max 32 concurrent)
# ==============================

FAST5_DIR="fast5"
OUTPUT_ROOT="fast5_jsonlgz"
MODEL_CKPT="models/nanopore_vq0_tokenizer.pth.epoch37.pth"
NUM_GPUS=4
MAX_CONCURRENT=32  # 总并发数（建议 = NUM_GPUS * 每卡安全并发数）
MEDF=5
LPF=0

mkdir -p "$OUTPUT_ROOT"

# 获取所有 .fast5 文件（递归）
mapfile -d '' all_files < <(find "$FAST5_DIR" -name "*.fast5" -print0)

if [ ${#all_files[@]} -eq 0 ]; then
    echo "❌ No .fast5 files found."
    exit 1
fi

echo "🔍 Found ${#all_files[@]} files. Running up to $MAX_CONCURRENT tasks concurrently..."

# 初始化任务计数器和 GPU 轮询索引
task_count=0
total=${#all_files[@]}

# 启动所有任务，但控制并发
for ((i=0; i<total; i++)); do
    fast5="${all_files[i]}"

    # 构造输出路径
    rel_path="${fast5#$FAST5_DIR/}"
    output_file="$OUTPUT_ROOT/${rel_path%.fast5}.jsonl.gz"
    mkdir -p "$(dirname "$output_file")"

    # 分配 GPU：按全局任务序号轮询（更均衡）
    gpu_id=$(( task_count % NUM_GPUS ))

    # 如果已达最大并发，等待任意一个任务结束
    if (( task_count >= MAX_CONCURRENT )); then
        wait -n  # 等待任意一个后台任务完成
    fi

    # 启动新任务
    echo "➡️  Submitting $(basename "$fast5") to GPU $gpu_id"
    python3 scripts/step03_vq0_tokenize_fast5.py \
        --fast5_file "$fast5" \
        --output_file "$output_file" \
        --model_ckpt "$MODEL_CKPT" \
        --gpu_id "$gpu_id" \
	--medf $MEDF 	\
    	--lpf $LPF	\
        > "${output_file}.log" 2>&1 &

    ((task_count++))
done

# 等待剩余所有任务完成
wait

echo "🎉 All $total files processed!"
