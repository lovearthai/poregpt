#!/bin/bash

# --- 配置区域 ---
# Python 解释器路径 (如有需要可调整，例如使用完整路径 /usr/bin/python3)
PYTHON_CMD="python3"

# 包含 .jsonl.gz 文件的目录
JSONLGZ_DIR="/mnt/nas_syy/default/poregpt/dataset/human_dna_032g/memap_mongoq30/jsonlgz_vqe340s147000" # 请将此路径修改为你的实际数据目录

# 码表大小
CODEBOOK_SIZE=390625 # 如果你的码表大小不同，请修改此值

# 并行处理的进程数 (可选)
# 如果留空或注释掉，Python 脚本将自动使用 CPU 核心数
NUM_PROCESSES=32 # 例如: NUM_PROCESSES=4

# Python 脚本文件名
SCRIPT_FILE="tongji_jsonlgz.py" # 确保此名称与你保存的 Python 代码文件名一致

# --- 配置结束 ---

echo "正在启动 token 频次统计脚本..."
echo "Python 命令: $PYTHON_CMD"
echo "脚本文件: $SCRIPT_FILE"
echo "数据目录: $JSONLGZ_DIR"
echo "码表大小: $CODEBOOK_SIZE"
if [[ -n "$NUM_PROCESSES" ]]; then
    echo "并行进程数: $NUM_PROCESSES"
else
    echo "并行进程数: 未指定 (将使用 CPU 核心数)"
fi
echo ""

# 检查 Python 脚本文件是否存在
if [[ ! -f "$SCRIPT_FILE" ]]; then
    echo "错误: 当前目录下未找到 Python 脚本 '$SCRIPT_FILE'。"
    exit 1
fi

# 检查数据目录是否存在
if [[ ! -d "$JSONLGZ_DIR" ]]; then
    echo "错误: 数据目录 '$JSONLGZ_DIR' 不存在。"
    exit 1
fi

# 构建命令参数
CMD_ARGS=("$SCRIPT_FILE" "$JSONLGZ_DIR" "$CODEBOOK_SIZE")
if [[ -n "$NUM_PROCESSES" ]]; then
    CMD_ARGS+=("--num_processes" "$NUM_PROCESSES")
fi

# 执行 Python 脚本
$PYTHON_CMD "${CMD_ARGS[@]}"

# 检查 Python 脚本的退出状态
if [[ $? -eq 0 ]]; then
    echo ""
    echo "Python 脚本执行成功！"
    echo "请检查生成的 'token_frequencies_sorted_by_count_desc.csv' 文件。"
else
    echo ""
    echo "错误: Python 脚本执行失败，退出状态码为 $?。请查看上方日志。"
    exit 1
fi
