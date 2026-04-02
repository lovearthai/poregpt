import argparse
import glob
import gzip
import json
import re
from collections import Counter
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm
import os

def count_tokens_in_file(filepath_with_total_lines):
    """统计单个 .jsonl.gz 文件中的 token 出现次数。"""
    filepath, estimated_total_lines = filepath_with_total_lines
    
    local_counts = Counter()
    # 正则表达式匹配 <|bwav:123456|> 格式的模式
    token_pattern = re.compile(r'<\|bwav:(\d+)\|>')
    
    # 如果未提供预估行数，则尝试估算，以获得更好的进度条效果
    if estimated_total_lines <= 0:
        # 备选方案：通过读取文件一次来估算（对非常大的文件可能较慢）
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                estimated_total_lines = sum(1 for _ in f)
        except Exception:
             estimated_total_lines = 0 # 如果估算失败，进度条将只显示当前计数

    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            # 为此文件创建一个 tqdm 实例
            pbar = tqdm(
                iterable=f,
                total=estimated_total_lines,
                desc=f"正在处理 {os.path.basename(filepath)}",
                leave=False, # 完成后隐藏此进度条
                unit="行"
            )
            for line in pbar:
                data = json.loads(line.strip())
                text = data.get('text', '')
                # 在文本字符串中查找所有 token ID
                matches = token_pattern.findall(text)
                # 将匹配项转换为整数并更新计数器
                local_counts.update(int(match) for match in matches)
    except Exception as e:
        print(f"处理文件 {filepath} 时出错: {e}")
        
    return local_counts

def estimate_line_count(filepath):
    """辅助函数，用于估算 gzipped 文件中的行数。"""
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            # 一种快速计算行数而不将所有内容加载到内存中的方法
            count = sum(1 for _ in f)
        return count
    except Exception:
        return 0 # 如果估算失败则返回 0

def main():
    parser = argparse.ArgumentParser(
        description='统计 .jsonl.gz 文件中的 token 频次并计算权重。'
                    '包含码表空间中的所有码，即使那些出现频率为零的码。'
                    '显示单个文件和整体处理的进度条。'
    )
    parser.add_argument('jsonlgz_dir', type=str, help='包含 .jsonl.gz 文件的目录。')
    parser.add_argument('codebook_size', type=int, help='码表空间的总大小 (例如, 390625)。')
    parser.add_argument('--num_processes', type=int, default=None, 
                        help='用于并行处理的进程数。如果未指定，将使用 CPU 核心数。')
    args = parser.parse_args()

    # 验证 codebook_size
    if args.codebook_size <= 0:
        raise ValueError("码表大小必须是正整数。")

    # 递归查找所有 .jsonl.gz 文件
    pattern = f"{args.jsonlgz_dir}/**/*.jsonl.gz"
    file_paths = glob.glob(pattern, recursive=True)
    print(f"找到了 {len(file_paths)} 个 .jsonl.gz 文件。")

    if not file_paths:
        print("在指定目录中未找到 .jsonl.gz 文件。")
        return

    # 估算每个文件的行数，以便更好地显示进度条
    print("正在估算进度条所需的行数...")
    estimated_lines_list = []
    for fp in file_paths:
        estimated_lines_list.append(estimate_line_count(fp))
    
    # 将文件路径与其预估行数组合成列表
    file_info_list = list(zip(file_paths, estimated_lines_list))

    # 确定使用的进程数
    num_processes = args.num_processes or cpu_count()
    # 不要创建比文件数量还多的进程
    num_processes = min(num_processes, len(file_paths)) 
    print(f"使用 {num_processes} 个进程进行并行计数。")
    
    # 为映射操作创建一个全局 tqdm 实例
    with Pool(processes=num_processes) as pool:
        # 对文件列表的主循环使用 tqdm.tqdm
        chunk_results = list(tqdm(
            pool.imap(count_tokens_in_file, file_info_list),
            total=len(file_info_list),
            desc="总体进度",
            unit="文件"
        ))
    
    # 将来自所有进程的结果聚合到一个 Counter 中
    total_counts = Counter()
    for chunk_counter in chunk_results:
        total_counts.update(chunk_counter)

    # 计算找到的 token 总数
    total_tokens_found = sum(total_counts.values())
    print(f"\n在所有文件中找到的 token 总数: {total_tokens_found}")

    # 定义均匀分布代码的期望计数
    expected_count_per_code = total_tokens_found / args.codebook_size
    print(f"每个代码的期望计数 (如果是均匀分布): {expected_count_per_code:.2f}")

    # 为 DataFrame 准备数据，覆盖所有可能的代码 (0 到 codebook_size - 1)
    all_possible_codes = list(range(args.codebook_size))
    
    # 创建 DataFrame 的列列表
    token_ids = []
    counts = []
    weights = []

    for code in all_possible_codes:
        # 获取计数，默认为 0（如果未找到）
        count_for_code = total_counts.get(code, 0) 
        # 计算权重
        weight_for_code = count_for_code / expected_count_per_code if expected_count_per_code > 0 else 0.0
        
        token_ids.append(code)
        counts.append(count_for_code)
        weights.append(weight_for_code)

    # 创建 DataFrame
    df = pd.DataFrame({
        'token_id': token_ids,
        'count': counts,
        'weight': weights
    })

    # 按 'count' (出现频次) 降序排序 (最高频率在前)
    df.sort_values(by='count', ascending=False, inplace=True)
    # 排序后重置索引，以获得干净的从 0 开始的索引，并删除旧索引
    df.reset_index(drop=True, inplace=True)

    # 将排序后的 DataFrame 保存为 CSV
    output_filename = f"token_frequencies_sorted_by_count_desc.csv"
    df.to_csv(output_filename, index=False)
    print(f"按频次降序排序的结果已保存到 {output_filename}")

if __name__ == '__main__':
    main()
