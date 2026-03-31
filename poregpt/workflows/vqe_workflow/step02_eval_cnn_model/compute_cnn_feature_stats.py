import os
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm

# 定义读取块大小 (例如每次处理 50,000 行)，平衡 IO 与内存
CHUNK_SIZE = 512 * 3000 # 修改为整个 batch 的向量数 (12,288,000)

def _process_chunk(data_chunk, current_stats):
    """
    处理单个数据块，更新统计量
    """
    n_i = data_chunk.shape[0]
    if n_i == 0:
        return current_stats

    total_n, global_mean, global_M2, global_min, global_max = current_stats

    # 1. 更新 Min/Max (按维度)
    shard_min = np.min(data_chunk, axis=0)
    shard_max = np.max(data_chunk, axis=0)
    global_min = np.minimum(global_min, shard_min)
    global_max = np.maximum(global_max, shard_max)

    # 2. Welford 算法更新 Mean/M2 (按维度)
    shard_mean = np.mean(data_chunk, axis=0, dtype=np.float64)
    shard_var = np.var(data_chunk, axis=0, ddof=0, dtype=np.float64)
    shard_M2 = shard_var * n_i

    if total_n == 0:
        global_mean = shard_mean
        global_M2 = shard_M2
    else:
        delta = shard_mean - global_mean
        total_n_new = total_n + n_i
        global_mean = global_mean + delta * n_i / total_n_new
        global_M2 = (
            global_M2 +
            shard_M2 +
            (delta ** 2) * (total_n * n_i) / total_n_new
        )

    return (total_n + n_i, global_mean, global_M2, global_min, global_max)

def _calculate_chunk_stats(data_chunk):
    """
    计算单个 chunk 的统计信息 (严格按维度)
    """
    n = data_chunk.shape[0]
    mean_per_dim = np.mean(data_chunk, axis=0, dtype=np.float64)
    std_per_dim = np.std(data_chunk, axis=0, dtype=np.float64)
    min_per_dim = np.min(data_chunk, axis=0)
    max_per_dim = np.max(data_chunk, axis=0)
    
    # 每个维度的统计信息作为一个字典
    per_dim_stats = []
    for d in range(data_chunk.shape[1]):
        per_dim_stats.append({
            "dim": int(d),
            "count": int(n),
            "mean": float(mean_per_dim[d]),
            "std": float(std_per_dim[d]),
            "min": float(min_per_dim[d]),
            "max": float(max_per_dim[d])
        })
    
    return per_dim_stats


def print_chunk_stats_table(chunk_stats_list, start_idx, end_idx, feature_dim, max_print_dims=10):
    """
    以表格形式打印 chunk 的统计信息
    """
    print(f"     Chunk [{start_idx}:{end_idx}] - {feature_dim} 维度统计信息:", flush=True)
    # 打印表头
    header = f"{'Dim':<6} {'Count':<12} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}"
    print(f"       {header}", flush=True)
    print(f"       {'-' * len(header)}", flush=True)
    
    # 打印数据行
    for i, stat in enumerate(chunk_stats_list):
        if i >= max_print_dims:
            print(f"       ... (省略 {feature_dim - max_print_dims} 个维度)", flush=True)
            break
        row = f"{stat['dim']:<6} {stat['count']:<12,} {stat['min']:<12.6f} {stat['max']:<12.6f} {stat['mean']:<12.6f} {stat['std']:<12.6f}"
        print(f"       {row}", flush=True)


def compute_feature_statistics(shard_dir: str, output_json: str, feature_dim: int):
    """
    计算特征目录的全局统计量 (使用 memmap 加载)
    """
    # === 1. 收集有效 .npy 文件 ===
    shard_files = sorted([
        f for f in Path(shard_dir).glob("*.npy")
        if f.name != "shards.json" and not f.name.startswith(".")
    ])

    if not shard_files:
        raise ValueError(f"目录 {shard_dir} 中未找到有效的 .npy 特征文件！")

    print(f"🔍 找到 {len(shard_files)} 个特征分片，开始统计...", flush=True)

    # === 2. 初始化全局统计量 (按维度) ===
    total_n = 0
    global_mean = np.zeros(feature_dim, dtype=np.float64)
    global_M2 = np.zeros(feature_dim, dtype=np.float64)
    global_min = np.full(feature_dim, np.inf, dtype=np.float32)
    global_max = np.full(feature_dim, -np.inf, dtype=np.float32)

    # 存储每个文件的详细信息
    per_file_detailed_info = []

    # === 3. 流式处理每个 shard (memmap + 分块) ===
    # 外层进度条：遍历分片
    with tqdm(total=len(shard_files), desc="处理分片", unit="shard") as pbar_shards:
        for i, shard_path in enumerate(shard_files, 1):

            # 使用 memmap 加载当前分片
            current_memmap = np.memmap(shard_path, dtype='float32', mode='r')

            # --- 对当前分片应用相同的重塑逻辑 ---
            if current_memmap.ndim == 1:
                current_original_length = current_memmap.shape[0]
                if current_original_length % feature_dim != 0:
                    raise ValueError(
                        f"分片 {shard_path.name} 的长度 ({current_original_length}) 不能被 --feature-dim ({feature_dim}) 整除。"
                    )
                current_data_to_process = current_memmap.reshape(-1, feature_dim)
            elif current_memmap.ndim >= 2:
                if current_memmap.shape[-1] != feature_dim:
                     raise ValueError(
                         f"分片 {shard_path.name} 的最后一维 ({current_memmap.shape[-1]}) 与 --feature-dim ({feature_dim}) 不符。"
                     )
                current_data_to_process = current_memmap
            else:
                 raise ValueError(f"分片 {shard_path.name} 的维度 {current_memmap.ndim} 不符合要求。")

            total_rows_original = current_data_to_process.shape[0]
            
            # --- 核心修改：根据大小决定是否截断 ---
            if total_rows_original < CHUNK_SIZE:
                # 如果文件总样本数小于 CHUNK_SIZE，直接使用全部数据
                print(f"ℹ️  分片 '{shard_path.name}' 样本数 ({total_rows_original}) 小于 CHUNK_SIZE ({CHUNK_SIZE})，使用全部数据。", flush=True)
                current_data_to_process = current_data_to_process
                total_rows = total_rows_original
            else:
                # 如果文件总样本数大于等于 CHUNK_SIZE，执行原来的截断逻辑
                total_rows_truncated = (total_rows_original // CHUNK_SIZE) * CHUNK_SIZE
                
                if total_rows_truncated != total_rows_original:
                    print(f"⚠️  分片 '{shard_path.name}' 原始样本数 {total_rows_original} "
                          f"不能被 CHUNK_SIZE {CHUNK_SIZE} 整除。已截断至 {total_rows_truncated}。"
                          f"丢弃了 {total_rows_original - total_rows_truncated} 个样本。", flush=True)
                    
                # 对数据进行切片，只保留可以整除的部分
                current_data_to_process = current_data_to_process[:total_rows_truncated]
                total_rows = current_data_to_process.shape[0]

            # --- 为当前文件初始化临时统计量 (按维度) ---
            file_total_n = 0
            file_global_mean = np.zeros(feature_dim, dtype=np.float64)
            file_global_M2 = np.zeros(feature_dim, dtype=np.float64)
            file_global_min = np.full(feature_dim, np.inf, dtype=np.float32)
            file_global_max = np.full(feature_dim, -np.inf, dtype=np.float32)

            # 存储当前文件所有 chunks 的统计信息
            file_chunks_stats = []

            # 内层进度条：处理当前分片内的块 (用于当前文件的统计)
            # 关键修改：移除内层的 tqdm with 语句块，改为手动更新
            pbar_chunk = tqdm(total=total_rows, desc=f"分片 {i} (文件统计)", leave=False, unit="row")
            for start_idx in range(0, total_rows, CHUNK_SIZE):
                end_idx = start_idx + CHUNK_SIZE # 因为已经整除或小于chunk，这里end_idx不会超过total_rows
                chunk = current_data_to_process[start_idx:end_idx]
                if not chunk.flags['C_CONTIGUOUS']:
                    chunk = np.ascontiguousarray(chunk)

                # --- 计算当前 chunk 的统计信息 (按维度) ---
                chunk_per_dim_stats = _calculate_chunk_stats(chunk)
                
                # 打印当前 chunk 的统计信息 (按维度，格式化表格)
                print_chunk_stats_table(chunk_per_dim_stats, start_idx, end_idx, feature_dim)

                # 将 chunk 统计信息添加到当前文件列表中
                file_chunks_stats.append({
                    "start_index": int(start_idx),
                    "end_index": int(end_idx),
                    "dimensions": chunk_per_dim_stats
                })

                # 更新当前文件的统计量 (按维度)
                file_total_n, file_global_mean, file_global_M2, file_global_min, file_global_max = _process_chunk(
                    chunk, (file_total_n, file_global_mean, file_global_M2, file_global_min, file_global_max)
                )
                # 同时更新全局统计量 (按维度)
                total_n, global_mean, global_M2, global_min, global_max = _process_chunk(
                    chunk, (total_n, global_mean, global_M2, global_min, global_max)
                )

                pbar_chunk.update(len(chunk))
            pbar_chunk.close() # 手动关闭进度条

            # --- 计算并打印当前文件的累积统计信息 (按维度) ---
            file_global_std = np.sqrt(file_global_M2 / file_total_n)
            
            print(f"\n--- 文件统计汇总: {shard_path.name} ---", flush=True)
            print(f"  样本数量: {file_total_n:,}, 特征维度: {feature_dim}", flush=True)
            print(f"  每个维度的 Min/Max/Mean/Std:", flush=True)
            # 打印表头
            header = f"{'Dim':<6} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}"
            print(f"    {header}", flush=True)
            print(f"    {'-' * len(header)}", flush=True)
            # 打印数据行
            for d in range(min(10, feature_dim)): # 只打印前10个维度，避免刷屏
                row = f"{d:<6} {file_global_min[d]:<12.6f} {file_global_max[d]:<12.6f} {file_global_mean[d]:<12.6f} {file_global_std[d]:<12.6f}"
                print(f"    {row}", flush=True)
            if feature_dim > 10:
                print(f"    ... (省略 {feature_dim - 10} 个维度)", flush=True)
            print("-" * 60, flush=True)

            # 将当前文件的汇总统计和所有 chunk 统计信息保存
            per_file_detailed_info.append({
                "filename": shard_path.name,
                "samples_processed": int(file_total_n),
                "samples_original": int(total_rows_original),
                "is_truncated": total_rows_original > CHUNK_SIZE,
                "summary_stats": {
                    "count": int(file_total_n),
                    "per_dim": [
                        {
                            "dim": d,
                            "mean": float(file_global_mean[d]),
                            "std": float(file_global_std[d]),
                            "min": float(file_global_min[d]),
                            "max": float(file_global_max[d])
                        }
                        for d in range(feature_dim)
                    ]
                },
                "chunks": file_chunks_stats # 将所有 chunk 的按维度统计信息放在这里
            })

            # 显式删除 memmap 对象
            del current_memmap, current_data_to_process

            # 更新外层进度条
            pbar_shards.set_postfix_str(f"累计样本: {total_n:,}")
            pbar_shards.update(1)

    # === 4. 计算最终全局统计量 (按维度) ===
    global_std = np.sqrt(global_M2 / total_n)

    # 构建结果字典
    stats = {
        "total_tokens_processed": int(total_n),
        "total_tokens_discarded": sum(
            total_rows_original - ((total_rows_original // CHUNK_SIZE) * CHUNK_SIZE)
            for total_rows_original in [np.memmap(shard_path, dtype='float32', mode='r').size // feature_dim for shard_path in shard_files]
        ) if any(np.memmap(sh, dtype='float32', mode='r').size // feature_dim >= CHUNK_SIZE for sh in shard_files) else 0,
        "feature_dim": int(feature_dim),
        "global_stats_per_dim": [
            {
                "dim": d,
                "min": float(global_min[d]),
                "max": float(global_max[d]),
                "mean": float(global_mean[d]),
                "std": float(global_std[d])
            }
            for d in range(feature_dim)
        ],
        "per_file_detailed_info": per_file_detailed_info # 包含所有文件和它们的 chunk 详情
    }

    # === 5. 输出结果 ===
    print("\n✅ 统计完成！关键摘要:", flush=True)
    print(f"   总处理 Token 数: {total_n:,}", flush=True)
    print(f"   总丢弃 Token 数: {stats['total_tokens_discarded']:,}", flush=True)
    print(f"   特征维度: {feature_dim}", flush=True)
    print(f"   按维度统计信息已计算。", flush=True)
    print(f"   前5个维度的全局 Min/Max/Mean/Std:", flush=True)
    # 打印表头
    header = f"{'Dim':<6} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}"
    print(f"    {header}", flush=True)
    print(f"    {'-' * len(header)}", flush=True)
    # 打印数据行
    for d in range(min(10, feature_dim)):
        g_stat = stats["global_stats_per_dim"][d]
        row = f"{d:<6} {g_stat['min']:<12.6f} {g_stat['max']:<12.6f} {g_stat['mean']:<12.6f} {g_stat['std']:<12.6f}"
        print(f"    {row}", flush=True)
    if feature_dim > 5:
        print(f"    ... (省略 {feature_dim - 5} 个维度)", flush=True)

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n💾 详细统计已保存至: {output_json}", flush=True)

    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计特征目录的全局维度级统计量 (Memmap优化版)")
    parser.add_argument("shard_dir", type=str, help="包含 .npy 特征分片的目录")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="保存统计结果的 JSON 路径")
    parser.add_argument("--feature-dim", type=int, required=True,
                        help="特征维度。如果输入文件是一维的，将按此维度重塑为 (N, feature_dim)。")
    args = parser.parse_args()

    compute_feature_statistics(args.shard_dir, args.output, args.feature_dim)
