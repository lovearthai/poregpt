import os
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm

# 定义读取块大小 (例如每次处理 50,000 行)，平衡 IO 与内存
CHUNK_SIZE = 50000

def _process_chunk(data_chunk, current_stats):
    """
    处理单个数据块，更新统计量
    """
    n_i = data_chunk.shape[0]
    if n_i == 0:
        return current_stats

    total_n, global_mean, global_M2, global_min, global_max = current_stats

    # 1. 更新 Min/Max
    shard_min = np.min(data_chunk, axis=0)
    shard_max = np.max(data_chunk, axis=0)
    global_min = np.minimum(global_min, shard_min)
    global_max = np.maximum(global_max, shard_max)

    # 2. Welford 算法更新 Mean/M2
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

    print(f"🔍 找到 {len(shard_files)} 个特征分片，开始统计...")

    # === 2. 初始化全局统计量 (从第一个 shard 推断 feature_dim) ===
    # 使用 memmap 打开第一个文件
    first_memmap = np.memmap(shard_files[0], dtype='float32', mode='r')
    
    print(f"📄 第一个分片 '{shard_files[0].name}' 的原始形状: {first_memmap.shape}, 维度: {first_memmap.ndim}")

    # --- 新增逻辑：检查并重塑一维数组 ---
    if first_memmap.ndim == 1:
        original_length = first_memmap.shape[0]
        if original_length % feature_dim != 0:
            raise ValueError(
                f"第一个分片的长度 ({original_length}) 不能被指定的 --feature-dim ({feature_dim}) 整除。"
            )
        # 重塑为二维数组 (num_tokens, feature_dim)
        reshaped_memmap = first_memmap.reshape(-1, feature_dim)
        print(f"📐 一维数组已重塑为: {reshaped_memmap.shape}")
        data_to_use = reshaped_memmap
    elif first_memmap.ndim >= 2:
        if first_memmap.shape[-1] != feature_dim:
            raise ValueError(
                f"第一个分片的最后一维 ({first_memmap.shape[-1]}) 与指定的 --feature-dim ({feature_dim}) 不符。"
            )
        data_to_use = first_memmap
    else:
        raise ValueError(f"首个分片的维度 {first_memmap.ndim} 不符合要求，必须是一维或更高维度。")

    # --- 初始化统计量 ---
    inferred_feature_dim = data_to_use.shape[1]
    total_n = 0
    global_mean = np.zeros(inferred_feature_dim, dtype=np.float64)
    global_M2 = np.zeros(inferred_feature_dim, dtype=np.float64)
    global_min = np.full(inferred_feature_dim, np.inf, dtype=np.float32)
    global_max = np.full(inferred_feature_dim, -np.inf, dtype=np.float32)

    # === 3. 流式处理每个 shard (memmap + 分块) ===
    # 外层进度条：遍历分片
    with tqdm(total=len(shard_files), desc="处理分片", unit="shard") as pbar_shards:
        for i, shard_path in enumerate(shard_files, 1):
            # print(f"  正在处理分片 {i}/{len(shard_files)}: {shard_path.name}")

            # 使用 memmap 加载当前分片
            current_memmap = np.memmap(shard_path, dtype='float32', mode='r')
            
            # print(f"     当前分片原始形状: {current_memmap.shape}, 维度: {current_memmap.ndim}")

            # --- 对当前分片应用相同的重塑逻辑 ---
            if current_memmap.ndim == 1:
                current_original_length = current_memmap.shape[0]
                if current_original_length % feature_dim != 0:
                    raise ValueError(
                        f"分片 {shard_path.name} 的长度 ({current_original_length}) 不能被 --feature-dim ({feature_dim}) 整除。"
                    )
                current_data_to_process = current_memmap.reshape(-1, feature_dim)
                # print(f"     一维数组已重塑为: {current_data_to_process.shape}")
            elif current_memmap.ndim >= 2:
                if current_memmap.shape[-1] != feature_dim:
                     raise ValueError(
                         f"分片 {shard_path.name} 的最后一维 ({current_memmap.shape[-1]}) 与 --feature-dim ({feature_dim}) 不符。"
                     )
                current_data_to_process = current_memmap
            else:
                 raise ValueError(f"分片 {shard_path.name} 的维度 {current_memmap.ndim} 不符合要求。")

            total_rows = current_data_to_process.shape[0]
            # 记录当前分片的起始累计样本数，用于计算本次分片的增量
            shard_start_total_n = total_n

            # 内层进度条：处理当前分片内的块
            with tqdm(total=total_rows, desc=f"分片 {i}", leave=False, unit="row") as pbar_chunk:
                for start_idx in range(0, total_rows, CHUNK_SIZE):
                    end_idx = min(start_idx + CHUNK_SIZE, total_rows)
                    # 切片操作会触发该部分的磁盘读取，但只加载到内存中的一小块
                    chunk = current_data_to_process[start_idx:end_idx]

                    # 确保切片是连续的内存块 (可选，有时能加速计算)
                    if not chunk.flags['C_CONTIGUOUS']:
                        chunk = np.ascontiguousarray(chunk)

                    total_n, global_mean, global_M2, global_min, global_max = _process_chunk(
                        chunk, (total_n, global_mean, global_M2, global_min, global_max)
                    )
                    
                    # 更新内层进度条
                    pbar_chunk.update(len(chunk))

            # --- 实时打印当前分片的统计信息 ---
            # 计算当前分片的样本数
            shard_samples = total_n - shard_start_total_n
            
            # 计算当前分片的统计量 (基于当前块的累积)
            # 由于Welford算法的特性，global_* 是全局的，我们无法直接从中间状态得到当前分片的独立统计。
            # 但我们可以在处理完当前分片后，通过全局统计的变化来粗略估算当前分片的贡献。
            # 更准确的做法是在处理当前分片时，单独维护一个临时的Welford统计。
            # 这里为了简化，我们重新加载当前分片并计算其独立的统计。
            
            # 重新加载当前分片数据（仅用于打印）
            temp_memmap = np.memmap(shard_path, dtype='float32', mode='r')
            if temp_memmap.ndim == 1:
                temp_data = temp_memmap.reshape(-1, feature_dim)
            else:
                temp_data = temp_memmap
            temp_min = np.min(temp_data, axis=0)
            temp_max = np.max(temp_data, axis=0)
            temp_mean = np.mean(temp_data, axis=0)
            temp_std = np.std(temp_data, axis=0)
            
            print(f"\n📊 分片 {i}/{len(shard_files)} ('{shard_path.name}') 处理完成:")
            print(f"   - 本地样本数: {shard_samples:,}")
            print(f"   - 累计样本数: {total_n:,}")
            print(f"   - 本地特征范围: [{temp_min.min():.4f}, {temp_max.max():.4f}]")
            print(f"   - 本地特征均值范围: [{temp_mean.min():.6f}, {temp_mean.max():.6f}]")
            print(f"   - 本地特征标准差范围: [{temp_std.min():.6f}, {temp_std.max():.6f}]")
            print("-" * 80)

            # 显式删除 memmap 对象，释放其持有的文件句柄和内存视图
            del current_memmap, current_data_to_process, temp_memmap

            # 更新外层进度条
            pbar_shards.set_postfix_str(f"累计样本: {total_n:,}")
            pbar_shards.update(1)

    # === 4. 计算最终统计量 ===
    global_std = np.sqrt(global_M2 / total_n)

    # 构建结果字典
    stats = {
        "total_tokens": int(total_n),
        "feature_dim": int(inferred_feature_dim),
        "per_dim_stats": [
            {
                "dim": d,
                "min": float(global_min[d]),
                "max": float(global_max[d]),
                "mean": float(global_mean[d]),
                "std": float(global_std[d])
            }
            for d in range(inferred_feature_dim)
        ],
        "global_min": float(np.min(global_min)),
        "global_max": float(np.max(global_max)),
        "global_mean": float(np.mean(global_mean)),
        "global_std": float(np.mean(global_std))
    }

    # === 5. 输出结果 ===
    print("\n✅ 统计完成！关键摘要:")
    print(f"   总 Token 数: {total_n:,}")
    print(f"   特征维度: {inferred_feature_dim}")
    print(f"   全局范围: [{stats['global_min']:.4f}, {stats['global_max']:.4f}]")
    print(f"   平均均值: {stats['global_mean']:.6f} | 平均标准差: {stats['global_std']:.6f}")

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n💾 详细统计已保存至: {output_json}")

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
