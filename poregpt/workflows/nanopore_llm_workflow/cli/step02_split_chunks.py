import os
import numpy as np
import glob
import shutil
from tqdm import tqdm

input_dir = "./fast5_chunks_w12k"
output_dir = "./fast5_chunks_w12k_split10k"
os.makedirs(output_dir, exist_ok=True)

# 收集所有 chunks
all_chunks = []
print("Loading all chunks (this will take memory, run on a machine with enough RAM)...")
for f in tqdm(sorted(glob.glob(os.path.join(input_dir, "*.npy")))):
    data = np.load(f, allow_pickle=True)
    for item in data:
        if item['chunk_data'].shape[0] == 12000:
            all_chunks.append(item)

print(f"Total valid chunks: {len(all_chunks)}")

# 拆分保存（每 2000 chunks 一个文件）
chunks_per_file = 100000
for i in range(0, len(all_chunks), chunks_per_file):
    batch = all_chunks[i:i + chunks_per_file]
    out_path = os.path.join(output_dir, f"chunk_batch_{i//chunks_per_file:06d}.npy")
    np.save(out_path, batch)

print(f"Saved to {output_dir} ({len(all_chunks)//chunks_per_file + 1} files)")
