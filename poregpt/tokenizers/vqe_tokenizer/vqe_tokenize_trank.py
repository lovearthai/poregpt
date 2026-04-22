# -*- coding: utf-8 -*-
"""
Sequentially tokenize .npy files generated from fast5_to_chank.py using VQETokenizer.
Each chunk in a .npy file becomes one line in the corresponding .jsonl.gz file.
This version processes files sequentially, one after another.
"""

import os
import gzip
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from .vqe_tokenizer import VQETokenizer
import torch


def process_npy_file(npy_file_path, output_path, tokenizer, max_batch_size, layer=0,strategy="dolma"):
    """
    Process a single .npy file: load chunks, tokenize them in batches,
    and write results to a .jsonl.gz file.

    Args:
        npy_file_path (str or Path): Path to the input .npy file.
        tokenizer (VQETokenizer): An instance of the VQETokenizer class.
        output_path (str or Path): Path to save the output .jsonl.gz file.
        max_batch_size (int): Batch size for tokenization during inference.
        layer (int): The layer index used for merged tokenization.
    """
    npy_file_path = Path(npy_file_path)

    try:
        # Load the .npy file
        # Assuming the structure from your previous script: list of 1D arrays
        chunks_list = np.load(npy_file_path, allow_pickle=True)
        if not isinstance(chunks_list, list) and not isinstance(chunks_list, np.ndarray):
             print(f"Warning: {npy_file_path} does not contain a list or array. Skipping.")
             return

        if isinstance(chunks_list, np.ndarray):
            # If it's a 2D array where rows are chunks, convert to list
            if chunks_list.ndim == 2:
                chunks_list = [chunks_list[i] for i in range(chunks_list.shape[0])]
            # If it's a 1D array, wrap it in a list
            elif chunks_list.ndim == 1:
                 print(f"Warning: {npy_file_path} seems to contain a single 1D array. Wrapping as list.")
                 chunks_list = [chunks_list]
            else:
                 print(f"Warning: {npy_file_path} has unexpected shape {chunks_list.shape}. Skipping.")
                 return

        if len(chunks_list) == 0:
            print(f"Info: {npy_file_path} contains no chunks. Writing empty .jsonl.gz file.")
            with gzip.open(output_path, 'wt', encoding='utf-8') as f_out:
                pass # Create an empty file
            return

        # Prepare results list
        results = []
        num_chunks = len(chunks_list)

        # 这里需要获取实际的模型实例（处理 DDP 包装的情况）
        raw_model = tokenizer.model.module if hasattr(tokenizer.model, 'module') else tokenizer.model

        # Process chunks in batches
        for i in tqdm(range(0, num_chunks, max_batch_size), desc=f"Tokenizing {npy_file_path.name}", leave=False):
            batch_chunks = chunks_list[i:i+max_batch_size]

            # Ensure all chunks in the batch have the same length
            # Assuming each chunk should be 1200 or 40000 samples long
            batch_signal_np = np.array(batch_chunks, dtype=np.float32) # Shape: (B, L_chunk)

            # Prepare input tensor for the model
            # x 形状: [Batch_Size, 1, L_signal]
            # 例如: [32, 1, 1200] 表示 32 个样本，单通道信号点 1200 个
            x = torch.from_numpy(batch_signal_np).float().unsqueeze(1).to(tokenizer.device) 
            
            # 2. 模型推理：送入模型，获取重构信号、层级索引等
            with torch.no_grad():
                # level_indices 形状: [Batch_Size, Seq_Len, Num_Layers]
                # 维度含义: [样本序号, 下采样后的时间步, 每一层量化器的索引ID]
                # 例如: [32, 300, 2] 表示 32 个样本，每个序列长 300，由 Layer0 和 Layer1 两层 ID 构成
                recon, level_indices, _ = tokenizer.model(x)

            # 3. Tokenization: 使用模型的 tokenize_indices 方法将层级索引转换为单一的token ID序列
            # tokens_tensor 形状: [Batch_Size, Seq_Len]
            # 含义: 每一行是该样本对应的合并后的 Token ID 序列
            tokens_tensor = raw_model.tokenize_indices(level_indices, layer=layer)

            # 4. 结果处理：将张量移至CPU并转换为 numpy 数组
            tokens_np = tokens_tensor.cpu().numpy().astype(np.int64)
            level_indices_np = level_indices.cpu().numpy().astype(np.int64)
            
            # 信号处理：保留小数点后三位有效数字
            # x_np 和 recon_np 初始形状均为 [Batch_Size, L_signal]
            x_raw = x.squeeze(1).cpu().numpy().astype(np.float32)
            recon_raw = recon.squeeze(1).cpu().numpy().astype(np.float32)

            # --- 核心修改：利用字符串格式化彻底去除浮点冗余尾数 ---
            # 对 Batch 中的每一行信号进行高精度截断处理
            x_np_rounded = [
                [round(float(f"{val:.3f}"), 3) for val in row] 
                for row in x_raw
            ]
            recon_np_rounded = [
                [round(float(f"{val:.3f}"), 3) for val in row] 
                for row in recon_raw
            ]

            # ... 前面模型推理与 x_np_rounded 处理逻辑保持不变 ...
            # Iterate through the batch results
            for j in range(tokens_np.shape[0]):
                # --- 字段: tokens (List[int]) ---
                chunk_tokens_list = tokens_np[j].tolist()

                # --- 核心修改：按层拆分 tokens_layered ---
                # level_indices_np[j] 的形状是 [Seq_Len, Num_Layers]
                # 我们通过索引获取每一层的所有时间步 ID
                current_sample_layered = level_indices_np[j] # [Seq_Len, Num_Layers]
                num_layers = current_sample_layered.shape[1]
                
                # 动态创建层级字段字典
                layered_fields = {}
                for l_idx in range(num_layers):
                    # 提取第 l_idx 层的所有 token ID
                    # current_sample_layered[:, l_idx] 获取该层整列数据
                    layered_fields[f"tokens_layer{l_idx}"] = current_sample_layered[:, l_idx].tolist()

                # Format tokens into the string format for 'text' field
                token_strings = [f"<|bwav:{int(token_id)}|>" for token_id in chunk_tokens_list]
                joined_string = "".join(token_strings)

                chunk_id = f"{npy_file_path.stem}_chunk_{i+j}"
                
                if strategy == "maximum":
                    # 构造最终存储字典
                    output_item = {
                        "id": chunk_id,
                        "text": joined_string,
                        "tokens": chunk_tokens_list,        # 混合后的 Token ID
                        "x": x_np_rounded[j],               # 原始信号 (3位小数)
                        "recon": recon_np_rounded[j]        # 重建信号 (3位小数)
                    }
                    # 将动态生成的 tokens_layer0, tokens_layer1 等字段合并进去
                    output_item.update(layered_fields)
                else:
                    # 构造最终存储字典
                    output_item = {
                        "id": chunk_id,
                        "text": joined_string,
                    }

                results.append(output_item)
    except Exception as e:
        print(f"❌ Error processing {npy_file_path}: {e}")
        return

    # Write all results to the .jsonl.gz file
    try:
        with gzip.open(output_path, 'wt', encoding='utf-8') as f_out:
            for item in tqdm(results, desc="Writing to file", leave=False):
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✅ Wrote {len(results)} lines to {output_path}")
    except Exception as e:
        print(f"❌ Error writing to {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Tokenize a single .npy file using VQETokenizer.')
    parser.add_argument('-i', '--input-file', type=str, required=True, help='Input .npy file to tokenize.')
    parser.add_argument('-o', '--output-file', type=str, required=True, help='Output .jsonl.gz file to save tokens.')
    parser.add_argument('--model-ckpt', type=str, required=True, help='Path to the VQ tokenizer model checkpoint (.pth file).')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu).')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for tokenization.')
    parser.add_argument('--layer', type=int, default=0, help='Layer index for tokenize_indices.')
    parser.add_argument('--strategy', type=str, default="dolma", help='format')

    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    if not input_file.exists() or not input_file.is_file():
        print(f"Error: Input file does not exist: {input_file}")
        return

    output_parent_dir = output_file.parent
    if not output_parent_dir.exists():
        output_parent_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing file: {input_file.name}")
    print(f"Model checkpoint: {args.model_ckpt}")
    print(f"Device: {args.device} | Layer: {args.layer}")
    print("-" * 60)

    # Initialize the tokenizer once
    tokenizer = VQETokenizer(model_ckpt=args.model_ckpt, device=args.device)

    try:
        process_npy_file(input_file, output_file, tokenizer, args.batch_size, layer=args.layer)
        print(f"\n✅ File processed successfully: {output_file}")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
