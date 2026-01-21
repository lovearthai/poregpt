import os
import re
import gzip
import json
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

def extract_tokens(text):
    """Extract all tokens of the form <|bwav:...|>"""
    return re.findall(r"<\|bwav:[^|>]+\|>", text)

def split_with_overlap(tokens, window=8192, overlap=1024):
    """滑动窗口分割，带重叠"""
    if len(tokens) <= window:
        return [tokens]
    step = window - overlap
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + window
        chunks.append(tokens[start:end])
        if end >= len(tokens):
            break
        start += step
    return chunks

def process_single_file(args):
    """
    处理单个 .jsonl.gz 文件
    args: (input_path_str, output_dir_str, min_chunk_token_count)
    """
    input_path_str, output_dir_str, min_chunk_token_count = args
    input_path = Path(input_path_str)
    output_dir = Path(output_dir_str)
    output_path = output_dir / (input_path.name.replace('.jsonl.gz', '.split.jsonl.gz'))

    total_input_tokens = 0          # 原始唯一 token 总数（每条 read 提取一次）
    total_kept_chunks = 0           # 保留的 chunk 数量
    total_discarded_chunks = 0      # 丢弃的 chunk 数量
    total_tokens_in_kept_chunks = 0     # 所有保留 chunk 中的 token 出现总次数（含重复）
    total_tokens_in_discarded_chunks = 0  # 所有丢弃 chunk 中的 token 出现总次数（含重复）

    try:
        with gzip.open(input_path, 'rt', encoding='utf-8') as fin, \
             gzip.open(output_path, 'wt', encoding='utf-8') as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    read_id = item["id"]
                    text = item["text"]
                    tokens = extract_tokens(text)
                    if not tokens:
                        continue

                    num_tokens = len(tokens)
                    total_input_tokens += num_tokens

                    chunks = split_with_overlap(tokens, window=8192, overlap=1024)

                    kept_chunks = []
                    for chunk in chunks:
                        chunk_len = len(chunk)
                        if chunk_len >= min_chunk_token_count:
                            kept_chunks.append(chunk)
                            total_kept_chunks += 1
                            total_tokens_in_kept_chunks += chunk_len
                        else:
                            total_discarded_chunks += 1
                            total_tokens_in_discarded_chunks += chunk_len

                    # 写入保留的 chunks
                    for idx, chunk in enumerate(kept_chunks):
                        new_id = f"{read_id}_{idx:05d}"
                        new_text = "".join(chunk)
                        meta = {
                            "source_file": input_path.name,
                            "original_read_id": read_id
                        }
                        out_item = {
                            "id": new_id,
                            "text": new_text,
                            "meta": meta
                        }
                        fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")

                except Exception as e:
                    print(f"⚠️ Error processing line in {input_path.name}: {e}")

        # 注意：由于 overlap，以下等式通常不成立：
        # total_tokens_in_kept_chunks + total_tokens_in_discarded_chunks >= total_input_tokens

        summary = (
            f"✅ {input_path.name} | "
            f"input_tokens={total_input_tokens} | "
            f"kept_chunks={total_kept_chunks} | "
            f"discarded_chunks={total_discarded_chunks} | "
            f"tokens_in_kept_chunks={total_tokens_in_kept_chunks} | "
            f"tokens_in_discarded_chunks={total_tokens_in_discarded_chunks}"
        )
        print(summary)
        return summary

    except Exception as e:
        error_msg = f"💥 Failed {input_path.name}: {e}"
        print(error_msg)
        return error_msg

def main():
    parser = argparse.ArgumentParser(description="Parallel split JSONL.GZ by tokens with overlap and min chunk filtering.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with .jsonl.gz files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU count)")
    parser.add_argument("--min_chunk_token_count", type=int, default=1200, help="Minimum number of tokens a chunk must have to be kept (default: 1)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    jsonl_files = list(input_dir.glob("*.jsonl.gz"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl.gz files found in {input_dir}")

    task_args = [(str(f), str(output_dir), args.min_chunk_token_count) for f in sorted(jsonl_files)]

    workers = args.workers or min(cpu_count(), len(jsonl_files))
    print(f"🚀 Starting parallel processing with {workers} workers...")
    print(f"   ⚙️  Min chunk token count: {args.min_chunk_token_count}")

    with Pool(processes=workers) as pool:
        results = pool.map(process_single_file, task_args)

    for res in results:
        print(res)

    print("🎉 All files processed.")

if __name__ == "__main__":
    main()
