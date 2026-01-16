# run_tokenize.py
import argparse
import os
import torch
from nanopore_signal_tokenizer import VQTokenizer

def main():
    parser = argparse.ArgumentParser(description="Tokenize a single Nanopore FAST5 file using RVQTokenizer on a specified GPU.")
    parser.add_argument("--fast5_file", type=str, required=True, help="Path to input .fast5 file")
    parser.add_argument("--output_file", type=str, default=None, help="Output .jsonl.gz file path. If not provided, use <input>.jsonl.gz")
    parser.add_argument("--model_ckpt", type=str, default="nanopore_rvq_tokenizer.pth", help="Path to RVQ model checkpoint")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (e.g., 0, 1, 2, 3). Default: 0")
    args = parser.parse_args()

    # 自动推导 output_file
    if args.output_file is None:
        args.output_file = args.fast5_file + ".jsonl.gz"

    # 输入检查
    if not os.path.isfile(args.fast5_file):
        raise FileNotFoundError(f"Input FAST5 file not found: {args.fast5_file}")

    # 检查 GPU 是否可用
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a machine with GPU support.")
    if args.gpu_id >= torch.cuda.device_count():
        raise ValueError(f"Invalid gpu_id={args.gpu_id}. Only {torch.cuda.device_count()} GPU(s) available.")

    device = f"cuda:{args.gpu_id}"
    print(f"Using device: {device} ({torch.cuda.get_device_name(args.gpu_id)})")

    # 初始化 tokenizer 并指定设备
    tokenizer = VQTokenizer(
        model_ckpt=args.model_ckpt,
        device=device  # 👈 关键：指定具体 GPU
    )

    print(f"✅ Tokenizing: {args.fast5_file}")
    print(f"📤 Output:      {args.output_file}")

    # 调用单文件处理方法（假设你已定义 tokenize_fast5_file）
    tokenizer.tokenize_fast5(args.fast5_file, args.output_file)

    print("✅ Done.")

if __name__ == "__main__":
    main()
