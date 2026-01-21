# run_tokenize.py
import argparse
import os
import torch
from nanopore_signal_tokenizer import VQTokenizer

def main():
    parser = argparse.ArgumentParser(description="Tokenize a single Nanopore FAST5 file using VQTokenizer on a specified GPU.")
    parser.add_argument("--fast5_file", type=str, required=True, help="Path to input .fast5 file")
    parser.add_argument("--output_file", type=str, default=None, help="Output .jsonl.gz file path. If not provided, use <input>.jsonl.gz")
    parser.add_argument("--model_ckpt", type=str, default="nanopore_rvq_tokenizer.pth", help="Path to RVQ model checkpoint")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (e.g., 0, 1, 2, 3). Default: 0")
    parser.add_argument("--medf", type=int, default=0, help="Median filter window size (odd int, e.g., 5). 0 to disable.")
    parser.add_argument("--lpf", type=int, default=0, help="Low-pass filter cutoff frequency in Hz (e.g., 1000). 0 to disable.")

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
        device=device
    )

    print(f"✅ Tokenizing: {args.fast5_file}")
    print(f"📤 Output:      {args.output_file}")
    if args.medf > 0:
        print(f"🔧 Median filter: window={args.medf}")
    if args.lpf > 0:
        print(f"🔧 Low-pass filter: cutoff={args.lpf} Hz")

    # 调用 tokenize_fast5 并传入 medf/lpf
    tokenizer.tokenize_fast5(
        fast5_path=args.fast5_file,
        output_path=args.output_file,
        medf=args.medf,
        lpf=args.lpf
    )

    print("✅ Done.")

if __name__ == "__main__":
    main()
