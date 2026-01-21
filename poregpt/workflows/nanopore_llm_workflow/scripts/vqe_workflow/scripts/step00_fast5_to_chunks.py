#!/usr/bin/env python3
"""
Preprocess Nanopore FAST5 files using the nanopore_llm_workflow package.

Usage:
    python run_preprocess.py --fast5_dir ./fast5 --output_dir ./chunks
"""

import argparse
from nanopore_llm_workflow import process_fast5_to_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Nanopore FAST5 signals into normalized chunks for LLM training."
    )
    parser.add_argument(
        "--fast5_dir", type=str, required=True,
        help="Path to directory containing .fast5 files."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save output .npy chunk files."
    )
    parser.add_argument(
        "--window_size", type=int, default=12000,
        help="Length of each signal chunk in samples (default: 12000)."
    )
    parser.add_argument(
        "--stride", type=int, default=11400,
        help="Stride between chunks in samples (default: 11400)."
    )
    parser.add_argument(
        "--do_normalize", action="store_true",
        help="Enable median-MAD normalization (default: enabled)."
    )
    parser.add_argument(
        "--no_normalize", action="store_false", dest="do_normalize",
        help="Disable normalization."
    )
    parser.set_defaults(do_normalize=True)

    parser.add_argument(
        "--do_medianfilter", action="store_true",
        help="Apply median filter (default: disabled)."
    )
    parser.add_argument(
        "--do_lowpassfilter", action="store_true",
        help="Apply low-pass filter (default: disabled)."
    )
    parser.add_argument(
        "--default_fs", type=int, default=5000,
        help="Default sampling frequency (Hz) if not in metadata (default: 5000)."
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1,
        help="Number of parallel jobs (default: 1; use -1 for all CPUs)."
    )

    args = parser.parse_args()

    # Call the workflow function
    process_fast5_to_chunks(
        fast5_dir=args.fast5_dir,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        do_normalize=args.do_normalize,
        do_medianfilter=args.do_medianfilter,
        do_lowpassfilter=args.do_lowpassfilter,
        default_fs=args.default_fs,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
