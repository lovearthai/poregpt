#!/usr/bin/env python3

import os
import sys
import csv
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
import traceback
import argparse # 导入 argparse 模块


def process_single_pair(args):
    """
    Processes a single pair of TSV and NPY files based on arguments passed by the pool.

    Args:
        args (tuple): A tuple containing (tsv_file_path, npy_file_path, output_csv_path).
    """
    tsv_file, npy_file, output_csv = args
    print(f"Starting process for {tsv_file} and {npy_file}", file=sys.stderr)

    try:
        # Load the NPY array
        print(f"  Loading NPY file: {npy_file}")
        reference_arrays = np.load(npy_file)
        print(f"  Loaded array shape: {reference_arrays.shape}, dtype: {reference_arrays.dtype}")

        # Check if the number of rows match
        with open(tsv_file, 'r', newline='', encoding='utf-8') as f:
            num_rows = sum(1 for line in f) - 1  # Subtract 1 for header

        if reference_arrays.shape[0] != num_rows:
            error_msg = (
                f"  Error in {tsv_file} and {npy_file}: Row count mismatch. "
                f"TSV has {num_rows} data rows, NPY has {reference_arrays.shape[0]} rows."
            )
            print(error_msg, file=sys.stderr)
            return False, error_msg

        print(f"  Processing {num_rows} rows...")

        with open(tsv_file, 'r', newline='', encoding='utf-8') as tsv_f, \
             open(output_csv, 'w', newline='', encoding='utf-8') as csv_f:

            # Create TSV reader
            tsv_reader = csv.reader(tsv_f, delimiter='\t')

            # Create CSV writer
            # 增加了 'bases_raw' 字段
            fieldnames = ['fast5', 'read_id', 'chunk_start', 'chunk_size', 'alignment_identity', 'bases', 'bases_raw']
            csv_writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
            csv_writer.writeheader()

            # Read and skip header from TSV
            header = next(tsv_reader)
            if header[:3] != ['filename', 'read_id', 'chunk_start']:
                 print(f"  Warning for {tsv_file}: Unexpected TSV header format: {header[:3]}. Proceeding anyway.", file=sys.stderr)

            # Iterate through TSV rows and corresponding NPY rows
            for i, row in enumerate(tsv_reader):
                if len(row) < 5:
                    print(f"  Warning for {tsv_file}: Skipping malformed TSV row {i+2}: {row}", file=sys.stderr)
                    continue

                # Extract TSV fields
                filename = row[0]
                read_id = row[1]
                chunk_start = row[2]
                chunk_size = row[3]
                alignment_identity = row[4]

                # Get corresponding NPY array
                npy_row = reference_arrays[i]

                # Process the NPY array: remove zeros and convert to string
                # --- 新增: 获取原始数组的字符串 ---
                raw_values = npy_row
                bases_raw_string = ''.join(map(str, raw_values))

                # Filter out zeros
                non_zero_values = npy_row[npy_row != 0]
                # Convert to string representation (e.g., [1, 2, 3] -> "123")
                bases_string = ''.join(map(str, non_zero_values))

                # Create dictionary for writing
                output_row = {
                    'fast5': filename,
                    'read_id': read_id,
                    'chunk_start': chunk_start,
                    'chunk_size': chunk_size,
                    'alignment_identity': alignment_identity,
                    'bases': bases_string,        # 不含零的序列
                    'bases_raw': bases_raw_string # 包含零的原始序列
                }

                # Write the row to CSV
                csv_writer.writerow(output_row)

                # Optional: Print progress every 10000 rows
                if (i + 1) % 10000 == 0:
                    print(f"    Processed {i+1}/{num_rows} rows...", file=sys.stderr)

        success_msg = f"Successfully processed {tsv_file} -> {output_csv}"
        print(f"  {success_msg}", file=sys.stderr)
        return True, success_msg

    except FileNotFoundError as e:
        error_msg = f"  Error in {tsv_file} and {npy_file}: File not found - {e}"
        print(error_msg, file=sys.stderr)
        return False, error_msg
    except ValueError as e:
        error_msg = f"  Error in {tsv_file} and {npy_file}: {e}"
        print(error_msg, file=sys.stderr)
        return False, error_msg
    except Exception as e:
        error_msg = f"  An unexpected error occurred in {tsv_file} and {npy_file}: {e}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        return False, error_msg


def find_pairs_and_run_parallel(root_directory, num_processes=None):
    """
    Recursively finds pairs of 'out_summary.processed.tsv' and 'references.npy' in subdirectories
    and runs the processing function on them in parallel using multiprocessing.

    Args:
        root_directory (str): The root directory to start the search.
        num_processes (int, optional): Number of processes to use. Defaults to number of CPUs.
    """
    if num_processes is None:
        num_processes = cpu_count()

    print(f"🔍 Scanning directory '{root_directory}' for file pairs using {num_processes} processes...")

    # Find all matching pairs
    work_items = []
    for subdir, dirs, files in os.walk(root_directory):
        tsv_file = None
        npy_file = None

        for file in files:
            if file == 'out_summary.processed.tsv':
                tsv_file = os.path.join(subdir, file)
            elif file == 'references.npy':
                npy_file = os.path.join(subdir, file)

        if tsv_file and npy_file:
            # Construct output path: {subdir}/{subdir_name}.bc.csv
            subdir_name = Path(subdir).name
            output_csv = os.path.join(subdir, f"{subdir_name}.bc.csv")

            work_items.append((tsv_file, npy_file, output_csv))
            print(f"  Found pair: {tsv_file} & {npy_file} -> {output_csv}")

    if not work_items:
        print("❌ No matching file pairs found.")
        return

    print(f"🚀 Found {len(work_items)} pairs. Starting parallel processing...")

    # Use multiprocessing Pool to process items in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_pair, work_items)

    # Report results
    successful = 0
    failed = 0
    for success, message in results:
        if success:
            successful += 1
        else:
            failed += 1
            print(message, file=sys.stderr) # Errors were already printed by worker, but we can summarize

    print(f"\n🎉 Processing complete. Successful: {successful}, Failed: {failed}")


if __name__ == "__main__":
    # 使用 argparse 定义和解析命令行参数
    parser = argparse.ArgumentParser(
        description="Recursively merge TSV and NPY files into .bc.csv files in parallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # 自动为帮助信息添加默认值
    )
    parser.add_argument(
        'root_directory',
        type=str,
        help='The root directory to scan for file pairs.'
    )
    parser.add_argument(
        '--num_processes', '-n',
        type=int,
        default=cpu_count(), # 默认值为CPU核心数
        help='Number of parallel processes to use.'
    )

    args = parser.parse_args()

    # 验证 root_directory 是否为一个有效的目录
    if not os.path.isdir(args.root_directory):
        print(f"Error: The provided path '{args.root_directory}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # 调用主函数
    find_pairs_and_run_parallel(args.root_directory, args.num_processes)
