#!/usr/bin/env python3

import os
import sys
import csv
from pathlib import Path

def process_tsv_file(input_path, output_path):
    """Processes a single TSV file."""
    try:
        with open(input_path, 'r', newline='') as infile, \
             open(output_path, 'w', newline='') as outfile:

            reader = csv.reader(infile, delimiter='\t')
            writer = csv.writer(outfile, delimiter='\t')

            # --- 处理表头 ---
            header = next(reader)
            
            # 找到 'read_id' 列的索引
            try:
                read_id_col_index = header.index('read_id')
            except ValueError:
                print(f"  Error: Column 'read_id' not found in header of {input_path}. Header is: {header}", file=sys.stderr)
                return # 跳过此文件
            
            # 构造新表头
            new_header = []
            for i, col_name in enumerate(header):
                if i == read_id_col_index:
                    # 将原来的 'read_id' 列替换为三个新列
                    new_header.extend(['read_id', 'chunk_start', 'chunk_size'])
                else:
                    # 保留其他列名
                    new_header.append(col_name)
            
            # 写入新表头
            writer.writerow(new_header)

            # --- 处理数据行 ---
            for row_num, row in enumerate(reader, start=2): # 从第2行开始计数
                if len(row) < len(header):
                    print(f"  Warning: Skipping malformed row {row_num} in {input_path}: {row}", file=sys.stderr)
                    continue # 跳过格式不正确的行

                new_row = []
                for i, cell_value in enumerate(row):
                    if i == read_id_col_index:
                        # 对 'read_id' 列的值进行分割
                        read_id_full = cell_value
                        
                        # 按冒号分割 read_id
                        parts = read_id_full.split(':', 2) # 最多分割成3部分
                        if len(parts) >= 3:
                            read_id = parts[0]
                            chunk_start = parts[1]
                            chunk_size = parts[2]
                        elif len(parts) == 2: # 如果只有2部分，比如 "id:start"
                             read_id = parts[0]
                             chunk_start = parts[1]
                             chunk_size = ''
                        else: # 如果只有1部分或0部分
                             read_id = read_id_full
                             chunk_start = ''
                             chunk_size = ''
                        
                        # 将分割后的三个部分加入新行
                        new_row.extend([read_id, chunk_start, chunk_size])
                    else:
                        # 其他列的值直接加入新行
                        new_row.append(cell_value)
                
                # 写入新行
                writer.writerow(new_row)

        # 修复了这里的错误，将 'output_csv' 改为 'output_path'
        print(f"  Successfully processed: {input_path} -> {output_path}")
    except Exception as e:
        print(f"  Error processing {input_path}: {e}", file=sys.stderr)

def main():
    # Define the root directory to start the search
    # You can change this to any specific directory or pass it as an argument
    root_directory = '.' # Current directory by default
    if len(sys.argv) > 1:
        root_directory = sys.argv[1]

    root_path = Path(root_directory)
    if not root_path.is_dir():
        print(f"Error: The provided path '{root_directory}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning directory: {root_directory}")
    # Walk through all subdirectories
    for subdir, dirs, files in os.walk(root_directory):
        # Check each file in the current subdirectory
        for file in files:
            if file == 'out_summary.handle.tsv':
                input_file_path = Path(subdir) / file
                output_file_path = Path(subdir) / 'out_summary.processed.tsv'

                print(f"Found file to process: {input_file_path}")
                process_tsv_file(input_file_path, output_file_path)

    print("\nAll files have been processed.")

if __name__ == "__main__":
    main()
