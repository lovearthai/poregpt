#!/usr/bin/env python3

import sys
import csv

input_file = 'out_summary.handle.tsv'
output_file = 'out_summary.processed.tsv'

with open(input_file, 'r', newline='') as infile, \
     open(output_file, 'w', newline='') as outfile:

    reader = csv.reader(infile, delimiter='\t')
    writer = csv.writer(outfile, delimiter='\t')

    for row in reader:
        if len(row) < 2:
            continue # 跳过格式不正确的行

        filename = row[0]
        read_id_full = row[1]
        alignment_identity = row[2] if len(row) > 2 else '' # 处理可能缺少 alignment_identity 的行

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

        # 写入新行
        writer.writerow([filename, read_id, chunk_start, chunk_size, alignment_identity])

print(f"Processing complete. Output saved to {output_file}")
