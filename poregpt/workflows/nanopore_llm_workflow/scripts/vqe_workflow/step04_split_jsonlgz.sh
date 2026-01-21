# 或指定进程数（例如 8）
nohup python3 -u scripts/step04_split_jsonlgz.py \
  --input_dir fast5_jsonlgz_vqe20_spoch38000/validation \
  --output_dir fast5_jsonlgz_vqe20_spoch38000_split/validation \
  --workers 32 \
  --min_chunk_token_count 1200 \
  &> step04_split_jsonlgz.out &

