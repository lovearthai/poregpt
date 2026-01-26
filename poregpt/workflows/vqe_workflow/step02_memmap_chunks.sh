nohup python3 -u step02_memmap_chunks.py \
    --input_dir fast5_chunks_w12k_split10k \
    --output_dir fast5_chunks_w12k_split10k_memmap \
    --chunk_size 12000 \
    --dtype float32 \
    --num_workers 32 \
    &> step02_memmap_chunks.out &
