# 1. 定义变量
INPUT_DIR="/mnt/nas_syy/default/poregpt/dataset/human_dna_595g/basecall/validation"
OUTPUT_DIR="/mnt/nas_syy/default/poregpt/dataset/human_dna_595g/basecall/validation"

# 2. 调用脚本，将变量展开传递给参数
python3 merge_and_split_bccsv.py -i "$INPUT_DIR" -o "$OUTPUT_DIR"
