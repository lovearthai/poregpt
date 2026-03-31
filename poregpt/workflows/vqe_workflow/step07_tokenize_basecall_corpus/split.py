import pandas as pd
import os
from pathlib import Path

def split_csv_by_fast5(input_csv_path, output_dir=None):
    """
    根据 'fast5' 列将一个大的 CSV 文件分割成多个小 CSV 文件。

    Args:
        input_csv_path (str): 输入的 CSV 文件路径。
        output_dir (str, optional): 输出目录路径。如果为 None，则默认为输入文件所在的目录。
    """
    print(f"📖 正在读取 CSV: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"📊 从 CSV 加载了 {len(df)} 行。")

    if output_dir is None:
        # 如果未指定输出目录，则使用输入文件的目录
        output_dir = Path(input_csv_path).parent

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 按 'fast5' 列分组
    grouped = df.groupby('fast5')
    print(f"📁 找到 {len(grouped)} 个唯一的 FAST5 文件名。")

    for fast5_filename, group_df in grouped:
        # 构建输出文件名 (去掉 .fast5 后缀，加上 .csv)
        output_filename = Path(fast5_filename).stem + '.bc.csv'
        output_path = os.path.join(output_dir, output_filename)

        print(f"💾 正在将 {len(group_df)} 行写入 {output_path}")
        # 将当前组的数据保存为一个新的 CSV 文件
        # index=False 表示不保存行索引
        group_df.to_csv(output_path, index=False)

    print("🎉 分割完成！")


if __name__ == "__main__":
    # 设置输入和输出路径
    input_csv = "validation_cyclone.csv"
    # output_directory = "path/to/output/directory" # 可选：指定输出目录
    output_directory = None # 如果为 None，则输出到 input_csv 所在的目录

    split_csv_by_fast5(input_csv, output_directory)
