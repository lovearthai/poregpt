#!/usr/bin/env python3
import pandas as pd
import os
import sys
import argparse
from pathlib import Path

def merge_and_split_csvs(root_directory, output_directory, overwrite=False):
    """
    递归查找指定目录下所有 .bc.csv 文件，合并它们，
    然后根据 'fast5' 列将合并后的 DataFrame 分割成多个小 CSV 文件。
    """
    root_path = Path(root_directory).resolve()
    
    # 处理输出目录
    if output_directory:
        out_path = Path(output_directory).resolve()
    else:
        out_path = root_path
    
    # 确保输出目录存在
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ 无法创建输出目录 {out_path}: {e}")
        sys.exit(1)

    print(f"🔍 正在递归扫描目录: {root_path}")
    
    # 递归查找所有 .bc.csv 文件
    # 注意：如果输出目录在输入目录内部，我们可能需要避免读取刚刚生成的文件
    # 这里简单处理：先收集所有文件路径
    csv_files = list(root_path.rglob("*.bc.csv"))
    
    # 过滤掉可能已经存在于输出目录中的目标文件（防止死循环或重复读取刚生成的文件）
    # 如果输出目录和输入目录不同，这一步可以跳过，但为了安全起见保留
    filtered_files = []
    for f in csv_files:
        # 简单的启发式检查：如果文件在输出目录且符合命名规则，可能是旧数据，视情况保留或跳过
        # 这里我们主要防止读取刚才脚本生成的文件（如果输出在输入内）
        # 更严谨的做法是记录生成前的文件列表，这里简化处理，假设用户知道自己在做什么
        # 或者：如果输出目录是输入目录的子集，且文件名符合 *.bc.csv，可能会读到新写的。
        # 为了安全，我们只读取在脚本开始*之前*就存在的文件。
        # 由于我们是先列出文件再处理，所以只要不往 root_path 里写同名文件冲突即可。
        # 但如果 output_dir == root_dir，to_csv 会覆盖旧文件，这没问题。
        # 唯一的风险是如果逻辑是追加，但这里是覆盖。
        filtered_files.append(f)
    
    csv_files = filtered_files

    if not csv_files:
        print(f"❌ 在目录 '{root_directory}' 及其子目录中未找到任何 .bc.csv 文件。")
        return

    print(f"📂 找到了 {len(csv_files)} 个 .bc.csv 文件。")

    # 读取所有 CSV 文件并存入列表
    dfs_to_concat = []
    for i, csv_file in enumerate(csv_files):
        # 进度显示
        if (i + 1) % 100 == 0:
            print(f"   ... 已读取 {i+1}/{len(csv_files)} 个文件")
            
        try:
            # 使用 chunksize 或者低内存模式如果文件极大，这里假设内存足够
            df_temp = pd.read_csv(csv_file)
            
            if 'fast5' not in df_temp.columns:
                print(f"  ⚠️  警告: 文件 {csv_file} 缺少 'fast5' 列，将被跳过。")
                continue
            
            # 可选：添加一列记录来源文件，方便调试
            # df_temp['source_file'] = str(csv_file) 
            
            dfs_to_concat.append(df_temp)
        except Exception as e:
            print(f"  ❌ 读取 {csv_file} 时出错: {e}")

    if not dfs_to_concat:
        print("❌ 没有可合并的有效 CSV 文件。")
        return

    # 合并所有 DataFrame
    print("🔄 开始合并所有数据...")
    combined_df = pd.concat(dfs_to_concat, ignore_index=True)
    print(f"📊 合并后总共有 {len(combined_df)} 行数据。")

    # 按 'fast5' 列分组
    print("📁 正在按 'fast5' 列进行分组...")
    grouped = combined_df.groupby('fast5')
    total_groups = len(grouped)
    print(f"📦 分成了 {total_groups} 个组。")

    # 遍历每个组，创建单独的 CSV 文件
    for i, (fast5_filename, group_df) in enumerate(grouped):
        # 构建输出文件名 (去掉 .fast5 后缀，加上 .bc.csv)
        # 处理 fast5_filename 可能包含路径的情况，只取文件名部分
        fname = Path(fast5_filename).name 
        output_filename = Path(fname).stem + '.bc.csv'
        output_path = out_path / output_filename

        # 检查文件是否存在
        if output_path.exists() and not overwrite:
            print(f"  ⚠️  跳过: {output_path} 已存在 (使用 --overwrite 强制覆盖)")
            continue

        print(f"💾 [{i+1}/{total_groups}] 正在写入: {output_path.name} ({len(group_df)} 行)")
        
        try:
            group_df.to_csv(output_path, index=False)
        except Exception as e:
            print(f"  ❌ 写入 {output_path} 失败: {e}")

    print("🎉 合并和分割过程全部完成！")
    print(f"📍 输出目录: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="递归合并目录下的 .bc.csv 文件，并按 'fast5' 列分割成新文件。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法：在当前目录查找，输出到当前目录
  python merge_split_bc.py

  # 指定输入目录和输出目录
  python merge_split_bc.py -i /data/input -o /data/output

  # 强制覆盖已存在的输出文件
  python merge_split_bc.py -i ./data --overwrite
        """
    )

    parser.add_argument(
        '-i', '--input', 
        type=str, 
        default='.', 
        help='要搜索的根目录路径 (默认: 当前目录 ".")'
    )
    
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        default=None, 
        help='输出目录路径 (默认: 与输入目录相同)'
    )

    parser.add_argument(
        '--overwrite', 
        action='store_true', 
        help='如果输出文件已存在，则强制覆盖'
    )

    args = parser.parse_args()

    # 验证输入目录
    if not os.path.isdir(args.input):
        print(f"❌ 错误: 输入路径 '{args.input}' 不是一个有效的目录。")
        sys.exit(1)

    merge_and_split_csvs(args.input, args.output, args.overwrite)

if __name__ == "__main__":
    main()
