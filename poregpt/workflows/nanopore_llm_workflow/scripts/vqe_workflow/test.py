import os
import gzip
import json
from pathlib import Path

def process_jsonlgz_files(root_dir):
    """
    遍历指定目录下所有递归子目录中的.jsonl.gz文件，
    检查每个文件中是否有小于100个字符的行，如果有则删除该文件
    """
    root_path = Path(root_dir)
    
    # 找到所有.jsonl.gz文件
    jsonlgz_files = list(root_path.rglob("*.jsonl.gz"))
    
    deleted_count = 0
    
    for file_path in jsonlgz_files:
        try:
            should_delete = False
            
            # 读取并检查文件
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if len(line) < 100:
                        print(f"发现小于100字符的行 (第{line_num}行): {line[:50]}...")
                        should_delete = True
                        break
            
            if should_delete:
                print(f"删除文件: {file_path}")
                file_path.unlink()  # 删除文件
                deleted_count += 1
            else:
                print(f"保留文件: {file_path}")
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    print(f"\n总共删除了 {deleted_count} 个文件")

if __name__ == "__main__":
    # 替换为你的目录路径
    directory_path = "fast5_jsonlgz"
    process_jsonlgz_files(directory_path)
