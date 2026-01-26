import os
from pathlib import Path

def delete_small_jsonlgz_files(root_dir, min_size_bytes=1000):
    """
    删除指定目录及所有子目录下小于指定字节数的.jsonl.gz文件
    """
    root_path = Path(root_dir)
    
    # 找到所有.jsonl.gz文件
    jsonlgz_files = list(root_path.rglob("*.jsonl.gz"))
    
    deleted_count = 0
    
    for file_path in jsonlgz_files:
        try:
            # 获取文件大小
            file_size = file_path.stat().st_size
            
            if file_size < min_size_bytes:
                print(f"删除小文件 ({file_size} 字节): {file_path}")
                file_path.unlink()  # 删除文件
                deleted_count += 1
            else:
                print(f"保留文件 ({file_size} 字节): {file_path}")
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    print(f"\n总共删除了 {deleted_count} 个文件")

if __name__ == "__main__":
    # 替换为你的目录路径
    directory_path = "fast5_jsonlgz"
    delete_small_jsonlgz_files(directory_path, min_size_bytes=1000)
