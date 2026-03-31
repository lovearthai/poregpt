import pickle
import os
import glob
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

def parse_args():
    parser = argparse.ArgumentParser(description="PoreGPT Codebook Analysis Toolkit")
    
    # 路径配置
    parser.add_argument("--source_dir", type=str, required=True,
                        help="包含 .pkl 索引文件的目录路径")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="包含使用率报告的 CSV 路径 (用于热图标注)")
    parser.add_argument("--output_subdir", type=str, default="visualizations_heatmaps",
                        help="热图保存的子目录名称")
    
    # 参数配置
    parser.add_argument("--grid_size", type=int, default=256, 
                        help="Codebook 网格大小 (默认 256x256)")
    parser.add_argument("--use_log_scale", type=bool, default=True,
                        help="热图是否使用 Log 尺度")
    
    return parser.parse_args()

def get_step(filename):
    """从文件名中提取 step 数字"""
    match = re.search(r'step(\d+)', filename)
    return int(match.group(1)) if match else 0

def run_kl_analysis(args, pkl_files):
    """功能 1: 计算并绘制 KL 散度趋势"""
    print("\n--- 正在进行 KL 散度趋势分析 ---")
    total_codes = args.grid_size * args.grid_size
    steps, kl_divergences = [], []
    prev_prob = None

    for file_path in tqdm(pkl_files, desc="Calculating KL"):
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                indices = data["indices"].flatten()
            
            counts = np.bincount(indices, minlength=total_codes)
            prob = (counts + 1e-10) / (np.sum(counts) + 1e-10 * total_codes)

            if prev_prob is not None:
                kl_divergences.append(entropy(prob, prev_prob))
                steps.append(get_step(os.path.basename(file_path)))
            prev_prob = prob
        except Exception as e:
            print(f"⚠️ 跳过文件 {os.path.basename(file_path)}: {e}")

    if not kl_divergences:
        print("❌ 数据不足，无法绘制 KL 趋势图")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(steps, kl_divergences, marker='o', ls='-', color='b', markersize=4, alpha=0.7)
    plt.title("Codebook Distribution Evolution (KL Divergence)", fontsize=14)
    plt.xlabel("Training Steps")
    plt.ylabel(r"$D_{KL}(P_{i+1} || P_i)$")
    plt.grid(True, alpha=0.3)
    
    save_path = "codebook_kl_divergence_trend.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✨ KL 趋势图已保存至: {save_path}")

def run_heatmap_generation(args, pkl_files):
    """功能 2: 批量生成 Codebook 使用分布热图"""
    print(f"\n--- 正在生成热图 (存至: {args.output_subdir}) ---")
    
    # 1. 准备目录
    output_dir = os.path.join(os.getcwd(), args.output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # 2. 尝试读取 CSV 报表
    ratio_map = {}
    if args.csv_path and os.path.exists(args.csv_path):
        try:
            df = pd.read_csv(args.csv_path)
            ratio_map = dict(zip(df['checkpoint'].str.strip(), df['usage_ratio']))
            print("📊 CSV 使用率报表加载成功")
        except Exception as e:
            print(f"⚠️ 加载 CSV 失败: {e}")

    # 3. 定义颜色映射: 白 -> 红 -> 黑
    my_cmap = LinearSegmentedColormap.from_list('white_red_black', ["#FFFFFF", "#FF0000", "#000000"], N=256)

    # 4. 循环绘图
    for file_path in tqdm(pkl_files, desc="Generating Heatmaps"):
        file_name = os.path.basename(file_path)
        ckpt_key = file_name.replace(".pkl", "")
        current_ratio = ratio_map.get(ckpt_key, None)
        
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                indices = data["indices"].flatten()

            counts = np.bincount(indices, minlength=args.grid_size * args.grid_size)
            heatmap_data = counts.reshape((args.grid_size, args.grid_size))

            plt.figure(figsize=(11, 9), facecolor='white')
            display_data = np.log1p(heatmap_data) if args.use_log_scale else heatmap_data
            
            ax = sns.heatmap(display_data, cmap=my_cmap, vmin=0, 
                             cbar_kws={'label': 'Log(Frequency + 1)' if args.use_log_scale else 'Frequency'},
                             xticklabels=False, yticklabels=False)
            
            # 添加边框
            for _, spine in ax.spines.items():
                spine.set_visible(True); spine.set_color('#DDDDDD'); spine.set_linewidth(1)

            ratio_text = f"{current_ratio:.2%}" if current_ratio is not None else "N/A"
            plt.title(f"Checkpoint: {file_name}\nUsage Ratio: {ratio_text}", fontsize=14, fontweight='bold', pad=15)
            
            save_name = file_name.replace(".pkl", ".png")
            plt.savefig(os.path.join(output_dir, save_name), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"⚠️ 处理 {file_name} 出错: {e}")

def main():
    args = parse_args()
    
    # 获取并统一排序文件列表
    pkl_files = sorted(glob.glob(os.path.join(args.source_dir, "*.pkl")), 
                        key=lambda x: get_step(os.path.basename(x)))
    
    if not pkl_files:
        print(f"❌ 目录 {args.source_dir} 中未找到 .pkl 文件")
        return

    # 执行任务
    run_heatmap_generation(args, pkl_files)
    run_kl_analysis(args, pkl_files)
    
    print("\n✅ 所有任务处理完毕！")

if __name__ == "__main__":
    main()