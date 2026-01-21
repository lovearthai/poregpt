#!/usr/bin/env python3
import os
import torch
import torch.nn.functional as F
import argparse

# 可选：可视化支持
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    HAS_VIS = True
except ImportError:
    HAS_VIS = False

def load_codebook_from_checkpoint(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"📂 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    embed_keys = [k for k in ckpt.keys() if "_codebook.embed" in k]
    if not embed_keys:
        raise RuntimeError("No '_codebook.embed' found in checkpoint.")
    
    embed = ckpt[embed_keys[0]]
    if embed.ndim == 3:
        embed = embed.squeeze(0)
    elif embed.ndim != 2:
        raise ValueError(f"Unexpected codebook shape: {embed.shape}")
    
    print(f"✅ Codebook loaded: {embed.shape} (num_codes, dim)")
    return embed

def evaluate_orthogonality(codebook, plot=False, plot_path_sim=None, plot_path_hist=None):
    device = codebook.device
    num_codes, dim = codebook.shape
    
    # 归一化 + 相似度矩阵
    codebook_norm = F.normalize(codebook, p=2, dim=1)
    sim_matrix = torch.matmul(codebook_norm, codebook_norm.T)
    
    # 去除对角线（只分析不同码字对）
    mask = ~torch.eye(num_codes, dtype=torch.bool, device=device)
    off_diag = sim_matrix[mask].cpu().numpy()  # 转为 numpy 便于绘图
    
    # 基础统计
    mean_abs = abs(off_diag).mean()
    max_sim = off_diag.max()
    min_sim = off_diag.min()
    std_sim = off_diag.std()
    
    # 打印报告
    print("\n" + "="*70)
    print("📊 VQ Codebook Orthogonality Report")
    print("="*70)
    print(f"Codebook size      : {num_codes}")
    print(f"Embedding dim      : {dim}")
    print(f"Mean |similarity|  : {mean_abs:.6f}")
    print(f"Max similarity     : {max_sim:.6f}")
    print(f"Min similarity     : {min_sim:.6f}")
    print(f"Std of similarity  : {std_sim:.6f}")
    print("-"*70)
    
    # 多阈值分布
    print("Distribution of |cosine similarity| (off-diagonal pairs):")
    for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        ratio = (abs(off_diag) < th).mean()
        print(f"% pairs with |sim| < {th:3.1f} : {ratio:6.2%}")
    
    high_ratio = (abs(off_diag) > 0.9).mean()
    print(f"\n⚠️  % pairs with |sim| > 0.9 : {high_ratio:.2%} (potential code collapse)")
    
    grade = (
        "✅ Excellent" if mean_abs < 0.05 else
        "👍 Good"       if mean_abs < 0.1  else
        "⚠️  Fair"       if mean_abs < 0.2  else
        "❌ Poor"
    )
    print("-"*70)
    print(f"Orthogonality grade: {grade}")
    print("="*70)
    
    # 可视化
    if plot and HAS_VIS:
        # === 1. 热力图 ===
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            sim_matrix.cpu().numpy(),
            center=0,
            cmap='coolwarm',
            square=True,
            cbar_kws={'shrink': 0.8}
        )
        plt.title('Codebook Cosine Similarity Matrix')
        plt.xlabel('Codeword Index')
        plt.ylabel('Codeword Index')
        plt.tight_layout()
        if plot_path_sim:
            plt.savefig(plot_path_sim, dpi=150)
            print(f"📈 Similarity heatmap saved to: {plot_path_sim}")
        plt.close()

        # === 2. 相似度分布直方图 ===
        plt.figure(figsize=(10, 6))
        bins = torch.arange(-1.0, 1.01, 0.01).numpy()  # -1.0 到 1.0，步长 0.01
        plt.hist(off_diag, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        plt.xlim(-1.0, 1.0)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency (Number of Pairs)')
        plt.title('Distribution of Off-Diagonal Cosine Similarities (bin size = 0.01)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        if plot_path_hist:
            plt.savefig(plot_path_hist, dpi=150)
            print(f"📊 Histogram saved to: {plot_path_hist}")
        plt.close()
    elif plot and not HAS_VIS:
        print("\n⚠️  Plot requested but seaborn/matplotlib not installed.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate orthogonality of VQ codebook from checkpoint.")
    parser.add_argument(
        "ckpt_path",
        type=str,
        help="Path to the model checkpoint file (e.g., models/model.pth)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots: similarity heatmap and histogram"
    )
    parser.add_argument(
        "--plot_path_sim",
        type=str,
        default=None,
        help="Path to save the similarity heatmap (e.g., plots/sim.png). "
             "Default: <ckpt>_similarity.png"
    )
    parser.add_argument(
        "--plot_path_hist",
        type=str,
        default=None,
        help="Path to save the similarity distribution histogram (e.g., plots/hist.png). "
             "Default: <ckpt>_similarity_hist.png"
    )
    args = parser.parse_args()
    
    # 设置默认绘图路径
    if args.plot:
        base = os.path.splitext(args.ckpt_path)[0]
        if args.plot_path_sim is None:
            args.plot_path_sim = base + "_similarity.png"
        if args.plot_path_hist is None:
            args.plot_path_hist = base + "_similarity_hist.png"
    
    codebook = load_codebook_from_checkpoint(args.ckpt_path)
    evaluate_orthogonality(
        codebook,
        plot=args.plot,
        plot_path_sim=args.plot_path_sim,
        plot_path_hist=args.plot_path_hist
    )

if __name__ == "__main__":
    main()
