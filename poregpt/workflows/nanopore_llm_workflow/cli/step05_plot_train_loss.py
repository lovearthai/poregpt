import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 CSV 文件
csv_path = "train_loss.csv"
df = pd.read_csv(csv_path)

# 检查是否包含 codebook_usage 列
has_usage = 'codebook_usage' in df.columns

# 创建两个子图：上下布局
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# ========================
# 上图：原始线性尺度
# ========================
ax1.plot(df['epoch'], df['recon_loss'], label='Reconstruction Loss', color='tab:blue')
ax1.plot(df['epoch'], df['commit_loss'], label='Commitment Loss', color='tab:orange')
ax1.plot(df['epoch'], df['total_loss'], label='Total Loss', color='tab:green')

ax1.set_ylabel('Loss (linear)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, linestyle='--', alpha=0.5)

# 如果有 codebook_usage，添加右侧 Y 轴
if has_usage:
    ax2 = ax1.twinx()
    ax2.plot(df['epoch'], df['codebook_usage'], label='Codebook Usage', color='tab:red', linestyle='-.')
    ax2.set_ylabel('Codebook Usage', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
else:
    ax1.legend(loc='upper right')

# ========================
# 下图：log10(loss) 尺度
# ========================
# 避免除零：加 epsilon
eps = 1e-8
ax3.plot(df['epoch'], np.log10(df['recon_loss'] + eps), label='log10(Recon Loss)', color='tab:blue')
ax3.plot(df['epoch'], np.log10(df['commit_loss'] + eps), label='log10(Commit Loss)', color='tab:orange')
ax3.plot(df['epoch'], np.log10(df['total_loss'] + eps), label='log10(Total Loss)', color='tab:green')

ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss (log10 scale)', color='black')
ax3.tick_params(axis='y', labelcolor='black')
ax3.grid(True, linestyle='--', alpha=0.5)

# 如果有 codebook_usage，在下图也显示（线性右轴，不取 log）
if has_usage:
    ax4 = ax3.twinx()
    ax4.plot(df['epoch'], df['codebook_usage'], label='Codebook Usage', color='tab:red', linestyle='-.')
    ax4.set_ylabel('Codebook Usage', color='tab:red')
    ax4.tick_params(axis='y', labelcolor='tab:red')
    
    # 合并图例 for lower plot
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_4, labels_4 = ax4.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper right')
else:
    ax3.legend(loc='upper right')

# 全局标题
plt.suptitle('Training Loss and Codebook Usage (Linear and Log10 Scale)')
plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为 suptitle 留空间
plt.savefig("train_loss_and_usage_log.png", dpi=300)
plt.show()
