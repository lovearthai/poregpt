import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, window_size: int = 33, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.window_size = window_size
        self.dropout_p = dropout

        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.nhead, self.head_dim

        # 1. 投影并变换为 SDPA 期望的标准形状 [B, H, T, D]
        # 这种形状最易触发 Flash Attention 优化内核
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # 2. 向量化生成局部注意力布尔掩码
        # True 表示该位置被屏蔽 (Masked)
        device = x.device
        indices = torch.arange(T, device=device)
        # 计算相对距离矩阵 [T, T]
        rel_pos = (indices[:, None] - indices[None, :]).abs()
        mask = rel_pos > (self.window_size // 2)

        # 3. 使用原生高效算子 SDPA
        # 内部自动处理内存管理，不显式存储 T x T 的 Attention Map
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,  # 广播到 [B, H, T, T]
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        # 4. 合并多头并恢复形状 [B, T, C]
        output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(output)

class LocalTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, window_size: int = 33,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        # 使用 Pre-Norm 结构提升长序列训练的稳定性
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = LocalAttention(d_model, nhead, window_size, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # 第一阶段：Attention (Pre-Norm)
        x = self.norm1(src)
        src = src + self.dropout1(self.self_attn(x))

        # 第二阶段：FFN (Pre-Norm)
        x = self.norm2(src)
        src = src + self.ffn(x)
        return src

    def forward_chunked(self, src: torch.Tensor, chunk_size: int = 500, overlap_size: int = 65) -> torch.Tensor:
        """
        分块处理长序列，保持梯度链完整性
        
        Args:
            src: [B, T, C]
            chunk_size: 基础块大小
            overlap_size: 重叠大小，应与window_size相当
        """
        B, T, C = src.shape
        
        if T <= chunk_size:
            return self.forward(src)
        
        # 分块处理
        outputs = []
        num_chunks = (T + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size + overlap_size, T)
            
            # 确保不会超出边界
            if end_idx > T:
                end_idx = T
                start_idx = max(0, end_idx - chunk_size - overlap_size)
            
            # 提取块
            chunk = src[:, start_idx:end_idx, :]  # [B, actual_chunk_size, C]
            
            # 通过当前层处理这个块
            chunk_output = self.forward(chunk)
            
            # 提取非重叠部分
            if i == 0:
                # 第一块：取全部有效部分
                effective_start = 0
                effective_end = min(chunk_size, end_idx - start_idx)
            else:
                # 后续块：跳过重叠部分（保留重叠部分的计算结果以保持梯度）
                effective_start = overlap_size
                effective_end = min(chunk_size + overlap_size, end_idx - start_idx)
            
            effective_output = chunk_output[:, effective_start:effective_end, :]
            outputs.append(effective_output)
        
        return torch.cat(outputs, dim=1)

class LocalTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 window_size: int = 33, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LocalTransformerEncoderLayer(d_model, nhead, window_size, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        # Pre-Norm 模式需要在最后加一层归一化
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output)
        return self.norm_final(output)

    def forward_chunked(self, src: torch.Tensor, chunk_size: int = 500, overlap_size: int = 65) -> torch.Tensor:
        """
        分块处理长序列，保持梯度链完整性
        
        Args:
            src: [B, T, C]
            chunk_size: 基础块大小
            overlap_size: 重叠大小，应与window_size相当
        """
        B, T, C = src.shape
        
        if T <= chunk_size:
            return self.forward(src)
        
        # 对每一层都进行分块处理
        output = src
        for layer in self.layers:
            output = layer.forward_chunked(output, chunk_size, overlap_size)
        
        return self.norm_final(output)
