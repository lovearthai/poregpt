import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from typing import Tuple, Dict
# 导入新的 CNN 模型
from .cnn_model import NanoporeCNNModel

class NanoporeVQEModel_V15(nn.Module):
    """
    Nanopore VQ Tokenizer for Direct RNA Sequencing (130 bps, 4 kHz)

    支持多种 CNN 架构配置，通过 `cnn_type` 切换：
        - cnn_type=0: 大容量非严格对称模型（默认）
        - cnn_type=1: 小容量严格对称模型（通道数 1→16→32→64）

    设计目标通用：
        - 感受野 ≈ 33 采样点（≈1 个 RNA 碱基）
        - 总下采样率 = 5×（每碱基 ≈6 个 tokens）
        - 输出 codebook_dim 维 latent，直接用于 VQ
        - Decoder 在 cnn_type=1 时严格对称于 encoder

    适用于：VQ tokenizer + LLM basecalling pipeline
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        codebook_dim: int = 64,
        codebook_decay: float = 0.99,
        codebook_emadc: int = 2,

        commitment_weight: float = 1.0,
        orthogonal_reg_weight: float = 1.0,
        codebook_diversity_loss_weight: float = 1.0,
        cnn_type: int = 0,
        learnable_codebook: bool= True,
        init_codebook_path: str = None,
        freeze_cnn: bool = False,
        cnn_checkpoint_path: str = None
    ):
        """
        初始化 NanoporeVQModel。

        Args:
            codebook_size (int): VQ 码本大小。
            codebook_dim (int): VQ 嵌入维度（即 encoder 最终输出通道数）。
            commitment_weight (float): VQ commitment loss 权重。
            orthogonal_reg_weight (float): 正交正则化权重。
            codebook_diversity_loss_weight (float): 码本多样性损失权重。
            cnn_type (int): CNN 架构类型。
                - 0: 默认大模型（1 → 64 → 128 → codebook_dim）
                - 1: 严格对称小模型（1 → 16 → 32 → 64），此时 codebook_dim 必须为 64
        """
        super().__init__()

        self.cnn_model = NanoporeCNNModel(cnn_type=cnn_type)
        cnn_out_dim = self.cnn_model.out_channels
        d_model = self.cnn_model.out_channels  # 自动设置为CNN输出维度
        self.codebook_dim = codebook_dim
        self.cnn_type = cnn_type
        self.latent_dim = codebook_dim
        self.codebook_size = codebook_size
        self.cnn_stride = self.cnn_model.stride
        self.RF = self.cnn_model.RF

        # --- 在这里添加激活函数 ---
        # 这个激活函数将在 VQ 之前应用，以规范化特征分布
        self.activation_func = nn.SiLU() # 或者使用 nn.ReLU()
        # --- 添加结束 --


        print(f"codebook_dim:{codebook_dim}")
       # ======================================================================
        # VECTOR QUANTIZATION (VQ)
        # ======================================================================
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if learnable_codebook == True:
            ema_update = False
        else:
            ema_update = True
        # 实例化 ResidualFSQ
        # 建议：将 ResidualFSQ 的内建 dim 设置为 fsq_level_n，
        # 这样它内部就不会再创建额外的线性层，由我们手动控制。
        
        self.project_in = nn.Linear(cnn_out_dim, codebook_dim)
        self.project_out = nn.Linear(codebook_dim, cnn_out_dim)


        # 原理：threshold_ema_dead_code 代码中的 expire_codes_ 函数会检测活跃度低于阈值的码，并强制用当前 Batch 中的随机向量替换它。这能像“强心针”一样不断激活坍缩的码。
        self.vq = VectorQuantize(
            dim=codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=True,
            kmeans_iters=10,
            decay=codebook_decay,
            threshold_ema_dead_code=codebook_emadc,
            commitment_weight=commitment_weight,
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,
            orthogonal_reg_weight=orthogonal_reg_weight,
            orthogonal_reg_max_codes=256,
            orthogonal_reg_active_codes_only=True,
            learnable_codebook=learnable_codebook,
            ema_update = ema_update,
        )
        
        # 如果有CNN检查点路径，加载权重
        if cnn_checkpoint_path:
            self._load_cnn_weights(cnn_checkpoint_path, freeze_cnn)
 

        if rank == 0:
            self._print_vq_config()
   
    def _load_cnn_weights(self, cnn_checkpoint_path, freeze_cnn=False):
        """从检查点加载CNN权重"""
        try:
            import os
            import torch
            
            if not os.path.isfile(cnn_checkpoint_path):
                print(f"⚠️ CNN checkpoint文件不存在: {cnn_checkpoint_path}")
                return
            
            print(f"📥 从 {cnn_checkpoint_path} 加载CNN权重")
            
            # 加载检查点
            cnn_ckpt = torch.load(cnn_checkpoint_path, map_location='cpu',weights_only=False)
            cnn_state_dict = cnn_ckpt.get('model_state_dict', cnn_ckpt)
            
            # 如果权重有'module.'前缀，去掉它
            if list(cnn_state_dict.keys())[0].startswith('module.'):
                cnn_state_dict = {k.replace('module.', ''): v for k, v in cnn_state_dict.items()}
            
            # 只加载encoder和decoder的权重
            encoder_decoder_keys = [k for k in cnn_state_dict.keys() 
                                   if k.startswith(('encoder.', 'decoder.'))]
            
            if not encoder_decoder_keys:
                print(f"⚠️ 在checkpoint中未找到encoder/decoder权重")
                return
            
            # 获取当前模型状态
            model_state = self.state_dict()
            loaded_keys = []
            
            for key in encoder_decoder_keys:
                if key in model_state and cnn_state_dict[key].shape == model_state[key].shape:
                    model_state[key] = cnn_state_dict[key]
                    loaded_keys.append(key)
            
            # 加载权重
            self.load_state_dict(model_state, strict=False)
            print(f"✅ 加载了 {len(loaded_keys)} 个encoder/decoder参数")
            
            # 冻结参数（如果需要）
            if freeze_cnn:
                print("🔒 冻结encoder和decoder参数")
                for name, param in self.named_parameters():
                    if name.startswith(('encoder.', 'decoder.')):
                        param.requires_grad = False
            
        except Exception as e:
            print(f"❌ 加载CNN权重失败: {e}")



    def _print_vq_config(self) -> None:
        """打印 VQ 配置信息（仅 rank 0）"""
        print("Intialized VectorQuantize with the following hyperparameters:")
        print(f"  dim: {self.latent_dim}")
        print(f"  codebook_size: {self.codebook_size}")
        print(f"  kmeans_init: True")
        print(f"  kmeans_iters: 10")
        print(f"  decay: 0.99")
        print(f"  threshold_ema_dead_code: 2")
        print(f"  commitment_weight: {self.vq.commitment_weight}")
        print(f"  codebook_diversity_loss_weight: {self.vq.codebook_diversity_loss_weight}")
        print(f"  orthogonal_reg_weight: {self.vq.orthogonal_reg_weight}")
        print(f"  orthogonal_reg_max_codes: 256")
        print(f"  orthogonal_reg_active_codes_only: True")
        print(f"  cnn_type: {self.cnn_type}")
        print("-" * 60)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入信号，形状 [B, 1, T]

        Returns:
            recon (torch.Tensor): 重建信号，[B, 1, T]
            indices (torch.Tensor): VQ 离散 token，[B, T//5]
            loss (torch.Tensor): VQ 总损失（标量）
            loss_breakdown (dict): 损失分项（commitment, diversity, ortho...）
        """
        # Encode: [B, 1, T] → [B, C, T//5]
        z_continuous = self.cnn_model.encode(x)

        # Permute for VQ: [B, C, N] → [B, N, C]
        z_permuted = z_continuous.permute(0, 2, 1)
        # ======================================================================
        # 3. 显式线性投影 (对齐维度)
        # ======================================================================
        # 将 CNN 输出维度 (例如 128) 映射到 FSQ 维度 (例如 4)
        z_projected = self.project_in(z_permuted) # [B, N, fsq_level_n]

        # Quantize
        z_vq, indices, loss, loss_breakdown = self.vq(
            z_projected, return_loss_breakdown=True
        )
        
        # 5. 显式投影回原始维度
        # 将量化后的维度 (例如 4) 映射回 CNN 解码器需要的维度 (例如 128)
        z_quantized_permuted = self.project_out(z_vq) # [B, N, C]

        # Back to [B, C, N] for decoder
        z_quantized = z_quantized_permuted.permute(0, 2, 1)

        # Decode
        recon = self.cnn_model.decoder(z_quantized)

        # Length alignment: ensure recon length == input length
        target_len = x.shape[-1]
        current_len = recon.shape[-1]
        if current_len > target_len:
            recon = recon[..., :target_len]
        elif current_len < target_len:
            recon = F.pad(recon, (0, target_len - current_len))

        return recon, indices, loss, loss_breakdown
