import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch import FSQ # <-- 添加这一行
from typing import Tuple, Dict
# 导入新的 CNN 模型
from .cnn_model import NanoporeCNNModel
from vector_quantize_pytorch import ResidualFSQ  # 确保你已经正确安装或引用了该类
from vector_quantize_pytorch import FSQ # <-- 添加这一行
from vector_quantize_pytorch import LFQ

class NanoporeVQEModel_V22(nn.Module):
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
        codebook_size: int = 65536,
        codebook_dim: int = 16,
        codebook_nqtz: int = 2,
        cnn_type: int = 0,
        freeze_cnn: bool = False,
        cnn_checkpoint_path: str = None,
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
        
        # 必须使用encoder最后一步是tanh的卷积层
        self.cnn_model = NanoporeCNNModel(cnn_type=cnn_type)
        # 设置 codebook_dim 根据 cnn_type
        self.codebook_dim = codebook_dim
        self.cnn_type = cnn_type
        self.codebook_size = codebook_size
        self.cnn_stride = self.cnn_model.stride
        self.RF = self.cnn_model.RF
        print(f"codebook_dim:{codebook_dim}")
        # 假设 self.cnn_model.out_channels (也就是原来的 codebook_dim) 是一个大于4的数，
        # 比如是 128
        cnn_output_dim = self.cnn_model.out_channels

        # 实例化 ResidualFSQ
        # 建议：将 ResidualFSQ 的内建 dim 设置为 fsq_level_n，
        # 这样它内部就不会再创建额外的线性层，由我们手动控制。
        # 直接实例化 FSQ，传入 dim 参数
        # 只要 dim (128) != len(fsq_levels) (4)，FSQ 内部会自动创建 project_in 和 project_out
       
        self.vq = LFQ(
            codebook_size = codebook_size,      # codebook size, must be a power of 2
            dim = codebook_dim,                   # this is the input feature dimension, defaults to log2(codebook_size) if not defined
            entropy_loss_weight = 0.1,  # how much weight to place on entropy loss
            diversity_gamma = 1.        # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
        )


        # 使用 LFQ（Lookup-Free Quantization） 时，必须将输入信号（或特征）的维度（dim）与 LFQ 层的 dim 参数对齐。这是 LFQ 工作机制决定的。
        # LFQ 的核心操作是对 输入向量的每一个维度独立进行二值化（量化为 +1 或 -1）。
        # 输入要求：LFQ 期望的输入张量形状是 [B, ..., D]，其中 D 必须等于你在初始化 LFQ 时指定的 dim。
        # 你的原始信号：Nanopore 原始信号通常是 1D 的，形状为 [B, 1, T]（T 是时间点数，可能长达数千甚至上万）。
        # CNN Encoder 的作用：你现有的 NanoporeCNNModel 正是扮演了这个“降维+特征提取”的角色。它的输出 z_cnn 形状是 [B, C, N]，其中：
        # C 是 CNN 的输出通道数（例如 64, 128, 256），这就是特征维度。
        # N 是下采样后的时间步长（例如 T/5）。
        # 因此，CNN Encoder 的输出通道数 C 就是你提供给 LFQ 的 dim。
        self.project_in = nn.Linear(cnn_output_dim,codebook_dim)
        self.project_out = nn.Linear(codebook_dim, cnn_output_dim)


        # RFSQ 的总有效“码本”是分层的
        self.num_quantizers = codebook_nqtz
        # 如果有初始codebook路径，加载它
        # 如果有CNN检查点路径，加载权重
        if cnn_checkpoint_path:
            self._load_cnn_weights(cnn_checkpoint_path, freeze_cnn)

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
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
            print("预训练模型权重键 (前几个):", list(cnn_state_dict.keys())[:5])
            print("当前模型权重键 (前几个):", list(self.state_dict().keys())[:5])
            # 预训练模型权重键 (前几个): ['encoder.0.mean_conv.weight', 'encoder.0.std_conv.weight', 'encoder.1.weight', 'encoder.1.bias', 'encoder.1.running_mean']
            # 当前模型权重键 (前几个): ['cnn_model.encoder.0.mean_conv.weight', 'cnn_model.encoder.0.std_conv.weight', 'cnn_model.encoder.1.weight', 'cnn_model.encoder.1.bias', 'cnn_model.encoder.1.running_mean']
            # --- 添加以下逻辑 ---
            # 假设当前模型的 encoder 部分是通过 'cnn_model.encoder' 这个属性访问的
            # 我们需要将预训练权重的 'encoder.xxxx' 映射到 'cnn_model.encoder.xxxx'
            mapped_cnn_state_dict = {}
            for k, v in cnn_state_dict.items():
                if k.startswith('encoder.'): # 如果原始键是以 'encoder.' 开头
                    new_k = 'cnn_model.' + k # 将其映射为 'cnn_model.encoder.xxxx'
                    mapped_cnn_state_dict[new_k] = v
                else:
                    # 如果不是以 'encoder.' 开头（例如 decoder 或其他部分），可以选择跳过或也进行相应映射
                    pass # 或者继续处理其他部分，如果需要的话
                        # 只加载encoder和decoder的权重
                        # encoder_decoder_keys = [k for k in cnn_state_dict.keys() if k.startswith(('encoder.', 'decoder.'))]
             # 现在使用映射后的字典
            cnn_state_dict = mapped_cnn_state_dict

            # 原来的筛选逻辑现在应该能找到匹配项了
            # 注意这里也改为 'cnn_model.encoder.'
            encoder_decoder_keys = [k for k in cnn_state_dict.keys() if k.startswith(('cnn_model.encoder.'))]
            if not encoder_decoder_keys:
                print(f"⚠️ 在checkpoint中未找到encoder/decoder权重")
                return
           # --- 添加结束 ---

            # 获取当前模型状态
            model_state = self.state_dict()
            loaded_keys = []

            for key in encoder_decoder_keys:
                if key in model_state and cnn_state_dict[key].shape == model_state[key].shape:
                    print(f"加载参数:{key}")
                    model_state[key] = cnn_state_dict[key]
                    loaded_keys.append(key)

            # 加载权重
            self.load_state_dict(model_state, strict=False)
            #print(f"✅ 加载了 {len(loaded_keys)} 个encoder/decoder参数")
            print(f"✅ 加载了 {len(loaded_keys)} 个encoder参数")

            # 冻结参数（如果需要）
            freeze_cnt = 0
            if freeze_cnn:
                #print("🔒 冻结encoder和decoder参数")
                print("🔒 冻结encoder参数")
                for name, param in self.named_parameters():
                    #if name.startswith(('encoder.', 'decoder.')):
                    #if name.startswith(('encoder.')):
                    if name.startswith(('cnn_model.encoder.')):      # <- 修改为新的前缀
                        freeze_cnt +=1
                        param.requires_grad = False
                        print(f"冻结参数:{name}")
            print(f"✅ 冻结了 {freeze_cnt} 个encoder参数")
        except Exception as e:
            print(f"❌ 加载CNN权重失败: {e}")




    def _print_vq_config(self) -> None:
        """打印 VQ 配置信息（仅 rank 0）"""
        print("Intialized VectorQuantize with the following hyperparameters:")
        print(f"  codebook_size: {self.codebook_size}")
        print(f"  kmeans_init: True")
        print(f"  kmeans_iters: 10")
        print(f"  decay: 0.99")
        print(f"  threshold_ema_dead_code: 2")
        #print(f"  commitment_weight: {self.vq.commitment_weight}")
        #print(f"  codebook_diversity_loss_weight: {self.vq.codebook_diversity_loss_weight}")
        #print(f"  orthogonal_reg_weight: {self.vq.orthogonal_reg_weight}")
        #print(f"  orthogonal_reg_max_codes: 256")
        #print(f"  orthogonal_reg_active_codes_only: True")
        print(f"  cnn_type: {self.cnn_type}")
        print("-" * 60)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播函数 (升级版：支持 CNN + Transformer 架构)。

        Args:
            x (torch.Tensor): 输入信号，形状 [B, 1, T]
                (例如: B=4, T=2560 -> )

        Returns:
            recon (torch.Tensor): 重建信号，[B, 1, T]
            indices (torch.Tensor): VQ 离散 token，[B, N] (N = T // 5)
            loss (torch.Tensor): VQ 总损失（标量）
            loss_breakdown (dict): 损失分项（commitment, diversity, ortho...）
        """


        # 1. CNN 编码器
        z_cnn = self.cnn_model.encode(x) # [B, C, N]

        # 2. 维度变换
        z_permuted = z_cnn.permute(0, 2, 1) # [B, N, C]

        # ======================================================================
        # 3. 显式线性投影 (对齐维度)
        # ======================================================================
        # 将 CNN 输出维度 (例如 128) 映射到 FSQ 维度 (例如 4)
        z_projected = self.project_in(z_permuted) # [B, N, fsq_level_n]

        # 4. 应用 Residual FSQ
        # 注意这里传入的是 z_projected
        z_quantized_projected, indices , entropy_aux_loss = self.vq(z_projected)

        # 5. 显式投影回原始维度
        # 将量化后的维度 (例如 4) 映射回 CNN 解码器需要的维度 (例如 128)
        z_quantized_permuted = self.project_out(z_quantized_projected) # [B, N, C]
        # z_quantized_permuted: [B, N, 256] (量化后的连续特征，用于 Decoder)
        # indices: [B, N] (离散的 Token ID，用于存储/下游任务)
        #   例如 indices:
        # ======================================================================
        # 5. 解码器准备 (Decoder Preparation)
        #    目标：将特征格式转换回 CNN 解码器需要的格式
        # ======================================================================
        # Decoder (反卷积网络) 需要的格式是 [B, Channels, Length]
        z_quantized = z_quantized_permuted.permute(0, 2, 1)
        # z_quantized: [B, 256, N] -> 例如:
        # ======================================================================
        # 6. 解码器 (Decoder - 信号重构)
        #    目标：将量化特征重构回原始信号空间
        # ======================================================================
        recon = self.cnn_model.decode(z_quantized)
        # recon: [B, 1, T_recon]
        #   (理论上 T_recon 应该等于 T，但为了防止反卷积导致的长度微小差异)

        # ======================================================================
        # 7. 长度对齐 (Length Alignment)
        #    目标：确保输出信号长度与输入完全一致
        # ======================================================================
        target_len = x.shape[-1]  # 输入信号的原始长度 (2560)
        current_len = recon.shape[-1] # 重构信号的当前长度

        if current_len > target_len:
            # 如果重构信号过长（通常由 Padding 引起），进行裁剪
            recon = recon[..., :target_len]
        elif current_len < target_len:
            # 如果重构信号过短，进行填充 (Pad)
            # F.pad 的参数是 (左填充, 右填充)，这里只在时间轴末尾填充
            recon = F.pad(recon, (0, target_len - current_len))
        return recon, indices,entropy_aux_loss
