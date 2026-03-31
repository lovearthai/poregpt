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

# 导入局部注意力模块
from .local_attention import LocalTransformerEncoder


class NanoporeVQEModel_V19(nn.Module):
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
        fsq_level_d: int = 5,
        fsq_level_n: int = 4,
        codebook_size: int = 8192,
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
        
        d_model = self.cnn_model.out_channels   # 自动设置为CNN输出维度
        codebook_dim = d_model
        # 设置 codebook_dim 根据 cnn_type
        self.fsq_level_n = fsq_level_n
        self.fsq_level_d = fsq_level_d
        self.codebook_dim = codebook_dim
        self.cnn_type = cnn_type
        self.codebook_size = codebook_size
        self.cnn_stride = self.cnn_model.stride
        self.RF = self.cnn_model.RF
        print(f"codebook_dim:{codebook_dim}")
        # 假设 self.cnn_model.out_channels (也就是原来的 codebook_dim) 是一个大于4的数，
        # 比如是 128
        cnn_output_dim = self.cnn_model.out_channels
        fsq_levels = [fsq_level_d] * fsq_level_n

        # 2. 核心检查逻辑：
        #    - isinstance(x, int): 确保是整数类型 (排除 4.0 这种 float)
        #    - x > 0: 确保是正数 (排除 0 和负数)
        if all(isinstance(x, int) and x > 0 for x in fsq_levels):
            # 3. 计算乘积: 4 * 4 * 4 * 5
            fsq_codebook_size = math.prod(fsq_levels)
            print(f"✅ 验证通过，码本大小: {fsq_codebook_size}")
        else:
            # 如果检查失败，找出具体原因（可选的调试代码）
            for i, x in enumerate(fsq_levels):
                if not isinstance(x, int):
                    print(f"❌ 索引 {i} 的值 {x} 不是整数类型 (可能是浮点数)")
                elif x <= 0:
                    print(f"❌ 索引 {i} 的值 {x} 不是正数")
            raise ValueError("fsq_levels 格式错误")
            return

        assert fsq_codebook_size == codebook_size, f"码本大小不匹配！计算值: {fsq_codebook_size}, 预期值: {codebook_size}"
        # 实例化 ResidualFSQ
        # 建议：将 ResidualFSQ 的内建 dim 设置为 fsq_level_n，
        # 这样它内部就不会再创建额外的线性层，由我们手动控制。
       

        if cnn_output_dim == 512:
            D_hidden1 = 128
            D_hidden2 = 128
        else:
            raise ValueError("not accepted cnn_out_dim")
        
        self.project_in = nn.Sequential(
            nn.Linear(cnn_output_dim, D_hidden1), 
            nn.SiLU(),           # 非线性激活
            nn.Linear(D_hidden1, D_hidden2),
            nn.SiLU(),
            nn.Linear(D_hidden2, fsq_level_n),

        )    
        self.project_out = nn.Linear(fsq_level_n, cnn_output_dim)

        # 实例化 ResidualFSQ
        # 注意：RFSQ 内部会自动根据 dim 和 levels 的长度处理投影 (project_in/out)
        self.vq = ResidualFSQ(
            levels = fsq_levels,
            num_quantizers = codebook_nqtz,
            dim = fsq_level_n,
            quantize_dropout = False,
            quantize_dropout_cutoff_index = 1, # 至少保留第一层量化
            # 如果需要固定总码本大小，可以调整 levels
        )
        print(f"self.vq.levels:{self.vq.levels}") 

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
        print(f"  fsq_level_n: {self.fsq_level_n}")
        print(f"  fsq_level_d: {self.fsq_level_d}")
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
        z_quantized_projected, level_indices = self.vq(z_projected)

        # 5. 显式投影回原始维度
        # 将量化后的维度 (例如 4) 映射回 CNN 解码器需要的维度 (例如 128)
        z_quantized_permuted = self.project_out(z_quantized_projected) # [B, N, C]

        # 修复 ResidualFSQ 返回的 indices 类型为 int32 的问题，统一转为 int64
        # indices = indices.to(torch.int64) 是一个 类型转换 (type casting) 操作。PyTorch 中的 tensor.to() 方法，当用于类型转换时，其行为取决于原始张量是否需要被梯度追踪（即 requires_grad 属性）。
        # 对不需要梯度的张量（常见情况）：在向量量化（VQ）的场景下，indices 通常是量化过程产生的离散索引。
        # 这个索引本身是一个 离散的、非连续的决策结果，它不是通过可导函数计算出来的，因此 通常不会被设计为需要梯度。
        # 在您提供的 V9 的 forward 函数中，indices 是 self.vq(z_permuted_for_vq) 的返回值。
        # ResidualFSQ (以及其内部的 FSQ) 在量化过程中，会执行类似 argmax 或 searchsorted 这样的操作来找到最近的码本向量索引。
        # 些操作在数学上是不可导的，因此其返回的 indices 通常会自然地继承一个 requires_grad=False 的属性
        # 当一个 requires_grad=False 的张量进行 to(torch.int64) 转换时，这是一个纯 CPU/GPU 上的数据类型转换，它 不参与梯度计算图，因此 完全不会影响任何反向传播链路。
        # 对需要梯度的张量（罕见情况）：如果一个张量 x 的 requires_grad=True，那么 x.to(torch.int64) 操作会中断梯度流。
        # 这是因为类型转换（例如从 float32 到 int64）改变了数据的性质，梯度无法在这种非连续的类型变换中流动。
        # 如果 indices 意外地被设置为 requires_grad=True，那么这行代码会切断从 indices 向前追溯的梯度路径。
        level_indices = level_indices.to(torch.int64)


        # 从 level_indices 的形状动态获取量化器数量
        B, N, indices_n_quantizers = level_indices.shape # 例如 B=4, N=512, n_quantizers=4
        # --- 新增代码：合并多层 indices 为单一的 uni_indices ---
        # 获取每层的码本大小
        levels = torch.tensor(self.vq.levels, dtype=torch.long, device=level_indices.device) # [L0, L1, L2, L3], e.g., [8192, 8192, 8192, 8192]

        # 断言确保量化器数量与 levels 数量匹配
        assert self.num_quantizers == indices_n_quantizers, f"Number of quantizers ({codebook_nqtz}) must match number of levels ({indices_n_quantizers})"


        # 计算 FSQ 的总码本大小 (L0 * L1 * ... * LN)
        fsq_size = torch.prod(levels).item() # e.g., 8192^4 或 8*5*5*5


        # --- 合并策略：让骨干层 (idx[0]) 拥有最高权重 ---
        # uni_token = idx[0] * fsq_size^(k-1) + idx[1] * fsq_size^(k-2) + ... + idx[k-1] * fsq_size^0
        # 其中 k = indices_n_quantizers
        # 直接计算权重: [fsq_size^(k-1), fsq_size^(k-2), ..., fsq_size^0]
        # 对应索引:    [idx[0],       idx[1],       ..., idx[k-1]]
        exponents = torch.arange(indices_n_quantizers - 1, -1, -1, dtype=torch.long, device=level_indices.device)
        # exponents = [k-1, k-2, ..., 1, 0]

        fsq_size_tensor = torch.tensor(fsq_size, dtype=torch.long, device=level_indices.device)
        multipliers = torch.pow(fsq_size_tensor, exponents)
        # multipliers = [fsq_size^(k-1), fsq_size^(k-2), ..., fsq_size^1, fsq_size^0]
        
        # 计算加权索引并求和
        # Broadcasting: level_indices [B, N, indices_n_q] * multipliers [indices_n_q] -> [B, N, indices_n_q]
        weighted_indices = level_indices * multipliers.view(1, 1, -1)
        # Sum along the last dimension (quantizer dimension) -> [B, N]
        uni_indices = torch.sum(weighted_indices, dim=-1) # [B, N]

        # --- Debug: 验证映射逻辑 ---
        # 为了简化，我们只验证第一个 batch 和第一个 time step (即 [0, 0, :])

        # --- Debug: 验证映射逻辑 ---
        if False and B > 0 and N > 0:
            debug_batch_idx = 0
            debug_time_idx = 0
            single_level_indices = level_indices[debug_batch_idx, debug_time_idx, :] # [indices_n_quantizers]
            single_uni_index = uni_indices[debug_batch_idx, debug_time_idx] # scalar

            # 手动计算验证
            manual_calculation = 0
            calculation_str_parts = []
            # exponents 是 [k-1, k-2, ..., 0]
            for i in range(indices_n_quantizers):
                idx_val = single_level_indices[i].item()
                power_val = exponents[i].item() # This will be k-1, k-2, ..., 0
                weight_val = fsq_size ** power_val
                contrib = idx_val * weight_val
                manual_calculation += contrib
                calculation_str_parts.append(f"{idx_val} * {fsq_size}^{power_val}")
            calculation_str = " + ".join(calculation_str_parts) + f" = {manual_calculation}"
            
            print(f"\n[DEBUG] Merging ResidualFSQ tokens (FSQ_SIZE={fsq_size}), backbone (idx[0]) has highest weight:")
            print(f"  Raw Indices (from each block): {single_level_indices.tolist()}")
            print(f"  Weights applied [highest->lowest]: {multipliers.tolist()}")
            print(f"  Calculation: {calculation_str}")
            print(f"  Final Uni Index: {single_uni_index.item()}")
            print(f"  Match: {'✓' if manual_calculation == single_uni_index.item() else '✗ MISMATCH!'}")
        # --- End Debug ---


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

        return recon, level_indices, uni_indices
