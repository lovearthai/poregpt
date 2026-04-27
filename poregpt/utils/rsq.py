import torch
from vector_quantize_pytorch import ResidualFSQ

# --- 辅助函数：统一基础逻辑 ---
def get_fsq_basis(levels, device):
    """ 计算 FSQ 进制基数 """
    return torch.cumprod(torch.tensor([1] + levels[:-1], device=device), dim=0)

# --- 函数 1: 输入单个整数 token_id ---


def get_rsq_vector_from_integer_bak(token_id, levels, num_quantizers, debug=False):
    """
    全新修改版：符合“个十百千”直觉的解算逻辑
    Layer 0 (骨干) 存储在 token_id 的高位，Layer 1 (细节) 存储在低位。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    levels_tensor = torch.tensor(levels, device=device)
    basis = get_fsq_basis(levels, device)
    
    # 单层容量 (例如 625)
    codebook_size = torch.prod(levels_tensor).item()

    if debug:
        print(f"\n{'='*20} FSQ DEBUG (V2: HIGH-ORDER FIRST) {'='*20}")
        print(f"[Input] Token ID: {token_id}")

    # 初始化层索引 [1, num_quantizers]
    layer_indices = torch.zeros((1, num_quantizers), dtype=torch.long, device=device)

    # --- 环节 1: 跨层级解算 (倒序解算，符合权重逻辑) ---
    temp_id = token_id
    
    # 核心修改：从最后一层(细节层)开始取余数，剩下的商留给前面的骨干层
    # 这样 token_id // codebook_size 得到的就是 Layer 0 的值
    for i in range(num_quantizers - 1, -1, -1):
        current_layer_id = temp_id % codebook_size
        layer_indices[0, i] = current_layer_id
        temp_id //= codebook_size

    # 打印调试信息（按 Layer 0 -> Layer N 顺序展示）
    if debug:
        for i in range(num_quantizers):
            cid = layer_indices[0, i].item()
            dim_coords = (cid // basis) % levels_tensor
            role = "骨干 (Skeleton)" if i == 0 else f"细节 (Refinement {i})"
            print(f"\n--- Layer {i} [{role}] ---")
            print(f"  > Layer Codebook ID: {cid}")
            print(f"  > Dimension Coordinates: {dim_coords.tolist()}")

        print(f"\n[Final Indices Tensor]: {layer_indices}")

    # --- 环节 2: 调用 API ---
    vector = get_rsq_vector_from_indices(layer_indices, levels, num_quantizers)

    if debug:
        print(f"\n[Output Vector] Sample values: {vector[0, :4].tolist()} ...")
        print(f"{'='*25} DEBUG END {'='*25}\n")

    return vector


def get_rsq_vector_from_integer(token_id, levels, num_quantizers, debug=False, use_fast=False):
    """
    增加了 use_fast 逻辑：针对特定的 levels 进行数学快速生成
    """
    # --- 快速数学路径 ---
    if use_fast and levels == [5, 5, 5, 5] and num_quantizers == 1:
        # FSQ 映射逻辑：将 [0, 4] 映射到 [-1.0, -0.5, 0, 0.5, 1.0]
        # 公式为: (coord - (level - 1) / 2) / ((level - 1) / 2) -> (coord - 2) / 2
        # 5进制分解：token_id = d3*5^3 + d2*5^2 + d1*5^1 + d0*5^0
        d3 = (token_id // 125) % 5
        d2 = (token_id // 25) % 5
        d1 = (token_id // 5) % 5
        d0 = token_id % 5

        # 转换为向量坐标并归一化
        # 注意：这里的顺序需要根据你 get_fsq_basis 的实现来匹配。通常是低位在前。
        vec = [ (float(d) - 2.0) / 2.0 for d in [d0, d1, d2, d3] ]

        # 模拟返回 tensor，确保与原 API 兼容
        return torch.tensor([vec], dtype=torch.float32)

    # --- 原始慢速路径 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    levels_tensor = torch.tensor(levels, device=device)
    basis = get_fsq_basis(levels, device)

    codebook_size = torch.prod(levels_tensor).item()

    if debug:
        print(f"\n{'='*20} FSQ DEBUG (V2: HIGH-ORDER FIRST) {'='*20}")
        print(f"[Input] Token ID: {token_id}")

    layer_indices = torch.zeros((1, num_quantizers), dtype=torch.long, device=device)
    temp_id = token_id

    for i in range(num_quantizers - 1, -1, -1):
        current_layer_id = temp_id % codebook_size
        layer_indices[0, i] = current_layer_id
        temp_id //= codebook_size

    if debug:
        for i in range(num_quantizers):
            cid = layer_indices[0, i].item()
            # 这里的 basis 处理逻辑需要与 fast 路径一致
            dim_coords = (cid // basis) % levels_tensor
            role = "骨干 (Skeleton)" if i == 0 else f"细节 (Refinement {i})"
            print(f"\n--- Layer {i} [{role}] ---")
            print(f" > Layer Codebook ID: {cid}")
            print(f" > Dimension Coordinates: {dim_coords.tolist()}")

    vector = get_rsq_vector_from_indices(layer_indices, levels, num_quantizers)

    if debug:
        print(f"\n[Output Vector] Sample values: {vector[0, :4].tolist()} ...")
        print(f"{'='*25} DEBUG END {'='*25}\n")

    return vector

def get_rsq_coords_from_integer(token_id, levels, num_quantizers, debug=False, use_fast=True):
    """
    从整数 Token ID 解码出每一层的维度坐标。

    Args:
        token_id: 输入的整数 Token ID (可以是 int 或 torch.Tensor)。
        levels: 每一维度的量化级别列表 (例如 [5, 5, 5, 5])。
        num_quantizers: 量化器的层数。
        debug: 是否打印调试信息。
        use_fast: 是否启用针对特定参数的快速数学路径。

    Returns:
        list: 包含每一层坐标的列表，例如 [[d1, d2, d3, d4], ...]。
    """
    
    # 🚀 快速路径：针对 num_quantizers=1 和 levels=[5,5,5,5] 的纯数学优化
    if use_fast and num_quantizers == 1 and levels == [5, 5, 5, 5]:
        val = token_id.item() if isinstance(token_id, torch.Tensor) else token_id
        
        # 根据权重 [1, 5, 25, 125] 进行拆解
        # 索引 0 对应权重 1
        d0 = val % 5
        
        # 索引 1 对应权重 5
        d1 = (val // 5) % 5
        
        # 索引 2 对应权重 25
        d2 = (val // 25) % 5
        
        # 索引 3 对应权重 125
        d3 = (val // 125) % 5
        
        # 按照索引顺序返回
        coords = [d0, d1, d2, d3]
        
        if debug:
            print(f"⚡ [Fast Path] Coords: {coords}")
            
        return [coords]
    # ==========================================================
    # 🐢 通用路径：原有的 PyTorch 逻辑
    # ==========================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    levels_tensor = torch.tensor(levels, device=device)
    basis = get_fsq_basis(levels, device) # 假设外部已定义此函数

    # 单层容量
    codebook_size = torch.prod(levels_tensor).item()

    if debug:
        print(f"\n{'='*20} DECODE COORDS DEBUG {'='*20}")
        print(f"[Input] Token ID: {token_id}")

    # 初始化层索引
    layer_indices = torch.zeros((1, num_quantizers), dtype=torch.long, device=device)

    # --- 环节 1: 跨层级解算 ---
    temp_id = token_id if isinstance(token_id, torch.Tensor) else torch.tensor(token_id, device=device)

    # 倒序解算
    for i in range(num_quantizers - 1, -1, -1):
        current_layer_id = temp_id % codebook_size
        layer_indices[0, i] = current_layer_id
        temp_id //= codebook_size

    # --- 环节 2: 计算每一层的维度坐标 ---
    all_dim_coords = []

    for i in range(num_quantizers):
        cid = layer_indices[0, i].item()
        # 将一维的 Codebook ID 还原为多维坐标
        dim_coords = (cid // basis) % levels_tensor
        all_dim_coords.append(dim_coords.tolist())

        if debug:
            role = "骨干 (L0)" if i == 0 else f"细节 (L{i})"
            print(f"Layer {i} [{role}] ID: {cid} -> Coords: {dim_coords.tolist()}")

    return all_dim_coords

# --- 函数 A：输入多层 indices 调用官方接口 ---
def get_rsq_vector_from_indices(indices, levels, num_quantizers):
    device = indices.device
    # 实例化 ResidualFSQ
    model = ResidualFSQ(
        levels = levels,
        num_quantizers = num_quantizers,
        bound_hard_clamp = True
    ).eval().to(device)

    with torch.no_grad():
        vector = model.get_output_from_indices(indices)
    return vector

# --- 函数 B：基于数学逻辑扩展 ---
def get_fsq_vector_from_indices_via_math(indices, levels, num_quantizers):
    device = indices.device
    levels_tensor = torch.tensor(levels, device=device).float()
    basis = get_fsq_basis(levels, device)

    final_vector = 0.
    for i in range(num_quantizers):
        layer_id = indices[:, i]
        layer_indices = (layer_id.unsqueeze(-1) // basis) % levels_tensor
        # 反量化
        layer_vector = 2 * (layer_indices / (levels_tensor - 1)) - 1
        # 应用残差缩放因子
        scale = levels_tensor ** -i
        final_vector += layer_vector * scale

    return final_vector

# --- 校验与使用示例 ---
if __name__ == "__main__":
    levels = [5, 5, 5, 5]
    num_q = 2 # 测试多层残差
    my_token_id = 358

    # 1. 测试单整数接口
    # ⚠️ 修正：你原代码 main 里的函数名写错了，这里已修正
    result_vector = get_rsq_vector_from_integer(my_token_id, levels, num_q)

    print(f"输入 ID: {my_token_id}")
    print(f"输出向量形状: {result_vector.shape}")
    print(f"输出向量前 4 维: {result_vector[0, :4]}")

    # 2. 对比校验 (API vs Math)
    mock_indices = torch.tensor([[10, 20]], device=result_vector.device) # 随机模拟两层 ID
    v_api = get_rsq_vector_from_indices(mock_indices, levels, num_q)
    v_math = get_fsq_vector_from_indices_via_math(mock_indices, levels, num_q)
    
    diff = torch.abs(v_api - v_math).max().item()
    print(f"\nAPI 与数学公式的一致性误差: {diff:.2e}")
