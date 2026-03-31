#export CUDA_VISIBLE_DEVICES=1,2
#export CUDA_VISIBLE_DEVICES=1,2
# ===== 1) 国产卡/MACA 运行时环境 =====
export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export MACA_CLANG=${MACA_PATH}/mxgpu_llvm
export DEVINFO_ROOT=${MACA_PATH}
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}
export PATH=${CUCC_PATH}:${MACA_PATH}/bin:${MACA_CLANG}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MACA_SMALL_PAGESIZE_ENABLE=1
export MCPYTORCH_DISABLE_PRINT=1

export MCCL_NET_GDR_LEVEL=7
export MCCL_MAX_NCHANNELS=18
export MCCL_P2P_LEVEL=SYS
export MCCL_LIMIT_RING_LL_THREADTHRESHOLDS=1
export FORCE_ACTIVATE_WAIT=1
export SET_DEVICE_NUMA_PREFERRED=1

export MAX_JOBS=60
export PYTORCH_ENABLE_SAME_RAND_A100=1
export MCCL_SOCKET_IFNAME=eth0

set -euo pipefail

# =========================
# 1) 国产卡 / MACA 运行时环境
# =========================
export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export MACA_CLANG=${MACA_PATH}/mxgpu_llvm
export DEVINFO_ROOT=${MACA_PATH}
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}

export PATH=${CUCC_PATH}:${MACA_PATH}/bin:${MACA_CLANG}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MACA_SMALL_PAGESIZE_ENABLE=1
export MCPYTORCH_DISABLE_PRINT=1
export MAX_JOBS=20
export PYTORCH_ENABLE_SAME_RAND_A100=1
export OMP_NUM_THREADS=1

# =========================
# 2) MCCL / NCCL：稳定优先
# =========================
unset MCCL_SOCKET_IFNAME
unset MCCL_NET_GDR_LEVEL
unset MCCL_MAX_NCHANNELS
unset MCCL_P2P_LEVEL
unset MCCL_LIMIT_RING_LL_THREADTHRESHOLDS
unset FORCE_ACTIVATE_WAIT
unset SET_DEVICE_NUMA_PREFERRED

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH

# =========================
# 3) 从平台环境变量读取多节点信息
# =========================
# 平台注入优先，没给就退化到单节点
NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${POD_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# 每个节点多少卡
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export WANDB_API_KEY=748830e9b9acdf804bb0baad0eb82e6ca2592354
torchrun --nproc_per_node=8 --master_port 29501 -m poregpt.tokenizers.vqe_tokenizer.vqe_train --config config.yaml 2>&1 | tee run.log
