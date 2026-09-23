#!/bin/bash
# serve-w8a8moe-dp2-node.sh — GLM-5.3-Flash W8A8-MoE DP2×TP16 双节点 (0923 sparse-layer 池补丁)
# 用法: bash serve-w8a8moe-dp2-node.sh <node_rank 0|1> [context_len] [max_total_tokens]
#   (容器内执行; 0=.33 master, 1=.34 worker)
#   默认: context_len=1048576 (模型原生 1M 上限), max_total_tokens=2000000 (显存拉满档)
#   回滚旧配置: 传 "500000 500000"; 0923 首版 700K 档: "700000 700000"
#
# 0923v2 显存拉满账本 (每卡, 700K 实测推算):
#   权重 21.71 + draft 0.85 + mamba 2.11 + 图 ~4.1 = 28.8 固定
#   池 per-token 13.64KB (主 12.41 + draft 1.23)
#   2M 池 ≈ 27.3GB → 总静态 ~56.1GB, 运行时余量 ~4.4GB
#   mem-fraction 0.95 (0.90 静态预算 57.6GB 会把池封顶在 ~14.7GB 增量, 必须提到 0.95)
#   若图捕获/首请求 OOM: 降 tokens 档位 1800000 (余 ~7.2GB) / 1600000 (余 ~9.9GB)
set -x
node_rank=$1
ctx=${2:-1048576}
tokens=${3:-2000000}
local_ip=190.168.24.33
if [ "$node_rank" = "1" ]; then local_ip=190.168.24.34; fi
nic_name=enp67s0f0np0

# --- env (照抄 0922 生产容器 inspect, 含 HCCL_DETERMINISTIC=true 算术修复) ---
export SGLANG_MIN_THINK_TOKENS=64
export TRITON_CACHE_DIR=/data/models/triton_cache
export TASK_QUEUE_ENABLE=1
export SGLANG_HICACHE_MAMBA_RATIO=3
export ATB_MATMUL_SHUFFLE_K_ENABLE=1
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=1
export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0
export ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE=0
export ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE=0
export ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=1
export ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=5
export ATB_COMPARE_TILING_EVERY_KERNEL=0
export ATB_SHARE_MEMORY_NAME_SUFFIX=
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/customize
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.0.0/opp
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export ASCEND_TOOLKIT_LATEST_HOME=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.0.0
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export TOOLCHAIN_HOME=/usr/local/Ascend/cann-9.0.0
export CMAKE_PREFIX_PATH=/usr/local/Ascend/cann-9.0.0/toolkit/tools/tikicpulib/lib/cmake:/usr/local/Ascend/cann-9.0.0/lib64/cmake
export ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_DETERMINISTIC=true
export LCCL_PARALLEL=0
export LCCL_DETERMINISTIC=0
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_IF_IP=$local_ip

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null

exec python3 -m sglang.launch_server \
  --model-path /model/GLM-5.3-Flash-W8A8-MoE-NPU \
  --served-model-name glm53-flash \
  --trust-remote-code \
  --tp 16 --dp 2 --enable-dp-attention --enable-dp-attention-local-control-broadcast \
  --nnodes 2 --node-rank $node_rank \
  --dist-init-addr 190.168.24.33:6379 \
  --quantization modelslim \
  --mem-fraction-static 0.95 \
  --context-length $ctx \
  --max-total-tokens $tokens \
  --max-running-requests 16 \
  --cuda-graph-max-bs-decode 16 \
  --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
  --max-mamba-cache-size 128 \
  --chunked-prefill-size 8192 \
  --watchdog-timeout 900 \
  --page-size 64 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --speculative-draft-model-path /model/mtp_experiment_0907/draft \
  --speculative-draft-model-quantization unquant \
  --mamba-radix-cache-strategy extra_buffer \
  --skip-server-warmup \
  --disable-overlap-schedule \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --host 0.0.0.0 --port 8078
