#!/bin/bash
# recreate-w8a8moe-700k.sh — .33/.34 重建 glm53-w8a8moe-dp2 容器, KV 池 500K→700K (sparse-layer 池补丁)
# 用法: NODE_RANK 环境变量或 $1 (0=.33, 1=.34); 由外层编排保证 .34 先 .33 后
set -e
# 用法: recreate-w8a8moe-700k.sh <rank 0|1> [context_len] [max_total_tokens]
#   默认拉满档: ctx=1048576, tokens=2000000, mem-frac 0.95 (见 serve 脚本头注释)
#   回滚: recreate-w8a8moe-700k.sh 0 500000 500000
NODE_RANK=${1:-0}
CTX=${2:-1048576}
TOKENS=${3:-2000000}
NODE_IP=190.168.24.33
[ "$NODE_RANK" = "1" ] && NODE_IP=190.168.24.34

docker rm -f glm53-w8a8moe-dp2 2>/dev/null || true
sleep 20

docker run -d --privileged --name glm53-w8a8moe-dp2 \
  --shm-size=32g --net host \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /data/models:/model \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /data/models/triton_cache:/data/models/triton_cache \
  -v /data/models/mtp_experiment_0907/sglang:/sgl-workspace/sglang/python/sglang \
  -v /data/models/patch/image_processing_glm5_next.py:/usr/local/python3.11.15/lib/python3.11/site-packages/transformers/models/glm5_next/image_processing_glm5_next.py \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/local/sbin:/usr/local/sbin \
  ascend-sglang-glm53:v1-arm sleep infinity

sleep 5
docker exec -d glm53-w8a8moe-dp2 bash -c "bash /model/scripts/serve-w8a8moe-dp2-node.sh $NODE_RANK $CTX $TOKENS > /serve.log 2>&1"
echo "node $NODE_RANK ($NODE_IP) launched, log: docker logs -f glm53-w8a8moe-dp2"
