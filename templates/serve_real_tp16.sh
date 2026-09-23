#!/bin/bash
# GLM-5.3-Flash BF16 REAL-WEIGHTS TP16 serving launcher (2-node 910B, 0831).
# Recovered from the proven glm53-smoke dummy run (docker inspect: 11 devices, 8 mounts,
# env-sourcing entrypoint) with exactly three changes: no --load-format dummy,
# port 8200 (user-specified external endpoint on node3), fresh container name
# (glm53-serve — old smoke containers left untouched).
# Deploy to BOTH nodes as /root/glm53-serve-run.sh; start rank1 (node4) first, then rank0 (node3).
#
# Phasing discipline (one variable at a time):
#   Phase 1 = this script as-is: real weights + smoke-proven conservative flags
#             (disable-cuda-graph, ctx 8192, bs 8, mem-frac 0.75 — proven at the same
#             40GB/card weight footprint; note §5d: NPU MLA KV pool ignores mem-frac on
#             this arch — --max-total-tokens is the real bound).
#   Phase 2 = MTP/NEXTN speculative args (user hard-requires MTP for production
#             acceptance) — only AFTER phase-1 quality gate passes.
#   Phase 3 = perf: cuda-graph, larger ctx/bs — decode-graph capture semantics is the
#             known open bug (references/glm53-decode-graph-hang-20260831.md).
#
# Prereq: weights verified-complete on BOTH nodes (node-local /data is NOT shared):
#   ls <dir> | wc -l  AND  du -sb <dir>  must match the source — smoke-era STUB dirs
#   (config-only, 24MB) fool existence checks; du everything.
# Usage: glm53-serve-run.sh <node-rank 0|1>
set -e
RANK=${1:?usage: glm53-serve-run.sh <0|1>}

docker inspect glm53-serve >/dev/null 2>&1 && { echo "ABORT: glm53-serve already exists"; exit 1; }
if [ "$RANK" = "0" ]; then
  ss -lnt | grep -q ":8200 " && { echo "ABORT: port 8200 busy"; exit 1; }
  ss -lnt | grep -q ":6000 " && { echo "ABORT: dist-init port 6000 busy"; exit 1; }
fi
ls /data/models/GLM-5.3-Flash-BF16/model-00120-of-00120.safetensors >/dev/null 2>&1 || { echo "ABORT: weights incomplete on $(hostname)"; exit 1; }

docker run -d --name glm53-serve \
  --privileged --network host --ipc shareable --shm-size 512g \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /data/models/sglang_full/sglang:/sgl-workspace/sglang/python/sglang \
  -v /var/queue_schedule:/var/queue_schedule \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /data/models:/model \
  -v /common:/common \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /data/models/patch/image_processing_glm5_next.py:/usr/local/python3.11.15/lib/python3.11/site-packages/transformers/models/glm5_next/image_processing_glm5_next.py \
  -v /data/models/triton_cache:/data/models/triton_cache \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/sbin:/usr/local/sbin \
  ascend-sglang-glm53:v1-arm \
  python3 -m sglang.launch_server --model-path /model/GLM-5.3-Flash-BF16 --trust-remote-code \
  --tp 16 --nnodes 2 --node-rank $RANK --dist-init-addr ${RANK0_IP:-<rank0-ip>}:6000 \
  --mem-fraction-static 0.75 --max-total-tokens 16384 --watchdog-timeout 900 \
  --disable-cuda-graph --page-size 64 --disable-radix-cache --skip-server-warmup \
  --disable-overlap-schedule --context-length 8192 --max-running-requests 8 \
  --host 0.0.0.0 --port 8200

echo "launched rank $RANK on $(hostname); logs: docker logs -f glm53-serve"
