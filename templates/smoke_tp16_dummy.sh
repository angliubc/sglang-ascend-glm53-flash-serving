#!/bin/bash
# TP16 dummy-weights smoke test for a ported model across two 8-NPU nodes
# (rank0 = this node, rank1 = the other node; run on BOTH with node-rank arg).
# Validates the FULL stack END-TO-END (config parse -> model init -> attention
# backend -> forward -> sampler -> decode -> HTTP 200) WITHOUT real weights:
# only config/tokenizer files needed in the model dir. Full-size BF16 dummy
# needs TP16 (642GB/8 > 64GB HBM per card).
#
# PROVEN 2026-08-31 on glm5_next / GLM-5.3-Flash (node3+node4, 38.5GB/card dummy):
# short (1-64 tok) and 809-token prefills both generated 32 tokens, HTTP 200,
# 30-35s/request (~3.5s/token eager decode). Dummy weights emit garbage tokens
# by design ("ookhiookhi...") — quality verdict needs real/quantized weights.
#
# Usage: smoke_tp16_dummy.sh <node-rank>    (0 on rank0 host, 1 on rank1 host)
# Every engine flag below is LOAD-BEARING (SKILL.md §5d has the failure each
# one prevents — do not trim):
#   --privileged --security-opt label=disable   NPU visibility (else is_available()=False)
#   --page-size 64                              DSA KV pool assert (memory_pool.py)
#   --disable-radix-cache                       hybrid models (MambaComponent page_size=1 assert)
#   --skip-server-warmup                        VLM warmup auto-sends an image -> vision-tower
#                                                inductor crash / 8-dim processor; curl text manually
#   --max-total-tokens 16384                    CRITICAL: this NPU MLA KV pool path IGNORES both
#                                                --context-length and --mem-fraction-static and eats
#                                                ~92% of post-weight avail (20GB+), leaving <1.5GB
#                                                for prefill transients -> OOM misreported as
#                                                aclnnBatchMatMul 207001 "Binary get function failed"
#   --watchdog-timeout 900                      first request JITs remaining triton kernels (~5 min);
#                                                default 300 kills a healthy first run mid-sampler
#   --disable-cuda-graph                        decode graph replay hang (kpool pool empty at capture,
#                                                populated at replay). REMOVE once capture-safety is
#                                                fixed — decode is ~10x slower without graphs.
#   TRITON_CACHE_DIR on a mounted path          else every container restart re-JITs everything;
#                                                precompile once per node SINGLE-PROCESS before serving
#                                                (scripts/precompile_triton_npu.py) — N ranks first-
#                                                executing the same @triton.autotune kernel race the
#                                                shared cache.

set -e
RANK=${1:?node-rank required}
IMG=ascend-sglang-glm53:v1-arm
MODEL=/model/GLM-5.3-Flash-BF16
RANK0_IP=${RANK0_IP:?set RANK0_IP=<rank0 node ip>}
DIST_PORT=6000
PORT=8100
# Optional fast-iteration overlay: pre-extract the image's sglang tree once
# (docker run --rm --entrypoint tar $IMG -cf - -C /sgl-workspace/sglang/python sglang \
#   | tar xf - -C /data/models/sglang_full/), rsync port/sglang/ over it (NO --delete —
#   the port tree is a partial overlay; --delete wipes the full base), md5-verify changed
#   files on both ends, then set OVERLAY=/data/models/sglang_full/sglang to skip
#   image rebuild+ship (~3 min/iteration vs ~12 min).
OVERLAY=${OVERLAY:-}
# Optional bind-mount over site-packages files the overlay can't reach (e.g. a
# patched transformers image processor): PATCH_DIR with files at their relative paths.
PATCH_DIR=${PATCH_DIR:-}

EXTRA_MOUNT=()
[ -n "$OVERLAY" ] && EXTRA_MOUNT+=(-v "$OVERLAY:/sgl-workspace/sglang/python/sglang")
[ -n "$PATCH_DIR" ] && EXTRA_MOUNT+=(-v "$PATCH_DIR:/patch" )

docker rm -f glm53-smoke 2>/dev/null || true
mkdir -p /data/models/triton_cache

docker run -d --name glm53-smoke \
  --privileged --security-opt label=disable \
  --network host \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /var/queue_schedule:/var/queue_schedule \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /data/models:/model \
  -e TRITON_CACHE_DIR=/data/models/triton_cache \
  -v /data/models/triton_cache:/data/models/triton_cache \
  "${EXTRA_MOUNT[@]}" \
  --shm-size=500g \
  $IMG \
  bash -c "
python3 -m sglang.launch_server \
  --model-path $MODEL \
  --load-format dummy \
  --trust-remote-code \
  --tp 16 --nnodes 2 --node-rank $RANK --dist-init-addr $RANK0_IP:$DIST_PORT \
  --page-size 64 \
  --context-length 8192 --max-running-requests 8 \
  --max-total-tokens 16384 \
  --mem-fraction-static 0.65 \
  --watchdog-timeout 900 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --skip-server-warmup \
  --host 0.0.0.0 --port $PORT \
  2>&1 | tee /model/glm53_smoke_rank$RANK.log
"
echo "smoke rank $RANK up${OVERLAY:+ (overlay-mounted sglang)}; follow: docker logs -f glm53-smoke"
echo "readiness: curl http://$RANK0_IP:$PORT/health"
echo "generate:  curl http://$RANK0_IP:$PORT/v1/completions -H 'Content-Type: application/json' \\"
echo "  -d '{\"model\": \"$MODEL\", \"prompt\": \"The capital city of France is\", \"max_tokens\": 16, \"temperature\": 0}'"
