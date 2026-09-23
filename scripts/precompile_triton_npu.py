#!/usr/bin/env python3
"""Single-process precompile of serving-path triton kernels (Ascend NPU).

WHY: when an SGLang scheduler (TP ranks) first executes a @triton.autotune
kernel, ALL ranks compile it simultaneously into the SAME TRITON_CACHE_DIR.
Concurrent cache writes can race -> corrupt binaries -> the kernel runs,
then UNRELATED later GEMMs fail with aclnnBatchMatMul 207001 "Binary get
function by entry Failed" (or hang), always on a subset of ranks/one node.
Pre-compiling in ONE process fills the cache cleanly; all ranks then hit it.

USAGE (in a privileged throwaway container that mounts BOTH the sglang
overlay and the persistent triton cache):

  docker run -d --name preprobe --privileged --security-opt label=disable \
    --device /dev/davinci0 --device /dev/davinci_manager \
    --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /data/models/sglang_full/sglang:/sgl-workspace/sglang/python/sglang \
    -v /data/models/scripts:/probe \
    -e TRITON_CACHE_DIR=/data/models/triton_cache \
    -v /data/models/triton_cache:/data/models/triton_cache \
    ascend-sglang-glm53:v1-arm sleep 3600
  # 0) NUKE any cache written by a raced multi-rank run first:
  rm -rf /data/models/triton_cache/*
  # 1) single-process precompile on EVERY node (node-local /data is not shared):
  docker exec preprobe python3 /probe/precompile_triton_npu.py

NOTE: autotune keys are shape-dependent — cover the realistic shape buckets
you expect at serving time (tiny prefill, mid, full chunked-prefill chunk),
or a novel shape at runtime will compile again (re-introducing the race).
"""
import sys
import time

import torch

sys.path.insert(0, "/sgl-workspace/sglang/python")

dev = "npu"


def step(name, fn):
    t0 = time.time()
    out = fn()
    torch.npu.synchronize()
    print(f"precompile {name}: OK {time.time()-t0:.1f}s", flush=True)
    return out


def precompile_kda_prefill(T: int, HK: int = 4, DK: int = 128, DV: int = 128):
    """KDA (gated delta) linear-attention PREFILL kernels used by
    ascend_kda_backend.extend — the ones first executed at the first real
    request after graph capture (decode graphs never touch them)."""
    from sglang.kernels.ops.attention.fla.kda import chunk_kda_scaled_dot_kkt_fwd
    from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
    from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
    from sgl_kernel_npu.fla.utils import prepare_chunk_indices

    B = 1
    q = torch.randn(B, T, HK, DK, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, HK, DK, dtype=torch.bfloat16, device=dev)
    g = -torch.rand(B, T, HK, dtype=torch.float32, device=dev)  # log decay
    beta = torch.rand(B, T, HK, dtype=torch.bfloat16, device=dev)
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)

    step(f"l2norm T={T}", lambda: l2norm_fwd(q.contiguous()))
    ci = step(f"prepare_chunk_indices T={T}", lambda: prepare_chunk_indices(cu, 64))
    step(f"chunk_local_cumsum T={T}", lambda: chunk_local_cumsum(
        g.contiguous(), chunk_size=64, cu_seqlens=cu, chunk_indices=ci))
    step(f"chunk_kda_scaled_dot_kkt_fwd T={T}", lambda: chunk_kda_scaled_dot_kkt_fwd(
        q=q, k=k, gk=g, beta=beta, scale=DK ** -0.5,
        cu_seqlens=cu, output_dtype=torch.float32))


if __name__ == "__main__":
    # Shape buckets: tiny prefill (few tokens), mid, full chunked-prefill chunk.
    for T in (8, 512, 8192):
        precompile_kda_prefill(T)
    print("PRECOMPILE DONE", flush=True)
