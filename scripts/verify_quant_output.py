#!/usr/bin/env python3
"""Verify quantization OUTPUT dtype coverage from safetensors headers (no weight load).

EXIT=0 / .quant_done markers do NOT prove a usable quantized model: msmodelslim
llm_ptq wraps only nn.Linear modules, so fused MoE experts (raw 3-D Parameters,
e.g. Glm5NextTextExperts gate_up_proj [E,2*I,H]) silently stay BF16. The
GLM-5.3-Flash W8A8 artifact (2026-08-31) measured 529 int8 linears (~7GB) +
608.8GB BF16 experts = 620GB (~578GiB) ≈ the original BF16 size — undeployable
on 8x64GB. This script reads ONLY each file's 8-byte little-endian header-length
prefix + JSON header (seconds even for 600GB files; no torch needed) and prints
per-module-category byte volume by dtype, plus a BF16-bulk verdict
(exit 1 when BF16 > 40% of tensor bytes — usable as a gate in armed launchers).

Usage:
  python3 verify_quant_output.py <file-or-dir> [more files/dirs...]
Remote (no need to copy the script):
  ssh root@host 'python3 - /data/models/GLM-5.3-Flash-W8A8/' < verify_quant_output.py
"""
import glob, json, math, os, struct, sys

DTYPE_SIZE = {"I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1,
              "F16": 2, "BF16": 2, "I16": 2,
              "F32": 4, "I32": 4, "U32": 4,
              "F64": 8, "I64": 8, "U64": 8}


def categorize(name):
    if ".mlp.experts." in name or ".experts." in name:
        return "routed_experts"
    if "shared_experts" in name:
        return "shared_experts"
    if "self_attn" in name or "attn" in name:
        return "attn"
    if "visual" in name or "vision" in name or "merger" in name:
        return "vision"
    if "lm_head" in name:
        return "lm_head"
    if "embed" in name:
        return "embedding"
    return "other(norm/router/etc)"


def main(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.safetensors")))
        else:
            files.append(p)
    if not files:
        sys.exit("no .safetensors files found in: %s" % " ".join(paths))

    agg = {}  # (category, dtype) -> bytes
    file_bytes = 0
    for path in files:
        file_bytes += os.path.getsize(path)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            b = math.prod(v["shape"]) * DTYPE_SIZE.get(v["dtype"], 2)
            key = (categorize(k), v["dtype"])
            agg[key] = agg.get(key, 0) + b

    total = sum(agg.values())
    int8 = sum(b for (c, d), b in agg.items() if d in ("I8", "U8"))
    bf16 = sum(b for (c, d), b in agg.items() if d in ("BF16", "F16"))

    print("%-22s %-12s %9s" % ("category", "dtype", "GB"))
    for (c, d), b in sorted(agg.items(), key=lambda x: -x[1]):
        if b < 1e6:  # hide scale/aux noise below 1MB
            continue
        print("%-22s %-12s %9.1f" % (c, d, b / 1e9))
    denom = max(total, 1)
    print("\ntensor bytes: %.1f GB | files: %.1f GB | int8 %.1f GB (%.1f%%) | bf16 %.1f GB (%.1f%%)"
          % (total / 1e9, file_bytes / 1e9, int8 / 1e9, 100 * int8 / denom,
             bf16 / 1e9, 100 * bf16 / denom))
    if bf16 > 0.4 * total:
        print("⚠ BF16 BULK — fused modules (MoE experts?) were NOT quantized; "
              "artifact ≈ source size. Route experts through modelslim_v1 "
              "declarative yaml instead of llm_ptq.")
        sys.exit(1)
    print("✅ quantized weights dominate — coverage looks real")
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
