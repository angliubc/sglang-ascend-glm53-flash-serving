# GLM-5.3-Flash (glm5_next) serving image for Ascend 910B — port overlay baked in.
#
# Build inputs (assembled by scripts/glm5_next_port_builder.py):
#   port/sglang/   the overlay tree (8 whole-file copies from the lmsysorg
#                  glm-5.3-flash fork + anchored patches to 7 daily-image files)
#
# COPY-only layers => cross-arch docker build works WITHOUT qemu (safe to build
# on an x86 host; the result still only RUNS on aarch64/910B).
# ⚠️ Pull the base with --platform linux/arm64 — the multi-arch registry serves
#    the PULLING host's arch, and an amd64 build is a 17GB dead end (measured).
FROM quay.io/ascend/sglang:main-cann9.0.0-910b

# SGLang tree with the glm5_next port overlay (model + configs + attention
# registry wiring + hybrid-arch plumbing + MHC communicators + kpool support).
COPY port/sglang/ /sgl-workspace/sglang/python/sglang/

# transformers 5.16.1 + aarch64 tokenizers 0.23.1 wheels must be installed on
# top (glm5_next lands in transformers at 5.16.1 — PR #48342; the daily image
# ships an older one). Air-gapped: pip install --no-index --no-deps <wheels>.
# RUN pip install --no-index --no-deps /wheels/transformers-5.16.1-py3-none-any.whl ...

# Persist triton JIT cache at a predictable mount point (bind a host dir here,
# or every restart re-JITs the KDA chunk kernels, minutes each on aarch64).
ENV TRITON_CACHE_DIR=/sgl-triton-cache
VOLUME /sgl-triton-cache

LABEL org.opencontainers.image.title="sglang-ascend-glm53-flash-serving" \
      org.opencontainers.image.description="GLM-5.3-Flash (glm5_next: hybrid KDA+DSA, 288-expert MoE) on Ascend 910B: port overlay + W8A8/W4A8 quantized serving" \
      org.opencontainers.image.licenses="Apache-2.0"
