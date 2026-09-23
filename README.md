# sglang-ascend-glm53-flash-serving

**GLM-5.3-Flash (glm5_next — hybrid KDA linear attention + DSA sparse attention, 288-expert MoE, 320B/18B) production inference on Ascend 910B with SGLang** — the full chain from a zero-upstream starting point: glm5_next existed in **no Ascend image and no upstream main** when this work started (2026-08-30). This repo packages the port, the quantization battle, the serving bring-up, and the converged production config as an agent-readable skill.

Measured (4 × Atlas 800T A2, 32 × 910B 64GB, RoCE, 2026-08/09): TP32 no-EP **36.4 tok/s single-stream decode, 4,327 tok/s on 80K-token prefill (18s)**, KV pool 680K tokens, radix cache enabled; tuned W4A8 TP16 + EAGLE D6/S5 reaches **36.5 tok/s single-stream at 94ms/step** (1.9× over no-spec).

## This repo is agent-readable

The entry point is [`SKILL.md`](SKILL.md) — structured as a skill package (fact card → 13 load-bearing conclusions → triage table → deep references):

- **Fact card**: model facts, upstream support matrix, converged topology, throughput
- **13 core conclusions** — each a measured failure mode, not a guess (207001=OOM misreports, KV pool walls, NoPE guards, fake quantization, async-stack misattribution)
- **Triage table**: symptom → first check
- `scripts/`, `templates/`, `references/` follow skill-package conventions

## Layout

```
SKILL.md                              agent entry: fact card + conclusions + triage
scripts/glm5_next_port_builder.py     port overlay builder (anchored patches + syntax gates)
scripts/precompile_triton_npu.py      single-process triton precompile (kills rank cache races)
scripts/verify_quant_output.py        safetensors header-only dtype audit (EXIT=0 ≠ compressed)
scripts/unfuse_equiv_probe.py         fused→unfused expert swap equivalence probe (4 discriminators)
scripts/e2e_longctx_battery.py        long-context E2E battery (needle/arithmetic/coherence)
scripts/bench_decode.py               SSE step-time probe (median gap = step time)
templates/smoke_tp16_dummy.sh         TP16 dummy-weights smoke (full-stack validation, no weights)
templates/serve_real_tp16.sh          real-weights launcher (smoke-proven flag set)
templates/quant_naive_w8a8_v5.py      llm_ptq pure-CPU recipe (⚠ skips fused experts — history)
templates/quant_experts_w4a8.py       v8 unfuse + experts-only W4A8 (self-checking, gated)
templates/glm53_flash_w8a8.yaml       modelslim_v1 declarative yaml (quantizes fused experts)
Dockerfile                            port image build (daily base + overlay COPY)
references/                           deep docs: port / smoke / bring-up / kpool / triton /
                                      KV-pool / quant battles / MTP crash chain / production
```

## The one-paragraph version

Ascend daily images ship NPU kernels *before* model code — the full KDA kernel suite was already in `main-cann9.0.0-910b` with zero glm5_next model files, so the port is a **Python-level overlay** (8 whole-file copies from the lmsysorg CUDA fork + anchored patches to 7 shared files + appended symbols), not a kernel port. The one hard gap — the kpool DSA indexer — closed by semantic mapping (native fused topk + interface-identical torch logits + exact-rounding torch reimplementations), validated on CPU before any NPU attempt. Serving bring-up then peeled ~20 distinct failure modes, the recurring classes being: NoPE (`qk_rope_head_dim=0`) guards resurfacing in every newly-lit attention path; the NPU MLA KV pool ignoring both `--context-length` and `--mem-fraction-static` (bound it with `--max-total-tokens` — its overflow misreports as `207001 "Binary get function failed"`); CANN async stacks blaming bystander ops (the peer node holds the real traceback); and triton JIT/cache races on first request. Quantization required defeating msmodelslim's silent skip of fused-MoE experts (pattern-matched yaml or unfuse-to-ModuleList), verified by header-only dtype audits. Production converged on TP32 no-EP (EP's deepep LL mode kills prefill 11×), 680K-token KV pool below the three memory walls, and radix cache via `--mamba-radix-cache-strategy extra_buffer` — without which long-context agent workloads collapse into a prefill-avalanche death spiral.

## Requirements

- 2-4 × 910B nodes (aarch64 — pull with `--platform linux/arm64`), RoCE for multi-node
- `quay.io/ascend/sglang:main-cann9.0.0-910b` (daily; KDA kernels) + transformers 5.16.1 (glm5_next = PR #48342)
- `lmsysorg/sglang:glm-5.3-flash` (CUDA fork — port source tree)
- GLM-5.3-Flash **BF16** weights (642GB; the FP8 main repo is N-card-only, 910B has no FP8) → W8A8 via the yaml (296GB) or experts-only W4A8
- msmodelslim 26.1.0 for quantization

## License

Apache-2.0. The port derives from SGLang's Apache-2.0 sources: `lmsysorg/sglang:glm-5.3-flash` overlaid onto `quay.io/ascend/sglang:main-cann9.0.0-910b`.
