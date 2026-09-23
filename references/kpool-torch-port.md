# GLM-5.3-Flash kpool indexer: torch port + NPU wiring (2026-08-30 evening)

Continuation of the port session (2026-08-30 evening): working tree /tmp/glm53_port/ on the
workstation (build_port.py, patch segments >=20), decision log PREP.md. Data pipeline: BF16
100/120 shards downloaded to shared storage; the quant node's launcher polls for 120/120 to
auto-run msmodelslim W8A8.

## The gap (recap)
lms fork kpool (`dsa_indexer_kpool.py`, 1767 lines) hard-depends on triton/DeepGEMM; Ascend daily has zero kpool; Huawei custom_ops `npu_indexer_compress_epilog` has no python consumer.

## Closure technique: semantic mapping, not kernel-by-kernel rewrite
All 9 triton kernel functions were semantically mapped first (read the CUDA `.cuh` / triton sources for exact semantics — especially rounding). Three reuse discoveries collapsed ~1 week of work into ~1 day:

1. `group_topk = topk // pool` — GLM-5.3-Flash (topk=2048, pool=4) → **512 ≤ 512** → routes to sglang's NATIVE fused moe kernel `fast_kpool_topk_transform_fused` (JIT CUDA `.cuh`, in-tree). No port needed for the topk transform. (Only group_topk > 512 needs `fast_topk_v2` + pure-torch expansion.)
2. daily dsv4 `indexer.py` already ships **`fp8_paged_mqa_logits_torch`** — pure-torch pooled-MQA-logits with an interface IDENTICAL to the kpool call site (`deep_gemm.fp8_paged_mqa_logits`, same 64×132 page layout). Drop-in replacement for the DeepGEMM decode logits.
3. `act_quant` (fp8 block quant) and `_hadamard_quantize_fp8` raise NameError on NPU → torch reimplementations replicating EXACT rounding semantics read off the triton source (compress = per-dim softmax mixing + Hadamard-128 rotation + fp8 quantize + dual-region cache write).

Implementation file written capture-safe (decode path free of host syncs).

## CPU numerical validation — run BEFORE any NPU attempt
13/13 tests passed against triton/CUDA semantics: topk set/tail/padding, compress math + cache layout, Hadamard orthogonality, gather readback, ragged logits.
Test-writing pitfalls hit (each cost a rerun):
- fp8 tolerance must be computed as RELATIVE error, not absolute
- `gather` results are non-contiguous → use `reshape`, not `view`
- page-table test data must be sized ≥ the actual max page index referenced

## NPU wiring bugs (each cost one smoke cycle)
1. **Branch not activating** — stack showed ordinary `dsa_npu_indexer`, not kpool. TWO root causes stacked:
   - (a) rsync overlay pushed `port/` CONTENTS into `.../sglang/sglang/` (nested dir) — correct source is `port/sglang/`. Silent stale overlay = yesterday's code. Fix: correct rsync + md5-verify all changed files both ends (5/5 MATCH) before firing smoke.
   - (b) `deepseek_v2.py` lacked the `get_dsa_index_kpool` import → kpool getter never resolved even with `index_kpool=4` in config (model passes text_config; field exists).
2. **IndexerKPool has no NPU path** — inherits only `MultiPlatformOp` → `forward_npu` → native → NotImplementedError. `DSANPUIndexerMixin.forward_npu` is NOT reusable (ordinary-indexer specific, rope-bearing → NoPE crash). Fix: `forward_npu = forward_cuda` alias on the class + disable dual-stream on NPU (set `DUAL_STREAM_TOKEN_THRESHOLD` to a huge value).
3. **Signature mismatch** — daily's NPU caller passes an 8-arg `forward_npu` form (`layer_scatter_modes`, `dynamic_scale`); kpool `forward_cuda` has the CUDA signature. Fix: adapter in the 8-arg form dropping the NPU-only args.

## Open at recall time (2026-08-30 20:19)
NoPE crash in the NPU FIA sparse-attention decode path: `ascend_backend.py:2260 forward_decode_graph` — `k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)` with `qk_rope_head_dim=0` (also `q_rope.view(-1,1,heads,0)` earlier in the block). Probe `probe_nope_fia.py` written to test FIA kernel acceptance of 0-dim rope empirically in a privileged node3 container (same probe pattern as mHC: `import custom_ops`, tiny tensors, read error strings).

## Progress marker at recall
TP16 dummy smoke (node3 rank0 + node4 rank1) now passes: server_args parse → config-class swap → 16-card model build (38.46GB/card) → KV pool (page 64) → **kpool branch ACTIVE** → graph capture stage. Remaining chain: FIA NoPE fix → capture completes → overnight W8A8 weights (auto) → real serving.
