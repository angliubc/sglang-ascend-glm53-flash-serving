# GLM-5.3-Flash NPU smoke phase — error→root-cause→fix catalog (2026-08-30 evening)

Context: ported overlay (see glm53-flash-port-20260830.md) deployed to node3/node4 as
`ascend-sglang-glm53:v1-arm` (daily main-cann9.0.0-910b arm64 base + transformers 5.16.1 +
tokenizers aarch64). Smoke = TP16 dummy weights, `--load-format dummy`, dist-init node3:6000.
Iteration loop: build_port.py → rsync overlay (no --delete) → relaunch both ranks → ~3-6 min read logs.

## Cycle 0 — container NPU invisible
- Error: `RuntimeError: torch_npu detected, but NPU device is not available or visible` /
  `Can't get ascend_hal device count`, `torch.npu.is_available()=False`.
- Red herring: same failure in BOTH daily and known-good v0.5.16 images → NOT an image problem.
- Root cause: launch flags. Production `sglang-tp32` containers run `--privileged
  --security-opt label=disable` (check `docker inspect --format '{{.HostConfig.Privileged}}'`).
- Fix: add both flags to the launcher. All NPU containers on 910B need them.

## Cycle 1 — rope_theta AttributeError (config-class swap)
- Error: `AttributeError: 'Glm5NextTextConfig' object has no attribute 'rope_theta'` at
  models/glm5_next.py:621 (Glm5NextDecoderLayer init).
- Root cause chain:
  - sglang's model code expects sglang's OWN configs (`sglang/srt/configs/glm5_next.py`,
    Glm5NextTextConfig with `rope_theta: float = 800000.0` etc.).
  - lmsysorg's fork makes that happen via `_CONFIG_REGISTRY` + `AutoConfig.register` in
    `utils/hf_transformers/common.py` — works only because their bundled transformers fork
    deliberately omits glm5_next from CONFIG_MAPPING.
  - transformers 5.16.1 release natively knows glm5_next → register fails "already used"
    (caught+warned) → AutoConfig returns the NATIVE class (NoPE: qk_rope_head_dim=0 REQUIRED,
    no rope_theta field at all — position info lives in the DSA indexer).
  - Verified both sides: native class has no rope_theta; lmsysorg's own fork class also
    doesn't (their sglang class is the source of it).
- Fix: patch daily's `sglang/srt/utils/hf_transformers/config.py` — add
  `_try_load_glm5_next_config(model, revision, **kwargs)` mirroring the existing
  `_try_load_longcat_config` precedent: `PretrainedConfig.get_config_dict` → if
  model_type == "glm5_next" → `Glm5NextConfig.from_pretrained(...)` (sglang's class; its
  __init__ coerces text_config/vision_config dicts into sglang sub-configs). Call it in
  `parse()` between the longcat try and AutoConfig.
- Verified on amd64 container (CPU): parser returns sglang classes, rope_theta=800000.0,
  45 layers, 288 experts, hc_mult 4.

## Cycle 1b — build script chained-patch bug (silent!)
- Symptom: patch reported applied but the port file lacked it; smoke crashed identically.
- Root cause: build script's `patch()` always read the PRISTINE daily tree and wrote the
  full file to port/ — consecutive patches to the same file overwrote each other. 3 files
  had lost patches (model_config SarvamMLA-or, dp_attention classmethod, the new def above).
- Fix: patch() reads from port tree if a previous step already produced it (reset port/ at
  script start for idempotency). Same for append_defs().
- Detection failure: `python3 build.py | grep -c "ANCHOR FAIL"` printed the count but the
  `&&` chain continued (grep exits 0 on match). Deployed a broken overlay twice. Gate with
  `if … | grep -q "ANCHOR FAIL"; then abort`.

## Cycle 2 — sharded_weight_loader signature skew
- Error: `TypeError: sharded_weight_loader() takes 1 positional argument but 2 were given`
  (glm5_next.py:460, Glm5NextLinearAttention head-shard rank getter).
- Root cause: lms fork adds optional `tp_rank_getter` (defaults get_parallel().attn_tp_rank);
  daily has only `shard_axis`.
- Fix: patch `model_loader/weight_utils.py` signature + body (`tp_rank = tp_rank_getter() if
  tp_rank_getter is not None else get_parallel().attn_tp_rank`).
- Systematic follow-up: AST scan of ALL symbols imported by ported files vs both trees →
  7 skews; 4 harmless (daily supersets: ColumnParallelBatchedLinear, MergedColumnParallel-
  RepeatedLinear, FusedMoE, RadixLinearAttention all just add optional params; GlmOcrVisionModel
  skew harmless because Glm5NextVisionModel overrides __init__ completely, never calls super).

## Cycle 3 — init_context / AttentionInputs MHC params
- Error: `TypeError: AttnTpContext.init_context() takes 3 positional arguments but 4 were given`
  (glm5_next.py:1296 passes text_config.mhc as 3rd arg).
- Root cause: lms adds `is_mhc=False` (MHC pre-gather makes DSA eligible for input_scattered:
  `and (is_mhc or not is_dsa)`) and `*, is_pre_gathered` on AttentionInputs (MHC pre-gathered
  DSA path must not re-gather in fetch_qkv_latent/fetch_hidden_states).
- Fix: patch daily's communicator.py — extend init_context signature + condition (KEEP daily's
  accessor `get_parallel().enable_attn_tp_input_scattered`, NOT the fork's
  `get_parallel().config.…`); add is_pre_gathered kwarg + attr + 2 guard conditions.
- Anchor lesson: transcribe anchors from the ACTUAL file read, not from memory of a similar
  file (cost 2 anchor failures: `self.hidden_states_local` first-line, `residual.dim()==3`
  not 2).

## Cycle 4 — kv_b_proj on KDA layers
- Error: `AttributeError: 'Glm5NextLinearAttention' object has no attribute 'kv_b_proj'` in
  `DeepseekV2WeightLoaderMixin.post_load_weights` (deepseek_weight_loader.py:542).
- Root cause: daily's mixin iterates kv_b_proj layers unguarded; glm5_next hybrid KDA layers
  have no kv_b_proj. lms has `if not hasattr(self_attn, "kv_b_proj"): continue`.
- Fix: port that guard into daily's mixin.
- AFTER this: **model fully builds** — `Load weight end. elapsed=27.65s,
  type=Glm5NextForConditionalGeneration, avail mem=22.10 GB, mem usage=38.46 GB` on all 16 ranks.

## Cycle 5 — DSA KV pool page size
- Error: `assert self.page_size == 64` (memory_pool.py:4544, _build_dsa_kv_pool).
- Fix: launch flag `--page-size 64` (not a port bug).

## Cycle 6 — mHC kernel parameterization mismatch
- Error: `NameError: name 'deep_gemm' is not defined` (deep_gemm_wrapper/entrypoint tf32 path)
  ← lms mhc dispatch defaulted to tilelang/deep_gemm path on NPU where deep_gemm isn't imported.
- Investigation (probe script in privileged container, real sizes):
  - `import custom_ops` (Huawei CANN pkg, site-packages) registers torch.ops.custom.*;
    importing sglang's mhc module does NOT.
  - npu_hc_pre constraints (from error strings): x 4-D; hc_mult hardcoded 4; hidden d ∈
    {4096, 7168}; hc_fn must be (24, hc_dim) = dsv4 mixing form ((2+hc)*hc rows) — glm5_next
    uses head form (4, hc_dim) → kernel NOT usable without algebraic remapping.
  - GLM-5.3-Flash text config: hidden 4096, hc_mult 4, 45 layers — sizes match, params don't.
- Fix: route NPU → torch fallback in the dispatchers: `_mhc_pre_dispatch`:
  `if _NPU or not envs.SGLANG_OPT_USE_TILELANG_MHC_PRE…: _mhc_pre_torch(...)`; same for
  `_mhc_post_dispatch`. `_NPU = hasattr(torch, "npu") and torch.npu.is_available()`.
- Post-deploy optimization idea: dsv4↔glm5_next mHC parameterization mapping for the fused kernel.

## Cycle 7 — hybrid forward_metadata + NoPE triple guard
- Errors in sequence:
  1. `'MHCLayerCommunicator' object has no attribute 'maybe_prefetch_next_full_attention_kv'`
     → lms base-class no-op stub; append to daily's LayerCommunicator before postprocess_layer.
  2. `'NoneType' object has no attribute 'seq_lens_cpu_int'` (dsa_npu_indexer.forward_npu:35
     reading `get_attn_backend().forward_metadata`) → base AttentionBackend has CLASS-LEVEL
     `forward_metadata = None`; HybridLinearAttnBackend wrapper never sets it; the DSA child
     backend owns the real metadata. Fix: property+setter on the hybrid delegating to
     `self.full_attn_backend.forward_metadata`.
  3. `'NoneType' object has no attribute 'is_neox_style'` (mla_npu forward_dsa_prepare_npu:380)
     → NoPE: rotary_emb None. Guard `m.rotary_emb is not None and m.rotary_emb.is_neox_style`
     + wrap the layer-0 cache select and rotary call.
  4. `assert qk_rope_dim > 0` (fused_split_qk_norm) → route NoPE to the unfused split path:
     add `and m.qk_rope_head_dim > 0` to the fused-path condition.
  5. `cannot reshape tensor of 0 elements into shape [8, -1, 0]` (rope_variant.forward_npu:511)
     → INDEXER rotary with rotary_dim=0. This exposed the real remaining gap (below).

## The remaining hard gap: KPool indexer
- glm5_next's DSA indexer is the KPOOL variant: config `index_kpool: 4`,
  `index_kpool_compress: true`, `index_kpool_always_select_tail: true`,
  `indexer_types: all "full"`, `index_topk: 2048`, `indexer_rope_interleave: true`.
- lms fork: `dsa_indexer_kpool.py` (1767 lines, IndexerKPool) selected when
  `get_dsa_index_kpool(config) > 1`, `skip_rope` plumbed for NoPE. CUDA-only deps in MAIN
  paths (not optional branches): `_topk_from_kpool_logits` → triton
  `topk_from_pooled_history_logits` (unconditional); decode logits via
  `deep_gemm.fp8_paged_mqa_logits`; `_compress_write*` triton; `kpool_fp8_index.py` = 1720
  lines of triton. File has `if is_npu(): import custom_ops` at top (NPU work STARTED, not
  finished — zero custom_ops usage inside).
- daily: no kpool anywhere. `dsa_backend.py` HAS `get_indexer_metadata` (line 3387) ✓.
  NPU plain-indexer path exists (dsa_npu_indexer + paged_mqa_logits NPU kernels) but plain
  ≠ kpool semantics and checkpoints carry kpool weights (compress_gate etc.) — can't force plain.
- Huawei custom_ops ships `npu_indexer_compress_epilog`, `npu_quant_lightning_indexer`,
  `npu_gather_selection_kv_cache` — no python consumer in daily; z.ai's Ascend kpool support
  is in an unreleased internal branch.
- Port plan (next session): torch reimplementations of (a) topk_from_pooled_history_logits,
  (b) compress-write (decode + extend), (c) pooled paged MQA logits (or reuse NPU paged_mqa
  kernel over pooled layout if its interface allows); wire IndexerKPool into daily's
  deepseek_v2.py Indexer construction (mirror lms indexer_cls selection + skip_rope);
  validate numerics on CPU (no NPU needed) before overlay deployment.
- Dependencies to carry over: kpool needs `DUAL_STREAM_TOKEN_THRESHOLD` (const, from lms
  dsa_indexer) and `cp_zigzag_full_plan_rows` (fn, from lms dsa/utils) appended to daily files.

## Misc facts established this session
- quay arm64 daily image = 6.3GB compressed / 16.5GB unpacked (amd64 17GB); manifest digest
  sha256:c2e5ccf5…, 19 layers.
- Concurrent pulling: 6-worker layer downloads ~2.8MB/s; single 4.1GB layer stalled 160KB/s →
  10-segment Range download 5.8MB/s. Assemble LEGACY docker-save tar (OCI layout rejected by
  classic daemon: "blobs/json" error).
- the CUDA host docker disk FULL (no space left) — don't use it as a pull relay without cleanup.
- Local→910B direct scp ~79MB/s; docker save|ssh load of 16.6GB image ≈ 7-10 min/node parallel.
- DOCKER_BUILDKIT=0 required when FROM references a local-only tag (buildkit resolves bare
  names against registry mirrors).
- DSA NoPE math: GLM-5.3-Flash DSA has NO rope on attention at all (qk_rope_head_dim=0,
  NoPE); position info entirely via the indexer. transformers' Glm5NextTextConfig is a
  @strict dataclass-style config with class-level defaults and validate_architecture raising
  if qk_rope_head_dim > 0.
- mHC tensor protocol (lms hc_pre/hc_post): x=[s, n·hidden] bf16 → (layer_input [s,hidden],
  h_res [s,n·n] fp32, h_post [s,n] fp32, norm_fused); dsv4's NPU kernel protocol differs
  (4-D x, 24-row mixing matrices).
