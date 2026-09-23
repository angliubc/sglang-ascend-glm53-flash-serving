# GLM-5.3-Flash 长上下文 serve-time kpool topk — IndexerKPool NPU token级移植 (2026-09-07)

Closes the §5c "0904 caveat" (kpool indexer dead in serving → full attention) and the
2048-context ceiling of the MTP bring-up (`glm53-mtp-eagle-nextn-20260904.md` 坑④/⑦).
Result: 8100 服务 node1/node2 TP16, **context 2048→32768 with MTP retained**, three graph
families captured, E2E needle verified. Tree: `/data/models/mtp_experiment_0907/sglang`
(overlay-mounted; all patches have .bak). Boot order: worker (node2) first, master (node1)
+20s (per existing pair-restart rule).

## Decision: token-level approximation instead of full kpool port
Full kpool needs `get_indexer_metadata` (pooled page tables, pooled cache bookkeeping,
write plans) on the ascend backend — days. `kpool_torch_npu.py` HAS the scoring kernels
(`fp8_mqa_logits`, `fast_kpool_topk_transform_fused`) but they're pool-granularity /
gather-based (token-granularity at 32K context gathers [T,span,H,D] → 50GB+ — unusable).
Token-level alternative (hours): score per-token with the module's OWN indexer weights
(wq_b/wk/weights_proj/k_norm — identical in IndexerKPool and plain Indexer) via
`npu_lightning_indexer` (fused CANN, paged, bf16 keys + per-head weights, all modes
incl. verify qlen patterns — op-test validated 6e-4).
- ≤2048 windows: lightning returns all-visible → EXACT trained semantics.
- >2048: selection granularity differs (per-token top-2048 vs kpool 4-token blocks +
  compress gate). Always-select-tail is EMULATED (below) so the tail is never lost.
- kpool compress_ape/compress_gate weights simply unused. Scoring-order note: kpool's
  extra scale factors (n_heads^-0.5, softmax_scale, fp8 q_scale) are uniform per query →
  topk ORDER invariant → token selection unaffected by dropping them.

## The forward_npu design (dsa_indexer_kpool.py)
1. q = wq_b(q_lora) [T,32,128]; k = k_norm(wk(x)) [T,128] fp32; weights = weights_proj(x.float()) bf16.
2. ROPE: guard `if not self.skip_rope and getattr(self,"rotary_emb",None) is not None:`
   — GLM's kpool indexer is NoPE (skip_rope=True → rotary_emb NEVER CREATED, reading it
   = AttributeError at capture). Copy DSANPUIndexerMixin.forward_npu math for both
   neox/non-neox branches under the guard; no-rope path just views k → [T,1,128].
3. **Shadow bf16 index-K cache** (the core trick): per indexer instance
   `_npu_shadow_k = torch.zeros(pool.size + pool.page_size, 1, head_dim, bf16)`,
   allocated LAZILY on first forward (graph warmup runs eager → allocates before capture).
   Slot-order layout mirrors the main KV pool: write `shadow[out_cache_loc.long()] = k.bf16`
   (graph-safe scatter, same pattern as set_kv_buffer); read is a zero-copy VIEW
   `shadow.view(-1, page_size, 1, head_dim)` → [num_pages,64,1,128] PA_BSND, reusing the
   SAME block_tables as the main pool. fp8 packed cache write (SetKAndS triton) SKIPPED
   entirely — its only consumers were the dead CUDA path.
   Memory: ~140MB/layer at 550K tokens; 11 DSA layers → 1.55GB/card (fits the ~9GB
   post-capture headroom; avail after everything: 5.32GB).
4. lightning call verbatim from the plain mixin: qlen = prefill `extend_seq_lens.cumsum`,
   verify/draft_extend `arange(ndt, ndt+T, ndt)` (ndt from **get_spec()**, see crash ②),
   decode `arange(1,T+1)`; kvlen = `fm.seq_lens_cpu_int or fm.seq_lens`; block_table
   sliced to #reqs for prefill. sparse_count=index_topk, sparse_mode=3.
5. **Per-query tail + dedup**: tail = positions[:,None] + arange(-(kpool-2),1) (the query's
   own last kpool-1 positions — causally correct for every mode, no group expansion
   needed), mask <0 → -1. dup = (topk[:,:,None] == tail[:,None,:]).any(-1) → set dupes in
   topk to -1, return cat([tail, topk_dedup]) → **width index_topk + kpool - 1 = 2051**.
   At ≤2048 the topk is all-visible so the tail fully dedups → set is EXACTLY the window.

## Capture-time crash chain (each = one read + one-line fix + relaunch; 6 boots)
① `AttributeError: 'IndexerKPool' object has no attribute 'rotary_emb'` — skip_rope=True
   (NoPE). The plain-mixin template assumes rope; guard it (design step 2).
② `AttributeError: '_AscendKDAHybrid' object has no attribute 'speculative_num_draft_tokens'`
   — for GLM (hybrid linear+DSA), `get_attn_backend()` returns the HYBRID WRAPPER, not
   the raw ascend backend. Draft-token count source of truth: `get_spec().speculative_num_draft_tokens`
   (import from `sglang.srt.runtime_context` — the same source the backend itself uses).
③ `AclNN_Parameter_Error(EZ1001): 2051 and 2048 cannot broadcast` at DRAFT-EXTEND capture
   (target verify + draft decode graphs had already PASSED) — `eagle_worker_v2.py:266`
   `self.dsa_index_topk = getattr(hf_config, "index_topk")` = 2048 sizes BOTH seed-buffer
   allocations (graph-runner line ~240 and `_get_dsa_extend_topk_buf` line ~904); kpool
   output is wider. Fix AT THE SINGLE DEFINITION: widen by `index_kpool-1` when
   `index_kpool>1 and index_kpool_always_select_tail`. Note: the on-disk draft
   config.json has NO index_* keys — the runtime hf_config is the merged/main config
   (empirically carries them; guard with getattr anyway).

## E2E validation battery (run INSIDE the serving container via /model NFS path)
- Needle: 300-doc prompt, ~10,000 tokens, needle at doc #3 (~9,800 tokens from the
  query) → HIT. Proves prefill >2048 + verify-graph replay + sparse chain end-to-end.
- Arithmetic battery at greedy: 12×14 ✓ 45+38 ✓ 60km/1.5h ✓ — but 17×23 → 529 (23²).
  **Known-bug separation discipline**: the OLD deployment on the SAME tree @2048 answered
  289 (17²) — same question already failed, same failure MODE (squares one operand) →
  pre-existing MHC-missing quality issue, NOT a port regression. A same-question failure
  on old+new does not indict the change; compare failure MODES, and check EOS behavior
  on a case that needs it (needle stopped at 7 tokens ✓).
- accept len: `docker logs | grep -aoE 'accept len: [0-9.]+' | awk avg` → 2.99/6 (old
  range 2.15–3.27 → MTP intact at long context).
- Graph-replay proof: decode log lines `npu graph: True` at `#full token: 10048`.
- Throughput separation: scheduler `gen throughput` (decode-only, ~37 tok/s vs old 41.8)
  vs wall-clock aggregate (prefill-dominated at 16×1.8K prompts — NOT comparable to the
  old short-prompt baseline; state prompt shape when comparing).
- Benign startup tracebacks: urllib ConnectionRefused during the health-probe loop
  before HTTP binds — ignore unless after "fired up".

## Open items
- Granularity approximation unquantified vs trained kpool blocks (needle works; a
  quality A/B vs the full kpool port is future work if quality complaints appear).
- Greedy ramble-past-answer pattern (answers correct then repeats) — same tree at 2048
  showed a cleaner stop; worth checking with temp>0 or against the MHC-patched tree.
- MHC_FP32_RESIDUAL quality A/B still pending user approval (separate from this port).
- Context ceiling: pool 550K tokens → 32768×~16 concurrent; native max is 1M
  (max_position_embeddings=1,048,576) if a bigger pool is ever wanted.
- Scripts: e2e_test_0907.py / tp_test_0907.py in /data/models/mtp_experiment_0907/
  (also /home/ang/workspace/mtp-8100/). Patches: dsa_indexer_kpool.py (forward_npu),
  eagle_worker_v2.py (seed width), deepseek_v2_attention_mla_npu.py (P1: target-layer
  forward_npu direct call), ascend_backend.py (P2: forward_sparse zero-rope + 3-D→4-D
  KV view) — P1/P2 + op_test_longctx.py 11/11 (6e-4) predate this session.
