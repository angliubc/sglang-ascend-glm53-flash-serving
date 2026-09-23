# Decode-step hang on NPU multi-node serving (2026-08-31 late night)

Arc: after the KV-pool fix (`--max-total-tokens`) the prefill completed through ALL 45 layers
(py-spy: ranks in `sampler.py:158 forward`) — then decode hung forever. Watchdog 900 prevented
the premature kill; the request just sat. This file is the diagnostic ladder for that class:
**first token never arrives, no crash, no error, containers alive.**

## Symptom signature
- `curl` times out at `-m N` (HTTP 000) — server never responds, never 500s.
- Scheduler logs show `Prefill batch, #new-seq: 1, #new-token: 64` but **zero `Decode batch` lines**.
- py-spy on BOTH nodes' ranks: identical frame `process_batch_result_decode →
  _normalize_decode_outputs (batch_result_processor.py:1026)` — that line is `next_token_ids.tolist()`,
  the D2H sync. All ranks blocked there = the device queue never drained past the decode step's
  op chain (graph replay → logits all-gather → argmax).
- `npu-smi info`: **AICore 118–122%** (>100% = aggregate across cores) with HBM flat = a kernel
  SPINNING, not idle. This is the key discriminator:
  - AICore ~0% → host-side deadlock / waiting on a dead peer (collective will never complete).
  - AICore >100% → device-side spin: HCCL all-gather busy-wait polling for a peer's data, or a
    kernel in an infinite loop (e.g. replayed graph with garbage loop bounds / OOB indices).
- Sampler ops are innocent bystanders: a standalone probe of `torch.argmax([1,154820])` and even
  the exact w_vc `bmm(out=transposed-view)` pattern on the "failing" devices passes in 0.00s.
  Same rule as the prefill arc: **the op at the sync point is the reporter, not the culprit.**

## Gotcha: zero decode logs ≠ zero decode progress
`decode_log_interval` defaults to 40 — a request with `max_tokens < 40` NEVER prints a
`Decode batch` line. Don't conclude "decode never started" from absent logs; py-spy the
scheduler (`pgrep -f scheduler_TP0`) and look for `process_batch_result_decode`.

## Isolation lever: `--disable-cuda-graph`
Decode on NPU runs inside captured decode graphs (bs ladder captured at startup). Capture
EXECUTES the decode forward once with the THEN-current state; replay re-executes the recorded
op sequence with fixed addresses and baked control flow. If the capture-time state differs
from serving-time state (e.g. **kpool pool EMPTY at capture, populated after prefill**), the
replayed graph can: (a) replay a baked empty-pool branch (wrong results, no hang), or
(b) feed downstream kernels garbage/out-of-bounds indices — `npu_sparse_flash_attention`
requires in-bounds int32 indices [T,1,K]; the torch kpool transform pads invalid slots with
**-1** (`torch.where(valid, out, -1)`), which is OUT OF BOUNDS for that kernel → device spin.
`--disable-cuda-graph` runs decode eagerly with real pool state:
- works without graphs → the computation is right; fix the CAPTURE SEMANTICS (make every
  kpool decode read pool state from TENSORS, never host ints; never bake data-dependent
  branches) or exclude the indexer from graph capture.
- still hangs without graphs → the eager decode path itself (small-pool topk=2048 over 16
  entries, -1 indices reaching the sparse kernel, or the HCCL all-gather itself).
Cost: eager decode is slower (~tens of ms/step on 45 layers × 320B) — fine for smoke verdicts.

## Watchdog kill vs request hang — how they look from the client
- Watchdog firing: HTTP 500 (or 000 on later requests) + `node3` logs show `SIGQUIT received … one
  child failed` + `crash dump already performed, skipping` cascade + tokenizer_manager
  `asyncio.exceptions.CancelledError` in the ASGI traceback. The CHILD's real crash dump is
  EARLIER in `docker logs` — grep `-B60 "watchdog timeout"` for the py-spy dump of the stuck
  rank; `grep -B25 "hit an exception"` on the PEER node for the actual exception.
- Genuine request hang (watchdog raised to 900): nothing in the logs at all — only py-spy +
  npu-smi tell the story.

## Ladder that closed this arc (order matters)
1. Raise `--watchdog-timeout` first — separate "slow first pass" from "hung" (a 300s watchdog
   kills a healthy ~5-min first request; the kill masquerades as a 500 with CancelledError).
2. py-spy both nodes' TP0 + TP15 → confirm all ranks at the same frame (`.tolist()` sync).
3. `npu-smi` AICore → spinning vs idle discrimination (see above).
4. Standalone probe the op AT the sync point (argmax, exact bmm pattern) on the failing
   devices via `ASCEND_RT_VISIBLE_DEVICES` — all-pass ⇒ bystander confirmed.
5. `--disable-cuda-graph` rerun → splits "graph replay semantics" from "eager computation".
6. If eager works: audit every capture-safety rule in the decode path (tensor-driven sizes,
   no host branches, no -1 indices into index-bound kernels — remap -1 → a valid dummy slot
   or gate the kernel call).

## RESOLVED: no-graph mode generates end-to-end (2026-08-31 ~00:10)
- `--disable-cuda-graph` run: **HTTP 200**, 8/8 tokens + `finish_reason=length` (30.5s incl. prefill).
- 809-token prompt (chunked-mhc + KDA chunked prefill + kpool over 13 pages) + 32 decode
  tokens: HTTP 200 in 35s. **The eager small-pool/-1-index path is ELIMINATED** — the hang is
  confirmed to be graph-replay capture semantics, now purely a perf item (decode graphs ≈ 10x).
- Output text is garbage ("ookhiookhi…") — correct for `--load-format dummy` random weights;
  quality validation requires real/quantized weights.
- Perf at this stage: ~3.5s/token eager decode (45 layers × 320B TP16, both nodes). Fix
  direction for graphs: warm pool at capture, tensor-driven sizes (no host branches), remap
  -1 indices to a valid dummy slot before the index-bound sparse kernel — or exclude the
  indexer from capture.
- Final smoke flag set: `--mem-fraction-static 0.65 --max-total-tokens 16384
  --watchdog-timeout 900 --disable-cuda-graph --page-size 64 --disable-radix-cache
  --skip-server-warmup` (privileged, TRITON_CACHE_DIR mounted).
- BF16 120/120 at the download host; node3 forwarding (113/120 at handoff); auto W8A8 launcher armed —
  verified: quant container has 11 devices + `torch_npu.npu.is_available()=True`; the
  "npu-smi not in exec PATH" error inside glm53-quant is cosmetic (the launcher never calls
  npu-smi; python-side NPU access is what matters). Remaining: kpool graph fix (perf), W8A8
  real-weight deploy + quality check, topology decision re GLM-5.2 on node1+node2.
