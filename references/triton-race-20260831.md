# Triton cache races + async-stack misattribution (2026-08-30/31 night)

Continuation of `glm53-serving-bringup-20260830.md`: after that night's 8 fixes,
the server came up (graph capture ✓, "fired up" ✓) but the FIRST REAL REQUEST
still failed in three successive failure modes that all pointed the wrong way.

## Failure sequence (all on the same overlay build)

1. **HTTP 500 after ~500s** (first request). Watchdog (300s) killed the cluster.
   node3 py-spy: every TP rank blocked at `_mhc_pre_torch` sinkhorn loop line —
   an `add_Tensor` enqueue inside `OpCommand::RunOpApiV2 → eventfd_read`.
2. **node4 TP15 held the REAL traceback**: plain `F.linear(x, layer.weight)` in
   `linear.py:1632` (unquant.py apply) → `aclnnBatchMatMul 207001` on Device 7.
   CANN's own footer: "the operator is called asynchronously, the stacktrace
   may be inaccurate ... set ASCEND_LAUNCH_BLOCKING=1".
3. Earlier runs the same night: node4 TP12 / TP10 bmm 207001 "at mhc" — same
   misattribution; node3 ranks passed the identical op.

## Root-cause analysis

- **Async stacks blame the bystander.** The reported op is whichever enqueue
  hits the next sync point after the device dies — mhc (a sync-heavy chain)
  collected the blame three times while the actual failure was (a) a plain
  linear GEMM on the peer node, and (b) once a genuine memory issue (see
  207001=OOM in the bring-up reference). Ground truth requires either the
  PEER node's scheduler traceback, or `ASCEND_LAUNCH_BLOCKING=1` (sync mode,
  accurate stacks, big perf cost — debug only).
- **Why the peer's GEMM died: triton cache write race was the prime suspect —
  FALSIFIED as the (sole) root cause.**
  - Suspect rationale: serving compiles the KDA prefill triton kernels at
    first request; 8 ranks per node compile the SAME @triton.autotune kernel
    simultaneously into the SAME shared TRITON_CACHE_DIR. A partially-written
    cache entry read by another rank → corrupt device binary → kernel "runs",
    then unrelated later GEMMs fail/hang.
  - **The clean-cache experiment (2026-08-31 ~23:00)**: caches nuked on both
    nodes → single-process precompile each (`scripts/precompile_triton_npu.py`:
    l2norm 20s, prepare_chunk_indices 0s, chunk_local_cumsum 4.7s,
    chunk_kda_scaled_dot_kkt_fwd 127s; + T=8 / T=8192 bucket passes were 4-5s
    cache HITS — autotune keys for these kernels are NOT strongly
    shape-dependent) → smoke restarted → first request STILL **HTTP 500 in
    177s** (vs 502s before — cache worked, zero compile time) with node4 TP14
    `aclnnBatchMatMul 207001` on a plain F.linear. Same signature: node4 only,
    a different device each run (TP10=dev2, TP12=dev4, TP14=dev6, TP15=dev7
    — never node3, never the same device twice in a row).
  - Conclusion: the compile race is real and the precompile is still REQUIRED
    (it eliminates watchdog-vs-JIT and the compile-time flakiness class), but
    the one-node-only 207001 has a different root cause → see RESOLUTION below.
  - Ruled out along the way: co-tenant containers on node4 (sglang-tp32 with
    all 8 devices mapped, co-tenant vLLM containers — all ZERO processes inside,
    0 HBM held → harmless empty shells; AICore ~99% + 3.5GB HBM observed
    during smoke startup was the dummy-weight randn init, not contention);
    "KDA triton kernel poisons the device" (poison probe below).

## RESOLUTION (2026-08-31 ~23:40): oversized KV pool (ignores context-length AND mem-fraction)

The `ASCEND_LAUNCH_BLOCKING=1` serving run (env added to the docker run line
on BOTH nodes) delivered the exact failing site in one shot:

- **HTTP 500 in 88s** (fast — no compile time; also proves blocking mode is
  the cheapest precise diagnostic once you suspect memory, not a hang).
- Precise stack: `deepseek_v2.py:2239 forward_core` →
  `deepseek_v2_attention_mla_npu.py:529 forward_dsa_core_npu` →
  `torch.bmm(attn_output, m.w_vc, out=attn_bmm_output.view(...).transpose(0,1))`
  — the MLA w_vc projection bmm with `out=` into a transposed view, reached
  via `glm5_next.py:871` (the DSA attention core, NOT the indexer, NOT mhc).
- Full error text (grep the WHOLE block, not just the op name):
  `torch.OutOfMemoryError: ... call aclnnBatchMatMul failed, error code is 207001`
  + `[Error]: Failed to apply for memory.` +
  `Memory_Allocation_Failure(EL0004): Failed to allocate memory requested by
  APP module.` — **the op name lied; the exception TYPE told the truth.**

**Memory ledger (grep `Load weight end|KV Cache is allocated|Capture target
decode NPU graph begin|Capture target decode NPU graph end` per rank):**

| phase | number |
|---|---|
| Load weight end | avail 22.10GB, used 38.46GB |
| KV Cache allocated | **20.35GB, #tokens 419,904** |
| capture begin | avail 2.60GB |
| capture end | avail **1.26GB** |

`--context-length 8192` was set — and the pool still provisioned **419,904
tokens**: this NPU MLA pool path "maximizes capacity" from post-weight avail
(20.35/22.10 ≈ 92%) and ignores BOTH `--context-length` (caps per-request
length only) AND `--mem-fraction-static` (falsified below). Contrast: GLM-5.2
tp32 on the same image DOES respond to mem-fraction — the insensitivity is
arch/pool-path-specific. 64GB HBM −
38.46 weights − 20.35 pool − 1.34 graphs ≈ 1.26GB for ALL prefill transients
(KDA chunk workspaces, autotune, indexer buffers, mhc chunks) — the bmm's
aclnn workspace request failed → 207001 misreport.

**The node3-passes/node4-crashes asymmetry was allocation-order luck, not a node
defect**: both nodes sat at 1.26GB; node4's ranks tripped the wall first
(slightly different allocation timing via HCCL collectives), node3's ranks then
hung on the next cross-node collective waiting for the dead peer — the
watchdog finished the cluster. "One node only, different device each run" =
a shared resource wall, not hardware.

**Fix, in two steps (falsification first, then the working knob):**

1. `--mem-fraction-static 0.65` was deployed first — **did NOT shrink the
   pool**: next run's ledger showed `KV Cache ... #tokens: 419,776, KV size:
   20.34 GB` (unchanged), avail after capture 1.26GB → same 207001 class.
   The NPU MLA pool sizing on this path simply ignores the fraction.
2. `--max-total-tokens 16384` — **worked**: `#tokens: 16384, KV size: 0.80
   GB`, avail after capture ~20GB. This is the effective pool bound for
   smoke (and the safest lever whenever this pool path ignores fractions).

With W8A8 (~19GB/card) even more headroom frees up. For production, size
`--max-total-tokens` from real weights + target concurrency and leave
multi-GB headroom for prefill transients.

## Post-fix arc (2026-08-31 ~00:00-01:00): forward completes — watchdog kills in the SAMPLER

With the pool bounded (0.80GB) the first request changed character completely:
**HTTP 500 @ 521s, NO OOM, no 207001, no scheduler crash on either node.**
The watchdog dumps told the real story:

- node3 TP6 MainThread ACTIVE in `sampler.py:158 forward` (sampler executing),
  TP7 in `torch argmax → rtStreamSynchronizeWithTimeout` (sampler's greedy
  argmax in stream-sync) — **the prefill had completed ALL 45 layers** (mhc,
  KDA, kpool indexer, sparse attention, MoE) and the cluster was sampling
  when the default `watchdog_timeout=300` fired (it had gone stale during
  the ~5 min the request spent compiling NEW triton kernels for REAL serving
  shapes — the 4-kernel precompile covers the KDA chunk set but not every
  first-use kernel in the prefill path).
- The 500 + CancelledError at the HTTP layer was the SIGQUIT cascade, not a
  model failure.

**Final flag set for the confirmation run:** `--mem-fraction-static 0.65
--max-total-tokens 16384 --watchdog-timeout 900 --page-size 64
--disable-radix-cache --skip-server-warmup` + `TRITON_CACHE_DIR` mounted;
`ASCEND_LAUNCH_BLOCKING=1` REMOVED (diagnosis done — sync mode ≈2× op
latency). Milestone ladder after this fix: weights → pool → capture →
fired-up → **forward complete (sampler in py-spy)** → first token.

### Companion probe: exact-pattern replay on the failing devices

Before the ledger, run the EXACT failing op pattern (including `out=` view
gymnastics) standalone on the failing node's previously-failing devices:
`docker exec -e ASCEND_RT_VISIBLE_DEVICES=0 …`, `…=2`, `…=7` over the probe
container. w_vc bmm pattern passed on ALL of node4's devices → hardware/driver
theory dead in one cycle → serving-context memory. (Minor probe artifact:
`torch.ops.npu.batch_matmul_transpose` resolves only after some serving-side
import — don't chase that failure in a bare probe.)

## Poison-probe pattern (reusable)

Hypothesis "kernel A corrupts the device, later op B fails":
```python
def op_b(tag):  # the suspected victim, run in 3 phases
    ...
op_b("B-alone")        # passes? op B is fine alone
run_kernel_a()         # the suspected poisoner (real serving shapes)
op_b("B-after-A")      # fails? -> A is the poisoner. passes? -> theory dead
```
Ran this for chunk_kda_scaled_dot_kkt_fwd → mhc: PASSED → killed the
"triton kernel poisons the device" theory in one probe cycle and redirected
to the cache-race hypothesis. Note: import paths differ from what the
serving traceback shows — chunk_kda_scaled_dot_kkt_fwd lives in the sglang
tree (`sglang.kernels.ops.attention.fla.kda`, wrapper at :425), NOT in
sgl_kernel_npu (which only hosts the aclnn _npu variants: gate/prefill/
chunk_delta_h/solve_tril/target_verify).

## Watchdog note

`watchdog_timeout=300` (default) kills the whole scheduler tree if any
forward exceeds 5 min — during legitimate first-request JIT compilation
(aarch64 triton compile = minutes/kernel × several kernels) this fires
BEFORE the request can finish. Options: raise `--watchdog-timeout` for the
first-request run, or (better) precompile so the first request never compiles.

## Debugging checklist for "first request hangs/500s on multi-node NPU TP"

1. Which node actually crashed? Check BOTH nodes' `docker logs ... | grep -A40 "hit an exception"`.
2. Read the FAILING node's deepest frames — the healthy side's py-spy line is
   an innocent bystander (async enqueue).
3. Suspect JIT compile: py-spy the scheduler; `linalg_to_bin_enable_npu_compile_A2_A3`
   = triton-ascend compiling (minutes each). No scheduler logs + curl timeout
   + containers alive = compiling, not hung.
4. GEMM 207001 with a shape that passes standalone → device-state corruption
   or OOM-tiling; look at what RAN JUST BEFORE on that rank (first-execution
   triton kernels are prime suspects), and at concurrent cache writes.
5. Shared TRITON_CACHE_DIR + N ranks first-executing the same kernel = race.
   Nuke + single-process precompile per node, then restart. **If the failure
   then persists (clean cache, zero compile time), the race was not the root
   cause — go straight to ASCEND_LAUNCH_BLOCKING=1 for the accurate stack
   instead of iterating on more cache theories.**
6. One-node-only failures across runs: suspect node-level causes (peer-node
   cross-node collectives, device/driver state, memory layout) — and verify
   co-tenant containers are actually alive (process count + npu-smi per-
   process HBM) before counting them in.
7. **Read the exception TYPE, not the op name.** `torch.OutOfMemoryError`
   under a 207001 op failure = memory, full stop. Then grep the memory
   ledger per rank (`Load weight end|KV Cache is allocated|Capture ... begin/
   end`) and check the arithmetic against HBM. The NPU MLA KV pool ignores
   `--context-length` AND `--mem-fraction-static` (eats ~92% of post-weight
   avail) — bound it with `--max-total-tokens`; it is the usual hidden eater.
   Leave multi-GB headroom for prefill transients.
8. **After the pool fix, a 500 with NO crash + py-spy in `sampler.py`** =
   the forward completed; suspect the watchdog (300s default) killing the
   cluster mid-first-request-compile. Raise `--watchdog-timeout`, keep the
   triton cache warm, and re-request — the second request should skip the
   compile entirely.

## Live state at last update

- smoke.sh on both nodes (FINAL flag set): `--mem-fraction-static 0.65
  --max-total-tokens 16384 --watchdog-timeout 900 --page-size 64
  --disable-radix-cache --skip-server-warmup` + env
  `TRITON_CACHE_DIR=/data/models/triton_cache` (mounted; ASCEND_LAUNCH_BLOCKING
  removed after diagnosis). Confirmation request in flight (watcher armed):
  caches hot + pool 0.80GB + 900s watchdog — expected to be the first
  end-to-end token, or to surface the next (sampler-level) issue.
- BF16 download → node3 forwarding → auto W8A8 launcher chain running
  unattended (119/120 shards at last check, launcher window to 09:00).
- build_port.py section 19b = chunked mhc (CHUNK=1024, bf16 GEMM); port tree
  60 files, syntax-clean, rsync-verified on both nodes; triton caches on both
  nodes cleanly precompiled (T=512 main + T=8/8192 buckets).
- Remaining known-good-state notes: v1-arm tag on the workstation verified
  arm64/16,648,719,350 (a delayed notification showed an amd64 misbuild —
  stale artifact, no damage; dangling arm64 duplicates from the build
  experiments are cleanup candidates).
