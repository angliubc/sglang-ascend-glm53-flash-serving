#!/usr/bin/env python3
"""msmodelslim naive W8A8 quantization for a VLM (default adapter, transformers
generic forward). Template from the GLM-5.3-Flash (glm5_next) port — adapt
MODEL class, forward-slot layout, and disable_names regex to your arch.

V5 (2026-08-31, CURRENT): OUT must be a REAL filesystem path, never through a
symlink. Run#4 burned the full pipeline (57-min load + 6-min m2 + 17-min
calib, 12/12 samples done) and died at the LAST line that matters:
  calibrator.save(OUT, ...) ->
  ascend_utils/common/security/path.py get_valid_path ->
  ValueError: The value of the path cannot be soft link: /model/GLM-5.3-Flash-W8A8
The quant container's /model is a real dir holding per-model SYMLINKS
(/model/<Model> -> /data/models/<Model>); msmodelslim's security layer rejects
a symlinked WRITE path (READ paths like MODEL are not checked — loading through
/model/... worked in every run). Fix: OUT points at the real mount.
ARM-TIME gates (grep THIS file before firing): 'to(model.device)' PRESENT,
'npu:0' ABSENT, 'device_map' ABSENT, OUT line matching '/data/' or another
real (non-symlink) destination.

V4: calib tensors `.to(model.device)` (model-adaptive) + calib-device assert
before anti_outlier. Run#3 taught this: a stale hardcoded `.to("npu:0")` in
the calib list survived an in-place patch and only surfaced at anti_outlier's
FIRST forward (after 52-min load + 7-min init_dag): "Expected all tensors to
be on the same device, but got indices is on npu:0, different from other
tensors on cpu".

V3 history: pure-CPU route (no device_map). v1 bare device_map="auto" died
at load-finalize (crash A: planner packed cards to ~59.7GiB, fragmented
allocator couldn't serve the final 9GiB block). v2 added max_memory
50GiB/card caps + PYTORCH_NPU_ALLOC_CONF — STILL died (crash B: caps not
respected for the atomic 18GiB fused gate_up_proj = all 288 experts;
finalize's _init_weights → init.normal_ needed 18GiB with 18.58G free).
v3+ removes the NPU placement fight entirely: model in host RAM
(642GB < 1TB), msmodelslim dev_type="cpu". Zero meta-orphans possible →
no finalize device moves → load cannot OOM.

Measured phase timings on a 910B node (84 Kunpeng cores, 0831 runs; anchor ETAs):
- load 642GB: 52-57 min cold/semi-warm; ~10 min with FULL page-cache
  pre-warm (see SKILL §6a#9: reverse-order parallel cat; loader hits 2.6-3.4
  it/s vs 0.3 cold)
- anti_outlier m2: ~7 min init_dag (114 cores) + ~4 min forwards = ~6-7 min
- calibration: 12 samples × ~83s ≈ 17 min
- save: ~330GB in 5GB parts, minutes
- milestone ladder in the log: loading(+0s) → model loaded → calib samples →
  disable_names → anti_outlier m2 → anti_outlier done → W8A8 calibration →
  calibration done → saved → QUANT_DONE. The +Ns offsets are in CONTAINER
  clock (can lag host by 8-16h) — never correlate across hosts.

CPU-route viability prerequisites (check BOTH before this route):
1. MoE with small active params (GLM-5.3-Flash ~11B/token → CPU calib
   ~10-20 min/pass on 84 Kunpeng cores, full quant 1-2h). A DENSE model
   of this size would take days — check config (n_routed_experts /
   num_experts_per_tok) first.
2. CPU forward kernels exist: hybrid linear-attention archs (KDA) use
   @use_kernel_func_from_hub_with_fallback("chunk_kda", "fla") — the
   torch-native fallback only engages when fla is NOT importable. Verify
   `python3 -c "import fla"` FAILS in the container; if fla is installed,
   add a sys.meta_path blocker BEFORE importing transformers.
NPU stays 100% free during quant (co-tenant serving can use the cards).

Run INSIDE a container with transformers that knows the arch (e.g. 5.16.1
for glm5_next) and msmodelslim installed (--no-index --no-deps). NPU
devices are NOT required (plain CPU container works; torch_npu present is
harmless).

Recipe (naive run-1, max compression):
- anti_outlier m2 on a few text-only chat samples
- disable: vision merger mlp + vision blocks fc2 + lm_head; LLM fully W8A8
- act_method=2, mm_tensor=False (per-channel weights)
If output quality is broken, run-2 adds expert down_proj to disable_names.

Harmless msmodelslim CPU-route warnings (do not chase):
- "`cpu` is set as `dev_type`, `dev_id` cannot be specified manually!"
- "Not all elements in calib_data are torch.Tensor" (positional layout)
"""
import os, re, shutil, time
import torch
from transformers import AutoProcessor, Glm5NextForConditionalGeneration  # <- adapt class
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

MODEL = "/model/GLM-5.3-Flash-BF16"     # read path: symlink OK (not security-checked)
OUT = "/data/models/GLM-5.3-Flash-W8A8" # WRITE path: MUST be real, no symlink (v5)
N_CALIB = 12

t0 = time.time()
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time()-t0:.0f}s] {msg}", flush=True)

# 1. load — PURE CPU, NO device_map (v1/v2 died at finalize; see module docstring)
log(f"loading {MODEL} cpu-resident bf16 ...")
model = Glm5NextForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16,
    trust_remote_code=True, local_files_only=True,
).eval()
log("model loaded")
processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)

# 2. calib data — text-only chat. TENSORS FOLLOW THE MODEL'S DEVICE (v4).
# POSITIONAL layout must match the arch's forward signature (read it from the
# transformers wheel: zipfile + regex on modeling_<arch>.py). For glm5_next:
# (input_ids, attention_mask, position_ids, past_key_values, inputs_embeds,
#  labels, use_cache, pixel_values, ...) -> text-only needs just 3 slots.
CALIB_PROMPTS = [
    "请详细介绍量子计算的基本原理和主要技术路线。",
    "Explain the difference between transformer attention and state-space models.",
    "写一个Python函数，实现快速傅里叶变换，并注释关键步骤。",
    "The capital of France is",
    "总结一下分布式系统中的CAP定理及其在实际设计中的权衡。",
    "Solve: if 3x + 7 = 22, what is x? Show your reasoning step by step.",
    "用中文解释什么是混合专家模型（MoE），以及路由机制的作用。",
    "Write a SQL query to find the top 10 customers by total order value in 2024.",
    "描述一下photosynthesis的光反应和暗反应过程。",
    "Translate to English: 科学技术是第一生产力。",
    "def fibonacci(n):\n    # complete this function",
    "Explain why the sky is blue, in simple terms a child could understand.",
]
calib_data = []
for prompt in CALIB_PROMPTS[:N_CALIB]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    calib_data.append([inputs["input_ids"].to(model.device),
                       inputs["attention_mask"].to(model.device),
                       None])
log(f"calib samples: {len(calib_data)}")

# 2b. calib-device assert (v4): fail HERE, not after load+DAG inside anti_outlier
_dev = next(model.parameters()).device
for i, sample in enumerate(calib_data):
    for j, t in enumerate(sample):
        if isinstance(t, torch.Tensor):
            assert t.device == _dev, f"calib_data[{i}][{j}] on {t.device} != model on {_dev}"
log(f"calib device check OK ({_dev})")

# 3. disable list (naive run-1)
disable_names = []
for name, _ in model.named_modules():
    if re.search(r"merger\.mlp\.\d+$", name) or \
       re.search(r"visual\.blocks\.\d+\.mlp\.fc2$", name) or \
       name == "lm_head":
        disable_names.append(name)
log(f"disable_names: {len(disable_names)} modules")

# 4. anti-outlier (dev_type="cpu": msmodelslim moves nothing — model already cpu)
log("anti_outlier m2 ...")
anti_config = AntiOutlierConfig(w_bit=8, a_bit=8, anti_method="m2", dev_type="cpu", dev_id=0)
anti_outlier = AntiOutlier(model, calib_data=calib_data[:4], cfg=anti_config)
anti_outlier.process()
log("anti_outlier done")

# 5. quantize
log("W8A8 calibration ...")
quant_config = QuantConfig(w_bit=8, a_bit=8, disable_names=disable_names,
                           dev_type="cpu", dev_id=0, act_method=2, mm_tensor=False)
calibrator = Calibrator(model, quant_config, calib_data=calib_data, disable_level="L0")
calibrator.run()
log("calibration done")

# 6. save (v5: OUT real path — msmodelslim security check rejects symlinked
#    WRITE paths; died here in run#4 AFTER a fully successful calibration)
os.makedirs(OUT, exist_ok=True)
calibrator.save(OUT, save_type=["safe_tensor"], part_file_size=5 * 1024**3)
for f in os.listdir(MODEL):
    if f.endswith((".json", ".jinja", ".txt")) and not f.startswith("model"):
        shutil.copy(os.path.join(MODEL, f), os.path.join(OUT, f))
log(f"saved to {OUT}")
log("QUANT_DONE")
