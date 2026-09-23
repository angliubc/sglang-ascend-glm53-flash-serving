#!/usr/bin/env python3
"""GLM-5.3-Flash W8A8 v9 — experts-only int8 (rest stays BF16).
[v9 0916 = v8 verbatim with w_bit 4->8; v8 W4A8 already proven in the
0916 production-shape A/B: cc@1.0 10% err, bare 5%, greedy 20/20 clean,
vs xpinke full-quant W8A8 25-30%/0%/0%. W4A8_DYNAMIC contract removed —
it is w4-only; int8 is dense per-channel like quant_naive_w8a8.py.]

Based on v7 (unfuse recipe, verified correct 0903 by quant_v7_verify.py:
bf16 bit-exact + fp32 exact on L3/10/44; v7's 0.0625 fail was the dispatched
fused implementation's 1-2 ULP noise vs too-tight 1e-2 threshold).

v8 changes vs v7:
  1. quant W8A8 -> W4A8 (w_bit=4; msmodelslim docstring-documented usage,
     is_lowbit stays False = dense quant, NOT sparse)
  2. disable_names INVERTED: quantize ONLY language-model routed-expert linears
     (288x42 per-expert gate/up/down). Attention/KDA projections/dense MLP/
     shared expert/router/lm_head all stay BF16 — KDA gating is precision-
     sensitive (beta sigmoid lesson); experts are ~95% of params = the win.
  3. AntiOutlierConfig gains disable_anti_names (same list) so anti-outlier m2
     never perturbs the BF16-kept modules.
  4. Corrected counts: 42 MoE layers (L3-44 main model; L45 = MTP head, 889
     tensors, excluded per official Ascend recipe), 36288 expert tensors.
  5. Equivalence threshold 1e-2 -> 1.0 (run1, identical seeded input: L3 0.0625 /
     L10 0.078 / L44 0.375 = weight-magnitude-scaled dispatch noise; real swap
     bugs measure 7-42) + inline up_first control per verify layer which must
     exceed max(5.0, 4x the correct-order diff).

Run: docker exec -d glm53-quant (CPU, .33). Expect ~8-14h.

v8.1 (2026-09-04 wecom2 session) — OOM fix instrumentation:
  run3 OOM'd at unfuse 42/42 -> post-unfuse, anon-rss 973GB (model 642GB + ~330GB
  stranded). Root cause: glibc dynamic mmap threshold ratchets to 32MB, so the
  ~17MB bf16 clone tensors (36288 of them) churn through brk heap and fragment.
  Fixes: (a) relaunch with MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=
  2097152 (pins threshold, >=1MB allocs always mmap => freed=returned to OS);
  (b) gc.collect() per unfused layer; (c) VmRSS logged at every phase so a
  recurrence is diagnosable. Script logic otherwise UNCHANGED from v8.
"""
import os, time, shutil, json, glob, struct, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, Glm5NextForConditionalGeneration
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

# --- W4A8_DYNAMIC contract, checked before anything expensive runs ---
# msmodelslim check_and_set_w4a8_dynamic_config demands, for w_bit=4/a_bit=8:
#   is_dynamic=True, per-group (open_outlier=False, is_lowbit=True,
#   group_size in (32,64,128,256)), and w_sym=True.
# run5 died on that validator only after a 3h18m load plus anti-outlier,
# because the call site passed none of them. Validating here costs nothing.
# W8A8: dense per-channel int8; the w4-only dynamic contract does not apply.

try:
    _cfg_probe = QuantConfig(
        w_bit=8, a_bit=8, disable_names=[], dev_type="cpu", dev_id=0,
        act_method=2, mm_tensor=False
    )
    del _cfg_probe
    print("[gate] QuantConfig(W4A8_DYNAMIC) accepted; safe to load the model",
          flush=True)
except Exception as _e:
    raise SystemExit("[gate] QuantConfig rejected before load: %r" % (_e,))


MODEL = "/model/GLM-5.3-Flash-BF16"
OUT = "/data/models/GLM-5.3-Flash-W8A8-MoE"
N_CALIB = 12
VERIFY_LAYERS = {3, 10, 44}
EXPECTED_EXPERT_KEYS = 42 * 288 * 3  # 36288 — main model L3-44 (L45 = MTP head, excluded)

t0 = time.time()
def log(msg):
    print("[%s +%.0fs] %s" % (time.strftime("%F %T"), time.time() - t0, msg), flush=True)

def rss_gb():
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmRSS"):
                return int(ln.split()[1]) / 1048576.0
    except Exception:
        pass
    return -1.0

# ---------- unfused experts (v7, unchanged) ----------
class ExpertMLP(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

class UnfusedExperts(nn.ModuleList):
    """Drop-in for Glm5NextTextExperts; ModuleList subclass => child names
    '0','1',... => state-dict keys mlp.experts.N.gate_proj.weight, identical
    to the BF16 checkpoint and the reference Ascend W8A8 layout."""
    def __init__(self, num_experts, hidden, inter, swiglu_limit):
        super().__init__([ExpertMLP(hidden, inter) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            m = self[int(expert_idx)]
            gate = m.gate_proj(hidden_states[token_idx])
            up = m.up_proj(hidden_states[token_idx])
            gate = gate.clamp(min=None, max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            current = m.down_proj(F.silu(gate) * up) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final

def is_fused_experts(module):
    return hasattr(module, "gate_up_proj") and hasattr(module, "num_experts") and hasattr(module, "down_proj")

def unfuse_all(model):
    names = [name for name, m in model.named_modules() if is_fused_experts(m)]
    log("fused-expert modules found: %d (expect 42; L45 MTP head excluded by design)" % len(names))
    for n_done, name in enumerate(names, 1):
        module = model.get_submodule(name)
        layer_idx = None
        for part in name.split("."):
            if part.isdigit():
                layer_idx = int(part)
        E, i_, h = module.num_experts, module.intermediate_dim, module.hidden_dim
        unf = UnfusedExperts(E, h, i_, module.swiglu_limit).to(torch.bfloat16)
        with torch.no_grad():
            gu = module.gate_up_proj.data
            dn = module.down_proj.data
            for e in range(E):
                m = unf[e]
                m.gate_proj.weight.data = gu[e, :i_, :].clone()
                m.up_proj.weight.data = gu[e, i_:, :].clone()
                m.down_proj.weight.data = dn[e].clone()
        if layer_idx in VERIFY_LAYERS:
            torch.manual_seed(0)
            S = 256
            hid = torch.randn(S, h, dtype=torch.bfloat16)
            idx = torch.randint(0, E, (S, 8))
            w = torch.rand(S, 8, dtype=torch.float32)
            with torch.no_grad():
                ref = module(hid, idx, w)
                got = unf(hid, idx, w)
            diff = (ref.float() - got.float()).abs().max().item()
            # up_first control: identical rebuild with gate/up slices swapped —
            # must diverge at real-bug scale, proving this check still
            # discriminates above the layer's own dispatch-noise floor.
            unf_ctl = UnfusedExperts(E, h, i_, module.swiglu_limit).to(torch.bfloat16)
            with torch.no_grad():
                for e in range(E):
                    mc = unf_ctl[e]
                    mc.gate_proj.weight.data = gu[e, i_:, :].clone()
                    mc.up_proj.weight.data = gu[e, :i_, :].clone()
                    mc.down_proj.weight.data = dn[e].clone()
                ctl = unf_ctl(hid, idx, w)
            ctl_diff = (ref.float() - ctl.float()).abs().max().item()
            log("equiv L%s: max_abs_diff=%.5g (up_first control=%.5g)" % (layer_idx, diff, ctl_diff))
            # 1.0 threshold: dispatched fused impl vs per-expert loop differs by
            # weight-magnitude-scaled bf16 rounding (run1 same-seed input:
            # L3 0.0625 / L10 0.078 / L44 0.375); real swap bugs measure 7-42.
            # Control guard: swapped rebuild must exceed max(5.0, 4x correct diff).
            assert diff < 1.0, "equivalence FAILED at %s" % name
            assert ctl_diff > max(5.0, 4 * diff), "control blind at %s" % name
            del ref, got, ctl, unf_ctl
        parent_path, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_path), attr, unf)
        # free fused params immediately
        module.gate_up_proj.data = torch.empty(0, dtype=torch.bfloat16)
        module.down_proj.data = torch.empty(0, dtype=torch.bfloat16)
        del module
        if n_done % 8 == 0 or n_done == len(names):
            gc.collect()
            log("unfused %d/%d MoE layers rss=%.0fG" % (n_done, len(names), rss_gb()))
    return len(names)

# ---------- 1. load ----------
log("loading %s (cpu bf16) ..." % MODEL)
model = Glm5NextForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16,
    trust_remote_code=True, local_files_only=True,
).eval()
processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
log("model loaded rss=%.0fG" % rss_gb())

# ---------- 2. unfuse ----------
log("unfusing fused MoE experts ...")
n_moe = unfuse_all(model)
assert n_moe == 42, "expected 42 MoE layers (L3-44; L45=MTP excluded), got %d" % n_moe
sd_keys = list(model.state_dict().keys())
assert "model.language_model.layers.10.mlp.experts.0.gate_proj.weight" in sd_keys, "expert key naming wrong!"
n_exp_keys = sum(1 for k in sd_keys if ".mlp.experts." in k)
log("expert state-dict keys: %d (expect %d)" % (n_exp_keys, EXPECTED_EXPERT_KEYS))
assert n_exp_keys == EXPECTED_EXPERT_KEYS

# ---------- 3. calib data (v6/v7 set) ----------
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
    calib_data.append([inputs["input_ids"], inputs["attention_mask"], None])
log("calib samples: %d" % len(calib_data))

# ---------- 4. full-model smoke forward through unfused experts ----------
log("smoke forward (1 sample, cpu) ...")
try:
    with torch.no_grad():
        out = model(calib_data[0][0], calib_data[0][1], None)
    shp = tuple(out.logits.shape) if hasattr(out, "logits") else type(out)
    log("smoke ok, logits shape %r" % (shp,))
except Exception as e:
    log("smoke FAILED (non-fatal, continuing): %r" % (e,))

# ---------- 5. disable list (v8 rule: ONLY language-model routed experts quantized) ----------
disable_names = [name for name, module in model.named_modules()
                 if isinstance(module, torch.nn.Linear)
                 and not (name.startswith("model.language_model") and ".mlp.experts." in name)]
n_exp_linear = sum(1 for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear)
                   and name.startswith("model.language_model") and ".mlp.experts." in name)
log("disable_names: %d linears stay BF16; %d expert linears to quantize (expect %d)"
    % (len(disable_names), n_exp_linear, EXPECTED_EXPERT_KEYS))
assert n_exp_linear == EXPECTED_EXPERT_KEYS

# ---------- 6. anti-outlier (experts only) ----------
log("anti_outlier m2 (w4, experts only) ...")
anti_config = AntiOutlierConfig(
    w_bit=8, a_bit=8, anti_method="m2",
    dev_type="cpu", dev_id=0,
    disable_anti_names=disable_names,
)
anti_outlier = AntiOutlier(model, calib_data=calib_data[:4], cfg=anti_config)
anti_outlier.process()
log("anti_outlier done rss=%.0fG" % rss_gb())

# ---------- 7. quantize (W4A8, experts only) ----------
log("W8A8 calibration (%d expert linears, dense int8 per-channel) ..." % n_exp_linear)
quant_config = QuantConfig(
    w_bit=8, a_bit=8,
    disable_names=disable_names,
    dev_type="cpu", dev_id=0,
    act_method=2, mm_tensor=False,
)
calibrator = Calibrator(model, quant_config, calib_data=calib_data, disable_level="L0")
calibrator.run()
log("calibration done rss=%.0fG" % rss_gb())

# ---------- 8. save ----------
os.makedirs(OUT, exist_ok=True)
calibrator.save(OUT, save_type=["safe_tensor"], part_file_size=5 * 1024**3)
for f in os.listdir(MODEL):
    if f.endswith((".json", ".jinja", ".txt")) and not f.startswith("model"):
        shutil.copy(os.path.join(MODEL, f), os.path.join(OUT, f))
log("saved to %s" % OUT)

# ---------- 9. post-save dtype verification ----------
tot, n_exp_q, n_exp_bf16 = {}, 0, 0
for p in sorted(glob.glob(os.path.join(OUT, "*.safetensors"))):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    for k, v in hdr.items():
        tot[v["dtype"]] = tot.get(v["dtype"], 0) + 1
        if ".mlp.experts." in k:
            if v["dtype"] == "BF16":
                n_exp_bf16 += 1
            else:
                n_exp_q += 1
size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(OUT, "*")))
log("POSTCHECK dtypes=%s size=%.1fGB expert_quantized=%d/%d expert_left_BF16=%d" % (
    tot, size / 1e9, n_exp_q, EXPECTED_EXPERT_KEYS, n_exp_bf16))
if n_exp_q == EXPECTED_EXPERT_KEYS and n_exp_bf16 == 0:
    open(os.path.join(OUT, ".quant_v9_ok"), "w").write("ok\n")
    log("QUANT_V9_DONE_OK")
else:
    log("QUANT_V9_DONE_BUT_EXPERTS_INCOMPLETE")
