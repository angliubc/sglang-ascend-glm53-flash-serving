#!/usr/bin/env python3
"""Fused→unfused expert swap: equivalence probe (CPU, real weights, no full load).

When: any recipe that REPLACES a fused MoE experts module (Glm5NextTextExperts-
style gate_up_proj[E,2i,h] nn.Parameter) with per-expert nn.Linear must prove
the swap is mathematically exact before burning hours on calibration. An assert
fired at "max_abs_diff >= 1e-2" is NOT proof of a bug — it may be kernel-
blocking/ULP noise between the old module's DISPATCHED forward and the
reference loop.

Method (4 discriminators, verdict unambiguous):
  1. bf16 correct fusion order [gate;up]  → bit-exact 0 if swap is correct
  2. bf16 wrong order [up;gate]           → ~7-42 (real-bug magnitude reference)
  3. fp32 correct order                   → 0 proves mathematical equivalence
  4. matmul noise floor 1x[2i] vs 2x[i]   → blocking diff on this torch build

Only reads the target layers' expert tensors from checkpoint shards via
safetensors (index.json-guided, ~1s/shard warm) — no full model load.

Deployed verdict (2026-09-03, node3 glm53-quant, GLM-5.3-Flash-BF16, L3/10/44):
  bf16=0 / fp32=0 / noise=0 on all 3 layers; up_first 7.35/6.91/42.13 → the
  v7 recipe's 0.0625 failure was transformers' @use_experts_implementation
  dispatching the fused forward to a batched impl (1-2 ULP vs reference loop).
  Correct assert threshold: 0.25 (splits noise ~0.06 from real bugs ~7).

Usage:  python3 unfuse_equiv_probe.py [MODEL_DIR]   # default /model/GLM-5.3-Flash-BF16
Adapt:  key regex + Fused.forward reference loop for other architectures.
"""
import json, re, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import safe_open

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/model/GLM-5.3-Flash-BF16"
LAYERS = [3, 10, 44]          # first / mid / last MoE layer
S = 256                       # probe tokens

t0 = time.time()
def log(m):
    print("[%s +%.0fs] %s" % (time.strftime("%F %T"), time.time() - t0, m), flush=True)

torch.set_num_threads(64)

# ---------- checkpoint index facts ----------
idx = json.load(open(MODEL + "/model.safetensors.index.json"))
wm = idx["weight_map"]
pat = re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
moe_layers = set()
n_gate = 0
for k in wm:
    m = pat.search(k)
    if m:
        moe_layers.add(int(m.group(1)))
        if m.group(3) == "gate":
            n_gate += 1
log("index: MoE layers=%d (%s..%s), gate tensors=%d, expert tensors=%d"
    % (len(moe_layers), min(moe_layers), max(moe_layers), n_gate, n_gate * 3))
# NOTE (Glm5Next): main model L3-44 = 42 MoE layers; L45 = MTP head brings the
# checkpoint total to 43 — quantize only the 42, exclude MTP per official recipe.

cfg = open(MODEL + "/config.json").read()
mm = re.search(r'"swiglu_limit"\s*:\s*([\d.]+)', cfg)
LIMIT = float(mm.group(1)) if mm else 7.0
log("swiglu_limit=%g" % LIMIT)

k0 = next(k for k in wm if pat.search(k) and ".3.mlp.experts.0.gate_proj." in k)
with safe_open(MODEL + "/" + wm[k0], framework="pt", device="cpu") as f:
    w0 = f.get_tensor(k0)
I, H = w0.shape                                   # [moe_inter, hidden]
E = len({pat.search(k).group(2) for k in wm
         if pat.search(k) and ".layers.3." in k and ".gate_proj." in k})
log("dims: H=%d I=%d E=%d" % (H, I, E))

# ---------- targeted read of probe-layer expert tensors ----------
need = {}
for L in LAYERS:
    for k in wm:
        m = pat.search(k)
        if m and int(m.group(1)) == L:
            need[k] = (L, int(m.group(2)), m.group(3))
by_shard = {}
for k, v in need.items():
    by_shard.setdefault(wm[k], []).append(k)
log("need %d tensors across %d shards" % (len(need), len(by_shard)))

W = {L: {"gate": {}, "up": {}, "down": {}} for L in LAYERS}
for shard in sorted(by_shard):
    with safe_open(MODEL + "/" + shard, framework="pt", device="cpu") as f:
        for k in by_shard[shard]:
            L, e, part = need[k]
            W[L][part][e] = f.get_tensor(k)
    log("read %s" % shard)

# ---------- fused reference (verbatim Glm5NextTextExperts math) ----------
class Fused(nn.Module):
    def __init__(self, E, h, i, limit, dtype):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(E, 2 * i, h, dtype=dtype))
        self.down_proj = nn.Parameter(torch.empty(E, h, i, dtype=dtype))
        self.num_experts, self.swiglu_limit = E, limit

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
            gu = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])
            gate, up = gu.chunk(2, dim=-1)
            gate = gate.clamp(min=None, max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            current = F.linear(F.silu(gate) * up, self.down_proj[expert_idx]) \
                * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final

class ExpertMLP(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

class UnfusedExperts(nn.ModuleList):
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

def build(L, dtype=torch.bfloat16, up_first=False):
    fused = Fused(E, H, I, LIMIT, dtype)
    unf = UnfusedExperts(E, H, I, LIMIT).to(dtype)
    # ^ .to(dtype) BEFORE loading: nn.Linear defaults to fp32; skipping this
    #   breaks the bf16 forward with a dtype-mismatch error (hit on 0903).
    with torch.no_grad():
        for e in range(E):
            g = W[L]["gate"][e].to(dtype)
            u = W[L]["up"][e].to(dtype)
            d = W[L]["down"][e].to(dtype)
            a, b = (u, g) if up_first else (g, u)    # wrong-order control
            fused.gate_up_proj.data[e, :I].copy_(a)
            fused.gate_up_proj.data[e, I:].copy_(b)
            fused.down_proj.data[e].copy_(d)
            m = unf[e]
            m.gate_proj.weight.data.copy_(g)
            m.up_proj.weight.data.copy_(u)
            m.down_proj.weight.data.copy_(d)
    return fused, unf

def check(L):
    log("=== layer %d ===" % L)
    torch.manual_seed(0)
    hid = torch.randn(S, H, dtype=torch.bfloat16)
    idxr = torch.randint(0, E, (S, 8))
    w = torch.rand(S, 8, dtype=torch.float32)

    for up_first in (False, True):
        fused, unf = build(L, torch.bfloat16, up_first)
        with torch.no_grad():
            r = fused(hid, idxr, w)
            g = unf(hid, idxr, w)
        d = (r.float() - g.float()).abs().max().item()
        log("  [%s] bf16 max_abs_diff = %.5g   (out std %.3g)"
            % ("up_first" if up_first else "gate_first", d, r.float().std().item()))
        del fused, unf, r, g

    fused, unf = build(L, torch.float32, False)
    with torch.no_grad():
        r = fused(hid.float(), idxr, w)
        g = unf(hid.float(), idxr, w)
    d32 = (r - g).abs().max().item()
    log("  [gate_first fp32] max_abs_diff = %.5g  ==> %s"
        % (d32, "EXACT-EQUIV (any bf16 diff = kernel noise)" if d32 < 1e-4
           else "REAL BUG in swap"))
    del fused, unf, r, g

    torch.manual_seed(1)
    x = torch.randn(S, H, dtype=torch.bfloat16)
    gW, uW = W[L]["gate"][0], W[L]["up"][0]
    with torch.no_grad():
        full = F.linear(x, torch.cat([gW, uW], 0))
        split = torch.cat([F.linear(x, gW), F.linear(x, uW)], dim=-1)
    log("  noise floor (1x[2i] vs 2x[i] matmul, bf16) = %.5g"
        % (full - split).abs().max().item())

for L in LAYERS:
    check(L)
log("DONE")
