#!/usr/bin/env python3
"""glm5_next port overlay builder — lmsysorg CUDA dev build -> Ascend daily image.

Rebuilds the glm5_next (GLM-5.3-Flash) port overlay for
quay.io/ascend/sglang:main-cann9.0.0-910b from two source trees.

PREREQ: extract both sglang trees first (see references/glm53-flash-910b-prep-20260830.md §10.3):
  the CUDA host:  docker create --name X lmsysorg/sglang:glm-5.3-flash && docker cp X:/sgl-workspace/sglang/python/sglang /tmp/sgl_lmsysorg
  local:   docker create --name Y quay.io/ascend/sglang:main-cann9.0.0-910b && docker cp Y:/sgl-workspace/sglang/python/sglang daily/sglang
  mkdir -p /tmp/glm53_port/{lmsysorg,daily}; place trees as lmsysorg/sglang and daily/sglang

Then: python3 build_port.py  ->  port/ overlay tree
Dockerfile: FROM quay.io/ascend/sglang:main-cann9.0.0-910b  +  COPY port/sglang/ /sgl-workspace/sglang/python/sglang/
⚠️ pull base with --platform linux/arm64 (910B is aarch64).

Key decisions encoded here (2026-08-30):
- KDA attention -> AscendKDAAttnBackend/AscendKDAHybridLinearAttnBackend (mirror daily's kimi_linear
  NPU branch), NOT lmsysorg's plain KDAAttnBackend (CUDA-only path)
- server_args.py hunks dropped (that logic moved to arg_groups/overrides.py in daily)
- glm4v.py whole-file copy (daily 123-line subset lacks GLM video stack)
- model registration via EntryClass scan, no models/__init__.py wiring
- utils/***/ literal-asterisk dir = build artifact on both images, ignore
"""
import os, re, shutil, sys, py_compile

ROOT = os.path.dirname(os.path.abspath(__file__))
LMS = os.path.join(ROOT, "lmsysorg")
DAY = os.path.join(ROOT, "daily")
OUT = os.path.join(ROOT, "port")

COPY_FILES = [
    "sglang/srt/configs/glm5_next.py",
    "sglang/srt/models/glm5_next.py",
    "sglang/srt/models/glm5_next_nextn.py",
    "sglang/srt/layers/communicator_mhc.py",
    "sglang/srt/layers/communicator_mhc_hybrid_cp.py",
    "sglang/srt/multimodal/processors/glm4v.py",
]

errors = []

def patch(path, old, new, count=1):
    """Apply exact string patch to a daily file, write result to port tree."""
    src_path = os.path.join(DAY, path)
    dst_path = os.path.join(OUT, path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    src = open(src_path).read()
    n = src.count(old)
    if n != count:
        errors.append(f"ANCHOR FAIL ({n}x, want {count}x) in {path}:\n---\n{old[:300]}\n---")
        open(dst_path, "w").write(src)
        return
    open(dst_path, "w").write(src.replace(old, new))
    print(f"  patched {path} ({count} hunk)")

if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)

print("== copying new files from lmsysorg ==")
for f in COPY_FILES:
    dst = os.path.join(OUT, f)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.join(LMS, f), dst)
    print(f"  copied {f}")

print("== patching shared files (base: daily) ==")

# ---------- 1. configs/hybrid_arch.py ----------
patch("sglang/srt/configs/hybrid_arch.py",
"""def linear_attn_model_spec(model_config: ModelConfig):""",
"""def glm5_next_config(model_config: ModelConfig):
    hf_config = model_config.hf_config
    if hf_config.model_type == "glm5_next" and not model_config.is_draft_model:
        return hf_config.get_text_config()
    return None


def linear_attn_model_spec(model_config: ModelConfig):""")

patch("sglang/srt/configs/hybrid_arch.py",
"""        or kimi_linear_config(model_config)
        or hybrid_lightning_config(model_config)
    )
    if existing:
        return existing
    result = _get_linear_attn_registry_result(model_config)
    return result[1] if result else None""",
"""        or kimi_linear_config(model_config)
        or glm5_next_config(model_config)
        or hybrid_lightning_config(model_config)
    )
    if existing:
        return existing
    result = _get_linear_attn_registry_result(model_config)
    return result[1] if result else None""")

# ---------- 2. configs/__init__.py ----------
patch("sglang/srt/configs/__init__.py",
"""from sglang.srt.configs.falcon_h1 import FalconH1Config
from sglang.srt.configs.granitemoehybrid import GraniteMoeHybridConfig""",
"""from sglang.srt.configs.falcon_h1 import FalconH1Config
from sglang.srt.configs.glm5_next import Glm5NextConfig, Glm5NextTextConfig
from sglang.srt.configs.granitemoehybrid import GraniteMoeHybridConfig""")

patch("sglang/srt/configs/__init__.py",
'''    "MuseGlimmerConfig",
    "MuseGlimmerAssistantConfig",
    "KimiLinearConfig",''',
'''    "MuseGlimmerConfig",
    "MuseGlimmerAssistantConfig",
    "Glm5NextConfig",
    "Glm5NextTextConfig",
    "KimiLinearConfig",''')

# ---------- 3. configs/model_config.py ----------
patch("sglang/srt/configs/model_config.py",
'''            "GlmMoeDsaForCausalLM",
            "GlmMoeDsaForCausalLMNextN",
            "LongcatFlashForCausalLM",''',
'''            "GlmMoeDsaForCausalLM",
            "GlmMoeDsaForCausalLMNextN",
            "Glm5NextForConditionalGenerationNextN",
            "Glm5NextForConditionalGeneration",
            "LongcatFlashForCausalLM",''')

patch("sglang/srt/configs/model_config.py",
'''        if is_draft_model and self.hf_config.architectures[0] == "Qwen3NextForCausalLM":
            self.hf_config.architectures[0] = "Qwen3NextForCausalLMMTP"
            self.hf_config.num_nextn_predict_layers = 1
''',
'''        if is_draft_model and self.hf_config.architectures[0] == "Qwen3NextForCausalLM":
            self.hf_config.architectures[0] = "Qwen3NextForCausalLMMTP"
            self.hf_config.num_nextn_predict_layers = 1

        if (
            is_draft_model
            and self.hf_config.architectures[0] == "Glm5NextForConditionalGeneration"
        ):
            self.hf_config.architectures[0] = "Glm5NextForConditionalGenerationNextN"
            self.hf_text_config.architectures = list(self.hf_config.architectures)
            self.hf_text_config.num_nextn_predict_layers = 1
            self.hf_text_config.linear_attn_config = None
''')

patch("sglang/srt/configs/model_config.py",
'''            or "GlmMoeDsaForCausalLM" in self.hf_config.architectures
            or "GlmMoeDsaForCausalLMNextN" in self.hf_config.architectures''',
'''            or "GlmMoeDsaForCausalLM" in self.hf_config.architectures
            or "GlmMoeDsaForCausalLMNextN" in self.hf_config.architectures
            or "Glm5NextForConditionalGeneration" in self.hf_config.architectures
            or "Glm5NextForConditionalGenerationNextN" in self.hf_config.architectures''')

patch("sglang/srt/configs/model_config.py",
'''        elif "SarvamMLAForCausalLM" in self.hf_config.architectures:''',
'''        elif (
            "SarvamMLAForCausalLM" in self.hf_config.architectures
            or "Glm5NextForConditionalGeneration" in self.hf_config.architectures
        ):''')

patch("sglang/srt/configs/model_config.py",
'''        hc_mult = getattr(self.hf_text_config, "hc_mult", 1)
        self.spec_hidden_size = (
            self.hidden_size * hc_mult if hc_mult > 1 else self.hidden_size
        )
        # mHC-flattened hidden size; None when not running an mHC model
        # (e.g. non-DeepSeek-V4 configs without ``hc_mult``).
        self.hc_hidden_size = self.spec_hidden_size if hc_mult > 1 else None''',
'''        hc_mult = getattr(self.hf_text_config, "hc_mult", 1)
        if (
            getattr(self.hf_config, "model_type", None) == "glm5_next"
            or getattr(self.hf_text_config, "model_type", None) == "glm5_next_text"
        ) and not getattr(self.hf_text_config, "mhc", False):
            hc_mult = 1
        # mHC-flattened hidden size; None when not running an mHC model
        # (e.g. non-DeepSeek-V4 configs without ``hc_mult``).
        self.hc_hidden_size = self.hidden_size * hc_mult if hc_mult > 1 else None
        if hc_mult > 1 and not (
            getattr(self.hf_config, "model_type", None) == "glm5_next"
            or getattr(self.hf_text_config, "model_type", None) == "glm5_next_text"
        ):
            self.spec_hidden_size = self.hidden_size * hc_mult
        else:
            self.spec_hidden_size = self.hidden_size''')

# ---------- 4. layers/attention/attention_registry.py ----------
patch("sglang/srt/layers/attention/attention_registry.py",
'''from sglang.srt.configs.hybrid_arch import (
    hybrid_gdn_config,
    hybrid_lightning_config,
    kimi_linear_config,
    mamba2_config,
    mambaish_config,
)''',
'''from sglang.srt.configs.hybrid_arch import (
    glm5_next_config,
    hybrid_gdn_config,
    hybrid_lightning_config,
    kimi_linear_config,
    mamba2_config,
    mambaish_config,
)''')

patch("sglang/srt/layers/attention/attention_registry.py",
'''                linear_attn_backend = KDAAttnBackend(runner)
        elif hybrid_lightning_config(runner.model_config) is not None:''',
'''                linear_attn_backend = KDAAttnBackend(runner)
        elif glm5_next_config(runner.model_config) is not None:
            if _is_npu:
                from sglang.srt.hardware_backend.npu.attention.ascend_kda_backend import (
                    AscendKDAAttnBackend,
                    AscendKDAHybridLinearAttnBackend,
                )

                linear_attn_backend = AscendKDAAttnBackend(runner)
                hybrid_backend_cls = AscendKDAHybridLinearAttnBackend
            else:
                linear_attn_backend = KDAAttnBackend(runner)
        elif hybrid_lightning_config(runner.model_config) is not None:''')

# ---------- 5. mem_cache/kv_cache_builder.py ----------
patch("sglang/srt/mem_cache/kv_cache_builder.py",
'''from sglang.srt.configs.hybrid_arch import (
    hybrid_gdn_config,
    hybrid_lightning_config,
    kimi_linear_config,
    linear_attn_model_spec,
    mamba2_config,
)''',
'''from sglang.srt.configs.hybrid_arch import (
    glm5_next_config,
    hybrid_gdn_config,
    hybrid_lightning_config,
    kimi_linear_config,
    linear_attn_model_spec,
    mamba2_config,
)''')

patch("sglang/srt/mem_cache/kv_cache_builder.py",
'''        or kimi_linear_config(model_config) is not None
        or hybrid_lightning_config(model_config) is not None''',
'''        or kimi_linear_config(model_config) is not None
        or glm5_next_config(model_config) is not None
        or hybrid_lightning_config(model_config) is not None''')

# ---------- 6. arg_groups/overrides.py ----------
patch("sglang/srt/arg_groups/overrides.py",
'''    "PixtralForConditionalGeneration",
    "GlmMoeDsaForCausalLM",
    "LongcatFlashForCausalLM",
    "LongcatFlashForCausalLMNextN",
    "Dots3NoteForCausalLM",
)''',
'''    "PixtralForConditionalGeneration",
    "GlmMoeDsaForCausalLM",
    "Glm5NextForConditionalGeneration",
    "LongcatFlashForCausalLM",
    "LongcatFlashForCausalLMNextN",
    "Dots3NoteForCausalLM",
)''')

patch("sglang/srt/arg_groups/overrides.py",
'''        "GraniteMoeHybridForCausalLM",
        "NemotronHForCausalLM",''',
'''        "GraniteMoeHybridForCausalLM",
        "Glm5NextForConditionalGeneration",
        "NemotronHForCausalLM",''')

patch("sglang/srt/arg_groups/overrides.py",
'''        "PixtralForConditionalGeneration",
        "GlmMoeDsaForCausalLM",
        "LongcatFlashForCausalLM",
        "LongcatFlashForCausalLMNextN",
        "Dots3NoteForCausalLM",
    }
)''',
'''        "PixtralForConditionalGeneration",
        "GlmMoeDsaForCausalLM",
        "Glm5NextForConditionalGeneration",
        "LongcatFlashForCausalLM",
        "LongcatFlashForCausalLMNextN",
        "Dots3NoteForCausalLM",
    }
)''')

# ---------- 7. utils/common.py ----------
patch("sglang/srt/utils/common.py",
'''@dataclass
class VideoData:''',
'''GLM_MEDIA_CONFIG_KEYS = (
    "fps",
    "max_frames",
    "max_tokens_per_frame",
    "max_image_tokens",
)


@dataclass
class VideoData:''')

# ---------- verify ----------
print("\n== verification ==")
if errors:
    print(f"!! {len(errors)} ANCHOR FAILURES:")
    for e in errors:
        print(e)
    sys.exit(1)

ok = True
for root, dirs, files in os.walk(OUT):
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(root, f)
            try:
                py_compile.compile(p, doraise=True)
            except py_compile.PyCompileError as e:
                print(f"  SYNTAX FAIL {p}: {e}")
                ok = False
print("syntax: OK" if ok else "syntax: FAILED")
n = sum(len(fs) for _, _, fs in os.walk(OUT))
print(f"port tree: {n} files at {OUT}")
sys.exit(0 if ok else 1)
