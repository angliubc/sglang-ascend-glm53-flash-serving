# GLM-5.3-Flash W8A8 量化实战记录（2026-08-31，node3 glm53-quant 容器）

## 终局结论

**管线全通但产物是"混合量化"**：529 个 nn.Linear → int8（仅 ~7GB），
**融合 MoE 专家（570GB，~87% 参数）保持 BF16**。产物 578GB
`/data/models/GLM-5.3-Flash-W8A8/`（单文件 5032 张量，`.quant_done` 在）。

根因：transformers Glm5Next 的专家是 `Glm5NextTextExperts`（自定义 nn.Module，
`gate_up_proj = nn.Parameter([288, 2*I, H])` 裸 3D 参数），**不是 nn.Linear** —
msmodelslim 26.1.0 llm_ptq 流程（AntiOutlier/Calibrator）只包 nn.Linear，无
Glm5Next/融合 MoE 处理器（grep 实锤零支持）。578GB 两节点 TP16 可装（36G/卡）
但量化收益≈0。

## 先例：DSV4 W8A8 的专家是量化的

`/shared/models/DeepSeek-V4-Flash-w8a8-mtp/`（gluster 四节点共享）：
- 专家 per-expert int8（`layers.0.ffn.experts.0.w1.weight_scale` F32 + I8 权重，vLLM 风格无 model. 前缀）
- 用**新一代 modelslim_v1 声明式 yaml**（`DeepSeek-V4_best_practice.yaml`）：
  quarot(block 32) → flex_smooth_quant → linear_quant（**pattern 匹配** `*attn*`/`*ffn*`，
  非 isinstance-Linear）→ ascendv1_saver（part_file_size 4 → 70 片分片生效）
- `verified_model_types: [DeepSeek-V4]` — **Glm5Next 不在列**
- 攻专家量化的下一步 = 验证容器 msmodelslim 26.1.0 是否支持 modelslim_v1 spec + 3D 融合参数切片

## 六连败根因表（每死因不同，全部修根因）

| # | 死点 | 根因 | 修复 |
|---|------|------|------|
| 1 | 加载即 SafetensorError incomplete metadata | BF16 分片坏（node3 63/120，源头 the download host 19/120） | 头部校验（8B len + json + max(offsets)==filesize）定位；the transfer host→隧道2224→node3 只重推坏片；aria2c 修源头 |
| 2 | finalize `_init_weights` 18GiB normal_ OOM（NPU1 41.4G已占/18.58 free） | device_map=auto planner 过装 → in-load OOM 留 meta 孤儿 → finalize 在近满卡随机初始化巨张量 | **弃 NPU 放置流，转纯 CPU**（1TB 主存装 642GB，无规划=无孤儿） |
| 3 | m2 forward: "indices is on npu:0, different from other tensors on cpu" | calib 张量硬编码 `.to("npu:0")` | `.to(model.device)` 随模型自适应 |
| 4 | calibrator.save(): "path cannot be soft link" | 容器内 `/model/GLM-5.3-Flash-W8A8` 是软链，ascend_utils 安全检查拒绝 | OUT 用真实路径 `/data/models/...`（读路径软链无碍） |
| 5 | save: `input_scale.cpu()` NoneType | 纯文本 calib 永不前向视觉模块 → 挂钩但 input_scale=None；旧 disable 正则抄自 InternVL 命名 0 匹配（活模型视觉命名≠ckpt `model.visual.*`，转换映射改名） | disable **语言模型外一切 Linear**（`not name.startswith("model.language_model")`，结构性不猜名字） |
| 6 | （无 - 成功） | — | EXIT=0 + `.quant_done`；但见"终局结论"专家未量化 |

早期日志的 `MergeModulelist "Ckpt contains: 2"` 报错是坏分片时期遗物；
修复后加载报告 **0 MISSING / 0 UNEXPECTED / 0 转换错误**（量化数学跑在真权重上）。
#2 的 18GiB 张量 = layers.{7,10,...,25} 融合专家 gate_up_proj 随机初始化（当时误读为纯放置问题）。

## 纯 CPU 量化配方（v6 最终版，`templates/quant_naive_w8a8_v2.py` 同步）

- `from_pretrained(MODEL, torch_dtype=bf16)` **无 device_map**（防 #2 类放置崩溃）
- AntiOutlierConfig + QuantConfig 均 `dev_type="cpu"`
- calib 张量 `.to(model.device)`；OUT 真实路径
- disable = 非 `model.language_model` 的全部 Linear（视觉路径 BF16 保留，未来 VL 可用）
- 点火一律 `docker exec -d`（免疫 ssh 断连）+ 门禁（badlist=0/无标记/无残留/补丁在位/关键行 grep）+ EXIT=N 尾标记 + 成功才 touch `.quant_done`

## 性能档案（node3，2×NVMe RAID0 7.5T，1TB 主存，84+ 核）

- 裸盘读 3.4GB/s；页缓存读 2.5GB/s；**单线程 mmap 加载 ~130MB/s**（4KB 缺页上限，与盘速无关）
- transformers 本地 safetensors 加载 = 单线程 Python 循环（无内建并行；vLLM/SGLang loader 才有线程池）
- 冷缓存加载 52-57 分钟；**反向序 4 路 cat 预热**（120 片 ~5 分钟）后 ~10 分钟（2-3 it/s vs 0.3）
- **缓存动力学**：纯 CPU 流 RSS 650G 自挤页缓存（NPU 流 `.to(npu)` 释放源、缓存存活所以更快——"NPU 里快"的真相）
- m2：6.3 分钟（MoE 每 token 仅 ~11B 激活，CPU 前向便宜；fla 未装→transformers `use_kernel_func_from_hub_with_fallback` 自动走 torch 原生）
- 校准：12 样本 ×82s ≈ 22 分钟（#5）；#6 异常跑 92 分钟（未解，非阻塞）
- 保存：~10 分钟（part_file_size=5G 被 llm_ptq 老 saver 忽略 → 单文件 578GB，可加载但非预期布局；DSV4 新流程分片生效）

## 模型档案（量化相关）

- 321B 总参 MoE：45 层（42 sparse + 3 dense），288 专家/激活 8 + 1 共享，hidden 4096，moe_inter 2048
- 混合注意力：linear_attention（GDN/KDA，33 层）+ deepseek_sparse_attention（每 4 层）；nextn=1（MTP 层）
- 融合专家单张量：gate_up_proj [288, 8192, 4096] ≈ 18GiB bf16 / 9GiB int8 口径
- checkpoint 视觉权重命名 `model.visual.*`（InternVL 风格）≠ 活模型命名（转换映射加载，干净）

## v1 流程源码调查（0831 晚，任务②"专家量化攻坚"）

接上文"下一步"。调查结果（容器 msmodelslim 26.1.0）：

1. **v1 CLI 在**：`msmodelslim quant --model_path X --save_path Y --device cpu --config_path <yaml> --quant_type QuantType.W8A8`（`ms`/`msmodelslim` 两个入口均在 bin/）；`--device 'npu:0,1,2'` 多卡仅 v1 支持。DSV4 的 best_practice.yaml 正是此入口的配置。
2. **`LinearQuantProcessor._install_quantizer`（processor/quant/linear.py）只装 nn.Linear**：`if not isinstance(submodule, nn.Linear): continue` — include/exclude pattern 仅是名字过滤，叠在 isinstance 检查之上；`processor/quant/` 仅 linear/attention/autoround 三个处理器，全包 grep `TextExperts|fused.*expert|FusedMoE|num_experts` 零命中。
3. **DSV4/GLM-5.2 专家在 llm_ptq 下能量化的机制**：两者 transformers 实现是**逐专家 nn.Linear**（ModuleList，DSV4 产物 `experts.0.w1.weight_scale` 布局佐证；GLM-5.2 W4A8C8 的 230400 条目=75层×256专家×3矩阵×4张量同理）；Glm5Next 是融合 3D 自定义模块，Linear-only 检查拦不到。
4. ⚠️ **未闭环**：0901 外部实证（xpinke 产物，见 SKILL.md §6 与 `references/glm53-w8a8-xpinke-yaml-20260901.md`）证明 v1 yaml 路线能出**专家全 int8** 的 296GB 产物 — 与本调查的 Linear-only 源码矛盾。合理解释：v1 管线在 `processor/convert`/`container`（本次未深挖）或 saver 层存在拆装/3D 处理机制。**复刻 xpinke yaml 时以此为基准实测，勿单凭 linear.py 源码片段下结论。**
5. 0831 时点的悲观结论（"专家量化=拆装手术+sglang-Ascend 量化内核，数天级专项"）已被 xpinke 产物推翻一半——量化侧可产出；serving 侧（sglang-Ascend Glm5Next 量化权重加载）仍是独立的待验证项。
