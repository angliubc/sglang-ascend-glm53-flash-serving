# GLM-5.3-Flash v7 解融合自量化：fused MoE → 逐专家 Linear（2026-09-01）

根因与解法：msmodelslim（26.1.0）跳过融合专家是因为权重是 **nn.Parameter 不是 nn.Linear**。
不需要写适配器/EP——加载后把融合模块换装成逐专家 nn.Linear 的 ModuleList，rewriter 自动
wrap，saver 输出键名与 BF16 checkpoint / xpinke 件完全一致，**无需后处理再融合**。

> 状态（0903 CPU 验证完结）：v7 等价性失败**已定性为阈值假阳性，换装逻辑完全正确**。
> 探针 quant_v7_verify.py（真实权重 L3/10/44，容器 glm53-quant CPU）：重建 fused（[gate;up]，
> 与 `@use_experts_implementation` 默认 `is_concatenated=True` 互证）vs UnfusedExperts——
> **bf16 bit-exact（diff=0）×3 层、fp32 精确（0）×3 层、matmul 分块噪声底 0**；错序对照
> （up_first）7.35/6.91/42.13 = 真 bug 量级参考。v7 当年 diff=0.0625 的来源：transformers
> 加载模块经 `use_experts_implementation` **forward 被 dispatch 到批量实现**，与逐专家循环差
> 1-2 ULP（装饰器会替换 forward，inspect 实锤）。修复清单：①阈值 1e-2→0.25（舍入 ~0.06 与
> 真 bug ~7 之间取整）②`assert n_moe==43`→**42**（主模型 L3-44；L45=MTP 头 889 张量，官方配方
> 本就 exclude MTP，检测器漏掉它反而是正确行为）③EXPECTED_EXPERT_KEYS 37152→**36288**。
> checkpoint 事实：专家层=43（L3..L45）=主模型 42 + MTP 1；H=4096 I=2048 E=288 swiglu_limit=10。
> 脚本：node3 `/data/models/scripts/quant_v7_experts.py`；日志 `/data/models/quant_v7.log`；
> 验证探针 `/data/models/quant_v7_verify.py` + `quant_v7_verify.log`。
> 另：msmodelslim **CLI** `--quant_type QuantType.W4A8` 会 argparse 秒挂（enum choices vs str，
> 同事 0903 05:14 实锤 quant_w4a8.log）——走 Python API（QuantConfig）绕过。

## 模型结构事实（探测自 BF16 index.json + xpinke 分片头）

- 45 层：L0-2 稠密 MLP（inter 12288），L3-44 MoE；hidden 4096，moe_inter 2048，top-8
- 每层 **288 个路由专家** × 3 权重（gate/up [2048,4096]，down [4096,2048]）；主模型 L3-44 = **42 层 MoE**（42×288×3 = 36,288 专家张量 = 量化目标）；checkpoint 全部专家层 = 43（L3..L45，含 L45=MTP 头）= 37,152 张量——**MTP 按官方配方排除，勿量化**（0903 index 探针实锤）
- checkpoint 本身是**逐专家键名** `mlp.experts.N.{gate,up,down}_proj.weight`；transformers
  加载时 Glm5NextTextExperts 合并成内存融合参数 gate_up_proj [E,2i,h] / down_proj [E,h,i]
- shared_experts = 普通 Glm5NextTextMLP（nn.Linear，v6 已正常量化）；router gate 留 BF16
- `@use_experts_implementation` 装饰器只是给类挂 vllm 风格 dispatch 元数据，CPU reference
  forward 不受影响 → **加载后整模块替换**（而非类打补丁）最稳

## 解融合配方五要点

1. **UnfusedExperts 继承 nn.ModuleList**（不是"含 self.experts=ModuleList 属性的普通 Module"）：
   ModuleList 子模块名就是 '0','1',... → state_dict 键 = `mlp.experts.N.gate_proj.weight` ✓。
   写成普通 Module 属性则键变 `experts.experts.N.*` ✗（saver 产物键名错位）
2. 权重切片：gate=gu[e,:i,:]，up=gu[e,i:,:]，down=dn[e]；bf16 下直接 `.data` 赋值
3. forward 复刻原 reference loop（one_hot/mask/hit/index_add，照抄 Glm5NextTextExperts.forward）；
   gate_up 单线性 vs gate/up 两线性数学等价
4. **等价性自检**（换装前，首/中/末 3 个 MoE 层）：随机 hidden[256,4096] + 随机 top-8 路由，
   新旧模块输出 max_abs_diff < **0.25** 才继续（⚠️1e-2 太紧：对照基准的 transformers forward 被
   `@use_experts_implementation` 派发到批量实现，与逐专家循环天然差 1-2 ULP ~0.06；真 bug 量级
   7-42——0903 探针四判别实锤，见 `scripts/unfuse_equiv_probe.py`）；再全模型 smoke forward 1 条校准样本
5. 量化配置与 v6 全同（W8A8 / anti m2 / act_method=2 / mm_tensor=False / disable 全部非语言
   Linear），唯一变量 = experts 进管线 → 与 v6 可严格对比

## 内存坑（1TB RAM 机器 + 643GB bf16 模型）

- `list(model.named_modules())` 快照持有全部模块引用 → 换装后融合参数不释放，峰值
  ≈ 模型 + 全部解融合副本 ≈ 1.26TB → **OOM**。必须只收集名字列表，逐个 `get_submodule`
- 换装后立即 `module.gate_up_proj.data = torch.empty(0)` 释放；逐层瞬时峰值 +~29GB，可控
- NPU 被同事任务占满时 `dev_type=cpu` 是唯一路径；加载 643GB 约 40min（容器内 CPU bf16）

## 输出格式对照（自量化 26.1.0 vs xpinke 件）

| 维度 | 自量化 (msmodelslim 26.1.0) | xpinke/glm-5.3-flash-w8a8 |
|---|---|---|
| 专家张量键名 | `mlp.experts.N.*.weight` I8 + scale/offset（同布局） | 同 |
| weight_scale/offset | BF16 [out,1] | F32 [out,1] |
| 激活量化 | static：input_scale BF16[1] + deq_scale F32[out] + quant_bias I32[out] | dynamic：不存 input_scale |
| serving 路径 | 与已上线 GLM-5.2 W4A8C8 同格式（vllm-ascend AscendV1 static） | dynamic 变体 |
| desc 文件名 | quant_model_description_w8a8.json | quant_model_description.json |
| desc 键名 | 待 v7 出件确认 | 短名 model.layers.*（疑为 vllm 侧名；tensor 本身是长名 model.language_model.layers.*） |

注意：xpinke 的 desc（152KB）**不含逐专家条目**（loader 按命名约定推断；其 w4a8 仓另有 10MB
.perexp.backup 全量版）；26.1.0 saver 会写全量 desc。两种件引擎加载兼容性到货后实测，
质量对比以评测分数为准（用户指示：自量化 vs 下载件同批评测对比）。

## v8：直接上专家-only W4A8（0903 17:14 点火）

配方 = v7 + 五处修正（用户拍板跳过 W8A8 中间件）。脚本模板 `templates/quant_experts_w4a8.py`
（部署件 node3 `/data/models/scripts/quant_v8_w4a8.py`，日志 `/data/models/quant_v8_w4a8.log`）：

- **量化范围收窄为仅专家**：36,288 个专家线性层（L3-44 每专家 gate/up/down）int4；attention/
  KDA 投影/dense MLP/共享专家/router/lm_head 全保 BF16。理由：专家 ~305B/320B 参数=收益主体，
  KDA 门控精度敏感（beta sigmoid 教训）。disable_names 反转：disable 一切 Linear **除了**
  `name.startswith("model.language_model") and ".mlp.experts." in name`，并断言 enable 数==36288
- **`AntiOutlierConfig` 加 `disable_anti_names=disable_names`**（v7 未用此参数）——anti_outlier
  m2 不再扰动保 BF16 的模块
- `QuantConfig(w_bit=4, a_bit=8, act_method=2, mm_tensor=False)` = 包内 docstring 官方 W4A8 用法；
  **is_lowbit 必须保持 False**——那是稀疏量化开关（config_utils.py:123
  `is_sparse_quant = co_sparse or (is_lowbit and w_bit==4)`）；group_size=64 默认=分组 int4 scale
- 断言修正：n_moe==42、EXPECTED_EXPERT_KEYS==36288、等价阈值 1e-2→0.25
- 产物 `/data/models/GLM-5.3-Flash-W4A8-v8`（预计 ~100-170GB）、8-14h CPU（glm53-quant/mslim26）
- 验收门（POSTCHECK）：`expert_quantized==36288 且 expert_left_BF16==0` → `.quant_v8_ok` 标记
- 监控模式：风险时刻=校准器启动（W4 CPU 路径首跑，加载后 ~1.5-2h）——一次性 cron 巡检
  （ISO 时刻 + 只读 + 报阶段/崩溃行）
- 出件后链路：dtype 审计（`scripts/safetensors_dtype_audit.py`）→ serving 侧 W4 加载实测
  （glm5_next+W4 无人跑过；xpinke w4a8 168GB 下载件=并行保底）
- **首程里程碑实况（0903 晚巡检）**：加载 28min 完成 → 解融合探测 **n_moe=42/42 精确命中**（L45 MTP
  排除断言在真实产物上验证）→ 等价性自检三连**全绿**：L3=0.0625、L10=0.078125 —— 与 v7 探针预测的
  1-2 ULP 噪声量级逐位吻合（v7 的 1e-2 阈值就是死在这两个数上；0.25 阈值修复在产线上验证有效）；
  资源 55 核/735GB RSS。阶段梯：加载(+28min) → unfuse(逐层) → anti_outlier → W4A8 校准(最长) → 保存

## 等价性探针（方法沉淀，可复用）

`scripts/unfuse_equiv_probe.py`：不加载全模型，按 index.json 定点读目标层专家张量（页缓存热
~1s/片），重建 fused（[gate;up]）vs UnfusedExperts 同种子四判别：bf16 正序（应=0）/ bf16 错序
（~7-42 = 真 bug 量级参考）/ fp32 正序（=0 即数学等价铁证）/ matmul 分块噪声底。适用于任何
fused→unfused 换装等价性验证（改 regex 与 forward 参考循环即可适配他架构）。核心教训：
**对照基准若来自 transformers 加载的真实模块，其 forward 可能被 `@use_experts_implementation`
派发到批量实现（与逐专家循环差 1-2 ULP）——等价阈值必须按"真 bug 量级 vs ULP 噪声"定宽
（0.25），不能拍脑袋 1e-2**。

## 相关

- 事故链与 xpinke 件细节：`glm53-flash-quant.md`
- 下载→转发管线（.verified + 增量校验守护进程）：model-distribution skill
