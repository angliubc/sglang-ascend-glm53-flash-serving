---
name: sglang-ascend-glm53-flash-serving
description: GLM-5.3-Flash (glm5_next: hybrid KDA linear + DSA sparse attention, 288-expert MoE) on Ascend 910B with SGLang — full chain from source port (glm5_next absent from every Ascend image) through msmodelslim W8A8/W4A8 quantization of fused-MoE experts, serving bring-up, MTP/EAGLE speculative decoding, kpool indexer NPU port, sparse-layer DSA KV pool (76% dead weight cut), to the converged DP2×TP16 dual-node 1M-context production config. When porting a brand-new model arch to 910B, quantizing fused-MoE models with msmodelslim, debugging NPU serving (207001 misreports, KV pool walls, triton cache races, NoPE guards), or deploying GLM-5.3-Flash on A2 hardware.
version: 1.1.0
license: Apache-2.0
metadata:
  hardware: "Atlas 800T A2 (Kunpeng aarch64), 8 × Ascend 910B 64GB per node, up to 4 nodes"
  software: "quay.io/ascend/sglang main-cann9.0.0-910b daily (sglang 0.5.19.dev, CANN 9.0.0) + glm5_next overlay; msmodelslim 26.1.0"
  weights: "GLM-5.3-Flash BF16 (642GB) → W8A8-MoE 302GB (v9 experts-only int8, 0923 现役) / modelslim_v1 yaml 296GB / experts-only W4A8"
  measured: "2026-08/09, port 2026-08-30 → TP32 converged 2026-09-08, W4A8 tuned 2026-09-18, W8A8+sparse-pool 1M-ctx 2026-09-23"
---

# GLM-5.3-Flash (glm5_next) on Ascend 910B — SGLang port/quant/serve skill

## 事实卡（一眼定夺，全部实测）

| 项 | 值 |
|---|---|
| 模型 | `glm5_next` = `Glm5NextForConditionalGeneration`（VLM）；320B 总参 / 18B 激活；288 路由专家+1 共享；**45 层 = 34 KDA 线性注意力 + 11 DSA 稀疏**；MLA（kv_lora 512）；MTP = layer 45；原生 1M ctx；vocab 154820 |
| 上游支持 | **零**：sglang/vllm main、全部 quay ascend/sglang tag（stable v0.5.16 → daily）均无 glm5_next；唯一参考实现 = `lmsysorg/sglang:glm-5.3-flash`（CUDA fork） |
| 移植量 | Python 级 overlay：**8 整文件拷贝 + 7 共享文件补丁 + 追加符号**——daily 镜像已带全套 KDA NPU kernel（`sgl_kernel_npu.fla.*` + `AscendKDAAttnBackend`），真正的 kernel 缺口只有 kpool indexer 一个 |
| 量化 | 910B 无 FP8 → 源必须 BF16 642GB。llm_ptq **静默跳过融合专家**（578GB 假量化）；可用路线 = modelslim_v1 yaml（296GB，专家全 int8）或 v8/v9 解融合（仅专家 W4A8/int8）。产物必须过 header-only dtype 审计 |
| 生产拓扑（0923 定版） | **W8A8-MoE 件（v9 experts-only int8, 302GB）DP2×TP16 双机；ctx 1,048,576（模型原生 1M）；KV 池 1.8M token（21.32GB/卡）；运行时余 4.97GB** |
| 0918→0923 演进 | W4A8 TP16（700K 池）→ W8A8 v9 件 + **sparse-layer 池补丁（45 层→11 层 DSA 实体化，51.8→12.4KB/tok，砍掉 76% 死权重）** → DP2×TP16 1M ctx 拉满 |
| MTP | EAGLE 2.2×（1.5→2.8-3.5 tok/s 起步，调优后 36.5）；draft = BF16 layer-45（从 FP8 主仓 `bf16 = fp8 × scale_inv` 反量化），draft config 关 `index_share_for_mtp_iteration` |
| READY | 权重加载 145s（74 片 W8A8）+ 内存池/tree-cache 静默 ~9min + 图捕获；冷启动全程 8-13min 属正常 |

## 目录

| 路径 | 内容 |
|---|---|
| `scripts/sparse_layer_dsa_pool.py` | **sparse-layer DSA KV 池**（glm5_next 45 层只有 11 层 DSA 需要 KV；砍 76% 死权重，KV 51.8→12.4KB/tok）→ 放 `srt/mem_cache/` |
| `scripts/patch_sparse_dsa_pool.py` | 上述池的 `kv_cache_configurator.py` 接线 patch（锚点唯一断言 + ast.parse 门禁） |
| `scripts/verify_ctx700k.py` | 池补丁正确性验证（短算术 + 41 万 token 暗号检索 + radix 二次检索；暗号 max_tokens 给 8192 防 thinking 烧预算假阴性） |
| `scripts/glm5_next_port_builder.py` | 移植 overlay 构建器（锚点唯一断言 + py_compile 门禁；从两棵源树重建 port 树） |
| `scripts/precompile_triton_npu.py` | 单进程 triton 预编译（杀多 rank 缓存写竞争） |
| `scripts/verify_quant_output.py` | 量化产物 dtype 覆盖审计（只读 safetensors 头，秒级，exit 1 门禁） |
| `scripts/unfuse_equiv_probe.py` | 融合→解融合专家换装等价性四判别探针 |
| `scripts/e2e_longctx_battery.py` | 长上下文 E2E 验证组（needle/算术/连贯性） |
| `scripts/bench_decode.py` | SSE 步时探针（median gap=步时，tokens/event≈accept len） |
| `templates/smoke_tp16_dummy.sh` | TP16 dummy 权重冒烟（全栈验证无需真权重；每个 flag 都是 load-bearing） |
| `templates/serve_real_tp16.sh` | 真权重 TP16 启动器（0918 W4A8 档；0923 生产用 serve-w8a8moe-dp2-node.sh） |
| `templates/quant_naive_w8a8_v5.py` | llm_ptq 纯 CPU 量化配方（v5：真实路径 OUT + 设备自适应 calib；**注意 llm_ptq 跳专家**——用 yaml 或 v8/v9） |
| `templates/quant_v9_w8a8_moe.py` | **v9 生产件配方（0923 现役 W8A8-MoE 302GB）**：v8 换 w_bit 8；仅 42 层路由专家 int8（36,288 张量），KDA/注意力/router/lm_head 留 BF16 |
| `templates/serve-w8a8moe-dp2-node.sh` | **0923 生产 serve**（DP2×TP16 双机，ctx 1M / 池 1.8M / mem-frac 0.95；env 全量照抄含 HCCL_DETERMINISTIC=true） |
| `templates/recreate-w8a8moe-700k.sh` | 容器重建编排（rm+run+serve；worker 先 master 后；serve 输出必须 `> /serve.log`） |
| `templates/quant_experts_w4a8.py` | v8 解融合仅专家 W4A8（含等价性自检 + POSTCHECK 门禁） |
| `templates/glm53_flash_w8a8.yaml` | modelslim_v1 声明式 yaml（外部验证可出专家全 int8 296GB 件） |
| `references/w8a8moe-dp2-sparse-pool-20260923.md` | 0923 定版：v9 件 + sparse-layer 池 + DP2×TP16 1M ctx 拉满（含 2M 桠死因与降档公式） |
| `references/` | 深度文档：移植 / NPU 冒烟 8 轮 / serving bring-up / kpool / triton 竞争 / KV 池墙 / 量化战役 / MTP 崩溃链 / 生产定版 |

## 快速路径（从零到服务）

```bash
# 0) 移植：两棵源树 → port overlay → 镜像（⚠️ 910B 是 aarch64，基底必须 arm64）
docker pull --platform linux/arm64 quay.io/ascend/sglang:main-cann9.0.0-910b
python3 scripts/glm5_next_port_builder.py     # 需先抽两棵树（见脚本 docstring）
# Dockerfile: FROM daily + COPY port/sglang/ /sgl-workspace/sglang/python/sglang/

# 1) 冒烟（TP16 双节点 dummy 权重，只需 config/tokenizer 小文件）
templates/smoke_tp16_dummy.sh <node-rank>     # 每个 flag 防的什么坑见文件头注释

# 2) 量化（BF16 → W8A8/W4A8），产物过审计
python3 scripts/verify_quant_output.py <quant-dir>   # EXIT=0 ≠ 可部署，BF16>40% 即废

# 3) 生产（0923 定版 = W8A8-MoE DP2×TP16 双机 1M ctx，见 references/w8a8moe-dp2-sparse-pool-20260923.md）
# 必备补丁：sparse_layer_dsa_pool.py + patch_sparse_dsa_pool.py（KV 池砍 76% 死权重）
#         + modelslim.py get_moe_scheme 融合键回退（层级单键 mlp.experts.weight）
# serve:  templates/serve-w8a8moe-dp2-node.sh <rank> [ctx] [tokens]
```

## 核心结论（决策时直接引用）

1. **新架构移植先查 kernel 库再估工作量**。Ascend daily 镜像先落 NPU kernel、后落模型代码——glm5_next 移植时 KDA 全套 kernel 已在（fork 的模型文件是纯 Python，整文件拷贝即用）；唯一 HARD GAP 是 kpool indexer（lms fork 硬依赖 triton/DeepGEMM）。结论：**Python-level overlay 而非 kernel 移植**，1 天完成 95%。
2. **kpool 缺口的闭合 = 语义映射而非逐 kernel 重写**：`group_topk = topk//pool` ≤512 直接走 sglang 原生融合 kernel；daily dsv4 的 `fp8_paged_mqa_logits_torch` 与 DeepGEMM 调用点接口相同，drop-in；其余 triton 用复刻舍入语义的 torch 版。CPU 13 测试先行，再上 NPU。
3. **serving 期 indexer 死亡是隐形退化不是崩溃**：ascend backend 不 override `get_indexer_metadata`（基类返回 None）→ `IndexerKPool.forward_npu` 委托 `forward_cuda` 在 metadata 检查处退出 → **DSA 层全跑全注意力**（正确但慢，还饿死 MTP seed）。修复 = token 级近似（影子 bf16 index-K cache + `npu_lightning_indexer` + per-query tail 去重），hours 级；≤2048 窗口 = 精确训练语义。
4. **msmodelslim llm_ptq 只包 nn.Linear → 融合专家静默 BF16**。GLM-5.3-Flash 实测：EXIT=0、529 个小 Linear int8 (~7GB) + 专家 570GB BF16 = 578GB 假量化。**验收只认 dtype 审计**。可行路线：modelslim_v1 yaml（pattern 匹配能命中裸 3D Parameter）或解融合换装（ModuleList 子类 → 键名自动对齐 checkpoint，saver 输出即目标布局）。
5. **NPU MLA KV 池同时无视 `--context-length` 和 `--mem-fraction-static`**（此路径吃掉 post-weight avail 的 ~92%）→ **`--max-total-tokens` 是唯一有效的池上限**。不设 → prefill 瞬态只剩 ~1.3GB → OOM 被 CANN 误报成 `aclnnBatchMatMul 207001 "Binary get function by entry Failed"`。诊断铁律：同 op+shape 空闲探针容器能过、在跑服务里挂 → 查内存，别查二进制。
6. **CANN 异步栈指认旁观者**：报错的 op 是 device 死后撞到的下一个 sync 点。多节点 TP 挂起时**先查对端节点容器**——活着一侧的 py-spy 是无辜的 enqueue 行，死掉的对端才有真 traceback。`ASCEND_LAUNCH_BLOCKING=1`（诊断专用，~2× 慢）一发定案。
7. **NoPE（`qk_rope_head_dim=0`）bug 在每条新点亮的注意力路径上复发**：`npu_fused_infer_attention_score` 接受完全省略 rope kwargs 的 MLA 调用（数学上精确），但拒绝 0/1 维 rope 张量 → 14 处 `if qk_rope_head_dim > 0` 守卫。陷阱：NPU MLA KV 池把 packed buffer 当 rope 返回——NoPE 下 rope cache 必须置 None，不能 reshape-0。
8. **首请求三类假象**：① triton JIT 数分钟/个（py-spy 见 `linalg_to_bin_enable_npu_compile_A2_A3` = 编译非死锁；TRITON_CACHE_DIR 持久化 + 单进程预编译）；② 300s 默认 watchdog 杀健康首请求（`--watchdog-timeout 900`）；③ 无 decode 日志 ≠ 无 decode 进度（`decode_log_interval=40`）。里程碑梯：weights → pool → capture → fired-up → **forward 完成（py-spy 在 sampler）** → 首 token。
9. **EP 判决（0908 定案）：通用服务别用 EP**。tp32+EP32(deepep LL) prefill 被 256-chunk 硬限拖死 11×（431 vs 4327 tok/s）；deepep 只赢高并发 decode 吞吐。no-EP 全 TP 是 prefill/decode 均衡解。
10. **KV 池墙与死权重（0923 修订）**：三道墙数据（95 万图捕获 OOM / 88 万 HCCL / 74 万首请求 OOM）是 **45 层全配池**（49-52KB/tok）时代的。`sparse_layer_dsa_pool` 补丁后 12.4KB/tok：W8A8 权重 21.7GB/卡下 **池 1.8M + ctx 1M + mem-frac 0.95 + 运行时余量 ≥5GB = 拉满定版**（2M 桠 aclnnCat 207001 实锤——图内存随池容量增长非常量：700K 桠 4.14GB → 1.9M 桠 7.97GB）。池构造期 OOM 诊断法：`Tried to allocate X` ÷ token 数 = 单层 buffer 字节 → 反推已完成层数。
11. **长上下文 agent 工作流必须开 radix cache，且 KDA+EAGLE+page64 需要 `--mamba-radix-cache-strategy extra_buffer`**（普通 radix 在混合架构上崩；~1G/rank 开销）。90K 前缀命中后 TTFT <1s vs 全量重 prefill 25-40s。0923 实战：17.8 万 token 真实 agent 会话每轮只 prefill 新增 1-3K，accept 3.8-4.4。
12. **MTP/EAGLE 七层崩溃链每层一个修法**（NoPE rotary 守卫 → 空张量 view 短路 → KV 池 0 宽 scatter 守卫 → indexer-dead seed → FIA blockSize 128 约束 → V1/V2 全灭 → per-draft-position `_npu_paged_attention_mla` 改道，`context_lens` 必须 CPU int32）。draft 权重：量化件按设计丢 layer 45 → 从 FP8 主仓反量化提取（cos 0.9993）。
13. **吞吐对比必须用步频口径**（吞吐÷accept_len）——decode 吞吐被 accept len 波动污染（同配置 25-43 tok/s 摆动）；accept len 任务相关，A/B 必须同 prompt 同温度。
14. **混合 KDA 模型的 KV 池只该实体化 DSA 层**（0923 sparse-layer 补丁）：`_build_dsa_kv_pool` 用全层数建池时 76% 是 KDA 死 latent（KDA 状态在 mamba 池，`_transfer_full_attention_id` 挡写）。所有外部访问走方法入口（get_key/value_buffer、index_k 系）→ 子类翻译层 id 即可，patch 面收敛。两签名陷阱：`kwargs.pop("layer_num")`、`set_kv_buffer` 是 MLA 签名（layer_id_override 非 k_scale）。**算子 OOM（207001）是 forward 流内 fatal，进程必退无恢复——只有池级满载走 retraction**；OOM 后 KV 损坏会以"暗号数字错一位"形式出现，别误诊为补丁 bug。

## 分诊表（症状 → 第一检查）

| 症状 | 第一检查 |
|---|---|
| 容器内 `torch.npu.is_available()=False` | 启动 flag 缺 `--privileged --security-opt label=disable`（不是镜像问题） |
| 启动即 `assert page_size == 64` | 加 `--page-size 64`（DSA KV 池硬约束，非 bug） |
| `AttributeError: no attribute 'rope_theta'` | config-class swap：release transformers 原生认识 glm5_next → AutoConfig 返回原生类；patch `hf_transformers/config.py` 让 sglang 类先赢（参照 `_try_load_longcat_config`） |
| prefill 期 207001 | 十有八九是 OOM 误报 → grep 内存账本（`Load weight end/KV Cache is allocated/Capture … end`）对 HBM 算术；用 `--max-total-tokens` 压池 |
| 首请求挂起无日志 | py-spy scheduler：编译行=JIT；`.tolist()` D2H + AICore>100%=device 自旋；AICore~0%=死等对端 |
| 首 token 永不到达、无崩溃 | decode graph replay 捕获语义（kpool 池捕获时空、回放时满；-1 填充索引越界）→ `--disable-cuda-graph` 隔离定责 |
| 词沙拉（温度 0 亦然） | 换真权重后首次质量验证必须长生成+多 prompt（短输出连贯是假阴性）；`--debug-tensor-dump-layers` 逐层激活范数找坏层 |
| 工具调用漏进 content | 无 `--tool-call-parser`；glm5_next 原生格式 = `[think]…[/think]` + XML 风格 arg 标签 → `--reasoning-parser glm45 --tool-call-parser glm47` |
| `Glm5NextForConditionalGeneration is not supported for encoder disaggregation` | arg_groups/pd_disaggregation_hook.py 白名单补该 arch（启动期死，0 HBM） |
| 量化 EXIT=0 但产物≈原大 | dtype 审计（`scripts/verify_quant_output.py`）——llm_ptq 跳过融合专家 |
| mHC `NameError: deep_gemm` / F.linear 207001 | NPU 路由到 torch fallback（fused hc kernel 参数化不兼容，libopapi 无二进制）；fallback 必须 **chunked ~1024 token + bf16 GEMM**（16k prefill fp32 中间态 3-4GB） |
| 图捕获 OOM / 首请求 workspace OOM | KV 池超墙 → 降 `--max-total-tokens`（三道墙见结论 10） |
| 多节点 TP 挂起 | 先查对端容器（结论 6）；TP16 成对重启（worker 先 master 后，只重启一台 = 全下） |

## 何时不用这条路

- **只要 W8A8 起服务、不想自己移植**：社区现成件已存在（ModelScope `Eco-Tech/GLM-5.3-Flash-w8a8`，含完整 layer-45 MTP，333GB；vllm-ascend 官方 A2 教程已收录双节点 DP2×TP8 配方）。本仓库的价值 = 无现成件时的全链方法论 + SGLang 侧深度调优史。
- 910B 不支持 FP8/MXFP8——官方主仓 FP8 件（328GB）在此硬件不可加载，必须走 BF16→W8A8/W4A8。
- vllm-ascend 消费同一量化件时 desc 键格式不同（模块树前缀 + MoE per-expert 三件套），SGLang 能吃 ≠ vllm 能吃。

## License

Apache-2.0. Ported code derives from SGLang (Apache-2.0) sources: `lmsysorg/sglang:glm-5.3-flash` fork overlaid onto `quay.io/ascend/sglang:main-cann9.0.0-910b`.
