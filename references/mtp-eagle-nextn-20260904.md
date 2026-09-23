# GLM-5.3 MTP (EAGLE/nextn) bring-up on 910B — 七层崩溃链 + verify 路径改道 (2026-09-04)

node1/node2 TP16, target=W8A8-xpinke (int8 weight-only, KV bf16, page_size=64),
draft=BF16 L45 反量化 (`/common/glm53_flash_mtp_bf16`, 14.87GB, 889 tensors)。
**结论: 跑通。1.4-1.5 → 2.8-3.5 tok/s (2.2x), accept len 2.5/4 (rate 0.43→0.54 爬升), 质量 4/4 连贯, 零调度器异常。**
bind mount 同 porting skill 的 overlay 体系; **node1=/data/models/sglang_full, node2=/local_data/models/sglang_full 路径不同**。

## 启动配方 (v5-final, /common/glm53_w8a8/run-node{0,1}-graph.sh)

```
--speculative-algorithm EAGLE \
--speculative-draft-model-path /common/glm53_flash_mtp_bf16 \
--speculative-draft-model-quantization unquant \
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```
(基座参数沿 v3: --disable-cuda-graph --context-length 32768 --max-total-tokens 131072 --page-size 64 --disable-radix-cache --watchdog-timeout 900)

- 此 build spec_info 枚举**无 NEXTN** → 传 `EAGLE`; draft arch 按 `num_nextn_predict_layers` 自动切 NextN 类 (Glm5NextForConditionalGenerationNextN, 89 行, 不替换 decoder=DeepseekV2DecoderLayer(is_nextn=True))
- draft 量化继承 target 的 modelslim → 加载器找 quant 描述文件失败 → 必须 `--speculative-draft-model-quantization unquant`
- draft config (text_config) 必须设 `index_share_for_mtp_iteration=false` (见坑④)
- draft 放 /common NFS 双节点同路径免拷贝

## draft 从何而来 ("开箱不可用"根因)

原版 GLM-5.3-Flash 是 **FP8(E4M3) 128×128 块量化发布版** (experts F8+scale_inv[16,32], 头部 BF16;
safetensors 头非标须手动 data_offsets 解析)。W8A8-xpinke 量化**按设计丢弃 layers.45 (MTP 层
1760 张量, 自带 MoE experts)** — SGLang 代码端 nextn 支持齐全 (load_weights is_nextn=, nextn 重映射
model.layers.45→model.decoder.*), 纯权重缺失。提取: `bf16 = fp8 × scale_inv` (128×128 块乘,
L3 cos=0.9993), 去 quantization_config, 保原始命名。
⚠️ node1 原版副本 per-expert 格式, node3 融合 expert 格式 — 扫描/反量化一律用 node1 版。

## 七层崩溃洋葱 (每层: 症状→根因→修复)

1. **rotary_dim=0 撞 triton** — GLM `qk_rope_head_dim=0` 设计值 (DSA APE), draft 继承 DeepSeek
   路径无守卫 → `tl.arange(0, D_ROPE//2)` 编译炸。
   修: rotary_embedding/base.py fused 分支加 `and self.rotary_dim > 0` 守卫。
2. **native 空 reshape** — `view(16, -1, 0)` 空 rope 段歧义错。
   修: forward_native/forward_npu 入口对 `rotary_dim == 0` 短路返回 (恒等)。
3. **KV pool 空 scatter** — NPU MLA pool 把 v_buffer 复用为 rope 段缓存 (DeepSeek 惯例), GLM
   rope 宽 0 → `v_buffer[...].view(-1,1,0)` 炸。
   修: memory_pool_npu.py npu_scatter_nd_update_ 的 v_buffer 写加 0 宽守卫 (读路径前人已有)。
4. **DSA seed 全 -1 → aclnnSparseFlashAttention 561002** (8 rank 全灭, 最深一层):
   - 链: `base_attn_backend.get_indexer_metadata()` 基类默认 `return None` ("don't support
     indexer"), **ascend backend 未 override** → `IndexerKPool.forward_npu` 只是
     `return self.forward_cuda(...)` → forward_cuda 入口 `metadata is None → return None`
   - 即 **NPU 上所有 indexer 调用永远返回 None → 主模型 DSA 从未启用, 一直全注意力退化跑**
     (质量正常但慢 — 速度根因之一, 与 KDA 移植层并列)
   - draft 的 seed 机制 (index_share_for_mtp_iteration) 要 indexer 返回 topk → 拿不到 →
     seed buffer 分配了没人填 (全 -1) → draft decode -1 索引越界
   - 修: draft config `index_share_for_mtp_iteration=false` 关 seed → draft 与主模型一致走
     dense DSA (正确性由 target verify 保证, 接受率略降)
   - 排障路径备忘: nextn publish 探针 (deepseek_nextn.py) 显示 decoder 返回 topk=None;
     mla_npu prepare 探针显示全模型 11 个 DSA 层 indexer 全返回 None → 定位 KPool 转发链
5. **verify 撞 CANN blockSize** — aclnnFusedInferAttentionScoreV3: "In no quant GQA (QS > 1)
   scenario, when page attention enable, blockSize(64) should be a multiple of 128, and
   should be in range of [128, 1024]"。试 `--page-size 128` → GLM KV pool 硬 assert
   `page_size == 64` (memory_pool.py:4544) 堵死 (启动即崩, 勿再试)。
6. **换算子全灭** — npu_fused_infer_attention_score_v2 内部调 V4 → 同 128 限制; V1 非分页
   TND 只支持 headDim 64/128/192 (MLA latent 512 不行, CheckFeatureLayout); sparse_mode=3
   还必须传 atten_mask (CheckFeatureMask)。
7. **终极修复 (v4v→v5)** — TARGET_VERIFY + DRAFT_EXTEND_V2 改路由到新方法
   `forward_verify_paged_mla` (ascend_backend.py, 插在 forward_mtp 前 ~74 行):
   **每个 draft 位置 j 跑一次 `_npu_paged_attention_mla`** (decode 实证配方, page 64 可跑,
   latent 512 可跑), `context_lens = seq_len - ndt + j + 1` (draft j attend prefix+j+1 —
   数学上精确等价 verify 因果语义)。KV 先 set_kv_buffer 写 draft tokens 再读全 cache。
   - `context_lens` 必须是 **CPU int32 tensor** (对齐 decode 的
     `seq_lens_cpu_int = seq_lens.cpu().int()`); 传 list 报 "Expected a value of type
     'Tensor'", 传 int64 报 "PagedAttentionOperation setup failed" (C++ 帧无信息)
   - 路由: forward_extend 中 `is_target_verify() or is_draft_extend_v2()` → 新方法
     (原 forward_mtp 保留但 GLM 不可达; 其 gather+V1 改写对 headDim 512 不可用)
   - DRAFT_EXTEND_V2 也要走此路 — decode 轮间的 draft extend 原本也路由 forward_mtp
     (首请求 prefill→draft extend→draft decode→verify 全过后, 崩在第二轮 _draft_extend_for_decode)

## CANN 算子约束速查 (此 port 实测, 通用)

| 场景 | 算子 | 约束 |
|------|------|------|
| no-quant GQA + paged (block_table) | FIA V3/V4 (V1/V2 内部同) | blockSize 必须 128 倍数 ∈[128,1024] |
| no-quant 无 rope + TND 非分页 | V1 | headDim 仅 64/128/192 (MLA latent 512 不行) |
| sparse_mode=3 | 所有 FIA | 必须传 atten_mask (非 null) |
| page_size=128 | — | GLM KV pool 硬 assert ==64 |
| decode 单 token | _npu_paged_attention_mla | page 64 + latent 512 均可跑 |

## NPU 排障技巧 (本 session 新增, 通用)

- **异步报错陷阱**: NPU 算子异步执行, Python traceback 指向**下一个 launch 的 host 代码**
  (PA-MLA 的错报在 MHC layernorm 除法行)。看 "The current working operator name is XXX"
  才是真凶; "XXXOperation setup failed" + 纯 C++ 帧 = 参数类型/shape 不匹配 → 对照同文件
  decode 路径已验证调用逐参比对 (dtype/device/layout)
- **env 门控探针** (`if os.environ.get("SGLANG_DEBUG_DSA"): logger.warning(...)`) 插桩,
  一轮重启拿实锤, 不猜; 成功后删 env 即零开销 (代码可留)
- 多阶段链路排障顺序: 先让 prefill 通 → draft extend → draft decode → verify → **第二轮
  draft extend v2** (每轮一个新崩溃, 逐层剥; verify 通≠全通)

## 已知残留

- **主模型 DSA indexer NPU 未实现** → 全注意力退化 (速度根因之一) — 若实现
  get_indexer_metadata + KPool NPU 原生路径, 主模型可恢复 sparse; draft 也可重开 seed
- 接受率 ~0.5 (非理想): draft dense DSA 与训练分布偏差 + bf16 draft vs W8A8 target; 可调
  steps/topk
- **DeepSeek 等 quant-KV 模型勿在此树上跑 MTP verify** (路由已改成 PA-MLA, 会读错
  quant cache — forward_mtp 原 V3 路径才支持 antiquant)
- graph 模式仍被 KDA NaN 堵 (另一线); MTP 现役 eager+overlap

## 补丁文件清单 (双机同步, /tmp 母本)

- `layers/rotary_embedding/base.py` — rotary_dim 守卫+短路 (node1 备份 base.py.bak.0904)
- `hardware_backend/npu/memory_pool_npu.py` — v_buffer 0 宽写守卫
- `hardware_backend/npu/attention/ascend_backend.py` — **forward_verify_paged_mla 新方法 +
  verify/draft_extend_v2 路由** (核心) + env 门控调试日志
- `hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` — is_nextn forward_npu
  分支 (KPool 下等效无效, 保留) + PREP/CORE 探针
- `models/deepseek_nextn.py` / `speculative/eagle_worker_v2.py` — publish/seed 探针
- draft: `/common/glm53_flash_mtp_bf16/` (config 已关 index_share_for_mtp_iteration)
