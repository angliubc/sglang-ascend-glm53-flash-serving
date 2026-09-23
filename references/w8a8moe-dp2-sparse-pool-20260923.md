# W8A8-MoE 件 + sparse-layer 池 + DP2×TP16 1M ctx（2026-09-21/23）

0918 W4A8 定版之后的两代演进：v9 experts-only int8 件上线，随后 sparse-layer 池补丁把 KV 死权重砍掉，DP2×TP16 双机拉到模型原生 1M context。

## W8A8-MoE 件（v9，2026-09-18~21 量产）

- **配方**：`templates/quant_v9_w8a8_moe.py` = v8 verbatim 换 `w_bit 4→8`。仅 language-model 路由专家（42 层 × 288 expert × gate/up/down = 36,288 张量）量化 int8 dense per-channel；注意力/KDA 投影/dense MLP/shared expert/router/lm_head 全留 BF16（KDA 门控精度敏感教训）
- **为何弃 W4A8 改 W8A8**：W4A8 多步工具调用有 int4 16 离散值精度退化（0918 A/B 已见 cc@1.0 10% err）；W8A8 无此问题且权重 302GB（vs BF16 642GB / W4A8 更小）在 DP2×TP16 下 21.71GB/卡，显存账成立
- **dtype 构成**（quant_model_description.json）：W8A8_DYNAMIC 108,865 + FLOAT 1,774
- **运行验证**（0922-23 三天生产 + 0923 定档测试）：17×23=391 ✓、41 万 token 暗号检索逐字 PASS、EAGLE accept len 3.4-4.5、HCCL_DETERMINISTIC=true 下无算术漂移
- ⚠️ v9 run 日志止于 unfuse 40/42（容器滚动覆盖），完整 POSTCHECK 记录未留存；产物正确性由 3 天生产 + 定档测试背书

## sparse-layer DSA 池补丁（2026-09-23，本 repo 的核心增量）

**发现**：fork `_build_dsa_kv_pool` 用 `layer_info.num_effective_layers`（=45 全层）建 `DSATokenToKVPool`，但 glm5_next 只有 11 层 DSA（[3,7,...,43]）需要 KV latent+index——34 层 KDA 状态全在 mamba 池，`_transfer_full_attention_id` 对非 DSA 层 id 直接 raise，**76% 池字节是永远无人读写的死重量**（51.8KB→12.4KB/token）。W4A8 时代同样带着此开销（权重小掩盖了它）。

**补丁**（两节点 `/data/models/mtp_experiment_0907/sglang/` 同步）：
1. `scripts/sparse_layer_dsa_pool.py` → `srt/mem_cache/sparse_layer_dsa_pool.py`：`SparseLayerDSATokenToKVPool(DSATokenToKVPool)` 只实体化 11 层，公共入口做全局层 id→dense id 翻译
2. `scripts/patch_sparse_dsa_pool.py`：`kv_cache_configurator.py` 接线 patch（`_build_dsa_kv_pool` else 分支；backup `.bak-sparse-0923`）

**可行性依据**：所有外部访问（ascend_backend 25 处 get_key/value_buffer、dsa_indexer 系 get_index_k_with_scale_buffer）都走方法入口；`k_buffer[i]` 直访仅 DSV4 路径（glm5_next 不走）。

**两签名陷阱**（踩过）：
- 子类 `__init__` 必须 `kwargs.pop("layer_num")`——configurator 显式传了，直接叠加会 `TypeError: multiple values`
- `set_kv_buffer` 是 **MLA 签名** `(layer, loc_info, cache_k, cache_v, layer_id_override=None)`，不是 MHA 的 k_scale/v_scale——EAGLE draft 路径按位置传 7 参，签名错即 `TypeError: takes 5 to 6 positional but 7 given`

## DP2×TP16 拉满档（0923 定版）

```
--tp 16 --dp 2 --enable-dp-attention --enable-dp-attention-local-control-broadcast
--mem-fraction-static 0.95 --context-length 1048576 --max-total-tokens 1800000
（EAGLE D6/S5、page 64、mamba extra_buffer 等与 0918 相同）
```

**显存账**（每卡 60.49GB 可用）：权重 21.71 + draft 0.85 + mamba 2.11 + KV 池 21.32（1.8M token）+ 图 ~8 = 54.5，**运行时余量 4.97GB**

**2M 档死因（勿再试）**：`mem-frac 0.95` 下池自适应到 1,916,544，余 3.09GB——60 万 token 冷 prefill 的 aclnnCat 算子 workspace 打爆（CANN 207001，forward 流内 fatal 无恢复路径，进程必退）。**图内存随池 token 寻址空间增长（700K 档 4.14GB → 1.9M 档 7.97GB），不是常量**。拉满公式 = 运行时余量 ≥5GB，池每 +100K 吃 ~1.36GB 静态 + 图增量。

**验证**（1.8M 档，用打死 2M 档的同一 60 万 token 负载）：
- 41 万 token 暗号逐字抄写 PASS + 数字追问 PASS（209s/206s）
- accept len 3.4-4.5（基线区间）、网关端到端 42 ✓
- 17.8 万 token 真实 Claude Code 会话长跑：radix 每轮只 prefill 新增 1-3K token，accept 3.8-4.4

## 运维

- serve：`templates/serve-w8a8moe-dp2-node.sh <rank> [ctx] [tokens]`（默认 1M/1.8M；回滚 `500000 500000`）
- 重建容器：`templates/recreate-w8a8moe-700k.sh <rank> [ctx] [tokens]`（.34 worker 先、.33 master 后）
- 日志：容器内 `/serve.log`（⚠️ `docker exec -d` 必须 `> /serve.log`，裸跑 stdout 被丢弃，崩溃无日志）
- 正确性验证：`scripts/verify_ctx700k.py <api>`（短算术 + 41 万 token 暗号检索 + radix 二次检索；暗号 max_tokens 给 8192——512 会被 thinking 烧满出空 content，是老病不是池错位）

## 调试弯路（勿重复）

- transformers 自带 `Glm5NextTextConfig` 无 `full_attention_layer_ids`；fork 属性只在 `_try_load_glm5_next_config`（`utils/hf_transformers/config.py`）替换 config 类后才有。容器外 AutoConfig 验证会误报 KeyError
- `init_unified_mamba_pools` 看着正确（`len(full_attention_layer_ids)`）但生产 dispatch 走不到它——`use_mla + is_dsa` 优先命中 `_build_dsa_kv_pool`。**判池构成只信启动日志 `KV Cache is allocated` 实测数字反推**
- OOM 后的 KV 损坏会以"暗号数字错一位"（7719→7791）形式出现，别误诊为池补丁 bug——先查 Error 日志有没有 207001

## 0923 下午事故：视觉塔 swiglu @torch.compile 运行时重编译 → 跨机死锁（已修复）

**现象**：服务运行 1h 后（07:35 UTC）全 rank 静默卡死，无任何 batch 日志，请求挂死或返回错误短响应；900s watchdog 16 rank 集体自杀，SIGQUIT 级联死透。

**根因链**（py-spy + watchdog 栈实锤）：
1. 首个带图像请求路由到 .34（DP1），vision tower `Glm5NextVisionMLP.forward` 首次走到 `swiglu_clamped`（`glm5_next.py:169`）
2. 该函数带 `@torch.compile`（:138）——`server_args.enable_torch_compile=False` 管不到模块私有装饰器；`--skip-server-warmup` 又保证启动期无请求预热，视觉塔新 shape 首次触发 dynamo 编译
3. torch_npu inductor `_make_launchers` 每核 `synchronize` 等 device 排空 ↔ device 队列里跨机 TP16 集合通信在等 .33（DP0）对齐 → 双向互等，硬死锁
4. .33 侧表象：卡在 `dsa_indexer_kpool.py:189`（npu_lightning_indexer op launch，队列堵满）——indexer 是**受害者不是凶手**，别修错靶

**修复**：`glm5_next.py` 的 swiglu_clamped 改为 `_swiglu_clamped_impl` + 条件编译（`_is_npu` 走 eager，GPU 才 compile）。.33 早在 09-15 就打过此补丁（当时因另一签名 `KeyError: s94+1 in detect_flattened_axis`），但只改了单节点副本——.34 仍是 @torch.compile 旧版，带图请求恰好分到 .34 引爆。

**教训（制度化）**：sglang 源码是每节点 bind 各自的 `/data/models/mtp_experiment_0907/sglang`，无共享真值——**改源码必须两节点同改 + md5 对齐**（当前 `glm5_next.py = 79a1d3e1191a80f3246e2b5cab578f55`）。现役脚本唯一入口已归档 `/command/sglang-flash/recreate-w8a8moe-700k.sh`（旧版全部移入 `/command/old/sglang-flash-variants/`）。

**判死要领**：跨机部署下"单节点 op 卡住"栈（indexer/通信 op launch）优先怀疑对端在编译或 GC——先 py-spy 对端，别急着改本端算子。
