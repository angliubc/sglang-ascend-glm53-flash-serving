# GLM-5.3-Flash 910B 生产收敛：TP32 no-EP 定版（2026-09-08）与 0918 调优

从真权重上线（词沙拉定位、conv1d 批 bug、MTP 闭合）到四机生产定版的完整决策链。
本文件是部署侧终点；此前链路见各 reference（移植 → 冒烟 → 量化 → serving → MTP → kpool）。

## 定版参数（TP32 no-EP 四机，0908）

```
--model-path <W8A8-xpinke dir> --served-model-name glm53-flash
--trust-remote-code --tp 32 --nnodes 4 --node-rank <N> --dist-init-addr <master>:6300
--quantization modelslim --mem-fraction-static 0.92
--max-total-tokens 680000 --context-length 650000
--max-running-requests 32 --max-mamba-cache-size 32 --chunked-prefill-size 8192
--watchdog-timeout 900 --cuda-graph-max-bs-decode 32 --page-size 64
--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1
--speculative-num-draft-tokens 6 --speculative-draft-model-path <bf16-draft dir>
--speculative-draft-model-quantization unquant
--mamba-radix-cache-strategy extra_buffer
--reasoning-parser glm45 --tool-call-parser glm47
--skip-server-warmup --host 0.0.0.0 --port 80
```

要点：
- **`--restart no`**（Docker 默认）。带 `--restart unless-stopped` 的部署脚本会让
  docker stop 后守护进程自动重建容器，"停不掉"全因于此；watchdog cron 是另一层
  自动拉起，杀进程前一并清。
- **parser 必须显式给**：不带 `--tool-call-parser/--reasoning-parser` 时工具调用整体
  漏进 content（tool_calls=null）、思考不分离。glm5_next 原生格式 = `[think]…[/think]`
  + XML 风格 `[arg_key]…[/arg_value]` 标签（非 GLM-5.2 旧格式）。
- 容器 `--restart no` + 依赖补丁树 bind-mount（补丁随宿主树存活，docker rm 不丢，
  但换树重部署会静默丢——重部署前 `grep -rn <补丁标记> <树>` 验证）。

## 必备 serving 补丁（两枚，quant 件适配）

1. **`layers/quantization/modelslim/modelslim.py` `get_moe_scheme` 融合键回退**：
   W8A8-xpinke 件 routed experts 用层级单键 `model.layers.N.mlp.experts.weight`（值
   W8A8_DYNAMIC，42 层）；原代码只认逐专家键 `{prefix}.0.gate_proj.weight` 式 →
   ValueError。
2. **W4A8C8 类件**：`moe_methods.py` w2 scale_bias 按 TP 槽位预存，SGLang loader 的
   tp_size 恒 1 → 全列加载 + 全列求和 → 行并行 AR 后放大 tp 倍 → 输出乱码。修复 =
   从自身 int4 权重 + 原始 f32 scale 现算本 rank 输入切片的和（须在位打包**之前**读
   原始 f32 scale）。判别特征：W8A8/Dense 正常 + 首 token 即乱 + accept≈0。

## EP vs no-EP 判决（0908 实测定案）

| 配置 | decode 单流 | prefill 80K | 结论 |
|---|---|---|---|
| tp32 + EP32 (deepep LL) | 46.9 tok/s | **431 tok/s (186s)** | ❌ prefill 被 256-chunk 硬限拖死 11× |
| **tp32 no-EP（定版）** | 36.4 tok/s | **4327 tok/s (18s)** | ✅ 均衡 |
| tp16 no-EP 双机 | 43.3 tok/s | 4653 tok/s (17s) | decode +16% 但 KV 池仅 55 万 |

deepep low_latency 只赢高并发 decode 吞吐；**通用服务（prefill+decode 混合）别用 EP**。
另：投机+deepep 下 decode 图 bs 必须是 attn_tp(32) 的倍数，running/graph < 32 必死
`capture_bs=[]`。

## KV 池三道墙（68 万 token 池的由来）

GLM-5.3-Flash 混合架构 ~49-52KB/token（45 层全配 latent buffer——移植版
`layer_setup.py` 不按层类型过滤，DSA/MLA 池按全部 45 层分配）：

- 95 万 → 图捕获 OOM
- 88 万 → HCCL 惰性窗口分配失败（561000）
- 74 万 → 首请求算子 workspace OOM（207001）
- **68 万 + ~5G 余量 = 稳态**

池构造期 OOM 反推法：`Tried to allocate X` ÷ token 数 = 单层 buffer 字节；
（已分配 − 静态）÷ buffer = 已完成层数 → 对照 config `layer_types` 定位。

## Radix cache + KDA/mamba 混合架构（0908 关键修复）

长上下文 agent 工作流（90K+ token/轮）**必须开 radix cache**：
- `--disable-radix-cache` 下每轮全量重 prefill 90K（TTFT 25-40s）；多 agent 并发时
  decode 被 prefill 链饿死 → 上游 stale 检测判死掐流 → 重试 → 又一轮 90K → 雪崩
- 开 radix 后 90K 前缀命中，每轮只 prefill 新增 ~300 token，TTFT <1s

GLM-5.3-Flash（KDA + EAGLE + page_size=64）需要 **`--mamba-radix-cache-strategy
extra_buffer`**：普通 radix 在 KDA 混合架构上崩（mamba/KDA state 管理不兼容）；
~1G/rank 开销。启动日志确认：`Tree cache initialized: impl=UnifiedRadixCache
hybrid_ssm=True`。

## 0918 调优定版（W4A8 TP16 双机 + EAGLE）

单流 vs 并发是权衡轴（MTP 大 batch verify 反噬聚合吞吐）：

### EAGLE 投机参数 A/B（短中文 temp0.7，usage 口径）

| 配置 | decode | 步时 | accept len |
|---|---|---|---|
| no-spec | 19.1 tok/s | 52.5ms/步 | — |
| **D6/S5（steps5/draft6/topk1，定版）** | **36.5 tok/s** | 94ms/步 | 3.3-3.7 |
| D8/S7 | 31.0 tok/s | ~120ms/步 | 3.2-4.0（接受率 0.42 随链深衰减） |

- **94ms/步拆解**：verify ~70ms（75%，KDA 34 层 + W4A8 MoE kernel）+ draft ~15ms +
  跨机 AR ~8ms。verify 6-token 仅比单步贵 ~60%（KDA chunk 扫描对候选批量并行——
  "verify 6 倍代价"假设证伪）；D6/S5 已在收益曲线峰值，勿再加链。
- **draft 质量**：接受率 0.45-0.53；draft = 单层 BF16（1.15G，~3ms/步）——瓶颈是
  verify kernel 成本不是 draft。
- **吞吐对比必须用步频口径**（吞吐÷accept_len）：decode 吞吐被 accept len 波动污染
  （同配置 25-43 tok/s 摆动）；accept len 任务相关（创作 ~5.4-5.8 vs 推理 2.95-3.5），
  A/B 必须同 prompt 同温度。
- 单流解码在 16-rank NPU graph 端到端下 HCCL ring 调优（`HCCL_ALGO="level0:NA;
  level1:ring"`，引号必须）<0.5%，无害保留。

## 启动脚本纪律（此 fork）

- **整行删除/插入，勿 sed 注释**：`#` 行切断 `\` 续行链 → 其后参数静默丢失，症状
  = resolved ServerArgs 缺参 / port 回落默认。
- TP16/TP32 成对重启：**worker 先、master 后**（间隔 ~12-20s；全并行双发有时错过
  gloo rendezvous 窗口，集群不形成）；**绝不只重启一台**（死等 rendezvous 的对端
  挂死全部 collectives）。
- 改后 `bash -n` + `/proc/$(pgrep -f launch_server)/environ` 验证 env。

## 冷启动画像（W8A8 74 片）

权重加载 145s（20.06GB/rank）→ **内存池/tree-cache 阶段双节点静默 ~9min（零日志，
勿误判挂死）** → /health 200。全程 8-13min 正常；同配置重建 7m49s。

错误扫描纪律：健康容器 `grep -ciE 'error|traceback|oom'` = 56-76 全是良性 registry
探测行；`grep -v 'Ignore import error'` 后再判；新窗口 raw 计数先 `sort | uniq -c`
聚类核对全属已知良性才算零新增。

## 质量验证纪律（真权重）

- **短输出连贯是假阴性**：48-token 短生成连贯 ≠ 质量 PASS——词沙拉未在短生成复现
  的实锤案例。必须长生成 + 多 prompt + 温度 0 复测。
- 已定位并修复的质量 bug（移植层）：① KDA prefill beta 门缺 sigmoid（短 prompt 正常
  长 prompt 词沙拉——chunk.py 契约要求 pre-sigmoided beta）；② CANN
  `torch.ops.npu.causal_conv1d` N>1 批损坏（只有 seq0 读自己的 slot 窗口、所有序列
  尾窗写 cache_indices[0]；N=1 完全正确——串行测试永远干净的根因）→ per-sequence
  循环修复。
- 巡检口径：容器名/端口/模型名有漂移史——**先 `docker ps` + `ss -tlnp` 实测再探端口**，
  勿信固化值；`/metrics` 404 = 未开 `--enable-metrics`，不是服务死。
