# GLM-5.3-Flash W8A8 modelslim_v1 yaml — 外部验证产物（2026-09-01）

## 产物位置
`/data/models/GLM-5.3-Flash-W8A8-xpinke/` on a 910B node（296GB，74 片×~4.3GB）

## yaml 原件（`GLM-5.3-Flash_best_practice.yaml`）
```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: linear_quant
    qconfig:
      act:
        scope: per_token
        dtype: int8
        symmetric: true
        method: minmax
      weight:
        scope: per_channel
        dtype: int8
        symmetric: true
        method: minmax
    include:
    - '*language_model.layers*.self_attn*'
    exclude:
    - '*indexer*'
    - '*norm*'
  - type: linear_quant
    qconfig:
      act:
        scope: per_token
        dtype: int8
        symmetric: true
        method: minmax
      weight:
        scope: per_channel
        dtype: int8
        symmetric: true
        method: minmax
    include:
    - '*language_model.layers*.mlp*'
    exclude:
    - '*mlp.gate'
    - '*shared_experts*'
```

## 产物 dtype 分布（全 74 片聚合）
- int8: 36,773 张量（attn + routed experts 全量化）
- float32: 73,546 张量（weight_scale / weight_offset，量化元数据）
- bf16: 998 张量（norm / embedding / shared_experts / gate 等未量化部分）

## 与 llm_ptq 产物对比
| 维度 | llm_ptq（0831 我们的） | modelslim_v1 yaml（xpinke） |
|------|----------------------|---------------------------|
| 大小 | 578GB | 296GB |
| 专家 dtype | BF16（未量化） | int8 ✅ |
| 分片 | 单文件（part_file_size 被忽略） | 74 片×~4.3GB ✅ |
| int8 张量数 | 529 | 36,773 |
| TP8 每卡 | 72GB ❌ | 37GB ✅ |
| 压缩率 | ~0%（vs BF16 源） | 49% |

## 部署可行性（0901 实测）
- TP8 单节点 8×64GB：37GB/卡权重 + 27GB 余量做 KV cache → **可行**
- node1/node3 当前跑 GLM-5.2 PD 生产（HBM ~满），需先停生产服务
- node1 上 GLM-5.3-Flash FP8 下载不完整（32/62 分片），不可用
- 产物无 `.quant_done` 标记，部署前需验证加载

## 关键差异分析
yaml 的 `include` 用 pattern 匹配（`*mlp*`）而非 `isinstance(nn.Linear)` —
这是 modelslim_v1 与 llm_ptq 的根本区别：pattern 匹配能命中
`Glm5NextTextExperts` 的 `gate_up_proj`/`down_proj`（裸 nn.Parameter 3D），
而 llm_ptq 的 AntiOutlier/Calibrator 只遍历 `nn.Linear` 子类。
