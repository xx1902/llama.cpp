# Anchor Family 与三级缓存实验设计

## 1. 实验目标

这组实验验证三个问题：

1. 调用频率能否提高 anchor family 的有效 KV 命中。
2. 联合频率、预测、prefix、fanout 和成本，能否在命中收益与存储占用之间取得更好折中。
3. Host full、Host delta 和预测 GPU-ready 三种能力是否具有可测量的独立作用。

## 2. 当前量化模型

Variant 调用频率采用 EMA 衰减：

\[
f_{j,t}=f_{j,last}\gamma^{t-last}+1,\qquad \gamma=0.95
\]

Family 调用频率：

\[
F_i=\sum_{j\in Family_i}f_{j,t}
\]

默认收益：

\[
B_i=4\ln(1+F_i)+20P_i+0.002L_i+3D_i
+\frac{4}{1+A_i}+Q_i
\]

其中：

- `P`：预测概率。
- `L`：prefix token 数。
- `D`：delta fanout。
- `A`：距离最近访问的请求数。
- `Q`：shared system=8、user context=5、其他=1。

成本：

\[
C_i=0.05(M_i^{GPU}+M_i^{Host})+0.0005T_i^{materialize}
\]

价值密度：

\[
Score_i=\frac{B_i-C_i}{\max(1,M_i^{GPU}+M_i^{Host})}
\]

## 3. 实验假设

| 假设 | 预期现象 |
|---|---|
| H1 Frequency 优于纯 Recency | 有效命中率或总命中 token 增加 |
| H2 Frequency+Recency 更稳定 | 热点变化时兼顾旧热点与近期热点 |
| H3 Prediction 能保护即将访问的 family | 预测有效期内 family 淘汰次数下降 |
| H4 Full latency 提高绝对复用收益 | 长 prefix 和高 fanout family 保留更多 |
| H5 Full density 降低单位收益的存储成本 | Host/GPU 峰值下降，但命中可能下降 |
| H6 Delta 降低 Host 中重复 KV 占用 | 压缩分支物理字节低于 full-KV equivalent |
| H7 正确 GRU 预取建立 GPU-ready KV | 正确预测位置 TTFT 下降 |

## 4. Family 评分消融

统一参数：

```text
相同 workload
相同请求顺序
相同 prefix chunk
相同缓存容量
每种策略重复 3 次
```

策略：

| Case | 开启项 |
|---|---|
| `recency` | 最近访问 |
| `frequency` | 调用频率 |
| `frequency_recency` | 频率 + 最近访问 |
| `frequency_prediction` | 频率 + 最近访问 + 预测保护 |
| `full_latency` | 全部收益与成本，不按 MiB 归一化 |
| `full_density` | 全部收益与成本，按 MiB 归一化 |

主要指标：

```text
any_chunk_prefix_hit_rate
full_prefix_hit_rate
total_chunk_hit_tokens
peak_host_full_mb
peak_host_delta_mb
family_evictions
active_prediction_family_evictions
TTFT change relative to same-run baseline
```

TTFT 容易受到不同运行轮次 GPU 波动影响。Family 消融首先使用命中 token、存储和淘汰事件判断，
TTFT 必须报告三次均值与误差条。

## 5. 三级状态消融

| Case | GPU exact | Host full | Host delta | Prediction ready |
|---|---:|---:|---:|---:|
| `gpu_exact_only` | 是 | 否 | 否 | 否 |
| `gpu_plus_host_full` | 是 | 是 | 否 | 否 |
| `host_anchor_delta` | 是 | 是 | 是 | 否 |
| `gru_gpu_ready` | 是 | 是 | 是 | 是 |

需要报告：

- GPU exact/partial prefix 命中。
- Host full 与 Host delta 峰值。
- 压缩分支的 full-KV equivalent。
- Delta encode、load、materialize 工作量。
- 正确/错误预取数量与成本。
- TTFT、后台 overrun 和 background fit rate。

当前实验是逻辑状态消融，不代表已经存在独立 GPU compact-delta 物理池。

## 6. 分组消融

比较：

```text
Random
Semantic
Transition
Hybrid
```

控制不变：

- LoRA 总数为 87。
- Group 数为 10。
- 每组 8-9 个 LoRA。
- 请求轨迹、prefix 文本和 delta pair 数量一致。

指标：

```text
adapter/app family purity
cross-App transition locality
KV cosine gap = (1-cosine) x 1000
KV relative L2
delta logical saved rate
family fanout
prefix hit tokens
```

由于 Q8 数据格式固定，分组更可能改善重建误差和可用性，而不一定改变理论压缩率。

## 7. 预测消融

比较：

```text
No prediction
GRU file
GRU online
Oracle
```

指标：

- Top-1/Top-k accuracy。
- GRU model inference 和 HTTP query latency。
- 正确预测位置与错误预测位置。
- Prefetch full/delta/materialize 次数。
- 正确与错误预取工作量。
- GPU-ready 命中和 TTFT。

Oracle 只表示收益上限，不能作为最终系统结果。

## 8. 重复实验与统计

每个 case 至少重复 3 次：

```text
mean
sample standard deviation
error bar
```

正式结果不要混合不同代码版本、不同请求数或不同缓存容量。每次运行的参数保存在
`system_parameters.csv` 中。

## 9. 可视化设计

统一生成以下彩色图：

```text
01_prefix_delta_ablation.png/pdf
02_family_score_ablation.png/pdf
03_storage_tier_ablation.png/pdf
04_grouping_ablation.png/pdf
05_prediction_ablation.png/pdf
```

图中使用：

- 柱高表示重复实验均值。
- Error bar 表示样本标准差。
- Host full 与 Host delta 使用堆叠柱。
- TTFT 使用相对同轮 baseline 的变化率。
- KV cosine 使用 `(1-cosine)*1000`，避免数值都接近 1 时图形无法区分。

## 10. 运行顺序

```text
1. Family 三次重复
2. Storage 三次重复
3. Grouping 三次重复
4. 启动 GRU 服务
5. Prediction 三次重复
6. 汇总 CSV
7. 生成图片
```

