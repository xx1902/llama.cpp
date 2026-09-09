# Adaptive reuse v2 消融实验 2_1

本文对 `Adaptive reuse v2` 的边界信号和成本项做独立消融。旧版 v2 文档、代码入口和
`output/optimized_prefix_*` 结果保持不变；本实验使用 `_2_1` 后缀。

## 1. v2 reference

```text
score_v2(s,e)
  = |L(s,e) - T|
  + 10 * future_nodes(e)
  + 80 * prefetch_overflow(e)
  - W_reuse * reuse_boundary(e)
  - 18 * natural_boundary(e)

W_reuse = 48 + 64 * context_reaccess_rate + 64 * exact_prefix_rate
```

`L` 和前三个成本项使用 estimated-token 等价量纲；系数是可消融的相对权重，不是硬件
时间常数。

## 2. 消融定义

| 方法 | 删除内容 |
|---|---|
| v2 reference | 不删除 |
| v2 no Sentence | 不加入句末/换行候选；保留 workload segment 边界 |
| v2 no Reuse | 不加入 reuse point，也不计算 reuse reward |
| v2 no future-node cost | 令 `10 * future_nodes = 0` |
| v2 no prefetch cost | 令 `80 * prefetch_overflow = 0` |
| v2 length only | 同时删除 reuse、future、prefetch 和 natural reward，只保留长度项 |

v2 本来不包含 Keyword 边界，所以 `v2 no Keyword` 与 reference 在代码和输入上完全
相同。为了避免把两次相同 workload 的运行噪声误解释为 Keyword 贡献，没有重复创建
一个伪消融 row；Keyword 的真实实验见 `1_1`。

## 3. 在线结果

| 数据集 | Reference | No Sentence | No Reuse | No future | No prefetch | Length only |
|---|---:|---:|---:|---:|---:|---:|
| MSC | +67.31% +/- 9.09 | +66.09% +/- 11.59 | +66.51% +/- 10.07 | +66.50% +/- 9.84 | +66.24% +/- 11.26 | +68.89% +/- 10.25 |
| Taskmaster | +64.79% +/- 14.54 | +66.43% +/- 13.98 | +64.49% +/- 16.44 | +65.99% +/- 14.50 | +65.52% +/- 13.50 | +64.85% +/- 14.24 |
| LMSYS-33K | -2.43% +/- 68.04 | +7.61% +/- 61.68 | +0.45% +/- 66.68 | +2.19% +/- 66.73 | +3.12% +/- 66.68 | +12.05% +/- 53.23 |
| ShareGPT52K | +45.20% +/- 15.22 | -50.28% +/- 157.73 | +46.44% +/- 13.90 | -53.78% +/- 159.85 | +44.13% +/- 14.41 | -49.81% +/- 158.48 |
| XSum | -233.33% +/- 215.83 | -223.42% +/- 201.42 | -233.12% +/- 210.51 | -233.76% +/- 211.38 | -224.26% +/- 210.04 | -229.60% +/- 207.91 |
| OPUS-100 | +41.11% +/- 2.49 | +40.94% +/- 2.96 | +41.80% +/- 2.45 | +41.73% +/- 2.39 | +42.14% +/- 2.38 | +41.71% +/- 2.43 |

## 4. 结果解释

### 4.1 MSC 和 Taskmaster

六种方法的 token coverage 分别稳定在约 86% 和 95%，TTFT CI 高度重叠。它们的核心
优势来自稳定 append-only 祖先；当历史本身非常规则时，评分项只是在相近边界间微调。

### 4.2 ShareGPT

六种方法的平均 chunk 数都为 `3.2`、平均约 `231` token，但边界位置不同：

- reference/no Reuse/no prefetch 为正收益约 44%~46%；
- no Sentence/no future/length only 为负，且 CI 达到约 +/-158%。

no Sentence 的自然边界覆盖从 `1.00` 降到 `0.55`；no future 和 length-only 改变了长
history 的内部祖先。5 请求中一个长请求是否命中会主导均值，因此这些结果说明边界
稳定性重要，但还不能给出精确权重最优值。

### 4.3 XSum 和 OPUS-100

XSum 的 reference first article 被切为 `[303, 299]`，多数消融仍被两 chunk 预取目标
约束成相同结构，所以所有结果都在约 `-224%~-234%`，差异小于 CI。OPUS-100 每条
前缀约 59 token、只有一个 chunk，所有消融都是 99.1% token coverage 和约 41% TTFT
改善。这两组共同说明：当硬性长度/预取约束已经唯一决定结构时，软评分项不会改变结果。

## 5. 结论

1. v2 的成本项在长、可产生多个候选节点的 ShareGPT 上才有可观察作用。
2. 对短 OPUS 和两 chunk XSum，消融几乎退化为同一个结构。
3. `future_nodes` 与 Sentence 边界应保留作为长上下文保护，但当前 5 请求不足以重新拟合 `10/18/80` 权重。
4. Reuse reward 在这批 smoke 中不是唯一收益来源；连续数据的 append-only 结构本身已提供强祖先命中。
5. 下一轮权重消融应固定边界 trace、重复 3~5 个进程和请求顺序，不能只增加单次请求数。

## 6. 图和逐数据集报告

- [MSC 2_1 图](output/ablation_v2_5/msc_continuous_prefix_method_effect_2_1.png)
- [Taskmaster 2_1 图](output/ablation_v2_5/taskmaster_continuous_prefix_method_effect_2_1.png)
- [LMSYS-33K 2_1 图](output/ablation_v2_5/lmsys_continuous_prefix_method_effect_2_1.png)
- [ShareGPT52K 2_1 图](output/ablation_v2_5/sharegpt_continuous_prefix_method_effect_2_1.png)
- [XSum 2_1 图](output/ablation_v2_5/xsum_parallel_prefix_method_effect_2_1.png)
- [OPUS-100 2_1 图](output/ablation_v2_100/opus100_parallel_prefix_method_effect_2_1.png)

原始 CSV、日志和逐数据集 Markdown 位于 `output/ablation_v2_5/` 和
`output/ablation_v2_100/`。

