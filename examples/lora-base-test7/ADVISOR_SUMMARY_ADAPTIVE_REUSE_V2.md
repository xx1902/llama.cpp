# Adaptive reuse v2：导师汇报摘要

## 1. 实验目的

本实验研究多 LoRA 请求中的共享前缀复用问题。对每条请求，先把
`common_prefix_text` 切分为连续的 prefix chunks，再由前缀树和
`system-v2` 运行时复用已经建立的 KV 前缀。实验重点不是单纯提高命中
token 数，而是在以下目标之间取得平衡：

- 减少重复 prefill，降低首 token 延迟（TTFT）；
- 控制前缀树节点数量、预取数量和淘汰次数；
- 对连续对话和并列 LoRA 请求使用不同的切分粒度；
- 尽量保留句末、segment 末尾和历史复用点等稳定边界。

## 2. v2 方法概述

v1 的 Reuse-aware 方法固定使用约 128-token 目标，并给历史复用终点一个固定奖励：

```text
score_v1(s,e) = |L(s,e) - 128| - 72 * I(e in reuse_points)
```

v2 先根据 workload 的上下文回访率和完整共享前缀比例判断请求形态，再自适应选择
目标 chunk 长度：

- 连续对话：回访频繁时使用约 128-token chunk，回访稀疏或前缀较长时使用约
  192~256-token chunk，以减少节点管理开销；
- 并列请求：根据共享前缀总长度和最多预取 chunk 数计算目标长度。默认最多预取
  2 个 chunk，因此约 700-token 的 XSum 文章目标为两个约 350-token 的 chunk；
- 评分同时考虑长度偏差、未来节点数、预取溢出、历史复用点和自然边界：

```text
score_v2(s,e)
  = |L(s,e) - T|
  + 10 * future_nodes
  + 80 * prefetch_overflow
  - W_reuse * reuse_boundary
  - 18 * natural_boundary

W_reuse = 48 + 64 * context_reaccess_rate + 64 * exact_prefix_rate
```

其中 `L(s,e)` 是估算 token 数，`T` 是当前 workload 的目标 chunk 长度。上述系数是
切分阶段的相对代价，不是某一块 GPU 的固定毫秒数；最终 TTFT 仍需通过运行时测量。

## 3. v1/v2 对照结果

TTFT 优化率定义为：

```text
100 * (TTFT_full_prefill - TTFT_online) / TTFT_full_prefill
```

正值表示在线复用比 full-prefill 更快，负值表示在线策略反而更慢；`+/-` 为固定
seed bootstrap 均值置信区间半宽。v1 取该数据集五种旧切分方法中的最佳结果，作为
“最佳旧基线”，不是重新挑选 v2 后的结果。

| 数据集 | 请求数 | v1 最佳方法 | v1 TTFT 优化率 | v2 TTFT 优化率 | v2 - v1 | v1 token 覆盖 | v2 token 覆盖 | v1 节点峰值 | v2 节点峰值 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| MSC | 5 | Sentence | +63.19% +/- 12.91 | **+67.09% +/- 8.54** | +3.91 pp | 86.0% | 86.0% | 7 | 5 |
| Taskmaster | 5 | Sentence | +60.56% +/- 14.48 | **+64.24% +/- 13.35** | +3.68 pp | 95.0% | 95.0% | 7 | 5 |
| LMSYS-33K | 5 | Fixed 64/128 | +16.98% +/- 28.71 | **-5.42% +/- 69.90** | -22.40 pp | 36.0% | 59.4% | 9 | 6 |
| ShareGPT52K | 5 | Fixed 192/384 | +65.51% +/- 25.42 | **+44.76% +/- 13.54** | -20.75 pp | 96.8% | 72.5% | 8 | 13 |
| XSum | 5 | Fixed 192/384 | -418.54% +/- 346.02 | **-233.35% +/- 216.99** | +185.19 pp | 24.9% | 26.1% | 6 | 4 |
| OPUS-100 | 100 | Fixed 64/128 | +43.70% +/- 2.14 | **+41.62% +/- 2.36** | -2.08 pp | 99.1% | 99.1% | 31 | 32 |

`pp` 表示 percentage points（百分点）。例如 MSC 的 `+3.91 pp` 表示 v2 的 TTFT
改善比例比 v1 高 3.91 个百分点，而不是 3.91 倍。

## 4. 结果解读

### MSC 和 Taskmaster：连续追加场景的正例

两类请求都属于 append-only 结构：后续请求保留已有 system/context/history，
只在末尾追加新一轮对话或 task state。v2 根据较高的 context reaccess rate 使用
约 128-token 粒度，并优先对齐已有复用终点。这样在保持 86.0%（MSC）和 95.0%
（Taskmaster）token 覆盖的同时，把峰值节点从 7 降到 5，TTFT 改善约 3.7~3.9 个
百分点。这里的收益主要来自已有祖先节点复用，而不是关键词切分本身。

### LMSYS-33K：低回访导致估计不稳定

LMSYS 中很多请求来自新 conversation，真正回到同一 context 的请求较少。v2 虽然
把 token 覆盖从 36.0% 提高到 59.4%，并把节点峰值从 9 降到 6，但 5 条请求的顺序
会显著影响均值，结果为 `-5.42% +/- 69.90`。因此目前只能说明 v2 提高了结构命中
潜力，不能据此断言在线 TTFT 已稳定改善。

### ShareGPT52K：高回访和长 suffix 的权衡

ShareGPT 的对话回访率较高，但单轮 user/assistant 文本可能很长。v2 试图保留历史
边界并减少未来节点，却在这条 trace 中产生了更多有效节点（8 -> 13），token 覆盖
也从 96.8% 降到 72.5%。TTFT 仍为正，但低于 v1，说明长轮次的 suffix 计算、节点
管理和容量压力已经抵消了一部分复用收益。

### XSum：并列长共享前缀仍有跨 LoRA 成本

XSum 的请求可以抽象为：

```text
R0 = system + article + summarize
R1 = system + article + question-answer
R2 = system + article + headline-rewrite
```

v2 识别到完整共享文章前缀，并将其压缩到约两个大 chunk，节点峰值从 6 降到 4，
TTFT 相对 v1 的回退幅度由约 418.5% 降至 233.4%。但第一条 LoRA 仍需建立完整
article KV，后续 LoRA 还要承担 variant 物化、调度和未命中 suffix 的成本；默认
预取只覆盖两个 chunk，因而高 exact-prefix 比例并不保证 TTFT 为正。

### OPUS-100：短共享前缀的稳定正例

OPUS-100 的共享源文本平均约 59 token，通常只形成一到两个小 chunk。100 请求中
runtime token coverage 保持 99.1%，v2 的 TTFT 改善为 `+41.62% +/- 2.36`，与
Fixed 64/128 的 `+43.70% +/- 2.14` 接近。这个结果说明：当前缀短、跨 LoRA 请求
紧邻且 exact-prefix 命中稳定时，切分策略之间的差异较小，复用本身是主要收益来源。

## 5. 给导师的阶段性总结

> Adaptive reuse v2 在固定 chunk 的 Reuse-aware 方法基础上，引入了 workload-aware
> 的目标长度和代价函数：连续对话根据 context 回访率选择 128~256-token 粒度，并
> 对历史复用点加权；并列请求则根据共享前缀长度和预取预算合并为少量大 chunk，
> 同时惩罚未来节点增长和预取溢出。初步结果显示，v2 对 MSC 和 Taskmaster 这类
> append-only 连续请求最有效，在保持 86.0%/95.0% token 覆盖的同时将节点峰值从 7
> 降到 5，TTFT 额外改善约 3.7~3.9 个百分点；对 XSum，v2 将长共享前缀造成的
> TTFT 回退明显减轻，但跨 LoRA KV 物化和长 suffix 计算仍使结果为负；对 LMSYS 和
> ShareGPT，v2 暴露出低回访或长轮次带来的方差和容量权衡；OPUS-100 的 100 请求
> 结果最稳定，约 99% token 被复用，TTFT 改善约 42%。因此当前结论是：切分策略必须
> 与请求形态、回访概率和预取预算联合设计，不能用单一 chunk 大小覆盖所有数据集。
> 目前除 OPUS-100 外其余结果仍是 5 请求 smoke test，下一步应在相同请求顺序下扩展
> 到 100 请求、重复多个 request order，并报告配对 TTFT CI 后再做最终性能结论。

## 6. 结论边界和下一步

当前结果可以支持“v2 的设计方向合理、不同 workload 需要不同粒度”这一阶段性判断，
但不能支持 v2 已经在全部数据集上优于 v1。下一轮建议：

1. 将 MSC、Taskmaster、LMSYS、ShareGPT、XSum 都扩展到 100 请求，并至少重复 3 个
   request order；
2. 对 XSum 单独比较 `max_prefetch_chunks=2/4/8/0`，验证预取预算是否是回退主因；
3. 对 ShareGPT 限制单轮追加长度，区分长 suffix 成本和切分边界成本；
4. 对 LMSYS 改变 context 热度，验证回访率阈值和 `W_reuse` 的稳定性；
5. 最终同时报告 TTFT、p95、token coverage、节点峰值、淘汰次数和 cell wall time，
   不仅看单一平均 TTFT。

## 7. 复现入口

- 算法和完整说明：[`PREFIX_SPLITTING_ALGORITHM_ANALYSIS_2.md`](PREFIX_SPLITTING_ALGORITHM_ANALYSIS_2.md)
- v2 切分实现：`analyze_results.py` 中的 `split_prefix_adaptive_v2`
- 在线实验入口：`run_prefix_method_experiments.py --methods adaptive_reuse_v2`
- v2 smoke 输出：`output/optimized_prefix_5/`
- OPUS-100 100 请求输出：`output/optimized_prefix_100/`
