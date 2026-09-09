# Adaptive reuse v2：前缀切分优化与实验分析

本文是 `PREFIX_SPLITTING_ALGORITHM_ANALYSIS.md` 的 v2 对照版。旧版文件和
`output/all_prefix_5`、`output/opus100_prefix_100` 中的结果保持不变；本文件只解释
`adaptive_reuse_v2` 和 `output/optimized_prefix_*` 中的新结果。

## 1. 目标和问题定义

一次请求的 `common_prefix_text` 是 LoRA 真正执行任务前必须读取的上下文。切分器把它
拆为一串连续 chunk，运行时为每个 chunk 建立前缀树节点：

```text
root -> H(chunk_0) -> H(H(chunk_0), chunk_1) -> ...
```

只有字节级前缀完全相同的祖先节点才能复用。因此，切分器需要同时优化四件事：

1. 命中历史祖先，减少重复 prefill；
2. 控制 chunk 数和节点数，减少树管理、淘汰和 LoRA 调度开销；
3. 在并列请求中把共享前缀放入有限的预取槽位；
4. 在句末、段落和已有复用点结束，避免为了凑长度破坏稳定边界。

单独最大化 token 命中率是不够的。ShareGPT 的历史回访率很高，但每次增长很长；XSum
有大量跨 LoRA exact-prefix pair，却仍可能因长文章构建和跨 LoRA KV 物化而回退。

## 2. v1 的不足

旧版 `Reuse-aware` 对候选终点使用固定评分：

```text
score_v1(s, e) = |L(s,e) - 128| - 72 * I(e ∈ reuse_points)
```

其中 `L(s,e)` 是估算 token 数，`I` 是指示函数。这个分数有三个结构性缺陷：

- 固定的 `72` 不随 context 回访率或 exact-prefix 概率变化；
- 不惩罚未来节点数，长文章可能产生过多 chunk；
- 不知道并列请求的预取预算，超过 `max_prefetch_chunks_per_lora=2` 后仍会继续切成同样的粒度。

`Sentence` 和 `Keyword` 虽然提高了自然边界覆盖，但可能产生很多很短的节点。短节点
增加哈希、树查找和淘汰次数，命中收益可能被管理开销抵消。`Fixed 192/384` 则相反，
节点少但中间复用点更稀疏，短请求或长尾 suffix 的命中粒度不够细。

## 3. v2 算法

### 3.1 工作负载分类

对每个 workload 预先统计：

```text
context_reaccess_rate = 回访已有 context 的请求数 / 请求总数
exact_prefix_rate     = 出现完整共享前缀的请求数 / 请求总数
```

当 `exact_prefix_rate >= 0.25` 或请求带有
`exact_shared_english_intersection` 时，判为并列模式；否则按连续模式处理。

### 3.2 目标 chunk 长度

并列模式令：

```text
T = clamp(ceil(total_tokens / max_prefetch_chunks), 96, 512)
min_tokens = max(48, floor(0.55*T))
max_tokens = max(T, floor(1.35*T))
```

默认 `max_prefetch_chunks=2`。因此一个约 700-token 的 XSum 文章目标为两个约
350-token chunk，而不是六到十个 64/128-token chunk。

连续模式根据回访密度选择目标：

```text
context_reaccess_rate < 0.70  -> T = 192
context_reaccess_rate >= 0.70 -> T = 128
total_tokens > 512            -> T = max(T, 192)
total_tokens > 768            -> T = 256
```

稀疏回访时用大 chunk 减少一次性节点开销；频繁回访时用 128 左右的 chunk 保留局部
追加复用。连续模式的有效范围是 `96 <= L <= 288`（目标 192）或
`128 <= L <= 384`（目标 256）。

### 3.3 量化评分公式

对当前位置 `s` 和候选终点 `e`，定义：

```text
L(s,e)              = estimated_tokens(text[s:e])
remaining(e)        = estimated_tokens(text[e:])
future_nodes(e)     = ceil(remaining(e) / max(1, T))
reuse_boundary(e)   = 1[e ∈ reuse_points]
natural_boundary(e) = 1[e ∈ natural_boundaries]
```

并列模式的预取溢出量为：

```text
slots_left          = max(0, max_prefetch_chunks - (chunks_already_emitted + 1))
prefetch_overflow   = max(0, future_nodes(e) - slots_left)
```

复用权重随 workload 变化：

```text
W_reuse = 48 + 64 * context_reaccess_rate + 64 * exact_prefix_rate
```

最终分数为：

```text
score_v2(s,e)
  = |L(s,e) - T|                  # 长度偏差，token-equivalent
  + 10 * future_nodes(e)          # 后续节点数量成本
  + 80 * prefetch_overflow        # 超出预取预算的惩罚
  - W_reuse * reuse_boundary(e)   # 历史复用点奖励
  - 18 * natural_boundary(e)      # 自然边界奖励
```

选择 `score_v2` 最小的候选终点；没有可行候选时退化为目标长度的 fixed chunk。
所有项都以“估算 token 等价成本”为量纲，因而可以直接做消融。系数不是硬件时间的
物理常数，而是调度目标中的相对权重：`80 > 10 > 18` 表示当前配置优先避免预取溢出，
其次减少未来节点，再用自然边界做温和奖励；`W_reuse` 则在复用概率高时超过长度偏差，
值得为历史复用点多保留一点边界偏移。

### 3.4 伪代码

```text
classify workload
choose T, min_tokens, max_tokens
for each chunk start s:
    if remaining(s) <= max_tokens:
        emit remaining text
        break
    candidates = natural_boundaries ∪ reuse_points
    viable = candidates with min_tokens <= L(s,e) <= max_tokens
    for e in viable:
        calculate future_nodes, prefetch_overflow, W_reuse
        calculate score_v2(s,e)
    choose e with minimum score
    emit text[s:e]
```

## 4. v1/v2 对比结果

五个连续/并列 smoke workload 使用 `5` 条请求；OPUS-100 使用 `100` 条请求。表中的
TTFT change 定义为：

```text
100 * (TTFT_full_prefill - TTFT_online) / TTFT_full_prefill
```

正值表示在线 TTFT 降低，负值表示在线 TTFT 增加。v1 取五个旧方法中该数据集的
最高点，仅用于和 v2 做“最佳旧基线”比较；不是重新挑选运行结果。

| 数据集 | v1 最佳方法 | v1 TTFT | v2 TTFT | v2 - v1 | v1 token 覆盖 | v2 token 覆盖 | v1 节点峰值 | v2 节点峰值 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MSC | Sentence | +63.19% +/- 12.91 | **+67.09% +/- 8.54** | +3.91 pp | 86.0% | 86.0% | 7 | 5 |
| Taskmaster | Sentence | +60.56% +/- 14.48 | **+64.24% +/- 13.35** | +3.68 pp | 95.0% | 95.0% | 7 | 5 |
| LMSYS-33K | Fixed 64/128 | +16.98% +/- 28.71 | **-5.42% +/- 69.90** | -22.40 pp | 36.0% | 59.4% | 9 | 6 |
| ShareGPT52K | Fixed 192/384 | +65.51% +/- 25.42 | **+44.76% +/- 13.54** | -20.75 pp | 96.8% | 72.5% | 8 | 13 |
| XSum | Fixed 192/384 | -418.54% +/- 346.02 | **-233.35% +/- 216.99** | +185.19 pp | 24.9% | 26.1% | 6 | 4 |
| OPUS-100 | Fixed 64/128 | +43.70% +/- 2.14 | **+41.62% +/- 2.36** | -2.08 pp | 99.1% | 99.1% | 31 | 32 |

MSC、Taskmaster 和 XSum 的前五行来自 5-request smoke，置信区间很宽；OPUS-100 的
100-request 结果更稳定。ShareGPT 的 v2 结构目标减少了部分长度偏差，但在这次 trace
中为了对齐历史边界形成了较多有效节点，因而 token 覆盖低于大 chunk 的 v1，说明
“减少未来节点”与“保留完整祖先”之间仍有真实 trade-off。

## 5. 各数据集的解释

### MSC 和 Taskmaster：v2 的主要正例

两者都是 append-only 连续对话/任务状态：

```text
R0 = S + C0 + T0
R1 = S + C0 + T0 + T1
R2 = S + C0 + T0 + T1 + T2
```

后续请求从已有祖先继续追加，v2 用 128 左右的 chunk 对齐追加点，又把节点数从约
7 降到 5。MSC 的 token 覆盖为 86.0%，Taskmaster 为 95.0%，所以减少的在线工作主要
是新增 suffix，而不是重新构建整段历史。

### LMSYS-33K：回访稀疏导致不稳定

LMSYS 中许多请求属于新 context，只有短 system prefix 可以复用。v2 把平均 chunk 数
从 2.6 降到 1.6、节点峰值从 9 降到 6，但 5 条请求中命中哪一个长聊天会显著改变
均值，结果为 `-5.42% +/- 69.90`。这更像 workload 热度不足和样本方差问题，而不是
切分边界本身已被证明错误。

### ShareGPT52K：长轮次使收益受 suffix 和容量限制

ShareGPT 的 context 回访高，但单轮 user/assistant 文本可能带来数百 token。v2 的
平均 chunk 仍为 3.2，节点峰值为 13；虽然 TTFT 回退幅度相比旧版长尾方法更小，
`72.5%` token 覆盖仍低于 Fixed 192/384 的 `96.8%`。这说明长请求应进一步采用
“每轮上限 + 热点祖先保护”或增大 GPU/Host 容量，不能只继续调整自然边界权重。

### XSum：并列长前缀仍受跨 LoRA 成本约束

XSum 的并列请求形如：

```text
R0 = S + article + summarize
R1 = S + article + question-answer
R2 = S + article + headline-rewrite
```

v2 识别到 exact-prefix pair，并把约 700-token article 压缩到平均 2 个 chunk，节点峰值
从 6 降到 4，TTFT 从约 `-418.5%` 改善到 `-233.3%`。但第一个 LoRA 仍要完整建立
article KV，后续 LoRA 还要支付 variant 物化、调度和未命中 suffix；默认预取最多覆盖
两个 chunk，后半段不能保证后台完成。因此重复概率高并不等于在线 TTFT 必然为正。

### OPUS-100：短并列前缀正例

OPUS-100 的共享源前缀平均约 59 token，每个 source 通常只对应一到两个小 chunk。
100 请求中 runtime token coverage 为 99.1%，v2 TTFT 为 `+41.62% +/- 2.36`，与旧版
最佳 Fixed 64/128 的 `+43.70% +/- 2.14` 接近。这里边界差异已经小于 LoRA 切换和
测量噪声，重点是 exact pair 紧邻且共享前缀足够短。

## 6. 量化指标和判断规则

### 6.1 结构指标

```text
structure_token_coverage = 可落在方法 chunk 边界上的 prefix token / prefix token
natural_boundary_coverage = 命中自然边界的切分终点 / 可用自然边界终点
mean_chunks_per_context   = 每个 context 的 chunk 数平均值
unique_nodes               = 结构分析序列中不同祖先节点的数量
```

这些是离线结构指标，不能替代 runtime 命中。尤其是 `structure_token_coverage` 高，
只说明切分边界稳定，不代表该节点已经被同一 LoRA 或跨 LoRA 请求实际建立。

### 6.2 Runtime 指标

```text
request_hit_rate = count(reused_tokens > 0) / request_count
token_hit_rate   = sum(reused_tokens) / sum(prefix_tokens)
```

建议把收益判为“可靠”至少同时满足：`token_hit_rate >= 0.80`、TTFT CI 不跨 0、且
`runtime_peak_nodes` 与淘汰次数没有明显上升。仅有 request hit 而 token coverage 很低
时，通常只是命中了短 system prefix。

### 6.3 预期收益的近似分解

```text
net_saving ≈ α * reused_tokens
             - β * uncached_suffix_tokens
             - γ * unique_nodes
             - δ * lora_variant_materialization
             - ε * host_gpu_transfer
```

`α~ε` 是机器和运行时相关的时间系数，实验中不把它们伪装成固定常数；v2 的
`score_v2` 只负责在切分阶段用可观察代理量近似这一目标。TTFT、p95 和 cell wall time
仍需通过实际运行测量。

## 7. 参数消融建议

下一轮可以固定请求顺序，只改变一个参数：

1. XSum：`max_prefetch_chunks` 取 `2/4/8/0`，验证预取溢出项 `80` 是否是主因；
2. ShareGPT：把单轮追加截断到 `128/256/384` token，观察 `future_nodes` 与容量淘汰；
3. LMSYS：把 context 回访率重采样到 `25%/50%/75%`，验证 `W_reuse` 的自适应效果；
4. 对 `W_reuse` 的三个系数和自然边界奖励 `18` 做网格消融，报告均值、p95 和配对 CI；
5. 每个配置至少重复 3 个 request order，避免 5-request smoke 的偶然顺序主导结论。

## 8. 可复现实验文件

- v2 代码：`analyze_results.py` 的 `split_prefix_adaptive_v2`；
- 在线运行器：`run_prefix_method_experiments.py --methods adaptive_reuse_v2`；
- v1/v2 图和逐数据集报告：`output/optimized_prefix_5/`；
- OPUS-100 100 请求 v2：`output/optimized_prefix_100/`；
- 跨数据集汇总：`output/optimized_prefix_5/ALL_PREFIX_METHOD_RESULTS_2.md`。

