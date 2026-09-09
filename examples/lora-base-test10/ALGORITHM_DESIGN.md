# test10 算法设计与实验结论

## 1. 当前应该优化什么

“从根节点逐级匹配，首次不匹配时建立新分支”本身就是 RadixAttention 的典型行为，不适合作为主要创新点。当前更值得推进的是三个相互配合的机制：

1. **存储单位**：以 `conversation_id × LoRA_id` 的完整路径作为保护、计分和淘汰单位，避免只淘汰叶节点后留下无法形成完整命中的碎片。
2. **异步提交**：当前请求先用已有树匹配并立即 prefill；Sentence/关键词切分与推理并行，完成后再把本次已生成的 KV 区间提交到树中，服务后续请求。
3. **异构自适应调度**：GPU 有余量时使用低优先级 CUDA stream；GPU 饱和但 CPU 有余量时使用 CPU worker；两侧都忙时延后到 prefill 后或空闲期。

因此论文故事不应说“发明了另一种 Radix tree”，而应说：针对端侧多 App、多 LoRA、同一会话跨时间返回的场景，设计了 sequence-aware KV 生命周期和不阻塞 TTFT 的异步建树流水线。

## 2. 数据集选择

数据由 `examples/lora-base-test4/build_87_lora_workloads.py` 生成。

| 文件 | 请求数 | 实际 App/LoRA 覆盖 | 用途 |
|---|---:|---:|---|
| `delta/delta_requests.jsonl` | 696 | 87/87 | 本次切分和并行覆盖实验 |
| `grouped/grouped_requests.jsonl` | 600 | 21/21 | 在线时序、复用、淘汰实验 |
| `delta/delta_pairs.jsonl` | 616 对 | 87 映射池 | 相同/跨 group 的 delta 对比 |

`delta_requests` 含 80 个上下文，来源为 XSum 与 ShareGPT52K；App/LoRA 一一映射和 group 由 LSApp 87-App trace 与语义家族共同构造。本次对 87 个 LoRA 做确定性分层轮询，每个 LoRA 取一条，共 87 条。

## 3. 通用 chunk 方法

### 3.1 Sentence + fixed

先用中英文句末标点分句，再按目标大小装箱。单句超过上限时退化为固定长度切分：

```text
sentences = split_by_punctuation(text)
for sentence in sentences:
    if len(sentence) > L:
        flush(current_chunk)
        split sentence every L tokens
    elif len(current_chunk) + len(sentence) > L:
        flush(current_chunk)
    append sentence
```

本实验同时测 `L = 64/128/256`，并保留 fixed-128 基线。这个方法快、确定、无需模型，适合作为在线 fast path。

### 3.2 TF-IDF topic discontinuity

纯粹把高 TF-IDF 关键词位置设为边界并不合理，因为关键词通常属于主题内部。test10 改为在固定上限之前寻找左右局部词分布差异最大的点。

对候选边界 `b`，取左右各 32 个 token，计算 IDF 加权 Jaccard 距离：

```text
D(b) = 1 - Σ_w min(tf_left(w), tf_right(w)) × idf(w)
             / Σ_w max(tf_left(w), tf_right(w)) × idf(w)
```

每个目标边界 `t = start + 128` 只在 `[t-24, t]` 搜索，并加一个很小的长度惩罚，防止总选择过早边界：

```text
b* = argmax_b D(b) - 0.08 × (t-b)/24
```

这仍是轻量统计语义，不等同于 KeyBERT/BGE-M3。它主要验证边界是否真的改变，而不是验证语义质量。

## 4. 与推理并行的正确时序

切分如果影响树匹配，就不能在当前请求“先使用、后计算”。可实施的做法是 fast path + delayed commit：

```text
请求到达
  ├─ 现有树：从 root 做 token/radix 匹配，得到可复用 KV
  ├─ 高优先级：立即执行缺失前缀的 prefill
  └─ 后台分析：Sentence/TF-IDF/embedding 计算候选边界
                         │
prefill 完成并产生 KV ──┴─ 校验 request generation/version
                              └─ 按边界引用 KV span，增量提交新路径
```

关键约束：

- 当前请求只复用提交前已经存在的节点；新边界只帮助未来请求。
- 后台任务携带 `request_id + conversation_id + lora_id + generation`，过期结果不能写回。
- prefill 需要保留可切片的 token→KV 位置；不应为了建树重新跑一次 prefill。
- 相同文本采用稳定切分版本号，避免新旧分词配置产生不可比较的边。

## 5. 两种并行路径

### 5.1 CPU worker + GPU inference

```text
CPU worker:  [ TF-IDF / sentence analysis ]
GPU stream:  [       inference proxy       ]
wall time:   max(CPU, GPU) + submit/sync overhead
```

优点是 CPU 与 GPU 资源隔离较好，推理 kernel 饱和时仍可能并行。缺点是占用 CPU、共享内存带宽并增加能耗；移动 SoC 上 CPU/GPU 可能共享 DRAM，因此并非完全无竞争。

### 5.2 CUDA high/low priority streams

```text
high priority (-5): [ inference proxy ]
low priority  ( 0): [ keyword proxy   ]
```

优点是数据留在 GPU，且 GPU 有空闲 SM 时重叠充分。缺点是 stream priority 只提供调度提示，不会抢占已经运行的长 kernel，也不保证并发；推理占满 SM 或带宽时，后台任务会拖慢总时间。

### 5.3 推荐的自适应选择

```text
if gpu_utilization < U_low and inference_deadline_has_margin:
    launch keyword task on low-priority CUDA stream
elif cpu_utilization < C_high and thermal_budget_ok:
    enqueue keyword task to persistent CPU worker
else:
    defer task until post-prefill or idle window
```

这是本次最有价值的结论：两种方法没有固定胜负，选择取决于当前资源余量。后续真实端侧实现可用 EMA 记录两类任务耗时，并把 SLO miss、温度和功耗纳入决策。

## 6. 建树、存储和淘汰示例

假设 WorkBuddy 中同一会话先后发生：

```text
conversation C7 + LoRA-ticket
  Q1: 帮我订 9 月 10 日去北京的票
  Q2: 几分钟后，帮我订 9 月 15 日的返程票
```

已有树仍按 Radix 方式工作：Q2 从 root 逐级匹配 Q1 的系统提示、用户资料和去程上下文，在“返程日期”首次不匹配处新开分支。

真正的变化在存储生命周期：

- 节点可由多个 sequence owner 引用，公共前缀只存一份。
- `C7 × LoRA-ticket` 的整条路径有统一的 `last_access`、reuse、recompute cost 和 future probability。
- 淘汰时移除 owner 对整条路径的引用；只有 owner 为空的节点才释放。
- 如果另一个 LoRA 共享 base KV，但 LoRA delta 不兼容，只复用明确标记为 LoRA-independent 的层/区间，不能按文本相同就盲目复用。

一个可继续验证的路径得分为：

```text
score(path) =
    α × normalized_recency
  + β × reuse_frequency
  + γ × predicted_return_probability
  + δ × recompute_cost
  + ε × complete_path_bonus
  - ζ × bytes
  - η × fragmentation_penalty
```

其中 `complete_path_bonus` 和 `fragmentation_penalty` 对应老师提出的“完整对话序列替换”，可与普通 leaf-LRU 做消融。

## 7. 本次实测结果

硬件为 RTX 2080 Ti；每个 87-LoRA 样本运行 3 次，先按请求取中位数，再跨请求求均值。`headroom` 使用较少 blocks，`saturated` 使用更多 blocks。

CUDA 迭代量以 640 个前缀 token 为基准，并在 `0.5–2.5` 倍范围内随 `estimated_common_prefix_tokens` 线性变化，因此长度分箱图反映的是代理负载随序列增长后的调度趋势。

| 负载 | CPU 串行 | CPU 并行 | CPU 收益 | CUDA 串行 | 双 stream | CUDA 收益 |
|---|---:|---:|---:|---:|---:|---:|
| headroom | 0.823 ms | 0.780 ms | 5.3% | 1.345 ms | 0.782 ms | 41.9% |
| saturated | 2.835 ms | 2.708 ms | 4.5% | 3.700 ms | 3.836 ms | -3.7% |

切分结构统计：

| 方法 | 平均 chunk 数 | 聚合平均 chunk token |
|---|---:|---:|
| fixed-128 | 4.43 | 116.53 |
| Sentence+fixed-64 | 10.55 | 48.87 |
| Sentence+fixed-128 | 4.83 | 106.82 |
| Sentence+fixed-256 | 2.52 | 204.85 |
| TF-IDF discontinuity-128 | 4.62 | 111.60 |

TF-IDF-128 与 fixed/Sentence-128 的 chunk 数仍然接近，说明统计语义边界没有带来结构上的巨大变化，这与前期观察一致。下一步不要只比较 chunk 数，而要测边界稳定性、未来请求覆盖 token、完整路径命中率以及真实 TTFT。

## 8. 结果不能证明什么

- CUDA 两个 kernel 是调度代理，不是 BGE-M3 或 llama.cpp prefill。
- Python/C++ TF-IDF 实现不同，绝对延迟不用于比较模型优劣。
- 当前未测真实 tokenizer、KV 显存占用、答案质量、功耗与温度。
- 桌面 RTX 2080 Ti 不是移动 GPU/NPU，结论只能作为端侧实验设计依据。

## 9. 下一步实验顺序

1. 把调度 hook 接到 test8 的真实 prefill 前后，仅记录时间，不先改 KV 所有权。
2. 用真实 tokenizer 替换正则 token，固定 Sentence 64/128/256 作为三条通用基线。
3. 将已生成 KV 按 delayed-commit 边界登记到树，验证不发生二次 prefill。
4. 在 grouped 在线 trace 上联测 sequence-aware 淘汰：leaf-LRU、sequence-LRU、带 `complete_path_bonus` 的 score。
5. 加入真实轻量 embedding 模型后，再测边界质量、TTFT、能耗和 SLO；否则不能声称 BGE-M3 方案有效。
