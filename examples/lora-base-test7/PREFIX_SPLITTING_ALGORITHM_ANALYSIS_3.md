# LoRA-focus filter v3：分类式前缀切分与实验 3

本文提出第三种前缀切分方法：`LoRA-focus filter v3`。它尝试在 prefill 之前区分：

- **LoRA-private resident**：跨 LoRA 复用低，但在同一 LoRA 内重复的稳定提示词；
- **shared reuse**：跨 LoRA 或同 context 反复出现、适合建立公共祖先的文本；
- **user focus**：与当前 task 词汇重合、靠近上下文尾部的当前关注内容；
- **cold bulk**：低复用、低关注的一次性长文本。

当前实现是轻量 filter/text-classification proxy，不读取模型真实 attention。原因是获取
attention 必须先完成一次模型前向，反而会增加 TTFT；真正的 attention-aware 版本更适合
使用离线训练得到的先验分类器。

## 1. 频率 profile

只使用请求的 `common_prefix_text`、`task`、`lora_id` 和 `context_id`，不读取答案和
runtime 命中结果。profile 按请求因果更新：第 `i` 条请求只能使用 `0..i` 的文本统计，
不会观察后续请求。对过滤停用词后的 term `t` 定义：

```text
support(t)       = min(1, (request_count(t) - 1) / 3)
cross_lora(t)    = (distinct_loras(t) - 1) / max(1, total_loras - 1)
concentration(t) = requests_in_current_lora(t) / max(1, request_count(t))
context_repeat(t)= min(1, (requests_in_context(t) - 1) / max(1, context_requests - 1))
```

对候选 chunk 内所有 term 取平均：

```text
Shared = mean(support(t) * cross_lora(t))
Private = mean(support(t) * concentration(t) * (1 - cross_lora(t)))
Context = mean(context_repeat(t))
R = 0.55 * Shared + 0.45 * Context
```

当前用户关注分数使用廉价 attention proxy：

```text
F = min(1,
        0.50 * task_term_overlap
      + 0.30 * chunk_end_position_ratio
      + 0.20 * role_marker)
```

`role_marker` 检测 `User:`、`Question:`、`Profile:`、`State:` 等结构标记。

## 2. 动态长度和评分

```text
discount = min(160, 96*R + 80*F + 64*Private)
T_filter = clamp(round(256 - discount), 96, 512)
```

并列请求首先满足预取预算：

```text
T_base = clamp(ceil(total_prefix_tokens / max_prefetch_chunks), 96, 512)
T = max(T_base, T_filter)
```

连续请求使用 `T_base=192`，超过 512 token 时为 256。实验第一次运行发现 ShareGPT
被关注分数过度切细，因此最终版本加入硬保护：并列请求或连续前缀超过 512 token 时，
`T=T_base`，分类分数只选择边界，不再缩小 chunk。

最终评分为：

```text
score_v3(s,e)
  = |L(s,e) - T|
  + 10 * future_nodes
  + 80 * prefetch_overflow
  + 32 * min(Private, max(R,F))
  - 64 * R
  - 48 * Private
  - 56 * F
  - 48 * reuse_boundary
  - 18 * natural_or_keyword_boundary
```

混合惩罚鼓励 LoRA-private 与 shared/focus 文本在不同 chunk 中结束。得分最低的候选
被选中；没有候选时按 `T_base` fixed fallback。

## 3. “常备”的实际含义和限制

v3 会把分类角色写入 `prefix_method_chunk_roles`，例如
`lora_private_resident`。但当前 llama.cpp 实验运行时没有“按 chunk 永久 pin 某个 LoRA
variant”的接口，因此 resident 目前表示“隔离成稳定节点、便于同 LoRA 再命中”，不是
物理上的永不淘汰。真正实现常备还需要 admission/pinning policy 配合。

更重要的是，当前六个 workload 的公共前缀主要是共享 system、对话历史或 source
article；LoRA 特有 task 通常位于 `common_prefix_text` 之后。最终 role 统计为：

| 数据集 | shared reuse | user focus | cold bulk | LoRA-private resident |
|---|---:|---:|---:|---:|
| MSC | 1 | 4 | 0 | 0 |
| Taskmaster | 0 | 5 | 0 | 0 |
| LMSYS-33K | 1 | 5 | 1 | 0 |
| ShareGPT52K | 6 | 9 | 1 | 0 |
| XSum | 6 | 4 | 0 | 0 |
| OPUS-100 | 48 | 52 | 0 | 0 |

因此这批数据能测试 shared/focus filter，不能证明 LoRA-private resident 的收益。要验证
该项，需要在不改变语义的前提下把真实 LoRA system prompt 放入 prefix，或构造一个
带显式 per-LoRA system segment 的 workload。

## 4. 具体例子

### 4.1 ShareGPT 长上下文

一个约 1150-token 前缀最终切为：

```text
[251, 262, 259, 250, 129]
roles = [shared_reuse, shared_reuse, user_focus, user_focus, user_focus]
```

这是单条最长示例；全体 5 请求平均仍为 3.2 chunks。第一次无保护版本平均为 4.4
chunks、节点峰值 20；加入 `total_tokens>512` 保护后，平均回到 3.2、节点峰值 10。

### 4.2 XSum 并列长文章

602-token 文章切为：

```text
[303, 299]
roles = [user_focus, user_focus]
```

它与 v2 相同，因为两 chunk 预取预算比分类软分数更强。

### 4.3 OPUS-100 短并列源

第一条 52-token 源被标为单个 `user_focus` chunk；随着因果 profile 积累，后续相同
source pair 会逐渐被标为 `shared_reuse`。所有 v3 消融结构相同，所以 100 请求
TTFT 差异主要是运行噪声。

假设未来 workload 中存在只在 LoRA-A 重复的：

```text
You are the medical-coding adapter. Follow ICD-10 output rules.
```

且它至少出现 4 次、没有出现在其它 LoRA，则 `support≈1`、`cross_lora=0`、
`concentration≈1`，Private 接近 1，会被隔离为 `lora_private_resident`。这是算法支持的
目标情形，但不是当前数据实际观测到的情形。

## 5. 消融方法

| 方法 | 说明 |
|---|---|
| Adaptive reuse v2 | 同进程族 reference |
| LoRA-focus filter v3 | 完整 R/Private/F 分类和成本保护 |
| v3 no LoRA-private | 令 Private=0 |
| v3 no user-focus | 令 F=0 |
| v3 cost only | 令 R=Private=F=0，只保留长度、节点、预取和边界成本 |

## 6. 在线结果

| 数据集 | v2 | v3 full | No private | No focus | Cost only |
|---|---:|---:|---:|---:|---:|
| MSC | +64.78% +/- 9.38 | +67.03% +/- 9.60 | +65.67% +/- 8.09 | +67.31% +/- 7.83 | +67.52% +/- 9.07 |
| Taskmaster | +62.21% +/- 14.31 | +62.92% +/- 14.87 | +64.12% +/- 12.61 | +65.72% +/- 14.09 | +63.26% +/- 14.82 |
| LMSYS-33K | -3.48% +/- 75.33 | -33.20% +/- 125.62 | -38.35% +/- 132.87 | -34.86% +/- 129.17 | -36.54% +/- 129.71 |
| ShareGPT52K | +45.34% +/- 14.12 | +14.68% +/- 62.48 | +17.82% +/- 63.92 | +14.73% +/- 67.11 | +18.22% +/- 62.88 |
| XSum | -251.26% +/- 228.35 | -218.84% +/- 203.98 | -243.11% +/- 223.07 | -216.94% +/- 198.69 | -241.96% +/- 219.34 |
| OPUS-100 | +40.57% +/- 2.41 | +40.39% +/- 2.70 | +41.97% +/- 2.48 | +40.05% +/- 2.59 | +40.71% +/- 2.63 |

## 7. 结论

1. v3 在 MSC、Taskmaster、OPUS 上保持正收益，但没有显著超过 v2；CI 均重叠。
2. 长上下文保护消除了 ShareGPT 的过度节点增长，但 5 请求 TTFT 仍只有 `+14.68% +/- 62.48`，不能声称 v3 优于 v2。
3. LMSYS 回访稀疏，filter 会根据少量 term 统计改变边界，方差比 v2 更大，不适合直接启用。
4. XSum 由两 chunk 预算和跨 LoRA 物化主导，文本分类无法解决系统级瓶颈。
5. No private 与 full 接近不是 private 信号“无用”，而是当前 prefix 中根本没有被分类为 private 的真实段。
6. 下一版更合理的部署方式是 gating：仅当 profile 有足够支持样本、预测节点不增加且 `R/F` 超过阈值时启用 filter，否则退回 v2。

## 8. 其它可选机制

- **真实 attention**：可用离线采样得到 layer/head 聚合注意力，再训练小分类器；在线直接取 attention 会先付出 prefill 成本。
- **TextRank/RAKE**：适合产生关键词边界，但 1_1 已显示“更多关键词边界”不等于更低 TTFT。
- **小型文本分类器**：可以预测 private/shared/focus/cold，质量可能高于规则，但需要独立训练集和推理开销。
- **KV admission + pinning**：这是实现 LoRA-private 常备的真正系统组件，切分器只能提供边界和标签。

## 9. 图和逐数据集报告

- [MSC v3 图](output/focus_filter_v3_5/msc_continuous_prefix_method_effect_3.png)
- [Taskmaster v3 图](output/focus_filter_v3_5/taskmaster_continuous_prefix_method_effect_3.png)
- [LMSYS-33K v3 图](output/focus_filter_v3_5/lmsys_continuous_prefix_method_effect_3.png)
- [ShareGPT52K v3 图](output/focus_filter_v3_5/sharegpt_continuous_prefix_method_effect_3.png)
- [XSum v3 图](output/focus_filter_v3_5/xsum_parallel_prefix_method_effect_3.png)
- [OPUS-100 v3 图](output/focus_filter_v3_100/opus100_parallel_prefix_method_effect_3.png)

原始 CSV、日志、分类后的 workload 和逐数据集 Markdown 位于
`output/focus_filter_v3_5/` 与 `output/focus_filter_v3_100/`。
