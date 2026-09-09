# 前缀切分算法与六个数据集结果分析

## 1. 文档范围

本文说明 `lora-base-test7` 中五种前缀切分方法的实际实现，并解释六个真实数据集上的结构指标和 TTFT 结果。

结果规模需要先区分：

- `MSC`、`Taskmaster`、`LMSYS-33K`、`ShareGPT52K`、`XSum` 的组合图来自 `5 requests x 5 methods` 的 smoke 实验，用于验证构造、切分和运行链路。5 条请求的置信区间很宽，不能作为最终平均性能结论。
- `OPUS-100` 的组合图来自 `100 requests x 5 methods` 的完整实验，结论相对稳定。
- 所有实验中的物理 GGUF LoRA 主要用于触发 LoRA 切换和 KV 复用路径，不代表这些 adapter 真正针对对应数据集训练过；因此本文讨论的是缓存/调度行为，不是生成质量。

相关图和报告：

- [MSC 图](output/all_prefix_5/msc_continuous_prefix_method_effect.png) · [MSC 报告](output/all_prefix_5/MSC_CONTINUOUS_PREFIX_METHOD_RESULTS.md)
- [Taskmaster 图](output/all_prefix_5/taskmaster_continuous_prefix_method_effect.png) · [Taskmaster 报告](output/all_prefix_5/TASKMASTER_CONTINUOUS_PREFIX_METHOD_RESULTS.md)
- [LMSYS-33K 图](output/all_prefix_5/lmsys_continuous_prefix_method_effect.png) · [LMSYS 报告](output/all_prefix_5/LMSYS_CONTINUOUS_PREFIX_METHOD_RESULTS.md)
- [ShareGPT52K 图](output/all_prefix_5/sharegpt_continuous_prefix_method_effect.png) · [ShareGPT 报告](output/all_prefix_5/SHAREGPT_CONTINUOUS_PREFIX_METHOD_RESULTS.md)
- [XSum 图](output/all_prefix_5/xsum_parallel_prefix_method_effect.png) · [XSum 报告](output/all_prefix_5/XSUM_PARALLEL_PREFIX_METHOD_RESULTS.md)
- [OPUS-100 100 请求图](output/opus100_prefix_100/opus100_parallel_prefix_method_effect.png) · [OPUS-100 报告](output/opus100_prefix_100/OPUS100_PARALLEL_PREFIX_METHOD_RESULTS.md)

## 2. 公共定义

### 2.1 前缀、chunk 和节点

对每条请求，workload 生成器先写出 `common_prefix_text`。它是模型在真正的 task 指令或答案前必须读取的上下文，例如：

```text
Continue this long-term conversation naturally.
Participant profiles: ...
Earlier session: ...
User: ...
Assistant: ...
```

切分方法把这段文本拆成 `chunk_0, chunk_1, ...`。运行器把这些 chunk 写入 `prefix_segments`，并用 `system-v2` 运行时执行。前缀树中的一个节点表示从父节点继续追加一个 chunk 后形成的 KV 前缀：

```text
root -> H(chunk_0) -> H(H(chunk_0), chunk_1) -> ...
```

两个请求只有在从 root 到某个节点的完整字节前缀一致时，才可以复用这个节点；仅仅“语义相似”不能命中。

### 2.2 估算 token 数

当前离线切分没有调用模型 tokenizer，而是使用与 `build_real_workloads.py` 相同的轻量估算：

```text
ascii_count = ASCII 字符数量
estimated_tokens(text) = ceil(ascii_count / 4 + non_ascii_count / 1.5)
```

结果至少为 1。它适合比较切分趋势，但不是 Qwen tokenizer 的精确 token 数，因此图中写的是 `estimated tokens`。

### 2.3 自然边界

`natural_boundaries` 收集三类位置：

1. workload 已标记的 segment 结束位置，例如 `shared_system`、`persona_context`、`conversation_history`、`source_article` 的结束位置；
2. 句号、问号、感叹号、右引号附近的句末位置；
3. 换行位置和整段文本末尾。

自然边界不是模型理解出来的语义边界，而是文本结构边界。中文或清洗后的非 ASCII 文本仍然按照字符位置计算。

### 2.4 Runtime 的命中定义

对在线请求，脚本把下面两种 token 数取最大值：

```text
chunk_hit_tokens = Runtime 返回的已命中 chunk token
legacy_hit_tokens = same_lora_variant_hit ? prefix_tokens : 0
reused_tokens = max(chunk_hit_tokens, legacy_hit_tokens)
```

因此：

- `Runtime request hit`：`reused_tokens > 0` 的请求比例；
- `Runtime token coverage`：`sum(reused_tokens) / sum(prefix_tokens)`；
- `Complete prefix hit`：该请求的 `common_prefix_hash` 在同一结构分析序列中已经出现过的比例；
- `TTFT change`：

```text
(matched full-prefill TTFT - online TTFT) / matched full-prefill TTFT
```

正数表示 TTFT 降低，负数表示 TTFT 增加。在线点估计采用逐请求配对相对变化的平均值，置信区间用固定 seed 的 bootstrap mean CI 计算。

## 3. 五种切分算法

### 3.1 Fixed 64/128

这是按 workload segment 类型使用两级固定目标长度的算法：

```text
shared_system      -> 64 estimated tokens
其它 segment         -> 128 estimated tokens
```

算法步骤：

1. 先按 `prefix_segments` 保留 system/context 的结构边界；
2. 对每个 segment 调用 `fixed_chunks`；
3. `fixed_chunks` 根据剩余文本的估算 token 数换算字符结束位置；
4. 如果向前扩展会超过目标，就回退一个字符；如果未超过目标，就逐字符扩展到最大可用长度；
5. 最后一段不足目标长度时原样保留。

例：MSC 的一条约 256-token 前缀，实际得到：

```text
[37, 93, 127]
```

第一个 37-token chunk 是较短的 system/persona 结构，后两个接近 128-token 的 chunk 保存连续对话历史。它的优点是边界数量可预测、节点大小接近运行时的复用粒度；缺点是文本可能在句子中间断开。

### 3.2 Fixed 192/384

结构与 Fixed 64/128 相同，只是目标长度更大：

```text
shared_system      -> 192 estimated tokens
其它 segment         -> 384 estimated tokens
```

例如 LMSYS 一条约 556-token 前缀得到：

```text
[19, 384, 153]
```

更大的 chunk 减少节点数和树管理开销，但单个 chunk 的复用条件更严格：只要 chunk 内有一个字符不同，整个 chunk 就不能命中。对于短请求，大目标还可能退化成一个大 chunk，降低可复用的中间节点数量。

### 3.3 Sentence

Sentence 不是简单地每 128 token 截断，而是：

1. 用 segment 末尾、句末标点和换行生成自然单元；
2. 依次把自然单元累加到当前 chunk；
3. 累加后超过 128 token 时关闭当前 chunk；
4. 单个自然单元超过 `1.5 x 128 = 192` token 时，先关闭当前 chunk，再对这个大单元使用 fixed 128 切分。

例如 XSum 的约 602-token 文章前缀，Sentence 产生：

```text
[126, 113, 113, 112, 125, 15]
```

这些边界都落在句子/segment 末尾，所以自然边界覆盖率为 1.00；但它仍然有 6 个节点，且长文章的后半部分必须逐节点构建或预取。

### 3.4 Keyword

Keyword 在自然单元之上增加显式任务/对话关键词触发点。当前正则关注：

```text
user, assistant, system, task, context, article, question, answer,
session, profile, instruction, dialogue, summary, headline, translate,
source, target, however, meanwhile, therefore, because, finally, next, then
```

算法步骤：

1. 先得到自然单元；
2. 如果单元超过 288 token，先关闭当前 chunk，并用 192-token fixed chunks 拆分；
3. 否则把单元追加到当前 chunk；
4. 如果新单元以关键词开头，或者追加后超过 192 token，就关闭当前 chunk；
5. 最后保留未满的 chunk。

例如 MSC 同一条约 256-token 前缀得到：

```text
[144, 9, 18, 12, 16, 18, 11, 21, 11]
```

关键词把 `User:`、`Assistant:` 等边界切得非常细。这样自然边界覆盖率仍然高，但节点数量从 2~3 个增加到 9 个，树管理和调度开销可能抵消命中收益。Keyword 不是“关键词语义摘要”，只是关键词触发的边界规则。

### 3.5 Reuse-aware

Reuse-aware 试图把历史可复用的前缀终点纳入切分候选：

```text
candidates = natural_boundaries union reuse_points
```

其中 `reuse_points` 是同一 `context_id` 内、当前文本的严格前缀长度。对每个 chunk 起点：

1. 枚举候选终点；
2. 只保留估算长度在 64~192 token 的候选；
3. 代价为 `abs(size - 128)`；
4. 如果终点属于历史复用点，代价减去 72 的 reuse bonus；
5. 选择代价最小的终点；
6. 没有可行候选时，退回普通 fixed 128。

当前 v1 的候选得分可以明确写成：

```text
设当前 chunk 起点为 s，候选终点为 e：
L(s,e) = estimated_tokens(text[s:e])

可行条件：64 <= L(s,e) <= 192

score_v1(s,e)
    = |L(s,e) - 128|
      - 72 * I(e in reuse_points)

e* = argmin score_v1(s,e)
```

其中 `I(condition)` 是指示函数，条件成立取 1，否则取 0。`72` 是历史 prefix 终点的固定奖励，单位与 token 偏差项一致。该公式没有显式考虑 context 回访概率、节点总数、预取预算或 Host/GPU 搬运成本，这也是 v2 需要优化的部分。

例如 MSC 约 256-token 前缀得到：

```text
[144, 112]
```

它比 Keyword 少很多节点，并且优先对齐历史终点；但它依赖历史请求顺序。对于没有 exact prefix reuse 的连续数据集，`reuse_points` 为空，Reuse-aware 主要退化为自然边界加 128-token 近似，不会凭空创造复用。

## 4. 从切分到 Runtime 的实验流程

每个方法的在线实验执行以下流程：

```text
原始 workload
    -> 读取前 request_limit 条请求
    -> 对每个 context 计算 reuse_points
    -> split_prefix(row, method_id, reuse_points)
    -> 把每个 chunk 写入 prefix_segments
    -> 使用相同的 system-v2 + tiered/oracle runtime
    -> 每个请求与同一 cell 内的 full-prefill baseline 配对
    -> 汇总 TTFT、命中率、节点数和 Host KV
```

前缀方法的 runtime 参数使用 4096-token 的通用 runtime chunk，同时将离线切出的边界写入 `prefix_segments`；因此图中的方法差异主要来自离线边界，而不是更换模型或更换 cache backend。公共配置仍限制 `max-prefetch-chunks-per-lora=2`，所以长前缀不能因为使用 oracle 就无限预取。

## 5. 数据集构造与复用概率

| 数据集 | 形式 | 请求/上下文构造 | 100 请求复用统计 |
|---|---|---|---|
| MSC | 连续 | 多个五 session 对话，按会话历史逐步增长；包含离开、返回和 LoRA 切换 | 4 contexts；context reaccess `96%`；return-after-gap `45.8%`；append-only `100%`；same-LoRA reread `33.3%` |
| Taskmaster | 连续 | 多个独立任务对话，追加历史和 task-state | 7 contexts；context reaccess `93%`；return-after-gap `38.7%`；append-only `100%`；same-LoRA reread `30.1%` |
| LMSYS-33K | 连续 | 41 个普通多轮聊天上下文，按选定对话逐步增长 | context reaccess `59%`；return-after-gap 仅 `6.8%`；append-only `100%`；same-LoRA reread `23.7%` |
| ShareGPT52K | 连续 | 17 个真实对话，保留相邻 user-assistant 轮次并追加历史 | context reaccess `83%`；return-after-gap `28.9%`；append-only `100%`；same-LoRA reread `30.1%`；平均前缀较长 |
| XSum | 并列 | 每篇文章派发给 summarization、QA、headline rewrite 三种逻辑 LoRA | 34 contexts；66 个跨 LoRA pair；exact-prefix reuse `66%`；每个文章前缀通常数百 token |
| OPUS-100 | 并列 | 每个英文源派发给日文和另一种语言 LoRA | 50 contexts；50 个跨 LoRA pair；exact-prefix reuse `50%`；平均前缀约 `59` token |

注意：MSC 和 Taskmaster 的 `exact_prefix_reuse_rate=0` 并不表示没有可复用内容。它们是 append-only：第 2 个请求包含第 1 个请求，但完整 prefix hash 不同；可复用的是已有 chunk/祖先节点，而不是“完整 prefix hash 再次出现”。

## 6. 为什么 MSC 和 Taskmaster 效果好

### 6.1 MSC

MSC 的 workload 明确模拟“连续对话、离开后返回、LoRA 中间切换”：

```text
R0 = system + persona + session-1 history
R1 = system + persona + session-1 history + new turn
R2 = system + persona + session-1 history + new turn + new turn
```

所以后续请求一定包含前一轮的文本前缀。100 请求统计中 96% 请求回到已有 context，且 100% 是 append-only。5 请求图中：

- Fixed 64/128：token coverage `81.0%`，TTFT change `+55.65%`；
- Sentence：token coverage `86.0%`，TTFT change `+63.19%`；
- Reuse-aware：token coverage `86.0%`，TTFT change `+61.22%`；
- Sentence/Reuse-aware 的节点峰值只有 7，Keyword 则有 9 个节点且 TTFT 只有 `+24.18%`。

原因是：

1. 历史文本增长是稳定的，父节点在后续请求中仍然存在；
2. 平均前缀约 198 token，不会像 XSum/ShareGPT 那样需要构建十几个长 chunk；
3. 句子和 reuse-aware 能把边界放在完整历史段或较稳定的自然位置，命中 token 多；
4. cache build 之后只需追加新的一小段，而不是重新 prefill 全部历史。

代表性例子中，约 256-token 前缀被 Reuse-aware 切成 `[144, 112]`。下一轮只增加最后一个对话单元时，第一个祖先节点可以直接返回，新增计算集中在最后一段。

### 6.2 Taskmaster

Taskmaster 的文本模式是“任务状态不断增长”：

```text
system + task instruction + user/assistant turns + structured task state
```

100 请求统计中 93% 请求回到已有 context，100% append-only。5 请求图中：

- Fixed 64/128：token coverage `87.8%`，TTFT change `+50.29%`；
- Fixed 192/384：token coverage `87.8%`，TTFT change `+54.59%`；
- Sentence：token coverage `95.0%`，TTFT change `+60.56%`；
- Reuse-aware：token coverage `95.0%`，TTFT change `+56.03%`。

代表性约 258-token 前缀的切法为：

```text
Fixed 64/128 -> [39, 9, 128, 83]
Fixed 192/384 -> [39, 9, 211]
Sentence -> [128, 128, 3]
Reuse-aware -> [128, 128, 3]
```

Taskmaster 的自然单元通常比普通闲聊短，Sentence/Reuse-aware 可以用 2~3 个接近 128-token 的节点覆盖大部分历史；新请求只需计算新增 task-state。它比 MSC 更接近“状态追加”而不是跨 session 的长文本拼接，所以在小样本中收益也较稳定。

## 7. 后四个数据集的原因分析

这里需要纠正一个容易造成误解的说法：后四个数据集并不都差。`LMSYS-33K` 的收益不稳定、`ShareGPT52K` 在长请求上容易回退、`XSum` 明显回退；但 `OPUS-100` 是短前缀并列复用，100 请求结果整体是好的。

### 7.1 LMSYS-33K：回访稀疏，复用机会不足

LMSYS 有 41 个 context，100 请求中 context reaccess 只有 `59%`，return-after-gap 只有 `6.8%`。很多新 context 的前缀只有通用 system 文本：

```text
Continue the ordinary chat conversation and answer the latest user message.
```

它约 19 token，单独命中它带来的收益很小；一旦进入某个聊天，后续 prefix 又可能突然变成 291、556 token 的长历史。代表性 556-token 请求的结构为：

```text
Fixed 64/128 -> [19, 128, 128, 128, 128, 25]
Fixed 192/384 -> [19, 384, 153]
Sentence -> [127, 99, 117, 115, 99]
Keyword -> [19, 11, 181, 186, 160]
Reuse-aware -> [127, 131, 118, 127, 55]
```

5 请求图中 token coverage 只有 `13.4%~46.8%`，TTFT change 从 `+16.98%` 到 `-25.93%`，所有 CI 都很宽。原因不是切分器必然错误，而是：

1. 新 context 很多，已有父节点来不及形成稳定热度；
2. 跨请求回访间隔短，系统更像一次性聊天集合而不是同一会话持续追加；
3. 19-token system prefix 的收益不足以支付节点/LoRA 调度成本；
4. 5 条请求样本太小，哪一个长对话刚好被命中会显著改变平均 TTFT。

因此 LMSYS 更适合作为“普通聊天 baseline”，不适合仅凭它证明 prefix cache 的理论收益。

### 7.2 ShareGPT52K：复用率不低，但每次增长太大

ShareGPT 有 17 个真实对话，context reaccess `83%`、append-only `100%`，看上去比 LMSYS 更适合复用；问题在于单轮内容很长。5 请求样本的平均前缀约 `739.8` token，代表性 1150-token 前缀为：

```text
Fixed 64/128 -> [19, 128, 128, 128, 128, 128, 128, 128, 128, 108]
Fixed 192/384 -> [19, 384, 384, 364]
Sentence -> [124, 128, 123, 72, 127, 113, 120, 128, 126, 93]
Keyword -> [19, 190, 173, 25, 108, 13, 176, 71, 160, 153, 66]
Reuse-aware -> [124, 128, 130, 126, 132, 134, 123, 128, 128, 1]
```

5 请求图中 Fixed 192/384 的 token coverage 为 `96.8%`、TTFT change 为 `+65.51%`，是当前 smoke 样本中最好的切法；但 Sentence 和 Keyword 分别出现 `-43.33%`、`-91.81%`。100 请求核心实验还出现过 `foreground_capacity_failed`，说明长历史和多 LoRA variant 会把 GPU/cache 容量推到上限。

具体原因是：

1. 一次 user-assistant 轮次可能引入数百 token，追加后不能只构建一个很小的 suffix；
2. Fixed 64/128、Sentence、Keyword 产生 6~8 个节点，节点数和管理开销迅速增加；
3. 复用的祖先节点虽多，但新增长尾仍然很长，TTFT 主要由长尾 prefill 决定；
4. 长对话会积累多个 LoRA variant，GPU-only 策略可能触发容量不足或淘汰。

所以 ShareGPT 证明的是“高 context reaccess 不等于高端到端收益”：还必须同时控制每次增长的 token 数、节点数量和容量。

### 7.3 XSum：并列复用真实存在，但前缀太长且预取覆盖不足

XSum 是并列结构：同一篇文章分别交给 summarization、main-event QA、headline rewrite。一个请求对例如：

```text
R0 = system + article + summarization task
R1 = system + article + QA task
R2 = system + article + headline task
```

100 请求中有 66 个 exact-prefix cross-LoRA pair，理论复用概率是 `66%`，但每个 article 前缀通常是数百 token。代表性 602-token 文章的切法为：

```text
Fixed 64/128 -> [18, 128, 128, 128, 128, 72]
Fixed 192/384 -> [18, 384, 200]
Sentence -> [126, 113, 113, 112, 125, 15]
Keyword -> [18, 192, 118, 183, 91]
Reuse-aware -> [126, 113, 136, 138, 91]
```

5 请求图中 runtime token coverage 仅 `5.9%~24.9%`，TTFT change 为 `-418.54%` 到 `-642.23%`，Host peak 达到 `49.4~103.9 MiB`。100 请求核心结果同样明显回退。

关键原因：

1. 每个 pair 第一个请求必须完整建立 489~986 token 的文章 prefix；
2. 公共配置 `max-prefetch-chunks-per-lora=2`，oracle 也最多提前覆盖两个 chunk，文章后半部分仍需前台构建；
3. cross-LoRA 切换后，旧 LoRA 的 KV 不能直接作为新 LoRA 的完整执行 KV，只能复用系统允许的 prefix/chunk 部分；
4. article 是长而不变的 source，复用的是“长前缀”，但每个不同 LoRA 仍有自己的 variant、调度和物化成本；
5. 因此结构上的 `complete prefix hit=60%` 并没有转化成相同幅度的 Runtime token hit。

XSum 的结果说明：并列请求的重复概率只是必要条件，不是充分条件；还要考虑共享前缀长度、可预取深度和跨 LoRA 物化开销。

### 7.4 OPUS-100：不是差数据集，而是短前缀并列复用的正例

OPUS-100 每个英文源只派发给两个语言 LoRA。代表性 52-token 请求为：

```text
Fixed 64/128 -> [34, 18]
Fixed 192/384 -> [34, 18]
Sentence -> [52]
Keyword -> [12, 40]
Reuse-aware -> [52]
```

100 请求中有 50 个 exact-prefix pair，Runtime request hit `99.0%`、token coverage `99.1%`。TTFT change 为：

| 方法 | TTFT change |
|---|---:|
| Fixed 64/128 | `+43.70% +/- 2.14` |
| Fixed 192/384 | `-12.46% +/- 13.73` |
| Sentence | `+29.91% +/- 10.02` |
| Keyword | `+22.71% +/- 13.90` |
| Reuse-aware | `+32.60% +/- 11.94` |

OPUS 的收益来自：

1. prefix 很短，完整 source 通常只有一个或两个小 chunk；
2. 第二个语言 LoRA 紧跟在第一个请求后面，exact-prefix reuse 位置清晰；
3. 不需要构建 XSum 那样的长文章 suffix，也不会产生大量 Host KV；
4. 5 种方法即使边界不同，实际覆盖的 token 几乎相同。

Fixed 192/384 这次出现负值，不应被解释为“192/384 算法一定差”：它的请求命中和 token 覆盖仍是 `99.1%`，而 `[-26.84%, +0.62%]` 的 CI 横跨 0。更合理的解释是这次重跑中切换/调度噪声超过了边界差异；需要更多独立重复或固定请求顺序后再判断。

## 8. 综合结论

### 8.1 影响收益的四个因素

可以把端到端收益近似理解为：

```text
收益 ~= 命中的 prefix token 省下的 prefill
       - 未命中 suffix 的 prefill
       - chunk/节点管理开销
       - LoRA 切换与 KV 物化开销
       - 淘汰、Host/GPU 搬运和预取等待
```

MSC/Taskmaster 同时满足“高回访 + append-only + 中等前缀长度 + 少量节点”，所以更容易为正。

LMSYS 缺少稳定回访；ShareGPT 有回访但每次增长太大；XSum 有 exact cross-LoRA pair 但 source 太长且预取深度有限。OPUS-100 同时满足“exact pair + 短 prefix + 紧邻切换”，因此是并列复用的正例。

### 8.2 对实验结论的正确表述

当前数据支持以下表述：

1. 对连续、append-only、上下文中等长度的 MSC/Taskmaster，结构化 prefix chunk 可以显著减少后续请求的 prefill；
2. 对普通聊天 LMSYS，复用收益取决于 context 回访密度，不能只看“多轮对话”标签；
3. 对 ShareGPT，长轮次会让 cache capacity 和 suffix prefill 成为主要瓶颈；
4. 对 XSum，跨 LoRA 的 exact prefix reuse 仍可能因为长 source 和有限预取而端到端回退；
5. 对 OPUS-100，短而精确的并列 prefix 是当前实验中最适合验证复用收益的正例；
6. `5 requests` 图只用于结构和趋势检查，正式论文式结论应使用至少 100 请求、多个请求顺序和多次独立重复。

## 9. 建议的后续对照实验

为了把“切分算法问题”和“工作负载问题”分开，建议下一轮固定 workload，只改变一个因素：

1. XSum 固定 66 个 pair，分别测试 `max-prefetch-chunks-per-lora = 2, 4, 8, 0`；
2. ShareGPT 固定同一批对话，比较原始长轮次和每轮截断到 128/256 token；
3. LMSYS 把 41 个 context 的回访率从 `6.8%` 提升到 `25%/50%`，观察收益何时转正；
4. 每种方法运行 3~5 个不同 request order，报告均值、p95 和配对 CI；
5. 同时记录 `prefix_tokens`、`chunk_hit_tokens`、`cache_nodes`、`family_evictions` 和 `foreground_capacity_failed`，避免只看 TTFT 百分比。
