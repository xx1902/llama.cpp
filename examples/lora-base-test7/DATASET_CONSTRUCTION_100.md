# 六个真实数据集的 100 请求构造说明

本文说明 `test7` 使用的六个真实数据集如何转换成请求 workload，并解释
上下文重读、LoRA 切换、prefix 重复和负 TTFT speedup 的含义。

结论先说：负 speedup 不足以证明数据集有问题。当前 100 请求 workload
均通过结构校验，真实文本和真实 reference 均保留。负 speedup 主要表示
当前缓存策略的构建、复制、跨 LoRA 转换、淘汰或容量管理成本超过了省下的
prefill 成本。不过，这些 workload 仍是系统实验 trace，不是线上真实流量
分布；物理 GGUF LoRA 只是用于模拟切换，并没有针对下面的逻辑任务训练。

## 一、100 请求生成流程

### 1. LSApp 路由生成

请求到达顺序来自本地 LSApp 请求记录，先按照 5 分钟时间片切分：

1. `close_time <= open_time` 的记录原样保留。
2. 持续时间不超过 5 分钟的记录不切分。
3. 超过 5 分钟的记录按连续的 5 分钟区间切分。
4. 单条记录最多取 200 分钟，避免一个长应用无限生成请求。
5. 切分后的记录按照时间排序，取前 100 条作为 `source_trace_100.jsonl`。
6. 每个请求从 LSApp 的 `app_name` 映射到物理 LoRA ID 和 GGUF 路径。

生成命令：

```powershell
D:\anaconda\envs\qwen2.5_vl\python.exe -B `
  examples/lora-base-datasets/build_lsapp_requests.py `
  --request-count 100 `
  --time-step-min 5 `
  --max-duration-min 200 `
  --output-dir examples/lora-base-datasets/output/lsapp_100
```

### 2. 真实数据转换

```powershell
D:\anaconda\envs\qwen2.5_vl\python.exe -B `
  examples/lora-base-datasets/build_real_workloads.py `
  --request-count 100 `
  --route-jsonl examples/lora-base-datasets/output/lsapp_100/source_trace_100.jsonl `
  --lora-groups-json D:\ecnu_experiment\datasets\mobilora_workloads_87_original\lora_groups.json `
  --output-dir examples/lora-base-datasets/output/real_100
```

每个请求包含以下关键字段：

| 字段 | 含义 |
|---|---|
| `request_id` | `0` 到 `99` 的连续请求编号 |
| `context_id` | 真实会话、任务、文章或翻译源的上下文编号 |
| `common_prefix_text` | 可被 KV cache 复用的公共前缀 |
| `prompt` | 公共前缀加当前任务指令的完整输入 |
| `reference_response` | 数据集中的真实 assistant、翻译或摘要文本 |
| `lora_id` / `adapter_path` | LSApp 路由映射出的物理 LoRA |
| `arrival_ms` | 由 LSApp 5 分钟切片顺序映射出的到达时间 |

## 二、概率和重复率的定义

下面的比例是 workload 层面的机会率，不是 llama.cpp 运行时最终命中率。
例如，上下文曾经出现过，并不代表对应 LoRA 的 KV 一定仍在 GPU 上。

设请求总数为 `N=100`。连续 workload 按 `arrival_ms` 排序；并列 workload
按同一个 `context_id` 内的请求顺序计算。

| 指标 | 计算方式 | 说明 |
|---|---|---|
| 上下文重读率 | `context_seen_before / N` | 当前请求的 `context_id` 之前是否出现过 |
| 完整 prefix 重复率 | `exact_prefix_seen_before / N` | 当前 `common_prefix_hash` 是否在同一上下文出现过 |
| 立即同 LoRA 重读率 | `same_lora_as_previous / context_reaccesses` | 当前请求是否紧接着用同一个 LoRA 重读上下文 |
| 切换 LoRA 重读率 | `switched_lora / context_reaccesses` | 重读上下文时是否换了 LoRA |
| 历史同 LoRA 重读率 | `same_lora_seen_before / context_reaccesses` | 当前 LoRA 是否曾在该上下文出现过，不要求相邻 |
| 间隔返回率 | `return_after_gap / lora_transitions` | 离开某 LoRA 后，之后又返回该 LoRA |
| append-only 率 | `append_only_extension / context_reaccesses` | 新 prompt 是否以前一个完整 prompt 为前缀 |

第一条请求没有“之前的请求”，所以不会计入重读分母。并列 workload 中，
`return_after_gap` 和 `append-only` 没有连续对话含义，因此记为 0 或 n/a。

## 三、100 请求统计

| Workload | 形式 | 请求 | 上下文 | pair | 上下文重读 | 完整 prefix 重复 | 立即同 LoRA | 切换 LoRA | 历史同 LoRA | 间隔返回 | append-only |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MSC | 连续 | 100 | 4 | 0 | 96.0% | 0.0% | 33.3% | 66.7% | 79.2% | 45.8% | 100.0% |
| Taskmaster | 连续 | 100 | 7 | 0 | 93.0% | 0.0% | 30.1% | 69.9% | 68.8% | 38.7% | 100.0% |
| LMSYS-33K | 连续 | 100 | 41 | 0 | 59.0% | 0.0% | 23.7% | 76.3% | 30.5% | 6.8% | 100.0% |
| ShareGPT52K | 连续 | 100 | 17 | 0 | 83.0% | 0.0% | 30.1% | 69.9% | 59.0% | 28.9% | 100.0% |
| OPUS-100 | 并列 | 100 | 50 | 50 | 50.0% | 50.0% | 0.0% | 100.0% | 0.0% | n/a | 0.0% |
| XSum | 并列 | 100 | 34 | 66 | 66.0% | 66.0% | 0.0% | 100.0% | 0.0% | n/a | 0.0% |

注意：连续 workload 的完整 prefix 重复率为 0 是设计结果。每一轮都把
新的 user/assistant 回合追加到历史后面，所以它是“增长前缀”，不是完全
相同的 prefix。并列 workload 则相反，同一文章或同一句英文被发送给不同
LoRA，因此有完整 prefix 重复，但没有 append-only 增长。

## 四、各数据集的详细构造算法

### 1. MSC：多 session 连续对话

**源文件**：`datasets/msc/msc/msc/msc_dialogue/session_5/valid.txt`。

算法：

1. 读取 MSC 的多 session JSON 记录。
2. 保留包含足够真实 user/assistant pair 的记录，按源文件顺序选择。
3. 每个源记录建立一个独立 `context_id`，不把不同人物的对话拼接到一起。
4. 将 participant persona 和 session boundary 放在公共前缀中。
5. 对每个真实 user/assistant pair 生成一条请求。
6. 当前 user turn 放入 `prompt`，真实 assistant turn 放入
   `reference_response`。
7. 下一条请求把上一轮完整对话追加到 `conversation_history`。

100 条请求最终来自 4 个 MSC 源记录。上下文重读率 96%，但完整 prefix
重复率 0%；同 LoRA 紧邻重读 33.3%，切换 LoRA 重读 66.7%，历史上曾使用
过同一 LoRA 的重读为 79.2%，离开后返回率为 45.8%。

这代表 MSC 很适合测试“上下文长期存在，但 LoRA 可能离开后再回来”的
场景，不适合把完整相同 prefix 当作主要收益来源。

### 2. Taskmaster-1：带任务状态增长的连续对话

**源文件**：`datasets/taskmaster/TM-1-2019/self-dialogs.json`。

算法：

1. 读取 self-dialogue 记录，要求存在足够的真实 `USER -> ASSISTANT` pair。
2. 每个任务对话单独建立一个上下文。
3. 保留任务 instruction、历史对话和源数据中的槽位/任务状态。
4. 当前 user turn 作为请求任务，真实 assistant turn 作为 reference。
5. 完成一轮后更新序列化 task state，再生成下一条 prompt。
6. 不把电影、订餐、维修、打车等不同任务强行拼成一个上下文。

100 条请求来自 7 个独立任务上下文。上下文重读率 93.0%，完整 prefix
重复率 0%；立即同 LoRA 重读 30.1%，切换 LoRA 重读 69.9%，历史同 LoRA
重读 68.8%，间隔返回率 38.7%，append-only 为 100%。

该 workload 重点测试任务状态增长和 LoRA 返回，不是同 prefix 的并列复用。

### 3. LMSYS-33K：普通连续聊天 baseline

**源文件**：本地 Chatbot Arena parquet 文件。

筛选条件：

- 只保留英文对话；
- 排除被标记为异常或不适合实验的记录；
- 使用有效的 winning model branch；
- 至少包含两个真实 user/assistant pair。

算法：每条源对话独立建立上下文，按 winning branch 生成连续 prompt。每条
新请求追加上一轮 user 和 assistant 文本，因此仍然是 append-only。

100 条请求来自 41 个短上下文。上下文重读率只有 59.0%，是连续数据集中
最低的；立即同 LoRA 重读 23.7%，切换 LoRA 重读 76.3%，历史同 LoRA
重读 30.5%，间隔返回率 6.8%，append-only 为 100%。

因此 LMSYS 是低复用的普通聊天 baseline。若该数据集上的缓存策略收益不高，
不能据此判断缓存实现错误，因为大部分请求根本没有足够长的同上下文复用。

### 4. ShareGPT52K：真实通用多轮连续聊天

**源目录**：`datasets/sharegpt52k_dataset`，本地包含 52,180 条对话。

算法：

1. 接受 `human/user` 到 `gpt/chatgpt/assistant/bing` 的相邻 turn pair。
2. 读取 `value`，缺失时回退到 `text`。
3. 去除 HTML 标签并反转义 HTML entity，不重写正文语义。
4. 每条源对话最多取 12 个 pair，避免单个极长 transcript 独占 trace。
5. 每条源对话独立建立 `context_id`，不同对话之间不制造共享 prefix。
6. 后续请求保留之前完整的 user/assistant 历史。

100 条请求来自 17 个上下文。上下文重读率 83.0%，完整 prefix 重复率 0%；
立即同 LoRA 重读 30.1%，切换 LoRA 重读 69.9%，历史同 LoRA 重读 59.0%，
间隔返回率 28.9%，append-only 为 100%。

ShareGPT 的主要特点是 prompt 长度明显高于 LMSYS。它在 test7 的
GPU-only 运行中有 6 条请求触发 `foreground_capacity_failed`，原因是
物理 KV 超过 `max-cache-tokens=61440`，这是容量压力，不是 JSON 或文本
构造错误。tiered 策略可以通过淘汰继续运行，但会增加大量 materialize、
迁移和重建时间。

### 5. OPUS-100：同一英文源的并列翻译

**源数据**：OPUS-100 的 `en-ja` 与 `de/es/fr/ru/zh` parquet 交集。

算法：

1. 从 `en-ja` 建立 English -> Japanese 映射。
2. 分别扫描 German、Spanish、French、Russian、Chinese 文件。
3. 只保留两边都有的完全相同 English source。
4. 过滤过短、异常、包含 URL 或非自然英文的样本。
5. 每个 English source 生成两个请求：Japanese anchor 和另一种语言 child。
6. 两个请求的 `common_prefix_text` 字节级相同，但 `lora_id` 不同。
7. 翻译结果直接使用 parquet 中的真实目标语言文本。

100 条请求由 50 个 context 和 50 个跨 LoRA pair 组成。上下文重读率和
完整 prefix 重复率均为 50.0%，所有重读都是切换 LoRA。没有 append-only
增长，也没有连续对话意义上的间隔返回。

这是最理想的并列 workload：prefix 短、重复明确、任务边界清晰。test7 中
tiered oracle 的 token 命中率约 99%，TTFT 改善约 42%，说明系统确实能从
这种并列复用中获益。

### 6. XSum：同一文章的并列多任务请求

**源数据**：本地 XSum dataset。

筛选条件：文章长度约 1,600--4,000 字符，summary 长度约 40--500 字符。

算法：

1. 选择真实新闻文章和真实 summary。
2. 对每篇文章建立一个上下文。
3. 为同一文章生成三个任务角色：summary、main-event QA、headline rewrite。
4. 三个角色共享完全相同的 article prefix，但使用不同逻辑 LoRA。
5. summary 使用真实 XSum summary 作为 reference。
6. QA 和 headline 没有对应的官方任务标签，因此使用 XSum summary 作为
   proxy reference；这只用于系统请求，不用于质量结论。

100 条请求由 33 个完整三任务上下文和最后一个只包含 summary 的部分上下文
组成，共 34 个 context、66 个跨 LoRA pair。上下文重读率和完整 prefix
重复率为 66.0%，所有重读都是切换 LoRA。

XSum 的并列机会率很高，但文章 prefix 较长，实际 128-token chunk 的
token 命中率只有约 20%。因此会出现“结构上重复、请求级命中很高，但 TTFT
仍然回退”的情况：大部分文章仍需 materialize 或重新构建，跨 LoRA KV
转换成本也被长 prefix 放大。

## 五、为什么会出现负 speedup

speedup 的定义是：

```text
speedup = 100 * (full_prefill_TTFT - cache_path_TTFT) / full_prefill_TTFT
```

所以负数表示 cache path 更慢，不表示数据集损坏，也不表示“减少了负的
时间”。常见原因如下：

1. **机会率低**：LMSYS 的上下文重读只有 59%，很多请求是新上下文。
2. **完整 prefix 不重复**：连续 workload 虽然 append-only 为 100%，但每轮
   都增长，不能直接命中完整旧 prefix。
3. **LoRA 切换成本**：连续 workload 的切换重读率在 66.7%--76.3%，并列
   workload 更是 100%。需要构建或转换 child-LoRA 的 KV。
4. **token 覆盖不足**：XSum 结构上有 66% prefix 重复，但实际 chunk 命中
   只有约 20%，未命中的长 suffix 仍然要 prefill。
5. **缓存容量不足**：ShareGPT 的 GPU-only 有 6 条请求触发
   `foreground_capacity_failed`；长上下文会迅速消耗 61,440 token 配额。
6. **分层存储开销**：host full、host delta、GPU materialize、delta 构建、
   eviction 和后台队列都会增加 TTFT 或 wall time。
7. **并行实验的 baseline 差异**：并列请求的 child LoRA 不是同一物理 KV
   variant；若没有可用 delta，系统仍可能执行 full-KV build。

因此，“上下文重读率高”只是必要条件，不是 TTFT 获益的充分条件。至少还要
同时观察 `prefix token hit rate`、`same/cross-LoRA`、`physical_cache_tokens`、
淘汰次数、`delta_build_ms` 和 `materialize_ms`。

## 六、test7 已完成的 100 请求实验

核心矩阵命令：

```powershell
D:\anaconda\envs\qwen2.5_vl\python.exe -B `
  examples/lora-base-test7/run_experiments.py `
  --request-limit 100 `
  --strategies legacy_exact,gpu_only_128,tiered_oracle_128 `
  --output-dir examples/lora-base-test7/output/core_100_matrix
```

结果文件：

- [RESULTS.md](output/core_100_matrix/RESULTS.md)：总体结果和每个数据集结论。
- [DIAGNOSIS.md](output/core_100_matrix/DIAGNOSIS.md)：逐 cell 的有效请求数、
  TTFT、命中率、缓存峰值、模式和失败原因。
- `aggregate_results.csv`：策略汇总。
- `diagnostic_cells.csv`：诊断表格原始数据。

本次核心矩阵中 17 个 cell 成功，ShareGPT/GPU-only 有 1 个失败 cell，
具体是 100 条请求中的 6 条容量失败。完整 11 策略矩阵命令保留在
`examples/lora-base-test7/README.md`，但在当前 GPU 和长上下文配置下需要
数小时，不应把未完成的策略写成已完成结果。

## 七、复现实验的注意事项

1. 这些百分比是当前 100 请求 trace 的描述统计，不是公开数据集的总体概率。
2. 若要估计真实线上复用概率，需要更多用户、更多时间窗口和多次随机抽样。
3. 连续和并列 workload 必须分开报告，不能把 append-only 和 exact-prefix
   重复率合成一个“总复用率”。
4. 物理 LoRA 与逻辑任务不匹配，所以只能解释缓存/切换/KV 行为，不能解释
   生成质量。
5. XSum 的 QA/headline reference 是代理文本，不能用这批请求计算任务质量。

