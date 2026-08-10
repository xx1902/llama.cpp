# 前缀切分与 KV 缓存评分整理

本文是 `lora-base-test4` 当前实现的内部实验整理，并说明 `lora-base-test5` 与前缀切分的关系。文档用于后续人工复核和实验设计，不代表已经完成的实现。

## 结论先行

1. 当前切分点不是由得分决定的。切分点由前缀语义段边界和固定 token 步长决定：`shared_system=64`、`user_context=128`、其他段使用 `prefix_chunk_tokens=128`。
2. 当前得分是缓存家族的保留、淘汰和后台 delta 压缩优先级得分，不是“这个位置值不值得切分”的得分。
3. `test4/test5` 默认使用 `Qwen2.5-1.5B-Instruct-f16.gguf`，基础模型为 F16。Q8 只用于跨 LoRA 的 KV delta，不是基础模型量化。
4. 现有实验输出显示切分策略仍有明显优化空间：在线回放中前缀 token 复用率约 10.2%，完整前缀命中为 0；48 次淘汰全部因 `max_cache_nodes=8` 触发。
5. 下一步应先把“候选切分评分”和“缓存淘汰评分”分开，并将成本改成实测的毫秒、字节和重建误差，再调权重。

## 代码路径

| 环节 | 当前实现 | 位置 |
| --- | --- | --- |
| 生成前缀语义段 | 生成 `shared_system`、`user_context`；连续对话数据集还可生成 `assistant_history` | `examples/lora-base-test4/build_87_lora_workloads.py:499`；`examples/lora-base-datasets/build_lsapp_requests.py:360` |
| 前缀/完整 prompt 分词 | 分别分词，要求 `common_prefix_text` 是完整 prompt 的精确 token 前缀 | `examples/lora-base-test4/lora-base-test4.cpp:1036` |
| 语义段 token 边界 | 对每个累计段文本重新分词，与完整 prefix 做 token 级公共前缀匹配 | `examples/lora-base-test4/lora-base-test4.cpp:1090` |
| 前缀树建路径 | 每个段按固定步长生成累计前缀节点 | `examples/lora-base-test4/lora-base-test4.cpp:2500` |
| GPU 前缀命中 | 沿路径从深到浅找同 LoRA 的 `gpu_full` variant | `examples/lora-base-test4/lora-base-test4.cpp:3382` |
| 缺失节点填充 | 将请求前缀 eval 后，为路径上的每个节点保存一个 variant | `examples/lora-base-test4/lora-base-test4.cpp:2889` |
| 家族评分 | 计算频率、预测、深度、扇出、近期性、段类型和成本 | `examples/lora-base-test4/lora-base-test4.cpp:2219` |
| 淘汰/降级 | 先处理 delta 降级和 full KV offload，再按最低家族分数淘汰叶节点 | `examples/lora-base-test4/lora-base-test4.cpp:2318`、`2419` |
| delta 验证与压缩 | 先 probe，再构建 Q8 delta，按采样比例做重建校验 | `examples/lora-base-test4/lora-base-test4.cpp:2612` |

## 当前前缀切分流程

### 1. 数据集先定义“可共享的前缀”

请求记录中同时保存：

- `common_prefix_text` 和 `common_prefix_hash`；
- `prefix_segments`，例如 `shared_system`、`user_context`、`assistant_history`；
- `prompt = common_prefix_text + task`；
- `estimated_common_prefix_tokens`，仅用于数据集侧估计，不参与运行时切分决策。

`test5` 的 delta pair 要求 anchor 和 child 的 `common_prefix_text` 完全相同；因此它只能验证“同一前缀、不同 LoRA”的 delta 路径，不能覆盖连续对话中前缀逐轮增长的场景。`examples/lora-base-datasets/README.md` 已明确记录了这一限制。

### 2. 运行时做精确 token 校验

运行时对 `common_prefix_text` 和完整 `prompt` 分词，要求：

```text
tokens(full)[0 : len(tokens(prefix))] == tokens(prefix)
```

如果不成立，请求被判定为 `tokenize_or_prefix_failed`。这一步是必要的，因为字符级公共前缀不能保证 token 边界稳定。

### 3. 段内固定步长切分

`ensure_chunk_path` 按 `prefix_segments` 遍历：

- `shared_system` 使用 `system_chunk_tokens`，默认 64；
- `user_context` 使用 `context_chunk_tokens`，默认 128；
- 其他段使用 `prefix_chunk_tokens`，默认 128；
- 每个节点保存从 token 0 到当前深度的累计 prefix，而不是只保存新增 chunk。

例如某个 prefix 的 token 数为 208，system 段结束于 18，则可能形成深度 `18 -> 146 -> 208`。因此 18 不是根据评分选出的断点，只是语义段结束；146 和 208 是固定步长产生的断点。

### 4. 命中与构建

请求到达后先从路径末端向前寻找同 LoRA 的 GPU full variant：

- 命中：复制该节点到请求序列，跳过已缓存前缀的 prefill；
- 未命中：对缺失部分 eval，然后把路径中缺少的节点全部加入缓存；
- host delta 当前不会在普通前台命中路径直接 materialize，主要由预取或后台路径恢复。

因此当前实现是“先按固定规则产生节点，再在容量压力下淘汰”，而不是“先评分再决定是否创建节点”。

## 当前评分模型

对一个节点的 anchor family，定义：

- `F`：所有 variant 的衰减访问频率之和；
- `P`：所有 variant 的预测概率之和；
- `N_delta`：可用 delta child 数；
- `D`：节点深度 token 数；
- `A`：距上次访问的请求数；
- `G/H`：GPU/host 占用 MB；
- `M`：历史 materialize 时间 ms。

默认权重下，代码计算：

```text
segment = 8 (shared_system), 5 (user_context), 1 (other)

benefit = 4 * log1p(F)
        + 20 * P
        + 0.002 * D
        + 3 * N_delta
        + 4 / (1 + A)
        + 1 * segment

cost = 0.05 * (G + H) + 0.0005 * M

score = (benefit - cost) / max(1, G + H)  // family_normalize_by_mb=true
```

这个分数目前用于：

1. `family_evict`：容量不足时淘汰最低分的叶节点；
2. `gpu_full_to_host`：GPU 压力下选择低价值 full variant 做 host offload；
3. `tier3_delta_demote`：优先将有 delta 的 GPU variant 降为 host delta；
4. 后台 delta 压缩队列的优先级。

它不是切分候选的收益函数，不能回答“在 token 146 处增加一个 checkpoint 是否值得”。

## 量化与正确性

### 基础模型和 LoRA

`test4` 和 `test5` 的默认模型路径都是 `Qwen2.5-1.5B-Instruct-f16.gguf`，并通过 `n_gpu_layers=99` 尽可能放到 GPU。LoRA 适配器通过 `llama_adapter_lora_init` 加载；代码没有对基础模型或 LoRA 再做 Q4/Q8 模型量化。适配器文件名中的 `f16` 只说明对应文件的存储格式，实际配置仍应以 GGUF metadata 为准。

### 跨 LoRA KV delta

对于同一 token 前缀的 anchor/child KV，代码分别处理 K 和 V：

```text
scale[t] = max(abs(delta[t, :])) / 127
q8[t, j]  = clip(round(delta[t, j] / scale[t]), -127, 127)
reconstruct = anchor + q8 * scale
```

`q8` 为 int8，`scale` 为每个 token 的 FP32 标量。delta 大小为 `q8 bytes + scale bytes`，逻辑节省率按 `full_kv_bytes_equivalent` 计算。当前 test5 的代表性结果为：

| 模式 | 平均 delta 时间 | 平均 TTFT | 逻辑节省率 | 重建 cosine | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| inference-only | 0 ms | 29.04 ms | 0 | 不适用 | 12/12 成功 |
| cuda-sync | 5.78 ms | 28.58 ms | 49.22% | 0.999999 | 12/12 成功 |
| cuda-overlap | 10.86 ms | 25.37 ms | 49.22% | 0.999999 | 12/12 成功 |
| cpu-sync | 619.78 ms | 642.72 ms | 49.22% | 0.999999 | 12/12 成功 |

### 当前质量门槛

delta candidate 需要通过两层门槛：probe 阶段要求 `kv_cos_avg > 0.98` 或 `kv_l2_avg < 0.05`，随后还要求 `kv_cos_avg >= 0.97` 且 `kv_l2_avg <= 0.25`。后台重建校验默认只抽样 10%（`delta_validation_rate=0.1`）。这些阈值应按任务质量指标继续校准，不能只看 KV cosine。

## 当前主要问题

### A. 切分和评分没有闭环

所有候选位置都先建节点，评分只在容量不足时使用。低复用节点会先消耗序列槽位和树节点，再被淘汰，造成建树和淘汰开销。

### B. 节点上限成为主要瓶颈

在 `delta_matrix_debug_sync_30` 中，48 次 cache event 全部是 `family_evict`，reason 全部为 `nodes`。这说明当前样本首先受 `max_cache_nodes=8` 限制，而不是 host delta 1 GiB 或 GPU token 3072 限制。

### C. 评分的量纲失衡

现有事件中，平均 `segment_benefit` 约 5.06，平均 `memory_cost` 约 0.076；prediction benefit 因 `prefetch_policy=none` 始终为 0。固定段奖励会压过真实的内存和重建成本，直接修改原始权重难以解释。

### D. GPU 成本估计过于粗略

当 `materialized_kv_bytes` 不可用时，代码用 `depth_tokens * 1024` 估算 GPU bytes。当前样本中 18 token 的 full KV 等价大小约为 516096 bytes，即约 28 KiB/token，和 1 KiB/token 的 fallback 相差很大。评分可能严重低估长 prefix 的资源占用。

### E. 分数没有体现 delta 的真实收益

host delta 被计入成本，但“相对于完整 child KV 节省了多少字节”和“下次恢复节省了多少时间”没有作为 benefit 进入 family score。这样会把已经压缩成功的 variant 误判为单纯的 host 成本。

### F. test5 尚未覆盖连续增长前缀

test5 只接受 anchor/child 的精确相同 prefix，主要验证 Q8 delta 的 CPU/CUDA sync/overlap。它目前不是 test4 的前缀树或切分策略实现，不能用 test5 结果直接证明动态切分有效。

## 推荐的下一版切分评分

下面是一个候选 V1 方案，先作为离线 ablation 的定义，不应直接当成最终权重。

### 1. 候选位置

- 语义段结束是 hard boundary，不能跨段合并；
- 长段内每 32 或 64 token 生成候选点；
- 候选点必须通过 token 级 exact-prefix 校验；
- 保留最小 chunk（建议 32）和最大 chunk（建议 256）约束，避免产生大量极小节点或过长未命中区间。

### 2. 用“边际收益”而不是累计深度

对节点 `i`，令父节点深度为 `d_parent`，候选新增区间为 `[d_parent, d_i)`。定义：

```text
saved_ms(i) = P(reuse_i) * (prefill_ms(parent, i) - restore_ms(i))
cross_lora_ms(i) = P(cross_lora_reuse_i) * delta_or_prefill_saving_ms(i)

net_gain(i) = saved_ms(i)
            + cross_lora_ms(i)
            + prediction_gain_ms(i)
            + semantic_prior_ms(i)
            - gpu_price_ms(i)
            - host_price_ms(i)
            - build_ms(i)
            - materialize_ms(i)
            - node_overhead_ms(i)
            - quality_risk_ms(i)
```

切分准入使用 `net_gain(i) > admission_threshold`。缓存淘汰再使用 `net_gain / resource_cost` 做 value density；这样“是否创建”和“容量不足时淘汰谁”不会共用一个不透明分数。

### 3. 特征和候选归一化

先把每个特征转换到 `[0, 1]` 或直接转换成毫秒/字节影子价格，再调权重。建议的初始实验权重如下，数值仅用于第一轮 ablation：

| 类别 | 特征 | 初始权重 |
| --- | --- | ---: |
| benefit | 同 LoRA 复用概率 | 0.35 |
| benefit | 可节省的 prefill ms | 0.30 |
| benefit | 跨 LoRA fanout / delta 复用概率 | 0.15 |
| benefit | 预测概率带来的提前准备收益 | 0.10 |
| benefit | 语义边界先验 | 0.10 |
| cost | GPU bytes | 0.35 |
| cost | host delta/full bytes | 0.20 |
| cost | restore/materialize ms | 0.20 |
| cost | delta build ms | 0.15 |
| cost | 节点和序列槽位开销 | 0.10 |

这组权重的目的，是让真实复用收益和资源成本处于同一量纲；不是把现有的 `family_*_weight` 原值按比例替换。prediction 没有启用时应固定为 0 或做单独实验，不能让未观测特征影响排序。

## 后续实施顺序

### P0：先补可观测性和基线

1. 记录每个候选点的 `parent_depth`、`marginal_tokens`、reuse 次数、reuse 概率、prefill/restore/build/materialize ms、真实 GPU/host bytes、delta quality。
2. 同一 trace 对比固定 chunk `32/64/128/256`，同时报告 TTFT p50/p95、前缀 token 命中率、节点数、物理 KV 使用率和重建误差。
3. 将 `max_cache_nodes`、`max_cache_variants`、`max_cache_tokens` 分开做压力实验，确认瓶颈来自哪个资源。
4. 连续增长前缀必须加入 test4 online replay；test5 继续作为同 prefix 跨 LoRA delta 微基准。

### P1：实现候选准入

1. `ensure_chunk_path` 只生成候选，不立即把所有候选加入缓存。
2. 按父节点的边际区间计算 `net_gain`，只准入正收益且满足资源预算的节点。
3. 在请求完成后更新 reuse 统计，避免用当前请求的结果提前污染候选评分。

### P2：修正资源和调度模型

1. 用实际 KV buffer/page 统计或 `full_kv_bytes_equivalent` 替代 `depth * 1024` fallback。
2. 区分 logical token、physical bytes、host delta bytes，避免用逻辑 token 总和代替物理占用。
3. 对 host delta 引入“压缩节省”和“恢复代价”两项，不能只计 host cost。
4. 将节点上限改成可解释的内存预算；`max_cache_nodes` 仅作为安全上限，不应成为主要策略。

### P3：预测与权重调优

1. 校准 predictor probability；一个 family 的多候选概率应使用 `1 - product(1 - p)` 或明确的 top-k 预算，不直接求和。
2. 用按 session 划分的 train/validation/test trace 做网格搜索或 Bayesian tuning，优化 `TTFT_p95 + lambda_memory * memory + lambda_error * quality_error`。
3. 每次只改变一个因素，保留 `no-score`、`fixed-chunk`、`score-admission`、`score-admission+delta` 四个 ablation。
4. `family_frequency_decay` 用半衰期解释。当前 0.95 的请求级半衰期约 14 个请求；应根据真实复用间隔测试 0.975/0.99，而不是凭直觉增大 frequency weight。

## 验收标准

下一版至少应同时满足：

- 与 fixed-128 相比，前缀 token 命中率和 TTFT p95 有统计显著改善；
- 在相同 GPU/host 预算下，节点淘汰次数下降；
- delta 重建 cosine、L2 和模型输出质量不劣于当前门槛；
- 能解释每个被准入或拒绝的切分点：收益、成本、分数和触发的资源约束都能在 CSV 中复现；
- 在连续增长前缀和同 prefix 跨 LoRA 两类 workload 上都通过，而不是只在 test5 的精确 prefix pair 上通过。

## AI 使用说明

本文由 AI 根据当前源码、实验 CSV 和已有数据集 README 辅助整理。提交到上游或作为正式实验结论前，应由贡献者逐项人工核对代码、重新运行实验并按 `CONTRIBUTING.md` 要求披露 AI 使用方式。
