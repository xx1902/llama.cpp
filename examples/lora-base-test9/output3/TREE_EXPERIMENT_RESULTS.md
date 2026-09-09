# test9 增量前缀树与端侧序列置换小实验结果

本报告由 `run_tree_experiment.py` 自动生成。实验不加载大模型，只验证建树、前缀匹配和缓存置换三个与模型 logits 无关的系统机制。

## 实验设置

- chunk 大小：`4` 个代理 token。
- flat/trie 叶子容量：`4`。序列策略的容量表示同时保留的 `conversation × LoRA` 序列数。
- 匹配均从 root 开始，逐级比较 chunk；遇到不存在的边立即停止匹配并从该位置创建新分支。
- 代理 token 是稳定的词/标点切分，不等同于 Qwen tokenizer；因此结果用于比较结构，不用于宣称真实 TTFT。

## 三种策略

1. `flat_leaf_lru`：每个 LoRA 保存一条完整 prefix，只有完整 prefix 再次出现时才命中；淘汰单位是整条叶记录。
2. `trie_leaf_lru`：按 chunk 从 root 增量建树，匹配到公共祖先即可复用；容量压力下逐个删除最低访问叶节点。
3. `trie_sequence_lru`：建树和匹配与 2 相同，但把 `conversation_id × lora_id` 的整条路径作为端侧原子置换单位；淘汰时移除该序列拥有的所有节点，仍被其它序列使用的公共祖先保留。

老师提出的 BFS/DFS 方向在这里落实为 root-to-leaf 增量遍历。对同一请求序列，BFS/DFS 只是节点访问顺序；真正改变系统性质的是“逐级匹配 + 分支创建”和“序列级置换”这两个组合。

## 结果

| workload | strategy | any-hit | complete-hit | token coverage | peak nodes | evictions | mean walk |
|---|---|---:|---:|---:|---:|---:|---:|
| workbuddy_return | flat_leaf_lru | 10.0% | 10.0% | 10.2% | 4 | 5 | 1.00 |
| workbuddy_return | trie_leaf_lru | 90.0% | 60.0% | 81.9% | 9 | 0 | 4.10 |
| workbuddy_return | trie_sequence_lru | 90.0% | 50.0% | 81.3% | 8 | 0 | 4.10 |
| workbuddy_return | trie_sequence_strict | 90.0% | 20.0% | 71.7% | 8 | 0 | 3.90 |
| server_shared_template | flat_leaf_lru | 0.0% | 0.0% | 0.0% | 4 | 12 | 1.00 |
| server_shared_template | trie_leaf_lru | 93.8% | 43.8% | 75.0% | 12 | 0 | 2.81 |
| server_shared_template | trie_sequence_lru | 93.8% | 25.0% | 68.8% | 6 | 0 | 2.81 |
| server_shared_template | trie_sequence_strict | 93.8% | 25.0% | 68.8% | 6 | 0 | 2.81 |
| long_document | flat_leaf_lru | 0.0% | 0.0% | 0.0% | 4 | 8 | 1.00 |
| long_document | trie_leaf_lru | 91.7% | 66.7% | 85.7% | 8 | 0 | 3.67 |
| long_document | trie_sequence_lru | 91.7% | 0.0% | 71.4% | 7 | 0 | 3.50 |
| long_document | trie_sequence_strict | 91.7% | 0.0% | 71.4% | 7 | 0 | 3.50 |
| msc_real | flat_leaf_lru | 0.0% | 0.0% | 0.0% | 4 | 26 | 1.00 |
| msc_real | trie_leaf_lru | 96.7% | 0.0% | 6.1% | 16 | 7105 | 16.47 |
| msc_real | trie_sequence_lru | 96.7% | 0.0% | 93.6% | 478 | 8 | 237.23 |
| msc_real | trie_sequence_strict | 96.7% | 0.0% | 6.1% | 31 | 11 | 16.47 |
| opus100_real | flat_leaf_lru | 0.0% | 0.0% | 0.0% | 4 | 26 | 1.00 |
| opus100_real | trie_leaf_lru | 96.7% | 50.0% | 72.8% | 16 | 74 | 8.23 |
| opus100_real | trie_sequence_lru | 96.7% | 50.0% | 72.8% | 22 | 26 | 8.23 |
| opus100_real | trie_sequence_strict | 96.7% | 0.0% | 48.9% | 21 | 30 | 6.07 |

## 如何解释

- `flat_leaf_lru` 的 complete-hit 低，说明整段 prefix 只要有一个日期、座位或新对话轮次变化，就无法利用较浅的公共上下文。
- 两种 trie 都可以在首个新 chunk 处停止并复用祖先，因此对“几分钟后回到同一会话、只改返程日期”的 WorkBuddy 场景更合适。
- `trie_sequence_lru` 的创新不在于产生更多叶节点，而在于把端侧淘汰从“随机删一个叶”改成“删掉一个 conversation × LoRA 序列”，避免留下难以恢复的半条会话路径。
- 服务器场景仍可使用细粒度 trie，因为多个用户共享 system/template 祖先；端侧则可以采用较粗 chunk 和序列级原子淘汰。
- 这个小实验没有测量 BGE-M3/KeyBERT 推理延迟，也没有把关键词模型并行接入 llama.cpp；它只隔离出建树和存储机制本身的收益，后续可在同一 trace 上叠加真实 tokenizer、GPU TTFT 和语义模型成本。

## 图片

- `tree_policy_comparison.png`：各 workload 的命中、token 覆盖、节点峰值和淘汰次数。
- `workbuddy_match_trace.png`：WorkBuddy 返回场景中每次请求实际复用的 prefix token。
- `tree_structure_comparison.png`：整段叶存储与 root-to-leaf 增量树的结构差异。

## 与 test8 的关系和下一步

`test8` 已经具备 chunked prefix tree、family 评分和三级 KV 驻留，但当前容量管理仍以节点/family 评分为主，尚未把端侧 `conversation × LoRA` 作为独立原子序列。test9 的 Python 小实验验证了这个机制值得接入，下一步可以在 C++ 中增加 `sequence_family_id`、序列级 LRU/收益评分和路径引用计数，再用真实 workload 做 GPU KV 回放。

本实验是结构性验证，不代表任何一种策略已经在真实模型上获得固定 TTFT 提升。

## Strict incremental build and overlap extension

`trie_sequence_strict` is the strict variant added for the follow-up experiment. The foreground walk starts at root, stops at the first missing chunk, creates only that edge, and queues the remaining suffix. One queued chunk is built in each later idle window. This separates request-critical matching from background materialization.

The strategy table above now contains four strategies: flat complete-prefix LRU, trie leaf LRU, trie sequence LRU, and strict sequence LRU with background build. The strict strategy is intentionally conservative: lower immediate coverage can be exchanged for less foreground tree work.

## Sentence + fixed fallback parallel timing

The splitter first uses sentence/newline boundaries and falls back to fixed chunks when a sentence exceeds 32 proxy tokens. The timing model is deterministic: serial TTFT = split + inference; parallel TTFT = max(split, inference). It demonstrates the overlap upper bound, not hardware performance.

| workload | tokens | chunks | split ms | inference ms | serial TTFT | parallel TTFT | overlap saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| workbuddy_sentence | 3 | 1 | 0.53 | 6.54 | 7.07 | 6.54 | 0.53 |
| long_document_sentence | 54 | 2 | 2.25 | 15.72 | 17.97 | 15.72 | 2.25 |
| long_sentence_fallback | 1 | 1 | 0.48 | 6.18 | 6.65 | 6.18 | 0.47 |
