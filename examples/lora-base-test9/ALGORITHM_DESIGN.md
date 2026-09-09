# test9 建议算法与 test8 不足分析

## 1. 对当前 test8 的判断

`test8` 已经可以验证多 LoRA prefix KV 的主要机制，但和老师这次提出的方向相比还有以下不足：

1. **建树和置换单位没有完全分离。** `ensure_chunk_path` 会按固定 `system/context` 步长创建累计节点，容量不足时 `ensure_cache_capacity_v2` 主要按照 family 分数和叶子节点淘汰；还没有把端侧的 `conversation_id × lora_id` 作为显式原子序列。
2. **节点是一次请求中连续创建的。** 运行时虽然从 root 查找父节点，但一旦进入 `ensure_chunk_path`，会为整条 prefix path 补齐所有 chunk。更严格的增量策略应在第一个未匹配 chunk 停止，先服务当前请求，再按预算逐级补建后续节点。
3. **引用关系不够明确。** `prefix_node` 记录 parent 和 variants，但没有独立的 `path_refcount`/`sequence_owners`，因此端侧整条序列淘汰需要额外判断哪些祖先仍被其它会话共享。
4. **上下文隔离策略需要显式化。** 当前 `find_chunk_node` 的匹配键主要是 `group + parent + token prefix + depth`，适合服务器跨用户共享公共 system/template；端侧如果涉及个人资料、订单、地址等敏感上下文，应支持 `namespace=context_id` 或按 segment 设置可共享级别。
5. **语义切分仍未形成在线闭环。** test7 的 Sentence/Keyword/Reuse-aware 主要是 workload 侧切分；BGE-M3/KeyBERT 没有和推理并行，也没有把模型前向延迟、预取预算和 chunk 收益放进统一决策。
6. **当前实验缺少真实 KV 成本。** test9 的代理 token 实验只比较结构、命中和淘汰；最终仍需要在 test8 中记录实际 `seq_cp`、materialize、Host copy、delta restore 和 TTFT。

## 2. 建议的双层设计

### 2.1 共享 Prefix Trie

服务器端维护一棵按 token chunk 组织的 trie。节点键为：

```text
(share_namespace, parent_node_id, chunk_hash, depth_tokens)
```

`share_namespace` 可以取：

- `global`：所有用户可共享的 system prompt；
- `tenant/group`：同一应用或工作流可共享的模板；
- `conversation`：仅同一会话可共享的个人上下文；
- `private`：不允许跨用户复用的敏感片段。

请求从 root 开始：

```text
node = root
for chunk in request.chunks:
    child = lookup(node, namespace, hash(chunk))
    if child is missing:
        break_at = current_depth
        break
    node = child
```

匹配到的最长祖先用于 `seq_cp`。第一个未匹配 chunk 进入“增量建树队列”，而不是在前台一次性建立整条新链。

### 2.2 端侧 Sequence Family

端侧把以下对象视为一个缓存单元：

```text
sequence_key = conversation_id × lora_id
```

一个 sequence family 保存：

- 当前会话的 root-to-leaf path；
- 每个节点的 KV sequence id；
- 最近访问时间、回访概率、可复用 token 数；
- 恢复成本、delta 质量和显存/Host 大小。

端侧置换分数可以写为：

```text
value_edge(sequence)
  = P_return × reusable_tokens × saved_ms_per_token
    + W_shared × shared_ancestor_tokens
    - W_gpu × gpu_bytes
    - W_restore × restore_ms
    - W_delta × delta_build_ms
```

当 `value_edge` 最低时，原子释放该 sequence 的所有私有路径；公共祖先只有在 `path_refcount==0` 时才释放。这样可以避免只淘汰一个叶子后留下“半条会话”或丢失 anchor 依赖。

### 2.3 服务器端 Node/Variant 置换

服务器端仍建议细粒度管理：

```text
value_server(node)
  = shared_users × P_reuse × reusable_tokens / (memory_bytes + epsilon)
```

优先保留高扇出的 global/tenant 祖先，低热度的 conversation/private 叶子先淘汰。服务器和端侧不应强制使用同一个容量单位。

## 3. 增量建树伪代码

```text
access(request):
    tokens = tokenize(request.prefix)
    chunks = split(tokens, Sentence + fixed fallback)
    node = root
    matched = 0

    for chunk in chunks:
        child = trie.lookup(node, namespace(chunk), hash(chunk))
        if child is missing:
            break
        node = child
        matched += len(chunk)

    if matched > 0:
        copy_kv(node.seq, request.seq)

    eval(tokens[matched:] + request.suffix, request.seq)

    enqueue_incremental_build(
        parent=node,
        remaining_chunks=chunks_after(matched),
        owner=conversation × lora)

    update_refcount_and_access_stats()
    schedule_prefetch_or_delta_in_idle_window()
```

这里的关键创新是：**匹配先于建树，建树先于存储，存储按共享祖先和端侧 sequence family 分层管理。**

## 4. Sentence + fixed fallback

为了保持通用性，不建议把 BGE-M3 作为每个请求的必经前台路径。更稳妥的默认切分是：

1. 先按显式 segment（system、profile、history、source article）分段；
2. 段内优先在句号、问号、感叹号、换行和 `User:`/`Assistant:` 边界处结束 chunk；
3. chunk 在 `[min_tokens, max_tokens]` 内时采用自然句边界；
4. 单句超过 `max_tokens` 时使用固定长度切分；
5. 对重复出现的历史前缀终点增加 reuse bonus，但不改变 token 正确性。

这个方法不要求语义模型在线运行，并且适用于英文、中文、对话和长文章。BGE-M3/KeyBERT 可以作为**离线分析或低优先级后台建议器**：

```text
当前请求到达
  ├─ 前台：tokenize -> root-to-leaf match -> prefill/decode
  └─ 后台：BGE-M3 embedding -> 主题变化候选 -> 下一次请求的 split hint
```

如果要验证并行是否值得，必须单独记录：

```text
semantic_model_ms、overlap_window_ms、split_hint_hit_rate、
extra_nodes、reused_tokens、TTFT_delta
```

不能只比较“切分后的边界看起来更语义化”。

## 5. WorkBuddy 场景的推荐流程

示例：

```text
t0: LoRA1：帮我订上海到北京的去程票，周五出发
t+5min: LoRA2：帮我订同一行程的返程票，下周一
t+20min: LoRA1：把返程改成周二，靠窗
```

建议保存：

- `global`：工作助手 system prompt；
- `conversation`：用户偏好、出发地、当前订单和历史对话；
- `conversation × LoRA1/LoRA2`：对应 adapter 下的完整 KV variant；
- `private`：身份证、手机号等不可跨用户共享字段。

第三次请求从 root 匹配到同一 conversation 的公共祖先，只对“返程日期/座位”的新 chunk 建分支。若显存不足，淘汰一个长期不回访的 `conversation × LoRA2` family，而不是删除 LoRA1 路径中的某个叶节点。

## 6. test9 小实验的意义和限制

`run_tree_experiment.py` 用三个合成 workload 和两个真实 JSONL workload 验证：

- flat 完整 prefix 存储对日期、座位和新增对话轮次非常敏感；
- root-to-leaf trie 能复用公共祖先，显著提升 any-hit 和 token coverage；
- sequence-level eviction 在端侧可以稳定控制“同时保留多少条会话×LoRA 路径”，但容量过小会牺牲服务器式多用户共享；
- trie 的节点数、淘汰次数和 walk steps 暴露了 chunk 太细或容量太小的代价。

它没有加载 GGUF、没有调用 Qwen tokenizer、没有生成真实 KV，也没有实现 BGE-M3 并行。因此下一步 C++ 验证应优先加入：

1. `sequence_family_id`、`sequence_owners` 和 `path_refcount`；
2. root-first longest-prefix match 的前台接口；
3. 增量建树队列和请求间预算；
4. namespace/shared-level；
5. 实测 KV copy/materialize/restore 时间；
6. Sentence + fixed fallback 与 semantic hint 的并行消融。
