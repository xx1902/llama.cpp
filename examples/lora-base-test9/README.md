# lora-base-test9：增量建树与端侧序列置换小实验

本目录验证老师提出的核心方向：前缀匹配从 root 开始逐级进行，遇到新文本再创建分支；端侧缓存可以把 `conversation × LoRA` 的整条路径作为原子置换单位，而不是只淘汰一个叶节点。

## 运行

```powershell
python examples/lora-base-test9/run_tree_experiment.py `
  --output-dir examples/lora-base-test9/output `
  --real-root examples/lora-base-datasets/output/real_30 `
  --real-limit 30
```

脚本默认同时运行三个人工 workload：

- `workbuddy_return`：几分钟后回到同一会话，只修改返程日期/座位；
- `server_shared_template`：多用户共享订票系统模板；
- `long_document`：长文档逐步增长前缀。

如果 `--real-root` 存在，还会自动加入 `msc_continuous` 和 `opus100_parallel` 的真实请求前缀。实验使用稳定的代理 token，不加载大模型，因此运行很快，重点是隔离比较建树、匹配和置换机制。

## 策略

1. `flat_leaf_lru`：完整 prefix 作为一条记录，只有整段再次出现才命中。
2. `trie_leaf_lru`：chunked root-to-leaf 增量建树，容量压力下逐个淘汰叶节点。
3. `trie_sequence_lru`：同样增量建树，但把 `conversation × LoRA` 路径作为端侧原子序列淘汰。

输出包括：

- `tree_experiment_results.csv/json`：汇总指标；
- `tree_request_trace.csv`：逐请求匹配和淘汰轨迹；
- `tree_policy_comparison.png`：跨 workload 对比图；
- `workbuddy_match_trace.png`：WorkBuddy 返回场景的复用曲线；
- `tree_structure_comparison.png`：叶记录与共享前缀树结构图；
- `TREE_EXPERIMENT_RESULTS.md`：自动生成的结果说明。

更完整的不足分析、算法定义、结果解释和接入 test8 的建议见 `TREE_EXPERIMENT_RESULTS.md`。
