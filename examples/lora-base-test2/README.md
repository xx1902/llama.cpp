# lora-base-test2 阅读指南

这是当前多 LoRA prefix-KV 实验的独立整理版。它不会覆盖旧的
`examples/lora-base-test`，可执行文件名为 `llama-lora-base-test2.exe`，默认结果写入
`examples/lora-base-test2/output`。

完整的代码结构、请求时序、三级缓存和在线 GRU 设计见
[代码结构与实现思路.md](./代码结构与实现思路.md)。

成套消融使用 [run_system_ablations.ps1](./run_system_ablations.ps1)，统一结果汇总使用
[analyze_system_ablations.py](./analyze_system_ablations.py)。

Anchor-Delta GPU 算子的可行性、基线和分阶段实验见
[ANCHOR_DELTA_OPERATOR_FEASIBILITY.md](./ANCHOR_DELTA_OPERATOR_FEASIBILITY.md)。

Anchor family、三级状态、分组和预测消融见
[FAMILY_CACHE_EXPERIMENT_DESIGN.md](./FAMILY_CACHE_EXPERIMENT_DESIGN.md)，彩色论文图由
[plot_system_ablations.py](./plot_system_ablations.py) 生成。

## 1. 系统边界

代码只研究从 prompt 开头开始的 prefix KV 复用，不研究任意中间 token 的 KV 复用。

1. 相同 LoRA、相同 token prefix：直接命中目标 LoRA 的完整 KV，跳过已命中部分 prefill。
2. 不同 LoRA、相同 token prefix：不能直接共用完整 KV。系统以 `anchor KV + Q8 delta`
   压缩保存目标 LoRA KV，需要先恢复为目标 LoRA 的完整 KV 才能用于推理。
3. delta 的直接收益是降低 Host/磁盘存储，不是降低当前请求 TTFT。
4. 预测只有在请求到达前把目标 KV 准备到 GPU，才能间接降低 TTFT。
5. GRU 离线训练、在线推理：Python 服务按用户维护历史，C++ 每次请求后实时获取 Top-k。
   原来的 JSONL 文件模式仍保留，用于复现实验和对照。
6. 当前在线循环仍是一请求一序列。混合 LoRA batch 和 fused CUDA 算子尚未接入本文件。

## 2. 建议阅读顺序

核心代码只有一个文件：[lora-base-test2.cpp](./lora-base-test2.cpp)。建议按下面顺序看：

| 顺序 | 模块 | 主要内容 |
|---:|---|---|
| 1 | `experiment_options` / `parse_options` | 所有命令行参数和实验开关 |
| 2 | `dataset_request` | 一条 LSApp 请求如何映射到 LoRA 和 prompt |
| 3 | `prefix_node` / `prefix_variant` | prefix tree 节点及同一节点上的多 LoRA KV 状态 |
| 4 | `tokenize_request` | system、user context、task suffix 的 token 边界 |
| 5 | `run_delta_experiment` | 离线比较相同 prefix 下不同 LoRA KV，并构造 Q8 delta |
| 6 | `run_online_experiment` | 旧版 exact-prefix 基线 |
| 7 | `family_value` | Anchor family 的频率、预测、热度和成本评分 |
| 8 | `ensure_chunk_path` | 按 chunk 逐级构建 prefix tree |
| 9 | `convert_variant_to_host_delta` | GPU full KV 转成 Host anchor + delta |
| 10 | `prefetch_oracle_chunks` | Oracle/GRU Top-k 驱动的目标 KV 准备 |
| 11 | `select_delta_compress_job` | 后台压缩候选的优先级选择 |
| 12 | `run_online_system_v2` | 请求、缓存、预测和后台任务的完整事件循环 |
| 13 | `save_*` / `main` | CSV 输出和程序入口 |

## 3. 核心数据结构

### `prefix_node`

一个节点表示从根到当前 chunk 的完整 token prefix。它保存父节点、token 深度、提示词分区、
anchor LoRA 和多个 LoRA variant。树只按照 token 内容逐级匹配，不按照“相似词”匹配。

### `prefix_variant`

同一 prefix 节点在某个 LoRA 下的 KV 状态：

- `gpu_full`：GPU 上已有完整目标 KV，可直接命中。
- `host_full`：Host 上保存完整 KV，需要复制/恢复后才能使用。
- `host_delta`：Host 上保存相对 anchor 的 Q8 delta，需要 `anchor + delta` 恢复。

### Anchor family

物理管理单位是某个 prefix 节点上的 anchor 及依赖它的 variants。family 评分为：

```text
benefit = frequency + prediction + prefix_length + delta_fanout
        + recency + segment_priority
cost    = memory + materialization
score   = benefit - cost
```

开启 `--family-normalize-by-mb 1` 后，代码按占用空间归一化，倾向保留单位 MiB 收益更高的
family；关闭后更偏向绝对延迟收益。

## 4. 在线请求流程

```text
读取一条 LSApp 请求
  -> 映射到实际 LoRA
  -> tokenize 并得到 system/context/suffix 边界
  -> 在 chunked prefix tree 中找最深节点
  -> 若目标 LoRA GPU full KV 命中，复制到请求序列
  -> 只计算未命中的 prefix chunk 和 suffix
  -> 记录跨 LoRA、相同 prefix 的 delta 候选
  -> 调用在线 GRU 服务取得 Top-k，保护候选 family，并建立预取任务
  -> 在下一请求前的空闲窗口执行允许完成的预取/压缩任务
  -> 按 family score 管理 GPU/Host 容量
  -> 输出每请求指标和 family 决策日志
```

这里的“后台”是用请求间逻辑空闲窗口模拟的单线程调度，不是真正独立的 CPU 线程或 CUDA
stream。因此它可以验证策略和工作量，但不能直接代表最终异步实现的并行性能。

## 5. 提示词分段

数据集可显式提供 `prefix_segments`。代码累积 tokenize 各段来取得准确边界：

- `shared_system`：默认每 64 token 一个 chunk，进入 prefix tree。
- `user_context`：默认每 128 token 一个 chunk，进入 prefix tree。
- task instruction/suffix：不写入共享 prefix tree，按当前请求正常计算。

“suffix 按请求计算”表示它仍参与正常 prefill，只是不作为可长期共享的 prefix tree 节点。

## 6. 输入文件

默认工作负载为 `D:/ecnu_experiment/datasets/mobilora_workloads_87_original`：

```text
lora_groups.json
delta/delta_requests.jsonl
delta/delta_pairs.jsonl
grouped/grouped_requests.jsonl
```

`lora_groups.json` 负责 87 个逻辑 App/LoRA 到真实 GGUF LoRA 的一对一映射，并标记 group
和 anchor。`file` 模式的 GRU 文件每行以当前 `request_id` 为键，给出下一请求候选；`service`
模式则将当前请求的 `user_id`、LoRA 和原始时间戳实时发送给 Python 服务。

## 7. 在线 GRU 接口

Python 服务 [gru_online_service.py](./gru_online_service.py) 为每个 `user_id` 独立保存最近 15 次
App 和时间特征。每完成一条请求，C++ 调用 `POST /predict`：

```json
{
  "request_id": 14,
  "user_id": 138,
  "lora_id": 21,
  "original_timestamp": "2018-01-04T05:23:02",
  "top_k": 3
}
```

历史不足 15 条时返回 `ready=false`；历史足够后返回 Top-k `lora_id + probability`。服务支持
`POST /reset` 清空所有用户状态，C++ 默认在每轮实验开始时调用，防止上一轮状态污染结果。
重复的 `request_id` 不会被追加两次。服务失联或超时时，本次请求只跳过预测预取，不会中断
前台 LLM 推理。

## 8. 主要输出

| 文件 | 用途 |
|---|---|
| `delta_prefix_probe.csv` | LoRA pair 的 KV cosine、L2、Q8 delta 大小和构建成本 |
| `delta_layer_probe.csv` | 每层 K/V 差异 |
| `online_request_results.csv` | 每请求 TTFT、命中 token、预测服务耗时、后台工作和存储统计 |
| `prefix_tree_summary.csv` | 最终 prefix tree 节点和 variant 状态 |
| `family_cache_events.csv` | family 换入、降级、淘汰原因及评分分项 |
| `system_parameters.csv` | 模型、LoRA、缓存上限和评分参数快照 |

## 9. 已实现与未实现

已经实现：87 LoRA、LSApp 顺序回放、固定提示词分区、chunked prefix tree、同 LoRA exact
prefix 命中、跨 LoRA Q8 delta、延迟压缩、持久化 delta、Host/GPU 三级状态原型、GRU JSONL
对照、Python GRU 在线逐用户预测、Top-k 预取、anchor-family 量化评分和相关实验日志。

尚未实现：真正后台线程/CUDA stream、LoRA 动态换入换出、语义自适应切分、增量 chunk
delta、C++ 进程内 GRU、GRU + 聚类 + 全局/用户关系表融合、一个 App 多 LoRA 的分层预测、混合 LoRA batch
和 fused multi-LoRA CUDA 算子。上述项目不能作为当前代码已经完成的实验结论。

## 10. 构建与运行

先把 `examples/CMakeLists.txt` 中加入 `add_subdirectory(lora-base-test2)`，再构建：

```powershell
cd D:\ecnu_experiment\LLama.cpp\llama.cpp
cmake --build build --config Debug --target llama-lora-base-test2 -j
```

推荐先运行 [run_examples.ps1](./run_examples.ps1) 的 `Smoke`，确认 87 个 LoRA 和数据路径都正常。

在线 GRU 需要两个 PowerShell 窗口。第一个窗口启动服务：

```powershell
powershell -ExecutionPolicy Bypass -File D:\ecnu_experiment\LLama.cpp\llama.cpp\examples\lora-base-test2\start_gru_service.ps1
```

确认服务状态：

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
```

也可以先运行不加载 LLM 的状态管理测试：

```powershell
D:\anaconda\envs\qwen2.5_vl\python.exe D:\ecnu_experiment\LLama.cpp\llama.cpp\examples\lora-base-test2\test_gru_online_service.py
```

第二个窗口运行在线预测实验：

```powershell
powershell -ExecutionPolicy Bypass -File D:\ecnu_experiment\LLama.cpp\llama.cpp\examples\lora-base-test2\run_examples.ps1 -Case GruOnline -Requests 120
```

运行后重点查看 `prediction_service_ok`、`prediction_ready`、`prediction_query_ms`、
`prediction_inference_ms` 和 `predicted_lora_id`。`GruFile` 使用同一个 checkpoint 预生成的 JSONL，
可与 `GruOnline` 做一致性和通信开销对照。

汇总在线准确率、服务延迟、TTFT 和后台工作量：

```powershell
D:\anaconda\envs\qwen2.5_vl\python.exe D:\ecnu_experiment\LLama.cpp\llama.cpp\examples\lora-base-test2\summarize_gru_online.py D:\ecnu_experiment\LLama.cpp\llama.cpp\examples\lora-base-test2\output_gru_online
```
