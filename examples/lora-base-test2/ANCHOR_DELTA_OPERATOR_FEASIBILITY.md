# Anchor-Delta KV 算子可行性与实验路线

## 1. 研究问题

当前跨 LoRA KV 路径先保存 `anchor KV + Q8 delta`，需要使用目标 LoRA 时再恢复完整 KV。本文档
要验证三个问题：

1. Delta encode 的高后台成本是否主要来自小粒度 GPU/CPU 搬运和 CPU 标量量化。
2. 融合 `dequant + anchor add + paged KV write` 是否能显著降低 materialize 延迟和临时内存。
3. 在短复用周期或显存压力较高时，压缩 KV 直接参与 Attention 是否比完整 materialize 更合适。

## 2. 当前实现为什么慢

当前 `seq_delta_build_branch()` 和 `seq_delta_materialize_branch()` 位于
`src/llama-kv-cache.cpp`。

Encode 的当前流程是：

```text
for layer
  for token
    查找 anchor/child cell
    ggml_backend_tensor_get(anchor row)
    ggml_backend_tensor_get(child row)
    CPU 第一次遍历：计算 max_abs
    CPU 第二次遍历：Q8 量化
```

当 V cache 使用转置布局时，当前代码在 embedding 维度内部逐元素调用
`ggml_backend_tensor_get()`。这会产生大量极小粒度设备同步和 D2H 传输。

Materialize 同样按 layer、token 反复执行 tensor get/set；转置 V 路径还会逐元素读取 anchor、
反量化并写回目标 KV。当前实现首先用于验证语义正确性，并不是高性能 GPU 路径。

12 请求 Smoke 中：

```text
delta candidates       = 24
compressed branches    = 10
background delta time  = 36.61 s
mean per branch        = 3.66 s
Q8 storage reduction   = 49.22%
```

因此有明确的优化空间，但必须用分阶段实验确认收益来自哪里。

## 3. 算子定义

### 3.1 Fused Delta Encode

输入：

```text
anchor KV: FP16/BF16
child KV : FP16/BF16
cell map : token -> physical KV cell
```

每个量化 block：

\[
\Delta_i = KV_{child,i}-KV_{anchor,i}
\]

\[
s_b=\frac{\max_{i\in b}|\Delta_i|}{127}
\]

\[
q_i=\operatorname{clip}
\left(
\operatorname{round}(\Delta_i/s_b),-127,127
\right)
\]

输出：

```text
Q8 delta
per-block scale
branch metadata
```

GPU kernel 应在同一个 tile 内完成差值、reduction、scale 和量化，不生成完整 FP32 delta。

### 3.2 Fused Delta Materialize

输入：

```text
anchor KV
Q8 delta
scale
destination cell map
```

计算：

\[
KV_{target,i}=KV_{anchor,i}+s_bq_i
\]

在一个 kernel 中完成反量化、相加和目标 paged KV 写入，不产生反量化临时张量。

### 3.3 Anchor-Delta Attention

不写出完整目标 KV，在 Attention tile 内计算：

\[
K_t=K_a+s_Kq_K,\qquad V_t=V_a+s_Vq_V
\]

\[
O=\operatorname{softmax}
\left(\frac{QK_t^T}{\sqrt d}\right)V_t
\]

anchor 和 Q8 delta 在寄存器/shared memory 中组合，随后直接进入 attention 计算。

## 4. 分阶段实验

## Phase 0：现有路径分解计时

目标：证明当前 3.66 s/branch 具体耗在哪里。

需要增加以下计时：

| 指标 | 内容 |
|---|---|
| `cell_lookup_ms` | position 到 physical cell 的查找 |
| `anchor_d2h_ms` | Anchor KV 读取 |
| `child_d2h_ms` | Child KV 读取 |
| `quantize_cpu_ms` | max reduction 和 Q8 量化 |
| `delta_store_ms` | Delta vector 分配和保存 |
| `anchor_read_ms` | Materialize 时读取 anchor |
| `dequant_add_ms` | CPU 反量化和相加 |
| `target_h2d_ms` | 写回目标 KV |
| `backend_get_calls` | tensor_get 调用次数 |
| `backend_set_calls` | tensor_set 调用次数 |

判据：如果 D2H/H2D 与 backend 调用占主要时间，GPU fused 路径具有直接可行性。

## Phase 1：Bulk CPU Baseline

暂时不写 CUDA kernel，先优化数据搬运方式：

1. 预先构建 `position -> cell_id` 数组，避免每个位置扫描全部 cells。
2. 每层一次性读取连续 K/V 区间，不按 token/元素调用 backend get。
3. 在连续 Host buffer 上执行 CPU Q8 encode/materialize。
4. 每层一次性写回目标 KV。

这个 baseline 用来区分：

```text
算法本身慢
vs.
当前大量小传输和同步慢
```

如果 bulk CPU 已经比当前实现快一个数量级，就能证明传输粒度是主要瓶颈。

## Phase 2：Standalone CUDA Microbenchmark

先在独立 benchmark 中实现 fused encode/materialize，不立即修改 llama.cpp Attention。

建议文件：

```text
examples/anchor-delta-bench/anchor-delta-bench.cu
```

实验后端：

| 后端 | 说明 |
|---|---|
| `scalar_cpu` | 当前逐行/逐元素逻辑 |
| `bulk_cpu` | 整层搬运后 CPU 量化 |
| `cuda_encode` | GPU fused delta encode |
| `cuda_materialize` | GPU fused delta materialize |
| `full_kv_copy` | 直接搬运完整 FP16 KV 的参考上限 |

先使用 synthetic tensor 验证 kernel，再读取真实 Qwen2.5-1.5B KV snapshot。

## Phase 3：接入 llama.cpp KV Page

将 standalone kernel 接入真实 KV cell/page 布局：

```text
cell_ids
layer pointers
K/V strides
v_trans layout
dtype
```

需要分别覆盖：

- K 连续行布局。
- V 非转置布局。
- V 转置布局。
- 不连续 physical cell。
- 不同 prefix chunk 范围 `[p0,p1)`。

接入后保留 CPU reference，通过运行参数选择：

```text
--delta-operator scalar
--delta-operator bulk-cpu
--delta-operator cuda-fused
```

## Phase 4：Direct Delta Attention

只有 fused materialize 正确且性能稳定后，再修改 Attention。

对照组：

| 路径 | 说明 |
|---|---|
| Full KV Attention | 原始目标 LoRA 完整 KV |
| Materialize + Attention | Q8 delta 恢复完整 KV 后执行 Attention |
| Direct Delta Attention | Attention 内部组合 anchor + Q8 delta |

Direct Attention 首先只支持 decode、固定 head dimension 和连续 page，确认可行后再扩展。

## 5. 数据与实验矩阵

### 5.1 KV 数据

使用真实 87 LoRA 工作负载中的相同 prefix、不同 LoRA pair：

```text
高权重相似度 pair
中等相似度 pair
低相似度 pair
组内 pair
跨组 pair
```

Prefix 长度：

```text
64, 128, 256, 512, 1024 token
```

每个配置：

```text
10 次 warm-up
至少 100 次计时
报告 P50/P95/P99
```

### 5.2 Encode 实验

比较：

```text
Current Scalar
Bulk CPU
CUDA Fused Encode
```

输出：

- Encode latency。
- 有效处理带宽。
- backend get/set 次数。
- 临时 Host/GPU 内存。
- Q8 delta 大小和压缩率。
- 与 CPU reference 的字节/数值一致性。

### 5.3 Materialize 实验

比较：

```text
Current Scalar Materialize
Bulk CPU Materialize
CUDA Fused Materialize
Full KV Copy
```

输出：

- Materialize latency P50/P95。
- `materialized_kv_bytes / latency` 有效 GB/s。
- 相对 Full KV Copy 的延迟。
- 临时显存峰值。
- 重建 cosine、L2 和 max absolute error。

### 5.4 End-to-End 实验

在相同 LSApp 请求轨迹下比较：

```text
Prefix only
Deferred delta + scalar operator
Deferred delta + bulk CPU
Deferred delta + CUDA fused
GRU prefetch + CUDA fused
```

输出：

- 前台 TTFT。
- 后台总工作量。
- Background fit rate 和 overrun。
- Delta candidates/completed。
- GPU/Host KV 占用。
- 正确和错误预取成本。

## 6. 正确性验证

算子结果不能只看 KV cosine。

### Level 1：Tensor

将 CUDA 输出与当前 CPU Q8 reference 比较：

```text
max_abs_error
mean_abs_error
relative_l2
cosine
NaN/Inf count
```

### Level 2：Logits

对同一 prompt 比较：

```text
Full target KV
CPU Q8 materialize
CUDA Q8 materialize
Direct delta attention
```

记录：

```text
logits cosine
KL divergence
Top-1/Top-5 token agreement
```

### Level 3：Generation

固定 greedy decoding 和随机种子，比较：

```text
首个不同 token 的位置
完整 token agreement
任务质量指标
```

摘要任务可用 ROUGE，代码任务可用编译/单元测试通过率。只有 Tensor 相似但生成质量变化明显时，
不能认为 delta 路径正确。

## 7. 理论带宽分析

假设共有 `N` 个 FP16 KV 元素，忽略 scale 元数据。

Fused materialize 最少需要：

```text
读取 anchor FP16 : 2N bytes
读取 Q8 delta    : 1N bytes
写出 target FP16: 2N bytes
合计             : 5N bytes
```

如果先产生 FP16 delta 临时张量，还会增加一次临时写入和读取，接近 `9N bytes`。因此 fused
materialize 的主要目标是接近 `5N` 的理论流量。

Direct delta attention 每次读取约 `3N`，普通完整 KV attention 每次读取约 `2N`。若完整
materialize 一次需要约 `5N`，只按显存流量估算：

\[
3HN < 5N + 2HN \Rightarrow H < 5
\]

其中 `H` 是未来重复使用该 KV 的 Attention 次数。说明 direct delta attention 更适合：

- 只使用很少次数的预测候选。
- 预测置信度较低、不值得完整展开的候选。
- 显存压力很高，无法保存完整目标 KV。

当预计会被连续使用较多次时，提前 materialize 通常更合适。

## 8. 自适应路径决策

定义：

\[
C_{direct}=H\cdot T_{delta-attn}+\lambda M_{compact}
\]

\[
C_{materialize}=T_{transfer}+T_{reconstruct}
+H\cdot T_{normal-attn}+\lambda M_{full}
\]

选择：

\[
Mode^*=\arg\min(C_{direct},C_{materialize})
\]

其中 `H` 可以根据任务类型、历史生成长度或预测模块估计；`lambda` 随当前 GPU 内存压力变化。

## 9. 可行性判据

第一阶段认为 fused materialize 可行，需要同时满足：

1. CUDA 输出与 CPU Q8 reference 在设定 tolerance 内一致。
2. Fused materialize 显著快于当前 scalar 路径和 bulk CPU。
3. 相对 Full KV Copy 的额外延迟可控。
4. 不产生完整 FP32 delta 临时张量。
5. 真实 system-v2 中 background overrun 明显下降。
6. Logits 和生成质量没有不可接受退化。

Direct delta attention 只有在以下条件满足后继续：

1. 对 materialized Q8 KV 的 logits 结果一致。
2. 短 horizon 或高内存压力场景具有端到端收益。
3. Decode TGI 增量小于省去 materialize 带来的收益。

## 10. 推荐开发顺序

```text
1. 增加 current path 分项计时和调用次数
2. 实现 cell map + bulk CPU baseline
3. 编写 standalone fused materialize CUDA benchmark
4. 编写 fused encode CUDA benchmark
5. 接入 llama.cpp paged KV
6. 做 tensor/logits/generation 正确性实验
7. 接入 system-v2 和 GRU 预取
8. 最后尝试 direct delta attention
```

不建议第一步直接修改 FlashAttention。先证明 fused materialize 的数值和带宽收益，可以显著降低
实现风险，并为 direct delta attention 提供可靠 reference。

