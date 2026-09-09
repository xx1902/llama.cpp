# LoRA prefix-cache test7

This local experiment suite is the 100-request follow-up to test6. It reuses the `lora-base-test4` system-v2 runtime and
adds a reproducible cross-dataset matrix, aggregation, plots, and a Markdown
report. Reusing the runtime keeps the GPU full-KV, host full/Q8-delta, disk
delta-store, prefix-tree, and CUDA overlap paths identical to the implementation
already exercised by test4 and test5.

The default matrix covers six real 100-request workloads:

- MSC, Taskmaster, LMSYS, and ShareGPT52K continuous growing-prefix traces;
- OPUS-100 and XSum exact shared-prefix parallel traces.

It compares full-prefill and legacy exact-prefix baselines with GPU-only,
GPU-plus-host-full, tiered no-prefetch, oracle-prefetched fixed chunks of
32/64/128/256 tokens, an `oracle_full` 128-token run with
`max-prefetch-chunks-per-lora=0`, and cold/warm disk delta-store runs. Oracle
prefetch is an upper bound, not a deployable predictor result; `0` means
unlimited chunk coverage, not zero prefetch.

The default matrix uses `n_ctx=65536` and a 61440-token cache budget. Growing
prefix nodes can hold several LoRA variants, so the 100-request traces may
trigger capacity eviction. The runner treats any failed mode or non-positive
online TTFT as a failed cell instead of aggregating it.

## Build

```powershell
cmake --build build --config Release --target llama-lora-base-test7 --parallel 2
```

## Run and analyze

```powershell
D:\anaconda\envs\qwen2.5_vl\python.exe `
  examples/lora-base-test7/run_experiments.py
```

Use `--quick` for a five-request smoke run, `--force` to rerun completed cells,
and comma-separated `--datasets` or `--strategies` filters for a subset. The
runner stores complete commands and logs under `output/real_100_matrix/runs` and
then invokes `analyze_results.py`.

For the completed 100-request core experiment (legacy, GPU-only, and tiered
oracle at 128 tokens), use a separate output directory:

```powershell
D:\anaconda\envs\qwen2.5_vl\python.exe -B `
  examples/lora-base-test7/run_experiments.py `
  --request-limit 100 `
  --strategies legacy_exact,gpu_only_128,tiered_oracle_128 `
  --output-dir examples/lora-base-test7/output/core_100_matrix
```

The command without `--strategies` runs the full 11-strategy matrix and can
take hours on the long MSC, Taskmaster, and ShareGPT traces.

The analyzer writes:

- `aggregate_results.csv` and `delta_quality.csv`;
- `prefix_structure_analysis.csv`, an offline comparison of Fixed 64/128,
  Fixed 192/384, Sentence, Keyword, and Reuse-aware boundaries, enriched with
  online runtime columns after the prefix-method suite has run;
- `prefix_method_results.csv`, the online TTFT, return-hit, host-KV, eviction,
  and reusable-token measurements for those five boundary methods;
- TTFT heatmap, prefix-chunk ablation, storage ablation, and delta-quality PNGs;
- `datasets/<dataset>/RESULTS.md`, `final_effect.png`, and
  `prefix_split_structure.png` plus `prefix_method_online_effect.png` for each
  dataset;
- `RESULTS.md`, including methodology, cross-dataset results, per-dataset links,
  and limitations.

Generate a focused regression diagnosis after the matrix finishes:

```powershell
D:\anaconda\envs\qwen2.5_vl\python.exe -B `
  examples/lora-base-test7/diagnose_results.py
```

It writes ignored `diagnostic_cells.csv` and `DIAGNOSIS.md` files beside the
main report. The diagnosis separates low realized reuse from overhead that
remains even when the token hit rate is high.

Run the five prefix methods online with:

```powershell
D:\anaconda\envs\qwen2.5_vl\python.exe `
  examples/lora-base-test7/run_prefix_method_experiments.py
```

To reproduce the single-dataset screenshot-style example for OPUS-100:

```powershell
D:\anaconda\envs\qwen2.5_vl\python.exe -B `
  examples/lora-base-test7/run_prefix_method_experiments.py `
  --datasets opus100_parallel `
  --request-limit 100 `
  --output-dir examples/lora-base-test7/output/opus100_prefix_100

D:\anaconda\envs\qwen2.5_vl\python.exe -B `
  examples/lora-base-test7/make_prefix_method_figure.py `
  --dataset opus100_parallel `
  --request-limit 100 `
  --output-dir examples/lora-base-test7/output/opus100_prefix_100
```

The second command writes `opus100_parallel_prefix_method_effect.png`, a
2x2 structural chart plus the online result table, and
`OPUS100_PARALLEL_PREFIX_METHOD_RESULTS.md` with the same values in Markdown.

For the algorithm definitions, concrete chunk examples, workload reuse
probabilities, and the per-dataset explanation of positive/negative TTFT,
see [`PREFIX_SPLITTING_ALGORITHM_ANALYSIS.md`](PREFIX_SPLITTING_ALGORITHM_ANALYSIS.md).

The optimized `Adaptive reuse v2` splitter is documented separately so the
original v1 analysis remains reproducible. See
[`PREFIX_SPLITTING_ALGORITHM_ANALYSIS_2.md`](PREFIX_SPLITTING_ALGORITHM_ANALYSIS_2.md)
for the v1/v2 score formulas, units, workload-specific policy, ablation plan,
and the six-dataset comparison. Its smoke-suite summary is
[`output/optimized_prefix_5/ALL_PREFIX_METHOD_RESULTS_2.md`](output/optimized_prefix_5/ALL_PREFIX_METHOD_RESULTS_2.md),
and the OPUS-100 100-request comparison is under
`output/optimized_prefix_100/`.

Additional local-research experiments:

- [`PREFIX_SPLITTING_ABLATION_ANALYSIS_1_1.md`](PREFIX_SPLITTING_ABLATION_ANALYSIS_1_1.md):
  v1 Sentence/Keyword/Reuse candidate ablations;
- [`PREFIX_SPLITTING_ABLATION_ANALYSIS_2_1.md`](PREFIX_SPLITTING_ABLATION_ANALYSIS_2_1.md):
  v2 boundary, reuse, future-node, and prefetch-cost ablations;
- [`PREFIX_SPLITTING_ALGORITHM_ANALYSIS_3.md`](PREFIX_SPLITTING_ALGORITHM_ANALYSIS_3.md):
  the LoRA-affinity and user-focus filter v3 method, score, limitations, and results.

Their generated outputs use separate ignored directories:
`output/ablation_v1_{5,100}`, `output/ablation_v2_{5,100}`, and
`output/focus_filter_v3_{5,100}`. Figure/report suffixes are `_1_1`, `_2_1`,
and `_3`, respectively.

For a complete quick comparison across the other five datasets, use the
5-request smoke suite. It produces one figure and one report per dataset,
plus `ALL_PREFIX_METHOD_RESULTS.md`. This is intended to validate the
cross-dataset construction and chart pipeline; use the 100-request command
above for a stable OPUS-100 estimate and increase `--request-limit` for the
long continuous traces when runtime permits.

```powershell
D:\anaconda\envs\qwen2.5_vl\python.exe -B `
  examples/lora-base-test7/run_prefix_method_experiments.py `
  --datasets msc_continuous,taskmaster_continuous,lmsys_continuous,sharegpt_continuous,xsum_parallel `
  --request-limit 5 `
  --output-dir examples/lora-base-test7/output/all_prefix_5 `
  --continue-on-failure

$datasets = @("msc_continuous", "taskmaster_continuous", "lmsys_continuous", "sharegpt_continuous", "xsum_parallel")
foreach ($dataset in $datasets) {
  D:\anaconda\envs\qwen2.5_vl\python.exe -B `
    examples/lora-base-test7/make_prefix_method_figure.py `
    --dataset $dataset --request-limit 5 `
    --output-dir examples/lora-base-test7/output/all_prefix_5
}
D:\anaconda\envs\qwen2.5_vl\python.exe -B `
  examples/lora-base-test7/make_prefix_method_summary.py `
  --output-dir examples/lora-base-test7/output/all_prefix_5 --request-limit 5
```

The helper writes each method's boundaries into `prefix_segments`, runs the
same oracle-prefetch and tiered-storage policy, and pairs online TTFT with the
full-prefill baseline inside the same cell. Derived workloads and all result
files remain under the ignored `output/` directory.

This is a local research harness. The current physical adapters are routing
artifacts and are not semantically trained for the logical dataset roles.

中文数据集构造、100 请求统计和负 speedup 原因说明见
[`DATASET_CONSTRUCTION_100.md`](DATASET_CONSTRUCTION_100.md)。
