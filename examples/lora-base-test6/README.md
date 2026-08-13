# LoRA prefix-cache test6

This local experiment suite reuses the `lora-base-test4` system-v2 runtime and
adds a reproducible cross-dataset matrix, aggregation, plots, and a Markdown
report. Reusing the runtime keeps the GPU full-KV, host full/Q8-delta, disk
delta-store, prefix-tree, and CUDA overlap paths identical to the implementation
already exercised by test4 and test5.

The default matrix covers the five real 30-request workloads:

- MSC, Taskmaster, and LMSYS continuous growing-prefix traces;
- OPUS-100 and XSum exact shared-prefix parallel traces.

It compares full-prefill and legacy exact-prefix baselines with GPU-only,
GPU-plus-host-full, tiered no-prefetch, oracle-prefetched fixed chunks of
32/64/128/256 tokens, an `oracle_full` 128-token run with
`max-prefetch-chunks-per-lora=0`, and cold/warm disk delta-store runs. Oracle
prefetch is an upper bound, not a deployable predictor result; `0` means
unlimited chunk coverage, not zero prefetch.

The default matrix uses `n_ctx=65536` and a 61440-token cache budget. The MSC
30-request trace reaches about 42K physical cached tokens because growing
prefix nodes can hold several LoRA variants. Smaller budgets produced
`foreground_capacity_failed` rows. The runner treats any failed mode or
non-positive online TTFT as a failed cell instead of aggregating it.

## Build

```powershell
cmake --build build --config Release --target llama-lora-base-test6 --parallel 2
```

## Run and analyze

```powershell
D:\anaconda\envs\qwen2.5_vl\python.exe `
  examples/lora-base-test6/run_experiments.py
```

Use `--quick` for a five-request smoke run, `--force` to rerun completed cells,
and comma-separated `--datasets` or `--strategies` filters for a subset. The
runner stores complete commands and logs under `output/real_30_matrix/runs` and
then invokes `analyze_results.py`.

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

Run the five prefix methods online with:

```powershell
D:\anaconda\envs\qwen2.5_vl\python.exe `
  examples/lora-base-test6/run_prefix_method_experiments.py
```

The helper writes each method's boundaries into `prefix_segments`, runs the
same oracle-prefetch and tiered-storage policy, and pairs online TTFT with the
full-prefill baseline inside the same cell. Derived workloads and all result
files remain under the ignored `output/` directory.

This is a local research harness. The current physical adapters are routing
artifacts and are not semantically trained for the logical dataset roles.
