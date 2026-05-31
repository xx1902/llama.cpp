# PreCoLoRA Prediction Experiments

This folder refactors the original exploratory notebook into reusable scripts for
the thesis prediction experiments.

## Main Entry

```bash
python experiments/run_prediction_experiments.py --input "D:/ecnu_experiment/LoRA/datasets/lsapp.tsv.gz"
```

If the LSApp file uses a different path, pass it with `--input`.

## Outputs

The script writes results to `outputs/prediction/`:

- `metrics_summary.csv`
- `window_sweep.csv`
- `alpha_sweep.csv`
- `temperature_sweep.csv`
- `fig5_1_training_curve.png`
- `fig5_2_window_top3.png`
- `fig5_3_method_topk.png`
- `fig5_4_alpha_sweep.png`
- `fig5_5_temperature_sweep.png`
- `fig3_3_relation_heatmap.png`
- `fig3_4_relation_topology.png`

## Expected Input Columns

The script accepts either the original LSApp-style columns or a simplified CSV
with these columns:

- `user_id`
- `timestamp`
- `app_name`
- `event_type`

Only `Opened` events are used when `event_type` exists.
