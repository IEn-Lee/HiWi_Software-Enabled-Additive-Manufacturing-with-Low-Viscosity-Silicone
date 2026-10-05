# Free Segment-Level Residence-Time LP Optimizer v1.0

## What changed

This version removes control-block grouping and does not preserve the original feed-rate distribution.
Every adjustable G-code segment is an independent control variable.

Conceptually, the optimizer chooses each segment execution time `t_j` independently.
The implementation uses cumulative sparse variables `z`, where:

`t_j = z[j+1] - z[j]`

This keeps the LP sparse without forcing any two segments to share a scale.

## Stage 1: full-freedom feasibility

For all scored residence rows:

`tau_min - vmax <= tau_i <= tau_max + vmax`

The LP minimizes `vmax`.

- `minimal_worst_violation_s = 0` (within numerical tolerance): the fixed-event linear model found a legal independent-segment timing profile that puts every scored residence row inside the target window.
- `minimal_worst_violation_s > 0`: even after removing block/relative-speed constraints, the current linear model still cannot place every scored row inside the window under the supplied F/P bounds.

This is the key diagnostic for determining whether the previous block structure was the bottleneck.

## Stage 2: target quality without preserving the old speed profile

Stage 2 keeps the Stage-1 worst violation and minimizes sampled absolute error to `tau_ideal`.
It deliberately does NOT use:

- `block_rows`
- `block_duration_sec`
- `change_weight`
- `smooth_weight`
- shared block scale

`max_target_samples` still limits only how many residence rows enter the ideal-target objective. All scored rows remain constrained by the residence-time window/minimax envelope.

## Physical bounds

Extrusion segments use `Fmin_extrusion/Fmax_extrusion`.
Travel segments use `Fmin_travel/Fmax_travel` when `--enable_travel_opt` is enabled.
Optimizable G4 dwell rows use `Pmin_ms/Pmax_ms` when `--enable_dwell_opt` is enabled.

`--max_relative_F_change 0` and `--max_relative_P_change 0` mean no relative-to-original restriction; only absolute machine/process limits remain.

## Recommended first diagnostic run

Use direct tau mode first if the target window is already known:

```powershell
py residence_optimizer_free_segment_lp_v1_0.py `
  --input "LiQ5_sample_RTV-Body_Temp32_780s_segmented.csv" `
  --gcode_in "LiQ5_sample_RTV-Body_Temp32_780s_segmented.gcode" `
  --gcode_out "LiQ5_sample_RTV-Body_Temp32_780s_free_segment_optimized.gcode" `
  --out_dir "LiQ5_sample_RTV-Body_Temp32_780s_free_segment_optimization" `
  --target_mode direct `
  --tau_min 715.752232 `
  --tau_ideal 780.001652 `
  --tau_max 863.097453 `
  --Fmin_extrusion 1 `
  --Fmax_extrusion 5000 `
  --enable_travel_opt `
  --Fmin_travel 1 `
  --Fmax_travel 10000 `
  --enable_dwell_opt `
  --Pmax_ms 780000 `
  --max_relative_F_change 0 `
  --max_relative_P_change 0 `
  --max_target_samples 5000 `
  --target_weight 1.0 `
  --vmax_tolerance_s 0.01 `
  --true_validate `
  --vnozzle 574
```

For a pure theoretical feasibility check with the smallest LP, add:

```powershell
--stage1_only
```

Then focus on:

- `iteration_001/round_summary.json -> stage1.minimal_worst_violation_s`
- `iteration_001/round_summary.json -> stage1.predicted_full_window_feasible`

## Optional TRUE-FIFO relinearization

Use for example:

```powershell
--iterations 3 --true_validate
```

Each accepted TRUE result becomes the baseline for another full-freedom segment LP.
Unlike the incremental-scale optimizer, there is no trust-region scale window; every round can reconstruct each segment anywhere inside the absolute F/P bounds.

## Main outputs

- `iteration_XXX/optimized_prediction.csv`
- `iteration_XXX/segment_controls.csv`
- `iteration_XXX/candidate_optimized.gcode`
- `iteration_XXX/true_validation.csv` (when TRUE validation runs)
- `iteration_XXX/round_summary.json`
- `optimization_history.csv`
- `summary.json`

## Important interpretation

The Stage-1 zero-violation result is a statement about the fixed-event sparse linear model. A TRUE FIFO rebuild remains necessary because large timing changes can invalidate the baseline event mapping.
