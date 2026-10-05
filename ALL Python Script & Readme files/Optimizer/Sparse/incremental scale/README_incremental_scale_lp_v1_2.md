# Incremental Scale LP Optimizer v1.3

This version changes the optimizer in two connected ways:

1. **Explicit scale variables in linear programming**
   - Every control block has an LP variable `scale_b`.
   - `scale = 1` keeps the current timing.
   - `scale > 1` increases duration and lowers movement feed rate.
   - `scale < 1` decreases duration and raises movement feed rate.
   - Cumulative `z` variables remain only as a sparse clock representation and are linked by:

     `z[b+1] - z[b] = base_duration[b] * scale[b]`

2. **Incremental-scale learning through TRUE FIFO relinearization**
   - Set `--iterations` above 1.
   - Each round limits scale to a local trust window around 1, such as `[0.8, 1.2]`.
   - The LP solves Stage 1 and Stage 2 for the current accepted CSV/G-code.
   - The candidate G-code is rebuilt with TRUE FIFO.
   - The candidate is accepted only when TRUE metrics improve in this priority:
     1. lower worst violation;
     2. fewer violating rows;
     3. lower total violation;
     4. lower mean target error.
   - Accepted TRUE results become the next baseline. Rejected results are discarded and the local scale window shrinks.

## Suggested command

```powershell
py residence_optimizer_incremental_scale_lp_v1_3.py `
  --input "LiQ5_sample_RTV-Body_Temp32_780s_segmented.csv" `
  --gcode_in "LiQ5_sample_RTV-Body_Temp32_780s_segmented.gcode" `
  --gcode_out "LiQ5_sample_RTV-Body_Temp32_780s_incremental_optimized.gcode" `
  --out_dir "LiQ5_sample_RTV-Body_Temp32_780s_incremental_optimization" `
  --target_mode alpha --temp_c 32.0 --alpha_mode bounds `
  --alpha_min 0.887156 --alpha_ideal 0.933848 --alpha_max 0.980540 `
  --Fmin_extrusion 1 --Fmax_extrusion 5000 `
  --enable_travel_opt --Fmin_travel 1 --Fmax_travel 10000 `
  --enable_dwell_opt --Pmax_ms 780000 `
  --block_rows 50 --block_duration_sec 4 `
  --max_target_samples 5000 `
  --change_weight 0.10 --smooth_weight 0.05 --target_weight 1.00 `
  --vmax_tolerance_s 0.01 `
  --iterations 8 `
  --increment_scale_limit 0.20 `
  --max_increment_scale_limit 0.40 `
  --min_increment_scale_limit 0.01 `
  --trust_shrink 0.50 --trust_grow 1.15 `
  --patience 3 `
  --true_validate --vnozzle 574
```

## Main outputs

- `iteration_###/control_blocks.csv`: incremental scale selected in each round.
- `iteration_###/stage1_checkpoint.npz`: saved Stage 1 scale solution before Stage 2.
- `iteration_###/true_validation.csv`: TRUE FIFO result for that candidate.
- `incremental_history.csv`: accepted/rejected history and TRUE metrics.
- `cumulative_scale_by_seg_idx.csv`: final TRUE row duration divided by original row duration.
- `final_true_validation.csv`: best accepted authoritative CSV.
- `summary.json`: complete settings and iteration history.

## Important interpretation

This is sequential linear programming with TRUE FIFO feedback, not a neural-network training method. The scale learned in each round is incremental relative to the latest accepted baseline. The total scale relative to the original file is the product of accepted incremental changes; the authoritative result is exported per `seg_idx` in `cumulative_scale_by_seg_idx.csv`.
