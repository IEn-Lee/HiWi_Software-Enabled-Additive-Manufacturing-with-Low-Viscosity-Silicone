# Corrected FIFO-weight optimizer v1.2

This version keeps the existing control-block LP structure but replaces the residence-time surrogate model.

## What was wrong before

The old optimizer read the baseline scalar `t_in` from the CSV and treated it as one event position on the G-code time axis. In the TRUE FIFO builder, however, cycle >= 2 computes `t_in` as a volume-weighted average of multiple previous-cycle `t_out` values.

Therefore the old model effectively transformed an already-averaged time. It lost the actual FIFO source-volume dependencies, so after non-uniform timing changes its predicted `t_in` could diverge strongly from TRUE FIFO.

## Corrected model

For every FIFO piece, the optimizer now constructs an exact linear expression for its output time:

`T_out(i) = linear function of block timings`

For cycle >= 2, it rebuilds the FIFO source mapping from `cycle`, `cycle_local`, and `V_mm3`, reproducing the TRUE builder's volume consumption order:

`T_in(i) = sum_k w(i,k) * T_out(k)`

Then:

`tau(i) = T_out(i) - T_in(i)`

The weights `w(i,k)` depend only on extrusion volume/order and therefore remain fixed as long as E/volume, cycle assignment, G-code order, and V_nozzle do not change. The optimization remains a linear program.

## What stays unchanged

- extrusion/travel independent F limits
- optional travel optimization
- optional G4 dwell optimization
- control blocks (`block_rows`, `block_duration_sec`)
- Stage 1 minimax LP
- Stage 2 target/change/smooth objectives
- G-code patching
- optional TRUE FIFO validation

## New required CSV fields

The corrected model needs the FIFO structure, so the input CSV must contain:

- `cycle`
- `cycle_local`
- `V_mm3`

in addition to the previous columns.

## New diagnostics

At startup the optimizer prints:

- baseline tau reconstruction error
- baseline `t_in` reconstruction error
- baseline `t_out` reconstruction error
- FIFO cycle/piece counts
- FIFO fallback volume if previous-cycle source volume is insufficient

For a compatible CSV/TRUE-builder pair, `tin_abs_max`, `tout_abs_max`, and `tau_abs_max` should be near numerical/CSV-rounding error before optimization.

## Suggested test command

Use the same command you used for the previous CFFFP test, changing only the Python filename:

```powershell
python residence_optimizer_sparse_fifo_weighted_v1_2.py --input "CFFFP_testmodel_E0.9_Temp32_780s_segmented.csv" --gcode_in "CFFFP_testmodel_E0.9_Temp32_780s_segmented.gcode" --gcode_out "CFFFP_testmodel_E0.9_Temp32_780s_fifo_weighted_optimized.gcode" --out_dir "CFFFP_testmodel_E0.9_Temp32_780s_fifo_weighted_optimization" --target_mode direct --tau_min 741 --tau_ideal 780 --tau_max 819 --Fmin_extrusion 50 --Fmax_extrusion 5000 --enable_travel_opt --enable_dwell_opt --block_rows 50 --block_duration_sec 12 --max_target_samples 5000 --change_weight 0.10 --smooth_weight 0.05 --target_weight 1.00 --vmax_tolerance_s 0.01 --true_validate --vnozzle 574 --Fmin_travel 50 --Fmax_travel 10000
```

## What to compare with the old run

The decisive fields are still:

- `true_minus_pred_mean_s`
- `true_minus_pred_rmse_s`
- `true_minus_pred_abs_max_s`

The old test had RMSE around 883 s. If the TRUE validation module uses the same FIFO rules reconstructed here, this corrected model should reduce the prediction gap substantially. If it does not, the next step is to compare the exact `rebuild_csv_and_gcode_fifo_v1_2_precondition_autofill` implementation against the available FIFO builder version, especially cycle-1 initialization and clock-origin rules.

## Validation performed here

A synthetic 3-cycle FIFO case with unequal segment volumes, travel/dwell delays, and deliberately non-uniform block timing changes was tested. The corrected LP residence equations matched an independent FIFO volume-backfill recomputation exactly (maximum difference 0 in the test), while baseline `t_in`, `t_out`, and tau reconstruction errors were at floating-point precision.
