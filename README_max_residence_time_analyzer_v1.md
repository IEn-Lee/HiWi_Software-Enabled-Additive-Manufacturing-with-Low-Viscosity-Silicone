# Maximum Residence-Time Analyzer v1

## Purpose

This is a diagnostic tool for answering:

> Under the supplied minimum feedrates and maximum existing dwell times, what is the largest residence time physically reachable by timing-only G-code changes?

It does **not** generate a balanced production profile. It intentionally creates an all-slowest diagnostic profile.

## Logic

For fixed G-code order, fixed E/volume, fixed `V_nozzle`, and FIFO volume pairing:

- Slowing an extrusion command cannot reduce the residence time of material that is inside the nozzle during that command.
- Slowing an enabled travel command cannot reduce that residence time.
- Increasing an existing enabled optimizable G4 dwell cannot reduce it.

Therefore, setting every permitted timing control to its slowest allowed value simultaneously maximizes every residence time under those limits.

The program performs two checks:

1. **Predicted maximum:** maps baseline `t_in` and `t_out` event positions onto the all-slowest timeline.
2. **TRUE maximum:** optionally patches the G-code once and sends it through the existing FIFO builder.

Use the TRUE result as the authoritative result.

## Example for the LiQ5 Temp32 case

```bash
python max_residence_time_analyzer_v1.py \
  --input "LiQ5_sample_RTV-Body_Temp32_780s_segmented.csv" \
  --gcode_in "LiQ5_sample_RTV-Body_Temp32_780s_segmented.gcode" \
  --out_dir "LiQ5_Temp32_max_tau_analysis" \
  --Fmin_extrusion 100 \
  --enable_travel_opt \
  --Fmin_travel 100 \
  --enable_dwell_opt \
  --Pmax_ms 600000 \
  --required_tau_min 715.7522321536318 \
  --true_validate \
  --vnozzle 574 \
  --d_filament 15.55634919 \
  --e_mode filament
```

Set `Fmin_extrusion`, `Fmin_travel`, and `Pmax_ms` to real machine and printability limits. The result is only meaningful under those limits.

If travel or dwell must not be changed, omit the corresponding `--enable_*` flag.

## Select a particular segment

```bash
--target_seg_idx 12345
```

Without this option, the program selects the row with the lowest predicted maximum residence time, which is the true bottleneck under the fast model.

To select the row with the lowest baseline residence time instead:

```bash
--target_select min_baseline
```

## Important outputs

### `maximum_residence_summary.json`

First file to inspect.

Key TRUE fields:

```json
"true_maximum_profile": {
  "tau_min_s": 700.0,
  "bottleneck_seg_idx": 12345,
  "all_rows_reach_required_min": false,
  "unreachable_row_count": 15,
  "bottleneck_margin_s": -15.752
}
```

Interpretation:

- `tau_min_s`: lowest TRUE residence time after every permitted command is pushed to its slowest limit.
- `bottleneck_seg_idx`: segment that remains shortest.
- `all_rows_reach_required_min`: whether every scored row can reach `required_tau_min`.
- `bottleneck_margin_s`: `TRUE minimum tau - required_tau_min`.
  - Negative: timing-only control is physically insufficient under the supplied limits.
  - Zero or positive: the lower bound is reachable in the all-slowest profile, but satisfying the upper bound simultaneously still requires optimization.

### `maximum_residence_true_comparison.csv`

Row-by-row comparison between the fast prediction and TRUE FIFO result.

Important columns:

- `baseline_tau_s`
- `predicted_max_tau_s`
- `true_max_tau_s`
- `true_minus_predicted_tau_s`

### `maximum_residence_bottlenecks.csv`

The rows with the lowest predicted maximum tau, sorted from worst to best.

### `selected_target_contributions.csv`

Shows which G-code rows lie between the selected material's `t_in` and `t_out`, and how much each row can add to its residence time.

Useful columns:

- `control_kind`
- `overlap_fraction`
- `baseline_tau_contribution_s`
- `maximum_tau_contribution_s`
- `tau_gain_contribution_s`
- `maximum_profile_F_mm_per_min`
- `maximum_profile_P_ms`

### `maximum_residence_profile.gcode`

Diagnostic all-slowest G-code sent to TRUE FIFO. It is not intended as the final printing G-code.

## Decision rule

When `--true_validate` is enabled:

- If `true_maximum_profile.tau_min_s < required_tau_min`, then at least one formal print-path material unit cannot reach the lower residence-time requirement using only the enabled F/travel/dwell controls and supplied limits.
- If `true_maximum_profile.tau_min_s >= required_tau_min`, the lower bound is reachable, but a separate optimizer must still find a profile that also keeps all rows below `tau_max`.

## Assumptions

The physical conclusion depends on:

- Correct baseline segmented CSV.
- Same `V_nozzle`, `d_filament`, and `e_mode` in baseline and TRUE validation.
- Fixed G-code order and fixed E/volume.
- Feedrate and dwell limits representing actual feasible process limits.
- TRUE FIFO builder correctly representing the material flow.
