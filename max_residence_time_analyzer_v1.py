#!/usr/bin/env python3
"""
Determine the maximum physically reachable FIFO residence time under explicit
G-code timing limits.

The program is a diagnostic tool, not a production optimizer.

Core principle
--------------
For a fixed G-code order, fixed extrusion volume/E values, fixed V_nozzle, and
FIFO volume pairing, every residence time is monotonic with respect to the
execution durations that lie between its material-entry and material-exit
events. Therefore, the profile that maximizes all residence times is obtained
by applying the slowest allowed timing to every adjustable command:

* extrusion movement -> Fmin_extrusion
* optional travel movement -> Fmin_travel
* optional optimizable G4 dwell -> Pmax_ms

The tool:
1. Reads the authoritative baseline segmented CSV.
2. Builds a row-level maximum-duration profile.
3. Maps every baseline t_in/t_out event to the maximum-duration timeline and
   predicts the maximum reachable tau for every scored extrusion row.
4. Patches one diagnostic G-code containing the slowest allowed profile.
5. Optionally sends that G-code through the existing TRUE FIFO builder once.
6. Reports the bottleneck row and whether the required lower tau bound is
   physically reachable under the supplied limits.

Important interpretation
------------------------
"Maximum" always means maximum under the limits supplied on the command line.
The result is not an absolute material or machine limit. Fmin and Pmax must be
chosen from real machine, printability, and process constraints.

Example
-------
python max_residence_time_analyzer_v1.py \
  --input LiQ5_sample_RTV-Body_Temp32_780s_segmented.csv \
  --gcode_in LiQ5_sample_RTV-Body_Temp32_780s_segmented.gcode \
  --out_dir max_tau_test \
  --Fmin_extrusion 100 \
  --enable_travel_opt --Fmin_travel 100 \
  --enable_dwell_opt --Pmax_ms 600000 \
  --required_tau_min 715.752232 \
  --true_validate --vnozzle 574
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
SEGIDX_RE = re.compile(r"\bseg_idx\s*=\s*(?P<id>\d+)\b", re.IGNORECASE)


@dataclass
class Row:
    row_i: int
    seg_idx: int
    cmd: str
    d_mm: float
    F: float
    V_mm3: float
    t_s: float
    t_in: float
    t_out: float
    tau_s: float
    note: str
    is_extrusion: bool
    is_optimizable: bool


def finite_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_segments_csv(path: str) -> Tuple[List[Row], pd.DataFrame]:
    df = pd.read_csv(path)
    required = {
        "seg_idx", "cmd", "d_mm", "F_mm_per_min", "V_mm3", "t_s",
        "t_in", "t_out", "tau_s", "note",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    rows: List[Row] = []
    for row_i, rec in df.iterrows():
        note_raw = rec.get("note", "")
        note = "" if pd.isna(note_raw) else str(note_raw).strip()
        note_l = note.lower()
        rows.append(
            Row(
                row_i=int(row_i),
                seg_idx=int(rec["seg_idx"]),
                cmd=str(rec["cmd"]).strip().upper(),
                d_mm=finite_float(rec.get("d_mm", 0.0)),
                F=finite_float(rec.get("F_mm_per_min", 0.0)),
                V_mm3=finite_float(rec.get("V_mm3", 0.0)),
                t_s=max(0.0, finite_float(rec.get("t_s", 0.0))),
                t_in=finite_float(rec.get("t_in", math.nan), math.nan),
                t_out=finite_float(rec.get("t_out", math.nan), math.nan),
                tau_s=finite_float(rec.get("tau_s", math.nan), math.nan),
                note=note,
                is_extrusion=("extrusion" in note_l),
                is_optimizable=("optimizable" in note_l),
            )
        )

    order = np.argsort([r.seg_idx for r in rows], kind="stable")
    if not np.array_equal(order, np.arange(len(rows))):
        rows = [rows[int(i)] for i in order]
        df = df.iloc[order].reset_index(drop=True)
        for i, row in enumerate(rows):
            row.row_i = i

    segs = [r.seg_idx for r in rows]
    if len(set(segs)) != len(segs):
        raise ValueError("seg_idx values must be unique for reliable G-code patching")

    return rows, df


def has_positive_extrusion_volume(row: Row) -> bool:
    return row.is_extrusion and math.isfinite(row.V_mm3) and row.V_mm3 > EPS


def is_scored_extrusion_row(row: Row) -> bool:
    if not has_positive_extrusion_volume(row):
        return False
    if row.cmd == "G0" and "travel_extrusion" in row.note.lower():
        return False
    return True


def is_control_target_row(row: Row) -> bool:
    if row.is_extrusion and not has_positive_extrusion_volume(row):
        return False
    return row.is_extrusion or row.is_optimizable


def is_travel_row(row: Row) -> bool:
    if row.is_extrusion or row.is_optimizable:
        return False
    return row.d_mm > EPS and row.cmd in {"G0", "G1"}


def infer_feedrate(row: Row) -> float:
    if row.F > EPS:
        return row.F
    if row.d_mm > EPS and row.t_s > EPS:
        return 60.0 * row.d_mm / row.t_s
    return 0.0


def max_duration_for_movement(
    row: Row,
    fmin: float,
    max_relative_f_change: float,
) -> Tuple[float, float, float]:
    """Return (max_duration, resulting_F, scale)."""
    f0 = infer_feedrate(row)
    if row.t_s <= EPS or f0 <= EPS:
        return row.t_s, row.F, 1.0

    max_scale = f0 / fmin
    if max_relative_f_change > 0.0:
        r = min(max_relative_f_change, 0.95)
        max_scale = min(max_scale, 1.0 / (1.0 - r))

    max_scale = max(0.0, max_scale)
    t_max = row.t_s * max_scale
    f_new = f0 / max(max_scale, EPS)
    return t_max, f_new, max_scale


def build_maximum_duration_profile(
    rows: Sequence[Row],
    fmin_extrusion: float,
    fmin_travel: float,
    pmax_ms: float,
    enable_travel_opt: bool,
    enable_dwell_opt: bool,
    max_relative_f_change: float,
    max_relative_p_change: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    n = len(rows)
    t_max = np.array([r.t_s for r in rows], dtype=float)
    f_new = np.array([r.F for r in rows], dtype=float)
    p_new_ms = np.array([r.t_s * 1000.0 if r.cmd == "G4" else 0.0 for r in rows], dtype=float)
    scale = np.ones(n, dtype=float)
    kind = ["fixed"] * n

    for i, row in enumerate(rows):
        if row.t_s <= EPS:
            continue

        if row.cmd == "G4" and row.is_optimizable and enable_dwell_opt:
            t_limit = pmax_ms / 1000.0
            if max_relative_p_change > 0.0:
                t_limit = min(t_limit, row.t_s * (1.0 + max_relative_p_change))
            t_max[i] = max(0.0, t_limit)
            p_new_ms[i] = t_max[i] * 1000.0
            scale[i] = t_max[i] / row.t_s
            kind[i] = "dwell"
            continue

        if row.cmd == "G1" and is_control_target_row(row) and infer_feedrate(row) > EPS:
            t_max[i], f_new[i], scale[i] = max_duration_for_movement(
                row, fmin_extrusion, max_relative_f_change
            )
            kind[i] = "extrusion"
            continue

        if enable_travel_opt and is_travel_row(row) and infer_feedrate(row) > EPS:
            t_max[i], f_new[i], scale[i] = max_duration_for_movement(
                row, fmin_travel, max_relative_f_change
            )
            kind[i] = "travel"

    return t_max, f_new, p_new_ms, scale, kind


def map_event_times(
    event_times: np.ndarray,
    baseline_duration: np.ndarray,
    new_duration: np.ndarray,
) -> np.ndarray:
    """Map baseline event positions to a new timeline preserving row fraction."""
    base_edges = np.concatenate(([0.0], np.cumsum(baseline_duration)))
    new_edges = np.concatenate(([0.0], np.cumsum(new_duration)))
    total_base = float(base_edges[-1])
    total_new = float(new_edges[-1])
    n = baseline_duration.size

    out = np.empty(event_times.size, dtype=float)
    for q, raw_t in enumerate(event_times):
        t = float(raw_t)
        if not math.isfinite(t):
            out[q] = math.nan
            continue
        if t <= 0.0:
            out[q] = t
            continue
        if t >= total_base:
            out[q] = total_new + (t - total_base)
            continue

        i = int(np.searchsorted(base_edges, t, side="right") - 1)
        i = min(max(i, 0), n - 1)

        # Repeated edges may point at a zero-duration row. Move to the next
        # positive-duration row while preserving the same boundary location.
        while i < n - 1 and baseline_duration[i] <= EPS:
            i += 1

        base_t = baseline_duration[i]
        if base_t <= EPS:
            out[q] = new_edges[i]
            continue

        frac = (t - base_edges[i]) / base_t
        frac = min(1.0, max(0.0, frac))
        out[q] = new_edges[i] + frac * new_duration[i]

    return out


def predict_maximum_tau(
    rows: Sequence[Row],
    max_duration: np.ndarray,
) -> pd.DataFrame:
    scored = [r for r in rows if is_scored_extrusion_row(r)]
    if not scored:
        raise ValueError("No scored extrusion rows were found")

    tin = np.array([r.t_in for r in scored], dtype=float)
    tout = np.array([r.t_out for r in scored], dtype=float)
    tau0 = np.array([r.tau_s for r in scored], dtype=float)
    baseline_duration = np.array([r.t_s for r in rows], dtype=float)

    mapped_tin = map_event_times(tin, baseline_duration, max_duration)
    mapped_tout = map_event_times(tout, baseline_duration, max_duration)

    # Preserve any small authoritative baseline residual from CSV rounding or
    # builder-specific offsets.
    baseline_event_tau = tout - tin
    residual = tau0 - baseline_event_tau
    tau_max = mapped_tout - mapped_tin + residual

    return pd.DataFrame(
        {
            "row_i": [r.row_i for r in scored],
            "seg_idx": [r.seg_idx for r in scored],
            "cmd": [r.cmd for r in scored],
            "note": [r.note for r in scored],
            "V_mm3": [r.V_mm3 for r in scored],
            "baseline_t_in_s": tin,
            "baseline_t_out_s": tout,
            "baseline_tau_s": tau0,
            "predicted_max_t_in_s": mapped_tin,
            "predicted_max_t_out_s": mapped_tout,
            "predicted_max_tau_s": tau_max,
            "predicted_tau_gain_s": tau_max - tau0,
            "predicted_tau_gain_ratio": np.divide(
                tau_max,
                tau0,
                out=np.full_like(tau_max, np.nan),
                where=np.abs(tau0) > EPS,
            ),
        }
    )


def event_overlap_fractions(
    rows: Sequence[Row],
    t_in: float,
    t_out: float,
) -> np.ndarray:
    durations = np.array([r.t_s for r in rows], dtype=float)
    starts = np.concatenate(([0.0], np.cumsum(durations)[:-1]))
    ends = starts + durations
    overlap = np.maximum(0.0, np.minimum(ends, t_out) - np.maximum(starts, t_in))
    return np.divide(overlap, durations, out=np.zeros_like(overlap), where=durations > EPS)


def build_target_contributions(
    rows: Sequence[Row],
    target_row: Row,
    max_duration: np.ndarray,
    f_new: np.ndarray,
    p_new_ms: np.ndarray,
    scale: np.ndarray,
    kind: Sequence[str],
) -> pd.DataFrame:
    frac = event_overlap_fractions(rows, target_row.t_in, target_row.t_out)
    selected = np.flatnonzero(frac > EPS)
    records = []
    for i in selected:
        row = rows[int(i)]
        base_contrib = frac[i] * row.t_s
        max_contrib = frac[i] * max_duration[i]
        records.append(
            {
                "row_i": row.row_i,
                "seg_idx": row.seg_idx,
                "cmd": row.cmd,
                "note": row.note,
                "control_kind": kind[i],
                "overlap_fraction": frac[i],
                "baseline_row_duration_s": row.t_s,
                "maximum_row_duration_s": max_duration[i],
                "baseline_tau_contribution_s": base_contrib,
                "maximum_tau_contribution_s": max_contrib,
                "tau_gain_contribution_s": max_contrib - base_contrib,
                "baseline_F_mm_per_min": row.F,
                "maximum_profile_F_mm_per_min": f_new[i],
                "maximum_profile_P_ms": p_new_ms[i],
                "time_scale": scale[i],
            }
        )
    return pd.DataFrame(records)


def extract_seg_idx(comment: str) -> Optional[int]:
    match = SEGIDX_RE.search(comment or "")
    return int(match.group("id")) if match else None


def gcode_cmd_token(code: str) -> str:
    tokens = code.strip().split()
    if not tokens:
        return ""
    if tokens[0].upper().startswith("N") and tokens[0][1:].isdigit():
        tokens = tokens[1:]
        if not tokens:
            return ""
    token = tokens[0].upper()
    if token.startswith("G") and token[1:].isdigit():
        token = "G" + str(int(token[1:]))
    return token


def replace_or_append_parameter(parts: List[str], letter: str, value: str) -> List[str]:
    letter = letter.upper()
    output: List[str] = []
    replaced = False
    for token in parts:
        if token.upper().startswith(letter):
            output.append(f"{letter}{value}")
            replaced = True
        else:
            output.append(token)
    if not replaced:
        output.append(f"{letter}{value}")
    return output


def patch_gcode_by_segidx(
    gcode_in: str,
    gcode_out: str,
    rows: Sequence[Row],
    f_new: np.ndarray,
    p_new_ms: np.ndarray,
    enable_travel_opt: bool,
    enable_dwell_opt: bool,
) -> Tuple[int, int]:
    seg_to_i = {row.seg_idx: i for i, row in enumerate(rows)}
    patched = 0
    expected = 0

    def patchable(cmd: str, has_e: bool, has_p: bool, row: Row) -> Optional[str]:
        if cmd == "G1" and has_e and row.cmd == "G1" and is_control_target_row(row):
            return "F"
        if enable_travel_opt and cmd in {"G0", "G1"} and not has_e and is_travel_row(row):
            return "F"
        if enable_dwell_opt and cmd == "G4" and has_p and row.cmd == "G4" and row.is_optimizable:
            return "P"
        return None

    with open(gcode_in, "r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            raw = line.rstrip("\n")
            code, sep, comment = raw.partition(";")
            seg_idx = extract_seg_idx(comment if sep else "")
            if seg_idx is None or seg_idx not in seg_to_i:
                continue
            tokens = code.strip().split()
            cmd = gcode_cmd_token(code)
            has_e = any(t.upper().startswith("E") for t in tokens)
            has_p = any(t.upper().startswith("P") for t in tokens)
            if patchable(cmd, has_e, has_p, rows[seg_to_i[seg_idx]]) is not None:
                expected += 1

    os.makedirs(os.path.dirname(os.path.abspath(gcode_out)), exist_ok=True)
    with open(gcode_in, "r", encoding="utf-8", errors="ignore") as fin, open(
        gcode_out, "w", encoding="utf-8", newline=""
    ) as fout:
        for line in fin:
            raw = line.rstrip("\n")
            code, sep, comment = raw.partition(";")
            seg_idx = extract_seg_idx(comment if sep else "")
            if seg_idx is None or seg_idx not in seg_to_i:
                fout.write(line)
                continue

            i = seg_to_i[seg_idx]
            row = rows[i]
            parts = code.strip().split()
            cmd = gcode_cmd_token(code)
            has_e = any(t.upper().startswith("E") for t in parts)
            has_p = any(t.upper().startswith("P") for t in parts)
            mode = patchable(cmd, has_e, has_p, row)
            if mode == "F":
                parts = replace_or_append_parameter(parts, "F", f"{float(f_new[i]):.5f}")
            elif mode == "P":
                parts = replace_or_append_parameter(parts, "P", str(int(round(float(p_new_ms[i])))))
            else:
                fout.write(line)
                continue

            new_code = " ".join(parts)
            fout.write(f"{new_code} ; {comment.strip()}\n" if sep and comment.strip() else f"{new_code}\n")
            patched += 1

    return patched, expected


def run_true_validation(
    module_name: str,
    gcode_in: str,
    out_dir: str,
    vnozzle: float,
    d_filament: float,
    e_mode: str,
) -> Tuple[str, str]:
    module = importlib.import_module(module_name)
    rebuild_global = getattr(module, "rebuild_global")
    csv_out = os.path.join(out_dir, "maximum_profile_true_validation.csv")
    gcode_out = os.path.join(out_dir, "maximum_profile_true_rebuilt.gcode")
    rebuild_global(
        gcode_in=gcode_in,
        csv_out=csv_out,
        gcode_out=gcode_out,
        V_nozzle=vnozzle,
        d_filament=d_filament,
        e_mode=e_mode,
    )
    if not os.path.exists(csv_out):
        raise RuntimeError("TRUE FIFO builder did not create validation CSV")
    return csv_out, gcode_out


def merge_true_results(prediction: pd.DataFrame, true_csv: str) -> pd.DataFrame:
    true_df = pd.read_csv(true_csv)
    required = {"seg_idx", "tau_s"}
    missing = required.difference(true_df.columns)
    if missing:
        raise ValueError(f"TRUE validation CSV is missing columns: {sorted(missing)}")

    true_df = true_df.copy()
    true_df["seg_idx"] = pd.to_numeric(true_df["seg_idx"], errors="coerce")
    true_df["tau_s"] = pd.to_numeric(true_df["tau_s"], errors="coerce")
    true_df = true_df.dropna(subset=["seg_idx"]).copy()
    true_df["seg_idx"] = true_df["seg_idx"].astype(int)

    duplicate_count = int(true_df["seg_idx"].duplicated(keep=False).sum())
    if duplicate_count:
        print(f"Warning: TRUE CSV contains {duplicate_count} rows with duplicated seg_idx; keeping the last occurrence")
        true_df = true_df.drop_duplicates(subset=["seg_idx"], keep="last")

    keep_cols = ["seg_idx", "tau_s"]
    for col in ("t_in", "t_out", "note", "cmd"):
        if col in true_df.columns:
            keep_cols.append(col)
    true_small = true_df[keep_cols].rename(
        columns={
            "tau_s": "true_max_tau_s",
            "t_in": "true_max_t_in_s",
            "t_out": "true_max_t_out_s",
            "note": "true_note",
            "cmd": "true_cmd",
        }
    )

    merged = prediction.merge(true_small, on="seg_idx", how="left", validate="one_to_one")
    merged["true_minus_predicted_tau_s"] = merged["true_max_tau_s"] - merged["predicted_max_tau_s"]
    return merged


def choose_target(prediction: pd.DataFrame, target_seg_idx: Optional[int], target_select: str) -> pd.Series:
    if target_seg_idx is not None:
        match = prediction[prediction["seg_idx"] == int(target_seg_idx)]
        if match.empty:
            raise ValueError(f"target seg_idx={target_seg_idx} is not a scored extrusion row")
        return match.iloc[0]
    if target_select == "min_baseline":
        return prediction.loc[prediction["baseline_tau_s"].idxmin()]
    return prediction.loc[prediction["predicted_max_tau_s"].idxmin()]


def summarize_profile(
    prediction: pd.DataFrame,
    target: pd.Series,
    required_tau_min: Optional[float],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "scored_row_count": int(len(prediction)),
        "baseline": {
            "tau_min_s": float(prediction["baseline_tau_s"].min()),
            "tau_max_s": float(prediction["baseline_tau_s"].max()),
            "tau_mean_s": float(prediction["baseline_tau_s"].mean()),
        },
        "predicted_maximum_profile": {
            "tau_min_s": float(prediction["predicted_max_tau_s"].min()),
            "tau_max_s": float(prediction["predicted_max_tau_s"].max()),
            "tau_mean_s": float(prediction["predicted_max_tau_s"].mean()),
        },
        "selected_target": {
            "seg_idx": int(target["seg_idx"]),
            "baseline_tau_s": float(target["baseline_tau_s"]),
            "predicted_max_tau_s": float(target["predicted_max_tau_s"]),
            "predicted_gain_s": float(target["predicted_tau_gain_s"]),
            "predicted_gain_ratio": (
                None if not math.isfinite(float(target["predicted_tau_gain_ratio"]))
                else float(target["predicted_tau_gain_ratio"])
            ),
        },
    }
    if required_tau_min is not None:
        below = prediction["predicted_max_tau_s"] < required_tau_min - 1e-9
        summary["required_tau_min_s"] = float(required_tau_min)
        summary["predicted_reachability"] = {
            "all_rows_reach_required_min": bool(not below.any()),
            "unreachable_row_count": int(below.sum()),
            "unreachable_rate": float(below.mean()),
            "bottleneck_margin_s": float(prediction["predicted_max_tau_s"].min() - required_tau_min),
        }
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Determine maximum reachable residence time under explicit timing limits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Authoritative segmented CSV with baseline t_in/t_out/tau_s")
    ap.add_argument("--gcode_in", default=None, help="Segmented G-code containing seg_idx comments")
    ap.add_argument("--gcode_out", default=None, help="Diagnostic all-slowest G-code output")
    ap.add_argument("--out_dir", default="maximum_residence_time_analysis")

    ap.add_argument("--Fmin", type=float, default=None, help="Fallback minimum feedrate for extrusion and travel")
    ap.add_argument("--Fmin_extrusion", type=float, default=None)
    ap.add_argument("--Fmin_travel", type=float, default=None)
    ap.add_argument("--Pmax_ms", type=float, default=600000.0)
    ap.add_argument("--enable_travel_opt", action="store_true")
    ap.add_argument("--enable_dwell_opt", action="store_true")
    ap.add_argument("--max_relative_F_change", type=float, default=0.0,
                    help="Optional maximum fractional decrease from original F; 0 disables")
    ap.add_argument("--max_relative_P_change", type=float, default=0.0,
                    help="Optional maximum fractional increase from original dwell; 0 disables")

    ap.add_argument("--required_tau_min", type=float, default=None,
                    help="Required lower residence-time bound used for feasibility diagnosis")
    ap.add_argument("--target_seg_idx", type=int, default=None)
    ap.add_argument("--target_select", choices=["bottleneck", "min_baseline"], default="bottleneck",
                    help="Automatic target when target_seg_idx is omitted")
    ap.add_argument("--top_n", type=int, default=20, help="Number of lowest maximum-tau rows to export")

    ap.add_argument("--true_validate", action="store_true")
    ap.add_argument("--true_builder_module", default="rebuild_csv_and_gcode_fifo_v1_2_precondition_autofill")
    ap.add_argument("--vnozzle", type=float, default=574.0)
    ap.add_argument("--d_filament", type=float, default=15.55634919)
    ap.add_argument("--e_mode", choices=["filament", "mm3"], default="filament")
    return ap.parse_args()


def resolve_limits(args: argparse.Namespace) -> Tuple[float, float]:
    fext = args.Fmin_extrusion if args.Fmin_extrusion is not None else args.Fmin
    ftrav = args.Fmin_travel if args.Fmin_travel is not None else args.Fmin
    if fext is None:
        raise ValueError("Provide --Fmin_extrusion or --Fmin")
    if ftrav is None:
        ftrav = fext
    if fext <= 0.0 or ftrav <= 0.0:
        raise ValueError("Minimum feedrates must be > 0")
    if args.Pmax_ms < 0.0:
        raise ValueError("Pmax_ms must be >= 0")
    if not (0.0 <= args.max_relative_F_change < 1.0):
        raise ValueError("max_relative_F_change must satisfy 0 <= value < 1")
    if args.max_relative_P_change < 0.0:
        raise ValueError("max_relative_P_change must be >= 0")
    if args.true_validate and not args.gcode_in:
        raise ValueError("--true_validate requires --gcode_in")
    return float(fext), float(ftrav)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    try:
        fmin_extrusion, fmin_travel = resolve_limits(args)
        os.makedirs(args.out_dir, exist_ok=True)

        rows, source_df = read_segments_csv(args.input)
        print(f"Loaded {len(rows):,} rows")

        max_duration, f_new, p_new_ms, scale, control_kind = build_maximum_duration_profile(
            rows=rows,
            fmin_extrusion=fmin_extrusion,
            fmin_travel=fmin_travel,
            pmax_ms=args.Pmax_ms,
            enable_travel_opt=args.enable_travel_opt,
            enable_dwell_opt=args.enable_dwell_opt,
            max_relative_f_change=args.max_relative_F_change,
            max_relative_p_change=args.max_relative_P_change,
        )

        prediction = predict_maximum_tau(rows, max_duration)
        target = choose_target(prediction, args.target_seg_idx, args.target_select)
        target_row = rows[int(target["row_i"])]

        profile_df = source_df.copy()
        profile_df["control_kind"] = control_kind
        profile_df["maximum_time_scale"] = scale
        profile_df["maximum_t_s"] = max_duration
        profile_df["maximum_F_mm_per_min"] = f_new
        profile_df["maximum_P_ms"] = p_new_ms
        profile_csv = os.path.join(args.out_dir, "maximum_timing_profile.csv")
        profile_df.to_csv(profile_csv, index=False, float_format="%.8f")

        prediction_csv = os.path.join(args.out_dir, "maximum_residence_prediction.csv")
        prediction.to_csv(prediction_csv, index=False, float_format="%.8f")

        bottlenecks = prediction.nsmallest(max(1, args.top_n), "predicted_max_tau_s")
        bottleneck_csv = os.path.join(args.out_dir, "maximum_residence_bottlenecks.csv")
        bottlenecks.to_csv(bottleneck_csv, index=False, float_format="%.8f")

        contributions = build_target_contributions(
            rows, target_row, max_duration, f_new, p_new_ms, scale, control_kind
        )
        contributions_csv = os.path.join(args.out_dir, "selected_target_contributions.csv")
        contributions.to_csv(contributions_csv, index=False, float_format="%.8f")

        summary = {
            "input": os.path.abspath(args.input),
            "gcode_in": os.path.abspath(args.gcode_in) if args.gcode_in else None,
            "limits": {
                "Fmin_extrusion": fmin_extrusion,
                "Fmin_travel": fmin_travel,
                "travel_optimization_enabled": bool(args.enable_travel_opt),
                "Pmax_ms": float(args.Pmax_ms),
                "dwell_optimization_enabled": bool(args.enable_dwell_opt),
                "max_relative_F_change": float(args.max_relative_F_change),
                "max_relative_P_change": float(args.max_relative_P_change),
            },
            "assumption": (
                "G-code order, E/volume, V_nozzle, and FIFO volume pairing remain fixed; "
                "maximum means maximum under the supplied timing limits."
            ),
        }
        summary.update(summarize_profile(prediction, target, args.required_tau_min))

        print("=" * 78)
        print(f"Baseline minimum tau: {prediction['baseline_tau_s'].min():.6f} s")
        print(f"Predicted maximum-profile minimum tau: {prediction['predicted_max_tau_s'].min():.6f} s")
        print(
            f"Selected target seg_idx={int(target['seg_idx'])}: "
            f"baseline={float(target['baseline_tau_s']):.6f} s, "
            f"predicted maximum={float(target['predicted_max_tau_s']):.6f} s"
        )

        diagnostic_gcode = args.gcode_out
        if diagnostic_gcode is None and args.gcode_in:
            diagnostic_gcode = os.path.join(args.out_dir, "maximum_residence_profile.gcode")

        if args.gcode_in and diagnostic_gcode:
            patched, expected = patch_gcode_by_segidx(
                gcode_in=args.gcode_in,
                gcode_out=diagnostic_gcode,
                rows=rows,
                f_new=f_new,
                p_new_ms=p_new_ms,
                enable_travel_opt=args.enable_travel_opt,
                enable_dwell_opt=args.enable_dwell_opt,
            )
            coverage = patched / max(1, expected)
            summary["patch"] = {
                "gcode_out": os.path.abspath(diagnostic_gcode),
                "patched": int(patched),
                "expected": int(expected),
                "coverage": float(coverage),
            }
            print(f"G-code patch coverage: {patched}/{expected} ({coverage:.6f})")
            if coverage < 0.98:
                raise RuntimeError("G-code patch coverage is below 98%; TRUE result would be unreliable")

        if args.true_validate:
            assert diagnostic_gcode is not None
            true_csv, true_gcode = run_true_validation(
                module_name=args.true_builder_module,
                gcode_in=diagnostic_gcode,
                out_dir=args.out_dir,
                vnozzle=args.vnozzle,
                d_filament=args.d_filament,
                e_mode=args.e_mode,
            )
            comparison = merge_true_results(prediction, true_csv)
            comparison_csv = os.path.join(args.out_dir, "maximum_residence_true_comparison.csv")
            comparison.to_csv(comparison_csv, index=False, float_format="%.8f")

            valid = comparison["true_max_tau_s"].notna()
            matched = comparison.loc[valid]
            if matched.empty:
                raise RuntimeError("No scored rows could be matched to TRUE validation")

            errors = matched["true_minus_predicted_tau_s"].to_numpy(dtype=float)
            true_bottleneck = matched.loc[matched["true_max_tau_s"].idxmin()]
            target_true = matched[matched["seg_idx"] == int(target["seg_idx"])]

            true_summary: Dict[str, object] = {
                "matched_rows": int(len(matched)),
                "tau_min_s": float(matched["true_max_tau_s"].min()),
                "tau_max_s": float(matched["true_max_tau_s"].max()),
                "tau_mean_s": float(matched["true_max_tau_s"].mean()),
                "bottleneck_seg_idx": int(true_bottleneck["seg_idx"]),
                "prediction_error_mean_s": float(np.mean(errors)),
                "prediction_error_rmse_s": float(math.sqrt(np.mean(np.square(errors)))),
                "prediction_error_abs_max_s": float(np.max(np.abs(errors))),
                "csv": os.path.abspath(true_csv),
                "rebuilt_gcode": os.path.abspath(true_gcode),
                "comparison_csv": os.path.abspath(comparison_csv),
            }
            if not target_true.empty:
                true_summary["selected_target"] = {
                    "seg_idx": int(target_true.iloc[0]["seg_idx"]),
                    "true_max_tau_s": float(target_true.iloc[0]["true_max_tau_s"]),
                    "true_minus_predicted_tau_s": float(target_true.iloc[0]["true_minus_predicted_tau_s"]),
                }

            if args.required_tau_min is not None:
                below_true = matched["true_max_tau_s"] < args.required_tau_min - 1e-9
                true_summary["required_tau_min_s"] = float(args.required_tau_min)
                true_summary["all_rows_reach_required_min"] = bool(not below_true.any())
                true_summary["unreachable_row_count"] = int(below_true.sum())
                true_summary["unreachable_rate"] = float(below_true.mean())
                true_summary["bottleneck_margin_s"] = float(
                    matched["true_max_tau_s"].min() - args.required_tau_min
                )

            summary["true_maximum_profile"] = true_summary
            print(f"TRUE maximum-profile minimum tau: {true_summary['tau_min_s']:.6f} s")
            print(
                "TRUE vs predicted: "
                f"RMSE={true_summary['prediction_error_rmse_s']:.6f} s, "
                f"abs max={true_summary['prediction_error_abs_max_s']:.6f} s"
            )

        summary["outputs"] = {
            "maximum_timing_profile_csv": os.path.abspath(profile_csv),
            "maximum_residence_prediction_csv": os.path.abspath(prediction_csv),
            "bottleneck_csv": os.path.abspath(bottleneck_csv),
            "selected_target_contributions_csv": os.path.abspath(contributions_csv),
        }
        summary["elapsed_s"] = float(time.perf_counter() - started)

        summary_path = os.path.join(args.out_dir, "maximum_residence_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"Summary written to: {summary_path}")
        if args.required_tau_min is not None:
            source = summary.get("true_maximum_profile", summary.get("predicted_reachability", {}))
            reachable = source.get("all_rows_reach_required_min")
            if reachable is True:
                print("RESULT: Every scored row can reach the required lower tau bound under this maximum profile.")
            elif reachable is False:
                print("RESULT: At least one scored row cannot reach the required lower tau bound under the supplied limits.")

        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
