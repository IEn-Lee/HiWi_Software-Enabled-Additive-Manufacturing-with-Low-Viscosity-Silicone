#!/usr/bin/env python3
"""
Sparse global residence-time optimizer for segmented G-code.

Version 1.3: explicit scale LP with iterative incremental-scale learning.

Extrusion and travel retain independent feed-rate limits. Each control block's
normalized time scale is now an explicit LP decision variable. Cumulative
clock variables are retained only as a sparse bookkeeping layer, linked to
scales by linear equalities.

Core idea
---------
The segmented CSV already contains an authoritative baseline FIFO result
(`t_in`, `t_out`, and `tau_s`). If extrusion volume, segment order,
V_nozzle, and E values remain unchanged, changing feed rate or dwell time
changes event timing but not the FIFO material pairing.

This program therefore:
1. Infers each scored material's input/output event positions from the
   baseline timeline once.
2. Groups adjacent adjustable rows into a small number of control blocks.
3. Creates one explicit scale variable per control block. A scale of 1 keeps
   the original timing, >1 slows the block, and <1 speeds it up.
4. Links scales to sparse cumulative-time variables with linear equalities,
   then expresses every residence time from its entry/exit events.
5. Solves two global linear programs:
   - Stage 1: minimize the worst residence-time violation.
   - Stage 2: preserve that optimum while minimizing target error, speed
     changes, and block-to-block roughness.
6. Optionally repeats the optimization as incremental-scale learning:
   each LP round searches only a local scale window around 1, patches the
   current G-code, runs TRUE FIFO, accepts only a real improvement, and then
   rebuilds the next linear model from the accepted TRUE result.
7. Writes the best accepted G-code, TRUE CSV, iteration history, and cumulative
   per-row scale relative to the original input.

Incremental-scale learning example
----------------------------------
Use --iterations greater than 1. Each round searches a local scale around the
current accepted TRUE FIFO baseline, validates the candidate, and rebuilds the
next model only after acceptance.

python residence_optimizer_incremental_scale_lp_v1_3.py \
  --input model_segmented.csv \
  --gcode_in model_segmented.gcode \
  --gcode_out model_incremental_optimized.gcode \
  --target_mode direct --tau_min 715.75 --tau_ideal 780 --tau_max 863.10 \
  --Fmin_extrusion 1 --Fmax_extrusion 5000 \
  --enable_travel_opt --Fmin_travel 1 --Fmax_travel 10000 \
  --enable_dwell_opt --Pmax_ms 780000 \
  --block_rows 50 --block_duration_sec 4 \
  --iterations 8 --increment_scale_limit 0.20 \
  --trust_shrink 0.5 --trust_grow 1.15 \
  --true_validate --vnozzle 574

Important assumption
--------------------
This model is valid only while G-code order, extrusion volume/E values,
V_nozzle, and material-flow interpretation remain unchanged. It optimizes
only timing (F and optional G4 P).

Example (alpha bounds)
----------------------
python residence_optimizer_sparse_scale_lp_v1_2.py \
  --input model_segmented.csv \
  --gcode_in model_segmented.gcode \
  --gcode_out model_sparse_optimized.gcode \
  --temp_c 25 \
  --alpha_mode bounds \
  --alpha_min 0.906607 \
  --alpha_ideal 0.923072 \
  --alpha_max 0.938468 \
  --Fmin_extrusion 180 --Fmax_extrusion 3000 \
  --enable_travel_opt \
  --enable_dwell_opt \
  --Pmin_ms \
  --Pmax_ms \
  --block_rows 150 \
  --block_duration_sec 12 \
  --out_dir sparse_optimization

Example (direct tau bounds)
---------------------------
python residence_optimizer_sparse_scale_lp_v1_2.py \
  --input model_segmented.csv \
  --target_mode direct \
  --tau_min 1940 --tau_ideal 2000 --tau_max 2060 \
  --Fmin_extrusion 180 --Fmax_extrusion 3000 \
  --out_dir sparse_optimization

LiQ5 Example
py residence_optimizer_incremental_scale_lp_v1_3.py
  --input "LiQ5_sample_RTV-Body_Temp32_780s_segmented.csv"
  --gcode_in "LiQ5_sample_RTV-Body_Temp32_780s_segmented.gcode"
  --gcode_out "LiQ5_sample_RTV-Body_Temp32_780s_incremental_optimized.gcode"
  --out_dir "LiQ5_sample_RTV-Body_Temp32_780s_incremental_optimization"
  --target_mode alpha
  --temp_c 32.0
  --alpha_mode bounds
  --alpha_min 0.887156
  --alpha_ideal 0.933848
  --alpha_max 0.980540
  --Fmin_extrusion 1
  --Fmax_extrusion 5000
  --enable_travel_opt
  --Fmin_travel 1
  --Fmax_travel 10000
  --enable_dwell_opt
  --Pmax_ms 780000
  --block_rows 50
  --block_duration_sec 12
  --max_target_samples 5000
  --change_weight 0.10
  --smooth_weight 0.05
  --target_weight 1.00
  --vmax_tolerance_s 0.01
  --iterations 8 
  --increment_scale_limit 0.20 
  --max_increment_scale_limit 0.40 
  --min_increment_scale_limit 0.01 
  --trust_shrink 0.50 
  --trust_grow 1.15 
  --patience 3 
  --true_validate 
  --vnozzle 574
 
Alpha:
python residence_optimizer_incremental_scale_lp_v1_2.py --input "LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_segmented.csv" --gcode_in "LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_segmented.gcode" --gcode_out "LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_sparse_optimized.gcode" --out_dir "LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_sparse_optimization" --target_mode alpha --temp_c 32.0 --alpha_mode bounds --alpha_min 0.879469 --alpha_ideal 0.925747 --alpha_max 0.972045 --Fmin_extrusion 50 --Fmax_extrusion 5000 --enable_travel_opt --enable_dwell_opt --block_rows 50 --block_duration_sec 12 --max_target_samples 5000 --change_weight 0.10 --smooth_weight 0.05 --target_weight 1.00 --vmax_tolerance_s 0.01 --true_validate --vnozzle 574  --Fmin_travel 50 --Fmax_travel 10000

Residence Time:
python residence_optimizer_incremental_scale_lp_v1_2.py --input "LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_segmented.csv" --gcode_in "LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_segmented.gcode" --gcode_out "LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_sparse_optimized.gcode" --out_dir "LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_sparse_optimization" --target_mode direct --tau_min 770 --tau_ideal 800 --tau_max 840 --Fmin_extrusion 50 --Fmax_extrusion 5000 --enable_travel_opt --enable_dwell_opt --block_rows 50 --block_duration_sec 12 --max_target_samples 5000 --change_weight 0.10 --smooth_weight 0.05 --target_weight 1.00 --vmax_tolerance_s 0.01 --true_validate --vnozzle 574  --Fmin_travel 50 --Fmax_travel 10000 --iterations 8 --increment_scale_limit 0.20 --max_increment_scale_limit 0.40 --min_increment_scale_limit 0.01 --trust_shrink 0.50 --trust_grow 1.15 --patience 5
"""

from __future__ import annotations

import argparse
import csv
import importlib
import gc
import json
import math
import os
import shutil
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

EPS = 1e-12
R_GAS = 8.314

ALPHA_MODEL_PARAMS = {
    "A1": 1.00e15,
    "E1": 103.97e3,
    "A2": 1.96e12,
    "E2": 218.55e3,
    "m": 0.713,
    "n": 0.323,
}

SEGIDX_RE = re.compile(r"\bseg_idx\s*=\s*(?P<id>\d+)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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


@dataclass
class ControlBlock:
    index: int
    kind: str
    row_indices: List[int] = field(default_factory=list)
    start_row: int = 0
    end_row: int = 0
    base_duration: float = 0.0
    duration_lb: float = 0.0
    duration_ub: float = 0.0
    base_f_min: float = math.nan
    base_f_max: float = math.nan
    allowed_f_min: float = math.nan
    allowed_f_max: float = math.nan


@dataclass
class TimelinePiece:
    start_t: float
    end_t: float
    kind: str  # "fixed" or "variable"
    fixed_before: float
    variable_prefix_count: int
    block_index: Optional[int] = None


@dataclass
class EventRepresentation:
    fixed_offset: float
    z_weights: Dict[int, float]


@dataclass
class ResidenceEquation:
    row_i: int
    seg_idx: int
    baseline_tau: float
    constant: float
    z_weights: Dict[int, float]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

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
        note = str(rec.get("note", "") if not pd.isna(rec.get("note", "")) else "").strip()
        note_l = note.lower()
        rows.append(
            Row(
                row_i=row_i,
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
    if row.d_mm <= EPS or row.cmd not in {"G0", "G1"}:
        return False
    return True


def infer_feedrate(row: Row) -> float:
    if row.F > EPS:
        return row.F
    if row.d_mm > EPS and row.t_s > EPS:
        return 60.0 * row.d_mm / row.t_s
    return 0.0


def classify_adjustable_row(
    row: Row,
    enable_travel_opt: bool,
    enable_dwell_opt: bool,
) -> Optional[str]:
    if row.t_s <= EPS:
        return None

    if row.cmd == "G4" and row.is_optimizable and enable_dwell_opt:
        return "dwell"

    if row.cmd == "G1" and is_control_target_row(row) and infer_feedrate(row) > EPS:
        return "extrusion"

    if enable_travel_opt and is_travel_row(row) and infer_feedrate(row) > EPS:
        return "travel"

    return None


# ---------------------------------------------------------------------------
# Alpha <-> tau model
# ---------------------------------------------------------------------------

def celsius_to_kelvin(temp_c: float) -> float:
    return float(temp_c) + 273.15


def arrhenius_rate(A: float, E: float, temp_k: float) -> float:
    return float(A) * math.exp(-float(E) / (R_GAS * float(temp_k)))


def alpha_rhs_isothermal(_t: float, y: np.ndarray, temp_c: float, params: dict):
    alpha = min(max(float(y[0]), 1e-12), 1.0)
    temp_k = celsius_to_kelvin(temp_c)
    k1 = arrhenius_rate(params["A1"], params["E1"], temp_k)
    k2 = arrhenius_rate(params["A2"], params["E2"], temp_k)
    dadt = (k1 + k2 * alpha ** float(params["m"])) * (1.0 - alpha) ** float(params["n"])
    return [dadt]


def build_alpha_tau_lookup(
    temp_c: float,
    tau_upper_s: float,
    n_eval: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    tau_upper_s = max(1.0, float(tau_upper_s))
    t_eval = np.linspace(0.0, tau_upper_s, max(1000, int(n_eval)))
    sol = solve_ivp(
        fun=lambda t, y: alpha_rhs_isothermal(t, y, temp_c, ALPHA_MODEL_PARAMS),
        t_span=(0.0, tau_upper_s),
        y0=[0.0],
        t_eval=t_eval,
        method="BDF",
        rtol=1e-7,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Alpha ODE solver failed: {sol.message}")
    alpha = np.maximum.accumulate(np.clip(sol.y[0], 0.0, 1.0))
    return sol.t, alpha


def tau_from_alpha(alpha_target: float, tau_grid: np.ndarray, alpha_grid: np.ndarray) -> float:
    alpha_target = float(alpha_target)
    if alpha_target <= alpha_grid[0]:
        return float(tau_grid[0])
    if alpha_target >= alpha_grid[-1]:
        raise ValueError(
            f"Target alpha={alpha_target:.8f} exceeds lookup maximum "
            f"alpha={alpha_grid[-1]:.8f}; increase --tau_lookup_max."
        )
    alpha_unique, idx = np.unique(alpha_grid, return_index=True)
    return float(np.interp(alpha_target, alpha_unique, tau_grid[idx]))


def alpha_from_tau(tau: np.ndarray, tau_grid: np.ndarray, alpha_grid: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(tau, dtype=float), tau_grid, alpha_grid, left=alpha_grid[0], right=alpha_grid[-1])


# ---------------------------------------------------------------------------
# Control block creation
# ---------------------------------------------------------------------------

def row_scale_bounds(
    row: Row,
    kind: str,
    Fmin_extrusion: float,
    Fmax_extrusion: float,
    Fmin_travel: float,
    Fmax_travel: float,
    Pmin_ms: float,
    Pmax_ms: float,
    max_relative_F_change: float,
    max_relative_P_change: float,
) -> Tuple[float, float]:
    if kind in {"extrusion", "travel"}:
        F0 = infer_feedrate(row)
        if F0 <= EPS:
            raise ValueError(f"seg_idx={row.seg_idx} has no valid feed rate")

        if kind == "extrusion":
            Fmin_kind = Fmin_extrusion
            Fmax_kind = Fmax_extrusion
        else:
            Fmin_kind = Fmin_travel
            Fmax_kind = Fmax_travel

        # t_new = t_base * scale and F_new = F0 / scale.
        lo = F0 / Fmax_kind
        hi = F0 / Fmin_kind
        if max_relative_F_change > 0.0:
            r = min(float(max_relative_F_change), 0.95)
            lo = max(lo, 1.0 / (1.0 + r))
            hi = min(hi, 1.0 / (1.0 - r))
        return lo, hi

    if kind == "dwell":
        P0 = row.t_s * 1000.0
        if P0 <= EPS:
            raise ValueError(f"seg_idx={row.seg_idx} has zero baseline dwell and cannot be scaled")
        lo = Pmin_ms / P0
        hi = Pmax_ms / P0
        if max_relative_P_change > 0.0:
            r = min(float(max_relative_P_change), 0.95)
            lo = max(lo, 1.0 - r)
            hi = min(hi, 1.0 + r)
        return max(0.0, lo), max(0.0, hi)

    raise ValueError(f"Unknown block kind: {kind}")


def build_control_blocks(
    rows: Sequence[Row],
    enable_travel_opt: bool,
    enable_dwell_opt: bool,
    block_rows: int,
    block_duration_sec: float,
    Fmin_extrusion: float,
    Fmax_extrusion: float,
    Fmin_travel: float,
    Fmax_travel: float,
    Pmin_ms: float,
    Pmax_ms: float,
    max_relative_F_change: float,
    max_relative_P_change: float,
) -> Tuple[List[ControlBlock], np.ndarray]:
    blocks: List[ControlBlock] = []
    row_to_block = np.full(len(rows), -1, dtype=int)
    current: Optional[ControlBlock] = None
    current_scale_lo = 0.0
    current_scale_hi = math.inf
    positive_fixed_gap = False

    def finalize() -> None:
        nonlocal current, current_scale_lo, current_scale_hi
        if current is None or not current.row_indices:
            current = None
            return
        current.start_row = min(current.row_indices)
        current.end_row = max(current.row_indices)
        current.base_duration = float(sum(rows[i].t_s for i in current.row_indices))
        if current.base_duration <= EPS:
            current = None
            return
        if current_scale_lo > current_scale_hi + 1e-12:
            raise ValueError(
                f"Block {current.index} has incompatible bounds: "
                f"scale [{current_scale_lo:.6g}, {current_scale_hi:.6g}]"
            )
        current.duration_lb = current.base_duration * current_scale_lo
        current.duration_ub = current.base_duration * current_scale_hi
        feedrates = [infer_feedrate(rows[i]) for i in current.row_indices if current.kind != "dwell"]
        if feedrates:
            current.base_f_min = min(feedrates)
            current.base_f_max = max(feedrates)
        blocks.append(current)
        for i in current.row_indices:
            row_to_block[i] = current.index
        current = None
        current_scale_lo = 0.0
        current_scale_hi = math.inf

    for i, row in enumerate(rows):
        kind = classify_adjustable_row(row, enable_travel_opt, enable_dwell_opt)

        if kind is None:
            if row.t_s > EPS:
                finalize()
                positive_fixed_gap = True
            continue

        scale_lo, scale_hi = row_scale_bounds(
            row=row,
            kind=kind,
            Fmin_extrusion=Fmin_extrusion,
            Fmax_extrusion=Fmax_extrusion,
            Fmin_travel=Fmin_travel,
            Fmax_travel=Fmax_travel,
            Pmin_ms=Pmin_ms,
            Pmax_ms=Pmax_ms,
            max_relative_F_change=max_relative_F_change,
            max_relative_P_change=max_relative_P_change,
        )

        must_split = (
            current is None
            or positive_fixed_gap
            or current.kind != kind
            or (kind == "dwell")
            or len(current.row_indices) >= max(1, block_rows)
            or (
                block_duration_sec > 0.0
                and current.base_duration + row.t_s > block_duration_sec
                and len(current.row_indices) > 0
            )
            or max(current_scale_lo, scale_lo) > min(current_scale_hi, scale_hi) + 1e-12
        )

        if must_split:
            finalize()
            current = ControlBlock(index=len(blocks), kind=kind)
            if kind == "extrusion":
                current.allowed_f_min = Fmin_extrusion
                current.allowed_f_max = Fmax_extrusion
            elif kind == "travel":
                current.allowed_f_min = Fmin_travel
                current.allowed_f_max = Fmax_travel
            current_scale_lo = scale_lo
            current_scale_hi = scale_hi
        else:
            current_scale_lo = max(current_scale_lo, scale_lo)
            current_scale_hi = min(current_scale_hi, scale_hi)

        current.row_indices.append(i)
        current.base_duration += row.t_s
        positive_fixed_gap = False

        if kind == "dwell":
            finalize()

    finalize()

    # Reindex defensively after finalization.
    for b, block in enumerate(blocks):
        block.index = b
        for i in block.row_indices:
            row_to_block[i] = b

    if not blocks:
        raise ValueError("No adjustable control blocks were found")

    return blocks, row_to_block


# ---------------------------------------------------------------------------
# Sparse event model
# ---------------------------------------------------------------------------

def build_timeline_pieces(rows: Sequence[Row], blocks: Sequence[ControlBlock]) -> Tuple[List[TimelinePiece], float, float]:
    durations = np.array([r.t_s for r in rows], dtype=float)
    starts = np.concatenate(([0.0], np.cumsum(durations)[:-1]))
    ends = np.cumsum(durations)
    total_time = float(ends[-1]) if len(ends) else 0.0

    pieces: List[TimelinePiece] = []
    current_t = 0.0
    fixed_before = 0.0

    for block in blocks:
        block_start = float(starts[block.start_row])
        block_end = float(ends[block.end_row])

        if block_start > current_t + EPS:
            fixed_duration = block_start - current_t
            pieces.append(
                TimelinePiece(
                    start_t=current_t,
                    end_t=block_start,
                    kind="fixed",
                    fixed_before=fixed_before,
                    variable_prefix_count=block.index,
                )
            )
            fixed_before += fixed_duration

        pieces.append(
            TimelinePiece(
                start_t=block_start,
                end_t=block_end,
                kind="variable",
                fixed_before=fixed_before,
                variable_prefix_count=block.index,
                block_index=block.index,
            )
        )
        current_t = block_end

    if current_t < total_time - EPS:
        pieces.append(
            TimelinePiece(
                start_t=current_t,
                end_t=total_time,
                kind="fixed",
                fixed_before=fixed_before,
                variable_prefix_count=len(blocks),
            )
        )
        fixed_before += total_time - current_t

    return pieces, total_time, fixed_before


def represent_event(
    event_time: float,
    pieces: Sequence[TimelinePiece],
    piece_ends: np.ndarray,
    total_time: float,
    total_fixed_time: float,
    block_count: int,
) -> EventRepresentation:
    t = float(event_time)
    if t <= 0.0:
        return EventRepresentation(fixed_offset=t, z_weights={0: 1.0})
    if t >= total_time:
        return EventRepresentation(
            fixed_offset=total_fixed_time + (t - total_time),
            z_weights={block_count: 1.0},
        )

    piece_i = int(np.searchsorted(piece_ends, t, side="left"))
    piece_i = min(max(piece_i, 0), len(pieces) - 1)
    piece = pieces[piece_i]

    if piece.kind == "fixed":
        return EventRepresentation(
            fixed_offset=piece.fixed_before + max(0.0, t - piece.start_t),
            z_weights={piece.variable_prefix_count: 1.0},
        )

    assert piece.block_index is not None
    duration = max(piece.end_t - piece.start_t, EPS)
    frac = min(1.0, max(0.0, (t - piece.start_t) / duration))
    weights: Dict[int, float] = {}
    if 1.0 - frac > EPS:
        weights[piece.block_index] = 1.0 - frac
    if frac > EPS:
        weights[piece.block_index + 1] = frac
    return EventRepresentation(fixed_offset=piece.fixed_before, z_weights=weights)


def combine_event_difference(end: EventRepresentation, start: EventRepresentation) -> Tuple[float, Dict[int, float]]:
    weights: Dict[int, float] = {}
    for idx, value in end.z_weights.items():
        weights[idx] = weights.get(idx, 0.0) + value
    for idx, value in start.z_weights.items():
        weights[idx] = weights.get(idx, 0.0) - value
    weights = {idx: val for idx, val in weights.items() if abs(val) > 1e-14}
    return end.fixed_offset - start.fixed_offset, weights


def baseline_z(blocks: Sequence[ControlBlock]) -> np.ndarray:
    y0 = np.array([b.base_duration for b in blocks], dtype=float)
    return np.concatenate(([0.0], np.cumsum(y0)))


def dot_weights(weights: Dict[int, float], z: np.ndarray) -> float:
    return float(sum(value * z[idx] for idx, value in weights.items()))


def build_residence_equations(
    rows: Sequence[Row],
    blocks: Sequence[ControlBlock],
) -> Tuple[List[ResidenceEquation], Dict[str, float]]:
    pieces, total_time, total_fixed = build_timeline_pieces(rows, blocks)
    if not pieces:
        raise ValueError("Timeline has no pieces")
    piece_ends = np.array([p.end_t for p in pieces], dtype=float)
    z0 = baseline_z(blocks)

    equations: List[ResidenceEquation] = []
    skipped = 0
    baseline_residuals: List[float] = []

    for row_i, row in enumerate(rows):
        if not is_scored_extrusion_row(row):
            continue
        if not (math.isfinite(row.t_in) and math.isfinite(row.t_out) and math.isfinite(row.tau_s)):
            skipped += 1
            continue
        if row.t_out + 1e-9 < row.t_in:
            skipped += 1
            continue

        start_repr = represent_event(row.t_in, pieces, piece_ends, total_time, total_fixed, len(blocks))
        end_repr = represent_event(row.t_out, pieces, piece_ends, total_time, total_fixed, len(blocks))
        fixed_part, weights = combine_event_difference(end_repr, start_repr)

        predicted_at_baseline = fixed_part + dot_weights(weights, z0)
        residual = row.tau_s - predicted_at_baseline
        baseline_residuals.append(residual)

        equations.append(
            ResidenceEquation(
                row_i=row_i,
                seg_idx=row.seg_idx,
                baseline_tau=row.tau_s,
                constant=fixed_part + residual,
                z_weights=weights,
            )
        )

    if not equations:
        raise ValueError("No valid scored extrusion rows with finite t_in/t_out/tau_s")

    diagnostics = {
        "equation_count": len(equations),
        "skipped_scored_rows": skipped,
        "baseline_residual_abs_max": float(max(abs(v) for v in baseline_residuals)) if baseline_residuals else 0.0,
        "baseline_residual_rmse": float(math.sqrt(np.mean(np.square(baseline_residuals)))) if baseline_residuals else 0.0,
        "timeline_total_s": total_time,
        "timeline_fixed_s": total_fixed,
    }
    return equations, diagnostics


def evaluate_equations(equations: Sequence[ResidenceEquation], z: np.ndarray) -> np.ndarray:
    out = np.empty(len(equations), dtype=float)
    for i, eq in enumerate(equations):
        out[i] = eq.constant + dot_weights(eq.z_weights, z)
    return out


# ---------------------------------------------------------------------------
# LP matrix construction
# ---------------------------------------------------------------------------

def add_sparse_row(
    data: List[float],
    row_idx: List[int],
    col_idx: List[int],
    r: int,
    entries: Iterable[Tuple[int, float]],
) -> None:
    for c, value in entries:
        if abs(value) <= 1e-15:
            continue
        row_idx.append(r)
        col_idx.append(c)
        data.append(float(value))


def block_scale_bounds(block: ControlBlock) -> Tuple[float, float]:
    """Return the explicit LP bounds for one block time scale."""
    base = max(block.base_duration, EPS)
    return block.duration_lb / base, block.duration_ub / base


def build_scale_link_equalities(
    blocks: Sequence[ControlBlock],
    nvar: int,
    scale_start: int,
) -> Tuple[csr_matrix, np.ndarray]:
    """Build z[b+1] - z[b] = base_duration[b] * scale[b].

    The z variables preserve the very sparse entry/exit event representation.
    The scale variables are the actual control decisions optimized by HiGHS.
    """
    data: List[float] = []
    rr: List[int] = []
    cc: List[int] = []
    rhs = np.zeros(len(blocks), dtype=float)
    for b, block in enumerate(blocks):
        add_sparse_row(
            data, rr, cc, b,
            [
                (b + 1, 1.0),
                (b, -1.0),
                (scale_start + b, -block.base_duration),
            ],
        )
    return coo_matrix((data, (rr, cc)), shape=(len(blocks), nvar)).tocsr(), rhs


def build_stage1_problem(
    equations: Sequence[ResidenceEquation],
    blocks: Sequence[ControlBlock],
    tau_lo: float,
    tau_hi: float,
) -> Tuple[
    np.ndarray,
    csr_matrix,
    np.ndarray,
    csr_matrix,
    np.ndarray,
    List[Tuple[Optional[float], Optional[float]]],
    int,
    int,
]:
    B = len(blocks)
    z_count = B + 1
    scale_start = z_count
    vmax_idx = scale_start + B
    nvar = vmax_idx + 1

    data: List[float] = []
    rr: List[int] = []
    cc: List[int] = []
    rhs: List[float] = []
    r = 0

    for eq in equations:
        # tau <= tau_hi + vmax
        entries = list(eq.z_weights.items()) + [(vmax_idx, -1.0)]
        add_sparse_row(data, rr, cc, r, entries)
        rhs.append(tau_hi - eq.constant)
        r += 1

        # tau >= tau_lo - vmax
        entries = [(idx, -value) for idx, value in eq.z_weights.items()] + [(vmax_idx, -1.0)]
        add_sparse_row(data, rr, cc, r, entries)
        rhs.append(eq.constant - tau_lo)
        r += 1

    A_ub = coo_matrix((data, (rr, cc)), shape=(r, nvar)).tocsr()
    b_ub = np.asarray(rhs, dtype=float)
    A_eq, b_eq = build_scale_link_equalities(blocks, nvar, scale_start)

    c = np.zeros(nvar, dtype=float)
    c[vmax_idx] = 1.0

    bounds: List[Tuple[Optional[float], Optional[float]]] = [(None, None)] * nvar
    bounds[0] = (0.0, 0.0)
    for b, block in enumerate(blocks):
        bounds[scale_start + b] = block_scale_bounds(block)
    bounds[vmax_idx] = (0.0, None)

    return c, A_ub, b_ub, A_eq, b_eq, bounds, vmax_idx, scale_start


def select_target_equations(
    equations: Sequence[ResidenceEquation],
    tau_lo: float,
    tau_hi: float,
    max_samples: int,
) -> List[int]:
    n = len(equations)
    if max_samples <= 0 or n <= max_samples:
        return list(range(n))

    baseline = np.array([eq.baseline_tau for eq in equations], dtype=float)
    violation = np.maximum(tau_lo - baseline, 0.0) + np.maximum(baseline - tau_hi, 0.0)

    # Keep the worst baseline violations, then fill remaining slots evenly.
    worst_count = min(max_samples // 2, int(np.count_nonzero(violation > 0.0)))
    selected: set[int] = set()
    if worst_count > 0:
        worst = np.argpartition(violation, -worst_count)[-worst_count:]
        selected.update(int(i) for i in worst)

    remaining = max_samples - len(selected)
    if remaining > 0:
        even = np.linspace(0, n - 1, remaining, dtype=int)
        selected.update(int(i) for i in even)

    return sorted(selected)[:max_samples]


def smooth_pairs(blocks: Sequence[ControlBlock]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for b in range(1, len(blocks)):
        prev = blocks[b - 1]
        curr = blocks[b]
        if prev.kind != curr.kind:
            continue
        if curr.start_row > prev.end_row + 1:
            continue
        pairs.append((b - 1, b))
    return pairs


def build_stage2_problem(
    equations: Sequence[ResidenceEquation],
    blocks: Sequence[ControlBlock],
    tau_lo: float,
    tau_star: float,
    tau_hi: float,
    vmax_upper: float,
    max_target_samples: int,
    change_weight: float,
    smooth_weight: float,
    target_weight: float,
    time_weight: float,
) -> Tuple[
    np.ndarray,
    csr_matrix,
    np.ndarray,
    csr_matrix,
    np.ndarray,
    List[Tuple[Optional[float], Optional[float]]],
    Dict[str, object],
]:
    B = len(blocks)
    z_count = B + 1
    scale_start = z_count
    vmax_idx = scale_start + B
    d_start = vmax_idx + 1
    d_count = B
    pairs = smooth_pairs(blocks)
    q_start = d_start + d_count
    q_count = len(pairs)
    targets = select_target_equations(equations, tau_lo, tau_hi, max_target_samples)
    e_start = q_start + q_count
    e_count = len(targets)
    nvar = e_start + e_count

    data: List[float] = []
    rr: List[int] = []
    cc: List[int] = []
    rhs: List[float] = []
    r = 0

    # Residence constraints use sparse cumulative event-time variables z.
    for eq in equations:
        add_sparse_row(
            data, rr, cc, r,
            list(eq.z_weights.items()) + [(vmax_idx, -1.0)],
        )
        rhs.append(tau_hi - eq.constant)
        r += 1

        add_sparse_row(
            data, rr, cc, r,
            [(idx, -value) for idx, value in eq.z_weights.items()] + [(vmax_idx, -1.0)],
        )
        rhs.append(eq.constant - tau_lo)
        r += 1

    # Absolute deviation of each explicit scale from the baseline scale 1.
    for b in range(B):
        s_idx = scale_start + b
        d_idx = d_start + b
        # scale - 1 <= d
        add_sparse_row(data, rr, cc, r, [(s_idx, 1.0), (d_idx, -1.0)])
        rhs.append(1.0)
        r += 1
        # 1 - scale <= d  -> -scale - d <= -1
        add_sparse_row(data, rr, cc, r, [(s_idx, -1.0), (d_idx, -1.0)])
        rhs.append(-1.0)
        r += 1

    # Absolute scale difference between neighboring compatible blocks.
    for q_i, (a, b) in enumerate(pairs):
        q_idx = q_start + q_i
        sa_idx = scale_start + a
        sb_idx = scale_start + b
        add_sparse_row(data, rr, cc, r, [(sb_idx, 1.0), (sa_idx, -1.0), (q_idx, -1.0)])
        rhs.append(0.0)
        r += 1
        add_sparse_row(data, rr, cc, r, [(sa_idx, 1.0), (sb_idx, -1.0), (q_idx, -1.0)])
        rhs.append(0.0)
        r += 1

    # Sampled absolute ideal-target error.
    for local_i, eq_i in enumerate(targets):
        eq = equations[eq_i]
        e_idx = e_start + local_i
        add_sparse_row(data, rr, cc, r, list(eq.z_weights.items()) + [(e_idx, -1.0)])
        rhs.append(tau_star - eq.constant)
        r += 1
        add_sparse_row(
            data, rr, cc, r,
            [(idx, -value) for idx, value in eq.z_weights.items()] + [(e_idx, -1.0)],
        )
        rhs.append(eq.constant - tau_star)
        r += 1

    A_ub = coo_matrix((data, (rr, cc)), shape=(r, nvar)).tocsr()
    b_ub = np.asarray(rhs, dtype=float)
    A_eq, b_eq = build_scale_link_equalities(blocks, nvar, scale_start)
    c = np.zeros(nvar, dtype=float)

    if B > 0 and change_weight > 0.0:
        c[d_start:d_start + B] = change_weight / B

    if q_count > 0 and smooth_weight > 0.0:
        c[q_start:q_start + q_count] = smooth_weight / q_count

    tau_scale = max(1.0, 0.5 * (tau_hi - tau_lo))
    if e_count > 0 and target_weight > 0.0:
        c[e_start:e_start + e_count] = target_weight / (e_count * tau_scale)

    if time_weight > 0.0:
        total_variable = max(sum(b.base_duration for b in blocks), EPS)
        for b, block in enumerate(blocks):
            c[scale_start + b] += time_weight * block.base_duration / total_variable

    # Tiny objective on vmax improves numerical preference when tolerance is nonzero.
    c[vmax_idx] = 1e-9

    bounds: List[Tuple[Optional[float], Optional[float]]] = [(None, None)] * nvar
    bounds[0] = (0.0, 0.0)
    for b, block in enumerate(blocks):
        bounds[scale_start + b] = block_scale_bounds(block)
    bounds[vmax_idx] = (0.0, max(0.0, vmax_upper))
    for idx in range(d_start, nvar):
        bounds[idx] = (0.0, None)

    metadata: Dict[str, object] = {
        "scale_start": scale_start,
        "vmax_idx": vmax_idx,
        "d_start": d_start,
        "q_start": q_start,
        "e_start": e_start,
        "target_indices": targets,
        "smooth_pairs": pairs,
        "variable_count": nvar,
        "constraint_count": r,
        "equality_count": B,
    }
    return c, A_ub, b_ub, A_eq, b_eq, bounds, metadata


def solve_lp(
    c: np.ndarray,
    A_ub: csr_matrix,
    b_ub: np.ndarray,
    A_eq: Optional[csr_matrix],
    b_eq: Optional[np.ndarray],
    bounds: Sequence[Tuple[Optional[float], Optional[float]]],
    time_limit: float,
    label: str,
):
    options = {"presolve": True}
    if time_limit > 0.0:
        options["time_limit"] = float(time_limit)
    started = time.perf_counter()
    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=list(bounds),
        method="highs",
        options=options,
    )
    elapsed = time.perf_counter() - started
    if not result.success:
        raise RuntimeError(
            f"{label} failed: status={result.status}, message={result.message}, "
            f"elapsed={elapsed:.3f}s"
        )
    return result, elapsed


# ---------------------------------------------------------------------------
# Convert solution back to row timing / F / P
# ---------------------------------------------------------------------------

def solution_to_rows(
    rows: Sequence[Row],
    blocks: Sequence[ControlBlock],
    scales: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the LP-optimized block scales directly to row timing and F/P."""
    if len(scales) != len(blocks):
        raise ValueError("Scale vector length does not match control block count")

    F_new = np.array([r.F for r in rows], dtype=float)
    P_new_ms = np.array([r.t_s * 1000.0 if r.cmd == "G4" else 0.0 for r in rows], dtype=float)
    t_new = np.array([r.t_s for r in rows], dtype=float)
    row_scale = np.ones(len(rows), dtype=float)

    for block in blocks:
        scale = float(scales[block.index])
        if not math.isfinite(scale) or scale <= EPS:
            raise ValueError(f"Block {block.index} has invalid optimized scale={scale}")
        for i in block.row_indices:
            row = rows[i]
            row_scale[i] = scale
            t_new[i] = row.t_s * scale
            if block.kind == "dwell":
                P_new_ms[i] = t_new[i] * 1000.0
            else:
                F0 = infer_feedrate(row)
                if F0 > EPS:
                    F_new[i] = F0 / scale

    return F_new, P_new_ms, t_new, row_scale


# ---------------------------------------------------------------------------
# G-code patching and optional TRUE validation
# ---------------------------------------------------------------------------

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
    F_new: np.ndarray,
    P_new_ms: np.ndarray,
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
                parts = replace_or_append_parameter(parts, "F", f"{float(F_new[i]):.2f}")
            elif mode == "P":
                parts = replace_or_append_parameter(parts, "P", str(int(round(float(P_new_ms[i])))))
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
    V_nozzle: float,
    d_filament: float,
    e_mode: str,
) -> Tuple[str, str]:
    module = importlib.import_module(module_name)
    rebuild_global = getattr(module, "rebuild_global")
    csv_out = os.path.join(out_dir, "true_validation.csv")
    gcode_out = os.path.join(out_dir, "true_validation_rebuilt.gcode")
    rebuild_global(
        gcode_in=gcode_in,
        csv_out=csv_out,
        gcode_out=gcode_out,
        V_nozzle=V_nozzle,
        d_filament=d_filament,
        e_mode=e_mode,
    )
    if not os.path.exists(csv_out):
        raise RuntimeError("TRUE FIFO builder did not create validation CSV")
    return csv_out, gcode_out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def residence_metrics(tau: np.ndarray, tau_lo: float, tau_star: float, tau_hi: float) -> Dict[str, float]:
    low = np.maximum(tau_lo - tau, 0.0)
    high = np.maximum(tau - tau_hi, 0.0)
    violation = low + high
    return {
        "count": int(tau.size),
        "tau_min": float(np.min(tau)),
        "tau_max": float(np.max(tau)),
        "tau_mean": float(np.mean(tau)),
        "within_percent": float(100.0 * np.mean(violation <= 1e-7)),
        "violation_count": int(np.count_nonzero(violation > 1e-7)),
        "violation_rate": float(np.mean(violation > 1e-7)),
        "vmax_s": float(np.max(violation)),
        "violation_sum_s": float(np.sum(violation)),
        "mean_abs_target_error_s": float(np.mean(np.abs(tau - tau_star))),
    }


def write_outputs(
    out_dir: str,
    source_df: pd.DataFrame,
    rows: Sequence[Row],
    blocks: Sequence[ControlBlock],
    row_to_block: np.ndarray,
    equations: Sequence[ResidenceEquation],
    tau_pred: np.ndarray,
    alpha_pred: Optional[np.ndarray],
    F_new: np.ndarray,
    P_new_ms: np.ndarray,
    t_new: np.ndarray,
    row_scale: np.ndarray,
    scales: np.ndarray,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    output_df = source_df.copy()
    output_df["control_block"] = row_to_block
    output_df["time_scale"] = row_scale
    output_df["t_optimized_s"] = t_new
    output_df["F_optimized_mm_per_min"] = F_new
    output_df["P_optimized_ms"] = P_new_ms
    output_df["tau_predicted_s"] = np.nan
    if alpha_pred is not None:
        output_df["alpha_predicted"] = np.nan

    for eq_i, eq in enumerate(equations):
        output_df.at[eq.row_i, "tau_predicted_s"] = tau_pred[eq_i]
        if alpha_pred is not None:
            output_df.at[eq.row_i, "alpha_predicted"] = alpha_pred[eq_i]

    prediction_path = os.path.join(out_dir, "optimized_prediction.csv")
    output_df.to_csv(prediction_path, index=False, float_format="%.8f")

    block_records = []
    for block in blocks:
        scale = float(scales[block.index])
        y_new = block.base_duration * scale
        block_records.append(
            {
                "block": block.index,
                "kind": block.kind,
                "start_seg": rows[block.start_row].seg_idx,
                "end_seg": rows[block.end_row].seg_idx,
                "row_count": len(block.row_indices),
                "base_duration_s": block.base_duration,
                "optimized_duration_s": y_new,
                "time_scale": scale,
                "scale_lb": block.duration_lb / max(block.base_duration, EPS),
                "scale_ub": block.duration_ub / max(block.base_duration, EPS),
                "duration_lb_s": block.duration_lb,
                "duration_ub_s": block.duration_ub,
                "base_f_min": block.base_f_min,
                "base_f_max": block.base_f_max,
                "allowed_Fmin_mm_per_min": block.allowed_f_min,
                "allowed_Fmax_mm_per_min": block.allowed_f_max,
            }
        )
    block_path = os.path.join(out_dir, "control_blocks.csv")
    pd.DataFrame(block_records).to_csv(block_path, index=False, float_format="%.8f")
    return prediction_path, block_path


def compare_true_validation(
    validation_csv: str,
    equations: Sequence[ResidenceEquation],
    tau_pred: np.ndarray,
) -> Dict[str, float]:
    df = pd.read_csv(validation_csv)
    if "seg_idx" not in df.columns or "tau_s" not in df.columns:
        raise ValueError("TRUE validation CSV lacks seg_idx or tau_s")
    true_map = {
        int(rec["seg_idx"]): float(rec["tau_s"])
        for _, rec in df.iterrows()
        if not pd.isna(rec["seg_idx"]) and not pd.isna(rec["tau_s"])
    }
    pred = []
    truth = []
    for i, eq in enumerate(equations):
        if eq.seg_idx in true_map and math.isfinite(true_map[eq.seg_idx]):
            pred.append(float(tau_pred[i]))
            truth.append(float(true_map[eq.seg_idx]))
    if not pred:
        raise ValueError("No matching scored seg_idx values in TRUE validation CSV")
    err = np.asarray(truth) - np.asarray(pred)
    return {
        "matched_rows": len(pred),
        "true_minus_pred_mean_s": float(np.mean(err)),
        "true_minus_pred_rmse_s": float(math.sqrt(np.mean(err ** 2))),
        "true_minus_pred_abs_max_s": float(np.max(np.abs(err))),
    }


def scored_tau_from_csv(path: str) -> np.ndarray:
    """Read the authoritative TRUE FIFO tau values used for acceptance."""
    rows, _ = read_segments_csv(path)
    tau = np.asarray(
        [r.tau_s for r in rows if is_scored_extrusion_row(r) and math.isfinite(r.tau_s)],
        dtype=float,
    )
    if tau.size == 0:
        raise ValueError(f"No finite scored residence times were found in {path}")
    return tau


def apply_increment_scale_window(
    blocks: Sequence[ControlBlock],
    scale_min: float,
    scale_max: float,
) -> Dict[str, float]:
    """Intersect absolute machine bounds with the current local scale window.

    The LP scale is incremental: scale=1 keeps the current accepted timing,
    scale>1 slows the current block, and scale<1 speeds it up.
    """
    if not (0.0 < scale_min <= 1.0 <= scale_max):
        raise ValueError("Increment scale window must satisfy 0 < min <= 1 <= max")

    active_lows: List[float] = []
    active_highs: List[float] = []
    clipped = 0
    for block in blocks:
        absolute_lo, absolute_hi = block_scale_bounds(block)
        local_lo = max(absolute_lo, scale_min)
        local_hi = min(absolute_hi, scale_max)
        if local_lo > local_hi + 1e-12:
            raise ValueError(
                f"Block {block.index} has no feasible incremental scale: "
                f"absolute=[{absolute_lo:.8g}, {absolute_hi:.8g}], "
                f"local=[{scale_min:.8g}, {scale_max:.8g}]"
            )
        if local_lo > absolute_lo + 1e-12 or local_hi < absolute_hi - 1e-12:
            clipped += 1
        block.duration_lb = block.base_duration * local_lo
        block.duration_ub = block.base_duration * local_hi
        active_lows.append(local_lo)
        active_highs.append(local_hi)

    return {
        "requested_min": float(scale_min),
        "requested_max": float(scale_max),
        "active_min": float(min(active_lows)),
        "active_max": float(max(active_highs)),
        "clipped_block_count": int(clipped),
    }


def compare_metric_priority(
    candidate: Dict[str, float],
    current: Dict[str, float],
    tolerance_s: float,
) -> Tuple[bool, str]:
    """Accept TRUE results lexicographically.

    Priority follows the two-stage optimizer:
    1. lower worst violation;
    2. with effectively equal worst violation, fewer violating rows;
    3. then lower total violation;
    4. then lower mean absolute target error.
    """
    tol = max(0.0, float(tolerance_s))
    cand_v = float(candidate["vmax_s"])
    curr_v = float(current["vmax_s"])
    if cand_v < curr_v - tol:
        return True, "lower_true_vmax"
    if cand_v > curr_v + tol:
        return False, "higher_true_vmax"

    cand_count = int(candidate["violation_count"])
    curr_count = int(current["violation_count"])
    if cand_count < curr_count:
        return True, "fewer_true_violations"
    if cand_count > curr_count:
        return False, "more_true_violations"

    cand_sum = float(candidate["violation_sum_s"])
    curr_sum = float(current["violation_sum_s"])
    if cand_sum < curr_sum - tol:
        return True, "lower_true_violation_sum"
    if cand_sum > curr_sum + tol:
        return False, "higher_true_violation_sum"

    cand_target = float(candidate["mean_abs_target_error_s"])
    curr_target = float(current["mean_abs_target_error_s"])
    if cand_target < curr_target - tol:
        return True, "lower_true_target_error"
    return False, "no_material_true_improvement"


def cumulative_scale_table(original_csv: str, final_csv: str) -> pd.DataFrame:
    """Compute authoritative cumulative row time scale from TRUE CSV timing."""
    original_rows, _ = read_segments_csv(original_csv)
    final_rows, _ = read_segments_csv(final_csv)
    original_map = {r.seg_idx: r for r in original_rows}
    records: List[Dict[str, object]] = []
    for row in final_rows:
        base = original_map.get(row.seg_idx)
        if base is None:
            continue
        if base.t_s > EPS:
            cumulative = row.t_s / base.t_s
        elif row.t_s <= EPS:
            cumulative = 1.0
        else:
            cumulative = math.nan
        records.append(
            {
                "seg_idx": row.seg_idx,
                "cmd": row.cmd,
                "note": row.note,
                "original_t_s": base.t_s,
                "final_true_t_s": row.t_s,
                "cumulative_time_scale": cumulative,
                "original_F_mm_per_min": base.F,
                "final_F_mm_per_min": row.F,
            }
        )
    return pd.DataFrame(records)


def optimize_one_scale_round(
    *,
    args: argparse.Namespace,
    input_csv: str,
    gcode_in: Optional[str],
    round_dir: str,
    tau_lo: float,
    tau_star: float,
    tau_hi: float,
    tau_grid: Optional[np.ndarray],
    alpha_grid: Optional[np.ndarray],
    increment_scale_min: Optional[float],
    increment_scale_max: Optional[float],
    run_true: bool,
) -> Dict[str, object]:
    """Solve one explicit-scale LP around the current authoritative baseline."""
    os.makedirs(round_dir, exist_ok=True)
    rows, source_df = read_segments_csv(input_csv)
    blocks, row_to_block = build_control_blocks(
        rows=rows,
        enable_travel_opt=args.enable_travel_opt,
        enable_dwell_opt=args.enable_dwell_opt,
        block_rows=args.block_rows,
        block_duration_sec=args.block_duration_sec,
        Fmin_extrusion=args.Fmin_extrusion_resolved,
        Fmax_extrusion=args.Fmax_extrusion_resolved,
        Fmin_travel=args.Fmin_travel_resolved,
        Fmax_travel=args.Fmax_travel_resolved,
        Pmin_ms=args.Pmin_ms,
        Pmax_ms=args.Pmax_ms,
        max_relative_F_change=args.max_relative_F_change,
        max_relative_P_change=args.max_relative_P_change,
    )

    increment_window = None
    if increment_scale_min is not None and increment_scale_max is not None:
        increment_window = apply_increment_scale_window(
            blocks, increment_scale_min, increment_scale_max
        )

    equations, model_diag = build_residence_equations(rows, blocks)
    z0 = baseline_z(blocks)
    baseline_tau_model = evaluate_equations(equations, z0)
    baseline_tau_csv = np.asarray([eq.baseline_tau for eq in equations], dtype=float)
    reconstruction_abs_max = float(np.max(np.abs(baseline_tau_model - baseline_tau_csv)))
    baseline_metrics = residence_metrics(baseline_tau_csv, tau_lo, tau_star, tau_hi)

    print(f"  Loaded {len(rows):,} rows; {len(blocks):,} blocks; {len(equations):,} equations")
    if increment_window is not None:
        print(
            "  Increment scale window: "
            f"[{increment_window['requested_min']:.6g}, {increment_window['requested_max']:.6g}]"
        )
    print(
        f"  Current TRUE baseline: within={baseline_metrics['within_percent']:.3f}% "
        f"violations={baseline_metrics['violation_count']:,} "
        f"vmax={baseline_metrics['vmax_s']:.6f}s"
    )

    c1, A1, b1, Aeq1, beq1, bounds1, vmax_idx1, scale_start1 = build_stage1_problem(
        equations, blocks, tau_lo, tau_hi
    )
    print(
        f"  Stage 1 LP: variables={A1.shape[1]:,}, constraints={A1.shape[0]:,}, "
        f"equalities={Aeq1.shape[0]:,}, nonzeros={A1.nnz + Aeq1.nnz:,}"
    )
    stage1, stage1_sec = solve_lp(
        c1, A1, b1, Aeq1, beq1, bounds1, args.solver_time_limit, "Stage 1 minimax"
    )
    vmax_star = max(0.0, float(stage1.x[vmax_idx1]))
    B = len(blocks)
    stage1_z = np.asarray(stage1.x[:B + 1], dtype=float)
    stage1_scales = np.asarray(stage1.x[scale_start1:scale_start1 + B], dtype=float)
    np.savez_compressed(
        os.path.join(round_dir, "stage1_checkpoint.npz"),
        z=stage1_z,
        scales=stage1_scales,
        minimal_worst_violation_s=vmax_star,
    )
    print(f"  Stage 1 complete in {stage1_sec:.3f}s; minimal worst violation={vmax_star:.6f}s")

    # Release the first LP matrices before constructing Stage 2.
    del c1, A1, b1, Aeq1, beq1, bounds1
    gc.collect()

    solution_source = "stage2"
    stage2_sec = 0.0
    stage2_message = None
    target_sample_count = 0
    vmax_upper = vmax_star + max(0.0, args.vmax_tolerance_s)
    try:
        c2, A2, b2, Aeq2, beq2, bounds2, meta2 = build_stage2_problem(
            equations=equations,
            blocks=blocks,
            tau_lo=tau_lo,
            tau_star=tau_star,
            tau_hi=tau_hi,
            vmax_upper=vmax_upper,
            max_target_samples=args.max_target_samples,
            change_weight=args.change_weight,
            smooth_weight=args.smooth_weight,
            target_weight=args.target_weight,
            time_weight=args.time_weight,
        )
        target_sample_count = len(meta2["target_indices"])
        print(
            f"  Stage 2 LP: variables={A2.shape[1]:,}, constraints={A2.shape[0]:,}, "
            f"equalities={Aeq2.shape[0]:,}, nonzeros={A2.nnz + Aeq2.nnz:,}, "
            f"target_samples={target_sample_count:,}"
        )
        stage2, stage2_sec = solve_lp(
            c2, A2, b2, Aeq2, beq2, bounds2, args.solver_time_limit, "Stage 2 quality"
        )
        z_opt = np.asarray(stage2.x[:B + 1], dtype=float)
        scale_start2 = int(meta2["scale_start"])
        scales_opt = np.asarray(stage2.x[scale_start2:scale_start2 + B], dtype=float)
        stage2_message = str(stage2.message)
        del c2, A2, b2, Aeq2, beq2, bounds2
        gc.collect()
    except Exception as exc:
        if args.no_stage2_fallback:
            raise
        print(f"  Warning: Stage 2 failed; using saved Stage 1 scale solution: {exc}")
        solution_source = "stage1_fallback"
        z_opt = stage1_z
        scales_opt = stage1_scales
        stage2_message = f"fallback_after_error: {exc}"

    tau_pred = evaluate_equations(equations, z_opt)
    predicted_metrics = residence_metrics(tau_pred, tau_lo, tau_star, tau_hi)
    print(
        f"  Predicted candidate: within={predicted_metrics['within_percent']:.3f}% "
        f"violations={predicted_metrics['violation_count']:,} "
        f"vmax={predicted_metrics['vmax_s']:.6f}s"
    )

    F_new, P_new_ms, t_new, row_scale = solution_to_rows(rows, blocks, scales_opt)
    alpha_pred = alpha_from_tau(tau_pred, tau_grid, alpha_grid) if tau_grid is not None else None
    prediction_path, block_path = write_outputs(
        out_dir=round_dir,
        source_df=source_df,
        rows=rows,
        blocks=blocks,
        row_to_block=row_to_block,
        equations=equations,
        tau_pred=tau_pred,
        alpha_pred=alpha_pred,
        F_new=F_new,
        P_new_ms=P_new_ms,
        t_new=t_new,
        row_scale=row_scale,
        scales=scales_opt,
    )

    candidate_gcode = None
    patch_info = None
    if gcode_in:
        candidate_gcode = os.path.join(round_dir, "candidate_optimized.gcode")
        patched, expected = patch_gcode_by_segidx(
            gcode_in, candidate_gcode, rows, F_new, P_new_ms,
            args.enable_travel_opt, args.enable_dwell_opt,
        )
        patch_info = {
            "patched": int(patched),
            "expected": int(expected),
            "coverage": float(patched / max(1, expected)),
        }
        print(f"  Patched G-code: {patched}/{expected} ({patch_info['coverage']:.6f})")
        if expected > 0 and patch_info["coverage"] < 0.98:
            raise RuntimeError("G-code patch coverage is below 98%")

    true_csv = None
    true_gcode = None
    true_metrics = None
    true_comparison = None
    if run_true:
        if candidate_gcode is None:
            raise ValueError("TRUE validation requires a current G-code input")
        true_csv, true_gcode = run_true_validation(
            module_name=args.true_builder_module,
            gcode_in=candidate_gcode,
            out_dir=round_dir,
            V_nozzle=args.vnozzle,
            d_filament=args.d_filament,
            e_mode=args.e_mode,
        )
        true_comparison = compare_true_validation(true_csv, equations, tau_pred)
        true_metrics = residence_metrics(
            scored_tau_from_csv(true_csv), tau_lo, tau_star, tau_hi
        )
        print(
            f"  TRUE candidate: within={true_metrics['within_percent']:.3f}% "
            f"violations={true_metrics['violation_count']:,} "
            f"vmax={true_metrics['vmax_s']:.6f}s; "
            f"prediction RMSE={true_comparison['true_minus_pred_rmse_s']:.6f}s"
        )

    round_summary = {
        "input_csv": os.path.abspath(input_csv),
        "gcode_in": os.path.abspath(gcode_in) if gcode_in else None,
        "row_count": len(rows),
        "control_block_count": len(blocks),
        "model": model_diag,
        "baseline_reconstruction_abs_max_s": reconstruction_abs_max,
        "increment_window": increment_window,
        "baseline_metrics": baseline_metrics,
        "stage1": {
            "elapsed_s": stage1_sec,
            "minimal_worst_violation_s": vmax_star,
            "solver_message": str(stage1.message),
        },
        "stage2": {
            "elapsed_s": stage2_sec,
            "allowed_worst_violation_s": vmax_upper,
            "solver_message": stage2_message,
            "target_sample_count": target_sample_count,
            "solution_source": solution_source,
        },
        "scale_statistics": {
            "min": float(np.min(scales_opt)),
            "max": float(np.max(scales_opt)),
            "mean": float(np.mean(scales_opt)),
            "median": float(np.median(scales_opt)),
        },
        "predicted_metrics": predicted_metrics,
        "patch": patch_info,
        "true_metrics": true_metrics,
        "true_validation": true_comparison,
        "files": {
            "prediction_csv": os.path.abspath(prediction_path),
            "block_csv": os.path.abspath(block_path),
            "candidate_gcode": os.path.abspath(candidate_gcode) if candidate_gcode else None,
            "true_csv": os.path.abspath(true_csv) if true_csv else None,
            "true_gcode": os.path.abspath(true_gcode) if true_gcode else None,
        },
    }
    with open(os.path.join(round_dir, "round_summary.json"), "w", encoding="utf-8") as f:
        json.dump(round_summary, f, ensure_ascii=False, indent=2)

    return {
        "rows": rows,
        "row_to_block": row_to_block,
        "blocks": blocks,
        "equations": equations,
        "scales": scales_opt,
        "baseline_metrics": baseline_metrics,
        "predicted_metrics": predicted_metrics,
        "true_metrics": true_metrics,
        "true_comparison": true_comparison,
        "candidate_gcode": candidate_gcode,
        "true_csv": true_csv,
        "true_gcode": true_gcode,
        "round_summary": round_summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Sparse global FIFO residence-time optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Segmented CSV containing baseline TRUE FIFO timing")
    ap.add_argument("--gcode_in", default=None, help="Segmented G-code with seg_idx comments")
    ap.add_argument("--gcode_out", default=None, help="Patched optimized G-code output")
    ap.add_argument("--out_dir", default="sparse_optimization")

    ap.add_argument("--target_mode", choices=["alpha", "direct"], default="alpha")
    ap.add_argument("--temp_c", type=float, default=None)
    ap.add_argument("--alpha_mode", choices=["bounds", "buffer"], default="bounds")
    ap.add_argument("--alpha_min", type=float, default=None)
    ap.add_argument("--alpha_ideal", type=float, default=None)
    ap.add_argument("--alpha_max", type=float, default=None)
    ap.add_argument("--alpha_buffer", type=float, default=None)
    ap.add_argument("--tau_lookup_max", type=float, default=20000.0)
    ap.add_argument("--tau_min", type=float, default=None)
    ap.add_argument("--tau_ideal", type=float, default=None)
    ap.add_argument("--tau_max", type=float, default=None)

    # Independent feed-rate limits for extrusion and travel.
    ap.add_argument("--Fmin_extrusion", type=float, default=None,
                    help="Minimum feed rate for extrusion/control-target G1 rows")
    ap.add_argument("--Fmax_extrusion", type=float, default=None,
                    help="Maximum feed rate for extrusion/control-target G1 rows")
    ap.add_argument("--Fmin_travel", type=float, default=None,
                    help="Minimum feed rate for travel G0/G1 rows")
    ap.add_argument("--Fmax_travel", type=float, default=None,
                    help="Maximum feed rate for travel G0/G1 rows")
    # Backward-compatible fallbacks: applied to both kinds when the
    # corresponding new argument is omitted.
    ap.add_argument("--Fmin", type=float, default=None,
                    help="Legacy fallback minimum for extrusion and travel")
    ap.add_argument("--Fmax", type=float, default=None,
                    help="Legacy fallback maximum for extrusion and travel")
    ap.add_argument("--Pmin_ms", type=float, default=0.0)
    ap.add_argument("--Pmax_ms", type=float, default=600000.0)
    ap.add_argument("--max_relative_F_change", type=float, default=0.0,
                    help="Optional per-row limit relative to original F; 0 disables")
    ap.add_argument("--max_relative_P_change", type=float, default=0.0,
                    help="Optional dwell-duration limit relative to original P; 0 disables")
    ap.add_argument("--enable_travel_opt", action="store_true")
    ap.add_argument("--enable_dwell_opt", action="store_true")

    ap.add_argument("--block_rows", type=int, default=150,
                    help="Maximum adjustable rows per control block")
    ap.add_argument("--block_duration_sec", type=float, default=12.0,
                    help="Maximum baseline duration per block; 0 disables")
    ap.add_argument("--max_target_samples", type=int, default=5000,
                    help="Maximum residence rows used in ideal-target objective; constraints still use all rows")

    ap.add_argument("--change_weight", type=float, default=0.10)
    ap.add_argument("--smooth_weight", type=float, default=0.05)
    ap.add_argument("--target_weight", type=float, default=1.00)
    ap.add_argument("--time_weight", type=float, default=0.00)
    ap.add_argument("--vmax_tolerance_s", type=float, default=0.01,
                    help="Stage-2 allowance above the mathematically minimal worst violation")
    ap.add_argument("--solver_time_limit", type=float, default=0.0,
                    help="Seconds per LP; 0 means no explicit limit")

    # Incremental scale learning / sequential LP controls.
    ap.add_argument("--iterations", type=int, default=1,
                    help="Number of LP/TRUE-FIFO rounds; values >1 enable incremental-scale learning")
    ap.add_argument("--increment_scale_limit", type=float, default=0.20,
                    help="Initial local time-scale radius: scale is limited to [1-limit, 1+limit]")
    ap.add_argument("--max_increment_scale_limit", type=float, default=0.50,
                    help="Maximum local scale radius after successful rounds")
    ap.add_argument("--min_increment_scale_limit", type=float, default=0.01,
                    help="Stop when rejected rounds shrink the local scale radius below this value")
    ap.add_argument("--trust_shrink", type=float, default=0.50,
                    help="Multiplier applied to the scale radius after a rejected TRUE result")
    ap.add_argument("--trust_grow", type=float, default=1.15,
                    help="Multiplier applied to the scale radius after an accepted TRUE result")
    ap.add_argument("--accept_tolerance_s", type=float, default=0.001,
                    help="Tolerance for comparing TRUE FIFO metrics between rounds")
    ap.add_argument("--patience", type=int, default=3,
                    help="Stop after this many consecutive rejected TRUE candidates")
    ap.add_argument("--no_stage2_fallback", action="store_true",
                    help="Fail instead of falling back to the saved Stage-1 scale solution")

    ap.add_argument("--true_validate", action="store_true")
    ap.add_argument("--true_builder_module", default="rebuild_csv_and_gcode_fifo_v1_2_precondition_autofill")
    ap.add_argument("--vnozzle", type=float, default=574.0)
    ap.add_argument("--d_filament", type=float, default=15.55634919)
    ap.add_argument("--e_mode", choices=["filament", "mm3"], default="filament")
    return ap.parse_args()


def resolve_tau_targets(args: argparse.Namespace):
    if args.target_mode == "direct":
        values = (args.tau_min, args.tau_ideal, args.tau_max)
        if any(v is None or not math.isfinite(v) for v in values):
            raise ValueError("direct mode requires finite --tau_min --tau_ideal --tau_max")
        tau_lo, tau_star, tau_hi = map(float, values)
        if not (0.0 <= tau_lo < tau_star < tau_hi):
            raise ValueError("Require 0 <= tau_min < tau_ideal < tau_max")
        return tau_lo, tau_star, tau_hi, None, None

    if args.temp_c is None or args.alpha_ideal is None:
        raise ValueError("alpha mode requires --temp_c and --alpha_ideal")

    alpha_ideal = float(args.alpha_ideal)
    if args.alpha_mode == "buffer":
        if args.alpha_buffer is None or not (0.0 < args.alpha_buffer < 1.0):
            raise ValueError("alpha buffer mode requires 0 < --alpha_buffer < 1")
        if args.alpha_min is not None or args.alpha_max is not None:
            raise ValueError("Do not provide alpha_min/alpha_max in buffer mode")
        alpha_min = alpha_ideal * (1.0 - args.alpha_buffer)
        alpha_max = alpha_ideal * (1.0 + args.alpha_buffer)
    else:
        if args.alpha_min is None or args.alpha_max is None:
            raise ValueError("alpha bounds mode requires --alpha_min and --alpha_max")
        alpha_min = float(args.alpha_min)
        alpha_max = float(args.alpha_max)

    if not (0.0 <= alpha_min < alpha_ideal < alpha_max <= 1.0):
        raise ValueError("Require 0 <= alpha_min < alpha_ideal < alpha_max <= 1")

    tau_grid, alpha_grid = build_alpha_tau_lookup(args.temp_c, args.tau_lookup_max)
    tau_lo = tau_from_alpha(alpha_min, tau_grid, alpha_grid)
    tau_star = tau_from_alpha(alpha_ideal, tau_grid, alpha_grid)
    tau_hi = tau_from_alpha(alpha_max, tau_grid, alpha_grid)
    return tau_lo, tau_star, tau_hi, tau_grid, alpha_grid


def resolve_feedrate_bounds(args: argparse.Namespace) -> Tuple[float, float, float, float]:
    """Resolve independent extrusion/travel bounds with legacy fallback."""
    legacy_min = args.Fmin
    legacy_max = args.Fmax

    fmin_extrusion = (
        float(args.Fmin_extrusion)
        if args.Fmin_extrusion is not None
        else float(legacy_min) if legacy_min is not None else 180.0
    )
    fmax_extrusion = (
        float(args.Fmax_extrusion)
        if args.Fmax_extrusion is not None
        else float(legacy_max) if legacy_max is not None else 3000.0
    )
    fmin_travel = (
        float(args.Fmin_travel)
        if args.Fmin_travel is not None
        else float(legacy_min) if legacy_min is not None else fmin_extrusion
    )
    fmax_travel = (
        float(args.Fmax_travel)
        if args.Fmax_travel is not None
        else float(legacy_max) if legacy_max is not None else fmax_extrusion
    )
    return fmin_extrusion, fmax_extrusion, fmin_travel, fmax_travel


def validate_args(args: argparse.Namespace) -> None:
    (
        args.Fmin_extrusion_resolved,
        args.Fmax_extrusion_resolved,
        args.Fmin_travel_resolved,
        args.Fmax_travel_resolved,
    ) = resolve_feedrate_bounds(args)

    if (
        args.Fmin_extrusion_resolved <= 0.0
        or args.Fmax_extrusion_resolved <= args.Fmin_extrusion_resolved
    ):
        raise ValueError("Require 0 < Fmin_extrusion < Fmax_extrusion")
    if (
        args.Fmin_travel_resolved <= 0.0
        or args.Fmax_travel_resolved <= args.Fmin_travel_resolved
    ):
        raise ValueError("Require 0 < Fmin_travel < Fmax_travel")
    if args.Pmin_ms < 0.0 or args.Pmax_ms <= args.Pmin_ms:
        raise ValueError("Require 0 <= Pmin_ms < Pmax_ms")
    if args.block_rows < 1:
        raise ValueError("block_rows must be >= 1")
    if args.iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not (0.0 < args.increment_scale_limit < 1.0):
        raise ValueError("increment_scale_limit must satisfy 0 < value < 1")
    if not (0.0 < args.min_increment_scale_limit <= args.increment_scale_limit):
        raise ValueError("Require 0 < min_increment_scale_limit <= increment_scale_limit")
    if not (args.increment_scale_limit <= args.max_increment_scale_limit < 1.0):
        raise ValueError("Require increment_scale_limit <= max_increment_scale_limit < 1")
    if not (0.0 < args.trust_shrink < 1.0):
        raise ValueError("trust_shrink must satisfy 0 < value < 1")
    if args.trust_grow < 1.0:
        raise ValueError("trust_grow must be >= 1")
    if args.accept_tolerance_s < 0.0:
        raise ValueError("accept_tolerance_s must be >= 0")
    if args.patience < 1:
        raise ValueError("patience must be >= 1")
    if args.iterations > 1 and not args.gcode_in:
        raise ValueError("incremental-scale learning (--iterations > 1) requires --gcode_in")
    for name in ("change_weight", "smooth_weight", "target_weight", "time_weight"):
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name} must be >= 0")
    if args.true_validate and not args.gcode_in:
        raise ValueError("--true_validate requires --gcode_in")


def main() -> int:
    args = parse_args()
    started_total = time.perf_counter()
    try:
        validate_args(args)
        os.makedirs(args.out_dir, exist_ok=True)
        tau_lo, tau_star, tau_hi, tau_grid, alpha_grid = resolve_tau_targets(args)

        print("=" * 78)
        print("Sparse global residence-time optimizer")
        print("Mode: explicit LP scale variables + incremental TRUE-FIFO learning")
        print(f"Target tau window: [{tau_lo:.6f}, {tau_hi:.6f}] s; ideal={tau_star:.6f} s")
        print(
            "Feed-rate bounds: "
            f"extrusion=[{args.Fmin_extrusion_resolved:.6g}, {args.Fmax_extrusion_resolved:.6g}] mm/min; "
            f"travel=[{args.Fmin_travel_resolved:.6g}, {args.Fmax_travel_resolved:.6g}] mm/min"
        )

        original_csv = os.path.abspath(args.input)
        current_csv = original_csv
        current_gcode = os.path.abspath(args.gcode_in) if args.gcode_in else None
        initial_tau = scored_tau_from_csv(current_csv)
        current_true_metrics = residence_metrics(initial_tau, tau_lo, tau_star, tau_hi)
        initial_true_metrics = dict(current_true_metrics)

        incremental_mode = args.iterations > 1
        run_true_each_round = bool(args.true_validate or incremental_mode)
        radius = float(args.increment_scale_limit)
        consecutive_rejections = 0
        accepted_rounds = 0
        history: List[Dict[str, object]] = []
        best_round_result: Optional[Dict[str, object]] = None

        for attempt in range(1, args.iterations + 1):
            round_dir = os.path.join(args.out_dir, f"iteration_{attempt:03d}")
            print("-" * 78)
            print(
                f"Iteration {attempt}/{args.iterations}; accepted={accepted_rounds}; "
                f"current TRUE within={current_true_metrics['within_percent']:.3f}%, "
                f"vmax={current_true_metrics['vmax_s']:.6f}s"
            )

            inc_min = max(EPS, 1.0 - radius) if incremental_mode else None
            inc_max = 1.0 + radius if incremental_mode else None
            result = optimize_one_scale_round(
                args=args,
                input_csv=current_csv,
                gcode_in=current_gcode,
                round_dir=round_dir,
                tau_lo=tau_lo,
                tau_star=tau_star,
                tau_hi=tau_hi,
                tau_grid=tau_grid,
                alpha_grid=alpha_grid,
                increment_scale_min=inc_min,
                increment_scale_max=inc_max,
                run_true=run_true_each_round,
            )

            if run_true_each_round:
                candidate_metrics = result["true_metrics"]
                assert isinstance(candidate_metrics, dict)
                accepted, reason = compare_metric_priority(
                    candidate_metrics, current_true_metrics, args.accept_tolerance_s
                )
            else:
                candidate_metrics = result["predicted_metrics"]
                accepted, reason = True, "single_round_prediction"

            record: Dict[str, object] = {
                "attempt": attempt,
                "increment_scale_radius": radius if incremental_mode else None,
                "increment_scale_min": inc_min,
                "increment_scale_max": inc_max,
                "accepted": bool(accepted),
                "acceptance_reason": reason,
                "baseline_metrics": result["baseline_metrics"],
                "predicted_metrics": result["predicted_metrics"],
                "true_metrics": result["true_metrics"],
                "true_validation": result["true_comparison"],
                "round_dir": os.path.abspath(round_dir),
            }
            history.append(record)

            if accepted:
                accepted_rounds += 1
                consecutive_rejections = 0
                best_round_result = result
                if run_true_each_round:
                    assert result["true_csv"] is not None and result["true_gcode"] is not None
                    current_csv = str(result["true_csv"])
                    current_gcode = str(result["true_gcode"])
                    current_true_metrics = dict(candidate_metrics)
                else:
                    current_true_metrics = dict(candidate_metrics)
                    if result["candidate_gcode"] is not None:
                        current_gcode = str(result["candidate_gcode"])
                print(f"  ACCEPTED: {reason}")
                if incremental_mode:
                    radius = min(args.max_increment_scale_limit, radius * args.trust_grow)
            else:
                consecutive_rejections += 1
                print(f"  REJECTED: {reason}; keeping previous TRUE baseline")
                if incremental_mode:
                    radius *= args.trust_shrink
                    print(f"  Trust radius shrunk to {radius:.6g}")
                    if radius < args.min_increment_scale_limit:
                        print("  Stopping: incremental scale radius is below the configured minimum")
                        break
                    if consecutive_rejections >= args.patience:
                        print("  Stopping: rejection patience reached")
                        break

        history_path = os.path.join(args.out_dir, "incremental_history.csv")
        flat_history: List[Dict[str, object]] = []
        for rec in history:
            base = rec["baseline_metrics"] or {}
            pred = rec["predicted_metrics"] or {}
            true = rec["true_metrics"] or {}
            comp = rec["true_validation"] or {}
            flat_history.append(
                {
                    "attempt": rec["attempt"],
                    "accepted": rec["accepted"],
                    "acceptance_reason": rec["acceptance_reason"],
                    "increment_scale_radius": rec["increment_scale_radius"],
                    "baseline_within_percent": base.get("within_percent"),
                    "baseline_vmax_s": base.get("vmax_s"),
                    "predicted_within_percent": pred.get("within_percent"),
                    "predicted_vmax_s": pred.get("vmax_s"),
                    "true_within_percent": true.get("within_percent"),
                    "true_vmax_s": true.get("vmax_s"),
                    "true_violation_count": true.get("violation_count"),
                    "true_violation_sum_s": true.get("violation_sum_s"),
                    "true_mean_abs_target_error_s": true.get("mean_abs_target_error_s"),
                    "prediction_rmse_s": comp.get("true_minus_pred_rmse_s"),
                    "round_dir": rec["round_dir"],
                }
            )
        pd.DataFrame(flat_history).to_csv(history_path, index=False, float_format="%.8f")

        final_csv = current_csv
        final_gcode = current_gcode
        final_csv_copy = os.path.join(args.out_dir, "final_true_validation.csv")
        if os.path.abspath(final_csv) != os.path.abspath(final_csv_copy):
            shutil.copy2(final_csv, final_csv_copy)

        requested_gcode_out = args.gcode_out or os.path.join(args.out_dir, "incremental_scale_optimized.gcode")
        if final_gcode:
            os.makedirs(os.path.dirname(os.path.abspath(requested_gcode_out)), exist_ok=True)
            if os.path.abspath(final_gcode) != os.path.abspath(requested_gcode_out):
                shutil.copy2(final_gcode, requested_gcode_out)

        cumulative_path = os.path.join(args.out_dir, "cumulative_scale_by_seg_idx.csv")
        cumulative_scale_table(original_csv, final_csv).to_csv(
            cumulative_path, index=False, float_format="%.8f"
        )

        total_sec = time.perf_counter() - started_total
        summary = {
            "input": original_csv,
            "gcode_in": os.path.abspath(args.gcode_in) if args.gcode_in else None,
            "gcode_out": os.path.abspath(requested_gcode_out) if final_gcode else None,
            "mode": {
                "optimization_variable": "explicit_incremental_block_time_scale",
                "scale_interpretation": "1=current timing, >1 slower, <1 faster",
                "iterations_requested": args.iterations,
                "attempts_completed": len(history),
                "accepted_rounds": accepted_rounds,
                "true_fifo_relinearization": run_true_each_round,
            },
            "target": {
                "tau_min": tau_lo,
                "tau_ideal": tau_star,
                "tau_max": tau_hi,
                "target_mode": args.target_mode,
                "temp_c": args.temp_c,
                "alpha_min": args.alpha_min,
                "alpha_ideal": args.alpha_ideal,
                "alpha_max": args.alpha_max,
                "alpha_buffer": args.alpha_buffer,
            },
            "feedrate_bounds": {
                "extrusion": {
                    "Fmin_mm_per_min": args.Fmin_extrusion_resolved,
                    "Fmax_mm_per_min": args.Fmax_extrusion_resolved,
                },
                "travel": {
                    "enabled": bool(args.enable_travel_opt),
                    "Fmin_mm_per_min": args.Fmin_travel_resolved,
                    "Fmax_mm_per_min": args.Fmax_travel_resolved,
                },
            },
            "increment_learning": {
                "initial_scale_limit": args.increment_scale_limit,
                "final_scale_limit": radius,
                "minimum_scale_limit": args.min_increment_scale_limit,
                "maximum_scale_limit": args.max_increment_scale_limit,
                "trust_shrink": args.trust_shrink,
                "trust_grow": args.trust_grow,
                "accept_tolerance_s": args.accept_tolerance_s,
                "patience": args.patience,
                "acceptance_priority": [
                    "true_vmax_s",
                    "true_violation_count",
                    "true_violation_sum_s",
                    "true_mean_abs_target_error_s",
                ],
            },
            "initial_true_metrics": initial_true_metrics,
            "final_true_metrics": current_true_metrics,
            "history": history,
            "outputs": {
                "final_true_csv": os.path.abspath(final_csv_copy),
                "final_gcode": os.path.abspath(requested_gcode_out) if final_gcode else None,
                "incremental_history_csv": os.path.abspath(history_path),
                "cumulative_scale_csv": os.path.abspath(cumulative_path),
            },
            "total_elapsed_s": total_sec,
        }
        summary_path = os.path.join(args.out_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("=" * 78)
        final_metric_label = "Final TRUE" if run_true_each_round else "Final predicted"
        print(
            f"{final_metric_label}: within={current_true_metrics['within_percent']:.3f}% "
            f"violations={current_true_metrics['violation_count']:,} "
            f"vmax={current_true_metrics['vmax_s']:.6f}s"
        )
        print(f"Accepted rounds: {accepted_rounds}/{len(history)}")
        print(f"Final G-code:     {requested_gcode_out if final_gcode else 'not generated'}")
        print(f"Final reference CSV: {final_csv_copy}")
        print(f"History CSV:      {history_path}")
        print(f"Cumulative scale: {cumulative_path}")
        print(f"Summary JSON:     {summary_path}")
        print(f"Total elapsed:    {total_sec:.3f}s")
        print("=" * 78)
        return 0

    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
