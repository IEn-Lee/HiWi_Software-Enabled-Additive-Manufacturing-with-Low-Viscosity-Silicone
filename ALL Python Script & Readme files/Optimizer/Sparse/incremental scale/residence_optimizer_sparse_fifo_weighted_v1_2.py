#!/usr/bin/env python3
"""
Sparse global residence-time optimizer for segmented G-code.

Version 1.2: corrected FIFO-weight residence model.

Extrusion and travel retain independent feed-rate limits. The optimizer no
longer interprets the scalar baseline `t_in` as a single event on the G-code
timeline. Instead it reconstructs the same volume-weighted FIFO dependency
used by the TRUE FIFO builder.

Core idea
---------
The segmented CSV already contains an authoritative baseline FIFO result
(`t_in`, `t_out`, and `tau_s`). If extrusion volume, segment order,
V_nozzle, and E values remain unchanged, changing feed rate or dwell time
changes event timing but not the FIFO material pairing.

This program therefore:
1. Groups adjacent adjustable rows into control blocks exactly as before.
2. Reconstructs each FIFO-piece output time from the G-code row timeline.
3. Reconstructs each input time from the previous FIFO cycle using the same
   volume-weighted source mapping as the TRUE FIFO builder.
4. Builds residence time directly as t_out - weighted(t_out_sources), without
   treating baseline scalar t_in as a physical event position.
5. Solves two global linear programs:
   - Stage 1: minimize the worst residence-time violation.
   - Stage 2: preserve that optimum while minimizing target error, speed
     changes, and block-to-block roughness.
6. Patches the G-code once and can optionally run the existing TRUE FIFO
   builder once for final validation.

Important assumption
--------------------
This model is valid while G-code order, extrusion volume/E values, FIFO
cycle assignment, V_nozzle, and material-flow interpretation remain unchanged.
Under those conditions the FIFO volume weights are fixed, so the corrected
residence model remains linear in the timing variables. It optimizes only
timing (F and optional G4 P).

Example (alpha bounds)
----------------------
python residence_optimizer_sparse_v1.py \
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
python residence_optimizer_sparse_v1.py \
  --input model_segmented.csv \
  --target_mode direct \
  --tau_min 1940 --tau_ideal 2000 --tau_max 2060 \
  --Fmin_extrusion 180 --Fmax_extrusion 3000 \
  --out_dir sparse_optimization

LiQ5 Example
python residence_optimizer_sparse_v1.1_separate_travel.py 
  --input "LiQ5_sample_RTV-Body_Temp32_780s_segmented.csv" 
  --gcode_in "LiQ5_sample_RTV-Body_Temp32_780s_segmented.gcode" 
  --gcode_out "LiQ5_sample_RTV-Body_Temp32_780s_sparse_optimized.gcode" 
  --out_dir "LiQ5_sample_RTV-Body_Temp32_780s_sparse_optimization" 
  --target_mode alpha 
  --temp_c 32.0 
  --alpha_mode bounds 
  --alpha_min 0.887156 
  --alpha_ideal 0.933848 
  --alpha_max 0.980540 
  --Fmin_extrusion 1 
  --Fmax_extrusion 5000 
  --enable_travel_opt 
  --Fmin_travel 100 
  --Fmax_travel 12000 
  --enable_dwell_opt 
  --block_rows 150 
  --block_duration_sec 12 
  --max_target_samples 5000 
  --change_weight 0.10 
  --smooth_weight 0.05 
  --target_weight 1.00 
  --vmax_tolerance_s 0.01 
  --true_validate 
  --vnozzle 574

Alpha:
python residence_optimizer_sparse_v1.1_separate_travel.py --input "LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_segmented.csv" --gcode_in "LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_segmented.gcode" --gcode_out "LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_sparse_optimized.gcode" --out_dir "LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_sparse_optimization" --target_mode alpha --temp_c 32.0 --alpha_mode bounds --alpha_min 0.879469 --alpha_ideal 0.925747 --alpha_max 0.972045 --Fmin_extrusion 50 --Fmax_extrusion 5000 --enable_travel_opt --enable_dwell_opt --block_rows 50 --block_duration_sec 12 --max_target_samples 5000 --change_weight 0.10 --smooth_weight 0.05 --target_weight 1.00 --vmax_tolerance_s 0.01 --true_validate --vnozzle 574  --Fmin_travel 50 --Fmax_travel 10000

Residence Time:
python residence_optimizer_sparse_v1.1_separate_travel.py --input "LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_segmented.csv" --gcode_in "LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_segmented.gcode" --gcode_out "LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_sparse_optimized.gcode" --out_dir "LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_sparse_optimization" --target_mode direct --tau_min 770 --tau_ideal 800 --tau_max 840 --Fmin_extrusion 50 --Fmax_extrusion 5000 --enable_travel_opt --enable_dwell_opt --block_rows 50 --block_duration_sec 12 --max_target_samples 5000 --change_weight 0.10 --smooth_weight 0.05 --target_weight 1.00 --vmax_tolerance_s 0.01 --true_validate --vnozzle 574  --Fmin_travel 50 --Fmax_travel 10000
python residence_optimizer_sparse_v1.1_separate_travel.py --input "CFFFP_testmodel_E0.9_Temp32_780s_segmented.csv" --gcode_in "CFFFP_testmodel_E0.9_Temp32_780s_segmented.gcode" --gcode_out "CFFFP_testmodel_E0.9_Temp32_780s_sparse_optimized.gcode" --out_dir "CFFFP_testmodel_E0.9_Temp32_780s_sparse_optimization" --target_mode direct --tau_min 741 --tau_ideal 780 --tau_max 819 --Fmin_extrusion 50 --Fmax_extrusion 5000 --enable_travel_opt --enable_dwell_opt --block_rows 50 --block_duration_sec 12 --max_target_samples 5000 --change_weight 0.10 --smooth_weight 0.05 --target_weight 1.00 --vmax_tolerance_s 0.01 --true_validate --vnozzle 574  --Fmin_travel 50 --Fmax_travel 10000
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict, deque
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
    cycle: int
    cycle_local: int
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
        "seg_idx", "cmd", "cycle", "cycle_local", "d_mm",
        "F_mm_per_min", "V_mm3", "t_s", "t_in", "t_out", "tau_s", "note",
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
                cycle=int(finite_float(rec.get("cycle", -1), -1)),
                cycle_local=int(finite_float(rec.get("cycle_local", -1), -1)),
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


def is_fifo_piece_row(row: Row) -> bool:
    """Rows that participate in the finite-volume FIFO cycle mapping.

    This intentionally includes precondition pieces. They are not scored as
    printed product, but their t_out values are valid FIFO sources for the next
    cycle in the TRUE builder.
    """
    return (
        row.cmd == "G1"
        and row.cycle >= 1
        and math.isfinite(row.V_mm3)
        and row.V_mm3 > EPS
    )


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


def _scale_representation(rep: EventRepresentation, factor: float) -> EventRepresentation:
    return EventRepresentation(
        fixed_offset=rep.fixed_offset * factor,
        z_weights={idx: value * factor for idx, value in rep.z_weights.items() if abs(value * factor) > 1e-14},
    )


def _add_representations(parts: Sequence[EventRepresentation]) -> EventRepresentation:
    fixed = 0.0
    weights: Dict[int, float] = {}
    for rep in parts:
        fixed += rep.fixed_offset
        for idx, value in rep.z_weights.items():
            weights[idx] = weights.get(idx, 0.0) + value
    weights = {idx: value for idx, value in weights.items() if abs(value) > 1e-14}
    return EventRepresentation(fixed_offset=fixed, z_weights=weights)


def _difference_representation(end: EventRepresentation, start: EventRepresentation) -> EventRepresentation:
    fixed, weights = combine_event_difference(end, start)
    return EventRepresentation(fixed_offset=fixed, z_weights=weights)


def _weighted_representation(
    sources: Sequence[Tuple[EventRepresentation, float]],
) -> EventRepresentation:
    return _add_representations([_scale_representation(rep, weight) for rep, weight in sources])


def _fifo_piece_cycles(rows: Sequence[Row]) -> Dict[int, List[int]]:
    cycles: Dict[int, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if is_fifo_piece_row(row):
            cycles[row.cycle].append(i)
    for cycle, indices in cycles.items():
        indices.sort(key=lambda i: (rows[i].cycle_local, rows[i].seg_idx))
    return dict(sorted(cycles.items()))


def _find_fifo_time_origin(rows: Sequence[Row], cycles: Dict[int, List[int]]) -> Tuple[int, bool]:
    fifo_indices = [i for indices in cycles.values() for i in indices]
    if not fifo_indices:
        raise ValueError("No FIFO piece rows with cycle>=1 and positive V_mm3 were found")

    precondition = [i for i in fifo_indices if "precondition" in rows[i].note.lower()]
    if precondition:
        return min(precondition), True

    # The current TRUE builder starts the non-extrusion clock only when a
    # precondition piece exists. Without one, its non-extrusion delays are not
    # accumulated. We can still build the volume mapping, but exact travel/dwell
    # timing equivalence is not guaranteed. This fallback keeps compatibility.
    return min(fifo_indices), False


def _build_fifo_tout_representations(
    rows: Sequence[Row],
    blocks: Sequence[ControlBlock],
    cycles: Dict[int, List[int]],
) -> Tuple[Dict[int, EventRepresentation], Dict[str, object]]:
    """Build exact linear t_out expressions at FIFO-piece row ends.

    With a precondition row, the TRUE builder's absolute t_out is the elapsed
    G-code execution time from the first precondition-piece start through the
    current FIFO piece end. Representing those row-boundary events with the
    cumulative z variables is exact for the block timing model.
    """
    pieces, total_time, total_fixed = build_timeline_pieces(rows, blocks)
    if not pieces:
        raise ValueError("Timeline has no pieces")
    piece_ends = np.asarray([p.end_t for p in pieces], dtype=float)

    durations = np.asarray([row.t_s for row in rows], dtype=float)
    starts = np.concatenate(([0.0], np.cumsum(durations)[:-1]))
    ends = np.cumsum(durations)

    origin_row, has_precondition = _find_fifo_time_origin(rows, cycles)
    origin_time = float(starts[origin_row])
    origin_repr = represent_event(
        origin_time, pieces, piece_ends, total_time, total_fixed, len(blocks)
    )

    tout: Dict[int, EventRepresentation] = {}
    for indices in cycles.values():
        for row_i in indices:
            end_repr = represent_event(
                float(ends[row_i]), pieces, piece_ends, total_time, total_fixed, len(blocks)
            )
            tout[row_i] = _difference_representation(end_repr, origin_repr)

    return tout, {
        "fifo_time_origin_row_i": int(origin_row),
        "fifo_time_origin_seg_idx": int(rows[origin_row].seg_idx),
        "fifo_time_origin_has_precondition": bool(has_precondition),
    }


def _build_fifo_tin_representations(
    rows: Sequence[Row],
    cycles: Dict[int, List[int]],
    tout_repr: Dict[int, EventRepresentation],
) -> Tuple[Dict[int, EventRepresentation], Dict[str, object]]:
    """Reproduce TRUE FIFO volume backfill as linear source weights.

    For cycle >= 2, each current piece consumes volume from the previous cycle
    in FIFO order. Its t_in is the extracted-volume-weighted average of those
    source t_out values. If previous-cycle volume is insufficient, the TRUE
    builder uses the previous cycle's final t_out as fallback; the same rule is
    reproduced here.
    """
    tin: Dict[int, EventRepresentation] = {}
    fallback_volume_total = 0.0
    source_link_count = 0
    max_sources_per_piece = 0

    cycle_ids = sorted(cycles)
    if not cycle_ids:
        return tin, {
            "fifo_source_link_count": 0,
            "fifo_fallback_volume_mm3": 0.0,
            "fifo_max_sources_per_piece": 0,
        }

    # Current TRUE builder uses zero injection span for cycle 1 in the provided
    # implementation (inj_end_time remains zero). Precondition rows are not
    # scored, but their t_out remains available as the source for cycle 2.
    first_cycle = cycle_ids[0]
    for row_i in cycles[first_cycle]:
        tin[row_i] = EventRepresentation(fixed_offset=0.0, z_weights={})

    for cycle in cycle_ids[1:]:
        prev_indices = cycles.get(cycle - 1, [])
        curr_indices = cycles[cycle]
        if not prev_indices:
            raise ValueError(
                f"FIFO cycle {cycle} has no previous cycle {cycle - 1}; "
                "cannot reconstruct TRUE FIFO source mapping"
            )

        fifo = deque(
            {"row_i": src_i, "remaining": float(rows[src_i].V_mm3)}
            for src_i in prev_indices
        )
        fallback_i = prev_indices[-1]

        for dst_i in curr_indices:
            total_need = float(rows[dst_i].V_mm3)
            need = total_need
            if total_need <= EPS:
                tin[dst_i] = tout_repr[fallback_i]
                continue

            weighted_sources: List[Tuple[EventRepresentation, float]] = []
            used_sources = 0
            while need > EPS and fifo:
                head = fifo[0]
                take = min(float(head["remaining"]), need)
                if take > EPS:
                    weighted_sources.append((tout_repr[int(head["row_i"])], take / total_need))
                    used_sources += 1
                    source_link_count += 1
                head["remaining"] = float(head["remaining"]) - take
                need -= take
                if float(head["remaining"]) <= EPS:
                    fifo.popleft()

            if need > EPS:
                weighted_sources.append((tout_repr[fallback_i], need / total_need))
                fallback_volume_total += need
                source_link_count += 1
                used_sources += 1
                need = 0.0

            max_sources_per_piece = max(max_sources_per_piece, used_sources)
            tin[dst_i] = _weighted_representation(weighted_sources)

    return tin, {
        "fifo_source_link_count": int(source_link_count),
        "fifo_fallback_volume_mm3": float(fallback_volume_total),
        "fifo_max_sources_per_piece": int(max_sources_per_piece),
    }


def build_residence_equations(
    rows: Sequence[Row],
    blocks: Sequence[ControlBlock],
) -> Tuple[List[ResidenceEquation], Dict[str, object]]:
    """Build residence equations from TRUE FIFO volume dependencies.

    Correct model:
        t_out(i) = linear function of G-code/block timing
        t_in(i)  = sum_k w_ik * t_out(k), where w_ik comes from FIFO volume
        tau(i)   = t_out(i) - t_in(i)

    The old implementation instead mapped the already-averaged scalar baseline
    t_in back onto the G-code time axis. That loses the source-volume weights
    and becomes inaccurate when different source regions change timing by
    different amounts.
    """
    cycles = _fifo_piece_cycles(rows)
    if not cycles:
        raise ValueError("No FIFO cycle data could be reconstructed from the CSV")

    tout_repr, timing_diag = _build_fifo_tout_representations(rows, blocks, cycles)
    tin_repr, fifo_diag = _build_fifo_tin_representations(rows, cycles, tout_repr)
    z0 = baseline_z(blocks)

    equations: List[ResidenceEquation] = []
    skipped = 0
    raw_residuals: List[float] = []
    tin_residuals: List[float] = []
    tout_residuals: List[float] = []

    first_cycle_id = min(cycles)
    for row_i, row in enumerate(rows):
        if not is_scored_extrusion_row(row):
            continue
        if (
            row.cycle == first_cycle_id
            and math.isfinite(row.t_in)
            and abs(row.t_in) > 1e-6
        ):
            raise ValueError(
                "A scored FIFO row in the first cycle has non-zero t_in. "
                "The provided TRUE builder version uses zero cycle-1 injection span; "
                "this dataset appears to use a different cycle-1 initialization and "
                "needs an explicit injection-time dependency model."
            )
        if row_i not in tout_repr or row_i not in tin_repr:
            skipped += 1
            continue
        if not math.isfinite(row.tau_s):
            skipped += 1
            continue

        tau_repr = _difference_representation(tout_repr[row_i], tin_repr[row_i])
        predicted_tau_baseline = tau_repr.fixed_offset + dot_weights(tau_repr.z_weights, z0)
        residual = row.tau_s - predicted_tau_baseline
        raw_residuals.append(residual)

        if math.isfinite(row.t_out):
            pred_tout = tout_repr[row_i].fixed_offset + dot_weights(tout_repr[row_i].z_weights, z0)
            tout_residuals.append(row.t_out - pred_tout)
        if math.isfinite(row.t_in):
            pred_tin = tin_repr[row_i].fixed_offset + dot_weights(tin_repr[row_i].z_weights, z0)
            tin_residuals.append(row.t_in - pred_tin)

        # Keep only the tiny baseline residual as a numerical/CSV-rounding
        # calibration. The timing dependence itself comes from FIFO weights.
        equations.append(
            ResidenceEquation(
                row_i=row_i,
                seg_idx=row.seg_idx,
                baseline_tau=row.tau_s,
                constant=tau_repr.fixed_offset + residual,
                z_weights=tau_repr.z_weights,
            )
        )

    if not equations:
        raise ValueError("No valid scored extrusion rows could be modeled with FIFO volume weights")

    def abs_max(values: Sequence[float]) -> float:
        return float(max(abs(v) for v in values)) if values else 0.0

    def rmse(values: Sequence[float]) -> float:
        return float(math.sqrt(np.mean(np.square(values)))) if values else 0.0

    pieces, total_time, total_fixed = build_timeline_pieces(rows, blocks)
    diagnostics: Dict[str, object] = {
        "model_type": "fifo_volume_weighted_tout_dependencies",
        "equation_count": len(equations),
        "skipped_scored_rows": int(skipped),
        "fifo_cycle_count": int(len(cycles)),
        "fifo_piece_count": int(sum(len(v) for v in cycles.values())),
        "baseline_residual_abs_max": abs_max(raw_residuals),
        "baseline_residual_rmse": rmse(raw_residuals),
        "baseline_tin_residual_abs_max": abs_max(tin_residuals),
        "baseline_tin_residual_rmse": rmse(tin_residuals),
        "baseline_tout_residual_abs_max": abs_max(tout_residuals),
        "baseline_tout_residual_rmse": rmse(tout_residuals),
        "timeline_total_s": float(total_time),
        "timeline_fixed_s": float(total_fixed),
        **timing_diag,
        **fifo_diag,
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


def build_stage1_problem(
    equations: Sequence[ResidenceEquation],
    blocks: Sequence[ControlBlock],
    tau_lo: float,
    tau_hi: float,
) -> Tuple[np.ndarray, csr_matrix, np.ndarray, List[Tuple[Optional[float], Optional[float]]], int]:
    B = len(blocks)
    z_count = B + 1
    vmax_idx = z_count
    nvar = z_count + 1

    data: List[float] = []
    rr: List[int] = []
    cc: List[int] = []
    rhs: List[float] = []
    r = 0

    for eq in equations:
        # tau <= tau_hi + vmax  -> weights*z - vmax <= tau_hi - constant
        entries = list(eq.z_weights.items()) + [(vmax_idx, -1.0)]
        add_sparse_row(data, rr, cc, r, entries)
        rhs.append(tau_hi - eq.constant)
        r += 1

        # tau >= tau_lo - vmax -> -weights*z - vmax <= constant - tau_lo
        entries = [(idx, -value) for idx, value in eq.z_weights.items()] + [(vmax_idx, -1.0)]
        add_sparse_row(data, rr, cc, r, entries)
        rhs.append(eq.constant - tau_lo)
        r += 1

    for b, block in enumerate(blocks):
        # y = z[b+1] - z[b] <= ub
        add_sparse_row(data, rr, cc, r, [(b + 1, 1.0), (b, -1.0)])
        rhs.append(block.duration_ub)
        r += 1

        # y >= lb -> z[b] - z[b+1] <= -lb
        add_sparse_row(data, rr, cc, r, [(b, 1.0), (b + 1, -1.0)])
        rhs.append(-block.duration_lb)
        r += 1

    A_ub = coo_matrix((data, (rr, cc)), shape=(r, nvar)).tocsr()
    b_ub = np.asarray(rhs, dtype=float)
    c = np.zeros(nvar, dtype=float)
    c[vmax_idx] = 1.0
    bounds: List[Tuple[Optional[float], Optional[float]]] = [(None, None)] * nvar
    bounds[0] = (0.0, 0.0)
    bounds[vmax_idx] = (0.0, None)
    return c, A_ub, b_ub, bounds, vmax_idx


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
    List[Tuple[Optional[float], Optional[float]]],
    Dict[str, object],
]:
    B = len(blocks)
    z_count = B + 1
    vmax_idx = z_count
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

    # Residence constraints with shared minimax slack.
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

    # Duration bounds and absolute deviation from baseline duration.
    for b, block in enumerate(blocks):
        add_sparse_row(data, rr, cc, r, [(b + 1, 1.0), (b, -1.0)])
        rhs.append(block.duration_ub)
        r += 1

        add_sparse_row(data, rr, cc, r, [(b, 1.0), (b + 1, -1.0)])
        rhs.append(-block.duration_lb)
        r += 1

        d_idx = d_start + b
        # y - y0 <= d
        add_sparse_row(data, rr, cc, r, [(b + 1, 1.0), (b, -1.0), (d_idx, -1.0)])
        rhs.append(block.base_duration)
        r += 1
        # -(y-y0) <= d
        add_sparse_row(data, rr, cc, r, [(b, 1.0), (b + 1, -1.0), (d_idx, -1.0)])
        rhs.append(-block.base_duration)
        r += 1

    # Absolute difference between neighboring normalized block scales.
    for q_i, (a, b) in enumerate(pairs):
        q_idx = q_start + q_i
        ya = max(blocks[a].base_duration, EPS)
        yb = max(blocks[b].base_duration, EPS)
        entries = [
            (b + 1, 1.0 / yb), (b, -1.0 / yb),
            (a + 1, -1.0 / ya), (a, 1.0 / ya),
            (q_idx, -1.0),
        ]
        add_sparse_row(data, rr, cc, r, entries)
        rhs.append(0.0)
        r += 1
        add_sparse_row(data, rr, cc, r, [(idx, -value) for idx, value in entries[:-1]] + [(q_idx, -1.0)])
        rhs.append(0.0)
        r += 1

    # Sampled absolute ideal-target error.
    for local_i, eq_i in enumerate(targets):
        eq = equations[eq_i]
        e_idx = e_start + local_i
        add_sparse_row(data, rr, cc, r, list(eq.z_weights.items()) + [(e_idx, -1.0)])
        rhs.append(tau_star - eq.constant)
        r += 1
        add_sparse_row(data, rr, cc, r, [(idx, -value) for idx, value in eq.z_weights.items()] + [(e_idx, -1.0)])
        rhs.append(eq.constant - tau_star)
        r += 1

    A_ub = coo_matrix((data, (rr, cc)), shape=(r, nvar)).tocsr()
    b_ub = np.asarray(rhs, dtype=float)
    c = np.zeros(nvar, dtype=float)

    if B > 0 and change_weight > 0.0:
        for b, block in enumerate(blocks):
            c[d_start + b] = change_weight / (B * max(block.base_duration, EPS))

    if q_count > 0 and smooth_weight > 0.0:
        c[q_start:q_start + q_count] = smooth_weight / q_count

    tau_scale = max(1.0, 0.5 * (tau_hi - tau_lo))
    if e_count > 0 and target_weight > 0.0:
        c[e_start:e_start + e_count] = target_weight / (e_count * tau_scale)

    if time_weight > 0.0:
        total_variable = max(sum(b.base_duration for b in blocks), EPS)
        c[B] += time_weight / total_variable  # z[B] is total adjustable time.

    # Tiny objective on vmax improves numerical preference when tolerance is nonzero.
    c[vmax_idx] = 1e-9

    bounds: List[Tuple[Optional[float], Optional[float]]] = [(None, None)] * nvar
    bounds[0] = (0.0, 0.0)
    bounds[vmax_idx] = (0.0, max(0.0, vmax_upper))
    for idx in range(d_start, nvar):
        bounds[idx] = (0.0, None)

    metadata: Dict[str, object] = {
        "vmax_idx": vmax_idx,
        "d_start": d_start,
        "q_start": q_start,
        "e_start": e_start,
        "target_indices": targets,
        "smooth_pairs": pairs,
        "variable_count": nvar,
        "constraint_count": r,
    }
    return c, A_ub, b_ub, bounds, metadata


def solve_lp(
    c: np.ndarray,
    A_ub: csr_matrix,
    b_ub: np.ndarray,
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
    z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    F_new = np.array([r.F for r in rows], dtype=float)
    P_new_ms = np.array([r.t_s * 1000.0 if r.cmd == "G4" else 0.0 for r in rows], dtype=float)
    t_new = np.array([r.t_s for r in rows], dtype=float)
    row_scale = np.ones(len(rows), dtype=float)

    y = np.diff(z)
    for block in blocks:
        scale = y[block.index] / max(block.base_duration, EPS)
        for i in block.row_indices:
            row = rows[i]
            row_scale[i] = scale
            t_new[i] = row.t_s * scale
            if block.kind == "dwell":
                P_new_ms[i] = t_new[i] * 1000.0
            else:
                F0 = infer_feedrate(row)
                if scale > EPS and F0 > EPS:
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
    z: np.ndarray,
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

    y = np.diff(z)
    block_records = []
    for block in blocks:
        y_new = y[block.index]
        block_records.append(
            {
                "block": block.index,
                "kind": block.kind,
                "start_seg": rows[block.start_row].seg_idx,
                "end_seg": rows[block.end_row].seg_idx,
                "row_count": len(block.row_indices),
                "base_duration_s": block.base_duration,
                "optimized_duration_s": y_new,
                "time_scale": y_new / max(block.base_duration, EPS),
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
        print(f"Target tau window: [{tau_lo:.6f}, {tau_hi:.6f}] s; ideal={tau_star:.6f} s")
        print("FIFO model: volume-weighted source t_out dependencies (corrected model).")
        print("Assumption: E/volume, cycle assignment, G-code order, and V_nozzle stay fixed.")
        print(
            "Feed-rate bounds: "
            f"extrusion=[{args.Fmin_extrusion_resolved:.6g}, {args.Fmax_extrusion_resolved:.6g}] mm/min; "
            f"travel=[{args.Fmin_travel_resolved:.6g}, {args.Fmax_travel_resolved:.6g}] mm/min"
        )

        rows, source_df = read_segments_csv(args.input)
        print(f"Loaded {len(rows):,} CSV rows")

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
        print(f"Created {len(blocks):,} control blocks")

        equations, model_diag = build_residence_equations(rows, blocks)
        print(f"Built {len(equations):,} sparse residence equations using FIFO volume weights")
        print(
            "FIFO baseline check: "
            f"tau_abs_max={float(model_diag['baseline_residual_abs_max']):.3e}s, "
            f"tin_abs_max={float(model_diag['baseline_tin_residual_abs_max']):.3e}s, "
            f"tout_abs_max={float(model_diag['baseline_tout_residual_abs_max']):.3e}s"
        )
        if not bool(model_diag.get("fifo_time_origin_has_precondition", False)):
            print(
                "Warning: no precondition FIFO piece was found. The provided TRUE builder "
                "uses a special no-precondition non-extrusion timing convention, so exact "
                "travel/dwell equivalence is not guaranteed."
            )
        if float(model_diag.get("fifo_fallback_volume_mm3", 0.0)) > 1e-9:
            print(
                "Warning: FIFO source volume was insufficient for some pieces; "
                f"fallback volume={float(model_diag['fifo_fallback_volume_mm3']):.6g} mm^3"
            )
        if model_diag["skipped_scored_rows"]:
            print(f"Warning: skipped {model_diag['skipped_scored_rows']} scored rows with invalid FIFO mapping")

        z0 = baseline_z(blocks)
        baseline_tau_model = evaluate_equations(equations, z0)
        baseline_tau_csv = np.array([eq.baseline_tau for eq in equations], dtype=float)
        baseline_reconstruction_error = baseline_tau_model - baseline_tau_csv
        reconstruction_abs_max = float(np.max(np.abs(baseline_reconstruction_error)))
        print(f"Baseline reconstruction max error: {reconstruction_abs_max:.3e} s")

        baseline_metrics = residence_metrics(baseline_tau_csv, tau_lo, tau_star, tau_hi)
        print(
            f"Baseline: within={baseline_metrics['within_percent']:.3f}% "
            f"violations={baseline_metrics['violation_count']:,} "
            f"vmax={baseline_metrics['vmax_s']:.6f}s"
        )

        c1, A1, b1, bounds1, vmax_idx1 = build_stage1_problem(
            equations, blocks, tau_lo, tau_hi
        )
        print(
            f"Stage 1 LP: variables={A1.shape[1]:,}, constraints={A1.shape[0]:,}, "
            f"nonzeros={A1.nnz:,}"
        )
        stage1, stage1_sec = solve_lp(
            c1, A1, b1, bounds1, args.solver_time_limit, "Stage 1 minimax"
        )
        vmax_star = max(0.0, float(stage1.x[vmax_idx1]))
        print(f"Stage 1 complete in {stage1_sec:.3f}s; minimal worst violation={vmax_star:.6f}s")

        vmax_upper = vmax_star + max(0.0, args.vmax_tolerance_s)
        c2, A2, b2, bounds2, meta2 = build_stage2_problem(
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
        print(
            f"Stage 2 LP: variables={A2.shape[1]:,}, constraints={A2.shape[0]:,}, "
            f"nonzeros={A2.nnz:,}, target_samples={len(meta2['target_indices']):,}"
        )
        stage2, stage2_sec = solve_lp(
            c2, A2, b2, bounds2, args.solver_time_limit, "Stage 2 quality"
        )

        B = len(blocks)
        z_opt = np.asarray(stage2.x[:B + 1], dtype=float)
        tau_pred = evaluate_equations(equations, z_opt)
        optimized_metrics = residence_metrics(tau_pred, tau_lo, tau_star, tau_hi)
        print(
            f"Optimized: within={optimized_metrics['within_percent']:.3f}% "
            f"violations={optimized_metrics['violation_count']:,} "
            f"vmax={optimized_metrics['vmax_s']:.6f}s"
        )

        F_new, P_new_ms, t_new, row_scale = solution_to_rows(rows, blocks, z_opt)
        alpha_pred = alpha_from_tau(tau_pred, tau_grid, alpha_grid) if tau_grid is not None else None

        prediction_path, block_path = write_outputs(
            out_dir=args.out_dir,
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
            z=z_opt,
        )

        patched_info = None
        gcode_out = args.gcode_out
        if args.gcode_in:
            if not gcode_out:
                gcode_out = os.path.join(args.out_dir, "sparse_optimized.gcode")
            patched, expected = patch_gcode_by_segidx(
                args.gcode_in, gcode_out, rows, F_new, P_new_ms,
                args.enable_travel_opt, args.enable_dwell_opt,
            )
            patched_info = {"patched": patched, "expected": expected, "coverage": patched / max(1, expected)}
            print(f"Patched G-code: {patched}/{expected} lines -> {gcode_out}")
            if expected > 0 and patched < 0.98 * expected:
                print("Warning: G-code patch coverage is below 98%")

        true_comparison = None
        true_files = None
        if args.true_validate:
            assert gcode_out is not None
            validation_csv, validation_gcode = run_true_validation(
                module_name=args.true_builder_module,
                gcode_in=gcode_out,
                out_dir=args.out_dir,
                V_nozzle=args.vnozzle,
                d_filament=args.d_filament,
                e_mode=args.e_mode,
            )
            true_comparison = compare_true_validation(validation_csv, equations, tau_pred)
            true_files = {"csv": validation_csv, "gcode": validation_gcode}
            print(
                "TRUE validation: "
                f"RMSE={true_comparison['true_minus_pred_rmse_s']:.6f}s, "
                f"max_abs={true_comparison['true_minus_pred_abs_max_s']:.6f}s"
            )

        total_sec = time.perf_counter() - started_total
        summary = {
            "optimizer_version": "1.2_fifo_weighted_model",
            "residence_model": "fifo_volume_weighted_tout_dependencies",
            "input": os.path.abspath(args.input),
            "gcode_in": os.path.abspath(args.gcode_in) if args.gcode_in else None,
            "gcode_out": os.path.abspath(gcode_out) if gcode_out else None,
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
            "model": model_diag,
            "row_count": len(rows),
            "control_block_count": len(blocks),
            "baseline_reconstruction_abs_max_s": reconstruction_abs_max,
            "stage1": {
                "elapsed_s": stage1_sec,
                "minimal_worst_violation_s": vmax_star,
                "solver_message": stage1.message,
            },
            "stage2": {
                "elapsed_s": stage2_sec,
                "allowed_worst_violation_s": vmax_upper,
                "solver_message": stage2.message,
                "target_sample_count": len(meta2["target_indices"]),
            },
            "baseline_metrics": baseline_metrics,
            "optimized_metrics": optimized_metrics,
            "patch": patched_info,
            "true_validation": true_comparison,
            "true_files": true_files,
            "outputs": {
                "prediction_csv": os.path.abspath(prediction_path),
                "block_csv": os.path.abspath(block_path),
            },
            "total_elapsed_s": total_sec,
        }
        summary_path = os.path.join(args.out_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"Prediction CSV: {prediction_path}")
        print(f"Block CSV:      {block_path}")
        print(f"Summary JSON:   {summary_path}")
        print(f"Total elapsed:  {total_sec:.3f}s")
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
