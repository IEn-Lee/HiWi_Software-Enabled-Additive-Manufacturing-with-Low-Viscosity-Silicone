#!/usr/bin/env python3
"""
Free segment-level FIFO residence-time optimizer for segmented G-code.

Version 1.0
-----------
This version removes control-block grouping and does NOT preserve the original
feed-rate distribution. Every adjustable G-code segment is an independent
control degree of freedom.

Conceptual optimization variable
--------------------------------
For every adjustable segment j, the optimizer independently chooses a new
execution duration t_j within the physical timing bounds implied by:

* extrusion: Fmin_extrusion <= F_j <= Fmax_extrusion
* travel:    Fmin_travel    <= F_j <= Fmax_travel
* dwell:     Pmin_ms        <= P_j <= Pmax_ms

For movement rows, after solving:

    F_new = F_current / time_scale
    time_scale = t_new / t_current

The implementation keeps sparse cumulative-time variables z only as a memory-
efficient bookkeeping device. Each independent segment duration is exactly:

    t_j = z[j+1] - z[j]

No two adjustable segments are forced to share a scale or preserve their
original relative speeds.

Optimization stages
-------------------
Stage 1 (feasibility / minimax):
    Minimize the single worst residence-time violation across ALL scored rows.
    This answers whether the target window is reachable when every adjustable
    segment is free within its physical limits.

Stage 2 (quality):
    Keep the Stage-1 worst violation (within vmax_tolerance_s), then minimize
    absolute error to tau_ideal on a configurable sample of residence rows.
    There is deliberately no change_weight or smooth_weight: the original
    speed distribution is not preserved.

Optional TRUE-FIFO relinearization:
    With --iterations > 1, each accepted candidate is rebuilt by the existing
    TRUE FIFO builder and becomes the authoritative baseline for the next full-
    freedom LP round. Unlike the incremental-scale optimizer, there is no local
    scale trust window: every round may choose any physically legal segment
    speed.

Important assumption
--------------------
The linear timing model assumes G-code order, E/volume, V_nozzle, and the
baseline event interpretation stay fixed during one LP solve. TRUE validation
is recommended after large speed reconstruction.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
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
class SegmentControl:
    index: int
    row_i: int
    seg_idx: int
    kind: str
    base_duration: float
    duration_lb: float
    duration_ub: float
    base_F: float = math.nan
    allowed_f_min: float = math.nan
    allowed_f_max: float = math.nan


@dataclass
class TimelinePiece:
    start_t: float
    end_t: float
    kind: str  # fixed or variable
    fixed_before: float
    variable_prefix_count: int
    control_index: Optional[int] = None


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
# Input helpers
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


def classify_adjustable_row(row: Row, enable_travel_opt: bool, enable_dwell_opt: bool) -> Optional[str]:
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
# Alpha <-> tau
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


def build_alpha_tau_lookup(temp_c: float, tau_upper_s: float, n_eval: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
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
    return np.interp(np.asarray(tau, dtype=float), tau_grid, alpha_grid,
                     left=alpha_grid[0], right=alpha_grid[-1])


# ---------------------------------------------------------------------------
# One independent control per adjustable segment
# ---------------------------------------------------------------------------

def row_duration_bounds(
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
) -> Tuple[float, float, float, float, float]:
    """Return duration_lb, duration_ub, base_F, allowed_Fmin, allowed_Fmax."""
    if kind in {"extrusion", "travel"}:
        F0 = infer_feedrate(row)
        if F0 <= EPS:
            raise ValueError(f"seg_idx={row.seg_idx} has no valid feed rate")
        if kind == "extrusion":
            fmin, fmax = Fmin_extrusion, Fmax_extrusion
        else:
            fmin, fmax = Fmin_travel, Fmax_travel

        scale_lo = F0 / fmax
        scale_hi = F0 / fmin
        if max_relative_F_change > 0.0:
            r = min(float(max_relative_F_change), 0.95)
            scale_lo = max(scale_lo, 1.0 / (1.0 + r))
            scale_hi = min(scale_hi, 1.0 / (1.0 - r))
        if scale_lo > scale_hi + 1e-12:
            raise ValueError(f"seg_idx={row.seg_idx} has incompatible movement timing bounds")
        return row.t_s * scale_lo, row.t_s * scale_hi, F0, fmin, fmax

    if kind == "dwell":
        P0 = row.t_s * 1000.0
        if P0 <= EPS:
            raise ValueError(f"seg_idx={row.seg_idx} has zero baseline dwell and cannot be optimized")
        lo = max(0.0, Pmin_ms / 1000.0)
        hi = max(0.0, Pmax_ms / 1000.0)
        if max_relative_P_change > 0.0:
            r = min(float(max_relative_P_change), 0.95)
            lo = max(lo, row.t_s * (1.0 - r))
            hi = min(hi, row.t_s * (1.0 + r))
        if lo > hi + 1e-12:
            raise ValueError(f"seg_idx={row.seg_idx} has incompatible dwell timing bounds")
        return lo, hi, math.nan, math.nan, math.nan

    raise ValueError(f"Unknown control kind: {kind}")


def build_segment_controls(
    rows: Sequence[Row],
    enable_travel_opt: bool,
    enable_dwell_opt: bool,
    Fmin_extrusion: float,
    Fmax_extrusion: float,
    Fmin_travel: float,
    Fmax_travel: float,
    Pmin_ms: float,
    Pmax_ms: float,
    max_relative_F_change: float,
    max_relative_P_change: float,
) -> Tuple[List[SegmentControl], np.ndarray]:
    controls: List[SegmentControl] = []
    row_to_control = np.full(len(rows), -1, dtype=int)
    for i, row in enumerate(rows):
        kind = classify_adjustable_row(row, enable_travel_opt, enable_dwell_opt)
        if kind is None:
            continue
        lb, ub, F0, fmin, fmax = row_duration_bounds(
            row, kind,
            Fmin_extrusion, Fmax_extrusion,
            Fmin_travel, Fmax_travel,
            Pmin_ms, Pmax_ms,
            max_relative_F_change, max_relative_P_change,
        )
        control = SegmentControl(
            index=len(controls),
            row_i=i,
            seg_idx=row.seg_idx,
            kind=kind,
            base_duration=row.t_s,
            duration_lb=max(0.0, lb),
            duration_ub=max(0.0, ub),
            base_F=F0,
            allowed_f_min=fmin,
            allowed_f_max=fmax,
        )
        controls.append(control)
        row_to_control[i] = control.index
    if not controls:
        raise ValueError("No adjustable segments were found")
    return controls, row_to_control


# ---------------------------------------------------------------------------
# Sparse event model
# ---------------------------------------------------------------------------

def build_timeline_pieces(rows: Sequence[Row], controls: Sequence[SegmentControl]) -> Tuple[List[TimelinePiece], float, float]:
    durations = np.asarray([r.t_s for r in rows], dtype=float)
    starts = np.concatenate(([0.0], np.cumsum(durations)[:-1]))
    ends = np.cumsum(durations)
    total_time = float(ends[-1]) if len(ends) else 0.0

    pieces: List[TimelinePiece] = []
    current_t = 0.0
    fixed_before = 0.0

    for control in controls:
        start = float(starts[control.row_i])
        end = float(ends[control.row_i])
        if start > current_t + EPS:
            fixed_duration = start - current_t
            pieces.append(TimelinePiece(
                start_t=current_t,
                end_t=start,
                kind="fixed",
                fixed_before=fixed_before,
                variable_prefix_count=control.index,
            ))
            fixed_before += fixed_duration

        pieces.append(TimelinePiece(
            start_t=start,
            end_t=end,
            kind="variable",
            fixed_before=fixed_before,
            variable_prefix_count=control.index,
            control_index=control.index,
        ))
        current_t = end

    if current_t < total_time - EPS:
        pieces.append(TimelinePiece(
            start_t=current_t,
            end_t=total_time,
            kind="fixed",
            fixed_before=fixed_before,
            variable_prefix_count=len(controls),
        ))
        fixed_before += total_time - current_t

    return pieces, total_time, fixed_before


def represent_event(
    event_time: float,
    pieces: Sequence[TimelinePiece],
    piece_ends: np.ndarray,
    total_time: float,
    total_fixed_time: float,
    control_count: int,
) -> EventRepresentation:
    t = float(event_time)
    if t <= 0.0:
        return EventRepresentation(fixed_offset=t, z_weights={0: 1.0})
    if t >= total_time:
        return EventRepresentation(
            fixed_offset=total_fixed_time + (t - total_time),
            z_weights={control_count: 1.0},
        )

    piece_i = int(np.searchsorted(piece_ends, t, side="left"))
    piece_i = min(max(piece_i, 0), len(pieces) - 1)
    piece = pieces[piece_i]

    if piece.kind == "fixed":
        return EventRepresentation(
            fixed_offset=piece.fixed_before + max(0.0, t - piece.start_t),
            z_weights={piece.variable_prefix_count: 1.0},
        )

    assert piece.control_index is not None
    duration = max(piece.end_t - piece.start_t, EPS)
    frac = min(1.0, max(0.0, (t - piece.start_t) / duration))
    weights: Dict[int, float] = {}
    if 1.0 - frac > EPS:
        weights[piece.control_index] = 1.0 - frac
    if frac > EPS:
        weights[piece.control_index + 1] = frac
    return EventRepresentation(fixed_offset=piece.fixed_before, z_weights=weights)


def combine_event_difference(end: EventRepresentation, start: EventRepresentation) -> Tuple[float, Dict[int, float]]:
    weights: Dict[int, float] = {}
    for idx, value in end.z_weights.items():
        weights[idx] = weights.get(idx, 0.0) + value
    for idx, value in start.z_weights.items():
        weights[idx] = weights.get(idx, 0.0) - value
    weights = {idx: val for idx, val in weights.items() if abs(val) > 1e-14}
    return end.fixed_offset - start.fixed_offset, weights


def baseline_z(controls: Sequence[SegmentControl]) -> np.ndarray:
    y0 = np.asarray([c.base_duration for c in controls], dtype=float)
    return np.concatenate(([0.0], np.cumsum(y0)))


def dot_weights(weights: Dict[int, float], z: np.ndarray) -> float:
    return float(sum(value * z[idx] for idx, value in weights.items()))


def build_residence_equations(rows: Sequence[Row], controls: Sequence[SegmentControl]) -> Tuple[List[ResidenceEquation], Dict[str, float]]:
    pieces, total_time, total_fixed = build_timeline_pieces(rows, controls)
    if not pieces:
        raise ValueError("Timeline has no pieces")
    piece_ends = np.asarray([p.end_t for p in pieces], dtype=float)
    z0 = baseline_z(controls)

    equations: List[ResidenceEquation] = []
    skipped = 0
    residuals: List[float] = []
    for row_i, row in enumerate(rows):
        if not is_scored_extrusion_row(row):
            continue
        if not (math.isfinite(row.t_in) and math.isfinite(row.t_out) and math.isfinite(row.tau_s)):
            skipped += 1
            continue
        if row.t_out + 1e-9 < row.t_in:
            skipped += 1
            continue

        start_repr = represent_event(row.t_in, pieces, piece_ends, total_time, total_fixed, len(controls))
        end_repr = represent_event(row.t_out, pieces, piece_ends, total_time, total_fixed, len(controls))
        fixed_part, weights = combine_event_difference(end_repr, start_repr)
        pred0 = fixed_part + dot_weights(weights, z0)
        residual = row.tau_s - pred0
        residuals.append(residual)
        equations.append(ResidenceEquation(
            row_i=row_i,
            seg_idx=row.seg_idx,
            baseline_tau=row.tau_s,
            constant=fixed_part + residual,
            z_weights=weights,
        ))

    if not equations:
        raise ValueError("No valid scored extrusion rows with finite t_in/t_out/tau_s")
    diagnostics = {
        "equation_count": len(equations),
        "skipped_scored_rows": skipped,
        "baseline_residual_abs_max": float(max(abs(v) for v in residuals)) if residuals else 0.0,
        "baseline_residual_rmse": float(math.sqrt(np.mean(np.square(residuals)))) if residuals else 0.0,
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
# Sparse LPs
# ---------------------------------------------------------------------------

def add_sparse_row(data: List[float], row_idx: List[int], col_idx: List[int], r: int,
                   entries: Iterable[Tuple[int, float]]) -> None:
    for c, value in entries:
        if abs(value) <= 1e-15:
            continue
        row_idx.append(r)
        col_idx.append(c)
        data.append(float(value))


def add_segment_duration_constraints(
    controls: Sequence[SegmentControl],
    data: List[float], rr: List[int], cc: List[int], rhs: List[float], r_start: int,
) -> int:
    r = r_start
    for j, control in enumerate(controls):
        # duration_j = z[j+1] - z[j] <= ub
        add_sparse_row(data, rr, cc, r, [(j + 1, 1.0), (j, -1.0)])
        rhs.append(control.duration_ub)
        r += 1
        # duration_j >= lb -> z[j] - z[j+1] <= -lb
        add_sparse_row(data, rr, cc, r, [(j, 1.0), (j + 1, -1.0)])
        rhs.append(-control.duration_lb)
        r += 1
    return r


def build_stage1_problem(
    equations: Sequence[ResidenceEquation], controls: Sequence[SegmentControl],
    tau_lo: float, tau_hi: float,
) -> Tuple[np.ndarray, csr_matrix, np.ndarray, List[Tuple[Optional[float], Optional[float]]], int]:
    S = len(controls)
    z_count = S + 1
    vmax_idx = z_count
    nvar = z_count + 1

    data: List[float] = []
    rr: List[int] = []
    cc: List[int] = []
    rhs: List[float] = []
    r = 0

    for eq in equations:
        add_sparse_row(data, rr, cc, r, list(eq.z_weights.items()) + [(vmax_idx, -1.0)])
        rhs.append(tau_hi - eq.constant)
        r += 1
        add_sparse_row(data, rr, cc, r,
                       [(idx, -v) for idx, v in eq.z_weights.items()] + [(vmax_idx, -1.0)])
        rhs.append(eq.constant - tau_lo)
        r += 1

    r = add_segment_duration_constraints(controls, data, rr, cc, rhs, r)
    A_ub = coo_matrix((data, (rr, cc)), shape=(r, nvar)).tocsr()
    b_ub = np.asarray(rhs, dtype=float)
    c = np.zeros(nvar, dtype=float)
    c[vmax_idx] = 1.0
    bounds: List[Tuple[Optional[float], Optional[float]]] = [(None, None)] * nvar
    bounds[0] = (0.0, 0.0)
    bounds[vmax_idx] = (0.0, None)
    return c, A_ub, b_ub, bounds, vmax_idx


def select_target_equations(
    equations: Sequence[ResidenceEquation], tau_lo: float, tau_hi: float, max_samples: int
) -> List[int]:
    n = len(equations)
    if max_samples <= 0 or n <= max_samples:
        return list(range(n))
    baseline = np.asarray([eq.baseline_tau for eq in equations], dtype=float)
    violation = np.maximum(tau_lo - baseline, 0.0) + np.maximum(baseline - tau_hi, 0.0)
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


def build_stage2_problem(
    equations: Sequence[ResidenceEquation],
    controls: Sequence[SegmentControl],
    tau_lo: float,
    tau_star: float,
    tau_hi: float,
    vmax_upper: float,
    max_target_samples: int,
    target_weight: float,
    time_weight: float,
) -> Tuple[np.ndarray, csr_matrix, np.ndarray, List[Tuple[Optional[float], Optional[float]]], Dict[str, object]]:
    S = len(controls)
    z_count = S + 1
    vmax_idx = z_count
    targets = select_target_equations(equations, tau_lo, tau_hi, max_target_samples)
    e_start = vmax_idx + 1
    e_count = len(targets)
    nvar = e_start + e_count

    data: List[float] = []
    rr: List[int] = []
    cc: List[int] = []
    rhs: List[float] = []
    r = 0

    # Every scored row remains constrained by the Stage-1 minimax envelope.
    for eq in equations:
        add_sparse_row(data, rr, cc, r, list(eq.z_weights.items()) + [(vmax_idx, -1.0)])
        rhs.append(tau_hi - eq.constant)
        r += 1
        add_sparse_row(data, rr, cc, r,
                       [(idx, -v) for idx, v in eq.z_weights.items()] + [(vmax_idx, -1.0)])
        rhs.append(eq.constant - tau_lo)
        r += 1

    r = add_segment_duration_constraints(controls, data, rr, cc, rhs, r)

    # Only sampled residence rows add absolute-error variables. There are no
    # per-segment change or smoothness variables in the free-segment model.
    for local_i, eq_i in enumerate(targets):
        eq = equations[eq_i]
        e_idx = e_start + local_i
        add_sparse_row(data, rr, cc, r, list(eq.z_weights.items()) + [(e_idx, -1.0)])
        rhs.append(tau_star - eq.constant)
        r += 1
        add_sparse_row(data, rr, cc, r,
                       [(idx, -v) for idx, v in eq.z_weights.items()] + [(e_idx, -1.0)])
        rhs.append(eq.constant - tau_star)
        r += 1

    A_ub = coo_matrix((data, (rr, cc)), shape=(r, nvar)).tocsr()
    b_ub = np.asarray(rhs, dtype=float)
    c = np.zeros(nvar, dtype=float)
    tau_scale = max(1.0, 0.5 * (tau_hi - tau_lo))
    if e_count > 0 and target_weight > 0.0:
        c[e_start:e_start + e_count] = target_weight / (e_count * tau_scale)
    if time_weight > 0.0:
        # z[S] is total adjustable execution time because z[0]=0.
        total_base = max(sum(cn.base_duration for cn in controls), EPS)
        c[S] += time_weight / total_base
    c[vmax_idx] = 1e-9

    bounds: List[Tuple[Optional[float], Optional[float]]] = [(None, None)] * nvar
    bounds[0] = (0.0, 0.0)
    bounds[vmax_idx] = (0.0, max(0.0, vmax_upper))
    for idx in range(e_start, nvar):
        bounds[idx] = (0.0, None)

    meta = {
        "vmax_idx": vmax_idx,
        "e_start": e_start,
        "target_indices": targets,
        "variable_count": nvar,
        "constraint_count": r,
    }
    return c, A_ub, b_ub, bounds, meta


def solve_lp(c: np.ndarray, A_ub: csr_matrix, b_ub: np.ndarray,
             bounds: Sequence[Tuple[Optional[float], Optional[float]]],
             time_limit: float, label: str):
    options = {"presolve": True}
    if time_limit > 0.0:
        options["time_limit"] = float(time_limit)
    started = time.perf_counter()
    result = linprog(c=c, A_ub=A_ub, b_ub=b_ub,
                     bounds=list(bounds), method="highs", options=options)
    elapsed = time.perf_counter() - started
    if not result.success:
        raise RuntimeError(
            f"{label} failed: status={result.status}, message={result.message}, elapsed={elapsed:.3f}s"
        )
    return result, elapsed


# ---------------------------------------------------------------------------
# Convert independent segment durations back to G-code parameters
# ---------------------------------------------------------------------------

def solution_to_rows(
    rows: Sequence[Row], controls: Sequence[SegmentControl], z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    F_new = np.asarray([r.F for r in rows], dtype=float)
    P_new_ms = np.asarray([r.t_s * 1000.0 if r.cmd == "G4" else 0.0 for r in rows], dtype=float)
    t_new = np.asarray([r.t_s for r in rows], dtype=float)
    row_scale = np.ones(len(rows), dtype=float)
    durations = np.diff(z)
    if len(durations) != len(controls):
        raise ValueError("Optimized duration vector length does not match segment-control count")

    for control in controls:
        row = rows[control.row_i]
        new_t = float(durations[control.index])
        if not math.isfinite(new_t) or new_t < -1e-9:
            raise ValueError(f"seg_idx={row.seg_idx} has invalid optimized duration={new_t}")
        new_t = max(0.0, new_t)
        scale = new_t / max(row.t_s, EPS)
        row_scale[control.row_i] = scale
        t_new[control.row_i] = new_t
        if control.kind == "dwell":
            P_new_ms[control.row_i] = new_t * 1000.0
        else:
            F0 = infer_feedrate(row)
            if scale > EPS and F0 > EPS:
                F_new[control.row_i] = F0 / scale
    return F_new, P_new_ms, t_new, row_scale


# ---------------------------------------------------------------------------
# G-code patching / TRUE validation
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
    out: List[str] = []
    replaced = False
    for token in parts:
        if token.upper().startswith(letter):
            out.append(f"{letter}{value}")
            replaced = True
        else:
            out.append(token)
    if not replaced:
        out.append(f"{letter}{value}")
    return out


def patch_gcode_by_segidx(
    gcode_in: str,
    gcode_out: str,
    rows: Sequence[Row],
    F_new: np.ndarray,
    P_new_ms: np.ndarray,
    enable_travel_opt: bool,
    enable_dwell_opt: bool,
) -> Tuple[int, int]:
    seg_to_i = {r.seg_idx: i for i, r in enumerate(rows)}
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
            parts = code.strip().split()
            cmd = gcode_cmd_token(code)
            has_e = any(t.upper().startswith("E") for t in parts)
            has_p = any(t.upper().startswith("P") for t in parts)
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
                parts = replace_or_append_parameter(parts, "F", f"{float(F_new[i]):.5f}")
            elif mode == "P":
                parts = replace_or_append_parameter(parts, "P", str(int(round(float(P_new_ms[i])))))
            else:
                fout.write(line)
                continue
            new_code = " ".join(parts)
            fout.write(f"{new_code} ; {comment.strip()}\n" if sep and comment.strip() else f"{new_code}\n")
            patched += 1
    return patched, expected


def run_true_validation(module_name: str, gcode_in: str, out_dir: str,
                        V_nozzle: float, d_filament: float, e_mode: str) -> Tuple[str, str]:
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


def scored_tau_from_csv(path: str) -> np.ndarray:
    rows, _ = read_segments_csv(path)
    tau = np.asarray([r.tau_s for r in rows if is_scored_extrusion_row(r) and math.isfinite(r.tau_s)], dtype=float)
    if tau.size == 0:
        raise ValueError(f"No finite scored residence times were found in {path}")
    return tau


def compare_true_validation(validation_csv: str, equations: Sequence[ResidenceEquation], tau_pred: np.ndarray) -> Dict[str, float]:
    df = pd.read_csv(validation_csv)
    if "seg_idx" not in df.columns or "tau_s" not in df.columns:
        raise ValueError("TRUE validation CSV lacks seg_idx or tau_s")
    true_map = {
        int(rec["seg_idx"]): float(rec["tau_s"])
        for _, rec in df.iterrows()
        if not pd.isna(rec["seg_idx"]) and not pd.isna(rec["tau_s"])
    }
    pred: List[float] = []
    truth: List[float] = []
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


def compare_metric_priority(candidate: Dict[str, float], current: Dict[str, float], tolerance_s: float) -> Tuple[bool, str]:
    tol = max(0.0, float(tolerance_s))
    if candidate["vmax_s"] < current["vmax_s"] - tol:
        return True, "lower_true_vmax"
    if candidate["vmax_s"] > current["vmax_s"] + tol:
        return False, "higher_true_vmax"
    if candidate["violation_count"] < current["violation_count"]:
        return True, "fewer_true_violations"
    if candidate["violation_count"] > current["violation_count"]:
        return False, "more_true_violations"
    if candidate["violation_sum_s"] < current["violation_sum_s"] - tol:
        return True, "lower_true_violation_sum"
    if candidate["violation_sum_s"] > current["violation_sum_s"] + tol:
        return False, "higher_true_violation_sum"
    if candidate["mean_abs_target_error_s"] < current["mean_abs_target_error_s"] - tol:
        return True, "lower_true_target_error"
    return False, "no_material_true_improvement"


def write_outputs(
    out_dir: str,
    source_df: pd.DataFrame,
    rows: Sequence[Row],
    controls: Sequence[SegmentControl],
    row_to_control: np.ndarray,
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
    output_df["segment_control"] = row_to_control
    output_df["time_scale_diagnostic"] = row_scale
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

    durations = np.diff(z)
    records: List[Dict[str, object]] = []
    for control in controls:
        row = rows[control.row_i]
        new_t = float(durations[control.index])
        records.append({
            "control": control.index,
            "seg_idx": control.seg_idx,
            "row_i": control.row_i,
            "kind": control.kind,
            "base_duration_s": control.base_duration,
            "optimized_duration_s": new_t,
            "duration_lb_s": control.duration_lb,
            "duration_ub_s": control.duration_ub,
            "time_scale_diagnostic": new_t / max(control.base_duration, EPS),
            "base_F_mm_per_min": control.base_F,
            "optimized_F_mm_per_min": F_new[control.row_i],
            "allowed_Fmin_mm_per_min": control.allowed_f_min,
            "allowed_Fmax_mm_per_min": control.allowed_f_max,
            "P_optimized_ms": P_new_ms[control.row_i] if control.kind == "dwell" else math.nan,
        })
    control_path = os.path.join(out_dir, "segment_controls.csv")
    pd.DataFrame(records).to_csv(control_path, index=False, float_format="%.8f")
    return prediction_path, control_path


# ---------------------------------------------------------------------------
# One full-freedom LP round
# ---------------------------------------------------------------------------

def optimize_one_round(
    *, args: argparse.Namespace, input_csv: str, gcode_in: Optional[str], round_dir: str,
    tau_lo: float, tau_star: float, tau_hi: float,
    tau_grid: Optional[np.ndarray], alpha_grid: Optional[np.ndarray], run_true: bool,
) -> Dict[str, object]:
    os.makedirs(round_dir, exist_ok=True)
    rows, source_df = read_segments_csv(input_csv)
    controls, row_to_control = build_segment_controls(
        rows,
        args.enable_travel_opt,
        args.enable_dwell_opt,
        args.Fmin_extrusion_resolved,
        args.Fmax_extrusion_resolved,
        args.Fmin_travel_resolved,
        args.Fmax_travel_resolved,
        args.Pmin_ms,
        args.Pmax_ms,
        args.max_relative_F_change,
        args.max_relative_P_change,
    )
    equations, model_diag = build_residence_equations(rows, controls)
    z0 = baseline_z(controls)
    baseline_model = evaluate_equations(equations, z0)
    baseline_csv = np.asarray([eq.baseline_tau for eq in equations], dtype=float)
    reconstruction_abs_max = float(np.max(np.abs(baseline_model - baseline_csv)))
    baseline_metrics = residence_metrics(baseline_csv, tau_lo, tau_star, tau_hi)

    print(f"  Loaded {len(rows):,} rows")
    print(f"  Independent adjustable segments: {len(controls):,}")
    print(f"  Residence equations: {len(equations):,}")
    print(
        f"  TRUE baseline: within={baseline_metrics['within_percent']:.3f}% "
        f"violations={baseline_metrics['violation_count']:,} vmax={baseline_metrics['vmax_s']:.6f}s"
    )

    c1, A1, b1, bounds1, vmax_idx1 = build_stage1_problem(equations, controls, tau_lo, tau_hi)
    print(
        f"  Stage 1 free-segment LP: variables={A1.shape[1]:,}, constraints={A1.shape[0]:,}, "
        f"nonzeros={A1.nnz:,}"
    )
    stage1, stage1_sec = solve_lp(c1, A1, b1, bounds1, args.solver_time_limit, "Stage 1 free-segment minimax")
    vmax_star = max(0.0, float(stage1.x[vmax_idx1]))
    S = len(controls)
    z_stage1 = np.asarray(stage1.x[:S + 1], dtype=float)
    np.savez_compressed(
        os.path.join(round_dir, "stage1_checkpoint.npz"),
        z=z_stage1,
        minimal_worst_violation_s=vmax_star,
    )
    print(
        f"  Stage 1 complete in {stage1_sec:.3f}s; minimal worst violation={vmax_star:.6f}s; "
        f"predicted_window_feasible={vmax_star <= args.feasibility_tolerance_s}"
    )
    del c1, A1, b1, bounds1
    gc.collect()

    z_opt = z_stage1
    solution_source = "stage1"
    stage2_sec = 0.0
    stage2_message = "stage1_only"
    target_sample_count = 0
    vmax_upper = vmax_star + max(0.0, args.vmax_tolerance_s)

    if not args.stage1_only:
        try:
            c2, A2, b2, bounds2, meta2 = build_stage2_problem(
                equations, controls, tau_lo, tau_star, tau_hi,
                vmax_upper, args.max_target_samples,
                args.target_weight, args.time_weight,
            )
            target_sample_count = len(meta2["target_indices"])
            print(
                f"  Stage 2 free-segment LP: variables={A2.shape[1]:,}, constraints={A2.shape[0]:,}, "
                f"nonzeros={A2.nnz:,}, target_samples={target_sample_count:,}"
            )
            stage2, stage2_sec = solve_lp(c2, A2, b2, bounds2, args.solver_time_limit, "Stage 2 free-segment target")
            z_opt = np.asarray(stage2.x[:S + 1], dtype=float)
            solution_source = "stage2"
            stage2_message = str(stage2.message)
            del c2, A2, b2, bounds2
            gc.collect()
        except Exception as exc:
            if args.no_stage2_fallback:
                raise
            stage2_message = f"fallback_after_error: {exc}"
            print(f"  Warning: Stage 2 failed; using Stage 1 solution: {exc}")

    tau_pred = evaluate_equations(equations, z_opt)
    pred_metrics = residence_metrics(tau_pred, tau_lo, tau_star, tau_hi)
    print(
        f"  Predicted candidate: within={pred_metrics['within_percent']:.3f}% "
        f"violations={pred_metrics['violation_count']:,} vmax={pred_metrics['vmax_s']:.6f}s"
    )

    F_new, P_new_ms, t_new, row_scale = solution_to_rows(rows, controls, z_opt)
    alpha_pred = alpha_from_tau(tau_pred, tau_grid, alpha_grid) if tau_grid is not None else None
    prediction_path, control_path = write_outputs(
        round_dir, source_df, rows, controls, row_to_control, equations, tau_pred, alpha_pred,
        F_new, P_new_ms, t_new, row_scale, z_opt,
    )

    candidate_gcode = None
    patch_info = None
    if gcode_in:
        candidate_gcode = os.path.join(round_dir, "candidate_optimized.gcode")
        patched, expected = patch_gcode_by_segidx(
            gcode_in, candidate_gcode, rows, F_new, P_new_ms,
            args.enable_travel_opt, args.enable_dwell_opt,
        )
        coverage = patched / max(1, expected)
        patch_info = {"patched": patched, "expected": expected, "coverage": coverage}
        print(f"  Patched G-code: {patched}/{expected} ({coverage:.6f})")
        if expected > 0 and coverage < 0.98:
            raise RuntimeError("G-code patch coverage is below 98%; TRUE result would be unreliable")

    true_csv = None
    true_gcode = None
    true_metrics = None
    true_comparison = None
    if run_true:
        if candidate_gcode is None:
            raise ValueError("TRUE validation requires --gcode_in")
        true_csv, true_gcode = run_true_validation(
            args.true_builder_module, candidate_gcode, round_dir,
            args.vnozzle, args.d_filament, args.e_mode,
        )
        true_comparison = compare_true_validation(true_csv, equations, tau_pred)
        true_metrics = residence_metrics(scored_tau_from_csv(true_csv), tau_lo, tau_star, tau_hi)
        print(
            f"  TRUE candidate: within={true_metrics['within_percent']:.3f}% "
            f"violations={true_metrics['violation_count']:,} vmax={true_metrics['vmax_s']:.6f}s; "
            f"prediction_RMSE={true_comparison['true_minus_pred_rmse_s']:.6f}s"
        )

    durations = np.diff(z_opt)
    summary = {
        "input_csv": os.path.abspath(input_csv),
        "gcode_in": os.path.abspath(gcode_in) if gcode_in else None,
        "row_count": len(rows),
        "independent_segment_control_count": len(controls),
        "model": model_diag,
        "baseline_reconstruction_abs_max_s": reconstruction_abs_max,
        "baseline_metrics": baseline_metrics,
        "stage1": {
            "elapsed_s": stage1_sec,
            "minimal_worst_violation_s": vmax_star,
            "predicted_full_window_feasible": bool(vmax_star <= args.feasibility_tolerance_s),
            "feasibility_tolerance_s": args.feasibility_tolerance_s,
            "solver_message": str(stage1.message),
        },
        "stage2": {
            "enabled": not args.stage1_only,
            "elapsed_s": stage2_sec,
            "allowed_worst_violation_s": vmax_upper,
            "solver_message": stage2_message,
            "target_sample_count": target_sample_count,
            "solution_source": solution_source,
        },
        "optimized_segment_duration_statistics": {
            "min_s": float(np.min(durations)),
            "max_s": float(np.max(durations)),
            "mean_s": float(np.mean(durations)),
        },
        "predicted_metrics": pred_metrics,
        "patch": patch_info,
        "true_metrics": true_metrics,
        "true_validation": true_comparison,
        "files": {
            "prediction_csv": os.path.abspath(prediction_path),
            "segment_controls_csv": os.path.abspath(control_path),
            "candidate_gcode": os.path.abspath(candidate_gcode) if candidate_gcode else None,
            "true_csv": os.path.abspath(true_csv) if true_csv else None,
            "true_gcode": os.path.abspath(true_gcode) if true_gcode else None,
        },
    }
    with open(os.path.join(round_dir, "round_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "baseline_metrics": baseline_metrics,
        "predicted_metrics": pred_metrics,
        "true_metrics": true_metrics,
        "true_comparison": true_comparison,
        "stage1_vmax": vmax_star,
        "predicted_full_window_feasible": bool(vmax_star <= args.feasibility_tolerance_s),
        "candidate_gcode": candidate_gcode,
        "true_csv": true_csv,
        "true_gcode": true_gcode,
        "round_summary": summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Free independent-segment FIFO residence-time LP optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Segmented CSV containing baseline TRUE FIFO timing")
    ap.add_argument("--gcode_in", default=None, help="Segmented G-code with seg_idx comments")
    ap.add_argument("--gcode_out", default=None, help="Final reconstructed G-code output")
    ap.add_argument("--out_dir", default="free_segment_optimization")

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

    ap.add_argument("--Fmin_extrusion", type=float, default=None)
    ap.add_argument("--Fmax_extrusion", type=float, default=None)
    ap.add_argument("--Fmin_travel", type=float, default=None)
    ap.add_argument("--Fmax_travel", type=float, default=None)
    ap.add_argument("--Fmin", type=float, default=None, help="Legacy fallback minimum for both movement kinds")
    ap.add_argument("--Fmax", type=float, default=None, help="Legacy fallback maximum for both movement kinds")
    ap.add_argument("--Pmin_ms", type=float, default=0.0)
    ap.add_argument("--Pmax_ms", type=float, default=600000.0)
    ap.add_argument("--max_relative_F_change", type=float, default=0.0,
                    help="Optional per-round relative F limit; 0 means fully free within absolute F bounds")
    ap.add_argument("--max_relative_P_change", type=float, default=0.0,
                    help="Optional per-round relative dwell limit; 0 means fully free within P bounds")
    ap.add_argument("--enable_travel_opt", action="store_true")
    ap.add_argument("--enable_dwell_opt", action="store_true")

    ap.add_argument("--stage1_only", action="store_true",
                    help="Run only the full-freedom minimax feasibility LP")
    ap.add_argument("--max_target_samples", type=int, default=5000,
                    help="Stage-2 residence rows used for ideal-target error; all rows remain constrained")
    ap.add_argument("--target_weight", type=float, default=1.0)
    ap.add_argument("--time_weight", type=float, default=0.0,
                    help="Optional preference for shorter total adjustable execution time")
    ap.add_argument("--vmax_tolerance_s", type=float, default=0.01,
                    help="Stage-2 allowance above the Stage-1 minimal worst violation")
    ap.add_argument("--feasibility_tolerance_s", type=float, default=1e-6,
                    help="Stage-1 vmax at or below this value is reported as predicted fully feasible")
    ap.add_argument("--solver_time_limit", type=float, default=0.0,
                    help="Seconds per LP; 0 means no explicit solver limit")
    ap.add_argument("--no_stage2_fallback", action="store_true")

    ap.add_argument("--iterations", type=int, default=1,
                    help="Full-freedom LP/TRUE-FIFO relinearization rounds")
    ap.add_argument("--accept_tolerance_s", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=2,
                    help="Stop after this many consecutive rejected TRUE candidates")

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
    return (
        tau_from_alpha(alpha_min, tau_grid, alpha_grid),
        tau_from_alpha(alpha_ideal, tau_grid, alpha_grid),
        tau_from_alpha(alpha_max, tau_grid, alpha_grid),
        tau_grid,
        alpha_grid,
    )


def resolve_feedrate_bounds(args: argparse.Namespace) -> Tuple[float, float, float, float]:
    fmin_e = float(args.Fmin_extrusion) if args.Fmin_extrusion is not None else (
        float(args.Fmin) if args.Fmin is not None else 180.0
    )
    fmax_e = float(args.Fmax_extrusion) if args.Fmax_extrusion is not None else (
        float(args.Fmax) if args.Fmax is not None else 3000.0
    )
    fmin_t = float(args.Fmin_travel) if args.Fmin_travel is not None else (
        float(args.Fmin) if args.Fmin is not None else fmin_e
    )
    fmax_t = float(args.Fmax_travel) if args.Fmax_travel is not None else (
        float(args.Fmax) if args.Fmax is not None else fmax_e
    )
    return fmin_e, fmax_e, fmin_t, fmax_t


def validate_args(args: argparse.Namespace) -> None:
    (args.Fmin_extrusion_resolved, args.Fmax_extrusion_resolved,
     args.Fmin_travel_resolved, args.Fmax_travel_resolved) = resolve_feedrate_bounds(args)
    if not (0 < args.Fmin_extrusion_resolved < args.Fmax_extrusion_resolved):
        raise ValueError("Require 0 < Fmin_extrusion < Fmax_extrusion")
    if not (0 < args.Fmin_travel_resolved < args.Fmax_travel_resolved):
        raise ValueError("Require 0 < Fmin_travel < Fmax_travel")
    if not (0 <= args.Pmin_ms < args.Pmax_ms):
        raise ValueError("Require 0 <= Pmin_ms < Pmax_ms")
    if not (0.0 <= args.max_relative_F_change < 1.0):
        raise ValueError("max_relative_F_change must satisfy 0 <= value < 1")
    if args.max_relative_P_change < 0.0:
        raise ValueError("max_relative_P_change must be >= 0")
    if args.max_target_samples < 0:
        raise ValueError("max_target_samples must be >= 0")
    if args.target_weight < 0 or args.time_weight < 0:
        raise ValueError("target_weight and time_weight must be >= 0")
    if args.vmax_tolerance_s < 0 or args.feasibility_tolerance_s < 0:
        raise ValueError("vmax/feasibility tolerances must be >= 0")
    if args.iterations < 1:
        raise ValueError("iterations must be >= 1")
    if args.patience < 1:
        raise ValueError("patience must be >= 1")
    if args.accept_tolerance_s < 0:
        raise ValueError("accept_tolerance_s must be >= 0")
    if args.iterations > 1 and not args.gcode_in:
        raise ValueError("--iterations > 1 requires --gcode_in")
    if args.true_validate and not args.gcode_in:
        raise ValueError("--true_validate requires --gcode_in")


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    try:
        validate_args(args)
        os.makedirs(args.out_dir, exist_ok=True)
        tau_lo, tau_star, tau_hi, tau_grid, alpha_grid = resolve_tau_targets(args)

        print("=" * 78)
        print("Free independent-segment residence-time LP optimizer")
        print("Original relative speed distribution: NOT preserved")
        print("Grouping: disabled; one independent control per adjustable segment")
        print(f"Target tau window: [{tau_lo:.6f}, {tau_hi:.6f}] s; ideal={tau_star:.6f} s")
        print(
            "Feed-rate bounds: "
            f"extrusion=[{args.Fmin_extrusion_resolved:.6g}, {args.Fmax_extrusion_resolved:.6g}] mm/min; "
            f"travel=[{args.Fmin_travel_resolved:.6g}, {args.Fmax_travel_resolved:.6g}] mm/min"
        )

        original_csv = os.path.abspath(args.input)
        current_csv = original_csv
        current_gcode = os.path.abspath(args.gcode_in) if args.gcode_in else None
        initial_metrics = residence_metrics(scored_tau_from_csv(current_csv), tau_lo, tau_star, tau_hi)
        current_metrics = dict(initial_metrics)
        run_true_each_round = bool(args.true_validate or args.iterations > 1)
        accepted_rounds = 0
        consecutive_rejections = 0
        history: List[Dict[str, object]] = []
        first_stage1_vmax: Optional[float] = None
        first_predicted_feasible: Optional[bool] = None

        for attempt in range(1, args.iterations + 1):
            print("-" * 78)
            print(
                f"Iteration {attempt}/{args.iterations}; current TRUE within={current_metrics['within_percent']:.3f}% "
                f"vmax={current_metrics['vmax_s']:.6f}s"
            )
            round_dir = os.path.join(args.out_dir, f"iteration_{attempt:03d}")
            result = optimize_one_round(
                args=args,
                input_csv=current_csv,
                gcode_in=current_gcode,
                round_dir=round_dir,
                tau_lo=tau_lo,
                tau_star=tau_star,
                tau_hi=tau_hi,
                tau_grid=tau_grid,
                alpha_grid=alpha_grid,
                run_true=run_true_each_round,
            )
            if first_stage1_vmax is None:
                first_stage1_vmax = float(result["stage1_vmax"])
                first_predicted_feasible = bool(result["predicted_full_window_feasible"])

            if run_true_each_round:
                candidate_metrics = result["true_metrics"]
                assert isinstance(candidate_metrics, dict)
                accepted, reason = compare_metric_priority(candidate_metrics, current_metrics, args.accept_tolerance_s)
            else:
                candidate_metrics = result["predicted_metrics"]
                accepted, reason = True, "single_round_prediction"

            history.append({
                "attempt": attempt,
                "accepted": accepted,
                "acceptance_reason": reason,
                "stage1_vmax_s": result["stage1_vmax"],
                "predicted_full_window_feasible": result["predicted_full_window_feasible"],
                "baseline_metrics": result["baseline_metrics"],
                "predicted_metrics": result["predicted_metrics"],
                "true_metrics": result["true_metrics"],
                "true_validation": result["true_comparison"],
                "round_dir": os.path.abspath(round_dir),
            })

            if accepted:
                accepted_rounds += 1
                consecutive_rejections = 0
                current_metrics = dict(candidate_metrics)
                if run_true_each_round:
                    assert result["true_csv"] and result["true_gcode"]
                    current_csv = str(result["true_csv"])
                    current_gcode = str(result["true_gcode"])
                elif result["candidate_gcode"]:
                    current_gcode = str(result["candidate_gcode"])
                print(f"  ACCEPTED: {reason}")
            else:
                consecutive_rejections += 1
                print(f"  REJECTED: {reason}; keeping previous authoritative baseline")
                if consecutive_rejections >= args.patience:
                    print("  Stopping: rejection patience reached")
                    break

        history_path = os.path.join(args.out_dir, "optimization_history.csv")
        flat: List[Dict[str, object]] = []
        for rec in history:
            base = rec["baseline_metrics"] or {}
            pred = rec["predicted_metrics"] or {}
            true = rec["true_metrics"] or {}
            comp = rec["true_validation"] or {}
            flat.append({
                "attempt": rec["attempt"],
                "accepted": rec["accepted"],
                "acceptance_reason": rec["acceptance_reason"],
                "stage1_vmax_s": rec["stage1_vmax_s"],
                "predicted_full_window_feasible": rec["predicted_full_window_feasible"],
                "baseline_within_percent": base.get("within_percent"),
                "baseline_vmax_s": base.get("vmax_s"),
                "predicted_within_percent": pred.get("within_percent"),
                "predicted_vmax_s": pred.get("vmax_s"),
                "true_within_percent": true.get("within_percent"),
                "true_vmax_s": true.get("vmax_s"),
                "true_violation_count": true.get("violation_count"),
                "true_mean_abs_target_error_s": true.get("mean_abs_target_error_s"),
                "prediction_rmse_s": comp.get("true_minus_pred_rmse_s"),
                "round_dir": rec["round_dir"],
            })
        pd.DataFrame(flat).to_csv(history_path, index=False, float_format="%.8f")

        final_csv_copy = os.path.join(args.out_dir, "final_reference.csv")
        if os.path.abspath(current_csv) != os.path.abspath(final_csv_copy):
            shutil.copy2(current_csv, final_csv_copy)
        requested_gcode_out = args.gcode_out or os.path.join(args.out_dir, "free_segment_optimized.gcode")
        if current_gcode:
            os.makedirs(os.path.dirname(os.path.abspath(requested_gcode_out)), exist_ok=True)
            if os.path.abspath(current_gcode) != os.path.abspath(requested_gcode_out):
                shutil.copy2(current_gcode, requested_gcode_out)

        summary = {
            "input": original_csv,
            "gcode_in": os.path.abspath(args.gcode_in) if args.gcode_in else None,
            "gcode_out": os.path.abspath(requested_gcode_out) if current_gcode else None,
            "mode": {
                "optimization_unit": "independent_adjustable_segment",
                "preserve_original_speed_distribution": False,
                "control_grouping": False,
                "conceptual_decision_variable": "segment_execution_duration",
                "sparse_implementation": "cumulative_z_with_independent_segment_differences",
                "true_fifo_relinearization": run_true_each_round,
                "iterations_requested": args.iterations,
                "attempts_completed": len(history),
                "accepted_rounds": accepted_rounds,
            },
            "target": {
                "tau_min": tau_lo,
                "tau_ideal": tau_star,
                "tau_max": tau_hi,
                "target_mode": args.target_mode,
            },
            "feedrate_bounds": {
                "extrusion": [args.Fmin_extrusion_resolved, args.Fmax_extrusion_resolved],
                "travel": [args.Fmin_travel_resolved, args.Fmax_travel_resolved],
                "travel_enabled": bool(args.enable_travel_opt),
            },
            "theoretical_free_segment_feasibility_first_round": {
                "minimal_worst_violation_s": first_stage1_vmax,
                "predicted_full_window_feasible": first_predicted_feasible,
                "interpretation": (
                    "If true, the fixed-event linear model found a legal independent-segment timing profile "
                    "placing every scored residence row inside the target window. TRUE FIFO validation is separate."
                ),
            },
            "initial_metrics": initial_metrics,
            "final_metrics": current_metrics,
            "history": history,
            "outputs": {
                "final_reference_csv": os.path.abspath(final_csv_copy),
                "final_gcode": os.path.abspath(requested_gcode_out) if current_gcode else None,
                "history_csv": os.path.abspath(history_path),
            },
            "total_elapsed_s": float(time.perf_counter() - started),
        }
        summary_path = os.path.join(args.out_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("=" * 78)
        print(
            f"Final: within={current_metrics['within_percent']:.3f}% "
            f"violations={current_metrics['violation_count']:,} vmax={current_metrics['vmax_s']:.6f}s"
        )
        print(f"First-round theoretical free-segment feasibility: {first_predicted_feasible}")
        print(f"First-round minimal worst violation: {first_stage1_vmax:.6f}s")
        print(f"Final G-code: {requested_gcode_out if current_gcode else 'not generated'}")
        print(f"History CSV: {history_path}")
        print(f"Summary JSON: {summary_path}")
        print(f"Total elapsed: {time.perf_counter() - started:.3f}s")
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
