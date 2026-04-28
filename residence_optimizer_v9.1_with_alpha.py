"""
residence_optimizer_v9.1.py
- give alpha boundary to get residence time first
- use residence time for getting error
--------------------------------------------------------------------
CLI example:
python residence_optimizer_v8.1.py 
--input house_withoutdoor_xy30h5_940s_F800_rebuilt_segmented.csv 
--tau 940 --buffer 0.05 --vnozzle 723 --dnozzle 0.84 
--Kp 0.6 --Kd 0.1 --Ki 0.2 --Fmin 350 --Fmax 1200 
--dFmax 0.5 --deltaT_cap_ratio 0.3 --smooth_passes 1 
--max_iter 20 --out_dir optimization_outputs --out_prefix iteration_ 
--summary house_withoutdoor_xy30h5_940s_F800_optimized_summary.csv 
--enable_timer --use_boundary_window 
--gcode_in house_withoutdoor_xy30h5_940s_F800_rebuilt_segmented.gcode --gcode_out house_withoutdoor_xy30h5_940s_segmented_optimized_final.gcode 
--fifo_mode true --true_every 1 --save_intermediate 
--window_cap 1200 --horizon_sec 240 --horizon_mode past 
--enable_travel_opt --travel_gain 0.2 --travel_step_max 0.02 --travel_s_min 0.2 --travel_dFmax 0.05 
--window_direction backward

SingleLine Command:
residence_optimizer_v9.1_with_alpha.py --input CFFFP_testmodel_LH0.3_774s_Temp32_F0.2_T0.8_E0.95_Z+1_segmented.csv --temp_c 32.0 --alpha_min 0.916386 --alpha_ideal 0.929835 --alpha_max 0.942497 --vnozzle 723 --dnozzle 0.58 --Fmin 250 --Fmax 1200 --dFmax 0.4 --deltaT_cap_ratio 0.4 --smooth_passes 1 --out_dir optimization_outputs --out_prefix iteration_ --summary CFFFP_testmodel_LH0.3_774s_Temp32_F0.2_T0.8_E0.95_Z+1_optimized_summary.csv --enable_timer --gcode_in CFFFP_testmodel_LH0.3_774s_Temp32_F0.2_T0.8_E0.95_Z+1_segmented.gcode --gcode_out CFFFP_testmodel_LH0.3_774s_Temp32_F0.2_T0.8_E0.95_Z+1_optimized_final.gcode --fifo_mode true --true_every 1 --save_intermediate --window_cap 1200 --horizon_sec 240 --horizon_mode past --window_direction backward --max_iter 100 --Kp 0.2 --Ki 0.05 --Kd 0.1

"""


import csv
import math
import time
import argparse
import os
from typing import List, Dict, Tuple
from rebuild_csv_and_gcode_fifo_v1_1 import rebuild_global as true_fifo_rebuild
import pandas as pd
import re
import shutil
import sys
from collections import deque
import numpy as np
from scipy.integrate import solve_ivp


EPS = 1e-12
TAU_TOL = 0.01   # engineering tolerance in seconds
# ---------- Alpha model (fixed-T, alpha_in = 0) ----------

R_GAS = 8.314  # J/(mol*K)

# TODO:
# Replace these placeholders with your FINAL RTV-2 kinetic parameters
ALPHA_MODEL_PARAMS = {
    "A1": 1.00e15,      # s^-1
    "E1": 103.97e3,     # J/mol
    "A2": 1.96e12,      # s^-1
    "E2": 218.55e3,     # J/mol
    "m": 0.713,
    "n": 0.323
}

ALPHA_IN_DEFAULT = 0.0

def tau_to_const(t):
    """
    Convert tau to integer constant used for within judgement.
    rule:
    tau -> round to 1 decimal -> round to integer
    """
    if t is None or math.isnan(t):
        return None
    
    t1 = round(t, 1)
    return int(round(t1))

def celsius_to_kelvin(temp_c: float) -> float:
    return float(temp_c) + 273.15


def arrhenius_rate(A: float, E: float, temp_k: float) -> float:
    return float(A) * math.exp(-float(E) / (R_GAS * float(temp_k)))


def alpha_rhs_isothermal(t: float, y: np.ndarray, temp_c: float, params: dict):
    alpha = float(y[0])
    alpha = min(max(alpha, 1e-12), 1.0)

    T_k = celsius_to_kelvin(temp_c)
    k1 = arrhenius_rate(params["A1"], params["E1"], T_k)
    k2 = arrhenius_rate(params["A2"], params["E2"], T_k)
    m = float(params["m"])
    n = float(params["n"])

    dadt = (k1 + k2 * (alpha ** m)) * ((1.0 - alpha) ** n)
    return [dadt]


def solve_alpha_isothermal(temp_c: float,
                           residence_time_s: float,
                           params: dict,
                           alpha_in: float = ALPHA_IN_DEFAULT,
                           n_eval: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    if residence_time_s < 0:
        raise ValueError("residence_time_s must be >= 0")

    if residence_time_s == 0:
        return np.array([0.0]), np.array([alpha_in], dtype=float)

    t_eval = np.linspace(0.0, residence_time_s, n_eval)

    sol = solve_ivp(
        fun=lambda t, y: alpha_rhs_isothermal(t, y, temp_c, params),
        t_span=(0.0, residence_time_s),
        y0=[alpha_in],
        t_eval=t_eval,
        method="BDF",
        rtol=1e-7,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError(f"alpha ODE solver failed: {sol.message}")

    alpha = np.clip(sol.y[0], 0.0, 1.0)
    alpha = np.maximum.accumulate(alpha)  # enforce monotone
    return sol.t, alpha


def build_alpha_tau_lookup(temp_c: float,
                           params: dict,
                           tau_upper_s: float,
                           alpha_in: float = ALPHA_IN_DEFAULT,
                           n_eval: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    tau_upper_s = max(1.0, float(tau_upper_s))
    n_eval = max(1000, int(n_eval))
    tau_grid, alpha_grid = solve_alpha_isothermal(
        temp_c=temp_c,
        residence_time_s=tau_upper_s,
        params=params,
        alpha_in=alpha_in,
        n_eval=n_eval
    )
    alpha_grid = np.clip(alpha_grid, 0.0, 1.0)
    alpha_grid = np.maximum.accumulate(alpha_grid)
    return tau_grid, alpha_grid


def alpha_from_tau_lookup(tau_s: float,
                          tau_grid: np.ndarray,
                          alpha_grid: np.ndarray) -> float:
    if tau_s is None or math.isnan(tau_s):
        return float("nan")
    tau_s = float(tau_s)
    if tau_s <= tau_grid[0]:
        return float(alpha_grid[0])
    if tau_s >= tau_grid[-1]:
        return float(alpha_grid[-1])
    return float(np.interp(tau_s, tau_grid, alpha_grid))


def alpha_list_from_tau_list(tau_list: List[float],
                             tau_grid: np.ndarray,
                             alpha_grid: np.ndarray) -> List[float]:
    out = []
    for t in tau_list:
        out.append(alpha_from_tau_lookup(t, tau_grid, alpha_grid))
    return out


def tau_from_alpha_lookup(alpha_target: float,
                          tau_grid: np.ndarray,
                          alpha_grid: np.ndarray) -> float:
    alpha_target = float(alpha_target)

    if alpha_target <= alpha_grid[0]:
        return float(tau_grid[0])

    if alpha_target >= alpha_grid[-1]:
        raise ValueError(
            f"Target alpha={alpha_target:.5f} is above lookup max alpha={alpha_grid[-1]:.5f}. "
            f"Increase --tau_lookup_max or check temperature / kinetics."
        )

    alpha_unique, unique_idx = np.unique(alpha_grid, return_index=True)
    tau_unique = tau_grid[unique_idx]
    return float(np.interp(alpha_target, alpha_unique, tau_unique))


def append_alpha_column_to_segment_csv(csv_path: str,
                                       tau_grid: np.ndarray,
                                       alpha_grid: np.ndarray,
                                       tau_col: str = "tau_s",
                                       alpha_col: str = "alpha") -> None:
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)

    if tau_col not in df.columns:
        return

    alpha_vals = []
    for v in df[tau_col].tolist():
        try:
            vv = float(v)
            if math.isnan(vv):
                alpha_vals.append(float("nan"))
            else:
                alpha_vals.append(alpha_from_tau_lookup(vv, tau_grid, alpha_grid))
        except:
            alpha_vals.append(float("nan"))

    df[alpha_col] = alpha_vals
    df.to_csv(csv_path, index=False, float_format="%.5f")
    
# ---------- Data structures ----------

class Row:
    """
    Lightweight container for a single segment row from the CSV.

    The class parses a CSV dictionary into typed attributes and derives
    convenience flags for extrusion and optimization status based on the
    'note' field.

    Attributes:
        seg_idx (int): Segment index from the CSV.
        cmd (str): G-code command (e.g., "G1", "G0", "G4").
        d_mm (float): Segment length in millimeters.
        F (float): Feedrate in mm/min.
        V_mm3 (float): Extrusion volume in cubic millimeters.
        t_s (float): Segment duration in seconds.
        t_in (float): FIFO input time for this segment.
        t_out (float): FIFO output time for this segment.
        tau_s (float): Residence time (t_out - t_in).
        note (str): Annotation string from the CSV.
        is_extrusion (bool): True if `note` contains "extrusion".
        is_optimizable (bool): True if `note` contains "optimizable".
    """
    __slots__ = (
        "seg_idx","cmd","d_mm","F","V_mm3",
        "t_s","t_in","t_out","tau_s","note",
        "is_extrusion","is_optimizable"
    )
    def __init__(self, d: Dict[str,str]):
        self.seg_idx = int(d["seg_idx"])
        self.cmd = d["cmd"].strip()
        self.d_mm = float(d.get("d_mm","") or 0.0)
        self.F = float(d.get("F_mm_per_min","") or 0.0)
        self.V_mm3 = float(d.get("V_mm3","") or 0.0)
        self.t_s = float(d.get("t_s","") or 0.0)
        self.t_in = float(d.get("t_in","") or 0.0)
        self.t_out = float(d.get("t_out","") or 0.0)
        self.tau_s = float(d.get("tau_s","") or 0.0)
        self.note = d.get("note","").strip()
        note_l = self.note.lower()
        self.is_extrusion = ("extrusion" in note_l)
        self.is_optimizable = ("optimizable" in note_l)

# ---------- Helpers ----------
def area_circle(d_mm: float) -> float:
    """Return the cross-sectional area of a circle with diameter d_mm.

    Args:
        d_mm (float): Diameter in millimeters.

    Returns:
        float: Area in mm².
    """
    r = 0.5 * d_mm
    return math.pi * r * r


def r2(x: float) -> float:
    """Round a numeric value to two decimals.

    Args:
        x (float): Input value.

    Returns:
        float: Value rounded to two decimal places.
    """
    return round(float(x), 2)


def read_segments_csv(path: str) -> List[Row]:
    """Load a segment CSV file and return a list of Row objects.

    The CSV must follow the segment_fifo_builder_v6 format. Rows are
    automatically converted into typed 'Row' instances and sorted
    by segment index.

    Args:
        path (str): Path to the segments CSV.

    Returns:
        List[Row]: Parsed and sorted list of segment rows.
    """
    rows: List[Row] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(Row(d))
    rows.sort(key=lambda x: x.seg_idx)
    return rows


def infer_tau0_from_csv(rows: List[Row], fallback: float) -> float:
    """Infer an initial residence time τ₀ from the first extrusion row.

    The function scans the rows in order and returns:
        • the first non-zero t_out, or
        • if unavailable, the first non-zero tau_s, or
        • the provided fallback value.

    Args:
        rows (List[Row]): Parsed segment rows.
        fallback (float): Value to use if no valid τ is found.

    Returns:
        float: Inferred initial residence time τ₀.
    """
    for r in rows:
        if r.is_extrusion:
            if r.t_out > EPS:
                return r.t_out
            if r.tau_s > EPS:
                return r.tau_s
            break
    return fallback

def is_travel_row(r: Row) -> bool:
    """
    Travel row definition (robust):
      - not extrusion/optimizable
      - has motion length (d_mm > 0)
      - cmd is G0 or G1 (movement)
      - exclude dwell (G4) automatically (d_mm=0 or cmd not in G0/G1)
    """
    if (r.is_extrusion or r.is_optimizable):
        return False
    if r.d_mm <= EPS:
        return False
    c = (r.cmd or "").upper()
    if c not in ("G0", "G1"):
        return False
    # optional: accept note hint if你有
    # if "travel" in (r.note or "").lower(): return True
    return True

def infer_F0_row(r: Row, Fmin_fallback: float) -> float:
    """
    Prefer CSV feedrate; else infer from d_mm and t_s; else fallback.
    """
    if getattr(r, "F", 0.0) > EPS:
        return float(r.F)
    if getattr(r, "d_mm", 0.0) > EPS and getattr(r, "t_s", 0.0) > EPS:
        return 60.0 * float(r.d_mm) / float(r.t_s)
    return float(Fmin_fallback)

def update_travel_scale(s_travel: float,
                        tau_mean: float,
                        tau_lo: float,
                        tau_star: float,
                        travel_gain: float,
                        travel_step_max: float,
                        travel_s_min: float) -> float:
    """
    Only slows travel (monotone non-increasing s_travel).
    If tau_mean is below tau_lo, decrease s_travel slightly.
    """
    if tau_star <= EPS or math.isnan(tau_mean):
        return s_travel

    if tau_mean >= tau_lo:
        return s_travel

    e = (tau_star - tau_mean) / max(tau_star, EPS)
    step = travel_gain * e
    step = max(0.0, min(step, travel_step_max))
    s_new = s_travel * (1.0 - step)

    s_new = min(1.0, s_new)
    s_new = max(travel_s_min, s_new)

    return min(s_travel, s_new)

def apply_travel_scale_to_F(rows: List[Row],
                            F_curr: List[float],
                            F_next: List[float],
                            F0_travel: List[float],
                            s_travel: float,
                            travel_dFmax_ratio: float,
                            Fmin: float,
                            Fmax: float,
                            enable: bool) -> List[float]:
    """
    Apply global travel scaling to per-row feedrates while:
      - preserving relative travel structure (F0_travel[i] varies by row)
      - only slowing (never increasing)
      - per-iteration max change smaller than extrusion (travel_dFmax_ratio)
    """
    if not enable:
        return F_next

    out = F_next[:]
    s_travel = min(1.0, max(0.0, s_travel))

    for i, r in enumerate(rows):
        if not is_travel_row(r):
            continue

        F0 = F0_travel[i] if (i < len(F0_travel) and F0_travel[i] > EPS) else infer_F0_row(r, Fmin)
        # target based on original structure
        F_target = F0 * s_travel

        # only slow: cannot exceed current
        F_target = min(F_target, F_curr[i])

        # per-iter limit: max slow-down ratio
        # (e.g. 0.05 => at most 5% slower per iteration)
        F_low = max(Fmin, F_curr[i] * (1.0 - travel_dFmax_ratio))
        F_hi  = min(Fmax, F_curr[i])

        F_new = max(F_low, min(F_target, F_hi))
        out[i] = F_new

    return out

# ---------- FIFO 模擬 ----------
def simulate_v6_lite_fast(rows, F_list, V_nozzle, d_nozzle, tau0 = None, enable_travel_opt=False):
    """
    v6-lite FAST FIFO simulation (no external rebuild).

    Goal:
        Make A's FAST tau close to segment_fifo_builder_v6.py (B) WITHOUT calling it.

    Key properties (mirrors B Phase E logic):
        - Volume-based boundary slicing at exact multiples of V_nozzle (within each row)
        - Absolute time axis includes non-extrusion durations (travel/dwell/etc.)
        - Cycle 1:
            * prefill_delay = total non-piece time before first extrusion piece
            * t_in distributed uniformly over [0, T_fill] of cycle 1 pieces
            * t_out = t_in + prefill_delay
        - Cycle >= 2:
            * t_out uses absolute time axis (streaming)
            * t_in is FIFO backfilled from previous cycle outputs via volume-weighted mapping
        - Outputs are ROW-LEVEL aggregated:
            * row t_in = first piece t_in in that row
            * row t_out = last piece t_out in that row
            * tau = t_out - t_in

    Inputs:
        rows: list[Row] from A (must have: d_mm, F, V_mm3, is_extrusion, is_optimizable)
        F_list: list[float] feedrates aligned to rows (mm/min)
        V_nozzle: float (mm^3)
        d_nozzle: float (mm) used only when V_mm3 missing (fallback volume = area_nozzle * d_mm)

    Returns:
        (t_in_list, t_out_list, tau_list): lists aligned to rows
    """
    EPS = 1e-12
    N = len(rows)

    # ---------------- FIFO helper (copied from B, minimal changes) ----------------
    def fifo_backfill_times(pieces_in_cycle, prev_out_log, fallback_time):
        """
        pieces_in_cycle: list[{"V": float}]
        prev_out_log: deque/list of {"V": float, "t_in": float} where t_in is ABSOLUTE t_out of prev cycle parcels
        fallback_time: float
        """
        fifo = deque(prev_out_log) if prev_out_log is not None else deque()
        result = []
        for p in pieces_in_cycle:
            need = float(p["V"])
            acc_t = 0.0
            acc_v = 0.0

            while need > EPS and fifo:
                head = fifo[0]
                take = min(head["V"], need)
                acc_t += head["t_in"] * take
                acc_v += take
                head["V"] -= take
                need -= take
                if head["V"] <= EPS:
                    fifo.popleft()

            if need > EPS:
                acc_t += fallback_time * need
                acc_v += need
                need = 0.0

            result.append(acc_t / acc_v if acc_v > EPS else fallback_time)
        return result

    # ---------------- small geometry helpers ----------------
    def area_circle(d_mm):
        r = 0.5 * d_mm
        return math.pi * r * r

    A_noz = area_circle(d_nozzle)

    def is_piece_row(r):
        return bool(r.is_extrusion or r.is_optimizable)

    def get_F(i, r):
        if i < len(F_list) and F_list[i] > EPS:
            return float(F_list[i])
        if r.F > EPS:
            return float(r.F)
        return 1.0

    def row_t_s(i, r):
        # 1) extrusion/optimizable: recompute with updated F_list
        if is_piece_row(r):
            d = float(r.d_mm)
            F = get_F(i, r)
            if d > EPS and F > EPS:
                return 60.0 * d / F
            return float(getattr(r, "t_s", 0.0) or 0.0)

        # 2) travel rows: optionally recompute with updated F_list
        if enable_travel_opt and is_travel_row(r):
            d = float(r.d_mm)
            F = get_F(i, r)
            if d > EPS and F > EPS:
                return 60.0 * d / F
            return float(getattr(r, "t_s", 0.0) or 0.0)

        # 3) other non-piece rows (dwell etc.): trust CSV t_s
        return float(getattr(r, "t_s", 0.0) or 0.0)

    def row_V(r):
        if getattr(r, "V_mm3", 0.0) > EPS:
            return float(r.V_mm3)
        # fallback: approximate using nozzle cross-section * XY length
        return A_noz * float(r.d_mm)

    # ---------------- outputs (row-level) ----------------
    t_in_list  = [0.0] * N
    t_out_list = [0.0] * N
    tau_list   = [0.0] * N

    row_first_tin = [None] * N
    row_last_tout = [None] * N

    # ---------------- global absolute time axis ----------------
    global_time = 0.0

    # ---------------- cycle buffering state ----------------
    cycle_id = 1
    S_cycle_vol = 0.0
    cycle_pieces = []  # each piece: {"row_i","V","t_s","t_out_abs"}

    prev_out_log = None
    last_cycle_last_tout_abs = 0.0

    seen_first_piece = False
    prefill_delay_cycle1 = 0.0

    def apply_piece_timings(piece_timings):
        # piece_timings: list[(row_i, t_in, t_out)]
        for row_i, tin, tout in piece_timings:
            if row_first_tin[row_i] is None:
                row_first_tin[row_i] = tin
            row_last_tout[row_i] = tout

    def flush_cycle(cycle_id, cycle_pieces):
        """
        Compute t_in/t_out for all pieces in current cycle and update FIFO logs.
        Returns: None (updates outer scope vars via nonlocal)
        """
        nonlocal prev_out_log, last_cycle_last_tout_abs

        if not cycle_pieces:
            return

        M = len(cycle_pieces)

        if cycle_id == 1:
            # Cycle 1: t_in follows the fill-progress inside the first cycle
            # (use cumulative end time of each piece in the fill process)
            cum = 0.0
            base_t_in = []
            for p in cycle_pieces:
                cum += p["t_s"]
                base_t_in.append(cum)

            abs_t_out = [ti + prefill_delay_cycle1 for ti in base_t_in]

        else:
            # Cycle >= 2 uses absolute t_out axis from streaming simulation:
            abs_t_out = [p["t_out_abs"] for p in cycle_pieces]
            base_t_in = fifo_backfill_times(
                [{"V": p["V"]} for p in cycle_pieces],
                prev_out_log,
                last_cycle_last_tout_abs
            )

        # emit piece timings
        piece_timings = []
        for p, tin, tout in zip(cycle_pieces, base_t_in, abs_t_out):
            piece_timings.append((p["row_i"], tin, tout))

        apply_piece_timings(piece_timings)

        # update cycle end
        last_cycle_last_tout_abs = abs_t_out[-1]

        # prepare FIFO source for next cycle: each parcel is (V, t_out_abs)
        prev_out_log = deque({"V": cycle_pieces[i]["V"], "t_in": abs_t_out[i]} for i in range(M))

    # ---------------- main pass over rows ----------------
    for i, r in enumerate(rows):
        ts = row_t_s(i, r)

        if not is_piece_row(r) or ts <= EPS or float(r.d_mm) <= EPS:
            # non-piece row: only advances absolute time
            global_time += ts
            continue

        V_total = row_V(r)
        if V_total <= EPS:
            global_time += ts
            continue

        # first extrusion piece: record prefill_delay (time before first piece)
        if not seen_first_piece:
            seen_first_piece = True
            prefill_delay_cycle1 = global_time

        remaining_V = V_total
        # note: time per volume ratio assumed constant within row
        # => piece time is proportional to piece volume
        time_per_vol = ts / V_total

        while remaining_V > EPS:
            room = V_nozzle - S_cycle_vol

            # if cycle already full due to numeric drift, flush and start new cycle
            if room <= 1e-9:
                flush_cycle(cycle_id, cycle_pieces)
                cycle_id += 1
                S_cycle_vol = 0.0
                cycle_pieces = []
                continue

            take_V = remaining_V if remaining_V <= room + EPS else room
            take_ts = take_V * time_per_vol

            # advance absolute time to piece end
            global_time += take_ts
            t_out_abs_piece = global_time

            cycle_pieces.append({
                "row_i": i,
                "V": float(take_V),
                "t_s": float(take_ts),
                "t_out_abs": float(t_out_abs_piece),
            })

            S_cycle_vol += take_V
            remaining_V -= take_V

            # boundary reached -> flush cycle immediately
            if S_cycle_vol >= V_nozzle - 1e-9:
                flush_cycle(cycle_id, cycle_pieces)
                cycle_id += 1
                S_cycle_vol = 0.0
                cycle_pieces = []

    # flush the last partial cycle
    if cycle_pieces:
        flush_cycle(cycle_id, cycle_pieces)

    # ---------------- finalize row-level outputs ----------------
    for i, r in enumerate(rows):
        if is_piece_row(r):
            tin = row_first_tin[i]
            tout = row_last_tout[i]
            if tin is None or tout is None:
                t_in_list[i] = 0.0
                t_out_list[i] = 0.0
                tau_list[i] = 0.0
            else:
                t_in_list[i] = float(tin)
                t_out_list[i] = float(tout)
                tau_list[i] = float(tout - tin)
        else:
            t_in_list[i] = 0.0
            t_out_list[i] = 0.0
            tau_list[i] = 0.0

    return t_in_list, t_out_list, tau_list

def simulate_periodic_alignment(rows: List[Row],
                                F_list: List[float],
                                V_nozzle: float,
                                d_nozzle: float,
                                tau0: float,
                                enable_travel_opt: bool = False) -> Tuple[List[float], List[float], List[float]]:
    return simulate_v6_lite_fast(
        rows,
        F_list,
        V_nozzle,
        d_nozzle,
        tau0=tau0,
        enable_travel_opt=enable_travel_opt
    )

# --------- Real FIFO Builder v6 ---------
def simulate_fifo_v6_wrapper(
    gcode_template_path: str,
    rows: List[Row],
    F_vec: List[float],
    P_vec_ms: List[float],
    V_nozzle: float,
    d_nozzle: float,
    iteration: int,
    out_root: str,
    enable_travel_opt: bool = False,
    s_travel: float = 1.0,
    save_intermediate: bool = False,
) -> Tuple[bool, List[float], List[float], List[float], str, str]:

    """
    TRUE 模式物理引擎封裝：
      1) 依 seg_idx patch 模板 gcode（精準對齊）
      2) 呼叫 physical_builder 產生 CSV（tau_s）
      3) 讀回並對齊到 rows（同長度）
      4) 失敗時回傳 ok=False，外層保留 F_curr 不更新（你要求 #2）

    回傳:
      ok, t_in_list, t_out_list, tau_list, csv_path, gcode_path
    """

    # 工作資料夾（建議放在 out_dir 下面，避免散落）
    tmp_dir = os.path.join(out_root, "true_fifo_results", f"iter_{iteration:04d}")
    os.makedirs(tmp_dir, exist_ok=True)

    tmp_gcode = os.path.join(tmp_dir, "temp_in.gcode")
    tmp_csv   = os.path.join(tmp_dir, "temp_out.csv")
    tmp_outgcode = os.path.join(tmp_dir, "temp_out.gcode")

    # 0) 先 patch gcode（依 seg_idx）
    try:
        patched, expected = patch_gcode_by_segidx(
            gcode_template_path=gcode_template_path,
            gcode_out_path=tmp_gcode,
            rows=rows,
            F_vec=F_vec,
            P_vec_ms=P_vec_ms,
            enable_travel_opt=enable_travel_opt,
            s_travel=s_travel
        )
        if patched < max(1, int(0.98 * expected)):
            # patch 覆蓋率太低 → 高機率是模板沒 tag 或對齊已壞
            print(f"⛔ TRUE patch coverage too low: {patched}/{expected}. Skipping TRUE iteration.")
            return False, [float("nan")]*len(rows), [float("nan")]*len(rows), [float("nan")]*len(rows), tmp_csv, tmp_gcode
    except Exception as e:
        print(f"⛔ TRUE patch failed: {e}. Skipping TRUE iteration.")
        return False, [float("nan")]*len(rows), [float("nan")]*len(rows), [float("nan")]*len(rows), tmp_csv, tmp_gcode

    # 1) 呼叫物理引擎（你要可替換：physical_builder）
    try:
        true_fifo_rebuild(
            gcode_in=tmp_gcode,
            csv_out=tmp_csv,
            gcode_out=tmp_outgcode,
            V_nozzle=V_nozzle,
            d_filament=15.55634919,
            e_mode="filament",
        )

    except Exception as e:
        print(f"⛔ TRUE physical engine failed: {e}. Skipping TRUE iteration.")
        return False, [float("nan")]*len(rows), [float("nan")]*len(rows), [float("nan")]*len(rows), tmp_csv, tmp_gcode

    # 2) 檢查輸出 CSV 是否存在（你要求 #2：不存在就跳過、保留上一輪）
    if not os.path.exists(tmp_csv):
        print("⛔ TRUE physical engine produced no CSV. Skipping TRUE iteration.")
        return False, [float("nan")]*len(rows), [float("nan")]*len(rows), [float("nan")]*len(rows), tmp_csv, tmp_gcode

    # 3) 讀回結果並對齊 rows（同長度）
    try:
        df = pd.read_csv(tmp_csv)

        # 建立 seg_idx -> (t_in,t_out,tau) 映射
        m = {}
        for _, rr in df.iterrows():
            try:
                seg = int(rr["seg_idx"])
                tau = float(rr["tau_s"]) if (not pd.isna(rr["tau_s"])) else float("nan")
                tin = float(rr["t_in"])  if (not pd.isna(rr["t_in"]))  else float("nan")
                tout = float(rr["t_out"]) if (not pd.isna(rr["t_out"])) else float("nan")
                m[seg] = (tin, tout, tau)
            except:
                continue

        t_in_list  = [float("nan")] * len(rows)
        t_out_list = [float("nan")] * len(rows)
        tau_list   = [float("nan")] * len(rows)
        #
        hit = 0
        for i, r in enumerate(rows):
            key = r.seg_idx
            if key in m:
                t_in_list[i], t_out_list[i], tau_list[i] = m[key]
                hit += 1

        coverage = hit / max(1, len(rows))
        if coverage < 0.98:
            print(f"⛔ TRUE CSV seg_idx coverage too low: {hit}/{len(rows)} ({coverage:.3f}). Skipping TRUE iteration.")
            return False, [float("nan")]*len(rows), [float("nan")]*len(rows), [float("nan")]*len(rows), tmp_csv, tmp_gcode

        # 保存（建議：只在 save_intermediate 時做，避免爆磁碟；見下一條）
        if save_intermediate:
            dst = os.path.join(out_root, f"true_fifo_iter_{iteration:04d}")
            if not os.path.exists(dst):
                shutil.copytree(tmp_dir, dst)


        return True, t_in_list, t_out_list, tau_list, tmp_csv, tmp_gcode

    except Exception as e:
        print(f"⛔ TRUE read/align failed: {e}. Skipping TRUE iteration.")
        return False, [float("nan")]*len(rows), [float("nan")]*len(rows), [float("nan")]*len(rows), tmp_csv, tmp_gcode

# ---------- Metrics ----------

def risk_metrics(tau_list, rows, tau_lo: float, tau_hi: float):
    """
    Risk-based metrics over extrusion rows.
    Returns:
      violations_count: #segments outside [tau_lo, tau_hi]
      v_sum: sum of outside-distance (seconds)
      v_max: max outside-distance (seconds)
      p95_abs: 95th percentile of abs deviation from tau_star surrogate (distance to nearest bound)
      cvar95: mean of worst 5% abs distances (tail mean)
    """
    extr_vals = []
    dist = []

    for i, r in enumerate(rows):
        if not r.is_extrusion:
            continue
        if i >= len(tau_list):
            continue
        t = tau_list[i]
        if t is None or math.isnan(t):
            continue
        extr_vals.append(t)

        # distance to feasible window
        t_c = tau_to_const(t)
        tau_lo_c = tau_to_const(tau_lo)
        tau_hi_c = tau_to_const(tau_hi)

        if t_c < tau_lo_c:
            dist.append(tau_lo_c - t_c)
        elif t_c > tau_hi_c:
            dist.append(t_c - tau_hi_c)
        else:
            dist.append(0.0)

    if not dist:
        return 0, 0.0, 0.0, 0.0, 0.0

    violations_count = sum(1 for d in dist if d > EPS)
    v_sum = float(sum(dist))
    v_max = float(max(dist))

    # tail stats
    dist_sorted = sorted(dist)
    n = len(dist_sorted)
    # p95 index (robust)
    k95 = min(n - 1, int(math.floor(0.95 * (n - 1))))
    p95_abs = float(dist_sorted[k95])

    # CVaR95: mean of worst 5%
    tail_start = int(math.floor(0.95 * n))
    tail = dist_sorted[tail_start:] if tail_start < n else [dist_sorted[-1]]
    cvar95 = float(sum(tail) / max(1, len(tail)))

    return violations_count, v_sum, v_max, p95_abs, cvar95

def within_percentage_bounds(tau_list, rows, tau_lo, tau_hi) -> float:
    extr = []
    for i, r in enumerate(rows):
        if not r.is_extrusion:
            continue
        if i >= len(tau_list):
            continue
        v = tau_list[i]
        if v is None or math.isnan(v):
            continue
        extr.append(v)

    if not extr:
        return float("nan")

    tau_lo_c = tau_to_const(tau_lo)
    tau_hi_c = tau_to_const(tau_hi)

    ok = sum(
        1 for v in extr
        if tau_lo_c <= tau_to_const(v) <= tau_hi_c
    )
    return 100.0 * ok / len(extr)

def tau_stats(tau_list, rows):
    extr = []
    for i, r in enumerate(rows):
        if not r.is_extrusion:
            continue
        if i >= len(tau_list):
            continue
        v = tau_list[i]
        if v is None or math.isnan(v):
            continue
        extr.append(v)

    if not extr:
        return (0.0, 0.0, 0.0)

    return (min(extr), max(extr), sum(extr)/len(extr))

def global_error_abs_sum(tau_list, rows, tau_star: float) -> float:
    s = 0.0
    for i, r in enumerate(rows):
        if not r.is_extrusion:
            continue
        if i >= len(tau_list):
            continue
        v = tau_list[i]
        if v is None or math.isnan(v):
            continue
        s += abs(v - tau_star)
    return s

# ---------- Boundary & window_cap bounds ----------

def compute_boundaries_and_next_map(rows: List[Row],
                                    V_nozzle: float,
                                    d_nozzle: float) -> Tuple[List[int], List[int]]:
    """Identify FIFO volume boundaries and create a forward boundary map.

    This function rebuilds cycle boundaries purely from segment volumes.
    As volume accumulates, each point where the running sum crosses V_nozzle
    is recorded as a boundary index. A companion map (`next_of`) is produced:
    for each row index i, next_of[i] gives the next boundary row or -1 if none.

    Args:
        rows (List[Row]): Segments containing extrusion/optimizable flags and volume data.
        V_nozzle (float): FIFO cycle volume threshold.
        d_nozzle (float): Nozzle diameter used when rows lack explicit volume.

    Returns:
        tuple:
            boundaries (list[int]): Row indices marking cycle boundaries.
            next_of (list[int]): For each row, the next boundary index or -1.
    """
    A = area_circle(d_nozzle)
    N = len(rows)
    boundaries: List[int] = []
    S_cycle = 0.0
    # first_seen = False
    for i, r in enumerate(rows):
        if not (r.is_extrusion or r.is_optimizable):
            continue
        V_in = r.V_mm3 if r.V_mm3 > 0 else A * r.d_mm
        # if not first_seen:
        #     first_seen = True
        #     S_cycle += V_in
        #     continue
        if S_cycle + V_in >= V_nozzle + EPS:
            boundaries.append(i)
            overflow = (S_cycle + V_in) - V_nozzle
            S_cycle = overflow
        else:
            S_cycle += V_in
    next_of = [-1]*N
    bptr = 0
    next_b = boundaries[bptr] if boundaries else -1
    for i in range(N):
        while next_b != -1 and i > next_b:
            bptr += 1
            next_b = boundaries[bptr] if bptr < len(boundaries) else -1
        next_of[i] = next_b
    return boundaries, next_of

def compute_cycle_start_map(rows: List[Row],
                            V_nozzle: float,
                            d_nozzle: float) -> Tuple[List[int], List[int]]:
    """
    Build cycle boundary indices and a cycle_start map.

    boundaries: indices where a cycle reaches (or exceeds) V_nozzle.
    cycle_start[i]: for each row i, the FIRST row index that belongs to the same FIFO cycle as i.
                   non-piece rows get -1.

    This is purely volume-based (same idea as v6 builder).
    """
    A = area_circle(d_nozzle)
    N = len(rows)

    # 1) Identify boundaries (cycle end indices) in piece-row space
    boundaries: List[int] = []
    S = 0.0
    for i, r in enumerate(rows):
        if not (r.is_extrusion or r.is_optimizable):
            continue
        V = r.V_mm3 if r.V_mm3 > EPS else A * r.d_mm
        if S + V >= V_nozzle + EPS:
            boundaries.append(i)
            S = (S + V) - V_nozzle  # carry overflow
        else:
            S += V

    # 2) Assign cycle_start map by scanning and tracking current cycle start
    cycle_start = [-1] * N
    current_start = None
    bptr = 0
    current_boundary = boundaries[bptr] if boundaries else None

    for i, r in enumerate(rows):
        if not (r.is_extrusion or r.is_optimizable):
            continue

        if current_start is None:
            current_start = i  # first piece row starts the first cycle

        cycle_start[i] = current_start

        # When we hit a boundary index, next piece row starts a new cycle
        if current_boundary is not None and i == current_boundary:
            bptr += 1
            current_boundary = boundaries[bptr] if bptr < len(boundaries) else None
            current_start = None  # will be set at next piece row

    return boundaries, cycle_start

def compute_prev_boundary_map(boundaries: List[int], N: int) -> List[int]:
    """
    Build a map prev_of[i] = the last boundary index <= i, or -1 if none.
    Boundaries are row indices where a V_nozzle cycle boundary is hit.
    """
    prev_of = [-1] * N
    if not boundaries:
        return prev_of

    bptr = 0
    current = boundaries[bptr]
    last = -1

    for i in range(N):
        # Advance boundary pointer while the boundary is behind i
        while bptr < len(boundaries) and boundaries[bptr] <= i:
            last = boundaries[bptr]
            bptr += 1
        prev_of[i] = last
    return prev_of


def get_boundary_backward_window(rows: List[Row],
                                target_i: int,
                                window_cap: int,
                                boundaries: List[int],
                                prev_boundary_of: List[int]) -> List[int]:
    """
    Boundary-backward window (cycle-local backward slice).

    Logic:
      1) Determine the start of the current V_nozzle cycle:
         - If there is a previous boundary at index b (<= target_i),
           then cycle_start = b + 1
         - Else cycle_start = first "piece row" index in the file (fallback)
      2) Apply a backward cap: start_i = max(cycle_start, target_i - window_cap)
      3) Collect only "piece rows" in [start_i, target_i] inclusive
         (piece row := extrusion OR optimizable)

    Returns:
      A list of row indices in increasing order.
    """
    N = len(rows)
    if target_i < 0 or target_i >= N:
        return []

    # Helper: piece row definition
    def is_piece_row(r: Row) -> bool:
        return bool(r.is_extrusion or r.is_optimizable)

    # Find the first piece row as a robust fallback cycle start
    first_piece = None
    for k in range(N):
        if is_piece_row(rows[k]):
            first_piece = k
            break
    if first_piece is None:
        return []

    # Identify current cycle start using previous boundary
    prev_b = prev_boundary_of[target_i] if (0 <= target_i < len(prev_boundary_of)) else -1
    if prev_b != -1:
        cycle_start = prev_b + 1
    else:
        cycle_start = first_piece

    # Backward capped start index (must stay inside the same cycle)
    start_i = max(cycle_start, target_i - int(window_cap))

    # Collect indices within [start_i, target_i]
    win = []
    for j in range(start_i, target_i + 1):
        if is_piece_row(rows[j]):
            win.append(j)

    return win

def compute_cap_bounds_from_cycle(rows, V_nozzle, d_nozzle):
    """Estimate upper and lower bounds for window_cap from extrusion volumes.

    The function samples the first up-to-1000 extrusion rows, computes
    their mean volume, and uses it to approximate how many segments fill
    a full FIFO cycle. The result provides a stable search range for
    adaptive window sizing.

    Args:
        rows (list[Row]): Segment rows used to collect extrusion volumes.
        V_nozzle (float): FIFO cycle volume.
        d_nozzle (float): Nozzle diameter for volume fallback.

    Returns:
        tuple:
            upper (int): Suggested maximum window_cap.
            lower (int): Suggested minimum window_cap.
    """
    A = area_circle(d_nozzle)
    extr = [r for r in rows if (r.is_extrusion)]
    if not extr:
        return 10, 1

    # 取樣前 1000 段作平均（不足則全取）
    sample = extr[:min(len(extr), 1000)]
    avgV = sum(r.V_mm3 if r.V_mm3 > 0 else A * r.d_mm for r in sample) / len(sample)

    est_count = int(V_nozzle / max(avgV, 1e-6))
    # upper = max(1,int(est_count * 0.3))
    upper = max(1, 2000)
    lower = max(1, 30)
    # lower = max(1, int(round(upper * 0.001)))

    print(f"🔧 window_cap estimated via avgV: upper={upper}, lower={lower}, avgV={avgV:.6f}")
    return upper, lower

# ---------- Optimizer 核心 ----------

def optimize_feedforward(rows: List[Row],
                         F_current: List[float],
                         tau_current: List[float],
                         tau_lo: float,
                         tau_star: float,
                         tau_hi: float,
                         V_nozzle: float,
                         d_nozzle: float,
                         window_cap: int,
                         use_boundary_window: bool,
                         Kp: float,
                         Kd: float,
                         Fmin: float,
                         Fmax: float,
                         dFmax_ratio: float,
                         deltaT_cap_ratio: float) -> List[float]:
    """Compute a new feedrate profile using forward-looking τ error correction.

    This function implements the core feed-forward optimizer. For each
    extrusion/optimizable segment, it compares the current residence time τ
    against the target value (tau_star). If the deviation exceeds the buffer
    zone, the algorithm redistributes a time correction ΔT across a forward
    window of segments.

    Time corrections are converted into feedrate adjustments through:
        t = 60 * d_mm / F

    Key behaviors:
        * Uses a PD controller (Kp + Kd) to produce desired ΔT.
        * Enforces global caps via deltaT_cap_ratio * tau_star.
        * Supports two window modes:
              - boundary-based: window ends at next V_nozzle boundary
              - fixed length: window_cap rows ahead
        * Applies physical limits: Fmin, Fmax, and relative change (dFmax_ratio).
        * Distributes ΔT with quadratic weights (later segments get higher weight).
        * Smooths output using a 70/30 blend between old and target feedrates.

    Args:
        rows (List[Row]): Segment metadata (extrusion flags, geometry, volumes).
        F_current (list[float]): Feedrate values (mm/min) aligned to rows.
        tau_current (list[float]): Current τ values for each row.
        tau_star (float): Target residence time.
        buffer_ratio (float): Allowed relative deviation for τ before correction.
        V_nozzle (float): FIFO cycle volume threshold.
        d_nozzle (float): Nozzle diameter for fallback volume.
        window_cap (int): Maximum forward window length if not boundary-based.
        use_boundary_window (bool): Whether window ends at next cycle boundary.
        Kp (float): Proportional gain for PD controller.
        Kd (float): Derivative gain for PD controller.
        Fmin (float): Hard lower bound for feedrate.
        Fmax (float): Hard upper bound for feedrate.
        dFmax_ratio (float): Max relative feedrate change per iteration.
        deltaT_cap_ratio (float): Max |ΔT| as fraction of tau_star.

    Returns:
        list[float]: Updated feedrate list (same length as F_current).
    """

    N = len(rows)
    F_new = F_current[:]
    boundaries, cycle_start_of = compute_cycle_start_map(rows, V_nozzle, d_nozzle)
    _, next_boundary_of = compute_boundaries_and_next_map(rows, V_nozzle, d_nozzle)

    target_indices = [i for i,r in enumerate(rows) if (r.is_extrusion or r.is_optimizable)]
    if not target_indices:
        return F_new

    def clamp_rate(F_old: float, F_target: float) -> float:
        if F_old <= EPS:
            base = F_target
        else:
            base = F_target
        if F_old <= EPS:
            return max(Fmin, min(base, Fmax))
        dF_max = abs(F_old) * dFmax_ratio
        low = max(Fmin, F_old - dF_max)
        high = min(Fmax, F_old + dF_max)
        return max(low, min(base, high))

    prev_e = 0.0

    for target_i in target_indices:
        tau_i = tau_current[target_i]
        if tau_i is None or math.isnan(tau_i):
            continue

        if tau_i < tau_lo:
            err = (tau_lo - tau_i)
        elif tau_i > tau_hi:
            err = (tau_hi - tau_i)
        else:
            err = 0.0

        # inside window => no actuation
        if abs(err) <= EPS:
            prev_e = 0.0
            continue

        de = err - prev_e
        prev_e = err

        # 控制輸出 (時間修正)
        DELTA_T_STEP_CAP = max(1e-6, deltaT_cap_ratio * tau_star)
        u_des = Kp * err + Kd * de
        if u_des > DELTA_T_STEP_CAP:
            u_des = DELTA_T_STEP_CAP
        elif u_des < -DELTA_T_STEP_CAP:
            u_des = -DELTA_T_STEP_CAP

        # window 範圍
        if use_boundary_window:
            nb = next_boundary_of[target_i]
            end_i = nb if nb != -1 else (N - 1)
        else:
            end_i = min(N - 1, target_i + window_cap)

        win = [j for j in range(target_i, end_i + 1)
               if (rows[j].is_extrusion or rows[j].is_optimizable)]
        if not win:
            continue

        # 時間上下界
        t_base, t_min, t_max = [], [], []
        for j in win:
            dmm = rows[j].d_mm
            Fold = F_new[j] if F_new[j] > EPS else (rows[j].F if rows[j].F > EPS else Fmin)
            if dmm > EPS and Fold > EPS:
                tb = 60.0 * dmm / Fold
                t_base.append(tb)
                t_min.append(60.0 * dmm / Fmax)  # 最快
                t_max.append(60.0 * dmm / Fmin)  # 最慢
            else:
                tb = 0.0
                t_base.append(tb)
                t_min.append(tb)
                t_max.append(tb)

        Cplus = sum(t_max[k] - t_base[k] for k in range(len(win)))
        Cminus = sum(t_base[k] - t_min[k] for k in range(len(win)))

        if u_des >= 0.0:
            DeltaT = min(u_des, Cplus)
        else:
            DeltaT = -min(-u_des, Cminus)

        if abs(DeltaT) <= 1e-12:
            continue

        # 指數權重（後段較高，但極度平滑）
        alpha = 0.03   # 建議範圍 0.01–0.05，可微調強弱
        L = len(win)
        ranks = [math.exp(alpha * k) for k in range(L)]
        s = float(sum(ranks))
        weights = [rk / s for rk in ranks]


        # 實際調整
        for k, j in enumerate(win):
            dmm = rows[j].d_mm
            if dmm <= EPS:
                continue
            Fold = F_new[j] if F_new[j] > EPS else (rows[j].F if rows[j].F > EPS else Fmin)
            tb = t_base[k]
            dt = weights[k] * DeltaT
            new_ts = tb + dt
            if new_ts <= 0.0:
                continue
            # 約束在 min/max 內
            new_ts = max(t_min[k], min(new_ts, t_max[k]))
            F_tar = 60.0 * dmm / new_ts
            F_tar = clamp_rate(Fold, F_tar)
            F_new[j] = 0.9 * Fold + 0.1 * F_tar  # 平滑

    return F_new

# ---------- New Optimizer Hybrid ----------
def optimize_feedforward_hybrid(rows: List[Row],
                                F_current: List[float],
                                tau_current: List[float],
                                tau_lo: float,
                                tau_star: float,
                                tau_hi: float,
                                V_nozzle: float,
                                d_nozzle: float,
                                boundaries: List[int],
                                next_boundary_of: List[int],
                                cycle_start_of: List[int],
                                window_cap: int,
                                use_boundary_window: bool,
                                window_direction: str,
                                t_out_current: List[float],
                                horizon_sec: float,
                                horizon_mode: str,
                                horizon_w_future: float,
                                Kp: float,
                                Kd: float,
                                Ki_cycle: float,
                                Fmin: float,
                                Fmax: float,
                                dFmax_ratio: float,
                                deltaT_cap_ratio: float,
                                smooth_passes: int) -> List[float]:
    """
    Hybrid τ controller:
      - Local PD (per segment)
      - Cycle-level I (per FIFO cycle)
      - Forward window ΔT redistribution
      - Global smoothing over F_new
    """

    weight_cache = {}
    def get_exp_weights(L: int, alpha: float = 0.03):
        key = (L, alpha)
        w = weight_cache.get(key)
        if w is not None:
            return w
        ranks = [math.exp(alpha * k) for k in range(L)]
        s = float(sum(ranks)) or 1.0
        w = [rk / s for rk in ranks]
        weight_cache[key] = w
        return w


    N = len(rows)
    F_new = F_current[:]

    # 2) 為每個 row 指派 cycle_id
    cycle_of = [-1] * N
    if boundaries:
        bptr = 0
        current_cycle = 0
        current_boundary = boundaries[bptr]
        started = False

        for i, r in enumerate(rows):
            if not (r.is_extrusion or r.is_optimizable):
                continue

            if not started:
                started = True
                current_cycle = 0

            cycle_of[i] = current_cycle

            if i == current_boundary:
                bptr += 1
                current_cycle += 1
                current_boundary = boundaries[bptr] if bptr < len(boundaries) else None
    else:
        # 沒有 boundary 時，所有 extrusion/optimizable row 當成同一個 cycle
        cid = 0
        for i, r in enumerate(rows):
            if (r.is_extrusion or r.is_optimizable):
                cycle_of[i] = cid

    def seg_err_signed(t):
        if t is None or math.isnan(t):
            return 0.0
        if t < tau_lo:
            return (tau_lo - t)     # +
        if t > tau_hi:
            return (tau_hi - t)     # -
        return 0.0

    def seg_dist(t):
        return abs(seg_err_signed(t))

    # cycle-level aggregations (ONLY segments outside window)
    cycle_sum_signed = {}
    cycle_sum_dist   = {}
    cycle_cnt        = {}

    # also keep "where is the worst segment in that cycle"
    cycle_max_dist = {}
    cycle_max_i    = {}

    for i, r in enumerate(rows):
        cid = cycle_of[i]
        if cid < 0:
            continue
        if i >= len(tau_current):
            continue
        t = tau_current[i]
        if t is None or math.isnan(t):
            continue

        e = seg_err_signed(t)
        if abs(e) <= EPS:
            continue  # inside window => ignore

        d = abs(e)

        cycle_sum_signed[cid] = cycle_sum_signed.get(cid, 0.0) + e
        cycle_sum_dist[cid]   = cycle_sum_dist.get(cid, 0.0)   + d
        cycle_cnt[cid]        = cycle_cnt.get(cid, 0) + 1

        # record cycle-local worst segment
        if (cid not in cycle_max_dist) or (d > cycle_max_dist[cid]):
            cycle_max_dist[cid] = d
            cycle_max_i[cid] = i

    # finalized cycle errors:
    # - signed mean: for Ki control (directional)
    # - dist mean:  for optional debug
    cycle_err_signed = {}
    cycle_err_dist   = {}
    for cid in cycle_cnt.keys():
        n = cycle_cnt[cid]
        if n > 0:
            cycle_err_signed[cid] = cycle_sum_signed[cid] / n
            cycle_err_dist[cid]   = cycle_sum_dist[cid] / n
        else:
            cycle_err_signed[cid] = 0.0
            cycle_err_dist[cid]   = 0.0


    # 4) 目標 rows
    target_indices = [i for i, r in enumerate(rows) if (r.is_extrusion or r.is_optimizable)]
    if not target_indices:
        return F_new

    # 5) feedrate 限制（單段相對改變 + 絕對上下界）
    def clamp_rate(F_old: float, F_target: float) -> float:
        if F_old <= EPS:
            base = F_target
            return max(Fmin, min(base, Fmax))

        base = F_target
        dF_max = abs(F_old) * dFmax_ratio
        low = max(Fmin, F_old - dF_max)
        high = min(Fmax, F_old + dF_max)
        return max(low, min(base, high))

    prev_e = 0.0

    from bisect import bisect_left, bisect_right

    # precompute once
    is_target = [ (r.is_extrusion or r.is_optimizable) for r in rows ]
    target_idx = []
    target_tout = []
    for i in range(N):
        if not is_target[i]: 
            continue
        t = t_out_current[i]
        if t is None or math.isnan(t):
            continue
        target_idx.append(i)
        target_tout.append(float(t))

    def window_by_time(t0: float, t1: float) -> List[int]:
        # assume target_tout is non-decreasing
        L = bisect_left(target_tout, t0)
        R = bisect_right(target_tout, t1)
        win = target_idx[L:R]
        if not win:
            return []
        # hard cap to avoid blow-up when segments are tiny
        if window_cap > 0 and len(win) > window_cap:
            # keep closest ones to target in time (simple: keep first window_cap)
            win = win[:window_cap]
        return win

    # 6) 主迴圈：對每個目標 row 做 Hybrid 控制
    for target_i in target_indices:
        tau_i = tau_current[target_i]
        if tau_i is None or math.isnan(tau_i):
            continue

        # window-based signed error (inside => 0)
        if tau_i < tau_lo:
            err = (tau_lo - tau_i)      # +
        elif tau_i > tau_hi:
            err = (tau_hi - tau_i)      # -
        else:
            err = 0.0

        if abs(err) <= EPS:
            prev_e = 0.0
            continue

        de = err - prev_e
        prev_e = err

        # cycle-level I 項
        cid = cycle_of[target_i]
        e_cycle = cycle_err_signed.get(cid, 0.0) if cid >= 0 else 0.0

        # Hybrid 控制輸出 (時間修正 ΔT)
        DELTA_T_STEP_CAP = max(1e-6, deltaT_cap_ratio * tau_star)
        u_des = Kp * err + Kd * de + Ki_cycle * e_cycle

        # 夾住 ΔT
        if u_des > DELTA_T_STEP_CAP:
            u_des = DELTA_T_STEP_CAP
        elif u_des < -DELTA_T_STEP_CAP:
            u_des = -DELTA_T_STEP_CAP

        # window 範圍
        # window selection (place here)
        use_horizon = (
            horizon_sec is not None and horizon_sec > 0.0 and
            target_i < len(t_out_current) and
            (not math.isnan(t_out_current[target_i]))
        )

        # helper: apply deltaT on a given window (uses same constraints/weights)
        def apply_window_adjust(win_local: List[int], DeltaT_local: float) -> None:
            if not win_local or abs(DeltaT_local) <= 1e-12:
                return

            # time bounds for this window
            t_base_l, t_min_l, t_max_l = [], [], []
            for jj in win_local:
                dmm = rows[jj].d_mm
                Fold = F_new[jj] if F_new[jj] > EPS else (rows[jj].F if rows[jj].F > EPS else Fmin)
                if dmm > EPS and Fold > EPS:
                    tb = 60.0 * dmm / Fold
                    t_base_l.append(tb)
                    t_min_l.append(60.0 * dmm / Fmax)
                    t_max_l.append(60.0 * dmm / Fmin)
                else:
                    tb = 0.0
                    t_base_l.append(tb)
                    t_min_l.append(tb)
                    t_max_l.append(tb)

            Cplus_l  = sum(t_max_l[k] - t_base_l[k] for k in range(len(win_local)))
            Cminus_l = sum(t_base_l[k] - t_min_l[k] for k in range(len(win_local)))

            # clamp DeltaT into feasible capacity of THIS window
            if DeltaT_local >= 0.0:
                DeltaT_eff = min(DeltaT_local, Cplus_l)
            else:
                DeltaT_eff = -min(-DeltaT_local, Cminus_l)

            if abs(DeltaT_eff) <= 1e-12:
                return

            # exponential weights (same as你現在)
            weights = get_exp_weights(len(win_local), alpha=0.03)

            # apply
            for k, jj in enumerate(win_local):
                dmm = rows[jj].d_mm
                if dmm <= EPS:
                    continue
                Fold = F_new[jj] if F_new[jj] > EPS else (rows[jj].F if rows[jj].F > EPS else Fmin)
                tb = t_base_l[k]
                dt = weights[k] * DeltaT_eff
                new_ts = tb + dt
                if new_ts <= 0.0:
                    continue

                new_ts = max(t_min_l[k], min(new_ts, t_max_l[k]))
                F_tar = 60.0 * dmm / new_ts
                F_tar = clamp_rate(Fold, F_tar)
                F_new[jj] = 0.6 * Fold + 0.4 * F_tar

        # build window
        if use_horizon:
            t_ref = float(t_out_current[target_i])
            h = float(horizon_sec)

            if horizon_mode == "future":
                win = window_by_time(t_ref, t_ref + h)

            elif horizon_mode == "past":
                win = window_by_time(t_ref - h, t_ref)

            else:  # both
                win_past   = window_by_time(t_ref - h, t_ref)
                win_future = window_by_time(t_ref, t_ref + h)
                win = (win_past, win_future)

        else:
            # index-based window (only here we need start_i/end_i)
            if window_direction == "backward":
                start_i = max(0, target_i - window_cap)
                end_i = target_i
            else:
                start_i = target_i
                end_i = min(N - 1, target_i + window_cap)

            L = bisect_left(target_idx, start_i)
            R = bisect_right(target_idx, end_i)
            win = target_idx[L:R]

        # ----------------------------
        # compute DeltaT (same as before)
        # ----------------------------
        if use_horizon and horizon_mode == "both":
            win_past, win_future = win
            # 若兩邊都空就跳過
            if (not win_past) and (not win_future):
                continue

            # 先用「合併 window」估計 capacity → 再把 DeltaT 分配
            win_merged = (win_past + win_future)
            if not win_merged:
                continue

            # build merged bounds
            t_base, t_min, t_max = [], [], []
            for j in win_merged:
                dmm = rows[j].d_mm
                Fold = F_new[j] if F_new[j] > EPS else (rows[j].F if rows[j].F > EPS else Fmin)
                if dmm > EPS and Fold > EPS:
                    tb = 60.0 * dmm / Fold
                    t_base.append(tb)
                    t_min.append(60.0 * dmm / Fmax)
                    t_max.append(60.0 * dmm / Fmin)
                else:
                    tb = 0.0
                    t_base.append(tb)
                    t_min.append(tb)
                    t_max.append(tb)

            Cplus = sum(t_max[k] - t_base[k] for k in range(len(win_merged)))
            Cminus = sum(t_base[k] - t_min[k] for k in range(len(win_merged)))

            if u_des >= 0.0:
                DeltaT = min(u_des, Cplus)
            else:
                DeltaT = -min(-u_des, Cminus)

            if abs(DeltaT) <= 1e-12:
                continue

            wF = max(0.0, min(1.0, float(horizon_w_future)))
            wP = 1.0 - wF

            apply_window_adjust(win_past,   wP * DeltaT)
            apply_window_adjust(win_future, wF * DeltaT)
            continue

        else:
            # single window case: win is List[int]
            if not win:
                continue

            # time bounds
            t_base, t_min, t_max = [], [], []
            for j in win:
                dmm = rows[j].d_mm
                Fold = F_new[j] if F_new[j] > EPS else (rows[j].F if rows[j].F > EPS else Fmin)
                if dmm > EPS and Fold > EPS:
                    tb = 60.0 * dmm / Fold
                    t_base.append(tb)
                    t_min.append(60.0 * dmm / Fmax)
                    t_max.append(60.0 * dmm / Fmin)
                else:
                    tb = 0.0
                    t_base.append(tb)
                    t_min.append(tb)
                    t_max.append(tb)

            Cplus = sum(t_max[k] - t_base[k] for k in range(len(win)))
            Cminus = sum(t_base[k] - t_min[k] for k in range(len(win)))

            if u_des >= 0.0:
                DeltaT = min(u_des, Cplus)
            else:
                DeltaT = -min(-u_des, Cminus)

            if abs(DeltaT) <= 1e-12:
                continue

            # apply to single window using the helper
            apply_window_adjust(win, DeltaT)
            continue

    # 7) Global smoothing over F_new（可選）
    def smooth_feedrates(F_vals: List[float], passes: int) -> List[float]:
        if passes <= 0:
            return F_vals
        res = F_vals[:]
        for _ in range(passes):
            newF = res[:]
            for i in range(N):
                if not (rows[i].is_extrusion or rows[i].is_optimizable):
                    continue
                left = res[i - 1] if i - 1 >= 0 else res[i]
                right = res[i + 1] if i + 1 < N else res[i]
                newF[i] = 0.25 * left + 0.5 * res[i] + 0.25 * right
                newF[i] = max(Fmin, min(newF[i], Fmax))
            res = newF
        return res

    F_new = smooth_feedrates(F_new, smooth_passes)
    return F_new

def optimize_dwell_P_ms(
    rows: List[Row],
    P_current_ms: List[float],
    tau_current: List[float],
    tau_lo: float,
    tau_star: float,
    tau_hi: float,
    KpP: float = 0.2,
    dPmax_ratio: float = 0.10,
    Pmin_ms: float = 0.0,
    Pmax_ms: float = 600000.0,
) -> List[float]:
    """
    Dwell-only controller:
      - targets rows where cmd==G4 and is_optimizable
      - adjusts P (ms) directly (linear in time)
    """
    P_new = P_current_ms[:]

    for i, r in enumerate(rows):
        if not ((r.cmd or "").upper() == "G4" and r.is_optimizable):
            continue

        t = tau_current[i] if i < len(tau_current) else float("nan")
        if t is None or math.isnan(t):
            continue

        # distance to feasible window
        if t < tau_lo:
            err = (tau_lo - t)     # need MORE residence time -> increase P
        elif t > tau_hi:
            err = (tau_hi - t)     # negative -> decrease P
        else:
            err = 0.0

        if abs(err) <= EPS:
            continue

        # desired dwell change in ms
        dP = KpP * err * 1000.0

        # clamp per-iter change
        base = max(Pmin_ms, min(P_new[i], Pmax_ms))
        step_max = max(1.0, abs(base) * dPmax_ratio)
        dP = max(-step_max, min(step_max, dP))

        P_new[i] = max(Pmin_ms, min(Pmax_ms, base + dP))

    return P_new

# ---------- CSV Writers ----------
SEGIDX_RE = re.compile(r"\bseg_idx\s*=\s*(?P<id>\d+)\b", re.IGNORECASE)

def _extract_seg_idx_from_comment(comment: str):
    if not comment:
        return None
    m = SEGIDX_RE.search(comment)
    return int(m.group("id")) if m else None

# 2) 移除舊 seg_idx：這個 regex 不要捕捉群組，sub 時就不可能 "no such group"
SEGIDX_STRIP_RE = re.compile(r"(?:\s*\|\s*)?\bseg_idx\s*=\s*\d+\b", re.IGNORECASE)

def _strip_seg_idx(comment: str) -> str:
    if not comment:
        return ""
    c = SEGIDX_STRIP_RE.sub("", comment)
    # 清掉多餘的分隔符殘留
    c = re.sub(r"\s*\|\s*$", "", c).strip()
    return c

def _gcode_cmd_token(s: str) -> str:
    toks = s.strip().split()
    if not toks:
        return ""

    # Skip line number like: N123
    if toks[0].upper().startswith("N") and toks[0][1:].isdigit():
        toks = toks[1:]
        if not toks:
            return ""

    t = toks[0].upper()

    # Normalize G01 -> G1, G04 -> G4
    if t.startswith("G") and t[1:].isdigit():
        t = "G" + str(int(t[1:]))

    return t

def patch_gcode_by_segidx(
    gcode_template_path: str,
    gcode_out_path: str,
    rows: List[Row],
    F_vec: List[float],
    P_vec_ms: List[float],
    enable_travel_opt: bool = False,
    s_travel: float = 1.0,
) -> Tuple[int, int]:
    """
    Strategy A:
      - expected_count is computed from TEMPLATE gcode lines that are actually patchable:
          * F patch: G1 with E and seg_idx=... and rows[seg_idx] is (G1) and (extrusion or optimizable)
          * P patch: G4 with P and seg_idx=... and rows[seg_idx] is (G4) and (optimizable)
      - patch pass uses the exact same criteria -> expected == truly patchable
    """

    # map seg_idx -> row index (fast lookup)
    segidx_to_rowi_all = {r.seg_idx: i for i, r in enumerate(rows)}

    st = min(1.0, max(0.0, float(s_travel)))

    # ---------- PASS 1: compute expected_count from TEMPLATE (same criteria as patch) ----------
    expected = 0
    with open(gcode_template_path, "r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            raw = line.rstrip("\n")
            code_part, sep, comment_part = raw.partition(";")
            s = code_part.strip()
            c = comment_part.strip() if sep else ""

            if not s:
                continue

            seg_idx = _extract_seg_idx_from_comment(c)
            if seg_idx is None:
                continue

            cmd = _gcode_cmd_token(s)
            parts = s.split()
            has_E = any(p.upper().startswith("E") for p in parts)
            has_P = any(p.upper().startswith("P") for p in parts)

            # seg_idx must exist in rows
            if seg_idx not in segidx_to_rowi_all:
                continue
            i = segidx_to_rowi_all[seg_idx]
            rr = rows[i]

            # F patch expected: TEMPLATE must be G1 + E, and ROW must be target (extrusion/optimizable)
            if cmd == "G1" and has_E and ((rr.cmd or "").upper() == "G1") and (rr.is_extrusion or rr.is_optimizable):
                expected += 1
                continue

            # P patch expected: TEMPLATE must be G4 + P, and ROW must be optimizable dwell
            if cmd == "G4" and has_P and ((rr.cmd or "").upper() == "G4") and rr.is_optimizable:
                expected += 1
                continue

    # ---------- PASS 2: patch with the SAME criteria ----------
    patched = 0
    with open(gcode_template_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(gcode_out_path, "w", encoding="utf-8", newline="") as fout:

        for line in fin:
            raw = line.rstrip("\n")
            code_part, sep, comment_part = raw.partition(";")
            s = code_part.strip()
            c = comment_part.strip() if sep else ""

            if not s:
                fout.write(line)
                continue

            seg_idx = _extract_seg_idx_from_comment(c)
            cmd = _gcode_cmd_token(s)
            parts = s.split()
            has_E = any(p.upper().startswith("E") for p in parts)
            has_P = any(p.upper().startswith("P") for p in parts)

            # ---------- (1) F patch: G1 + E ----------
            if seg_idx is not None and cmd == "G1" and has_E and seg_idx in segidx_to_rowi_all:
                i = segidx_to_rowi_all[seg_idx]
                rr = rows[i]
                if ((rr.cmd or "").upper() == "G1") and (rr.is_extrusion or rr.is_optimizable):
                    F_new = round(float(F_vec[i]), 2)

                    new_parts = []
                    hasF = False
                    for p in parts:
                        if p.upper().startswith("F"):
                            new_parts.append(f"F{F_new}")
                            hasF = True
                        else:
                            new_parts.append(p)
                    if not hasF:
                        new_parts.append(f"F{F_new}")

                    new_code = " ".join(new_parts)
                    if sep:
                        fout.write(f"{new_code} ; {c}\n" if c else f"{new_code}\n")
                    else:
                        fout.write(f"{new_code}\n")

                    patched += 1
                    continue

            # ---------- (2) P patch: G4 + P ----------
            if seg_idx is not None and cmd == "G4" and has_P and seg_idx in segidx_to_rowi_all:
                i = segidx_to_rowi_all[seg_idx]
                rr = rows[i]
                if ((rr.cmd or "").upper() == "G4") and rr.is_optimizable:
                    P_new = int(round(float(P_vec_ms[i])))

                    new_parts = []
                    hasP2 = False
                    for p in parts:
                        if p.upper().startswith("P"):
                            new_parts.append(f"P{P_new}")
                            hasP2 = True
                        else:
                            new_parts.append(p)
                    if not hasP2:
                        new_parts.append(f"P{P_new}")

                    new_code = " ".join(new_parts)
                    if sep:
                        fout.write(f"{new_code} ; {c}\n" if c else f"{new_code}\n")
                    else:
                        fout.write(f"{new_code}\n")

                    patched += 1
                    continue

            # ---------- (3) travel scaling (unchanged) ----------
            if enable_travel_opt:
                is_g0 = (cmd == "G0")
                is_g1_travel = (cmd == "G1" and not has_E)
                if is_g0 or is_g1_travel:
                    new_parts = []
                    changed = False
                    for p in parts:
                        if p.upper().startswith("F"):
                            try:
                                val = float(p[1:])
                                new_parts.append(f"F{round(val * st, 2)}")
                                changed = True
                            except:
                                new_parts.append(p)
                        else:
                            new_parts.append(p)

                    if changed:
                        new_code = " ".join(new_parts)
                        if sep:
                            fout.write(f"{new_code} ; {c}\n" if c else f"{new_code}\n")
                        else:
                            fout.write(f"{new_code}\n")
                        continue

            fout.write(line)

    return patched, expected


def write_iteration_csv(path: str,
                        rows: List[Row],
                        F_old: List[float],
                        F_new: List[float],
                        tau_old: List[float],
                        tau_new: List[float],
                        tau_star: float,
                        tau_lo: float,
                        tau_hi: float,
                        tau_grid: np.ndarray,
                        alpha_grid: np.ndarray):
    """
    Per-iteration CSV with one extra alpha column (based on tau_new).
    """
    header = ["seg_idx","F_old","F_new","tau_old","tau_new","alpha","error","within%","optimizable"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for i, r in enumerate(rows):
            is_target = (r.is_extrusion or r.is_optimizable)

            tau_n = tau_new[i] if i < len(tau_new) else float("nan")
            alpha_n = alpha_from_tau_lookup(tau_n, tau_grid, alpha_grid)

            if is_target and (tau_n is not None) and (not math.isnan(tau_n)):
                err = tau_star - tau_n
                tau_n_c = tau_to_const(tau_n)
                tau_lo_c = tau_to_const(tau_lo)
                tau_hi_c = tau_to_const(tau_hi)

                within = 100.0 if tau_lo_c <= tau_n_c <= tau_hi_c else 0.0
                
            else:
                err = 0.0
                within = 100.0

            w.writerow([
                r.seg_idx,
                r2(F_old[i]),
                r2(F_new[i]),
                r2(tau_old[i]),
                r2(tau_new[i]),
                round(alpha_n, 5) if not math.isnan(alpha_n) else "",
                r2(err),
                r2(within),
                1 if is_target else 0
            ])

def write_summary_csv(path: str,
                      history: List[Dict[str,float]],
                      cli_used: str = ""):
    """Write the optimization history summary to a CSV file.

    Each record in `history` represents a single iteration and includes
    global error metrics, τ statistics, window_cap value, and iteration
    runtime. This CSV gives a high-level view of optimization progress.

    Columns:
        iteration         - Iteration index
        global_error      - Sum of |tau - tau_star| across extrusion rows
        tau_min           - Minimum τ among extrusion rows
        tau_max           - Maximum τ among extrusion rows
        tau_mean          - Mean τ among extrusion rows
        within%           - Percentage of extrusion rows within tolerance
        window_cap        - Window length used in that iteration
        iteration_runtime - Execution time for the iteration (seconds)

    Args:
        path (str): Output CSV path.
        history (list[dict]): List of iteration statistics dictionaries.
    """   
    header = ["iteration","global_error","tau_min","tau_max","tau_mean","within%","window_cap","iteration_runtime"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for rec in history:
            w.writerow([
                int(rec["iteration"]),
                r2(rec["global_error"]),
                r2(rec["tau_min"]),
                r2(rec["tau_max"]),
                r2(rec["tau_mean"]),
                r2(rec["within%"]),
                int(rec["window_cap"]),
                r2(rec["iteration_runtime"]),
            ])
        if cli_used:
            w.writerow([])
            w.writerow(["CLI", cli_used])

def write_runtime_log(path: str,
                      runtime_log: List[Dict[str,float]]):
    """Write per-iteration runtime measurements to a CSV file.

    This file tracks raw iteration execution time and the total runtime entry.

    Columns:
        iteration      - Iteration index or "TOTAL"
        runtime_sec    - Wall-clock time in seconds

    Args:
        path (str): Output CSV file path.
        runtime_log (list[dict]): List containing runtime measurements.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["iteration", 
                  "runtime_sec", 
                  "totaltime_sec", 
                  "mode", 
                  "tau_min", 
                  "tau_max", 
                  "tau_mean", 
                  "improve", 
                  "within", 
                  "window_cap", 
                  "phase",
                  "violations", "v_max", "p95", "cvar95"]
        w.writerow(header)
        for rec in runtime_log:
            w.writerow([rec.get("iteration", ""),
                        r2(rec.get("runtime_sec", 0.0)),
                        r2(rec.get("totaltime_sec", 0.0)),
                        rec.get("mode", ""),
                        r2(rec.get("tau_min", 0.0)),
                        r2(rec.get("tau_max", 0.0)),
                        r2(rec.get("tau_mean", 0.0)),
                        r2(rec.get("improve", 0.0)),
                        r2(rec.get("within", 0.0)),
                        rec.get("window_cap", ""),
                        rec.get("phase", ""),
                        rec.get("violations", 0),
                        r2(rec.get("v_max", 0.0)),
                        r2(rec.get("p95", 0.0)),
                        r2(rec.get("cvar95", 0.0)),
                        ])

# ---------- G-code patch（最終一次） ----------
def patch_gcode_feedrates(
    gcode_in_path: str,
    gcode_out_path: str,
    rows: List[Row],
    F_final: List[float],
    P_final_ms: List[float] = None,
    enable_travel_opt: bool = False,
    s_travel: float = 1.0
) -> None:
    out = open(gcode_out_path, "w", encoding="utf-8", newline="")
    fin = open(gcode_in_path, "r", encoding="utf-8", errors="ignore")

    # target indices for extrusion (G1E) by CSV flags
    extr_target = [i for i,r in enumerate(rows) if (r.cmd or "").upper()=="G1" and (r.is_extrusion or r.is_optimizable)]
    F_seq = [F_final[i] for i in extr_target]
    used_F = 0

    # dwell targets (G4 optimizable)
    dwell_target = [i for i, r in enumerate(rows) if ((r.cmd or "").upper() == "G4" and r.is_optimizable)]
    P_seq = None
    if P_final_ms is not None:
        P_seq = [P_final_ms[i] for i in dwell_target]
    used_P = 0

    for line in fin:
        raw = line.rstrip("\n")
        code_part, sep, comment_part = raw.partition(";")
        s = code_part.strip()
        c = comment_part.strip() if sep else ""

        cmd = _gcode_cmd_token(s)
        parts = s.split()
        has_E = any(p.upper().startswith("E") for p in parts)
        has_P = any(p.upper().startswith("P") for p in parts)

        # 1) extrusion: G1 + E -> patch F
        if cmd == "G1" and has_E and used_F < len(F_seq):
            row_i = extr_target[used_F]
            seg_idx = rows[row_i].seg_idx
            Fset = round(float(F_seq[used_F]), 2)
            used_F += 1

            new_parts = []
            hasF = False
            for p in parts:
                if p.upper().startswith("F"):
                    new_parts.append(f"F{Fset}")
                    hasF = True
                else:
                    new_parts.append(p)
            if not hasF:
                new_parts.append(f"F{Fset}")
            new_code = " ".join(new_parts)

            if sep:
                out.write(f"{new_code} ; {c} | seg_idx={seg_idx}\n" if c else f"{new_code} ; seg_idx={seg_idx}\n")
            else:
                out.write(f"{new_code} ; seg_idx={seg_idx}\n")
            continue

        # 2) dwell: G4 + P -> patch P(ms)
        if P_seq is not None and cmd == "G4" and has_P and used_P < len(P_seq):
            row_i = dwell_target[used_P]
            seg_idx = rows[row_i].seg_idx
            Pset = int(round(float(P_seq[used_P])))
            used_P += 1

            new_parts = []
            hasP = False
            for p in parts:
                if p.upper().startswith("P"):
                    new_parts.append(f"P{Pset}")
                    hasP = True
                else:
                    new_parts.append(p)
            if not hasP:
                new_parts.append(f"P{Pset}")
            new_code = " ".join(new_parts)

            if sep:
                out.write(f"{new_code} ; {c} | seg_idx={seg_idx}\n" if c else f"{new_code} ; seg_idx={seg_idx}\n")
            else:
                out.write(f"{new_code} ; seg_idx={seg_idx}\n")
            continue

        # 3) travel scaling (same as你原本，但用 token 判斷更穩)
        if enable_travel_opt and s:
            has_E2 = any(p.upper().startswith("E") for p in parts)
            is_travel = (cmd == "G0") or (cmd == "G1" and not has_E2)
            if is_travel:
                st = min(1.0, max(0.0, float(s_travel)))
                new_parts = []
                changed = False
                for p in parts:
                    if p.upper().startswith("F"):
                        try:
                            val = float(p[1:])
                            new_parts.append(f"F{round(val * st, 2)}")
                            changed = True
                        except:
                            new_parts.append(p)
                    else:
                        new_parts.append(p)

                if changed:
                    new_code = " ".join(new_parts)
                    out.write(f"{new_code} ; {c}\n" if sep and c else f"{new_code}\n")
                    continue

        out.write(line)

    fin.close()
    out.close()

    print(f"✅ Patched extrusion F: {used_F}/{len(F_seq)} lines.")
    if P_seq is not None:
        print(f"✅ Patched dwell P: {used_P}/{len(P_seq)} lines.")
    print(f"➡ Output: {gcode_out_path}")

# ---------- Main ----------

def main():
    """Run the residence-time optimization CLI pipeline.

    This entry point parses command-line arguments, configures logging,
    loads the segmented CSV, and iteratively adjusts feedrates to drive
    extrusion residence times toward a target value. It supports both a
    fast internal FIFO approximation and an external “true” FIFO model
    (segment_fifo_builder_v6), with optional cyclic adaptive window control.

    Main steps:
        1. Parse CLI arguments and create the output directory.
        2. Mirror stdout/stderr into a timestamped log file.
        3. Load segments from the input CSV and infer an initial τ₀.
        4. Initialize feedrates and window_cap based on nozzle volume.
        5. For each iteration:
            - Simulate residence times (FAST or TRUE FIFO).
            - Compute a new feedrate field via optimize_feedforward().
            - Re-simulate τ with updated feedrates.
            - Update global error, τ statistics, and within% coverage.
            - Optionally adapt window_cap (rise/fall cycle logic).
            - Optionally save per-iteration CSV and runtime log.
        6. Track the best iteration (minimal global error).
        7. Write a final summary CSV and runtime log.
        8. Optionally patch the segmented G-code with:
            - best feedrates (…_best.gcode)
            - final feedrates (gcode_out)

    Key CLI arguments:
        --input (str): Input segments CSV (segment_fifo_builder_v6 output).
        --tau (float): Target residence time τ* in seconds.
        --buffer (float): Allowed relative τ deviation (±ratio).
        --vnozzle (float): Mixing chamber volume V_nozzle in mm³.
        --dnozzle (float): Nozzle diameter in mm.
        --use_boundary_window (bool): If set, windows stop at V_nozzle boundaries.
        --window_cap (int): Initial forward window size (adapted over time).
        --Kp, --Kd (float): PD controller gains for ΔT correction.
        --Fmin, --Fmax (float): Feedrate bounds.
        --dFmax (float): Max relative feedrate step per row.
        --deltaT_cap_ratio (float): Max |ΔT| as a fraction of τ*.
        --max_iter (int): Maximum number of optimization iterations.
        --out_dir (str): Output directory for logs and CSV files.
        --out_prefix (str): Prefix for per-iteration CSV filenames.
        --summary (str): Summary CSV filename.
        --enable_timer (bool): If set, record per-iteration runtimes.
        --adaptive_window (bool): Enable cyclic window_cap adaptation.
        --cooldown (int): Min iterations between window_cap changes.
        --stall_improve (float): Relative improvement threshold for stall detection.
        --stall_runs (int): Required consecutive stalls to flip phase.
        --save_intermediate (bool): Save per-iteration CSV and G-code if enabled.
        --fifo_mode ({"fast","true"}): FAST = internal model, TRUE = v6 FIFO.
        --true_every (int): Use TRUE FIFO every N iterations when enabled.
        --gcode_in (str): Input segmented G-code path (required for fifo_mode=true).
        --gcode_out (str): Final patched G-code path (used with --save_intermediate).

    Behavior:
        * On empty input or missing gcode_in in TRUE mode, exits with a message.
        * On KeyboardInterrupt, falls back to the latest / best available solution.
        * Always writes the summary CSV; runtime log and G-code outputs are optional.

    Returns:
        None
    """
    ap = argparse.ArgumentParser(description="Residence-time optimizer with cyclic adaptive window.")
    ap.add_argument("--input", default="segments.csv", help="Input segments CSV")
    ap.add_argument("--temp_c", type=float, required=True,
                    help="Fixed process / nozzle temperature for alpha<->tau conversion [°C].")

    ap.add_argument("--alpha_min", type=float, required=True,
                    help="Lower alpha bound (supports 5 decimals). Example: 0.18000")
    ap.add_argument("--alpha_ideal", type=float, required=True,
                    help="Ideal alpha target (supports 5 decimals). Example: 0.22000")
    ap.add_argument("--alpha_max", type=float, required=True,
                    help="Upper alpha bound (supports 5 decimals). Example: 0.26000")

    ap.add_argument("--tau_lookup_max", type=float, default=20000.0,
                    help="Upper tau search limit [s] used to build alpha<->tau lookup.")
    ap.add_argument("--vnozzle", type=float, default=500.0, help="Mixing volume V_nozzle (mm^3)") 
    ap.add_argument("--dnozzle", type=float, default=0.5, help="Nozzle diameter (mm)")

    ap.add_argument("--use_boundary_window", action="store_true",
                    help="Forward window ends at next V_nozzle boundary if set; else fixed length.")
    
    ap.add_argument("--horizon_sec", type=float, default=0.0,
                help="Time-horizon window length in seconds (0 disables time-horizon mode).")
    ap.add_argument("--horizon_mode", choices=["future", "past", "both"], default="future",
                    help="Time-horizon mode: future-only, past-only, or both-sides.")
    ap.add_argument("--horizon_w_future", type=float, default=0.5,
                    help="When horizon_mode=both: fraction of DeltaT applied to future window (0..1).")

    ap.add_argument("--window_direction", choices=["forward", "backward"], default="backward",
                    help="Direction of optimization window. "
                        "'forward' distributes ΔT to later segments in the same cycle; "
                        "'backward' distributes ΔT to earlier segments in the same cycle.")
    ap.add_argument("--window_cap", type=int, default=250,
                    help="Initial window_cap (will be clamped into [lower, upper]).")


    ap.add_argument("--Kp", type=float, default=0.5)
    ap.add_argument("--Kd", type=float, default=0.1)
    ap.add_argument("--Fmin", type=float, default=500.0)
    ap.add_argument("--Fmax", type=float, default=6000.0)
    ap.add_argument("--dFmax", type=float, default=0.5,
                    help="Max relative change per row.")
    ap.add_argument("--deltaT_cap_ratio", type=float, default=0.5,
                    help="Max |ΔT| per target as fraction of tau*.")

    ap.add_argument("--max_iter", type=int, default=50)
    ap.add_argument("--out_dir", default="optimization_outputs")
    ap.add_argument("--out_prefix", default="iteration_")
    ap.add_argument("--summary", default="summary.csv")

    ap.add_argument("--enable_timer", action="store_true")
    ap.add_argument("--adaptive_window", action="store_true")
    ap.add_argument("--cooldown", type=int, default=5,
                    help="Min iterations between window_cap adjustments.")
    # cyclic rule uses固定閾值: (prev_ge - ge)/prev_ge < 0.002 持續 5 次
    #ap.add_argument("--stall_improve", type=float, default=0.002)
    ap.add_argument("--stall_runs", type=int, default=5)

    ap.add_argument("--save_intermediate", action="store_true",
                help="Save per-iteration CSV and G-code outputs if set. Otherwise only summary will be written.")
    
    ap.add_argument("--fifo_mode", choices=["fast", "true"], default="fast",
                help="Simulation mode: 'fast' = internal, 'true' = external FIFO (v6).")
    ap.add_argument("--true_every", type=int, default=10,
                help="TRUE scheduling switch: "
                    "set 1 for TRUE-only (every iteration uses physical engine); "
                    "set N>=2 for HYBRID (use TRUE every N iterations, FAST in between).")

    #------ NEW ------
    ap.add_argument("--Ki", type=float, default=0.2,
                    help="Cycle-level integrator gain (0.05-0.3 recommended).")
    ap.add_argument("--smooth_passes", type=int, default=1,
                    help="Number of global smoothing passes (0-2).")
    #-----------------
    
    ap.add_argument("--gcode_in", default=None)
    ap.add_argument("--gcode_out", default=None)
    
    #------ Travel Speed ------
    ap.add_argument("--enable_travel_opt", action="store_true",
                    help="Enable travel optimization via a single global scaling factor (only slows travel).")

    ap.add_argument("--travel_gain", type=float, default=0.20,
                    help="Travel scale controller gain (small). Typical 0.05-0.3.")

    ap.add_argument("--travel_step_max", type=float, default=0.02,
                    help="Max relative decrease of s_travel per iteration (e.g. 0.02 = 2%%).")

    ap.add_argument("--travel_s_min", type=float, default=0.20,
                    help="Lower bound of travel scaling factor s_travel (>=0, <=1).")

    ap.add_argument("--travel_dFmax", type=float, default=0.05,
                    help="Per-row max slow-down ratio for travel per iteration (e.g. 0.05 = 5%%). Smaller than extrusion.")
    #--------------------------

    args = ap.parse_args()
    if not (0.0 <= args.alpha_min <= 1.0 and 0.0 <= args.alpha_ideal <= 1.0 and 0.0 <= args.alpha_max <= 1.0):
        print("⛔ alpha values must be within [0, 1].")
        return

    if not (args.alpha_min <= args.alpha_ideal <= args.alpha_max):
        print("⛔ Require alpha_min <= alpha_ideal <= alpha_max.")
        return
    if args.true_every is None or args.true_every < 1:
        args.true_every = 1
    
    cli_string = " ".join(sys.argv)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = read_segments_csv(args.input)
    # ---------------------------------------------------------
    # Build alpha<->tau lookup at fixed T
    # ---------------------------------------------------------
    tau_grid_lookup, alpha_grid_lookup = build_alpha_tau_lookup(
        temp_c=args.temp_c,
        params=ALPHA_MODEL_PARAMS,
        tau_upper_s=args.tau_lookup_max,
        alpha_in=ALPHA_IN_DEFAULT,
        n_eval=4000
    )

    tau_lo = tau_from_alpha_lookup(args.alpha_min, tau_grid_lookup, alpha_grid_lookup)
    tau_star = tau_from_alpha_lookup(args.alpha_ideal, tau_grid_lookup, alpha_grid_lookup)
    tau_hi = tau_from_alpha_lookup(args.alpha_max, tau_grid_lookup, alpha_grid_lookup)

    print(f"🎯 Alpha target window @ T={args.temp_c:.2f} °C")
    print(f"   alpha_min   = {args.alpha_min:.5f} -> tau_lo    = {tau_lo:.5f} s")
    print(f"   alpha_ideal = {args.alpha_ideal:.5f} -> tau_star  = {tau_star:.5f} s")
    print(f"   alpha_max   = {args.alpha_max:.5f} -> tau_hi    = {tau_hi:.5f} s")
    # Precompute cycle/boundary maps once (FAST mode stable)
    boundaries, next_boundary_of = compute_boundaries_and_next_map(rows, args.vnozzle, args.dnozzle)
    _, cycle_start_of = compute_cycle_start_map(rows, args.vnozzle, args.dnozzle)

    if not rows:
        print("⛔ Empty input CSV.")
        return
    
    if args.fifo_mode == "true" and not args.gcode_in:
        print("⛔ Error: --gcode_in must be specified when using --fifo_mode true.")
        return
    # ✅ TRUE 模式：建立固定 template（帶 seg_idx tag）
    if args.fifo_mode == "true":
        args.gcode_template = args.gcode_in

    else:
        args.gcode_template = None

    
    cap_upper, cap_lower = compute_cap_bounds_from_cycle(rows, args.vnozzle, args.dnozzle)
    window_cap = max(cap_lower, min(args.window_cap, cap_upper))

    tau0 = infer_tau0_from_csv(rows, fallback=tau_star)
    F_curr = [r.F if r.F > EPS else args.Fmin for r in rows]
    # --- dwell control vector (ms) ---
    # default: use CSV t_s -> ms for G4 rows; otherwise 0
    P_curr_ms = [0.0] * len(rows)
    for i, r in enumerate(rows):
        if (r.cmd or "").upper() == "G4" and r.is_optimizable:
            # CSV t_s is seconds
            P_curr_ms[i] = max(0.0, float(r.t_s) * 1000.0)

    F0_travel = [infer_F0_row(r, args.Fmin) for r in rows]

    total_start = time.time()
    summary_hist = []
    runtime_log = []

    # =========================================================
    # [Iter 000] BASELINE: use original CSV tau_s (no simulate)
    # =========================================================
    iter0_start = time.time()

    tau_000 = []
    for r in rows:
        v = float(getattr(r, "tau_s", float("nan")))
        if v <= EPS:
            v = float("nan")
        tau_000.append(v)

    ge0 = global_error_abs_sum(tau_000, rows, tau_star)
    tau_min0, tau_max0, tau_mean0 = tau_stats(tau_000, rows)
    within0 = within_percentage_bounds(tau_000, rows, tau_lo, tau_hi)

    v_cnt0, v_sum0, v_max0, p95_0, cvar95_0 = risk_metrics(tau_000, rows, tau_lo, tau_hi)

    iter0_runtime = time.time() - iter0_start
    total_elapsed0 = time.time() - total_start

    print(
        f"[Iter 000] mode=FAST_CSV_INIT | Δt={iter0_runtime:.3f}s | Total={total_elapsed0:.1f}s | "
        f"τ_min={r2(tau_min0):>6}, τ_max={r2(tau_max0):>6}, τ_mean={r2(tau_mean0):>6} | "
        f"error={ge0:.2f} | Δerror=+0.00 | ratio=1.00000 | "
        f"within={r2(within0):>6}% | v_cnt={int(v_cnt0):>4} | v_max={r2(v_max0):>6}s | "
        f"p95={r2(p95_0):>6}s | cvar95={r2(cvar95_0):>6}s"
    )

    runtime_log.append({
        "iteration": 0,
        "runtime_sec": iter0_runtime,
        "totaltime_sec": total_elapsed0,
        "mode": "FAST_CSV_INIT",
        "tau_min": tau_min0,
        "tau_max": tau_max0,
        "tau_mean": tau_mean0,
        "improve": 1.0,
        "within": within0,
        "window_cap": window_cap,
        "phase": "init",
        "violations": int(v_cnt0),
        "v_max": float(v_max0),
        "p95": float(p95_0),
        "cvar95": float(cvar95_0),
    })

    summary_hist.append({
        "iteration": 0,
        "global_error": ge0,
        "tau_min": tau_min0,
        "tau_max": tau_max0,
        "tau_mean": tau_mean0,
        "within%": within0,
        "window_cap": window_cap,
        "iteration_runtime": iter0_runtime
    })

    prev_ge = ge0


    # ---- travel scaling state ----
    s_travel = 1.0
    F0_travel = [infer_F0_row(r, args.Fmin) for r in rows]  # keep original travel structure

    # --- best tracking ---
    def is_better_candidate(within, v_cnt, v_max, ge,
                        best_within, best_v_cnt, best_v_max, best_ge) -> bool:
        # within higher is better; others lower is better
        cand = (float(within), -int(v_cnt), -float(v_max), -float(ge))
        best = (float(best_within), -int(best_v_cnt), -float(best_v_max), -float(best_ge))
        return cand > best

    
    # baseline 不算 TRUE best（因為是 CSV init），所以 best_iter_true = -1
    best_F_true = None
    best_P_true = None
    best_F_fast = None
    best_P_fast = None
    
    best_iter_true = -1
    best_ge_true = float("inf")
    best_within_true = -1.0
    best_v_cnt_true = float("inf")
    best_v_max_true = float("inf")

    # optional: keep a FAST-best for debugging/inspection (won't be used for output in TRUE mode)
    best_iter_fast = -1
    best_ge_fast = float("inf")
    best_within_fast = -1.0
    best_v_cnt_fast = float("inf")
    best_v_max_fast = float("inf")

    phase = "rise"
    low_improve_count = 0
    prev_ge = None
    last_adjust_iter = 0
    check_count = 0

    F_final = F_curr[:]

    try:
        for it in range(1, args.max_iter + 1):
            iter_start = time.time()

            # 1) 取得 tau_old（依 TRUE-only / HYBRID 策略）
            do_true = (args.fifo_mode == "true") and (args.true_every == 1 or (it % args.true_every == 0))

            # ✅ Step A：第一輪一律用輸入 CSV 當 tau_old / t_in_old / t_out_old（包含 TRUE-only）
            if it == 1:
                t_in_old  = [float(r.t_in)  if getattr(r, "t_in", 0.0)  is not None else float("nan") for r in rows]
                t_out_old = [float(r.t_out) if getattr(r, "t_out", 0.0) is not None else float("nan") for r in rows]
                tau_old = []
                for r in rows:
                    v = float(getattr(r, "tau_s", float("nan")))
                    if v <= EPS:
                        v = float("nan")
                    tau_old.append(v)
                mode = "CSV_INIT"

            else:
                # it >= 2：依策略決定用 TRUE 或 FAST 取得 tau_old
                if args.fifo_mode == "true" and do_true:
                    mode = "TRUE"
                    ok_old, t_in_old, t_out_old, tau_old, _, _ = simulate_fifo_v6_wrapper(
                        gcode_template_path=args.gcode_template,
                        rows=rows,
                        F_vec=F_curr,
                        P_vec_ms=P_curr_ms,
                        V_nozzle=args.vnozzle,
                        d_nozzle=args.dnozzle,
                        iteration=it,
                        out_root=args.out_dir,
                        enable_travel_opt=args.enable_travel_opt,
                        s_travel=s_travel
                    )
                    if not ok_old:
                        # TRUE 引擎失敗 → 本輪跳過更新，保留 F_curr
                        iter_runtime = time.time() - iter_start
                        print(f"⚠️  [Iter {it:03d}] TRUE failed on tau_old. Skipping update, keeping previous F_curr.")
                        continue

                else:
                    # FAST（含 TRUE-HYBRID 的 FAST 輪）
                    # apply current dwell P to rows.t_s (so FAST simulation reflects it)
                    for i, r in enumerate(rows):
                        if (r.cmd or "").upper() == "G4" and r.is_optimizable:
                            r.t_s = float(P_curr_ms[i]) / 1000.0

                    t_in_old, t_out_old, tau_old = simulate_periodic_alignment(
                        rows, F_curr, args.vnozzle, args.dnozzle, tau0,
                        enable_travel_opt=args.enable_travel_opt
                    )
                    mode = "FAST"

            # 2) 計算新 F（先 F）
            F_next_base = optimize_feedforward_hybrid(
                rows=rows,
                F_current=F_curr,
                tau_current=tau_old,
                tau_lo=tau_lo,
                tau_star=tau_star,
                tau_hi=tau_hi,
                V_nozzle=args.vnozzle,
                d_nozzle=args.dnozzle,
                boundaries=boundaries,
                next_boundary_of=next_boundary_of,
                cycle_start_of=cycle_start_of,
                window_cap=window_cap,
                use_boundary_window=args.use_boundary_window,
                window_direction=args.window_direction,
                t_out_current=t_out_old,
                horizon_sec=args.horizon_sec,
                horizon_mode=args.horizon_mode,
                horizon_w_future=args.horizon_w_future,
                Kp=args.Kp,
                Kd=args.Kd,
                Ki_cycle=args.Ki,
                Fmin=args.Fmin,
                Fmax=args.Fmax,
                dFmax_ratio=args.dFmax,
                deltaT_cap_ratio=args.deltaT_cap_ratio,
                smooth_passes=args.smooth_passes
            )

            # 2.5) 再 dwell（第二）
            P_next_ms = optimize_dwell_P_ms(
                rows=rows,
                P_current_ms=P_curr_ms,
                tau_current=tau_old,
                tau_lo=tau_lo,
                tau_star=tau_star,
                tau_hi=tau_hi,
                KpP=0.15,
                dPmax_ratio=0.10,
                Pmin_ms=0.0,
                Pmax_ms=600000.0
            )

            # 2.8) 最後 travel（第三）
            if args.enable_travel_opt:
                _, _, tau_mean_old = tau_stats(tau_old, rows)
                s_travel = update_travel_scale(
                    s_travel=s_travel,
                    tau_mean=tau_mean_old,
                    tau_lo=tau_lo,
                    tau_star=tau_star,
                    travel_gain=args.travel_gain,
                    travel_step_max=args.travel_step_max,
                    travel_s_min=args.travel_s_min
                )

            F_next = apply_travel_scale_to_F(
                rows=rows,
                F_curr=F_curr,
                F_next=F_next_base,
                F0_travel=F0_travel,
                s_travel=s_travel,
                travel_dFmax_ratio=args.travel_dFmax,
                Fmin=args.Fmin,
                Fmax=args.Fmax,
                enable=args.enable_travel_opt
            )


            # 3) 取得 tau_new（同策略：TRUE 輪用 TRUE，HYBRID 的 FAST 輪用 FAST）
            if args.fifo_mode == "true" and do_true:
                mode = "TRUE"
                ok_new, _, _, tau_new, _, _ = simulate_fifo_v6_wrapper(
                    gcode_template_path=args.gcode_template,
                    rows=rows,
                    F_vec=F_next,
                    P_vec_ms=P_next_ms,
                    V_nozzle=args.vnozzle,
                    d_nozzle=args.dnozzle,
                    iteration=it,
                    out_root=args.out_dir,
                    enable_travel_opt=args.enable_travel_opt,
                    s_travel=s_travel
                )
                if not ok_new:
                    # ✅ 你要求 #2：TRUE 引擎失敗 → 本輪跳過更新，保留 F_curr
                    iter_runtime = time.time() - iter_start
                    print(f"⚠️  [Iter {it:03d}] TRUE failed on tau_new. Skipping update, keeping previous F_curr.")
                    continue
            else:
                # FAST（含 TRUE-HYBRID 的 FAST 輪）
                _, _, tau_new = simulate_periodic_alignment(
                    rows, F_next, args.vnozzle, args.dnozzle, tau0,
                    enable_travel_opt=args.enable_travel_opt
                )
            
            ge = global_error_abs_sum(tau_new, rows, tau_star)
            tau_min, tau_max, tau_mean = tau_stats(tau_new, rows)
            within = within_percentage_bounds(tau_new, rows, tau_lo, tau_hi)
            v_cnt, v_sum, v_max, p95_abs, cvar95 = risk_metrics(tau_new, rows, tau_lo, tau_hi)
            
            # ------------------------------------------------------------
            # Window-distance metrics based on tau_new
            # ------------------------------------------------------------
            def win_dist(t):
                if t is None or math.isnan(t):
                    return 0.0
                if t < tau_lo:
                    return (tau_lo - t)
                if t > tau_hi:
                    return (t - tau_hi)
                return 0.0

            # reuse your boundaries to rebuild cycle_of (same as hybrid)
            cycle_of_iter = [-1] * len(rows)
            if boundaries:
                bptr = 0
                current_cycle = 0
                current_boundary = boundaries[bptr]
                started = False
                for i, r in enumerate(rows):
                    if not (r.is_extrusion or r.is_optimizable):
                        continue
                    if not started:
                        started = True
                        current_cycle = 0
                    cycle_of_iter[i] = current_cycle
                    if i == current_boundary:
                        bptr += 1
                        current_cycle += 1
                        current_boundary = boundaries[bptr] if bptr < len(boundaries) else None
            else:
                for i, r in enumerate(rows):
                    if (r.is_extrusion or r.is_optimizable):
                        cycle_of_iter[i] = 0

            total_err = 0.0
            cycle_sum = {}
            cycle_cnt = {}
            cycle_worst_dist = {}
            cycle_worst_i = {}

            for i, r in enumerate(rows):
                cid = cycle_of_iter[i]
                if cid < 0:
                    continue
                if i >= len(tau_new):
                    continue
                d = win_dist(tau_new[i])
                if d <= EPS:
                    continue

                total_err += d
                cycle_sum[cid] = cycle_sum.get(cid, 0.0) + d
                cycle_cnt[cid] = cycle_cnt.get(cid, 0) + 1

                if (cid not in cycle_worst_dist) or (d > cycle_worst_dist[cid]):
                    cycle_worst_dist[cid] = d
                    cycle_worst_i[cid] = i

            # define cycle_err as MEAN distance (stable)
            max_cycle_err = 0.0
            max_cycle_id = None

            for cid in cycle_sum.keys():
                n = cycle_cnt.get(cid, 0)
                c_err = (cycle_sum[cid] / n) if n > 0 else 0.0
                if c_err > max_cycle_err:
                    max_cycle_err = c_err
                    max_cycle_id = cid

            # pick worst segment inside the worst cycle (safe, no continue)
            if max_cycle_id is not None and (max_cycle_id in cycle_worst_i):
                i_star = cycle_worst_i[max_cycle_id]
                worst_segidx = rows[i_star].seg_idx
                worst_dist = cycle_worst_dist.get(max_cycle_id, 0.0)

                # optional: if you only want extrusion worst in print, don't kill the iteration
                if not rows[i_star].is_extrusion:
                    i_star = None
                    worst_segidx = None
                    worst_dist = 0.0
            else:
                i_star = None
                worst_segidx = None
                worst_dist = 0.0

    
            # ✅ HYBRID best update gate:
            # - fast mode: allow best update
            # - true mode: allow best update only on TRUE iterations
            # - true-hybrid: only TRUE iterations should affect best
            if args.fifo_mode == "true":
                allow_best_update = (mode == "TRUE")   # 只在 TRUE 輪更新 best
            else:
                allow_best_update = True



            # --- debug: locate tau_min (extrusion only) ---
            extr_idx = [i for i, r in enumerate(rows)
                        if r.is_extrusion and i < len(tau_new) and (not math.isnan(tau_new[i]))]
            i_min = min(extr_idx, key=lambda i: tau_new[i]) if extr_idx else None

            if i_min is not None:
                tin = t_in_old[i_min] if i_min < len(t_in_old) else float("nan")
                tout = t_out_old[i_min] if i_min < len(t_out_old) else float("nan")
                print(f"τ_min@i={i_min}, seg_idx={rows[i_min].seg_idx}, tau={tau_new[i_min]:.3f}, "
                    f"t_in={tin:.3f}, t_out={tout:.3f}, note={rows[i_min].note}")
            # --------------------------------------------


            iter_runtime = time.time() - iter_start
            runtime_log.append({
                "iteration": it, 
                "runtime_sec": iter_runtime,
                "totaltime_sec":time.time() - total_start,
                "mode":mode,
                "tau_min":tau_min,
                "tau_max":tau_max,
                "tau_mean":tau_mean,
                "improve":(ge / prev_ge) if (prev_ge is not None and prev_ge > EPS) else 1.0,
                "within":within,
                "window_cap":window_cap,
                "phase":phase,
                # --- risk-based metrics ---
                "violations": int(v_cnt),
                "v_max": float(v_max),
                "p95": float(p95_abs),
                "cvar95": float(cvar95),
                })
            if args.enable_timer:
                print(f"🕒 Iteration {it} finished in {iter_runtime:.3f} s")

            # 4) 更新最佳解（Best Iteration Logic）
            if allow_best_update:
                # choose which "best bucket" to update
                if mode == "FAST":
                    if best_iter_fast < 0:
                        best_iter_fast = it
                        best_within_fast = within
                        best_v_cnt_fast = int(v_cnt)
                        best_v_max_fast = float(v_max)
                        best_ge_fast = ge
                        best_F_fast = F_next[:]
                        best_P_fast = P_next_ms[:]   # ✅ FAST 也記 dwell，之後 patch best.gcode 才一致
                    else:
                        if is_better_candidate(within, v_cnt, v_max, ge,
                                            best_within_fast, best_v_cnt_fast, best_v_max_fast, best_ge_fast):
                            best_iter_fast = it
                            best_within_fast = within
                            best_v_cnt_fast = int(v_cnt)
                            best_v_max_fast = float(v_max)
                            best_ge_fast = ge
                            best_F_fast = F_next[:]
                            best_P_fast = P_next_ms[:]

                elif mode == "TRUE":
                    if best_iter_true < 0:
                        best_iter_true = it
                        best_within_true = within
                        best_v_cnt_true = int(v_cnt)
                        best_v_max_true = float(v_max)
                        best_ge_true = ge
                        best_F_true = F_next[:]
                        best_P_true = P_next_ms[:]
                    else:
                        if is_better_candidate(within, v_cnt, v_max, ge,
                                            best_within_true, best_v_cnt_true, best_v_max_true, best_ge_true):
                            best_iter_true = it
                            best_within_true = within
                            best_v_cnt_true = int(v_cnt)
                            best_v_max_true = float(v_max)
                            best_ge_true = ge
                            best_F_true = F_next[:]
                            best_P_true = P_next_ms[:]

            # 5) 計算 improvement rate (ge/prev_ge) 與 stall
            if prev_ge is not None and prev_ge > EPS:
                ratio = ge / prev_ge
                improve = (prev_ge - ge) / prev_ge
                #if improve < args.stall_improve:
                if ge > prev_ge:
                    low_improve_count += 1
                else:
                    low_improve_count = 0
            else:
                ratio = 1.0
                improve = 0.0
                low_improve_count = 0
            
            # 5) 計算 error 
            if prev_ge is not None and prev_ge > EPS:
                error_delta = ge - prev_ge
                error_ratio = ge / prev_ge
            else:
                error_delta = 0.0
                error_ratio = 1.0

            # 6) 視需要調整 window_cap（循環模式）
            if args.adaptive_window and (it - last_adjust_iter) >= args.cooldown:
                # 追蹤連續在上限或下限的次數
                if "at_upper_count" not in locals():
                    at_upper_count = 0
                    at_lower_count = 0

                if phase == "rise":
                    if window_cap < cap_upper:
                        window_cap = min(cap_upper, math.ceil(window_cap * 1.1))
                        last_adjust_iter = it
                        at_upper_count = 0
                    else:
                        at_upper_count += 1
                        print(f"⚠️  Window_cap reached upper bound ({cap_upper}), count={at_upper_count}")
                        if at_upper_count >= args.stall_runs:
                            phase = "fall"
                            at_upper_count = 0
                            low_improve_count = 0
                            last_adjust_iter = it
                            print(f"🔻 Auto-switch to FALL (stayed at upper bound for {args.stall_runs} runs)")

                elif phase == "fall":
                    if window_cap > cap_lower:
                        window_cap = max(cap_lower, math.floor(window_cap * 0.95)) 
                        last_adjust_iter = it
                        at_lower_count = 0
                    else:
                        at_lower_count += 1
                        print(f"⚠️  Window_cap reached lower bound ({cap_lower}), count={at_lower_count}")
                        if at_lower_count >= args.stall_runs:
                            phase = "rise"
                            at_lower_count = 0
                            low_improve_count = 0
                            last_adjust_iter = it
                            print(f"🔺 Auto-switch to RISE (stayed at lower bound for {args.stall_runs} runs)")

            # 7) 寫本輪 iteration CSV
            if args.save_intermediate:
                iter_csv = os.path.join(args.out_dir, f"{args.out_prefix}{it}.csv")
                write_iteration_csv(
                    iter_csv,
                    rows,
                    F_curr,
                    F_next,
                    tau_old,
                    tau_new,
                    tau_star,
                    tau_lo,
                    tau_hi,
                    tau_grid_lookup,
                    alpha_grid_lookup
                )

            # 8) 更新 summary & log
            summary_hist.append({
                "iteration": it,
                "global_error": ge,
                "tau_min": tau_min,
                "tau_max": tau_max,
                "tau_mean": tau_mean,
                "within%": within,
                "window_cap": window_cap,
                "iteration_runtime": iter_runtime
            })
            summary_path = os.path.join(args.out_dir, args.summary)
            write_summary_csv(summary_path, summary_hist, cli_string)

            if args.enable_timer:
                runtime_path = os.path.join(args.out_dir, "runtime_log.csv")
                write_runtime_log(runtime_path, runtime_log)

            total_elapsed = time.time() - total_start
            imp_rate = (ge / prev_ge) if (prev_ge is not None and prev_ge > EPS) else 1.0

            print(f"[Iter {it:03d}] mode={mode:<5} | Δt={iter_runtime:.3f}s | Total={total_elapsed:.1f}s | "
                  f"τ_min={r2(tau_min):>6}, τ_max={r2(tau_max):>6}, τ_mean={r2(tau_mean):>6} | "
                  f"total_err={r2(total_err):>6}s | max_cycle_err={r2(max_cycle_err):>6}s@cycle={max_cycle_id} | worst_seg=seg_idx={worst_segidx}, d={r2(worst_dist)}s, row_i={i_star} | Δerror={error_delta:+.2f} | ratio={error_ratio:.5f} | "
                  f"within={r2(within):>6}% | v_cnt={v_cnt:>4} | v_max={r2(v_max):>6}s | p95={r2(p95_abs):>6}s | cvar95={r2(cvar95):>6}s | "
                  #f"window_cap={window_cap:>4} | phase={phase}"
                  )
            if args.fifo_mode == "true":
                # TRUE / TRUE-HYBRID: only report TRUE-best (authoritative)
                if best_iter_true >= 0:
                    print(f"Best TRUE iteration so far: {best_iter_true} (within={r2(best_within_true)}%, ge={r2(best_ge_true)})")
                else:
                    print("Best TRUE iteration so far: (none yet)")
            else:
                # FAST-only mode: report FAST-best
                if best_iter_fast >= 0:
                    print(f"Best iteration so far: {best_iter_fast} (v_cnt={best_v_cnt_fast}, v_max={r2(best_v_max_fast)}, within={r2(best_within_fast)}%, ge={r2(best_ge_fast)})")

            # 9) 收斂檢查
            if mode == "FAST":
                if v_cnt == 0:
                    tmp_check_count = check_count + 1
                    print(f"✅ Risk-converged: violations=0 ({tmp_check_count}/3) at iteration {it}")
                    if check_count >= 2:
                        print(f"🎉 Fully converged (violations=0) for {tmp_check_count} consecutive iterations. Stopping at {it}.")
                        F_final = F_next[:]
                        break
                    check_count += 1
                else:
                    check_count = 0

            else:
                # keep your old condition for TRUE mode
                if r2(ge) == 0.00 or within > 99.9999:
                    tmp_check_count = check_count + 1
                    print(f"✅ Converged condition hit ({tmp_check_count}/3) at iteration {it} (global_error=0 or within%=100)")
                    if check_count >= 2:
                        print(f"🎉 Fully converged after {tmp_check_count} consecutive iterations. Stopping at {it}.")
                        F_final = F_next[:]
                        break
                    check_count += 1
                else:
                    check_count = 0

            # 準備下一輪（平滑過渡）
            F_final = F_next[:]
            F_curr = [0.7 * F_curr[i] + 0.3 * F_next[i] for i in range(len(F_curr))]
            P_curr_ms = [0.7 * P_curr_ms[i] + 0.3 * P_next_ms[i] for i in range(len(P_curr_ms))]
            old_prev = prev_ge
            prev_ge = ge

            if old_prev is not None and old_prev > EPS:
                error_delta = ge - old_prev
                error_ratio = ge / old_prev
            else:
                error_delta = 0.0
                error_ratio = 1.0



        # end for

    except KeyboardInterrupt:
        print("\n⛔ Interrupted. Using last available F_final / best_F for output.")

    # ---------- 結尾：總時間 / summary / gcode ----------

    total_runtime = time.time() - total_start
    if args.enable_timer:
        runtime_log.append({"iteration": "TOTAL", "runtime_sec": total_runtime})
        runtime_path = os.path.join(args.out_dir, "runtime_log.csv")
        write_runtime_log(runtime_path, runtime_log)
        print(f"⏳ Total runtime: {total_runtime:.3f} s")

    # 最終 summary（再保險寫一次）
    summary_path = os.path.join(args.out_dir, args.summary)
    write_summary_csv(summary_path, summary_hist, cli_string)

    # G-code 輸出（只在最後一次）
    if args.gcode_in and args.gcode_out:
        base, ext = os.path.splitext(args.gcode_out)
        ext = ext or ".gcode"
        best_path  = f"{base}_best{ext}"
        final_path = args.gcode_out

        # choose best vectors
        if args.fifo_mode == "true":
            best_feed = best_F_true if best_F_true is not None else F_final
            best_P    = best_P_true if best_P_true is not None else P_curr_ms
            best_iter_out = best_iter_true if best_iter_true >= 0 else "N/A"
            best_ge_out = best_ge_true if best_iter_true >= 0 else float("nan")
        else:
            best_feed = best_F_fast if best_F_fast is not None else F_final
            best_P    = best_P_fast if best_P_fast is not None else P_curr_ms
            best_iter_out = best_iter_fast if best_iter_fast >= 0 else "N/A"
            best_ge_out = best_ge_fast if best_iter_fast >= 0 else float("nan")

        # --- BEST ---
        patch_gcode_feedrates(
            gcode_in_path=args.gcode_in,
            gcode_out_path=best_path,
            rows=rows,
            F_final=best_feed,
            P_final_ms=best_P,
            enable_travel_opt=args.enable_travel_opt,
            s_travel=s_travel
        )
        print(f"🏆 Best iteration {best_iter_out} (error={r2(best_ge_out)}) -> {best_path}")

        # --- FINAL ---
        # 你想要 final 用什麼？目前你的邏輯是用最後一輪/收斂的 F_final
        patch_gcode_feedrates(
            gcode_in_path=args.gcode_in,
            gcode_out_path=final_path,
            rows=rows,
            F_final=F_final,
            P_final_ms=P_curr_ms,
            enable_travel_opt=args.enable_travel_opt,
            s_travel=s_travel
        )
        print(f"✅ Final G-code -> {final_path}")

    
        # ---------------------------------------------------------
        # After patching final and best Gcodes, regenerate CSVs (AUTO)
        # ---------------------------------------------------------

        def rebuild_segments(gcode_path, out_csv_path):
            dummy_out = out_csv_path + ".rebuilt.gcode"
            true_fifo_rebuild(
                gcode_in=gcode_path,
                csv_out=out_csv_path,
                gcode_out=dummy_out,
                V_nozzle=args.vnozzle,
                d_filament=15.55634919,
                e_mode="filament",
            )

            if os.path.exists(dummy_out):
                os.remove(dummy_out)

            append_alpha_column_to_segment_csv(
                out_csv_path,
                tau_grid_lookup,
                alpha_grid_lookup,
                tau_col="tau_s",
                alpha_col="alpha"
            )

            print(f"📄 Build corresponding segment CSV (+alpha) → {out_csv_path}")

        # ✅ AUTO: only requires gcode_in/out
        if args.gcode_in and args.gcode_out:
            base, ext = os.path.splitext(args.gcode_out)
            best_path  = f"{base}_best{ext or '.gcode'}"
            final_path = args.gcode_out

            best_csv  = f"{base}_best_segments.csv"
            final_csv = f"{base}_final_segments.csv"

            rebuild_segments(best_path, best_csv)
            rebuild_segments(final_path, final_csv)


if __name__ == "__main__":
    main()
