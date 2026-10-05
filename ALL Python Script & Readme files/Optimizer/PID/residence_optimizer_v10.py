"""Optimize G-code feed rates using residence time and curing degree.

The optimizer evaluates the curing degree of RTV-2 material from its
residence time under isothermal conditions. It iteratively adjusts extrusion,
precondition, and optional travel feed rates to keep the predicted curing
degree within a specified range.

The optimizer first establishes an authoritative TRUE FIFO anchor. It then
validates a trust-limited PID probe candidate. An accepted candidate is
retained as a safe rollback checkpoint. HELD steps extrapolate along the
accepted direction with geometrically decreasing step sizes. The HELD
endpoint is finally TRUE-validated and either accepted or rolled back to
the safe candidate.

CLI example:
--vnozzle 723 --dnozzle 0.84 
--Kp 0.6 --Kd 0.1 --Ki 0.2 --Fmin 350 --Fmax 1200 
--dFmax 0.05 --deltaT_cap_ratio 0.05 --smooth_passes 1 
--max_iter 20 --out_dir optimization_outputs --out_prefix iteration_ 
--summary house_withoutdoor_xy30h5_940s_F800_optimized_summary.csv 
--enable_timer --use_boundary_window 
--gcode_in house_withoutdoor_xy30h5_940s_F800_rebuilt_segmented.gcode --gcode_out house_withoutdoor_xy30h5_940s_segmented_optimized_final.gcode 
--fifo_mode true --save_intermediate 
--window_cap 1200 --horizon_sec 240 --horizon_mode past 
--enable_travel_opt
--window_direction backward
--held_max_steps 4
--held_step_scale 0.5
--held_step_decay 0.5
--held_total_dFmax 0.05
--held_total_dPmax 0.10
--held_failure_shrink 0.5
--held_step_scale_min 0.05

SingleLine Command:
residence_optimizer_v9.3.py --input CFFFP_gasket1_LH0.2_Temp32_Z+1_F0.15_T0.15_segmented.csv --temp_c 32.0 --alpha_ideal 0.925747 --alpha_max 0.94426194  --alpha_min 0.90723206 --vnozzle 574 --dnozzle 0.58 --Fmin 180 --Fmax 3000 --dFmax 0.05 --deltaT_cap_ratio 0.05 --smooth_passes 2 --out_dir optimization_outputs --out_prefix iteration_ --summary CFFFP_gasket1_LH0.2_Temp32_Z+1_F0.15_T0.15_optimized_summary.csv --enable_timer --gcode_in CFFFP_gasket1_LH0.2_Temp32_Z+1_F0.15_T0.15_segmented.gcode --gcode_out CFFFP_gasket1_LH0.2_Temp32_Z+1_F0.15_T0.15_optimized_final.gcode --fifo_mode true --save_intermediate --enable_travel_opt --max_iter 300 --Kp 0.1 --Ki 0.05 --Kd 0.05 --window_cap 1000 --horizon_sec 600 --horizon_mode past
 
Debug test:
python residence_optimizer_v9.4.py --input "CFFFP_testmodel_XL_2040s_Z+1_F0.1_T0.1_segmented.csv" --gcode_in "CFFFP_testmodel_XL_2040s_Z+1_F0.1_T0.1_segmented.gcode" --gcode_out "CFFFP_testmodel_XL_2040s_Z+1_F0.1_T0.1_optimized.gcode" --temp_c 25.0 --alpha_ideal 0.923072 --alpha_max 0.938468  --alpha_min 0.906607 --vnozzle 574 --dnozzle 0.58 --true_interval_dFmax 0.05 --true_interval_dPmax 0.10 --trust_shrink 0.5 --trust_min 0.005 --trust_p_min 0.005 --Fmin 180 --Fmax 3000 --Kp 0.10 --Ki 0.05 --Kd 0.05 --dFmax 0.05 --deltaT_cap_ratio 0.05 --smooth_passes 1 --max_iter 200 --window_cap 1000 --horizon_sec 600 --horizon_mode past --window_direction backward --enable_travel_opt --out_dir "optimization_outputs" --summary "optimization_summary.csv" --enable_timer
python residence_optimizer_v10.py --input "CFFFP_testmodel_XL_2040s_Z+1_F0.1_T0.1_segmented.csv" --gcode_in "CFFFP_testmodel_XL_2040s_Z+1_F0.1_T0.1_segmented.gcode" --gcode_out "CFFFP_testmodel_XL_2040s_Z+1_F0.1_T0.1_optimized.gcode" --temp_c 25.0 --alpha_min 0.906607 --alpha_ideal 0.923072 --alpha_max 0.938468 --vnozzle 574 --dnozzle 0.58 --Fmin 180 --Fmax 3000 --Kp 0.10 --Ki 0.05 --Kd 0.05 --dFmax 0.05 --deltaT_cap_ratio 0.05 --smooth_passes 1 --true_interval_dFmax 0.05 --true_interval_dPmax 0.10 --trust_shrink 0.5 --trust_min 0.005 --trust_p_min 0.005 --held_max_steps 4 --held_step_scale 0.5 --held_step_decay 0.5 --held_total_dFmax 0.05 --held_total_dPmax 0.10 --held_failure_shrink 0.5 --held_step_scale_min 0.05 --window_cap 2500 --horizon_sec 600 --horizon_mode past --window_direction backward --enable_travel_opt --max_iter 300 --out_dir "optimization_outputs" --out_prefix "iteration_" --summary "optimization_summary.csv" --save_intermediate --enable_timer
"""


import csv
import math
import time
import argparse
import os
from typing import List, Dict, Tuple
from rebuild_csv_and_gcode_fifo_v1_2_precondition_autofill import rebuild_global as true_fifo_rebuild
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

# Final RTV-2 kinetic parameters identified from isothermal curing data.
# Activation energies are expressed in J/mol.
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
    """Convert residence time to an integer comparison value.

    The value is first rounded to one decimal place and then rounded to
    the nearest integer.

    Args:
        t: Residence time in seconds.

    Returns:
        The rounded integer value, or ``None`` if the input is ``None``
        or NaN.
    """
    if t is None:
        return None

    try:
        value = float(t)
    except (TypeError, ValueError):
        return None

    if math.isnan(value):
        return None

    return int(round(round(value, 1)))

def celsius_to_kelvin(temp_c: float) -> float:
    """Convert temperature from degrees Celsius to kelvin.

    Args:
        temp_c: Temperature in degrees Celsius.

    Returns:
        Temperature in kelvin.
    """
    return float(temp_c) + 273.15


def arrhenius_rate(
    A: float,
    E: float,
    temp_k: float,
) -> float:
    """Calculate an Arrhenius reaction-rate constant.

    Args:
        A: Pre-exponential factor.
        E: Activation energy in joules per mole.
        temp_k: Absolute temperature in kelvin.

    Returns:
        The calculated reaction-rate constant.
    """
    return float(A) * math.exp(-float(E) / (R_GAS * float(temp_k)))


def alpha_rhs_isothermal(
    t: float,
    y: np.ndarray,
    temp_c: float,
    params: dict,
):
    """Evaluate the isothermal curing-model differential equation.

    Args:
        t: Current integration time in seconds. This argument is required
            by the ODE solver but is not used explicitly.
        y: Current solver state containing the curing degree.
        temp_c: Isothermal temperature in degrees Celsius.
        params: Kinetic parameters containing ``A1``, ``E1``, ``A2``,
            ``E2``, ``m``, and ``n``.

    Returns:
        A one-element list containing the curing-degree derivative.
    """
    alpha = float(y[0])
    alpha = min(max(alpha, 1e-12), 1.0)

    T_k = celsius_to_kelvin(temp_c)
    k1 = arrhenius_rate(params["A1"], params["E1"], T_k)
    k2 = arrhenius_rate(params["A2"], params["E2"], T_k)
    m = float(params["m"])
    n = float(params["n"])

    dadt = (k1 + k2 * (alpha ** m)) * ((1.0 - alpha) ** n)
    return [dadt]


def solve_alpha_isothermal(
    temp_c: float,
    residence_time_s: float,
    params: dict,
    alpha_in: float = ALPHA_IN_DEFAULT,
    n_eval: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the curing-degree evolution under isothermal conditions.

    Args:
        temp_c: Constant processing temperature in degrees Celsius.
        residence_time_s: Total residence time in seconds.
        params: Kinetic-model parameters.
        alpha_in: Initial curing degree. Defaults to
            ``ALPHA_IN_DEFAULT``.
        n_eval: Number of evaluation points. Defaults to 400.

    Returns:
        A tuple containing the time grid and corresponding curing-degree
        values.

    Raises:
        ValueError: If ``residence_time_s`` is negative.
        RuntimeError: If the numerical ODE solver fails.
    """
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


def build_alpha_tau_lookup(
    temp_c: float,
    params: dict,
    tau_upper_s: float,
    alpha_in: float = ALPHA_IN_DEFAULT,
    n_eval: int = 3000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a lookup table between residence time and curing degree.

    Args:
        temp_c: Constant processing temperature in degrees Celsius.
        params: Kinetic-model parameters.
        tau_upper_s: Maximum residence time represented by the lookup.
        alpha_in: Initial curing degree. Defaults to
            ``ALPHA_IN_DEFAULT``.
        n_eval: Number of lookup points. Defaults to 3000.

    Returns:
        A tuple containing the residence-time grid and monotonic
        curing-degree grid.

    Raises:
        RuntimeError: If the underlying ODE solver fails.
    """
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


def alpha_from_tau_lookup(
    tau_s: float,
    tau_grid: np.ndarray,
    alpha_grid: np.ndarray,
) -> float:
    """Interpolate curing degree from residence time.

    Args:
        tau_s: Residence time in seconds.
        tau_grid: Residence-time lookup grid.
        alpha_grid: Curing-degree lookup grid.

    Returns:
        The interpolated curing degree. Returns NaN if ``tau_s`` is
        ``None`` or NaN. Values outside the lookup range are clamped to
        the nearest endpoint.
    """
    if tau_s is None or math.isnan(tau_s):
        return float("nan")
    tau_s = float(tau_s)
    if tau_s <= tau_grid[0]:
        return float(alpha_grid[0])
    if tau_s >= tau_grid[-1]:
        return float(alpha_grid[-1])
    return float(np.interp(tau_s, tau_grid, alpha_grid))


def alpha_list_from_tau_list(
    tau_list: List[float],
    tau_grid: np.ndarray,
    alpha_grid: np.ndarray,
) -> List[float]:
    """Convert multiple residence times to curing degrees.

    Args:
        tau_list: Residence times in seconds.
        tau_grid: Residence-time lookup grid.
        alpha_grid: Curing-degree lookup grid.

    Returns:
        A list of interpolated curing degrees.
    """
    out = []
    for t in tau_list:
        out.append(alpha_from_tau_lookup(t, tau_grid, alpha_grid))
    return out


def tau_from_alpha_lookup(
    alpha_target: float,
    tau_grid: np.ndarray,
    alpha_grid: np.ndarray,
) -> float:
    """Interpolate the residence time required for a curing degree.

    Args:
        alpha_target: Target curing degree.
        tau_grid: Residence-time lookup grid.
        alpha_grid: Monotonic curing-degree lookup grid.

    Returns:
        The interpolated residence time in seconds.

    Raises:
        ValueError: If ``alpha_target`` exceeds the maximum curing degree
            represented by the lookup.
    """
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


def append_alpha_column_to_segment_csv(
    csv_path: str,
    tau_grid: np.ndarray,
    alpha_grid: np.ndarray,
    tau_col: str = "tau_s",
    alpha_col: str = "alpha",
) -> None:
    """Add calculated curing degrees to a segment CSV file.

    The file is left unchanged if it does not exist or does not contain
    the requested residence-time column.

    Args:
        csv_path: Path to the segment CSV file.
        tau_grid: Residence-time lookup grid.
        alpha_grid: Curing-degree lookup grid.
        tau_col: Name of the residence-time column. Defaults to
            ``"tau_s"``.
        alpha_col: Name of the output curing-degree column. Defaults to
            ``"alpha"``.

    Returns:
        None.

    Raises:
        OSError: If the existing CSV cannot be read or overwritten.
    """
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
        except (TypeError, ValueError):
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
    
    def __init__(self, d: Dict[str, str]):
        """Initialize a segment row from CSV data.

        Args:
            d: CSV record containing segment geometry, timing, feed rate,
                volume, command, and note fields.

        Raises:
            KeyError: If a required CSV field is missing.
            ValueError: If a numeric field cannot be converted.
        """
        
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

# ---------- E-only / precondition support ----------

EONLY_D_FILAMENT = 15.55634919
EONLY_EMODE = "filament"

def volume_to_e_axis_mm(
    V_mm3: float,
    d_filament: float = EONLY_D_FILAMENT,
    e_mode: str = EONLY_EMODE,
) -> float:
    """Convert extrusion volume to an E-axis displacement.

    In filament mode, the E-axis value represents filament length. In
    cubic-millimetre mode, the E-axis value directly represents volume.

    Args:
        V_mm3: Extrusion volume in cubic millimetres.
        d_filament: Filament diameter in millimetres.
        e_mode: E-axis interpretation. Supported values are
            ``"filament"`` and ``"mm3"``.

    Returns:
        The corresponding E-axis displacement. Returns zero for
        negligible volume or invalid filament area.
    """
    V = float(V_mm3 or 0.0)

    if V <= EPS:
        return 0.0
    
    if e_mode == "mm3":
        return V

    if e_mode == "filament":
        A_fil = area_circle(float(d_filament))
        if A_fil <= EPS:
            return 0.0
        return V / A_fil

    raise ValueError(
        f"Unknown e_mode: {e_mode}. Expected 'filament' or 'mm3'."
    )


def row_control_len_mm(r: Row) -> float:
    """Determine the effective axis length used for time conversion.

    XY motion uses the geometric movement length. An E-only G1
    precondition or extrusion uses the equivalent filament displacement
    calculated from its volume.

    Args:
        r: Segment row to evaluate.

    Returns:
        The effective movement length in millimetres. Returns zero if
        the row has no controllable movement.
    """
    d = float(getattr(r, "d_mm", 0.0) or 0.0)

    if d > EPS:
        return d

    note_l = (getattr(r, "note", "") or "").lower()
    cmd = (getattr(r, "cmd", "") or "").upper()
    V = float(getattr(r, "V_mm3", 0.0) or 0.0)

    is_e_only_control_piece = (
        cmd == "G1"
        and V > EPS
        and (r.is_extrusion or r.is_optimizable or "precondition" in note_l)
    )

    if is_e_only_control_piece:
        return volume_to_e_axis_mm(V)

    return 0.0

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
    """Determine whether a segment row represents travel movement.

    Args:
        r: Segment row to evaluate.

    Returns:
        ``True`` for a non-extrusion, non-optimizable G0 or G1 movement
        with positive travel distance; otherwise, ``False``.
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

def is_actual_window_row(r: Row) -> bool:
    """Determine whether a row is counted in the actual-window report.

    Extrusion, optimizable, and travel rows are included. Raw, comment,
    and non-motion rows are excluded.

    Args:
        r: Segment row to evaluate.

    Returns:
        ``True`` if the row belongs to the reported actual window;
        otherwise, ``False``.
    """
    note_l = (getattr(r, "note", "") or "").lower()

    if "raw" in note_l:
        return False
    if "comment" in note_l:
        return False

    if r.is_extrusion or r.is_optimizable:
        return True

    if is_travel_row(r):
        return True

    return False

def is_pid_adjustable_row(
    r: Row,
    enable_travel_opt: bool,
) -> bool:
    """Determine whether a row can receive PID feed-rate adjustment.

    Args:
        r: Segment row to evaluate.
        enable_travel_opt: Whether travel rows may be adjusted.

    Returns:
        ``True`` for extrusion and optimizable rows, and optionally for
        travel rows. Returns ``False`` for raw, comment, and non-motion
        rows.
    """
    note_l = (getattr(r, "note", "") or "").lower()

    if "raw" in note_l:
        return False
    if "comment" in note_l:
        return False

    if r.is_extrusion or r.is_optimizable:
        return True

    if enable_travel_opt and is_travel_row(r):
        return True

    return False

def row_volume_for_boundary_window(
    r: Row,
    d_nozzle: float,
) -> float:
    """Calculate the row volume used for boundary-window construction.

    The CSV volume is preferred. If it is unavailable, volume is
    estimated from nozzle area and movement distance.

    Args:
        r: Extrusion or optimizable segment row.
        d_nozzle: Nozzle diameter in millimetres.

    Returns:
        The row volume in cubic millimetres.
    """
    if getattr(r, "V_mm3", 0.0) > EPS:
        return float(r.V_mm3)

    if getattr(r, "d_mm", 0.0) > EPS:
        return area_circle(d_nozzle) * float(r.d_mm)

    return 0.0


def build_vnozzle_boundary_window(
    rows: List[Row],
    target_idx: List[int],
    target_pos: int,
    V_nozzle: float,
    d_nozzle: float,
    window_direction: str,
) -> List[int]:
    """Build a volume-limited optimization window.

    The window contains only extrusion and optimizable rows. Travel, raw,
    and comment rows do not contribute to its accumulated volume.

    Args:
        rows: Complete segment-row list.
        target_idx: Row indices eligible for optimization.
        target_pos: Position of the target within ``target_idx``.
        V_nozzle: Maximum accumulated window volume.
        d_nozzle: Nozzle diameter used for volume fallback.
        window_direction: Scanning direction, normally ``"backward"`` or
            ``"forward"``.

    Returns:
        Sorted row indices belonging to the optimization window.
    """
    if not target_idx:
        return []

    if target_pos < 0 or target_pos >= len(target_idx):
        return []

    V_cap = float(V_nozzle)

    if V_cap <= EPS:
        return [target_idx[target_pos]]

    win = []
    acc_v = 0.0

    if window_direction == "backward":
        scan_range = range(target_pos, -1, -1)
    else:
        scan_range = range(target_pos, len(target_idx))

    for p in scan_range:
        j = target_idx[p]
        v = row_volume_for_boundary_window(rows[j], d_nozzle)

        if v <= EPS:
            continue

        # If target alone exceeds V_nozzle, keep target only.
        # Otherwise the target would never be optimized.
        if not win and v > V_cap + EPS:
            win.append(j)
            break

        if acc_v + v > V_cap + EPS:
            break

        win.append(j)
        acc_v += v

    win.sort()
    return win


def summarize_actual_window(
    rows: List[Row],
    win: List[int],
) -> dict:
    """Summarize the actual reported span of an optimization window.

    Travel rows between the first and last optimization targets are
    included, while raw and comment rows are excluded.

    Args:
        rows: Complete segment-row list.
        win: Optimization-window row indices.

    Returns:
        A dictionary containing ``start_seg``, ``end_seg``, and
        ``count``. Empty windows use ``None`` boundaries and zero count.
    """
    if not win:
        return {
            "start_seg": None,
            "end_seg": None,
            "count": 0,
        }

    start_i = min(win)
    end_i = max(win)

    actual_rows = [
        i for i in range(start_i, end_i + 1)
        if is_actual_window_row(rows[i])
    ]

    if not actual_rows:
        return {
            "start_seg": rows[start_i].seg_idx,
            "end_seg": rows[end_i].seg_idx,
            "count": 0,
        }

    return {
        "start_seg": rows[actual_rows[0]].seg_idx,
        "end_seg": rows[actual_rows[-1]].seg_idx,
        "count": len(actual_rows),
    }

def expand_target_window_to_pid_window(
    rows: List[Row],
    target_win: List[int],
    enable_travel_opt: bool,
) -> List[int]:
    """Expand a target window into the actual PID adjustment window.

    Args:
        rows: Complete segment-row list.
        target_win: Extrusion and optimizable target-row indices.
        enable_travel_opt: Whether intervening travel rows may be added.

    Returns:
        Row indices that may receive PID adjustment.
    """
    if not target_win:
        return []

    start_i = min(target_win)
    end_i = max(target_win)

    pid_win = []
    for i in range(start_i, end_i + 1):
        if is_pid_adjustable_row(rows[i], enable_travel_opt):
            pid_win.append(i)

    return pid_win

def update_window_debug_record(
    rows: List[Row],
    target_i: int,
    win: List[int],
    window_debug_max,
    window_debug_min,
):
    """Update maximum and minimum window-debug records.

    Args:
        rows: Complete segment-row list.
        target_i: Target row index.
        win: Current optimization-window indices.
        window_debug_max: Existing maximum-window record, or ``None``.
        window_debug_min: Existing minimum-window record, or ``None``.

    Returns:
        A tuple containing the updated maximum- and minimum-window
        records.
    """
    if not win:
        return window_debug_max, window_debug_min

    actual_summary = summarize_actual_window(rows, win)

    if actual_summary["count"] <= 0:
        return window_debug_max, window_debug_min

    rec = {
        "target_seg": rows[target_i].seg_idx,
        "start_seg": actual_summary["start_seg"],
        "end_seg": actual_summary["end_seg"],
        "count": actual_summary["count"],
    }

    if window_debug_max is None or rec["count"] > window_debug_max["count"]:
        window_debug_max = rec

    if window_debug_min is None or rec["count"] < window_debug_min["count"]:
        window_debug_min = rec

    return window_debug_max, window_debug_min

def infer_F0_row(r: Row, Fmin_fallback: float) -> float:
    """Infer the original feed rate of a segment row.

    Args:
        r: Segment row to evaluate.
        Fmin_fallback: Feed rate used when it cannot be inferred.

    Returns:
        The CSV feed rate, a feed rate derived from distance and duration,
        or the supplied fallback value.
    """
    if getattr(r, "F", 0.0) > EPS:
        return float(r.F)
    if getattr(r, "d_mm", 0.0) > EPS and getattr(r, "t_s", 0.0) > EPS:
        return 60.0 * float(r.d_mm) / float(r.t_s)
    return float(Fmin_fallback)

def update_travel_scale(
    s_travel: float,
    tau_mean: float,
    tau_lo: float,
    tau_star: float,
    travel_gain: float,
    travel_step_max: float,
    travel_s_min: float,
) -> float:
    """Update the global travel-speed scaling factor.

    Travel is slowed when the mean residence time is below the lower
    limit. The scaling factor is monotonic and never increases.

    Args:
        s_travel: Current travel-speed scaling factor.
        tau_mean: Current mean residence time.
        tau_lo: Lower acceptable residence-time boundary.
        tau_star: Target residence time.
        travel_gain: Gain applied to the normalized residence-time error.
        travel_step_max: Maximum scaling reduction per update.
        travel_s_min: Minimum allowed scaling factor.

    Returns:
        The updated travel-speed scaling factor.
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

def apply_travel_scale_to_F(
    rows: List[Row],
    F_curr: List[float],
    F_next: List[float],
    F0_travel: List[float],
    s_travel: float,
    travel_dFmax_ratio: float,
    Fmin: float,
    Fmax: float,
    enable: bool,
) -> List[float]:
    """Apply global scaling to travel feed rates.

    The original relative travel-speed structure is preserved. Travel
    feed rates can only decrease, and their per-iteration change is
    limited.

    Args:
        rows: Complete segment-row list.
        F_curr: Current feed rates aligned with ``rows``.
        F_next: Proposed next feed rates.
        F0_travel: Original travel feed rates.
        s_travel: Global travel scaling factor.
        travel_dFmax_ratio: Maximum relative travel-rate reduction.
        Fmin: Minimum allowed feed rate.
        Fmax: Maximum allowed feed rate.
        enable: Whether travel scaling is active.

    Returns:
        The updated feed-rate list.
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

    """Run the external physical FIFO rebuild for one iteration.

    The function patches a temporary G-code file using segment indices,
    invokes the external FIFO rebuilding engine, reads the generated CSV,
    and aligns its timing results with the optimizer rows.

    Args:
        gcode_template_path: Path to the segmented G-code template.
        rows: Segment rows aligned with the G-code indices.
        F_vec: Feed-rate values to apply.
        P_vec_ms: Dwell-time values in milliseconds.
        V_nozzle: FIFO control volume in cubic millimetres.
        d_nozzle: Nozzle diameter in millimetres.
        iteration: Current optimization iteration.
        out_root: Root directory for temporary and saved results.
        enable_travel_opt: Whether travel feed rates are patched.
        s_travel: Global travel scaling factor.
        save_intermediate: Whether to preserve intermediate files.

    Returns:
        A tuple containing:

        - Whether the external rebuild succeeded.
        - Row-aligned input times.
        - Row-aligned output times.
        - Row-aligned residence times.
        - Generated CSV path.
        - Patched G-code path.

    Notes:
        Failure is reported through a ``False`` status and NaN timing
        lists instead of propagating most internal exceptions.
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
            except Exception as e:
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
    """Calculate the percentage of extrusion rows within the limits.

    Args:
        tau_list: Residence times aligned with ``rows``.
        rows: Segment rows.
        tau_lo: Lower acceptable residence-time boundary.
        tau_hi: Upper acceptable residence-time boundary.

    Returns:
        The percentage of valid extrusion rows inside the limits.
        Returns NaN if no valid extrusion values are available.
    """
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
    """Calculate residence-time statistics for extrusion rows.

    Args:
        tau_list: Residence times aligned with ``rows``.
        rows: Segment rows.

    Returns:
        A tuple containing minimum, maximum, and mean residence time.
        Returns three zeros if no valid values are available.
    """
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

def global_error_abs_sum(
    tau_list,
    rows,
    tau_star: float,
) -> float:
    """Calculate total absolute residence-time error.

    Args:
        tau_list: Residence times aligned with ``rows``.
        rows: Segment rows.
        tau_star: Target residence time.

    Returns:
        The sum of absolute errors across valid extrusion rows.
    """
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
        * Distributes the time correction using exponential weights.
        * Applies a 90/10 blend between the previous and target feed rates.

    Args:
        rows (List[Row]): Segment metadata (extrusion flags, geometry, volumes).
        F_current (list[float]): Feedrate values (mm/min) aligned to rows.
        tau_current (list[float]): Current τ values for each row.
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
            L = row_control_len_mm(rows[j])
            Fold = F_new[j] if F_new[j] > EPS else (rows[j].F if rows[j].F > EPS else Fmin)
            if L > EPS and Fold > EPS:
                tb = 60.0 * L / Fold
                t_base.append(tb)
                t_min.append(60.0 * L / Fmax)  # 最快
                t_max.append(60.0 * L / Fmin)  # 最慢
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
            L = row_control_len_mm(rows[j])
            if L <= EPS:
                continue

            Fold = F_new[j] if F_new[j] > EPS else (rows[j].F if rows[j].F > EPS else Fmin)
            tb = t_base[k]
            dt = weights[k] * DeltaT
            new_ts = tb + dt

            if new_ts <= 0.0:
                continue

            new_ts = max(t_min[k], min(new_ts, t_max[k]))
            F_tar = 60.0 * L / new_ts
            F_tar = clamp_rate(Fold, F_tar)
            F_new[j] = 0.9 * Fold + 0.1 * F_tar

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
                                smooth_passes: int,
                                enable_travel_opt: bool,) -> Tuple[List[float], Dict[str, object]]:
    """
    Hybrid τ controller:
      - Local PD (per segment)
      - Cycle-level I (per FIFO cycle)
      - Forward window ΔT redistribution
      - Global smoothing over F_new
    """

    def get_exp_weights(
        L: int,
        alpha: float = 0.03,
    ) -> np.ndarray:
        """Return stable normalized exponential window weights.

        Args:
            L: Number of rows in the window.
            alpha: Exponential weighting coefficient.

        Returns:
            Normalized weights with length ``L``.
        """
        if L <= 0:
            return np.empty(0, dtype=float)

        # Shift the largest exponent to zero to prevent overflow.
        positions = np.arange(L, dtype=float)
        exponents = alpha * (positions - float(L - 1))
        weights = np.exp(exponents)

        total = float(np.sum(weights))

        if not math.isfinite(total) or total <= EPS:
            return np.full(L, 1.0 / float(L), dtype=float)

        return weights / total


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
        return F_new, {"max": None, "min": None}

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
    is_target = [(r.is_extrusion or r.is_optimizable) for r in rows]

    # For count-based and V_nozzle-boundary windows:
    # keep all extrusion/optimizable rows.
    target_idx = [i for i in range(N) if is_target[i]]

    # For horizon mode only:
    # keep only rows with valid t_out.
    target_idx_time = []
    target_tout = []

    for i in target_idx:
        t = t_out_current[i] if i < len(t_out_current) else float("nan")
        if t is None or math.isnan(t):
            continue

        target_idx_time.append(i)
        target_tout.append(float(t))
        
    def window_by_time(t0: float, t1: float) -> List[int]:
        # assume target_tout is non-decreasing
        L = bisect_left(target_tout, t0)
        R = bisect_right(target_tout, t1)
        win = target_idx_time[L:R]

        if not win:
            return []

        # Horizon mode uses count cap.
        # This whole function is ignored when use_boundary_window=True.
        count_cap = max(1, int(window_cap))

        if count_cap > 0 and len(win) > count_cap:
            #win = win[:count_cap]
            win = win[-count_cap:]

        return win

    window_debug_max = None
    window_debug_min = None

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
            (not use_boundary_window) and
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
                L = row_control_len_mm(rows[jj])
                Fold = F_new[jj] if F_new[jj] > EPS else (rows[jj].F if rows[jj].F > EPS else Fmin)

                if L > EPS and Fold > EPS:
                    tb = 60.0 * L / Fold
                    t_base_l.append(tb)
                    t_min_l.append(60.0 * L / Fmax)
                    t_max_l.append(60.0 * L / Fmin)
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
                L = row_control_len_mm(rows[jj])
                if L <= EPS:
                    continue

                Fold = F_new[jj] if F_new[jj] > EPS else (rows[jj].F if rows[jj].F > EPS else Fmin)
                tb = t_base_l[k]
                dt = weights[k] * DeltaT_eff
                new_ts = tb + dt

                if new_ts <= 0.0:
                    continue

                new_ts = max(t_min_l[k], min(new_ts, t_max_l[k]))
                F_tar = 60.0 * L / new_ts
                F_tar = clamp_rate(Fold, F_tar)
                F_new[jj] = 0.6 * Fold + 0.4 * F_tar

        # build window
        # Priority:
        #   1) use_boundary_window=True  -> V_nozzle-based boundary window
        #   2) use_horizon=True          -> time-horizon window
        #   3) otherwise                 -> target-count window

        horizon_both_active = False

        pos = bisect_left(target_idx, target_i)

        if pos >= len(target_idx) or target_idx[pos] != target_i:
            continue

        if use_boundary_window:
            # New boundary behavior:
            # Fully controlled by V_nozzle and target segment.
            # Ignore window_cap, horizon_sec, horizon_mode, horizon_w_future.
            win = build_vnozzle_boundary_window(
                rows=rows,
                target_idx=target_idx,
                target_pos=pos,
                V_nozzle=V_nozzle,
                d_nozzle=d_nozzle,
                window_direction=window_direction,
            )

        elif use_horizon:
            t_ref = float(t_out_current[target_i])
            h = float(horizon_sec)

            if horizon_mode == "future":
                win = window_by_time(t_ref, t_ref + h)

            elif horizon_mode == "past":
                win = window_by_time(t_ref - h, t_ref)

            else:  # both
                win_past = window_by_time(t_ref - h, t_ref)
                win_future = window_by_time(t_ref, t_ref + h)
                win = (win_past, win_future)
                horizon_both_active = True

        else:
            # Count-based target window:
            # window_cap means number of extrusion/optimizable rows.
            count_cap = max(1, int(window_cap))

            if window_direction == "backward":
                L = max(0, pos - count_cap + 1)
                R = pos + 1
            else:
                L = pos
                R = min(len(target_idx), pos + count_cap)

            win = target_idx[L:R]


        # update terminal debug statistics
        # Convert target window into actual PID window.
        # target_win controls boundary/window size.
        # pid_win is the actual set of rows receiving DeltaT.
        if horizon_both_active:
            target_win_past = win_past
            target_win_future = win_future

            pid_win_past = expand_target_window_to_pid_window(
                rows=rows,
                target_win=target_win_past,
                enable_travel_opt=enable_travel_opt,
            )

            pid_win_future = expand_target_window_to_pid_window(
                rows=rows,
                target_win=target_win_future,
                enable_travel_opt=enable_travel_opt,
            )

            debug_win = sorted(set(pid_win_past + pid_win_future))

        else:
            target_win = win

            pid_win = expand_target_window_to_pid_window(
                rows=rows,
                target_win=target_win,
                enable_travel_opt=enable_travel_opt,
            )

            debug_win = pid_win

        window_debug_max, window_debug_min = update_window_debug_record(
            rows=rows,
            target_i=target_i,
            win=debug_win,
            window_debug_max=window_debug_max,
            window_debug_min=window_debug_min,
        )

        # ----------------------------
        # compute DeltaT (same as before)
        # ----------------------------
        if horizon_both_active:
            win_past, win_future = win

            # Use PID windows, not target-only windows.
            win_past_pid = pid_win_past
            win_future_pid = pid_win_future

            if (not win_past_pid) and (not win_future_pid):
                continue

            win_merged = win_past_pid + win_future_pid
            if not win_merged:
                continue

            # build merged bounds
            t_base, t_min, t_max = [], [], []
            for j in win_merged:
                L = row_control_len_mm(rows[j])
                Fold = F_new[j] if F_new[j] > EPS else (rows[j].F if rows[j].F > EPS else Fmin)
                if L > EPS and Fold > EPS:
                    tb = 60.0 * L / Fold
                    t_base.append(tb)
                    t_min.append(60.0 * L / Fmax)
                    t_max.append(60.0 * L / Fmin)
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

            apply_window_adjust(win_past_pid,   wP * DeltaT)
            apply_window_adjust(win_future_pid, wF * DeltaT)
            continue
        
        else:
            # single window case:
            # win is target-only window.
            # pid_win is actual adjustable window.
            if not pid_win:
                continue

            # time bounds
            t_base, t_min, t_max = [], [], []
            for j in pid_win:
                L = row_control_len_mm(rows[j])
                Fold = F_new[j] if F_new[j] > EPS else (rows[j].F if rows[j].F > EPS else Fmin)
                if L > EPS and Fold > EPS:
                    tb = 60.0 * L / Fold
                    t_base.append(tb)
                    t_min.append(60.0 * L / Fmax)
                    t_max.append(60.0 * L / Fmin)
                else:
                    tb = 0.0
                    t_base.append(tb)
                    t_min.append(tb)
                    t_max.append(tb)

            Cplus = sum(t_max[k] - t_base[k] for k in range(len(pid_win)))
            Cminus = sum(t_base[k] - t_min[k] for k in range(len(pid_win)))

            if u_des >= 0.0:
                DeltaT = min(u_des, Cplus)
            else:
                DeltaT = -min(-u_des, Cminus)

            if abs(DeltaT) <= 1e-12:
                continue

            apply_window_adjust(pid_win, DeltaT)
            continue

    # 7) Global smoothing over F_new（可選）
    def smooth_feedrates(F_vals: List[float], passes: int) -> List[float]:
        if passes <= 0:
            return F_vals
        res = F_vals[:]
        for _ in range(passes):
            newF = res[:]
            for i in range(N):
                if not is_pid_adjustable_row(rows[i], enable_travel_opt):
                    continue
                left = res[i - 1] if i - 1 >= 0 else res[i]
                right = res[i + 1] if i + 1 < N else res[i]
                newF[i] = 0.25 * left + 0.5 * res[i] + 0.25 * right
                newF[i] = max(Fmin, min(newF[i], Fmax))
            res = newF
        return res

    F_new = smooth_feedrates(F_new, smooth_passes)

    window_debug = {
        "max": window_debug_max,
        "min": window_debug_min,
    }

    return F_new, window_debug

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
    """Extract a segment index from a G-code comment.

    Args:
        comment: Comment that may contain a ``seg_idx`` token.

    Returns:
        The segment index, or ``None`` if no valid token is found.
    """
    if not comment:
        return None
    m = SEGIDX_RE.search(comment)
    return int(m.group("id")) if m else None

# 2) 移除舊 seg_idx：這個 regex 不要捕捉群組，sub 時就不可能 "no such group"
SEGIDX_STRIP_RE = re.compile(r"(?:\s*\|\s*)?\bseg_idx\s*=\s*\d+\b", re.IGNORECASE)

def _strip_seg_idx(comment: str) -> str:
    """Remove a segment index token from a G-code comment.

    Args:
        comment: Comment that may contain a segment index.

    Returns:
        The cleaned comment.
    """
    if not comment:
        return ""
    c = SEGIDX_STRIP_RE.sub("", comment)
    # 清掉多餘的分隔符殘留
    c = re.sub(r"\s*\|\s*$", "", c).strip()
    return c

def _gcode_cmd_token(s: str) -> str:
    """Extract and normalize the command token from G-code.

    Optional line numbers are ignored, and commands such as ``G01`` and
    ``G04`` are normalized to ``G1`` and ``G4``.

    Args:
        s: Executable G-code content.

    Returns:
        The normalized command token, or an empty string if none exists.
    """
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

            # Travel F patch expected:
            # TEMPLATE must be G0 or G1 without E, and ROW must be travel.
            if enable_travel_opt and ((cmd == "G0") or (cmd == "G1" and not has_E)) and is_travel_row(rr):
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

            # ---------- (3) Travel F patch: G0 or G1 without E ----------
            if enable_travel_opt and seg_idx is not None and seg_idx in segidx_to_rowi_all:
                i = segidx_to_rowi_all[seg_idx]
                rr = rows[i]

                is_g0 = (cmd == "G0")
                is_g1_travel = (cmd == "G1" and not has_E)

                if (is_g0 or is_g1_travel) and is_travel_row(rr):
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
    """Write detailed results for one optimization iteration.

    Args:
        path: Output CSV path.
        rows: Segment rows.
        F_old: Feed rates before the iteration.
        F_new: Feed rates after the iteration.
        tau_old: Residence times before the iteration.
        tau_new: Residence times after the iteration.
        tau_star: Target residence time.
        tau_lo: Lower acceptable residence-time boundary.
        tau_hi: Upper acceptable residence-time boundary.
        tau_grid: Residence-time lookup grid.
        alpha_grid: Curing-degree lookup grid.

    Returns:
        None.

    Raises:
        OSError: If the output file cannot be written.
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

def write_summary_csv(
    path: str,
    history: List[Dict[str, object]],
    cli_used: str = "",
) -> None:
    """Write TRUE-validated optimization history to a CSV file.

    Each history record represents either the initial CSV baseline or a
    retained state backed by TRUE-confirmed data. HELD_EXTRAPOLATE
    iterations are not included because they do not rebuild or validate
    the FIFO model.

    TRUE_PROBE_ACCEPTED records the validated safe candidate.
    TRUE_PROBE_UNCHANGED records an unchanged parameter state that completed
    a successful TRUE FIFO rebuild.
    TRUE_HELD_ROLLBACK records the retained safe candidate.
    TRUE_HELD_FIFO_FAILED never reports the unvalidated endpoint state.

    The residence-time and risk metrics always describe the state retained
    by the optimizer. Therefore, when a proposed candidate is rejected,
    the metrics in that row describe the previous accepted TRUE state.

    The ``accepted`` column describes the checkpoint decision:

    - 1: The proposed candidate was accepted.
    - 0: The proposed candidate was rejected and rolled back.
    - blank: No candidate decision was made, such as the CSV baseline,
    initial TRUE anchor, or a FIFO-engine failure.

    A trust ratio of 0.05 means that the candidate feed rates were limited
    to a cumulative change of plus or minus 5 percent relative to the
    current TRUE anchor.

    Args:
        path: Output CSV path.
        history: TRUE-validated optimization history records.
        cli_used: Command-line invocation used for the optimization.

    Returns:
        None.

    Raises:
        OSError: If the output file cannot be written.
    """

    def number_or_blank(value: object, decimals: int = 2) -> object:
        """Convert a finite numeric value for CSV output.

        Args:
            value: Value to convert.
            decimals: Number of decimal places to preserve.

        Returns:
            Rounded numeric value, or an empty string for missing and
            non-finite values.
        """
        if value is None or value == "":
            return ""

        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""

        if not math.isfinite(number):
            return ""

        return round(number, decimals)

    def integer_or_blank(value: object) -> object:
        """Convert a value to an integer for CSV output.

        Args:
            value: Value to convert.

        Returns:
            Integer value, or an empty string if conversion fails.
        """
        if value is None or value == "":
            return ""

        try:
            return int(value)
        except (TypeError, ValueError):
            return ""

    def accepted_or_blank(value: object) -> object:
        """Convert a checkpoint decision to 1, 0, or blank.

        Args:
            value: Boolean checkpoint decision or None.

        Returns:
            1 for accepted, 0 for rejected, or an empty string when no
            decision was made.
        """
        if value is None or value == "":
            return ""

        return 1 if bool(value) else 0

    header = [
        "iteration",
        "mode",
        "accepted",
        "global_error",
        "tau_min",
        "tau_max",
        "tau_mean",
        "within%",
        "violations",
        "v_max",
        "p95",
        "cvar95",
        "window_cap",
        "trust_ratio_used",
        "trust_ratio_next",
        "trust_p_ratio_used",
        "trust_p_ratio_next",
        "iteration_runtime",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for record in history:
            writer.writerow([
                integer_or_blank(record.get("iteration")),
                record.get("mode", ""),
                accepted_or_blank(record.get("accepted")),
                number_or_blank(record.get("global_error")),
                number_or_blank(record.get("tau_min")),
                number_or_blank(record.get("tau_max")),
                number_or_blank(record.get("tau_mean")),
                number_or_blank(record.get("within%")),
                integer_or_blank(record.get("violations")),
                number_or_blank(record.get("v_max")),
                number_or_blank(record.get("p95")),
                number_or_blank(record.get("cvar95")),
                integer_or_blank(record.get("window_cap")),
                number_or_blank(
                    record.get("trust_ratio_used"),
                    decimals=6,
                ),
                number_or_blank(
                    record.get("trust_ratio_next"),
                    decimals=6,
                ),
                number_or_blank(
                    record.get("trust_p_ratio_used"),
                    decimals=6,
                ),
                number_or_blank(
                    record.get("trust_p_ratio_next"),
                    decimals=6,
                ),
                number_or_blank(
                    record.get("iteration_runtime"),
                    decimals=3,
                ),
            ])

        if cli_used:
            writer.writerow([])
            writer.writerow(["CLI", cli_used])

def write_runtime_log(
    path: str,
    runtime_log: List[Dict[str, object]],
    cli_used: str = "",
) -> None:
    """Write optimizer runtime and checkpoint decisions to CSV.

    Args:
        path: Output runtime CSV path.
        runtime_log: Runtime and checkpoint records.
        cli_used: Command-line invocation used for the run.

    Returns:
        None.

    Raises:
        OSError: If the output file cannot be written.
    """

    def number_or_blank(
        value: object,
        decimals: int = 3,
    ) -> object:
        """Return a rounded finite number or an empty CSV cell."""
        if value is None or value == "":
            return ""

        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""

        if not math.isfinite(number):
            return ""

        return round(number, decimals)

    def accepted_or_blank(value: object) -> object:
        """Return 1, 0, or an empty CSV cell."""
        if value is None or value == "":
            return ""

        return 1 if bool(value) else 0

    header = [
        "iteration",
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
        "violations",
        "v_max",
        "p95",
        "cvar95",
        "based_on_true",
        "held_step",
        "held_step_scale",
        "accepted",
        "trust_ratio_used",
        "trust_ratio_next",
        "trust_p_ratio_used",
        "trust_p_ratio_next",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for record in runtime_log:
            writer.writerow([
                record.get("iteration", ""),
                number_or_blank(
                    record.get("runtime_sec"),
                    decimals=3,
                ),
                number_or_blank(
                    record.get("totaltime_sec"),
                    decimals=3,
                ),
                record.get("mode", ""),
                number_or_blank(record.get("tau_min")),
                number_or_blank(record.get("tau_max")),
                number_or_blank(record.get("tau_mean")),
                number_or_blank(record.get("improve"), decimals=6),
                number_or_blank(record.get("within")),
                record.get("window_cap", ""),
                record.get("phase", ""),
                record.get("violations", ""),
                number_or_blank(record.get("v_max")),
                number_or_blank(record.get("p95")),
                number_or_blank(record.get("cvar95")),
                record.get("based_on_true", ""),
                record.get("held_step", ""),
                number_or_blank(
                    record.get("held_step_scale"),
                    decimals=6,
                ),
                accepted_or_blank(record.get("accepted")),
                number_or_blank(
                    record.get("trust_ratio_used"),
                    decimals=6,
                ),
                number_or_blank(
                    record.get("trust_ratio_next"),
                    decimals=6,
                ),
                number_or_blank(
                    record.get("trust_p_ratio_used"),
                    decimals=6,
                ),
                number_or_blank(
                    record.get("trust_p_ratio_next"),
                    decimals=6,
                ),
            ])

        if cli_used:
            writer.writerow([])
            writer.writerow(["CLI", cli_used])

# ---------- G-code patch（最終一次） ----------
"""
Legacy sequential patcher.

WARNING:
This function patches extrusion feedrates by the order of G1+E commands.
It is unsafe for segmented G-code containing pre_infill, prime lines,
or any non-print extrusion. Do not use for final/best optimized output.
Use patch_gcode_by_segidx() instead.
"""
def limit_relative_target(
    anchor: List[float],
    proposal: List[float],
    max_ratio: float,
    lower_bound: float,
    upper_bound: float,
    reference_floor: float = 1.0,
) -> List[float]:
    """Limit cumulative changes relative to a TRUE-confirmed anchor.

    Args:
        anchor: TRUE-confirmed parameter values.
        proposal: Parameter values proposed by the controller.
        max_ratio: Maximum relative change from each anchor value.
        lower_bound: Absolute lower parameter bound.
        upper_bound: Absolute upper parameter bound.
        reference_floor: Minimum reference magnitude used when an anchor
            value is zero or very small.

    Returns:
        The limited target vector.
    """
    if len(anchor) != len(proposal):
        raise ValueError("anchor and proposal must have the same length.")

    limited = []

    for old, new in zip(anchor, proposal):
        old = float(old)
        new = float(new)

        reference = max(abs(old), float(reference_floor))
        max_change = reference * float(max_ratio)

        low = max(float(lower_bound), old - max_change)
        high = min(float(upper_bound), old + max_change)

        limited.append(max(low, min(new, high)))

    return limited


def extrapolate_held_step(
    current: List[float],
    safe_candidate: List[float],
    direction: List[float],
    step_scale: float,
    total_max_ratio: float,
    lower_bound: float,
    upper_bound: float,
    reference_floor: float,
) -> List[float]:
    """Advance one open-loop HELD step along an accepted direction."""

    if not (
        len(current)
        == len(safe_candidate)
        == len(direction)
    ):
        raise ValueError("HELD vectors must have the same length.")

    proposal = [
        float(value) + float(step_scale) * float(delta)
        for value, delta in zip(current, direction)
    ]

    # Total HELD extrapolation is limited relative to the safe candidate.
    return limit_relative_target(
        anchor=safe_candidate,
        proposal=proposal,
        max_ratio=total_max_ratio,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        reference_floor=reference_floor,
    )


def candidate_rank(
    within: float,
    violations: int,
    violation_max: float,
    global_error: float,
) -> tuple:
    """Build a risk-first candidate comparison tuple.

    A higher tuple represents a better candidate. Violation count and
    maximum violation are prioritized over coverage and total error.
    """
    within_value = float(within)
    if math.isnan(within_value):
        within_value = float("-inf")

    return (
        -int(violations),
        -float(violation_max),
        within_value,
        -float(global_error),
    )

# ---------- Main ----------

def main():
    """Run the residence-time optimization CLI pipeline.

    Residence time is evaluated using the authoritative external TRUE FIFO
    model. The FAST option is retained only for command-line compatibility
    and is disabled for optimization.

    Main steps:
        1. Parse CLI arguments and create the output directory.
        2. Mirror stdout/stderr into a timestamped log file.
        3. Load segments from the input CSV and infer an initial τ₀.
        4. Initialize feedrates and window_cap based on nozzle volume.
        5. For each iteration:
            - Rebuild residence times at authoritative TRUE FIFO checkpoints.
            - Compute a new feedrate field via optimize_feedforward().
            - Re-simulate τ with updated feedrates.
            - Update global error, τ statistics, and within% coverage.
            - Optionally adapt window_cap (rise/fall cycle logic).
            - Optionally save per-iteration CSV and runtime log.
        6. Track the best retained TRUE state using risk-first ordering.
        7. Write a final summary CSV and runtime log.
        8. Optionally patch the segmented G-code with:
            - best feedrates (…_best.gcode)
            - final feedrates (gcode_out)

    Key CLI arguments:
        --input (str): Input segments CSV (segment_fifo_builder_v6 output).
        --vnozzle (float): Mixing chamber volume V_nozzle in mm³.
        --dnozzle (float): Nozzle diameter in mm.
        --use_boundary_window (bool): If set, windows stop at V_nozzle boundaries.
        --window_cap (int): Initial forward window size (adapted over time).
        --Kp, --Kd (float): PD controller gains for ΔT correction.
        --Fmin, --Fmax (float): Feedrate bounds.
        --dFmax (float): Max relative feedrate step per row.
        --deltaT_cap_ratio (float): Max |ΔT| as a fraction of τ*.
        --max_iter (int): Maximum number of state-machine steps.
        --out_dir (str): Output directory for logs and CSV files.
        --out_prefix (str): Prefix for per-iteration CSV filenames.
        --summary (str): Summary CSV filename.
        --enable_timer (bool): If set, record per-iteration runtimes.
        --adaptive_window (bool): Enable cyclic window_cap adaptation.
        --cooldown (int): Minimum resolved TRUE anchors between window_cap changes.
        --stall_improve (float): Relative improvement threshold for stall detection.
        --stall_runs (int): Required consecutive stalls to flip phase.
        --save_intermediate (bool): Save per-iteration CSV and G-code if enabled.
        --fifo_mode ({"fast","true"}): TRUE is required; FAST is disabled.
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

    ap.add_argument(
        "--max_iter",
        type=int,
        default=50,
        help=(
            "Maximum number of state-machine steps. With four HELD steps, "
            "one full cycle uses one TRUE probe, four HELD steps, and one "
            "TRUE endpoint validation."
        ),
    )
    ap.add_argument("--out_dir", default="optimization_outputs")
    ap.add_argument("--out_prefix", default="iteration_")
    ap.add_argument("--summary", default="summary.csv")

    ap.add_argument("--enable_timer", action="store_true")
    ap.add_argument("--adaptive_window", action="store_true")
    ap.add_argument("--cooldown", type=int, default=5,
                    help="Minimum resolved TRUE anchors between window_cap adjustments.")
    # cyclic rule uses固定閾值: (prev_ge - ge)/prev_ge < 0.002 持續 5 次
    #ap.add_argument("--stall_improve", type=float, default=0.002)
    ap.add_argument("--stall_runs", type=int, default=5)

    ap.add_argument("--save_intermediate", action="store_true",
                help="Save per-iteration CSV and G-code outputs if set. Otherwise only summary will be written.")
    
    ap.add_argument(
        "--fifo_mode",
        choices=["fast", "true"],
        default="true",
        help=(
            "FIFO simulation mode. TRUE is required for optimization. "
            "FAST is retained only for command-line compatibility and is disabled."
        ),
    )

    ap.add_argument(
        "--true_interval_dFmax",
        type=float,
        default=0.05,
        help=(
            "Maximum relative feed-rate change of a probe candidate "
            "from the current TRUE anchor."
        ),
    )

    ap.add_argument(
        "--true_interval_dPmax",
        type=float,
        default=0.10,
        help=(
            "Maximum relative dwell-time change of a probe candidate "
            "from the current TRUE anchor."
        ),
    )

    ap.add_argument(
        "--trust_shrink",
        type=float,
        default=0.5,
        help="Factor used to reduce the trust region after a rejected candidate.",
    )

    ap.add_argument(
        "--trust_min",
        type=float,
        default=0.005,
        help="Minimum cumulative relative feed-rate trust region.",
    )

    ap.add_argument(
        "--trust_p_min",
        type=float,
        default=0.005,
        help="Minimum cumulative relative dwell-time trust region.",
    )

    ap.add_argument(
        "--held_max_steps",
        type=int,
        default=4,
        help="Number of open-loop HELD extrapolation steps after an accepted probe.",
    )

    ap.add_argument(
        "--held_step_scale",
        type=float,
        default=0.5,
        help="Initial HELD step relative to the accepted probe direction.",
    )

    ap.add_argument(
        "--held_step_decay",
        type=float,
        default=0.5,
        help="Geometric decay applied to consecutive HELD step sizes.",
    )

    ap.add_argument(
        "--held_total_dFmax",
        type=float,
        default=0.05,
        help="Maximum cumulative HELD feed-rate extrapolation from the safe candidate.",
    )

    ap.add_argument(
        "--held_total_dPmax",
        type=float,
        default=0.10,
        help="Maximum cumulative HELD dwell extrapolation from the safe candidate.",
    )

    ap.add_argument(
        "--held_failure_shrink",
        type=float,
        default=0.5,
        help="Reduction applied to the HELD base step after a worse endpoint.",
    )

    ap.add_argument(
        "--held_step_scale_min",
        type=float,
        default=0.05,
        help="Minimum adaptive HELD base-step scale.",
    )

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
                    help=(
                        "Enable direct PID adjustment of travel feedrates inside the selected PID window. "
                        "Travel rows are not target segments and are not used for window boundary or V_nozzle accumulation."
                    ))
    #--------------------------

    args = ap.parse_args()
    if not (0.0 <= args.alpha_min <= 1.0 and 0.0 <= args.alpha_ideal <= 1.0 and 0.0 <= args.alpha_max <= 1.0):
        print("⛔ alpha values must be within [0, 1].")
        return

    if not (args.alpha_min <= args.alpha_ideal <= args.alpha_max):
        print("⛔ Require alpha_min <= alpha_ideal <= alpha_max.")
        return
    
    if args.fifo_mode != "true":
        print(
            "⛔ The FAST FIFO optimizer is disabled because its timing model "
            "does not match the authoritative TRUE FIFO rebuild. "
            "Please use --fifo_mode true."
        )
        return

    if not (0.0 < args.true_interval_dFmax <= 1.0):
        print("⛔ --true_interval_dFmax must be within (0, 1].")
        return

    if not (0.0 < args.true_interval_dPmax <= 1.0):
        print("⛔ --true_interval_dPmax must be within (0, 1].")
        return

    if not (0.0 < args.trust_shrink < 1.0):
        print("⛔ --trust_shrink must be within (0, 1).")
        return

    if not (0.0 < args.trust_min <= args.true_interval_dFmax):
        print(
            "⛔ --trust_min must be positive and no larger than "
            "--true_interval_dFmax."
        )
        return
    
    if not (0.0 < args.trust_p_min <= args.true_interval_dPmax):
        print(
            "⛔ --trust_p_min must be positive and no larger than "
            "--true_interval_dPmax."
        )
        return
    
    if args.held_max_steps < 0:
        print("⛔ --held_max_steps must be >= 0.")
        return

    if not (0.0 < args.held_step_scale <= 1.0):
        print("⛔ --held_step_scale must be within (0, 1].")
        return

    if not (0.0 < args.held_step_decay < 1.0):
        print("⛔ --held_step_decay must be within (0, 1).")
        return

    if not (0.0 < args.held_total_dFmax <= 1.0):
        print("⛔ --held_total_dFmax must be within (0, 1].")
        return

    if not (0.0 < args.held_total_dPmax <= 1.0):
        print("⛔ --held_total_dPmax must be within (0, 1].")
        return

    if not (0.0 < args.held_failure_shrink < 1.0):
        print("⛔ --held_failure_shrink must be within (0, 1).")
        return

    if not (
        0.0
        < args.held_step_scale_min
        <= args.held_step_scale
    ):
        print(
            "⛔ --held_step_scale_min must be positive and no larger "
            "than --held_step_scale."
        )
        return
    
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

    if args.use_boundary_window:
        # Boundary mode:
        # window_cap is ignored.
        # Keep it as int only for logging compatibility.
        window_cap = int(args.window_cap)

        # Adaptive window should not be used in boundary mode.
        # The boundary is controlled by V_nozzle and target segment.
        cap_lower = 1
        cap_upper = 1

        if args.adaptive_window:
            print("⚠️  --adaptive_window is ignored because --use_boundary_window is enabled.")
            args.adaptive_window = False

        if args.horizon_sec and args.horizon_sec > 0:
            print("⚠️  --horizon_sec is ignored because --use_boundary_window is enabled.")

    else:
        cap_upper, cap_lower = compute_cap_bounds_from_cycle(rows, args.vnozzle, args.dnozzle)
        window_cap = max(cap_lower, min(int(args.window_cap), cap_upper))

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
        f"[Iter 000] mode=CSV_INIT | Δt={iter0_runtime:.3f}s | Total={total_elapsed0:.1f}s | "
        f"τ_min={r2(tau_min0):>6}, τ_max={r2(tau_max0):>6}, τ_mean={r2(tau_mean0):>6} | "
        f"error={ge0:.2f} | Δerror=+0.00 | ratio=1.00000 | "
        f"within={r2(within0):>6}% | v_cnt={int(v_cnt0):>4} | v_max={r2(v_max0):>6}s | "
        f"p95={r2(p95_0):>6}s | cvar95={r2(cvar95_0):>6}s"
    )

    runtime_log.append({
        "iteration": 0,
        "runtime_sec": iter0_runtime,
        "totaltime_sec": total_elapsed0,
        "mode": "CSV_INIT",
        "tau_min": tau_min0,
        "tau_max": tau_max0,
        "tau_mean": tau_mean0,
        "improve": 0.0,
        "within": within0,
        "window_cap": window_cap,
        "phase": "init",
        "violations": int(v_cnt0),
        "v_max": float(v_max0),
        "p95": float(p95_0),
        "cvar95": float(cvar95_0),
        "based_on_true": "",
        "held_step": "",
        "held_step_scale": "",
        "accepted": "",
        "trust_ratio_used": "",
        "trust_ratio_next": float(args.true_interval_dFmax),
        "trust_p_ratio_used": "",
        "trust_p_ratio_next": float(args.true_interval_dPmax),
    })
    
    summary_hist.append({
        "iteration": 0,
        "mode": "CSV_INIT",
        "accepted": None,
        "global_error": ge0,
        "tau_min": tau_min0,
        "tau_max": tau_max0,
        "tau_mean": tau_mean0,
        "within%": within0,
        "violations": int(v_cnt0),
        "v_max": float(v_max0),
        "p95": float(p95_0),
        "cvar95": float(cvar95_0),
        "window_cap": window_cap,
        "trust_ratio_used": None,
        "trust_ratio_next": float(args.true_interval_dFmax),
        "trust_p_ratio_used": None,
        "trust_p_ratio_next": float(args.true_interval_dPmax),
        "iteration_runtime": iter0_runtime,
    })

    # ---- travel scaling state ----
    s_travel = 1.0
    F0_travel = [infer_F0_row(r, args.Fmin) for r in rows]  # keep original travel structure

    # --- best tracking ---
    def is_better_candidate(
        within,
        v_cnt,
        v_max,
        ge,
        best_within,
        best_v_cnt,
        best_v_max,
        best_ge,
    ) -> bool:
        """Return whether a candidate is better using risk-first ordering."""
        return candidate_rank(
            within,
            v_cnt,
            v_max,
            ge,
        ) > candidate_rank(
            best_within,
            best_v_cnt,
            best_v_max,
            best_ge,
        )

    
    # baseline 不算 TRUE best（因為是 CSV init），所以 best_iter_true = -1
    best_F_true = None
    best_P_true = None
    
    best_iter_true = -1
    best_ge_true = float("inf")
    best_within_true = -1.0
    best_v_cnt_true = float("inf")
    best_v_max_true = float("inf")
    
    # Authoritative TRUE anchor.
    true_anchor_valid = False
    true_anchor_iteration = 0

    true_anchor_F = F_curr[:]
    true_anchor_P_ms = P_curr_ms[:]

    true_anchor_t_in = None
    true_anchor_t_out = None
    true_anchor_tau = None

    # Accepted TRUE probe candidate; always safe for rollback/output.
    safe_candidate_iteration = 0

    safe_candidate_F = None
    safe_candidate_P_ms = None

    safe_candidate_t_in = None
    safe_candidate_t_out = None
    safe_candidate_tau = None

    # Open-loop HELD extrapolation.
    held_active = False
    held_step_index = 0

    held_direction_F = [0.0] * len(rows)
    held_direction_P_ms = [0.0] * len(rows)

    held_F_work = F_curr[:]
    held_P_work_ms = P_curr_ms[:]

    current_held_step_scale = float(args.held_step_scale)

    # Candidate trust regions.
    current_trust_ratio = float(args.true_interval_dFmax)
    current_trust_P_ratio = float(args.true_interval_dPmax)

    F_final = F_curr[:]
    P_final_ms = P_curr_ms[:]

    phase = "rise"
    low_improve_count = 0
    prev_ge = None
    accepted_checkpoint_count = 0
    last_adjust_checkpoint = 0
    check_count = 0

    try:
        for it in range(1, args.max_iter + 1):
            iter_start = time.time()
            
            mode = ""
            decision_accepted = None
            allow_best_update = False
            anchor_resolved = False
            probe_unchanged = False
            stop_after_iteration_reason = None

            based_on_true_iteration = 0
            trust_ratio_used = None
            trust_P_ratio_used = None
            window_debug = {}

            retained_F = None
            retained_P_ms = None
            retained_t_in = None
            retained_t_out = None
            retained_tau = None

            record_F_old = None
            record_tau_old = None

            if not true_anchor_valid:
                mode = "TRUE_ANCHOR"
                based_on_true_iteration = 0

                record_F_old = F_curr[:]
                record_tau_old = tau_000[:]

                ok_anchor, anchor_t_in, anchor_t_out, anchor_tau, _, _ = (
                    simulate_fifo_v6_wrapper(
                        gcode_template_path=args.gcode_template,
                        rows=rows,
                        F_vec=F_curr,
                        P_vec_ms=P_curr_ms,
                        V_nozzle=args.vnozzle,
                        d_nozzle=args.dnozzle,
                        iteration=it,
                        out_root=args.out_dir,
                        enable_travel_opt=args.enable_travel_opt,
                        s_travel=s_travel,
                    )
                )

                if not ok_anchor:
                    mode = "TRUE_ANCHOR_FAILED"
                    iter_runtime = time.time() - iter_start

                    runtime_log.append({
                        "iteration": it,
                        "runtime_sec": iter_runtime,
                        "totaltime_sec": time.time() - total_start,
                        "mode": mode,
                        "tau_min": float("nan"),
                        "tau_max": float("nan"),
                        "tau_mean": float("nan"),
                        "improve": float("nan"),
                        "within": float("nan"),
                        "window_cap": window_cap,
                        "phase": phase,
                        "violations": "",
                        "v_max": float("nan"),
                        "p95": float("nan"),
                        "cvar95": float("nan"),
                        "based_on_true": 0,
                        "held_step": "",
                        "held_step_scale": "",
                        "accepted": None,
                        "trust_ratio_used": "",
                        "trust_ratio_next": current_trust_ratio,
                        "trust_p_ratio_used": "",
                        "trust_p_ratio_next": current_trust_P_ratio,
                    })

                    print(
                        f"⚠️ [Iter {it:03d}] Initial TRUE anchor failed. "
                        "No parameter update was applied."
                    )
                    continue

                true_anchor_valid = True
                true_anchor_iteration = it

                true_anchor_F = F_curr[:]
                true_anchor_P_ms = P_curr_ms[:]
                true_anchor_t_in = anchor_t_in[:]
                true_anchor_t_out = anchor_t_out[:]
                true_anchor_tau = anchor_tau[:]

                retained_F = true_anchor_F[:]
                retained_P_ms = true_anchor_P_ms[:]
                retained_t_in = true_anchor_t_in[:]
                retained_t_out = true_anchor_t_out[:]
                retained_tau = true_anchor_tau[:]

                F_final = retained_F[:]
                P_final_ms = retained_P_ms[:]

                allow_best_update = True
                anchor_resolved = True

            elif held_active and held_step_index < args.held_max_steps:
                step_scale = (
                    current_held_step_scale
                    * (args.held_step_decay ** held_step_index)
                )

                held_F_work = extrapolate_held_step(
                    current=held_F_work,
                    safe_candidate=safe_candidate_F,
                    direction=held_direction_F,
                    step_scale=step_scale,
                    total_max_ratio=args.held_total_dFmax,
                    lower_bound=args.Fmin,
                    upper_bound=args.Fmax,
                    reference_floor=args.Fmin,
                )

                held_P_work_ms = extrapolate_held_step(
                    current=held_P_work_ms,
                    safe_candidate=safe_candidate_P_ms,
                    direction=held_direction_P_ms,
                    step_scale=step_scale,
                    total_max_ratio=args.held_total_dPmax,
                    lower_bound=0.0,
                    upper_bound=600000.0,
                    reference_floor=1000.0,
                )

                held_step_index += 1

                F_curr = held_F_work[:]
                P_curr_ms = held_P_work_ms[:]

                # 未經 TRUE FIFO，只能保留 safe candidate 作為輸出。
                F_final = safe_candidate_F[:]
                P_final_ms = safe_candidate_P_ms[:]

                iter_runtime = time.time() - iter_start

                runtime_log.append({
                    "iteration": it,
                    "runtime_sec": iter_runtime,
                    "totaltime_sec": time.time() - total_start,
                    "mode": "HELD_EXTRAPOLATE",
                    "tau_min": "",
                    "tau_max": "",
                    "tau_mean": "",
                    "improve": "",
                    "within": "",
                    "window_cap": window_cap,
                    "phase": phase,
                    "violations": "",
                    "v_max": "",
                    "p95": "",
                    "cvar95": "",
                    "based_on_true": safe_candidate_iteration,
                    "held_step": held_step_index,
                    "held_step_scale": step_scale,
                    "accepted": None,
                    "trust_ratio_used": "",
                    "trust_ratio_next": current_trust_ratio,
                    "trust_p_ratio_used": "",
                    "trust_p_ratio_next": current_trust_P_ratio,
                })

                print(
                    f"[Iter {it:03d}] mode=HELD_EXTRAPOLATE | "
                    f"step={held_step_index}/{args.held_max_steps} | "
                    f"scale={step_scale:.6f}"
                )

                continue
            
            elif held_active:
                based_on_true_iteration = safe_candidate_iteration

                record_F_old = safe_candidate_F[:]
                record_tau_old = safe_candidate_tau[:]

                ok_held, held_t_in, held_t_out, held_tau, _, _ = (
                    simulate_fifo_v6_wrapper(
                        gcode_template_path=args.gcode_template,
                        rows=rows,
                        F_vec=held_F_work,
                        P_vec_ms=held_P_work_ms,
                        V_nozzle=args.vnozzle,
                        d_nozzle=args.dnozzle,
                        iteration=it,
                        out_root=args.out_dir,
                        enable_travel_opt=args.enable_travel_opt,
                        s_travel=s_travel,
                    )
                )

                held_endpoint_accepted = False

                if ok_held:
                    safe_ge = global_error_abs_sum(
                        safe_candidate_tau, rows, tau_star
                    )
                    safe_within = within_percentage_bounds(
                        safe_candidate_tau, rows, tau_lo, tau_hi
                    )
                    (
                        safe_v_cnt,
                        _,
                        safe_v_max,
                        _,
                        _,
                    ) = risk_metrics(
                        safe_candidate_tau, rows, tau_lo, tau_hi
                    )

                    held_ge = global_error_abs_sum(
                        held_tau, rows, tau_star
                    )
                    held_within = within_percentage_bounds(
                        held_tau, rows, tau_lo, tau_hi
                    )
                    (
                        held_v_cnt,
                        _,
                        held_v_max,
                        _,
                        _,
                    ) = risk_metrics(
                        held_tau, rows, tau_lo, tau_hi
                    )

                    held_endpoint_accepted = (
                        candidate_rank(
                            held_within,
                            held_v_cnt,
                            held_v_max,
                            held_ge,
                        )
                        > candidate_rank(
                            safe_within,
                            safe_v_cnt,
                            safe_v_max,
                            safe_ge,
                        )
                    )

                if held_endpoint_accepted:
                    mode = "TRUE_HELD_ACCEPTED"
                    decision_accepted = True
                    allow_best_update = True

                    retained_iteration = it
                    retained_F = held_F_work[:]
                    retained_P_ms = held_P_work_ms[:]
                    retained_t_in = held_t_in[:]
                    retained_t_out = held_t_out[:]
                    retained_tau = held_tau[:]

                    current_held_step_scale = min(
                        args.held_step_scale,
                        current_held_step_scale * 1.10,
                    )

                else:
                    mode = (
                        "TRUE_HELD_ROLLBACK"
                        if ok_held
                        else "TRUE_HELD_FIFO_FAILED"
                    )
                    decision_accepted = False if ok_held else None

                    retained_iteration = safe_candidate_iteration
                    retained_F = safe_candidate_F[:]
                    retained_P_ms = safe_candidate_P_ms[:]
                    retained_t_in = safe_candidate_t_in[:]
                    retained_t_out = safe_candidate_t_out[:]
                    retained_tau = safe_candidate_tau[:]

                    # 只有成功執行 FIFO、但 endpoint 較差時才縮小。
                    if ok_held:
                        current_held_step_scale = max(
                            args.held_step_scale_min,
                            current_held_step_scale
                            * args.held_failure_shrink,
                        )

                true_anchor_valid = True
                true_anchor_iteration = retained_iteration
                true_anchor_F = retained_F[:]
                true_anchor_P_ms = retained_P_ms[:]
                true_anchor_t_in = retained_t_in[:]
                true_anchor_t_out = retained_t_out[:]
                true_anchor_tau = retained_tau[:]

                held_active = False
                held_step_index = 0
                anchor_resolved = True

                F_curr = retained_F[:]
                P_curr_ms = retained_P_ms[:]
                F_final = retained_F[:]
                P_final_ms = retained_P_ms[:]
                
            else:
                based_on_true_iteration = true_anchor_iteration

                record_F_old = true_anchor_F[:]
                record_tau_old = true_anchor_tau[:]

                tau_old = true_anchor_tau[:]
            
            
                F_next_base, window_debug = optimize_feedforward_hybrid(
                    rows=rows,
                    F_current=true_anchor_F,
                    tau_current=true_anchor_tau,
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
                    t_out_current=true_anchor_t_out,
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
                    smooth_passes=args.smooth_passes,
                    enable_travel_opt=args.enable_travel_opt
                )

                # 2.5) 再 dwell（第二）
                P_next_ms = optimize_dwell_P_ms(
                    rows=rows,
                    P_current_ms=true_anchor_P_ms,
                    tau_current=true_anchor_tau,
                    tau_lo=tau_lo,
                    tau_star=tau_star,
                    tau_hi=tau_hi,
                    KpP=0.15,
                    dPmax_ratio=0.10,
                    Pmin_ms=0.0,
                    Pmax_ms=600000.0
                )

                # 2.8) Travel PID adjustment
                # Travel rows are already included in the PID adjustment window
                # inside optimize_feedforward_hybrid() when enable_travel_opt=True.
                F_next = F_next_base

                # Preserve the trust regions actually used to build this candidate.
                # The current trust values may be changed after acceptance or rejection.
                trust_ratio_used = current_trust_ratio
                trust_P_ratio_used = current_trust_P_ratio

                # Limit the cumulative feed-rate change relative to the TRUE anchor.
                F_next = limit_relative_target(
                    anchor=true_anchor_F,
                    proposal=F_next_base,
                    max_ratio=trust_ratio_used,
                    lower_bound=args.Fmin,
                    upper_bound=args.Fmax,
                    reference_floor=args.Fmin,
                )

                P_next_ms = limit_relative_target(
                    anchor=true_anchor_P_ms,
                    proposal=P_next_ms,
                    max_ratio=trust_P_ratio_used,
                    lower_bound=0.0,
                    upper_bound=600000.0,
                    reference_floor=1000.0,
                )

                # 3) Validate the proposed target with the authoritative TRUE FIFO model.
                ok_new, t_in_new, t_out_new, tau_new, csv_new, gcode_new = (
                    simulate_fifo_v6_wrapper(
                        gcode_template_path=args.gcode_template,
                        rows=rows,
                        F_vec=F_next,
                        P_vec_ms=P_next_ms,
                        V_nozzle=args.vnozzle,
                        d_nozzle=args.dnozzle,
                        iteration=it,
                        out_root=args.out_dir,
                        enable_travel_opt=args.enable_travel_opt,
                        s_travel=s_travel,
                    )
                )
                
                if not ok_new:
                    mode = "TRUE_PROBE_FIFO_FAILED"

                    F_curr = true_anchor_F[:]
                    P_curr_ms = true_anchor_P_ms[:]
                    F_final = true_anchor_F[:]
                    P_final_ms = true_anchor_P_ms[:]

                    held_active = False
                    held_step_index = 0

                    iter_runtime = time.time() - iter_start

                    runtime_log.append({
                        "iteration": it,
                        "runtime_sec": iter_runtime,
                        "totaltime_sec": time.time() - total_start,
                        "mode": mode,
                        "tau_min": "",
                        "tau_max": "",
                        "tau_mean": "",
                        "improve": "",
                        "within": "",
                        "window_cap": window_cap,
                        "phase": phase,
                        "violations": "",
                        "v_max": "",
                        "p95": "",
                        "cvar95": "",
                        "based_on_true": true_anchor_iteration,
                        "held_step": "",
                        "held_step_scale": "",
                        "accepted": None,
                        "trust_ratio_used": trust_ratio_used,
                        "trust_ratio_next": current_trust_ratio,
                        "trust_p_ratio_used": trust_P_ratio_used,
                        "trust_p_ratio_next": current_trust_P_ratio,
                    })

                    print(
                        f"⚠️ [Iter {it:03d}] TRUE probe FIFO failed. "
                        "Keeping the current TRUE anchor."
                    )
                    continue
                
                # Evaluate the previous TRUE-confirmed anchor.
                old_ge = global_error_abs_sum(
                    tau_old,
                    rows,
                    tau_star,
                )
                old_within = within_percentage_bounds(
                    tau_old,
                    rows,
                    tau_lo,
                    tau_hi,
                )
                (
                    old_v_cnt,
                    old_v_sum,
                    old_v_max,
                    old_p95,
                    old_cvar95,
                ) = risk_metrics(
                    tau_old,
                    rows,
                    tau_lo,
                    tau_hi,
                )

                # Evaluate the new TRUE candidate.
                candidate_ge = global_error_abs_sum(
                    tau_new,
                    rows,
                    tau_star,
                )
                candidate_within = within_percentage_bounds(
                    tau_new,
                    rows,
                    tau_lo,
                    tau_hi,
                )
                (
                    candidate_v_cnt,
                    candidate_v_sum,
                    candidate_v_max,
                    candidate_p95,
                    candidate_cvar95,
                ) = risk_metrics(
                    tau_new,
                    rows,
                    tau_lo,
                    tau_hi,
                )

                candidate_changed = (
                    any(
                        abs(new - old) > 1e-9
                        for old, new in zip(true_anchor_F, F_next)
                    )
                    or
                    any(
                        abs(new - old) > 1e-9
                        for old, new in zip(true_anchor_P_ms, P_next_ms)
                    )
                )

                candidate_accepted = (
                    candidate_changed
                    and candidate_rank(
                        candidate_within,
                        candidate_v_cnt,
                        candidate_v_max,
                        candidate_ge,
                    )
                    > candidate_rank(
                        old_within,
                        old_v_cnt,
                        old_v_max,
                        old_ge,
                    )
                )
                
                if candidate_accepted:
                    mode = "TRUE_PROBE_ACCEPTED"
                    decision_accepted = True
                    allow_best_update = True

                    safe_candidate_iteration = it

                    safe_candidate_F = F_next[:]
                    safe_candidate_P_ms = P_next_ms[:]

                    safe_candidate_t_in = t_in_new[:]
                    safe_candidate_t_out = t_out_new[:]
                    safe_candidate_tau = tau_new[:]
                    
                    retained_F = safe_candidate_F[:]
                    retained_P_ms = safe_candidate_P_ms[:]
                    retained_t_in = safe_candidate_t_in[:]
                    retained_t_out = safe_candidate_t_out[:]
                    retained_tau = safe_candidate_tau[:]

                    held_direction_F = [
                        new - old
                        for old, new in zip(true_anchor_F, safe_candidate_F)
                    ]

                    held_direction_P_ms = [
                        new - old
                        for old, new in zip(
                            true_anchor_P_ms,
                            safe_candidate_P_ms,
                        )
                    ]

                    held_F_work = safe_candidate_F[:]
                    held_P_work_ms = safe_candidate_P_ms[:]
                    held_step_index = 0
                    held_active = args.held_max_steps > 0

                    F_curr = safe_candidate_F[:]
                    P_curr_ms = safe_candidate_P_ms[:]

                    # Safe even if interrupted during HELD.
                    F_final = safe_candidate_F[:]
                    P_final_ms = safe_candidate_P_ms[:]

                    current_trust_ratio = min(
                        args.true_interval_dFmax,
                        current_trust_ratio * 1.10,
                    )

                    current_trust_P_ratio = min(
                        args.true_interval_dPmax,
                        current_trust_P_ratio * 1.10,
                    )

                    # If HELD is disabled, candidate immediately becomes the new anchor.
                    if not held_active:
                        anchor_resolved = True
                        true_anchor_valid = True
                        true_anchor_iteration = it

                        true_anchor_F = safe_candidate_F[:]
                        true_anchor_P_ms = safe_candidate_P_ms[:]

                        true_anchor_t_in = safe_candidate_t_in[:]
                        true_anchor_t_out = safe_candidate_t_out[:]
                        true_anchor_tau = safe_candidate_tau[:]
                        
                elif not candidate_changed:
                    mode = "TRUE_PROBE_UNCHANGED"
                    decision_accepted = None
                    allow_best_update = True
                    anchor_resolved = True
                    probe_unchanged = True

                    # The parameters are unchanged, but the identical state
                    # completed a successful authoritative TRUE FIFO rebuild.
                    true_anchor_valid = True
                    true_anchor_iteration = it
                    true_anchor_F = F_next[:]
                    true_anchor_P_ms = P_next_ms[:]
                    true_anchor_t_in = t_in_new[:]
                    true_anchor_t_out = t_out_new[:]
                    true_anchor_tau = tau_new[:]

                    retained_F = true_anchor_F[:]
                    retained_P_ms = true_anchor_P_ms[:]
                    retained_t_in = true_anchor_t_in[:]
                    retained_t_out = true_anchor_t_out[:]
                    retained_tau = true_anchor_tau[:]

                    F_curr = retained_F[:]
                    P_curr_ms = retained_P_ms[:]
                    F_final = retained_F[:]
                    P_final_ms = retained_P_ms[:]

                    held_active = False
                    held_step_index = 0
                
                else:
                    mode = "TRUE_PROBE_REJECTED"
                    decision_accepted = False
                    allow_best_update = False
                    anchor_resolved = False
                    # Keep the same authoritative anchor.
                    F_curr = true_anchor_F[:]
                    P_curr_ms = true_anchor_P_ms[:]

                    t_in_new = true_anchor_t_in[:]
                    t_out_new = true_anchor_t_out[:]
                    tau_new = true_anchor_tau[:]

                    F_next = true_anchor_F[:]
                    P_next_ms = true_anchor_P_ms[:]

                    F_final = true_anchor_F[:]
                    P_final_ms = true_anchor_P_ms[:]

                    retained_F = true_anchor_F[:]
                    retained_P_ms = true_anchor_P_ms[:]
                    retained_t_in = true_anchor_t_in[:]
                    retained_t_out = true_anchor_t_out[:]
                    retained_tau = true_anchor_tau[:]

                    held_active = False
                    held_step_index = 0

                    # Only a valid but worse candidate shrinks candidate trust.
                    if candidate_changed:
                        current_trust_ratio = max(
                            args.trust_min,
                            current_trust_ratio * args.trust_shrink,
                        )

                        current_trust_P_ratio = max(
                            args.trust_p_min,
                            current_trust_P_ratio * args.trust_shrink,
                        )
            
                        if (
                            trust_ratio_used <= args.trust_min + EPS
                            and trust_P_ratio_used
                            <= args.trust_p_min + EPS
                        ):
                            mode = "TRUE_PROBE_REJECTED_AT_MIN_TRUST"
                            stop_after_iteration_reason = (
                                "The candidate was rejected while both trust "
                                "regions were already at their minimum values."
                            )
            
            F_next = retained_F[:]
            P_next_ms = retained_P_ms[:]
            t_in_new = retained_t_in[:]
            t_out_new = retained_t_out[:]
            tau_new = retained_tau[:]
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

            # --- debug: locate tau_min (extrusion only) ---
            extr_idx = [
                i
                for i, r in enumerate(rows)
                if (
                    r.is_extrusion
                    and i < len(tau_new)
                    and not math.isnan(tau_new[i])
                )
            ]

            i_min = (
                min(extr_idx, key=lambda i: tau_new[i])
                if extr_idx
                else None
            )

            if i_min is not None:
                tin = (
                    t_in_new[i_min]
                    if i_min < len(t_in_new)
                    else float("nan")
                )
                tout = (
                    t_out_new[i_min]
                    if i_min < len(t_out_new)
                    else float("nan")
                )

                print(
                    f"τ_min@i={i_min}, seg_idx={rows[i_min].seg_idx}, "
                    f"tau={tau_new[i_min]:.3f}, "
                    f"t_in={tin:.3f}, t_out={tout:.3f}, "
                    f"check={tout - tin:.3f}, "
                    f"note={rows[i_min].note}"
                )
            # --------------------------------------------

            iter_runtime = time.time() - iter_start

            # 這一個 runtime_log.append() 才是要替換的位置。
            runtime_log.append({
                "iteration": it,
                "runtime_sec": iter_runtime,
                "totaltime_sec": time.time() - total_start,
                "mode": mode,
                "tau_min": tau_min,
                "tau_max": tau_max,
                "tau_mean": tau_mean,
                "improve": (
                    (prev_ge - ge) / prev_ge
                    if prev_ge is not None and prev_ge > EPS
                    else 0.0
                ),
                "within": within,
                "window_cap": window_cap,
                "phase": phase,
                "violations": int(v_cnt),
                "v_max": float(v_max),
                "p95": float(p95_abs),
                "cvar95": float(cvar95),
                "based_on_true": based_on_true_iteration,
                "held_step": "",
                "held_step_scale": "",
                "accepted": decision_accepted,
                "trust_ratio_used": trust_ratio_used,
                "trust_ratio_next": current_trust_ratio,
                "trust_p_ratio_used": trust_P_ratio_used,
                "trust_p_ratio_next": current_trust_P_ratio,
            })

            if args.enable_timer:
                print(
                    f"🕒 Iteration {it} finished "
                    f"in {iter_runtime:.3f} s"
                )

            # 4) 更新最佳解（Best Iteration Logic）
            if allow_best_update:
                if best_iter_true < 0 or is_better_candidate(
                    within,
                    v_cnt,
                    v_max,
                    ge,
                    best_within_true,
                    best_v_cnt_true,
                    best_v_max_true,
                    best_ge_true,
                ):
                    best_iter_true = it
                    best_within_true = within
                    best_v_cnt_true = int(v_cnt)
                    best_v_max_true = float(v_max)
                    best_ge_true = ge
                    best_F_true = retained_F[:]
                    best_P_true = retained_P_ms[:]

            # Update checkpoint-based statistics only after the
            # authoritative anchor has been resolved.
            if prev_ge is not None and prev_ge > EPS:
                ratio = ge / prev_ge
                improve = (prev_ge - ge) / prev_ge
            else:
                ratio = 1.0
                improve = 0.0

            if anchor_resolved:
                accepted_checkpoint_count += 1

                if prev_ge is not None and prev_ge > EPS:
                    if ge > prev_ge:
                        low_improve_count += 1
                    else:
                        low_improve_count = 0
                else:
                    low_improve_count = 0
            
            # 5) 計算 error 
            if prev_ge is not None and prev_ge > EPS:
                error_delta = ge - prev_ge
                error_ratio = ge / prev_ge
            else:
                error_delta = 0.0
                error_ratio = 1.0

            # 6) 視需要調整 window_cap（循環模式）
            if (
                anchor_resolved
                and args.adaptive_window
                and (
                    accepted_checkpoint_count - last_adjust_checkpoint
                    >= args.cooldown
                )
            ):
                # 追蹤連續在上限或下限的次數
                if "at_upper_count" not in locals():
                    at_upper_count = 0
                    at_lower_count = 0

                if phase == "rise":
                    if window_cap < cap_upper:
                        window_cap = min(cap_upper, math.ceil(window_cap * 1.1))
                        last_adjust_checkpoint = accepted_checkpoint_count
                        at_upper_count = 0
                    else:
                        at_upper_count += 1
                        print(f"⚠️  Window_cap reached upper bound ({cap_upper}), count={at_upper_count}")
                        if at_upper_count >= args.stall_runs:
                            phase = "fall"
                            at_upper_count = 0
                            low_improve_count = 0
                            last_adjust_checkpoint = accepted_checkpoint_count
                            print(f"🔻 Auto-switch to FALL (stayed at upper bound for {args.stall_runs} runs)")

                elif phase == "fall":
                    if window_cap > cap_lower:
                        window_cap = max(cap_lower, math.floor(window_cap * 0.95)) 
                        last_adjust_checkpoint = accepted_checkpoint_count
                        at_lower_count = 0
                    else:
                        at_lower_count += 1
                        print(f"⚠️  Window_cap reached lower bound ({cap_lower}), count={at_lower_count}")
                        if at_lower_count >= args.stall_runs:
                            phase = "rise"
                            at_lower_count = 0
                            low_improve_count = 0
                            last_adjust_checkpoint = accepted_checkpoint_count
                            print(f"🔺 Auto-switch to RISE (stayed at lower bound for {args.stall_runs} runs)")

            # 7) 寫本輪 iteration CSV
            if args.save_intermediate:
                iter_csv = os.path.join(args.out_dir, f"{args.out_prefix}{it}.csv")
                write_iteration_csv(
                    iter_csv,
                    rows,
                    record_F_old,
                    retained_F,
                    record_tau_old,
                    retained_tau,
                    tau_star,
                    tau_lo,
                    tau_hi,
                    tau_grid_lookup,
                    alpha_grid_lookup
                )

            # 8) 更新 summary & log
            summary_hist.append({
                "iteration": it,
                "mode": mode,
                "accepted": decision_accepted,
                "global_error": ge,
                "tau_min": tau_min,
                "tau_max": tau_max,
                "tau_mean": tau_mean,
                "within%": within,
                "violations": int(v_cnt),
                "v_max": float(v_max),
                "p95": float(p95_abs),
                "cvar95": float(cvar95),
                "window_cap": window_cap,
                "trust_ratio_used": trust_ratio_used,
                "trust_ratio_next": current_trust_ratio,
                "trust_p_ratio_used": trust_P_ratio_used,
                "trust_p_ratio_next": current_trust_P_ratio,
                "iteration_runtime": iter_runtime,
            })
            summary_path = os.path.join(args.out_dir, args.summary)
            write_summary_csv(summary_path, summary_hist, cli_string)

            if args.enable_timer:
                runtime_path = os.path.join(args.out_dir, "runtime_log.csv")
                write_runtime_log(runtime_path, runtime_log, cli_string)

            total_elapsed = time.time() - total_start
            imp_rate = (ge / prev_ge) if (prev_ge is not None and prev_ge > EPS) else 1.0

            window_debug_max = window_debug.get("max") if window_debug else None
            window_debug_min = window_debug.get("min") if window_debug else None

            if window_debug_max is not None:
                window_max_msg = (
                    f"win_max: target_seg={window_debug_max['target_seg']}, "
                    f"range=[{window_debug_max['start_seg']}, {window_debug_max['end_seg']}], "
                    f"count={window_debug_max['count']}"
                )
            else:
                window_max_msg = "win_max: N/A"

            if window_debug_min is not None:
                window_min_msg = (
                    f"win_min: target_seg={window_debug_min['target_seg']}, "
                    f"range=[{window_debug_min['start_seg']}, {window_debug_min['end_seg']}], "
                    f"count={window_debug_min['count']}"
                )
            else:
                window_min_msg = "win_min: N/A"

            print(f"[Iter {it:03d}] mode={mode:<5} | Δt={iter_runtime:.3f}s | Total={total_elapsed:.1f}s | "
                f"τ_min={r2(tau_min):>6}, τ_max={r2(tau_max):>6}, τ_mean={r2(tau_mean):>6} | "
                f"total_err={r2(total_err):>6}s | max_cycle_err={r2(max_cycle_err):>6}s@cycle={max_cycle_id} | "
                f"worst_seg=seg_idx={worst_segidx}, d={r2(worst_dist)}s, row_i={i_star} | "
                f"Δerror={error_delta:+.2f} | ratio={error_ratio:.5f} | "
                f"within={r2(within):>6}% | v_cnt={v_cnt:>4} | v_max={r2(v_max):>6}s | "
                f"p95={r2(p95_abs):>6}s | cvar95={r2(cvar95):>6}s")

            print(f"           {window_max_msg} | {window_min_msg}")

            if best_iter_true >= 0:
                print(
                    f"Best retained TRUE iteration so far: "
                    f"{best_iter_true} "
                    f"(v_cnt={best_v_cnt_true}, "
                    f"v_max={r2(best_v_max_true)}, "
                    f"within={r2(best_within_true)}%, "
                    f"ge={r2(best_ge_true)})"
                )
            else:
                print("Best retained TRUE iteration so far: none")

            # 9) Convergence check based only on resolved TRUE anchors.
            if anchor_resolved:
                if v_cnt == 0:
                    check_count += 1

                    print(
                        f"✅ TRUE risk-converged: violations=0 "
                        f"({check_count}/3) at iteration {it}"
                    )

                    if check_count >= 3:
                        print(
                            "🎉 Fully TRUE-converged for three resolved "
                            f"anchors. Stopping at iteration {it}."
                        )

                        F_final = retained_F[:]
                        P_final_ms = retained_P_ms[:]
                        break
                else:
                    check_count = 0

            # HELD/probe intermediate states must not modify prev_ge.
            if anchor_resolved:
                prev_ge = ge

            if (
                probe_unchanged
                and v_cnt > 0
                and not args.adaptive_window
            ):
                print(
                    "⛔ Optimization stalled: the PID produced no parameter "
                    "change while violations remain."
                )
                break

            if stop_after_iteration_reason is not None:
                print(
                    "⛔ Optimization stalled: "
                    f"{stop_after_iteration_reason}"
                )
                break

        # end for

    except KeyboardInterrupt:
        print("\n⛔ Interrupted. Using last available F_final / best_F for output.")

    # ---------- 結尾：總時間 / summary / gcode ----------

    total_runtime = time.time() - total_start
    if args.enable_timer:
        runtime_log.append({
            "iteration": "TOTAL",
            "runtime_sec": total_runtime,
            "totaltime_sec": total_runtime,
            "mode": "TOTAL",
            "tau_min": "",
            "tau_max": "",
            "tau_mean": "",
            "improve": "",
            "within": "",
            "window_cap": "",
            "phase": "",
            "violations": "",
            "v_max": "",
            "p95": "",
            "cvar95": "",
            "based_on_true": "",
            "held_step": "",
            "held_step_scale": "",
            "accepted": "",
            "trust_ratio_used": "",
            "trust_ratio_next": current_trust_ratio,
            "trust_p_ratio_used": "",
            "trust_p_ratio_next": current_trust_P_ratio,
        })
        runtime_path = os.path.join(args.out_dir, "runtime_log.csv")
        write_runtime_log(runtime_path, runtime_log, cli_string)
        print(f"⏳ Total runtime: {total_runtime:.3f} s")

    # 最終 summary（再保險寫一次）
    summary_path = os.path.join(args.out_dir, args.summary)
    write_summary_csv(summary_path, summary_hist, cli_string)

    if (
        args.gcode_in
        and args.gcode_out
        and not true_anchor_valid
    ):
        print(
            "⛔ No successful TRUE FIFO checkpoint is available. "
            "Final and best G-code outputs are skipped."
        )
        return

    # G-code 輸出（只在最後一次）
    if args.gcode_in and args.gcode_out:
        base, ext = os.path.splitext(args.gcode_out)
        ext = ext or ".gcode"
        best_path  = f"{base}_best{ext}"
        final_path = args.gcode_out

        # choose best vectors
        best_feed = (
            best_F_true
            if best_F_true is not None
            else F_final
        )
        best_P = (
            best_P_true
            if best_P_true is not None
            else P_final_ms
        )

        best_iter_out = (
            best_iter_true
            if best_iter_true >= 0
            else "N/A"
        )
        best_ge_out = (
            best_ge_true
            if best_iter_true >= 0
            else float("nan")
        )

        # --- BEST ---
        patched, expected = patch_gcode_by_segidx(
            gcode_template_path=args.gcode_in,
            gcode_out_path=best_path,
            rows=rows,
            F_vec=best_feed,
            P_vec_ms=best_P,
            enable_travel_opt=args.enable_travel_opt,
            s_travel=s_travel
        )

        print(f"✅ Best patch coverage: {patched}/{expected}")

        if expected <= 0 or patched < max(1, int(0.98 * expected)):
            raise RuntimeError(f"Final BEST patch coverage too low: {patched}/{expected}")


        # --- FINAL ---
        patched, expected = patch_gcode_by_segidx(
            gcode_template_path=args.gcode_in,
            gcode_out_path=final_path,
            rows=rows,
            F_vec=F_final,
            P_vec_ms=P_final_ms,
            enable_travel_opt=args.enable_travel_opt,
            s_travel=s_travel
        )

        print(f"✅ Final patch coverage: {patched}/{expected}")

        if expected <= 0 or patched < max(1, int(0.98 * expected)):
            raise RuntimeError(f"Final patch coverage too low: {patched}/{expected}")

    
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
