# -*- coding: utf-8 -*-
"""
rtv2_engineering_model_v1a.py
------------------------------------------------------------
V1-A Engineering Model for RTV-2
Goal:
    Map residence time and nozzle temperature to degree of cure alpha.

Assumptions:
    - Isothermal inside nozzle
    - Scenario A: alpha_in = 0
    - Kamal-type cure kinetics
    - Prepared for future upgrade to V1-B

Inputs:
    - nozzle temperature [°C]
    - residence time [s]

Outputs:
    - alpha at requested condition
    - alpha-map csv
    - alpha-map plot
    - alpha(t) curves for selected temperatures
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
import argparse


# ============================================================
# 1. USER PARAMETERS
# ============================================================

# ---- Universal gas constant ----
R_GAS = 8.314  # J/(mol*K)

# ---- Kinetic parameters ----
# Replace these later with your final jointly-fitted parameters
# Current placeholders are based on your screenshot-style values
MODEL_PARAMS = {
    "A1": 1.00e15,      # s^-1
    "E1": 103.97e3,     # J/mol
    "A2": 1.96e12,      # s^-1
    "E2": 218.55e3,     # J/mol
    "m": 0.713,
    "n": 0.323
}

# ---- Initial condition for V1-A ----
ALPHA_IN_DEFAULT = 0.0

# ---- Numerical safety limits ----
ALPHA_MIN = 1e-9
ALPHA_MAX = 1.0

# ---- Default map ranges ----
DEFAULT_TEMP_C_MIN = 20.0
DEFAULT_TEMP_C_MAX = 50.0
DEFAULT_TEMP_C_STEP = 1.0

DEFAULT_TIME_S_MIN = 0.0
DEFAULT_TIME_S_MAX = 1500.0
DEFAULT_TIME_S_STEP = 2.0

# ---- Output file names ----
OUT_ALPHA_MAP_CSV = "alpha_map_v1a.csv"
OUT_ALPHA_MAP_PNG = "alpha_map_v1a.png"
OUT_ALPHA_CURVES_PNG = "alpha_curves_v1a.png"


# ============================================================
# 2. CORE KINETICS
# ============================================================

def celsius_to_kelvin(temp_c: float) -> float:
    """Convert temperature from °C to K."""
    return temp_c + 273.15


def arrhenius_rate(A: float, E: float, temp_k: float) -> float:
    """Compute Arrhenius rate constant."""
    return A * math.exp(-E / (R_GAS * temp_k))


def kamal_rhs_isothermal(t: float, y: np.ndarray, temp_c: float, params: dict) -> list[float]:
    """
    Isothermal Kamal-type cure kinetics:
        dα/dt = [k1(T) + k2(T) * α^m] * (1 - α)^n
    """
    alpha = float(y[0])

    # clip for numerical stability
    alpha_clipped = min(max(alpha, ALPHA_MIN), ALPHA_MAX)

    temp_k = celsius_to_kelvin(temp_c)
    k1 = arrhenius_rate(params["A1"], params["E1"], temp_k)
    k2 = arrhenius_rate(params["A2"], params["E2"], temp_k)
    m = params["m"]
    n = params["n"]

    dadt = (k1 + k2 * (alpha_clipped ** m)) * ((1.0 - alpha_clipped) ** n)

    return [dadt]


def solve_alpha_isothermal(
    temp_c: float,
    residence_time_s: float,
    params: dict,
    alpha_in: float = ALPHA_IN_DEFAULT,
    n_eval: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve alpha(t) from t=0 to t=residence_time_s for constant temperature.
    Returns:
        t_eval: time array
        alpha: alpha(t)
    """
    if residence_time_s < 0:
        raise ValueError("residence_time_s must be >= 0")

    if residence_time_s == 0:
        return np.array([0.0]), np.array([alpha_in], dtype=float)

    t_eval = np.linspace(0.0, residence_time_s, n_eval)

    sol = solve_ivp(
        fun=lambda t, y: kamal_rhs_isothermal(t, y, temp_c, params),
        t_span=(0.0, residence_time_s),
        y0=[alpha_in],
        t_eval=t_eval,
        method="BDF",
        rtol=1e-7,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    alpha = np.clip(sol.y[0], 0.0, 1.0)
    return sol.t, alpha


def alpha_at_condition(
    temp_c: float,
    residence_time_s: float,
    params: dict,
    alpha_in: float = ALPHA_IN_DEFAULT
) -> float:
    """
    Return alpha at one specific (temperature, residence time) condition.
    """
    _, alpha = solve_alpha_isothermal(
        temp_c=temp_c,
        residence_time_s=residence_time_s,
        params=params,
        alpha_in=alpha_in,
        n_eval=200
    )
    return float(alpha[-1])


# ============================================================
# 3. MAP GENERATION
# ============================================================

def build_alpha_map(
    temp_values_c: np.ndarray,
    time_values_s: np.ndarray,
    params: dict,
    alpha_in: float = ALPHA_IN_DEFAULT
) -> np.ndarray:
    """
    Build 2D alpha map:
        rows -> temperatures
        cols -> residence times
    """
    alpha_map = np.zeros((len(temp_values_c), len(time_values_s)), dtype=float)

    for i, temp_c in enumerate(temp_values_c):
        for j, t_res in enumerate(time_values_s):
            alpha_map[i, j] = alpha_at_condition(
                temp_c=temp_c,
                residence_time_s=t_res,
                params=params,
                alpha_in=alpha_in
            )
        print(f"[INFO] Finished temperature {temp_c:.1f} °C ({i+1}/{len(temp_values_c)})")

    return alpha_map


def save_alpha_map_csv(
    temp_values_c: np.ndarray,
    time_values_s: np.ndarray,
    alpha_map: np.ndarray,
    out_csv: str
) -> pd.DataFrame:
    """
    Save alpha map as CSV.
    Rows are temperatures, columns are residence times.
    """
    df = pd.DataFrame(alpha_map, index=temp_values_c, columns=time_values_s)
    df.index.name = "Temperature_C"
    df.columns.name = "ResidenceTime_s"
    df.to_csv(out_csv, float_format="%.6f")
    return df


def plot_alpha_map(
    temp_values_c: np.ndarray,
    time_values_s: np.ndarray,
    alpha_map: np.ndarray,
    out_png: str
) -> None:
    """
    Plot alpha contour map.
    """
    plt.figure(figsize=(10, 6))

    X, Y = np.meshgrid(time_values_s, temp_values_c)
    contour = plt.contourf(X, Y, alpha_map, levels=30)
    plt.colorbar(contour, label="Degree of cure α")

    # Optional contour lines
    cs = plt.contour(X, Y, alpha_map, levels=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    plt.clabel(cs, inline=True, fontsize=9, fmt="α=%.1f")

    plt.xlabel("Residence time [s]")
    plt.ylabel("Nozzle temperature [°C]")
    plt.title("RTV-2 V1-A Engineering Model: α-map")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_alpha_curves(
    selected_temps_c: list[float],
    max_time_s: float,
    params: dict,
    alpha_in: float,
    out_png: str
) -> None:
    """
    Plot alpha(t) for selected temperatures.
    """
    plt.figure(figsize=(10, 6))

    for temp_c in selected_temps_c:
        t, alpha = solve_alpha_isothermal(
            temp_c=temp_c,
            residence_time_s=max_time_s,
            params=params,
            alpha_in=alpha_in,
            n_eval=500
        )
        plt.plot(t, alpha, label=f"{temp_c:.1f} °C")

    plt.xlabel("Residence time [s]")
    plt.ylabel("Degree of cure α")
    plt.title("RTV-2 V1-A Engineering Model: α(t) curves")
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ============================================================
# 4. INTERPOLATOR FOR FAST QUERY
# ============================================================

def build_alpha_interpolator(
    temp_values_c: np.ndarray,
    time_values_s: np.ndarray,
    alpha_map: np.ndarray
) -> RegularGridInterpolator:
    """
    Build interpolator so that alpha can be queried quickly without solving ODE again.
    """
    return RegularGridInterpolator(
        points=(temp_values_c, time_values_s),
        values=alpha_map,
        bounds_error=False,
        fill_value=None
    )


def query_alpha_from_map(
    interpolator: RegularGridInterpolator,
    temp_c: float,
    residence_time_s: float
) -> float:
    """
    Query alpha from the precomputed map.
    """
    value = interpolator([[temp_c, residence_time_s]])
    return float(np.clip(value[0], 0.0, 1.0))


# ============================================================
# 5. FUTURE UPGRADE PLACEHOLDER (V1-B)
# ============================================================

def estimate_alpha_in_future_v1b():
    """
    Placeholder for future V1-B.

    Future idea:
        alpha_in = alpha(t_pre, T_pre)
    or
        alpha_in = alpha from measured upstream thermal history

    Currently not used in V1-A.
    """
    raise NotImplementedError("This is reserved for V1-B upgrade.")


# ============================================================
# 6. MAIN DEMO
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Map residence time and nozzle temperature to degree of cure alpha.")
    ap.add_argument("--temp_c", type=float, default=30, help="Temperature when printing")
    ap.add_argument("--residence_time", type=float, default=1122, help="Residence time for RTV-2 mixing")
    ap.add_argument("--chart_create", choices= ["on", "off"], default="off", help="Output alpha chart")
    args = ap.parse_args()
    
    # --------------------------------------------------------
    # A) Single query example
    # --------------------------------------------------------
    query_temp_c = args.temp_c
    query_residence_time_s = args.residence_time

    alpha_single = alpha_at_condition(
        temp_c=query_temp_c,
        residence_time_s=query_residence_time_s,
        params=MODEL_PARAMS,
        alpha_in=ALPHA_IN_DEFAULT
    )

    print("=" * 70)
    print("Single condition result")
    print(f"Nozzle temperature : {query_temp_c:.2f} °C")
    print(f"Residence time     : {query_residence_time_s:.2f} s")
    print(f"alpha_out          : {alpha_single:.6f}")
    print("=" * 70)

    # --------------------------------------------------------
    # B) Build alpha map
    # --------------------------------------------------------
    if args.chart_create == "on":
        temp_values_c = np.arange(
            DEFAULT_TEMP_C_MIN,
            DEFAULT_TEMP_C_MAX + DEFAULT_TEMP_C_STEP,
            DEFAULT_TEMP_C_STEP
        )

        time_values_s = np.arange(
            DEFAULT_TIME_S_MIN,
            DEFAULT_TIME_S_MAX + DEFAULT_TIME_S_STEP,
            DEFAULT_TIME_S_STEP
        )

        print("[INFO] Building alpha map ...")
        alpha_map = build_alpha_map(
            temp_values_c=temp_values_c,
            time_values_s=time_values_s,
            params=MODEL_PARAMS,
            alpha_in=ALPHA_IN_DEFAULT
        )

        # --------------------------------------------------------
        # C) Save csv
        # --------------------------------------------------------
        print(f"[INFO] Saving alpha map CSV -> {OUT_ALPHA_MAP_CSV}")
        save_alpha_map_csv(
            temp_values_c=temp_values_c,
            time_values_s=time_values_s,
            alpha_map=alpha_map,
            out_csv=OUT_ALPHA_MAP_CSV
        )

        # --------------------------------------------------------
        # D) Plot alpha map
        # --------------------------------------------------------
        print(f"[INFO] Saving alpha map figure -> {OUT_ALPHA_MAP_PNG}")
        plot_alpha_map(
            temp_values_c=temp_values_c,
            time_values_s=time_values_s,
            alpha_map=alpha_map,
            out_png=OUT_ALPHA_MAP_PNG
        )

        # --------------------------------------------------------
        # E) Plot alpha(t) curves
        # --------------------------------------------------------
        selected_temps_c = [30, 35, 40, 45, 50, 55, 60]
        print(f"[INFO] Saving alpha curves -> {OUT_ALPHA_CURVES_PNG}")
        plot_alpha_curves(
            selected_temps_c=selected_temps_c,
            max_time_s=DEFAULT_TIME_S_MAX,
            params=MODEL_PARAMS,
            alpha_in=ALPHA_IN_DEFAULT,
            out_png=OUT_ALPHA_CURVES_PNG
        )

        # --------------------------------------------------------
        # F) Fast query from precomputed map
        # --------------------------------------------------------
        interpolator = build_alpha_interpolator(
            temp_values_c=temp_values_c,
            time_values_s=time_values_s,
            alpha_map=alpha_map
        )

        alpha_fast = query_alpha_from_map(
            interpolator=interpolator,
            temp_c=47.5,
            residence_time_s=412.0
        )

        print("[INFO] Fast interpolated query from alpha-map")
        print(f"Query at T=47.5 °C, t=412 s -> alpha ≈ {alpha_fast:.6f}")
        print("[INFO] Done.")


if __name__ == "__main__":
    main()