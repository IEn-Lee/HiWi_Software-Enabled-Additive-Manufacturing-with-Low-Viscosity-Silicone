# -*- coding: utf-8 -*-
"""
RTV_model.py
------------------------------------------------------------
Engineering Model for RTV-2

Goal:
    Map residence time and nozzle temperature to degree of cure alpha.

Assumptions:
    - Isothermal condition inside the nozzle
    - Scenario A: alpha_in = 0
    - Kamal-type cure kinetics
    - Prepared for future upgrade to V1-B

Inputs:
    - Nozzle temperature [°C]
    - Residence time [s]
    - Optional chart creation flag from the command line

Outputs:
    - Alpha at the requested condition
    - Alpha-map CSV file
    - Alpha-map contour plot
    - Alpha(t) curves for selected temperatures

How to Use:
    Run a single alpha query:
    python RTV_model.py --temp_c 30 --residence_time 1122

    Generate the alpha map, CSV file, and plots:
    python RTV_model.py --temp_c 30 --residence_time 1122 --chart_create on

    The script first computes alpha for the requested temperature and
    residence time. If chart creation is enabled, it also builds a full
    temperature-time alpha map and exports the corresponding CSV and plots.
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
DEFAULT_TEMP_C_MAX = 40.0
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
    """Convert temperature from Celsius to Kelvin.

    Args:
        temp_c: Temperature in degrees Celsius.

    Returns:
        Temperature in Kelvin.
    """
    return temp_c + 273.15


def arrhenius_rate(
    A: float, 
    E: float, 
    temp_k: float
) -> float:
    """Compute the Arrhenius reaction rate constant.

    Args:
        A: Pre-exponential factor.
        E: Activation energy in J/mol.
        temp_k: Absolute temperature in Kelvin.

    Returns:
        Arrhenius rate constant at the given temperature.
    """
    return A * math.exp(-E / (R_GAS * temp_k))


def kamal_rhs_isothermal(
    t: float, 
    y: np.ndarray, 
    temp_c: float, 
    params: dict
) -> list[float]:
    """Evaluate the right-hand side of the isothermal Kamal cure model.

    The model is defined as:

        dα/dt = [k1(T) + k2(T) * α^m] * (1 - α)^n

    where k1 and k2 are temperature-dependent Arrhenius rate constants.

    Args:
        t: Current integration time in seconds. This value is required by the ODE solver but is not explicitly used for the isothermal model.
        y: Current state vector. The first element is the degree of cure alpha.
        temp_c: Nozzle temperature in degrees Celsius.
        params: Dictionary containing the Kamal model parameters.

    Returns:
        A single-element list containing dα/dt.
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
    """Solve the degree of cure over time under isothermal conditions.

    The function integrates the Kamal-type cure kinetics from t = 0 to the
    specified residence time at a constant nozzle temperature.

    Args:
        temp_c: Nozzle temperature in degrees Celsius.
        residence_time_s: Residence time in seconds.
        params: Dictionary containing the Kamal model parameters.
        alpha_in: Initial degree of cure at the nozzle inlet.
        n_eval: Number of time points used for evaluating the solution.

    Returns:
        A tuple containing:
            - Time array in seconds.
            - Degree of cure array corresponding to each time point.

    Raises:
        ValueError: If residence_time_s is negative.
        RuntimeError: If the ODE solver fails.
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
    """Compute the final degree of cure for one temperature-time condition.

    Args:
        temp_c: Nozzle temperature in degrees Celsius.
        residence_time_s: Residence time in seconds.
        params: Dictionary containing the Kamal model parameters.
        alpha_in: Initial degree of cure at the nozzle inlet.

    Returns:
        Final degree of cure at the outlet condition.
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
    """Build a two-dimensional alpha map over temperature and residence time.

    Rows correspond to nozzle temperatures, and columns correspond to residence
    times. Each map value is computed by solving the isothermal cure model.

    Args:
        temp_values_c: Array of nozzle temperatures in degrees Celsius.
        time_values_s: Array of residence times in seconds.
        params: Dictionary containing the Kamal model parameters.
        alpha_in: Initial degree of cure at the nozzle inlet.

    Returns:
        Two-dimensional array of degree of cure values.
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
    """Save the computed alpha map as a CSV file.

    Rows represent nozzle temperatures, and columns represent residence times.

    Args:
        temp_values_c: Array of nozzle temperatures in degrees Celsius.
        time_values_s: Array of residence times in seconds.
        alpha_map: Two-dimensional degree of cure map.
        out_csv: Output CSV file path.

    Returns:
        DataFrame containing the saved alpha map.
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
    """Create and save a contour plot of the alpha map.

    Args:
        temp_values_c: Array of nozzle temperatures in degrees Celsius.
        time_values_s: Array of residence times in seconds.
        alpha_map: Two-dimensional degree of cure map.
        out_png: Output image file path.

    Returns:
        None.
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
    """Create and save alpha(t) curves for selected temperatures.

    Args:
        selected_temps_c: List of nozzle temperatures in degrees Celsius.
        max_time_s: Maximum residence time shown in the plot.
        params: Dictionary containing the Kamal model parameters.
        alpha_in: Initial degree of cure at the nozzle inlet.
        out_png: Output image file path.

    Returns:
        None.
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
    """Build an interpolator for fast alpha-map queries.

    The interpolator allows alpha values to be estimated from the precomputed
    temperature-time map without solving the ODE again.

    Args:
        temp_values_c: Array of nozzle temperatures in degrees Celsius.
        time_values_s: Array of residence times in seconds.
        alpha_map: Two-dimensional degree of cure map.

    Returns:
        RegularGridInterpolator object for querying alpha values.
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
    """Query the degree of cure from a precomputed alpha-map interpolator.

    Args:
        interpolator: RegularGridInterpolator built from the alpha map.
        temp_c: Query temperature in degrees Celsius.
        residence_time_s: Query residence time in seconds.

    Returns:
        Interpolated degree of cure clipped to the range [0, 1].
    """
    value = interpolator([[temp_c, residence_time_s]])
    return float(np.clip(value[0], 0.0, 1.0))


# ============================================================
# 5. FUTURE UPGRADE PLACEHOLDER (V1-B)
# ============================================================

def estimate_alpha_in_future_v1b() -> None:
    """Placeholder for estimating inlet alpha in the future V1-B model.

    The future V1-B version may estimate alpha_in from upstream residence time,
    preheating temperature, or measured thermal history before the nozzle.

    Raises:
        NotImplementedError: Always raised because V1-B is not implemented yet.
    """
    raise NotImplementedError("This is reserved for V1-B upgrade.")


# ============================================================
# 6. MAIN DEMO
# ============================================================

def main() -> None:
    """Run the RTV-2 alpha model from command-line arguments.

    The function parses user-defined temperature, residence time, and chart
    creation options. It always performs a single-condition alpha calculation.
    If chart creation is enabled, it also generates the alpha map, CSV file,
    contour plot, alpha(t) curves, and a fast interpolation example.

    Returns:
        None.
    """
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