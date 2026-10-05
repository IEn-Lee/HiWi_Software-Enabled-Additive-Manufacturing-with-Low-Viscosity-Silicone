"""
segment_fifo_builder_v6_2_autofill.py
--------------------------------------------------------------------
CLI example:
python segment_fifo_builder_v6_2_autofill.py LiQ5_sample_RTV-Body_Temp32_780s_FT0.1.gcode \
      --out LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_segmented.csv \
      --gcode_out LiQ5_sample_RTV-Body_Temp32_780s_FT0.1_segmented.gcode \
      --vnozzle 574 --dnozzle 0.58 \
      --maxlen 1.0 \
      --d_filament 15.55634919 --emode filament \
      --res_time 0
      --auto_precondition_fill
      --precondition_fill_tol 1e-6
      
vnozzle 723(old)
--d_filament 15.55634919 (F400 Silicon)
--d_filament 1.75 (*79.02)
"""
import math
import csv
import argparse
import os
from collections import deque, defaultdict


# ---------------------------- Utilities ----------------------------

def r5(x)->float:
    """Return the area of a circle given its diameter.

    Args:
        d (float): Circle diameter in millimeters.
    
    Returns:
        float: Area in square millimeters.
    """
    return round(float(x), 5)

def fmt_fixed(x: float, nd: int = 5) -> str:
    """
    Format number in fixed-point decimal, NEVER scientific notation.
    - nd: number of decimal places to keep.
    - Strips trailing zeros and trailing dot.
    """
    if x is None:
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)

    # fixed-point formatting
    s = f"{v:.{nd}f}"
    # strip trailing zeros
    s = s.rstrip("0").rstrip(".")
    # avoid empty string for 0.00000
    return s if s else "0"

def area_circle(d):
    """Compute the XZ-plane distance between two points.

    Args:
        p1 (dict): First point with keys "X" and "Y".
        p2 (dict): Second point with keys "X" and "Y".
    
    Returns:
        float: Euclidean distance in the XY palne.
    """
    return math.pi * (0.5 * d) ** 2


def distance_xy(p1, p2):
    """Compute the XZ-plane distance between two points.

    Args:
        p1 (dict): First point with keys "X" and "Y".
        p2 (dict): Second point with keys "X" and "Y".
    
    Returns:
        float: Euclidean distance in the XY palne.
    """
    return math.hypot(p2["X"] - p1["X"], p2["Y"] - p1["Y"])


def volume_from_E(dE, d_filament=1.75, e_mode="filament"):
    """Convert E-axis increment (dE) into extrusion volume [mm^3].

    Args:
        dE (float): Relative E increment for this move.
        d_filament (float): Filament diameter in millimeters.
        e_mode (str): Interpretation mode, either:
            - "filament": E is filament length.
            - "mm3": E is direct volume.
    
    Returns:
        float: Extrusion volume in cubic millimeters.
    
    Raises:
        ValueError: If an unsupported e_mode is provided.
    """
    if abs(dE) < 1e-12:
        return 0.0
    if e_mode == "filament":
        return area_circle(d_filament) * dE
    if e_mode == "mm3":
        return dE
    raise ValueError(f"Unknown e_mode: {e_mode}")


def E_from_volume(V_mm3, d_filament=1.75, e_mode="filament"):
    """Convert volume [mm^3] back to E-axis increment.

    This is used by --auto_precondition_fill. In filament mode, E is
    filament length [mm], therefore E = V / filament_area. In mm3 mode,
    E is already a volume command.
    """
    V = float(V_mm3 or 0.0)
    if abs(V) < 1e-12:
        return 0.0
    if e_mode == "filament":
        A = area_circle(d_filament)
        if A <= 1e-12:
            raise ValueError("d_filament results in zero filament area.")
        return V / A
    if e_mode == "mm3":
        return V
    raise ValueError(f"Unknown e_mode: {e_mode}")


def split_extrusion_modal(last, move, max_len):
    """Split a single G1 extrusion move into shorter subsegments.

    The function divides an extrusion move with XY motion into pieces,
    ensuring each subsegment does not exceed a specified XY length.
    E increments are also split proportionally.

    Args:
        last (dict): Previous motion state with keys like "X", "Y", "Z", "E".
        move (dict): Current motion target in absolute coordinates.
        max_len (float): Maximum XY length allowed per subsegment (mm).
    
    Returns:
        list[dict]: A list of subsegments, each containing:
            - X1, Y1, Z1: start point
            - X2, Y2, Z2: end point
            - dE: partial extrusion increment
            - F: feedrate 
    """
    d_total = distance_xy(last, move)
    if d_total <= 1e-12:
        return []

    n = max(1, math.ceil(d_total / max_len))
    dx = (move["X"] - last["X"]) / n
    dy = (move["Y"] - last["Y"]) / n
    dz = (move["Z"] - last["Z"]) / n
    dE_total = move["E"] - last["E"]
    dE = dE_total / n

    out = []
    cx, cy, cz = last["X"], last["Y"], last["Z"]
    for _ in range(n):
        out.append({
            "X1": cx,
            "Y1": cy,
            "Z1": cz,
            "X2": cx + dx,
            "Y2": cy + dy,
            "Z2": cz + dz,
            "dE": dE,
            "F": move.get("F", last.get("F", 1000.0)),
        })
        cx += dx
        cy += dy
        cz += dz
    return out

def interpolate_point(p1, p2, frac):
    """Linear interpolation between two 3D point.
    
    Args:
        p1 (dict): Start point with keys "X", "Y", "Z".
        p2 (dict): End point with keys "X", "Y", "Z".
        frac (float): Interpolation fraction in [0, 1].

    Return:
        dict: Interpolated point with keys "X", "Y", "Z".
    """
    return {
        "X": p1["X"] + (p2["X"] - p1["X"]) * frac,
        "Y": p1["Y"] + (p2["Y"] - p1["Y"]) * frac,
        "Z": p1["Z"] + (p2["Z"] - p1["Z"]) * frac,
    }


def parse_g4_ts(parts):
    """Parse dwell time from a G4/G04 command.
    
    Supports both:
        - P<milliseconds>
        - S<seconds> (S has priority if both are present)

    Args:
        parts (list[str]): Tokenized G-code command line.
    
    Return:
        float: Dwell time in seconds.
    """
    t_s = 0.0
    p_ms = None
    s_s = None
    for tok in parts[1:]:
        if len(tok) < 2:
            continue
        c = tok[0].upper()
        try:
            v = float(tok[1:])
        except ValueError:
            continue
        if c == "P":
            p_ms = v
        elif c == "S":
            s_s = v
    if s_s is not None:
        t_s = float(s_s)
    elif p_ms is not None:
        t_s = float(p_ms) / 1000.0
    return max(0.0, t_s)

def has_optimizable(line: str) -> bool:
    """Check whether a G-code line includes the keyword 'optimizable'.

    The check is case-insensitive and applies to the entire raw line, including comments.

    Args:
        line (str): A single line of G-code text.

    Returns:
        bool: True if 'optimizable' appears in the line, otherwise Flase.
    """
    return "optimizable" in line.lower()

def has_precondition(line: str) -> bool:
    return "precondition" in (line or "").lower()

# FIFO helper: numerically stable O(N) mapping from previous cycle t_out to new cycle t_in
def fifo_backfill_times(pieces_in_cycle, prev_out_log, fallback_time):
    """Computer t_in values for extrusion pieces using FIFO volume mapping.

    Each piece in the current cycle pulls volume from the previous cycle's output log.
    Times are weighted by extracted volume to preserve physically accurate residence-time behavior.

    Args:
        pieces_in_cycle (list[dict]): Current cycle pieces, each containing "V".
        prev_out_log (collection.dque or None): A queue of dictionaties representing leftover output parcels from the previous cycle.
        Each entry must contain:
            - "V": remaining volume
            - "t_in": absolute t_out time of that parcel
        fallback_time (float): Time used if there is insufficient previous volume (safety fallback).

    Returns:
        list[float]: t_in values (absolute times) for each piece in order.
    """
    fifo = deque(prev_out_log) if prev_out_log is not None else deque()
    result = []
    for p in pieces_in_cycle:
        need = p["V"]
        acc_t = 0.0
        acc_v = 0.0

        while need > 1e-12 and fifo:
            head = fifo[0]
            take = min(head["V"], need)
            acc_t += head["t_out"] * take
            acc_v += take
            head["V"] -= take
            need -= take
            if head["V"] <= 1e-12:
                fifo.popleft()

        # 若還有缺，用 fallback_time 補（避免無限迴圈）
        if need > 1e-12:
            acc_t += fallback_time * need
            acc_v += need
            need = 0.0

        result.append(acc_t / acc_v if acc_v > 0 else fallback_time)

    return result

# ---------------------------- Core builder ----------------------------

def build_segments_and_segmented_gcode(
    gcode_path,
    csv_out_path,
    gcode_out_path,
    V_nozzle=723.0,
    d_nozzle=0.4,
    max_len=1.0,
    d_filament=15.55634919,
    e_mode="filament",
    res_time=5000.0,  # kept for compatibility; cycles>=2 FIFO mapping uses t_s only
    auto_precondition_fill=False,
    precondition_target_volume=None,
    precondition_fill_tol=1e-6,
):
    """Generate segments extrusion pieces and FIFO timing from relative-E G-code.

    This function implements a finite-volume FIFO nozzle model on top of a relative-E (M83) G-code file.
    It splits extrusion moves into geometric subsegments, slices them by volume boundaries (V_nozzle), assigns cycle indices, and computes timing parameters for each extrusion piece.

    The results are exported as:
        * A segmented G-code file containing fine-grained G1 SEG moves.
        * A CSV table with per-segment geometry, volume, timing, and notes.
    
    Phases:
        A. Parse G-code into a timeline
            - Classify lines as raw, travel, pre-infill, extrusion groups, or dwell.
            - Handle relative E semantics and detect first XY+E extrusion.
        B. Flatten extrusions & boundary slicing
            - Flatten extrusion groups into a list.
            - Slice by V_nozzle so each piece falls into a single FIFO cycles.
        C. Tag cycles & expand timeline
            - Assign cycle and cycle-local indices to piece.
            - Reinsert pieces into the global timeline.
        D. Compute durations for non-extrusion motions
            - Estimate t_s for travel and other motion commands.
            - Accumulate non-extrusion delays between extrusion pieces.
        E. Compute t_in / t_out per cycle (FIFO backfill)
            - Cycle 1: t_in distributed over fill time, t_out includes prefill delay.
            - Cycle >= 2: t_out based on cumulative duration; t_in backfilled from previous cycle's t_out using volume-weighted FIFO.
        F. Emit segmented G-code and CSV
            - Write G-code with segmented extrusions and annotation.
            - Write CSV rows with geometry, timing, and note fields.
        G. Finish CSV output
            - Close files and print basic statistics. 
        
    Special handling:
        * Pre-infill: G1 with positive E but no XY motion before the first
          XY+E move is tagged as "pre_infill" and contributes to prefill_delay.
        * Dwell (G4/G04): Parsed with S (seconds) or P (milliseconds) and
          added to the timing model.
        * Lines containing the word "optimizable" (case-insensitive) have
          their CSV note field annotated with "optimizable".

    Args:
        gcode_path (str): Path to the input G-code file (relative E, M83).
        csv_out_path (str): Path to the output CSV file for segment data.
        gcode_out_path (str): Path to the output segmented G-code file.
        V_nozzle (float, optional): Mixing chamber volume per FIFO cycle in
            cubic millimeters. Controls cycle boundaries. Defaults to 723.0.
        d_nozzle (float, optional): Nozzle diameter in millimeters. Used only
            for logging metadata. Defaults to 0.4.
        max_len (float, optional): Maximum XY length per geometric extrusion
            subsegment in millimeters. Defaults to 1.0.
        d_filament (float, optional): Filament diameter in millimeters when
            `e_mode="filament"`. Defaults to 15.55634919.
        e_mode (str, optional): E-axis interpretation mode:
            - "filament": E values represent filament length [mm].
            - "mm3": E values represent direct volume [mm³].
            Defaults to "filament".
        res_time (float, optional): Legacy residence-time parameter. Kept for
            backward compatibility; cycles >= 2 use t_s-based FIFO mapping.
            Defaults to 5000.0.

    Returns:
        None: The function writes the segmented G-code and CSV to disk and
        prints a short summary to stdout.

    Raises:
        RuntimeError: If no extrusion pieces remain after boundary slicing.
        ValueError: If an unsupported `e_mode` is provided.
    """
    # -------- Phase A: Parse G-code into a timeline --------
    timeline = []
    subgroups = []            # extrusion groups (XY+E segments)
    seen_first_xy_extrusion = False
    armed_wall_outer = False   # seen ;TYPE:WALL-OUTER
    started_cycles = False     # first G1 with E after WALL-OUTER -> cycle/seg start


    with open(gcode_path, "r", encoding="utf-8", errors="ignore") as fin:
        last = {"X": 0.0, "Y": 0.0, "Z": 0.0, "E": 0.0, "F": 1000.0}

        for raw in fin:
            line = raw.rstrip("\n")
            stripped = line.strip()
            opt = has_optimizable(line)
            precond = has_precondition(line)
            # arm start when WALL-OUTER marker appears (do not modify/comment it)
            if ";type:wall-outer" in line.lower():
                armed_wall_outer = True


            if stripped == "" or stripped.startswith(";"):
                timeline.append({"type": "raw", "line": line, "optimizable": opt})
                continue

            parts = stripped.split()
            cmd = parts[0].upper()

            # Dwell
            if cmd in ("G4", "G04"):
                t_s = parse_g4_ts(parts)
                timeline.append({
                    "type": "g4",
                    "parts": parts,
                    "t_s": t_s,
                    "line": line,
                    "optimizable": opt,
                })
                continue

            # Other non-motion
            if cmd not in ("G0", "G1"):
                timeline.append({"type": "raw", "line": line, "optimizable": opt})
                continue

            # Modal move: treat E as RELATIVE
            move = last.copy()
            dE = 0.0
            has_E = False

            for p in parts[1:]:
                if len(p) < 2:
                    continue
                c = p[0].upper()
                try:
                    v = float(p[1:])
                except ValueError:
                    continue

                if c == "E":
                    has_E = True
                    dE = v
                    move["E"] = last["E"] + dE  # store absolute
                else:
                    move[c] = v

            if not has_E:
                move["E"] = last["E"]

            d_xy = distance_xy(last, move)
            F = move.get("F", last.get("F", 1000.0))

            is_xy_extr = (cmd == "G1" and dE > 1e-12 and d_xy > 1e-12)
            is_e_only  = (cmd == "G1" and dE > 1e-12 and d_xy <= 1e-12)

            # ---------------------------------------------------------
            # START RULE:
            #   cycle/seg start = first G1 with E AFTER ';TYPE:WALL-OUTER'
            # BEFORE started_cycles:
            #   any extrusion (XY+E or E-only) is treated as pre_infill (cycle 0)
            # ---------------------------------------------------------

            is_any_extr = (cmd == "G1" and dE > 1e-12)

            # New start rule:
            # First G1 with E and comment containing "precondition" starts the FIFO timer/cycle region.
            if (not started_cycles) and precond and is_any_extr:
                started_cycles = True

            # Backward-compatible fallback:
            # If no precondition is used, keep the old WALL-OUTER behavior.
            if (not started_cycles) and armed_wall_outer and is_any_extr:
                started_cycles = True

            # BEFORE started_cycles: everything with E is pre_infill
            if is_any_extr and (not started_cycles):
                t_s = 0.0
                if d_xy > 1e-12:
                    t_s = 60.0 * d_xy / F
                else:
                    t_s = 60.0 * abs(dE) / F if F > 1e-12 else 0.0

                V = volume_from_E(dE, d_filament, e_mode)
                timeline.append({
                    "type": "pre_infill",
                    "cmd": cmd,
                    "F": F,
                    "t_s": t_s,
                    "V": V,
                    "dE": dE,
                    "line": line,              # keep original line untouched
                    "optimizable": opt,
                })
                last = move.copy()
                continue

            # ---------------------------------------------------------
            # ✅ NEW RULE:
            #   After started_cycles, but BEFORE first XY+E extrusion,
            #   treat E-only G1 (no X/Y/Z) as pre_infill (cycle 0).
            #   This covers: G1 Fxxx Exxx ; optimizable  (no XYZ)
            # ---------------------------------------------------------
            if is_e_only and (not seen_first_xy_extrusion) and (not precond):
                t_s = 60.0 * abs(dE) / F if F > 1e-12 else 0.0
                V = volume_from_E(dE, d_filament, e_mode)
                timeline.append({
                    "type": "pre_infill",
                    "cmd": cmd,
                    "F": F,
                    "t_s": t_s,
                    "V": V,
                    "dE": dE,
                    "line": line,              # keep original line untouched
                    "optimizable": opt,
                })
                last = move.copy()
                continue

            # AFTER started_cycles: allow segmentation (extr_group)
            if is_xy_extr:
                seen_first_xy_extrusion = True
                subs = split_extrusion_modal(last, move, max_len)
                sublist = []
                for s in subs:
                    p1 = {"X": s["X1"], "Y": s["Y1"], "Z": s["Z1"]}
                    p2 = {"X": s["X2"], "Y": s["Y2"], "Z": s["Z2"]}
                    d_len = distance_xy(p1, p2)
                    if d_len > 1e-12:
                        t_s = 60.0 * d_len / s["F"]
                    elif abs(s["dE"]) > 1e-12:
                        t_s = 60.0 * abs(s["dE"]) / s["F"]
                    else:
                        t_s = 0.0
                    V = volume_from_E(s["dE"], d_filament, e_mode)
                    sublist.append({
                        "p1": p1,
                        "p2": p2,
                        "F": s["F"],
                        "d_len": d_len,
                        "t_s": t_s,
                        "V": V,
                        "dE": s["dE"],
                        "precondition": precond,
                    })
                gi = len(timeline)
                timeline.append({
                    "type": "extr_group",
                    "subs": sublist,
                    "optimizable": opt,
                })
                subgroups.append({"ti": gi, "subs": sublist})
                last = move.copy()
                continue

            # AFTER started_cycles: E-only extrusion participates as degenerate extr_group
            if is_e_only:
                t_s = 60.0 * abs(dE) / F if F > 1e-12 else 0.0
                V = volume_from_E(dE, d_filament, e_mode)
                p1 = {"X": last["X"], "Y": last["Y"], "Z": last["Z"]}
                p2 = {"X": last["X"], "Y": last["Y"], "Z": last["Z"]}
                sublist = [{
                    "p1": p1,
                    "p2": p2,
                    "F": F,
                    "d_len": 0.0,
                    "t_s": t_s,
                    "V": V,
                    "dE": dE,
                    "precondition": precond,
                }]
                gi = len(timeline)
                timeline.append({
                    "type": "extr_group",
                    "subs": sublist,
                    "optimizable": opt,
                })
                subgroups.append({"ti": gi, "subs": sublist})
                last = move.copy()
                continue


            # Other moves
            timeline.append({
                "type": "move",
                "cmd": cmd,
                "last": last.copy(),
                "move": move.copy(),
                "line": line,
                "optimizable": opt,
            })
            last = move.copy()

    # -------- Phase B: Flatten extrusions & boundary slicing --------
    flat = []
    for sg in subgroups:
        group_item = timeline[sg["ti"]]
        for sub in sg["subs"]:
            flat.append({
                "group_ti": sg["ti"],
                "optimizable": group_item.get("optimizable", False),
                **sub,
            })

    # Optional: automatically top-up the precondition volume so that the first
    # real extrusion starts at the next FIFO cycle. This only adds volume if
    # the current precondition volume is below the target. If the precondition
    # volume already exceeds the target, nothing is changed.
    if auto_precondition_fill:
        target_V = float(precondition_target_volume) if precondition_target_volume is not None else float(V_nozzle)
        tol = max(0.0, float(precondition_fill_tol))

        first_real_idx = None
        for ii, seg in enumerate(flat):
            if not seg.get("precondition", False):
                first_real_idx = ii
                break

        precond_end = first_real_idx if first_real_idx is not None else len(flat)
        precond_indices = [ii for ii in range(precond_end) if flat[ii].get("precondition", False)]
        precond_V = sum(float(flat[ii].get("V", 0.0) or 0.0) for ii in precond_indices)

        if precond_indices and precond_V < target_V - tol:
            deficit_V = target_V - precond_V
            last_pre = flat[precond_indices[-1]]
            F_fill = float(last_pre.get("F", 0.0) or 0.0)
            if F_fill <= 1e-12:
                raise RuntimeError("Cannot auto-fill precondition volume because the last precondition feedrate is invalid.")

            dE_fill = E_from_volume(deficit_V, d_filament=d_filament, e_mode=e_mode)
            t_s_fill = 60.0 * abs(dE_fill) / F_fill
            p = dict(last_pre.get("p2", last_pre.get("p1", {"X": 0.0, "Y": 0.0, "Z": 0.0})))

            fill_seg = {
                "group_ti": last_pre["group_ti"],
                "optimizable": True,
                "p1": p.copy(),
                "p2": p.copy(),
                "F": F_fill,
                "d_len": 0.0,
                "t_s": t_s_fill,
                "V": deficit_V,
                "dE": dE_fill,
                "precondition": True,
                "auto_precondition_fill": True,
            }
            insert_at = precond_end
            flat.insert(insert_at, fill_seg)
            print(
                f"🧩 Auto precondition fill inserted: "
                f"target_V={target_V:.8f} mm^3, "
                f"existing_V={precond_V:.8f} mm^3, "
                f"added_V={deficit_V:.8f} mm^3, "
                f"added_E={dE_fill:.8f}, F={F_fill:.3f}"
            )
        elif precond_indices:
            print(
                f"🧩 Auto precondition fill: no fill needed "
                f"(existing_V={precond_V:.8f} mm^3, target_V={target_V:.8f} mm^3)."
            )
        else:
            print("⚠️ Auto precondition fill enabled, but no precondition segment was found before the first real extrusion.")

    S = 0.0
    sliced = []
    for seg in flat:
        V_total = seg["V"]
        if V_total <= 1e-12:
            continue
        p1 = seg["p1"]
        p2 = seg["p2"]
        remain_V = V_total
        t_s_total = seg["t_s"]
        dE_total = seg["dE"]

        while remain_V > 1e-12:
            k = math.floor(S / V_nozzle) + 1
            boundary = k * V_nozzle
            room = boundary - S

            if remain_V <= room + 1e-12:
                frac = remain_V / V_total
                cut = interpolate_point(p1, p2, frac)
                sliced.append({
                    "group_ti": seg["group_ti"],
                    "p1": p1,
                    "p2": cut,
                    "F": seg["F"],
                    "d_len": distance_xy(p1, cut),
                    "t_s": t_s_total * frac,
                    "V": remain_V,
                    "dE": dE_total * frac,
                    "precondition": seg.get("precondition", False),
                    "auto_precondition_fill": seg.get("auto_precondition_fill", False),
                })
                S += remain_V
                remain_V = 0.0
            else:
                frac = room / V_total
                cut = interpolate_point(p1, p2, frac)
                take_t_s = t_s_total * frac
                take_dE = dE_total * frac
                sliced.append({
                    "group_ti": seg["group_ti"],
                    "p1": p1,
                    "p2": cut,
                    "F": seg["F"],
                    "d_len": distance_xy(p1, cut),
                    "t_s": take_t_s,
                    "V": room,
                    "dE": take_dE,
                    "precondition": seg.get("precondition", False),
                    "auto_precondition_fill": seg.get("auto_precondition_fill", False),
                })
                p1 = cut
                V_total -= room
                t_s_total -= take_t_s
                dE_total -= take_dE
                remain_V = V_total
                S = boundary

    if not sliced:
        raise RuntimeError("No extrusion pieces after boundary slicing.")

    # -------- Phase C: Tag cycles & expand timeline --------
    pieces = []
    S = 0.0
    cycle_idx = 1
    local_idx = 0
    for s in sliced:
        prev_S = S
        S += s["V"]
        if math.isclose(prev_S % V_nozzle, 0.0, abs_tol=1e-9) and prev_S > 0:
            cycle_idx += 1
            local_idx = 0
        local_idx += 1
        rec = dict(s)
        rec["cycle"] = cycle_idx
        rec["cycle_local"] = local_idx
        pieces.append(rec)

    group_map = defaultdict(list)
    for p in pieces:
        group_map[p["group_ti"]].append(p)

    expanded = []
    for i, item in enumerate(timeline):
        if item["type"] == "extr_group":
            gpieces = group_map.get(i, [])
            opt = item.get("optimizable", False)
            for p in gpieces:
                expanded.append({
                    "type": "piece",
                    "optimizable": p.get("optimizable", opt),
                    **p,
                    "precondition": p.get("precondition", False),
                    "auto_precondition_fill": p.get("auto_precondition_fill", False),
                })
        else:
            expanded.append(item)

    # -------- Phase D: time origin starts at first precondition piece --------
    nonextr_delay_up_to = []
    delay = 0.0
    time_started = False
    inj_end_time = 0.0

    for it in expanded:
        # First precondition piece defines t = 0.
        if (
            (not time_started)
            and it.get("type") == "piece"
            and it.get("precondition", False)
        ):
            time_started = True
            delay = 0.0
            inj_end_time = 0.0

        if it["type"] == "piece":
            nonextr_delay_up_to.append(delay if time_started else 0.0)
        else:
            if time_started:
                delay += it.get("t_s", 0.0)

    # -------- Phase E: Compute t_in / t_out per cycle (穩定 FIFO) --------
    max_cycle = pieces[-1]["cycle"]
    cycles = [[p for p in pieces if p["cycle"] == c]
              for c in range(1, max_cycle + 1)]

    def cum_durs(plist, inclusive=False):
        out, acc = [], 0.0
        for p in plist:
            if inclusive:
                acc += p["t_s"]
                out.append(acc)
            else:
                out.append(acc)
                acc += p["t_s"]
        return out

    # map (cycle,local) -> global ordinal index of piece
    global_index_map = {
        (p["cycle"], p["cycle_local"]): i
        for i, p in enumerate(pieces)
    }

    final_timing = {}

    last_cycle_last_tout_abs = 0.0
    prev_out_log = None  # deque of {"V", "t_in"} using ABSOLUTE t_out as source

    # helper: 對應每個 piece ordinal -> 該 piece 前的非 piece 累積時間
    nonextr_by_ord = nonextr_delay_up_to

    for cidx, cyc in enumerate(cycles, start=1):
        N = len(cyc)
        if N == 0:
            continue

        ord_indices = [
            global_index_map[(p["cycle"], p["cycle_local"])]
            for p in cyc
        ]

        if cidx == 1:
            offs_incl = cum_durs(cyc, inclusive=True)
            first_delay = nonextr_by_ord[ord_indices[0]]

            abs_t_out = []
            for i, off in enumerate(offs_incl):
                extra_delay = nonextr_by_ord[ord_indices[i]] - first_delay
                if extra_delay < 0:
                    extra_delay = 0.0
                abs_t_out.append(first_delay + off + extra_delay)

            # ✅ 你的指令：用 optimizable 段落頭尾時間 / cycle1 seg數量 -> 估每個 seg 的 t_in（逐步遞增）
            T_inj = inj_end_time  # 頭端是 0，所以直接用尾端時間長度
            if T_inj < 0:
                T_inj = 0.0

            step = (T_inj / N) if N > 0 else 0.0
            base_t_in = [i * step for i in range(N)]   # i=0..N-1  -> 0, step, 2step...


        else:
            # 擠出內部累積時間 (僅此 cycle 的 pieces)
            offs_incl = cum_durs(cyc, inclusive=True)

            # 本 cycle 第一個 piece 前的累積 non-extrusion 時間
            first_delay = nonextr_by_ord[ord_indices[0]]

            # 上一個 cycle 最後一個 piece 前的累積 non-extrusion 時間
            # 這樣 first_delay - prev_last_delay 就是「cycle 交界處新增的 non-extrusion」
            prev_cycle = cycles[cidx - 2]  # cidx 從 1 開始，所以前一 cycle 是 cidx-2
            prev_last_piece = prev_cycle[-1]
            prev_last_ord = global_index_map[(prev_last_piece["cycle"], prev_last_piece["cycle_local"])]
            prev_last_delay = nonextr_by_ord[prev_last_ord]

            inter_cycle_delay = first_delay - prev_last_delay
            if inter_cycle_delay < 0:
                inter_cycle_delay = 0.0  # 保底，避免負值造成時間倒退

            abs_t_out = []
            for i, off in enumerate(offs_incl):
                # cycle 內部 piece 之間的非擠出差值（保留你原本的設計）
                extra_delay = nonextr_by_ord[ord_indices[i]] - first_delay
                # ✅把「cycle 交界 delay」加回去
                abs_t_out.append(last_cycle_last_tout_abs + inter_cycle_delay + off + extra_delay)


            # 從上一循環的 t_out 分配 t_in (FIFO)
            base_t_in = fifo_backfill_times(
                cyc,
                prev_out_log,
                last_cycle_last_tout_abs,
            )

        # 寫入 timing
        for i, p in enumerate(cyc):
            t_in = base_t_in[i]
            t_out = abs_t_out[i]
            final_timing[(p["cycle"], p["cycle_local"])] = {
                "t_in": t_in,
                "t_out": t_out,
                "tau_s": t_out - t_in,
            }

        last_cycle_last_tout_abs = abs_t_out[-1]

        # 下一個 cycle 的 FIFO 來源：當前 cycle 的輸出
        prev_out_log = deque(
            {"V": cyc[i]["V"], "t_out": abs_t_out[i]}
            for i in range(N)
        )

    # -------- Phase F: Emit segmented G-code and CSV --------
    header = [
        "seg_idx", "cmd", "cycle", "cycle_local",
        "d_mm", "F_mm_per_min", "V_mm3",
        "t_s", "t_in", "t_out", "tau_s", "note",
    ]

    rows = []
    seg_idx = 0

    with open(gcode_out_path, "w", encoding="utf-8", newline="") as fout:
        def emit_gcode(line: str) -> int:
            """
            Write ONE gcode line and append seg_idx.
            - If line contains 'optimizable' (case-insensitive): write EXACTLY as-is (no extra seg_idx comment)
            - Else:
                * If comment-only line: append " seg_idx=N"
                * If already has comment: append " ; seg_idx=N"
                * Else: append " ; seg_idx=N"
            """
            nonlocal seg_idx
            seg_idx += 1

            # ✅ define s FIRST (fixes UnboundLocalError)
            s = (line or "").rstrip("\n")
            stripped = s.strip()

            # ✅ keep optimizable lines untouched
            # ✅ optimizable lines ALSO get seg_idx (keep content, just append seg_idx once)
            if "optimizable" in s.lower():
                if stripped.startswith(";"):
                    # comment-only line: append seg_idx directly
                    fout.write(f"{s} seg_idx={seg_idx}\n")
                else:
                    # code line with comment: append seg_idx without touching existing comment text
                    code, sep, cmt = s.partition(";")
                    if sep:
                        cmt2 = cmt.strip()
                        if cmt2:
                            fout.write(f"{code.rstrip()} ; {cmt2} ; seg_idx={seg_idx}\n")
                        else:
                            fout.write(f"{code.rstrip()} ; seg_idx={seg_idx}\n")
                    else:
                        # extremely rare: no ';' but contains 'optimizable' in code part
                        fout.write(f"{s} ; seg_idx={seg_idx}\n")
                return seg_idx


            if stripped.startswith(";"):
                fout.write(f"{s} seg_idx={seg_idx}\n")
            else:
                code, sep, cmt = s.partition(";")
                if sep:  # already has comment
                    cmt2 = cmt.strip()
                    if cmt2:
                        fout.write(f"{code.rstrip()} ; {cmt2} ; seg_idx={seg_idx}\n")
                    else:
                        fout.write(f"{code.rstrip()} ; seg_idx={seg_idx}\n")
                else:
                    fout.write(f"{s} ; seg_idx={seg_idx}\n")

            return seg_idx


        def add_csv_row(idx: int, *, cmd="", cycle="", cycle_local="", d_mm=0.0, F_mm_per_min=0.0,
                        V_mm3=0.0, t_s=0.0, t_in="", t_out="", tau_s="", note="") -> None:
            rows.append({
                "seg_idx": idx,
                "cmd": cmd,
                "cycle": cycle,
                "cycle_local": cycle_local,
                "d_mm": d_mm if isinstance(d_mm, str) else fmt_fixed(d_mm, 5),
                "F_mm_per_min": F_mm_per_min if isinstance(F_mm_per_min, str) else fmt_fixed(F_mm_per_min, 3),
                "V_mm3": V_mm3 if isinstance(V_mm3, str) else fmt_fixed(V_mm3, 5),
                "t_s": t_s if isinstance(t_s, str) else fmt_fixed(t_s, 5),
                "t_in": t_in,
                "t_out": t_out,
                "tau_s": tau_s,
                "note": note,
            })

        # --- ✅ 先把 header 也用 seg_idx 寫出，並同步寫入 CSV ---
        idx = emit_gcode(f"; Segmented G-code generated from: {os.path.basename(gcode_path)}")
        add_csv_row(idx, note="header/meta")

        idx = emit_gcode("; finite-volume FIFO cycles with boundary backfill (v6.2, relative E)")
        add_csv_row(idx, note="header/meta")

        idx = emit_gcode(f"; V_nozzle={V_nozzle}, d_nozzle={d_nozzle}, max_len={max_len}")
        add_csv_row(idx, note="header/meta")

        idx = emit_gcode(f"; d_filament={d_filament}, e_mode={e_mode}, res_time={res_time}")
        add_csv_row(idx, note="header/meta")

        idx = emit_gcode("M83 ; relative extrusion mode assumed")
        add_csv_row(idx, cmd="M83", note="mode_setup")

        # --- 原本 loop：每一行都 emit + CSV ---
        seen_first_piece = False
        
        for it in expanded:
            typ = it["type"]
            opt = it.get("optimizable", False)
            
            def note_for(base, precond=False):
                tags = []

                if base:
                    tags.append(base)

                if precond and "precondition" not in tags:
                    tags.append("precondition")

                if opt and "optimizable" not in tags:
                    tags.append("optimizable")

                return ",".join(tags)

            if typ == "raw":
                idx = emit_gcode(it.get("line", ""))
                add_csv_row(idx,cycle=0 if not seen_first_piece else "",cycle_local=0 if not seen_first_piece else "",t_s=it.get("t_s", 0.0),note=note_for("comment/raw"),)

            elif typ == "move":
                idx = emit_gcode(it.get("line", "").strip() + " ; TRAVEL")
                last_mv = it["last"]
                cur = it["move"]
                d_xy = distance_xy(last_mv, cur)
                F = cur.get("F", last_mv.get("F", 1000.0))
                dE = cur.get("E", last_mv.get("E", 0.0)) - last_mv.get("E", 0.0)
                V = volume_from_E(dE, d_filament, e_mode) if dE > 1e-12 else 0.0
                base_note = "travel_extrusion" if V > 0 else "travel"
                add_csv_row(
                    idx,
                    cmd=it["cmd"],
                    cycle=0 if not seen_first_piece else "",
                    cycle_local=0 if not seen_first_piece else "",
                    d_mm=d_xy,
                    F_mm_per_min=F,
                    V_mm3=V,
                    t_s=it.get("t_s", 0.0),
                    note=note_for(base_note),
                )

            elif typ == "g4":
                idx = emit_gcode(it["line"] + " ; DWELL")
                add_csv_row(idx,cmd="G4",cycle=0 if not seen_first_piece else "",cycle_local=0 if not seen_first_piece else "",t_s=it.get("t_s", 0.0),note=note_for("dwell"),)

            elif typ == "piece":
                seen_first_piece = True
                dE = it["dE"]
                tags = ["SEG"]

                if it.get("precondition", False):
                    tags.append("precondition")

                if opt:
                    tags.append("optimizable")

                if it.get("auto_precondition_fill", False):
                    tags.append("auto_fill")

                comment = " ".join(tags)

                gline = (
                    f"G1 X{fmt_fixed(it['p2']['X'], 5)} "
                    f"Y{fmt_fixed(it['p2']['Y'], 5)} "
                    f"Z{fmt_fixed(it['p2']['Z'], 5)} "
                    f"E{fmt_fixed(dE, 6)} "
                    f"F{round(it['F'], 3)} ; {comment}"
                )
                idx = emit_gcode(gline)

                key = (it["cycle"], it["cycle_local"])
                tinfo = final_timing.get(key, {"t_in": None, "t_out": None, "tau_s": None})

                base_note = "precondition" if it.get("precondition", False) else "extrusion"

                if it.get("precondition", False):
                    # Reporting convention for precondition rows:
                    # material is entering the FIFO volume but is not considered
                    # an output product segment yet. Internally, the cumulative
                    # cycle-1 output-time proxy is still used for FIFO backfill;
                    # in the CSV it is reported as cumulative t_in, while t_out
                    # is left blank and tau_s accumulates from the first
                    # precondition piece.
                    t_pre = tinfo.get("t_out", None)
                    csv_t_in = fmt_fixed(t_pre) if t_pre is not None else ""
                    csv_t_out = ""
                    csv_tau = fmt_fixed(t_pre) if t_pre is not None else ""
                else:
                    csv_t_in = fmt_fixed(tinfo["t_in"]) if tinfo["t_in"] is not None else ""
                    csv_t_out = fmt_fixed(tinfo["t_out"]) if tinfo["t_out"] is not None else ""
                    csv_tau = fmt_fixed(tinfo["tau_s"]) if tinfo["tau_s"] is not None else ""

                add_csv_row(
                    idx,
                    cmd="G1",
                    cycle=it["cycle"],
                    cycle_local=it["cycle_local"],
                    d_mm=it["d_len"],
                    F_mm_per_min=it["F"],
                    V_mm3=it["V"],
                    t_s=it["t_s"],
                    t_in=csv_t_in,
                    t_out=csv_t_out,
                    tau_s=csv_tau,
                    note=note_for(
                        base_note,
                        precond=it.get("precondition", False),
                    ) + (",auto_fill" if it.get("auto_precondition_fill", False) else ""),
                )
                
            elif typ == "pre_infill":
                idx = emit_gcode(it["line"])
                add_csv_row(
                    idx,
                    cmd="G1",
                    cycle=0,                 # 你要的：pre_infill cycle=0
                    cycle_local="",
                    d_mm=0.0,
                    F_mm_per_min=it["F"],
                    V_mm3=it["V"],
                    t_s=it["t_s"],
                    note=note_for("pre_infill"),
    )


            else:
                idx = emit_gcode(it.get("line", "").strip() if isinstance(it, dict) else "")
                add_csv_row(idx, note=note_for("unknown"))

# ✅ 這時候：G-code 寫了幾行，rows 就一定有幾列



    # -------- Phase G: Write CSV --------
    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    gcode_total = seg_idx
    csv_total = len(rows)
    print(f"✅ Done: {os.path.basename(gcode_out_path)} lines(seg_idx) = {gcode_total}")
    print(f"✅ Done: {os.path.basename(csv_out_path)} rows(seg_idx)  = {csv_total}")
    
    if gcode_total != csv_total:
        print(f"⚠️ WARNING: gcode seg_idx({gcode_total}) != csv rows({csv_total})")

    # print(f"✅ Segments saved to: {csv_out_path} | Total rows: {len(rows)}")
    # print(f"✅ Segmented G-code saved to: {os.path.abspath(gcode_out_path)}")


# ---------------------------- CLI ----------------------------

def main():
    """Entry point for the CLI segmentation tool.

    This function parses command-line arguments and runs the G-code
    segmentation pipeline. It exposes the main modeling parameters via
    flags and writes both CSV and segmented G-code outputs.

    Command-line arguments:
        gcode (str): Input G-code file (M83 relative E).
        --out (str): Output CSV filename. Defaults to "segments.csv".
        --gcode_out (str): Output segmented G-code filename.
            Defaults to "segmented.gcode".
        --vnozzle (float): Mixing chamber volume V_nozzle in mm³.
            Defaults to 200.0.
        --dnozzle (float): Nozzle diameter in mm (logged only).
            Defaults to 0.5.
        --maxlen (float): Max XY length per extrusion subsegment (mm).
            Defaults to 1.0.
        --d_filament (float): Filament diameter in mm if e_mode="filament".
            Defaults to 1.75.
        --emode (str): E-axis unit mode, "filament" or "mm3".
            Defaults to "filament".
        --res_time (float): Legacy residence-time parameter, kept for
            compatibility with older versions. Defaults to 300.0.

    Returns:
        None  
    """
    ap = argparse.ArgumentParser(
        description="Finite-volume FIFO (cycle) segment builder with boundary backfill (v6.2, relative E)."
    )
    ap.add_argument("gcode", help="Input G-code (M83 relative E)")
    ap.add_argument("--out", default="segments.csv", help="Output CSV filename")
    ap.add_argument("--gcode_out", default="segmented.gcode", help="Output segmented G-code filename")
    ap.add_argument("--vnozzle", type=float, default=200.0, help="Mixing chamber volume V_nozzle [mm^3]")
    ap.add_argument("--dnozzle", type=float, default=0.5, help="Nozzle diameter [mm] (logged only)")
    ap.add_argument("--maxlen", type=float, default=1.0, help="Max XY length per extrusion subsegment [mm]")
    ap.add_argument("--d_filament", type=float, default=1.75, help="Filament diameter [mm] if e_mode='filament'")
    ap.add_argument("--emode", choices=["filament", "mm3"], default="filament", help="E-axis unit mode")
    ap.add_argument(
        "--res_time",
        type=float,
        default=300.0,
        help="Legacy parameter; kept for compatibility (used indirectly for cycles >=2).",
    )
    ap.add_argument(
        "--auto_precondition_fill",
        action="store_true",
        help=(
            "If enabled, automatically inserts one E-only '; precondition optimizable' "
            "top-up segment when the precondition volume before the first real extrusion "
            "is below the target volume. The top-up uses the same feedrate as the last "
            "precondition segment. If the volume already exceeds the target, no action is taken."
        ),
    )
    ap.add_argument(
        "--precondition_target_volume",
        type=float,
        default=None,
        help="Target precondition volume [mm^3]. Defaults to --vnozzle when omitted.",
    )
    ap.add_argument(
        "--precondition_fill_tol",
        type=float,
        default=1e-6,
        help="Tolerance [mm^3] for deciding whether the precondition volume is already sufficient.",
    )
    args = ap.parse_args()


    build_segments_and_segmented_gcode(
        gcode_path=args.gcode,
        csv_out_path=args.out,
        gcode_out_path=args.gcode_out,
        V_nozzle=args.vnozzle,
        d_nozzle=args.dnozzle,
        max_len=args.maxlen,
        d_filament=args.d_filament,
        e_mode=args.emode,
        res_time=args.res_time,
        auto_precondition_fill=args.auto_precondition_fill,
        precondition_target_volume=args.precondition_target_volume,
        precondition_fill_tol=args.precondition_fill_tol,
    )

if __name__ == "__main__":
    main()