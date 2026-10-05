#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Rebuild segmented G-code and globally recompute FIFO residence times.

The script reads segmented G-code containing ``; SEG`` markers, reconstructs
the complete processing timeline, assigns extrusion pieces to FIFO cycles, and
calculates the input time, output time, and residence time of each segment.

Manual G-code commands containing the ``MANUAL`` marker are included in the
timing calculation. The script exports a rebuilt CSV file and a rebuilt G-code
file with globally renumbered ``seg_idx`` identifiers.

An optional automatic precondition-fill operation can insert an E-only segment
when the initial precondition volume is lower than the target FIFO volume.

CLI example:
python rebuild_csv_and_gcode_fifo_v1_1.py \
      CFFFP_testmodel_1116s_F0.3_T0.8_E0.95_Z+0.2_segmented.gcode \
      --out CFFFP_testmodel_1116s_F0.3_T0.8_E0.95_Z+0.2_segmented_rebuilt.csv \
      --gcode_out CFFFP_testmodel_1116s_F0.3_T0.8_E0.95_Z+0.2_segmented_rebuilt.gcode \
      --vnozzle 723 \
      --d_filament 15.55634919 --emode filament
      
Single Line:
python rebuild_csv_and_gcode_fifo_v1_1.py CFFFP_testmodel_1170s_x0.2.gcode --out CFFFP_testmodel_1170s_x0.2_rebuilt_segmented.csv --gcode_out CFFFP_testmodel_1170s_x0.2.gcode_rebuilt_segmented.gcode --vnozzle 723 --d_filament 15.55634919 --emode filament
"""

import argparse
import csv
import math
import os
import re
from collections import deque

SEGIDX_RE = re.compile(r"\bseg_idx\s*=\s*(\d+)\b", re.IGNORECASE)
SEGIDX_STRIP_RE = re.compile(r"(?:\s*\|\s*)?\bseg_idx\s*=\s*\d+\b", re.IGNORECASE)


def extract_segidx_from_line(line: str):
    """Extract the segment index from a G-code comment.

    Args:
        line: Complete G-code line that may contain a ``seg_idx`` token.

    Returns:
        The segment index as an integer, or ``None`` if no valid
        ``seg_idx`` token is found.
    """
    if not line:
        return None
    raw = line.rstrip("\n")
    code_part, sep, comment_part = raw.partition(";")
    if not sep:
        return None
    m = SEGIDX_RE.search(comment_part)
    return int(m.group(1)) if m else None


def strip_segidx_from_comment(comment: str) -> str:
    """Remove the segment index token from a G-code comment.

    Args:
        comment: Comment text that may contain a ``seg_idx`` token.

    Returns:
        The cleaned comment without the segment index.
    """
    if not comment:
        return ""
    c = SEGIDX_STRIP_RE.sub("", comment)
    c = re.sub(r"\s*\|\s*$", "", c).strip()
    return c


def ensure_comment_has_segidx(line: str, seg_idx: int) -> str:
    """Ensure that a G-code line contains the specified segment index.

    Any existing segment index is removed before the new index is appended.

    Args:
        line: Complete G-code line to update.
        seg_idx: Segment index to append to the comment.

    Returns:
        The updated G-code line ending with a newline character.
    """
    raw = line.rstrip("\n")
    code_part, sep, comment_part = raw.partition(";")
    code = code_part.rstrip()
    c = comment_part.strip() if sep else ""
    c_clean = strip_segidx_from_comment(c)
    new_c = (c_clean + " " if c_clean else "") + f"seg_idx={int(seg_idx)}"
    return f"{code} ; {new_c}\n"


# ---------------------------- Utilities ----------------------------

def fmt_fixed(x: float, nd: int = 5) -> str:
    """Format a numeric value without unnecessary trailing zeros.

    Args:
        x: Value to format.
        nd: Maximum number of decimal places. Defaults to 5.

    Returns:
        The formatted value. Returns an empty string if ``x`` is
        ``None``. Non-numeric values are converted directly to strings.
    """
    if x is None:
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def area_circle(d: float) -> float:
    """Calculate the area of a circle from its diameter.

    Args:
        d: Circle diameter.

    Returns:
        The circle area.
    """
    return math.pi * (0.5 * d) ** 2


def distance_xy(p1, p2) -> float:
    """Calculate the planar distance between two positions.

    Args:
        p1: Mapping containing the starting ``X`` and ``Y`` coordinates.
        p2: Mapping containing the ending ``X`` and ``Y`` coordinates.

    Returns:
        The Euclidean distance in the XY plane.

    Raises:
        KeyError: If either position does not contain ``X`` or ``Y``.
    """
    return math.hypot(p2["X"] - p1["X"], p2["Y"] - p1["Y"])


def volume_from_E(
    dE: float,
    d_filament: float,
    e_mode: str,
) -> float:
    """Convert an E-axis increment to extrusion volume.

    Args:
        dE: E-axis increment.
        d_filament: Filament diameter in millimetres.
        e_mode: Interpretation of the E value. Supported modes are
            ``"filament"`` and ``"mm3"``.

    Returns:
        The corresponding extrusion volume in cubic millimetres.

    Raises:
        ValueError: If ``e_mode`` is not ``"filament"`` or ``"mm3"``.
    """
    if abs(dE) < 1e-12:
        return 0.0
    if e_mode == "filament":
        return area_circle(d_filament) * dE
    if e_mode == "mm3":
        return dE
    raise ValueError(f"Unknown e_mode: {e_mode}")


def E_from_volume(
    V_mm3: float,
    d_filament: float,
    e_mode: str,
) -> float:
    """Convert extrusion volume to an E-axis increment.

    This function is used when generating automatic precondition top-up
    segments. In filament mode, the E value represents filament length.
    In cubic-millimetre mode, the E value directly represents volume.

    Args:
        V_mm3: Extrusion volume in cubic millimetres.
        d_filament: Filament diameter in millimetres.
        e_mode: Interpretation of the E value. Supported modes are
            ``"filament"`` and ``"mm3"``.

    Returns:
        The corresponding E-axis increment. Returns zero if the volume is
        negligible or the calculated filament area is invalid.

    Raises:
        ValueError: If ``e_mode`` is not ``"filament"`` or ``"mm3"``.
    """
    V = float(V_mm3 or 0.0)
    if abs(V) < 1e-12:
        return 0.0
    if e_mode == "filament":
        A = area_circle(d_filament)
        if A <= 1e-12:
            return 0.0
        return V / A
    if e_mode == "mm3":
        return V
    raise ValueError(f"Unknown e_mode: {e_mode}")


def parse_g4_ts(parts):
    """Extract the dwell duration from a tokenized G4 command.

    An S value is interpreted as seconds. A P value is interpreted as
    milliseconds. If both are present, the S value takes priority.

    Args:
        parts: Sequence of tokens representing a G4 command.

    Returns:
        The non-negative dwell duration in seconds. Returns zero if no
        valid S or P value is found.
    """
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
        return max(0.0, float(s_s))
    if p_ms is not None:
        return max(0.0, float(p_ms) / 1000.0)
    return 0.0


def has_optimizable(line: str) -> bool:
    """Return whether a line contains the ``optimizable`` marker."""
    return "optimizable" in (line or "").lower()


def has_precondition(line: str) -> bool:
    """Return whether a line contains the ``precondition`` marker."""
    return "precondition" in (line or "").lower()


def has_manual(line: str) -> bool:
    """Return whether a line contains the ``manual`` marker."""
    return "manual" in (line or "").lower()


def strip_segidx_anywhere(text: str) -> str:
    """Remove all segment index tokens from text.

    Args:
        text: Text that may contain one or more ``seg_idx`` tokens.

    Returns:
        The cleaned and whitespace-normalized text.
    """
    if not text:
        return ""
    t = re.sub(r"\s*\bseg_idx\s*=\s*\d+\b\s*", " ", text, flags=re.IGNORECASE)
    return " ".join(t.split()).strip()


def split_code_and_comment(line: str):
    """Split a G-code line into code and comment components.

    Args:
        line: Complete G-code line that may contain a semicolon comment.

    Returns:
        A tuple containing the G-code portion and comment portion.
        Returns two empty strings if ``line`` is ``None``.
    """
    if line is None:
        return "", ""
    if ";" in line:
        a, b = line.split(";", 1)
        return a.rstrip(), b.strip()
    return line.rstrip(), ""


# ---------------- FIFO helper: EXACTLY v6.1 ----------------
def fifo_backfill_times(
    pieces_in_cycle,
    prev_out_log,
    fallback_time,
):
    """Calculate FIFO input times using the previous cycle's output parcels.

    Each input piece is filled by volume-weighted sampling of output
    parcels from the previous cycle. If the available parcel volume is
    insufficient, ``fallback_time`` is used for the remaining volume.

    Args:
        pieces_in_cycle: Current-cycle pieces containing a ``V`` volume.
        prev_out_log: Previous-cycle parcels containing ``V`` and
            ``t_out`` values. May be ``None``.
        fallback_time: Time assigned to volume that cannot be matched to
            a previous output parcel.

    Returns:
        A list of volume-weighted input times corresponding to the
        current-cycle pieces.
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

        if need > 1e-12:
            acc_t += fallback_time * need
            acc_v += need
            need = 0.0

        result.append(acc_t / acc_v if acc_v > 0 else fallback_time)

    return result


# ---------------------------- Core ----------------------------

def rebuild_global(
    gcode_in: str,
    csv_out: str,
    gcode_out: str,
    V_nozzle: float,
    d_filament: float,
    e_mode: str,
    auto_precondition_fill: bool = False,
    precondition_target_volume: float = None,
    precondition_fill_tol: float = 1e-6,
):
    """Rebuild segmented G-code and globally recompute FIFO timing.

    The function parses the segmented G-code, reconstructs its motion
    timeline, assigns extrusion pieces to FIFO cycles, calculates input,
    output, and residence times, and exports aligned G-code and CSV files.

    Optionally, an E-only precondition segment is inserted when the
    existing precondition volume is below the requested target.

    Args:
        gcode_in: Path to the input segmented G-code file.
        csv_out: Path for the rebuilt CSV output.
        gcode_out: Path for the rebuilt G-code output.
        V_nozzle: Nozzle or FIFO control volume in cubic millimetres.
        d_filament: Filament diameter in millimetres.
        e_mode: Interpretation of E values. Supported modes are
            ``"filament"`` and ``"mm3"``.
        auto_precondition_fill: Whether to insert an automatic
            precondition top-up segment. Defaults to ``False``.
        precondition_target_volume: Target precondition volume in cubic
            millimetres. If ``None``, ``V_nozzle`` is used.
        precondition_fill_tol: Volume tolerance used to determine whether
            a precondition top-up is required. Defaults to ``1e-6``.

    Returns:
        None.

    Raises:
        OSError: If an input or output file cannot be opened.
        RuntimeError: If the input contains no segmented extrusion pieces.
        ValueError: If an unsupported E mode is provided.
    """
    # -------- Phase A: Parse segmented gcode into expanded timeline --------
    expanded = []
    last = {"X": 0.0, "Y": 0.0, "Z": 0.0, "F": 1000.0}
    seen_first_seg_piece = False   # ✅ once we see the first "; SEG" piece, pre_infill ends


    with open(gcode_in, "r", encoding="utf-8", errors="ignore") as fin:
        for raw in fin:
            line0 = raw.rstrip("\n")
            opt = has_optimizable(line0)
            manu = has_manual(line0)
            precond = has_precondition(line0)

            # remove old seg_idx for parsing only
            orig_line = line0
            clean_line = strip_segidx_anywhere(line0)
            stripped = clean_line.strip()

            if stripped == "" or stripped.startswith(";"):
                expanded.append({
                    "type": "raw",
                    "orig_line": orig_line,
                    "t_s": 0.0,
                    "optimizable": opt,
                    "manual": manu,
                    "precondition": precond,
                })
                continue

            code_part, comment_part = split_code_and_comment(clean_line)
            code_part = code_part.strip()
            comment_part = strip_segidx_anywhere(comment_part)
            parts = code_part.split()
            cmd = parts[0].upper()

            # dwell
            if cmd in ("G4", "G04"):
                t_s = parse_g4_ts(parts)
                expanded.append({
                    "type": "g4",
                    "orig_line": orig_line,
                    "parts": parts,
                    "t_s": t_s,
                    "optimizable": opt,
                    "manual": manu,
                    "precondition": precond,
                })
                continue

            # other non-motion
            if cmd not in ("G0", "G1"):
                expanded.append({
                    "type": "raw",
                    "orig_line": orig_line,
                    "optimizable": opt,
                    "manual": manu,
                    "t_s": 0.0,
                    "precondition": precond,
                })
                continue

            # modal move: relative E (M83) => dE is RELATIVE
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
                elif c in ("X", "Y", "Z", "F"):
                    move[c] = v

            d_xy = distance_xy(last, move)
            F = move.get("F", last.get("F", 1000.0))

            if d_xy > 1e-12 and F > 1e-12:
                t_s = 60.0 * d_xy / F
            elif abs(dE) > 1e-12 and F > 1e-12:
                t_s = 60.0 * abs(dE) / F
            else:
                t_s = 0.0

            c_upper = (comment_part or "").upper()
            is_precondition = ("PRECONDITION" in c_upper)

            is_piece = ("SEG" in c_upper) or is_precondition

            if is_piece:
                seen_first_seg_piece = True

            is_pre_infill = ("PRE_INFILL" in c_upper)

            implicit_pre_infill = (
                (not seen_first_seg_piece)
                and (cmd == "G1")
                and has_E
                and (dE > 1e-12)
                and (d_xy <= 1e-12)
                and (not is_precondition)
            )

            if is_piece:
                V = volume_from_E(dE, d_filament, e_mode)
                expanded.append({
                    "type": "piece",
                    "orig_line": orig_line,
                    "optimizable": opt,
                    "manual": manu,
                    "precondition": is_precondition,
                    "p1": {"X": last["X"], "Y": last["Y"], "Z": last["Z"]},
                    "p2": {"X": move["X"], "Y": move["Y"], "Z": move["Z"]},
                    "F": F,
                    "d_len": d_xy,
                    "t_s": t_s,
                    "V": V,
                    "dE": dE,
                })
            elif is_pre_infill or implicit_pre_infill:
                V = volume_from_E(dE, d_filament, e_mode) if (has_E and dE > 1e-12) else 0.0
                expanded.append({
                    "type": "pre_infill",
                    "orig_line": orig_line,
                    "cmd": cmd,
                    "F": F,
                    "t_s": t_s,
                    "V": V,
                    "dE": dE,
                    "line": code_part + ((" ; " + comment_part) if comment_part else ""),
                    "optimizable": opt,
                    "manual": manu,
                    "precondition": precond,
                })
            else:
                expanded.append({
                    "type": "move",
                    "orig_line": orig_line,
                    "cmd": cmd,
                    "last": last.copy(),
                    "move": move.copy(),
                    "line": code_part + ((" ; " + comment_part) if comment_part else ""),
                    "optimizable": opt,
                    "manual": manu,
                    "t_s": t_s,
                    "precondition": precond,
                })

            last = move.copy()

    # -------- Phase B.5: Optional automatic precondition volume top-up --------
    # Goal:
    #   If the precondition volume before the first real extrusion is slightly
    #   below the target volume, insert one E-only precondition/optimizable
    #   top-up piece with the same feedrate as the last precondition piece.
    #   This prevents the first real extrusion from being consumed to close
    #   the first FIFO cycle.
    if auto_precondition_fill:
        target_v = float(precondition_target_volume) if precondition_target_volume is not None else float(V_nozzle)
        tol_v = max(0.0, float(precondition_fill_tol))

        piece_indices = [i for i, it in enumerate(expanded) if it.get("type") == "piece"]
        first_real_piece_idx = next(
            (i for i in piece_indices if not expanded[i].get("precondition", False)),
            None
        )

        precond_indices = [
            i for i in piece_indices
            if expanded[i].get("precondition", False)
            and (first_real_piece_idx is None or i < first_real_piece_idx)
        ]

        precond_v = sum(float(expanded[i].get("V", 0.0) or 0.0) for i in precond_indices)
        deficit_v = target_v - precond_v

        if precond_indices and target_v > 1e-12 and deficit_v > tol_v:
            ref = expanded[precond_indices[-1]]
            F_ref = float(ref.get("F", 0.0) or 0.0)
            if F_ref <= 1e-12:
                F_ref = 1.0

            dE_fill = E_from_volume(deficit_v, d_filament, e_mode)
            t_s_fill = 60.0 * abs(dE_fill) / F_ref if F_ref > 1e-12 else 0.0

            p = ref.get("p2", ref.get("p1", {"X": 0.0, "Y": 0.0, "Z": 0.0}))
            p1 = {"X": float(p.get("X", 0.0)), "Y": float(p.get("Y", 0.0)), "Z": float(p.get("Z", 0.0))}
            p2 = dict(p1)

            fill_line = (
                f"G1 X{fmt_fixed(p2['X'], 5)} "
                f"Y{fmt_fixed(p2['Y'], 5)} "
                f"Z{fmt_fixed(p2['Z'], 5)} "
                f"E{fmt_fixed(dE_fill, 6)} "
                f"F{fmt_fixed(F_ref, 3)} ; SEG precondition optimizable auto_fill"
            )

            fill_piece = {
                "type": "piece",
                "orig_line": fill_line,
                "optimizable": True,
                "manual": False,
                "precondition": True,
                "auto_fill": True,
                "p1": p1,
                "p2": p2,
                "F": F_ref,
                "d_len": 0.0,
                "t_s": t_s_fill,
                "V": deficit_v,
                "dE": dE_fill,
            }

            # Insert immediately after the last precondition piece, not merely
            # before the first real extrusion. This keeps the top-up inside the
            # preconditioning block and before any travel/non-extrusion delay.
            insert_at = precond_indices[-1] + 1
            expanded.insert(insert_at, fill_piece)

            print(
                f"🧩 Auto precondition fill inserted: "
                f"target={target_v:.5f} mm^3, existing={precond_v:.5f} mm^3, "
                f"added={deficit_v:.8f} mm^3, E={dE_fill:.8f}, F={F_ref:.3f}"
            )
        elif precond_indices:
            print(
                f"✅ Precondition volume already sufficient: "
                f"existing={precond_v:.5f} mm^3, target={target_v:.5f} mm^3"
            )
        else:
            print("⚠️ Auto precondition fill requested, but no precondition piece was found before the first real extrusion.")

    pieces = [it for it in expanded if it["type"] == "piece"]
    if not pieces:
        raise RuntimeError("No piece lines found. Input must contain '; SEG'.")

    # -------- Phase C (fixed): cycle assignment by boundary crossing --------
    S = 0.0
    cycle_idx = 1
    local_idx = 0
    next_boundary = V_nozzle
    tol = max(1e-6 * V_nozzle, 1e-6)

    for p in pieces:
        if S >= next_boundary - tol:
            cycle_idx += 1
            local_idx = 0
            next_boundary += V_nozzle

        local_idx += 1
        p["cycle"] = cycle_idx
        p["cycle_local"] = local_idx
        S += p["V"]
        
    # -------- Phase D: time origin starts at first precondition piece --------
    nonextr_delay_up_to = []
    delay = 0.0
    time_started = False

    time_origin_idx = None
    for idx, it in enumerate(expanded):
        if it.get("type") == "piece" and it.get("precondition", False):
            time_origin_idx = idx
            break

    # Fallback for old files without precondition.
    if time_origin_idx is None:
        for idx, it in enumerate(expanded):
            if it.get("type") == "piece":
                time_origin_idx = idx
                break
        print("⚠️ No precondition found. Falling back to first piece as time origin.")

    for idx, it in enumerate(expanded):

        if (time_origin_idx is not None) and (idx == time_origin_idx):
            time_started = True
            delay = 0.0

        if it["type"] == "piece":
            nonextr_delay_up_to.append(delay if time_started else 0.0)
        else:
            if time_started:
                delay += float(it.get("t_s", 0.0) or 0.0)

    if nonextr_delay_up_to:
        print(
            f"⏱ first_piece_delay={nonextr_delay_up_to[0]:.5f}s "
            f"(time_origin_idx={time_origin_idx}, "
            f"origin_type=precondition)"
        )

    # -------- Phase E (EXACT v6.1): FIFO timing --------
    max_cycle = max(p["cycle"] for p in pieces)
    cycles = [[p for p in pieces if p["cycle"] == c] for c in range(1, max_cycle + 1)]

    def cum_durs(plist, inclusive=False):
        """Calculate cumulative durations for a sequence of pieces.

        Args:
            plist: Sequence of pieces containing ``t_s`` durations.
            inclusive: Whether each cumulative value includes the current
                piece's duration. Defaults to ``False``.

        Returns:
            A list of cumulative durations.
        """
        out, acc = [], 0.0
        for p in plist:
            if inclusive:
                acc += p["t_s"]
                out.append(acc)
            else:
                out.append(acc)
                acc += p["t_s"]
        return out

    global_index_map = {(p["cycle"], p["cycle_local"]): i for i, p in enumerate(pieces)}
    nonextr_by_ord = nonextr_delay_up_to

    final_timing = {}
    last_cycle_last_tout_abs = 0.0
    prev_out_log = None  # deque of {"V":..., "t_out":...}

    for cidx, cyc in enumerate(cycles, start=1):
        N = len(cyc)
        if N == 0:
            continue

        ord_indices = [global_index_map[(p["cycle"], p["cycle_local"])] for p in cyc]

        if cidx == 1:
            offs_incl = cum_durs(cyc, inclusive=True)
            first_delay = nonextr_by_ord[ord_indices[0]]

            abs_t_out = []
            for i, off in enumerate(offs_incl):
                extra_delay = nonextr_by_ord[ord_indices[i]] - first_delay
                if extra_delay < 0:
                    extra_delay = 0.0
                abs_t_out.append(first_delay + off + extra_delay)

            # Cycle 1 input time starts from the first precondition piece.
            base_t_in = cum_durs(cyc, inclusive=False)

        else:
            offs_incl = cum_durs(cyc, inclusive=True)

            first_delay = nonextr_by_ord[ord_indices[0]]

            prev_cycle = cycles[cidx - 2]
            prev_last_piece = prev_cycle[-1]
            prev_last_ord = global_index_map[(prev_last_piece["cycle"], prev_last_piece["cycle_local"])]
            prev_last_delay = nonextr_by_ord[prev_last_ord]

            inter_cycle_delay = first_delay - prev_last_delay
            if inter_cycle_delay < 0:
                inter_cycle_delay = 0.0

            abs_t_out = []
            for i, off in enumerate(offs_incl):
                extra_delay = nonextr_by_ord[ord_indices[i]] - first_delay
                abs_t_out.append(last_cycle_last_tout_abs + inter_cycle_delay + off + extra_delay)

            base_t_in = fifo_backfill_times(cyc, prev_out_log, last_cycle_last_tout_abs)

        for i, p in enumerate(cyc):
            t_in = base_t_in[i]
            t_out = abs_t_out[i]
            final_timing[(p["cycle"], p["cycle_local"])] = {
                "t_in": t_in,
                "t_out": t_out,
                "tau_s": t_out - t_in,
            }

        last_cycle_last_tout_abs = abs_t_out[-1]
        prev_out_log = deque({"V": cyc[i]["V"], "t_out": abs_t_out[i]} for i in range(N))

    # -------- Phase F: Emit rebuilt gcode + CSV (RENUMBER seg_idx globally, overwrite old) --------
    header = [
        "seg_idx", "cmd", "cycle", "cycle_local",
        "d_mm", "F_mm_per_min", "V_mm3",
        "t_s", "t_in", "t_out", "tau_s", "note",
    ]

    def build_note(
        base: str,
        manual_flag: bool,
        opt_flag: bool,
        precondition_flag: bool = False,
    ) -> str:
        """Build a comma-separated CSV note from active status tags.

        Args:
            base: Base note category.
            manual_flag: Whether to include the ``manual`` tag.
            opt_flag: Whether to include the ``optimizable`` tag.
            precondition_flag: Whether to include the ``precondition``
                tag. Defaults to ``False``.

        Returns:
            A comma-separated string containing the applicable tags.
        """
        tags = []

        if base:
            tags.append(base)

        if precondition_flag and "precondition" not in tags:
            tags.append("precondition")

        if manual_flag and "manual" not in tags:
            tags.append("manual")

        if opt_flag and "optimizable" not in tags:
            tags.append("optimizable")

        return ",".join(tags)

    rows = []
    seg_idx = 0

    SEGIDX_ANY_RE = re.compile(r"\bseg_idx\s*=\s*\d+\b", re.IGNORECASE)

    def strip_all_segidx(line: str) -> str:
        """Remove all segment index tokens from a complete G-code line.

        Args:
            line: G-code line that may contain segment index tokens.

        Returns:
            The cleaned line with normalized spacing.
        """
        if not line:
            return ""
        s = line.rstrip("\n")
        s = SEGIDX_ANY_RE.sub("", s)
        # clean up double spaces
        s = re.sub(r"[ \t]{2,}", " ", s)
        # clean up comment separators like " ; ; "
        s = re.sub(r"\s*;\s*;\s*", " ; ", s)
        # trim spaces around ';'
        s = re.sub(r"\s*;\s*", " ; ", s)
        return s.rstrip()

    def append_new_segidx(line_no_seg: str, new_idx: int) -> str:
        """Append a new segment index to a G-code line.

        Empty lines are converted into comment lines. Existing comment
        lines receive the index directly, while executable G-code lines
        receive the index inside a semicolon comment.

        Args:
            line_no_seg: G-code line from which old segment indices have
                already been removed.
            new_idx: New segment index to append.

        Returns:
            The updated G-code line ending with a newline character.
        """
        s = (line_no_seg or "").rstrip("\n")
        stripped = s.strip()

        if stripped == "":
            # keep empty line but still tag seg_idx
            return f"; seg_idx={new_idx}\n"

        if stripped.startswith(";"):
            return f"{s} seg_idx={new_idx}\n"

        code, sep, cmt = s.partition(";")
        code = code.rstrip()
        cmt = cmt.strip() if sep else ""

        if cmt:
            return f"{code} ; {cmt} seg_idx={new_idx}\n"
        else:
            return f"{code} ; seg_idx={new_idx}\n"

    def add_csv_row(idx: int, *, cmd="", cycle="", cycle_local="", d_mm=0.0, F_mm_per_min=0.0,
                    V_mm3=0.0, t_s=0.0, t_in="", t_out="", tau_s="", note="") -> None:
        """Append one formatted record to the CSV output rows.

        Args:
            idx: Global segment index.
            cmd: G-code command name.
            cycle: Assigned FIFO cycle.
            cycle_local: Local segment index within the FIFO cycle.
            d_mm: XY travel distance in millimetres.
            F_mm_per_min: Feed rate in millimetres per minute.
            V_mm3: Extrusion volume in cubic millimetres.
            t_s: Command duration in seconds.
            t_in: Calculated FIFO input time.
            t_out: Calculated FIFO output time.
            tau_s: Calculated residence time.
            note: Comma-separated status description.

        Returns:
            None.
        """
        rows.append({
            "seg_idx": int(idx),
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

    with open(gcode_out, "w", encoding="utf-8", newline="") as fout:
        for it in expanded:
            typ = it["type"]
            opt = it.get("optimizable", False)
            manu = it.get("manual", False)

            original_line = it.get("orig_line", "")
            if original_line is None:
                original_line = ""

            # 1) ALWAYS renumber every line in the output file
            seg_idx += 1
            no_old = strip_all_segidx(original_line)
            fout.write(append_new_segidx(no_old, seg_idx))

            # 2) CSV: keep your original model -> only for non-empty, non-pure-comment lines
            s = (no_old or "").strip()
            # (A) blank line
            if s == "":
                add_csv_row(
                    seg_idx,
                    cmd="",
                    d_mm=0.0, F_mm_per_min=0.0, V_mm3=0.0,
                    t_s=0.0,
                    note=build_note("blank", manu, opt),
                )
                continue

            # (B) pure comment line
            if s.startswith(";"):
                add_csv_row(
                    seg_idx,
                    cmd=";",
                    d_mm=0.0, F_mm_per_min=0.0, V_mm3=0.0,
                    t_s=0.0,
                    note=build_note("comment", manu, opt),
                )
                continue


            if typ == "piece":
                key = (it["cycle"], it["cycle_local"])
                tinfo = final_timing.get(key, {"t_in": None, "t_out": None, "tau_s": None})
                is_precond_piece = bool(it.get("precondition", False))
                base_note = "precondition" if is_precond_piece else "extrusion"

                if is_precond_piece:
                    # Precondition pieces represent material entering the FIFO,
                    # not a real object-side extrusion output. Therefore the CSV
                    # reports cumulative input time in t_in, leaves t_out blank,
                    # and uses cumulative elapsed residence time in tau_s.
                    precond_elapsed = tinfo.get("t_out", None)
                    csv_t_in = fmt_fixed(precond_elapsed, 5) if precond_elapsed is not None else ""
                    csv_t_out = ""
                    csv_tau = fmt_fixed(precond_elapsed, 5) if precond_elapsed is not None else ""
                else:
                    csv_t_in = fmt_fixed(tinfo["t_in"], 5) if tinfo["t_in"] is not None else ""
                    csv_t_out = fmt_fixed(tinfo["t_out"], 5) if tinfo["t_out"] is not None else ""
                    csv_tau = fmt_fixed(tinfo["tau_s"], 5) if tinfo["tau_s"] is not None else ""

                note = build_note(
                    base_note,
                    manu,
                    opt,
                    precondition_flag=is_precond_piece,
                )
                if it.get("auto_fill", False) and "auto_fill" not in note:
                    note = note + ",auto_fill" if note else "auto_fill"

                add_csv_row(
                    seg_idx,
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
                    note=note,
                )
            elif typ == "g4":
                add_csv_row(seg_idx, cmd="G4", t_s=it.get("t_s", 0.0), note=build_note("dwell", manu, opt))
            elif typ == "move":
                last_mv = it["last"]
                cur = it["move"]
                d_xy = distance_xy(last_mv, cur)
                F = cur.get("F", last_mv.get("F", 1000.0))
                add_csv_row(
                    seg_idx,
                    cmd=it["cmd"],
                    d_mm=d_xy,
                    F_mm_per_min=F,
                    V_mm3=0.0,
                    t_s=it.get("t_s", 0.0),
                    note=build_note("travel", manu, opt),
                )
            elif typ == "pre_infill":
                add_csv_row(
                    seg_idx,
                    cmd=it["cmd"],
                    F_mm_per_min=it["F"],
                    V_mm3=it["V"],
                    t_s=it["t_s"],
                    note=build_note("pre_infill", manu, opt),
                )
            else:
                # raw but not empty/comment (e.g., G28, G92, M83, M104...)
                tok = s.split()[0].upper() if s else ""
                add_csv_row(
                    seg_idx,
                    cmd=tok,
                    t_s=it.get("t_s", 0.0),
                    note=build_note("raw", manu, opt),
                )


    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    print(f"✅ Rebuilt CSV   : {os.path.abspath(csv_out)} | rows={len(rows)}")
    print(f"✅ Rebuilt G-code: {os.path.abspath(gcode_out)} | total_lines(seg_idx)={seg_idx}")
    print(f"✅ cycles_found  : {max(p['cycle'] for p in pieces)}")

# ---------------------------- CLI ----------------------------

def main():
    """Parse command-line arguments and run the global FIFO rebuild.

    The command-line interface collects the input and output paths,
    nozzle volume, filament diameter, E mode, and optional automatic
    precondition-fill settings.

    Returns:
        None.

    Raises:
        SystemExit: If required command-line arguments are missing or
            invalid.
        OSError: If an input or output file cannot be accessed.
        RuntimeError: If the input contains no segmented extrusion pieces.
    """
    ap = argparse.ArgumentParser(description="GLOBAL FIFO rebuild (v6.1) + MANUAL detection -> CSV note.")
    ap.add_argument("gcode", help="Input segmented gcode (must contain '; SEG')")
    ap.add_argument("--out", default="rebuilt_segments.csv", help="Output CSV filename")
    ap.add_argument("--gcode_out", default="rebuilt_segmented.gcode", help="Output rebuilt gcode filename")
    ap.add_argument("--vnozzle", type=float, required=True, help="V_nozzle [mm^3]")
    ap.add_argument("--d_filament", type=float, required=True, help="d_filament for E->V")
    ap.add_argument("--emode", choices=["filament", "mm3"], required=True, help="E mode")
    ap.add_argument(
        "--auto_precondition_fill",
        action="store_true",
        help="If enabled, insert one E-only precondition optimizable auto-fill piece when the precondition volume before the first real extrusion is below the target volume.",
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
        help="Volume tolerance [mm^3] for deciding whether auto precondition fill is needed.",
    )
    args = ap.parse_args()

    rebuild_global(
        gcode_in=args.gcode,
        csv_out=args.out,
        gcode_out=args.gcode_out,
        V_nozzle=args.vnozzle,
        d_filament=args.d_filament,
        e_mode=args.emode,
        auto_precondition_fill=args.auto_precondition_fill,
        precondition_target_volume=args.precondition_target_volume,
        precondition_fill_tol=args.precondition_fill_tol,
    )


if __name__ == "__main__":
    main()
