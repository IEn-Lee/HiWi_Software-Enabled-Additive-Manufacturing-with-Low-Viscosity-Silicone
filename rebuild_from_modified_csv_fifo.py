#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rebuild_from_modified_csv_fifo.py
---------------------------------
Rebuild FIFO timing (t_in/t_out/tau_s) and regenerate G-code values
based on a user-modified CSV (same schema as segment_fifo_builder_v6.1 output).

Inputs:
  1) modified CSV (seg_idx-aligned, same columns)
  2) reference old segmented G-code (used as geometry/text template)

Outputs:
  1) new CSV (same columns) with recomputed t_in/t_out/tau_s
  2) new G-code updated according to the modified CSV (by seg_idx mapping)

Key behavior:
- FIFO timing model matches your v6.1 "Phase E" logic:
  * time origin starts at first row that is "optimizable" (note contains 'optimizable')
  * inj_end_time accumulates t_s of NON-PIECE rows after time origin, and updates whenever
    a NON-PIECE row is optimizable
  * For cycle 1: t_in is a linear ramp over inj_end_time: i*(inj_end_time/N)
  * For cycles >=2: t_in is FIFO backfilled from previous cycle's absolute t_out
- non-extrusion delays are taken from the CSV row order:
  * piece rows: contribute to extrusion offsets (t_s)
  * non-piece rows: contribute to delay accumulator (t_s)
- G-code regeneration:
  * lines are mapped by seg_idx (must exist in the gcode comment)
  * for extrusion pieces (note contains 'extrusion' or cmd==G1 with cycle>=1):
      - F is updated from CSV F_mm_per_min
      - E is optionally updated from CSV V_mm3 (via --update_e)
  * for pre_infill rows (note contains 'pre_infill'):
      - F and E can also be updated similarly
  * other lines are kept as-is (except still keep original seg_idx, no reindex)

CLI example:
python rebuild_from_modified_csv_fifo.py \
  --csv_in house_modified.csv \
  --gcode_ref house_old_segmented.gcode \
  --csv_out house_rebuilt.csv \
  --gcode_out house_rebuilt.gcode \
  --emode filament --d_filament 15.55634919 \
  --update_e

"""

import csv
import math
import argparse
import re
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional


# ---------------------------- Formatting ----------------------------

def fmt_fixed(x: float, nd: int = 5) -> str:
    if x is None:
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def area_circle(d: float) -> float:
    return math.pi * (0.5 * d) ** 2


# ---------------------------- FIFO backfill ----------------------------

def fifo_backfill_times(pieces_in_cycle: List[dict], prev_out_log: Optional[deque], fallback_time: float) -> List[float]:
    """
    Volume-weighted FIFO mapping:
    current piece's t_in = weighted average of previous cycle parcels' t_out
    """
    fifo = deque(prev_out_log) if prev_out_log is not None else deque()
    result = []

    for p in pieces_in_cycle:
        need = float(p["V"])
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


# ---------------------------- CSV I/O ----------------------------

CSV_HEADER = [
    "seg_idx", "cmd", "cycle", "cycle_local",
    "d_mm", "F_mm_per_min", "V_mm3",
    "t_s", "t_in", "t_out", "tau_s", "note",
]


def _to_float(s: str, default: float = 0.0) -> float:
    try:
        if s is None:
            return default
        s2 = str(s).strip()
        if s2 == "":
            return default
        return float(s2)
    except Exception:
        return default


def _to_int(s: str, default: int = 0) -> int:
    try:
        if s is None:
            return default
        s2 = str(s).strip()
        if s2 == "":
            return default
        return int(float(s2))
    except Exception:
        return default


def read_modified_csv(csv_path: str) -> List[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        # accept same header; if extra columns exist, keep them
        for row in r:
            rows.append(row)
    return rows


# ---------------------------- Identify row types ----------------------------

def is_optimizable_row(row: dict) -> bool:
    note = (row.get("note") or "").lower()
    return "optimizable" in note


def is_piece_row(row: dict) -> bool:
    """
    We treat rows as "piece" if they represent segmented extrusion pieces:
    - cmd == G1 AND cycle >= 1 AND V_mm3 > 0
    - or note contains 'extrusion'
    """
    cmd = (row.get("cmd") or "").strip().upper()
    cyc = _to_int(row.get("cycle"), 0)
    V = _to_float(row.get("V_mm3"), 0.0)
    note = (row.get("note") or "").lower()

    if "extrusion" in note:
        return True
    if cmd == "G1" and cyc >= 1 and V > 1e-12:
        return True
    return False


def is_pre_infill_row(row: dict) -> bool:
    note = (row.get("note") or "").lower()
    return "pre_infill" in note


# ---------------------------- Timing rebuild ----------------------------

def recompute_piece_ts_from_F(rows, *, e_mode: str, d_filament: float) -> List[dict]:
    """
    Recompute t_s for piece/pre_infill rows based on NEW F:
      - if d_mm > 0:  t_s = 60 * d_mm / F
      - else (E-only): t_s = 60 * |E| / F, where E derived from V_mm3 and e_mode
    Returns a NEW list of rows (dict-copied) with updated t_s (string).
    """
    out = [dict(r) for r in rows]

    for r in out:
        if not (is_piece_row(r) or is_pre_infill_row(r)):
            continue

        F = _to_float(r.get("F_mm_per_min"), 0.0)
        if F <= 1e-12:
            continue  # can't recompute

        d_mm = _to_float(r.get("d_mm"), 0.0)

        if d_mm > 1e-12:
            t_s_new = 60.0 * d_mm / F
            r["t_s"] = fmt_fixed(t_s_new, 5)
            continue

        # E-only path
        V = _to_float(r.get("V_mm3"), 0.0)
        if abs(V) <= 1e-12:
            r["t_s"] = fmt_fixed(0.0, 5)
            continue

        E = compute_E_from_V(V, e_mode=e_mode, d_filament=d_filament)
        t_s_new = 60.0 * abs(E) / F
        r["t_s"] = fmt_fixed(t_s_new, 5)

    return out

def rebuild_fifo_timing_from_csv(rows: List[dict]) -> Tuple[List[dict], float]:
    """
    Returns: (updated_rows, inj_end_time)
    - updates t_in, t_out, tau_s fields for piece rows
    - leaves other rows unchanged (t_in/t_out/tau_s blank)
    """

    # Build list of piece ordinals and their non-extrusion delay before each piece
    nonextr_delay_up_to_piece: List[float] = []
    piece_ord_to_row_index: List[int] = []

    delay = 0.0
    time_started = False
    inj_end_time = 0.0

    for i, row in enumerate(rows):
        opt = is_optimizable_row(row)
        if (not time_started) and opt:
            time_started = True
            delay = 0.0  # time origin

        if is_piece_row(row):
            # delay accumulated from non-piece rows only
            nonextr_delay_up_to_piece.append(delay if time_started else 0.0)
            piece_ord_to_row_index.append(i)
        else:
            if time_started:
                delay += _to_float(row.get("t_s"), 0.0)
                if opt:
                    inj_end_time = delay

    # Extract pieces with (cycle, cycle_local, V, t_s, ord)
    pieces = []
    for ord_idx, row_i in enumerate(piece_ord_to_row_index):
        row = rows[row_i]
        cyc = _to_int(row.get("cycle"), 0)
        loc = _to_int(row.get("cycle_local"), 0)
        V = _to_float(row.get("V_mm3"), 0.0)
        t_s = _to_float(row.get("t_s"), 0.0)
        pieces.append({
            "ord": ord_idx,
            "row_i": row_i,
            "cycle": cyc,
            "cycle_local": loc,
            "V": V,
            "t_s": t_s,
        })

    if not pieces:
        raise RuntimeError("No piece rows detected in CSV. Check cmd/cycle/V_mm3/note fields.")

    # Group by cycle
    by_cycle = defaultdict(list)
    max_cycle = 0
    for p in pieces:
        by_cycle[p["cycle"]].append(p)
        max_cycle = max(max_cycle, p["cycle"])

    # Ensure cycle-local ordering inside each cycle
    cycles = []
    for c in range(1, max_cycle + 1):
        cyc = by_cycle.get(c, [])
        cyc.sort(key=lambda x: x["cycle_local"])
        cycles.append(cyc)

    def cum_inclusive(plist: List[dict]) -> List[float]:
        out = []
        acc = 0.0
        for p in plist:
            acc += p["t_s"]
            out.append(acc)
        return out

    final_timing: Dict[int, Dict[str, float]] = {}  # ord -> timing dict

    last_cycle_last_tout_abs = 0.0
    prev_out_log = None

    # convenient: nonextr_delay for each ord
    nonextr_by_ord = nonextr_delay_up_to_piece

    for cidx, cyc in enumerate(cycles, start=1):
        if not cyc:
            continue

        N = len(cyc)
        offs_incl = cum_inclusive(cyc)

        ords = [p["ord"] for p in cyc]
        first_delay = nonextr_by_ord[ords[0]]

        if cidx == 1:
            # abs t_out for cycle 1
            abs_t_out = []
            for j, off in enumerate(offs_incl):
                extra_delay = nonextr_by_ord[ords[j]] - first_delay
                if extra_delay < 0:
                    extra_delay = 0.0
                abs_t_out.append(first_delay + off + extra_delay)

            # cycle1 t_in: linear ramp over inj_end_time
            T_inj = max(0.0, inj_end_time)
            step = (T_inj / N) if N > 0 else 0.0
            base_t_in = [j * step for j in range(N)]

        else:
            # determine inter-cycle delay (non-extr added between cycles)
            prev_cyc = cycles[cidx - 2]
            if not prev_cyc:
                prev_last_ord = ords[0]  # fallback
                prev_last_delay = first_delay
            else:
                prev_last_ord = prev_cyc[-1]["ord"]
                prev_last_delay = nonextr_by_ord[prev_last_ord]

            inter_cycle_delay = first_delay - prev_last_delay
            if inter_cycle_delay < 0:
                inter_cycle_delay = 0.0

            abs_t_out = []
            for j, off in enumerate(offs_incl):
                extra_delay = nonextr_by_ord[ords[j]] - first_delay
                if extra_delay < 0:
                    extra_delay = 0.0
                abs_t_out.append(last_cycle_last_tout_abs + inter_cycle_delay + off + extra_delay)

            base_t_in = fifo_backfill_times(
                cyc,
                prev_out_log,
                last_cycle_last_tout_abs,
            )

        # write cycle timing
        for j, p in enumerate(cyc):
            t_in = base_t_in[j]
            t_out = abs_t_out[j]
            final_timing[p["ord"]] = {"t_in": t_in, "t_out": t_out, "tau_s": t_out - t_in}

        last_cycle_last_tout_abs = abs_t_out[-1]
        prev_out_log = deque({"V": cyc[j]["V"], "t_out": abs_t_out[j]} for j in range(N))

    # Apply to CSV rows
    out_rows = [dict(r) for r in rows]
    for ord_idx, row_i in enumerate(piece_ord_to_row_index):
        tinfo = final_timing.get(ord_idx)
        if not tinfo:
            continue
        out_rows[row_i]["t_in"] = fmt_fixed(tinfo["t_in"], 5)
        out_rows[row_i]["t_out"] = fmt_fixed(tinfo["t_out"], 5)
        out_rows[row_i]["tau_s"] = fmt_fixed(tinfo["tau_s"], 5)

    return out_rows, inj_end_time


# ---------------------------- G-code update by seg_idx ----------------------------

SEGIDX_RE = re.compile(r"(?:^|[;\s])seg_idx\s*=\s*(\d+)\s*$", re.IGNORECASE)

def extract_seg_idx(line: str) -> Optional[int]:
    s = line.rstrip("\n")
    m = SEGIDX_RE.search(s)
    if not m:
        return None
    return int(m.group(1))

def strip_trailing_seg_idx(line: str) -> str:
    """Remove the trailing 'seg_idx=N' token but keep other content unchanged."""
    s = line.rstrip("\n")
    # remove only the trailing seg_idx=... and possible spaces before it
    s2 = re.sub(r"\s*seg_idx\s*=\s*\d+\s*$", "", s, flags=re.IGNORECASE)
    return s2

def replace_token_value(line: str, letter: str, new_value_str: str) -> str:
    """
    Replace a token like 'F123.45' or 'E0.010' with new value, preserving other tokens.
    If token doesn't exist, append it before the first ';' comment (or at end).
    """
    s = line.rstrip("\n")
    # split code / comment
    code, sep, cmt = s.partition(";")

    # token regex in code part
    pat = re.compile(rf"(^|\s){re.escape(letter)}([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", re.IGNORECASE)

    if pat.search(code):
        def _repl(m):
            prefix = m.group(1)
            return f"{prefix}{letter}{new_value_str}"
        code2 = pat.sub(_repl, code, count=1)
    else:
        code2 = code.rstrip() + f" {letter}{new_value_str}"

    if sep:
        return code2.rstrip() + " ;" + cmt
    return code2.rstrip()

def compute_E_from_V(V_mm3: float, e_mode: str, d_filament: float) -> float:
    if abs(V_mm3) < 1e-12:
        return 0.0
    if e_mode == "mm3":
        return V_mm3
    if e_mode == "filament":
        A = area_circle(d_filament)
        if A <= 1e-12:
            return 0.0
        return V_mm3 / A
    raise ValueError(f"Unknown e_mode: {e_mode}")

def update_gcode_from_csv(
    gcode_ref_path: str,
    gcode_out_path: str,
    rows_by_segidx: Dict[int, dict],
    *,
    update_e: bool,
    e_mode: str,
    d_filament: float,
) -> Tuple[int, int]:
    """
    Returns (updated_lines, total_lines_written)
    """
    updated = 0
    total = 0

    with open(gcode_ref_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(gcode_out_path, "w", encoding="utf-8", newline="") as fout:

        for raw in fin:
            total += 1
            line = raw.rstrip("\n")
            seg_idx = extract_seg_idx(line)

            if seg_idx is None or seg_idx not in rows_by_segidx:
                fout.write(line + "\n")
                continue

            row = rows_by_segidx[seg_idx]
            cmd = (row.get("cmd") or "").strip().upper()
            note = (row.get("note") or "").lower()
            cyc = _to_int(row.get("cycle"), 0)

            # We keep original content (including comments) but update F/E in code section,
            # then re-append the original seg_idx token exactly once.
            base = strip_trailing_seg_idx(line).rstrip()

            # Decide whether to apply changes
            should_update_piece = is_piece_row(row)
            should_update_pre = is_pre_infill_row(row)

            if cmd == "G1" and (should_update_piece or should_update_pre):
                # Update F
                F = _to_float(row.get("F_mm_per_min"), None)
                if F is not None and F > 0:
                    base = replace_token_value(base, "F", fmt_fixed(F, 3))

                # Update E (optional)
                if update_e:
                    V = _to_float(row.get("V_mm3"), 0.0)
                    E_new = compute_E_from_V(V, e_mode=e_mode, d_filament=d_filament)
                    # Match old script: pieces used 6 decimals for E
                    base = replace_token_value(base, "E", fmt_fixed(E_new, 6))

                updated += 1

            # re-append seg_idx exactly once
            fout.write(base + f" seg_idx={seg_idx}\n")

    return updated, total

def build_segidx_motion_map_from_gcode(gcode_path: str) -> dict:
    """
    Parse reference segmented gcode and build:
      seg_idx -> {
        "has_xyz": bool,   # line contains any X/Y/Z token
        "d_xyz": float,    # 3D distance between consecutive positions (mm)
        "E": float or None # E value on that line (relative, as printed in segmented gcode)
      }

    Assumptions:
      - G1 SEG lines contain X/Y/Z (usually) and E and F
      - seg_idx=N exists at end of each line
      - We compute distance by tracking last known X/Y/Z state across file.
    """
    seg_map = {}

    last = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    with open(gcode_path, "r", encoding="utf-8", errors="ignore") as fin:
        for raw in fin:
            line = raw.rstrip("\n")
            si = extract_seg_idx(line)
            if si is None:
                continue

            stripped = strip_trailing_seg_idx(line).strip()
            if not stripped:
                continue

            parts = stripped.split()
            cmd = parts[0].upper()
            if cmd not in ("G0", "G1"):
                continue

            has_xyz = False
            cur = dict(last)
            E_val = None

            for tok in parts[1:]:
                if len(tok) < 2:
                    continue
                c = tok[0].upper()
                try:
                    v = float(tok[1:])
                except ValueError:
                    continue

                if c in ("X", "Y", "Z"):
                    has_xyz = True
                    cur[c] = v
                elif c == "E":
                    E_val = v

            dx = cur["X"] - last["X"]
            dy = cur["Y"] - last["Y"]
            dz = cur["Z"] - last["Z"]
            d_xyz = math.sqrt(dx*dx + dy*dy + dz*dz)

            seg_map[si] = {
                "has_xyz": has_xyz,
                "d_xyz": d_xyz,
                "E": E_val,
            }

            # update last state for next distance computation
            last = cur

    return seg_map

def recompute_piece_ts_from_F_using_gcode(rows, seg_motion_map, *, e_mode: str, d_filament: float):
    """
    Recompute t_s for piece/pre_infill rows using:
      - if G-code line has any X/Y/Z token:  t_s = 60*d_xyz/F
      - else (pure E-only):                 t_s = 60*|E|/F  (E derived from CSV V_mm3 + e_mode)

    seg_motion_map comes from build_segidx_motion_map_from_gcode().
    """
    out = [dict(r) for r in rows]

    for r in out:
        if not (is_piece_row(r) or is_pre_infill_row(r)):
            continue

        seg_idx = _to_int(r.get("seg_idx"), -1)
        if seg_idx < 0:
            continue

        F = _to_float(r.get("F_mm_per_min"), 0.0)
        if F <= 1e-12:
            continue

        m = seg_motion_map.get(seg_idx)

        # If we can detect XYZ from gcode -> follow your rule strictly
        if m is not None and m.get("has_xyz", False):
            d_xyz = float(m.get("d_xyz", 0.0))
            t_s_new = 60.0 * d_xyz / F if d_xyz > 1e-12 else 0.0
            r["t_s"] = fmt_fixed(t_s_new, 5)
            continue

        # Otherwise treat as E-only
        V = _to_float(r.get("V_mm3"), 0.0)
        if abs(V) <= 1e-12:
            r["t_s"] = fmt_fixed(0.0, 5)
            continue

        E = compute_E_from_V(V, e_mode=e_mode, d_filament=d_filament)
        t_s_new = 60.0 * abs(E) / F
        r["t_s"] = fmt_fixed(t_s_new, 5)

    return out

# ---------------------------- Main ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Rebuild FIFO timing from a modified v6.1-style CSV and regenerate a new G-code using an old segmented gcode as template."
    )
    ap.add_argument("--csv_in", required=True, help="Modified input CSV (same schema as v6.1 output)")
    ap.add_argument("--gcode_ref", required=True, help="Old segmented G-code (reference template with seg_idx=)")
    ap.add_argument("--csv_out", required=True, help="Output rebuilt CSV")
    ap.add_argument("--gcode_out", required=True, help="Output rebuilt G-code")

    ap.add_argument("--emode", choices=["filament", "mm3"], default="filament", help="E-axis unit mode for E-from-V conversion")
    ap.add_argument("--d_filament", type=float, default=15.55634919, help="Filament diameter (only used if --emode=filament)")
    ap.add_argument("--update_e", action="store_true", help="If set, update E in gcode from CSV V_mm3 (otherwise keep old E)")
    args = ap.parse_args()

    # Read CSV
    rows = read_modified_csv(args.csv_in)

    # Basic validation: seg_idx monotonic and unique
    segs = []
    for r in rows:
        si = _to_int(r.get("seg_idx"), -1)
        if si >= 0:
            segs.append(si)
    if not segs:
        raise RuntimeError("CSV has no seg_idx values.")
    if len(segs) != len(set(segs)):
        raise RuntimeError("CSV seg_idx is not unique. seg_idx must uniquely map rows to gcode lines.")

    # Rebuild FIFO timing
    seg_motion_map = build_segidx_motion_map_from_gcode(args.gcode_ref)

    rows2 = recompute_piece_ts_from_F_using_gcode(
        rows,
        seg_motion_map,
        e_mode=args.emode,
        d_filament=args.d_filament
    )

    rebuilt_rows, inj_end_time = rebuild_fifo_timing_from_csv(rows2)


    # Write rebuilt CSV (keep original columns order if possible; preserve extra columns too)
    # If extra columns exist, we keep them but still ensure core columns exist.
    all_fields = list(dict.fromkeys(list(rebuilt_rows[0].keys())))
    # Ensure core header columns appear first in the output
    for h in reversed(CSV_HEADER):
        if h in all_fields:
            all_fields.remove(h)
        all_fields.insert(0, h)

    with open(args.csv_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rebuilt_rows)

    # Build seg_idx -> row mapping for gcode update
    rows_by_segidx = {}
    for r in rebuilt_rows:
        si = _to_int(r.get("seg_idx"), -1)
        if si >= 0:
            rows_by_segidx[si] = r

    # Update gcode
    updated, total = update_gcode_from_csv(
        args.gcode_ref,
        args.gcode_out,
        rows_by_segidx,
        update_e=args.update_e,
        e_mode=args.emode,
        d_filament=args.d_filament,
    )

    print(f"✅ FIFO rebuilt from CSV. inj_end_time={fmt_fixed(inj_end_time, 5)}")
    print(f"✅ Wrote CSV : {args.csv_out}  (rows={len(rebuilt_rows)})")
    print(f"✅ Wrote G-code: {args.gcode_out}  (lines={total}, updated_lines={updated})")


if __name__ == "__main__":
    main()
