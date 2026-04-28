#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rebuild_csv_and_gcode_fifo.py
------------------------------------------------------------
Mode A: GLOBAL recompute (recommended).

Input:
  - segmented G-code (from v6.1) with piece lines containing "; SEG"
  - user may manually insert G0/G4/etc with comment "; MANUAL" to increase residence time

Output:
  1) rebuilt CSV (same columns as v6.1) with FIFO timing recomputed globally
  2) rebuilt segmented G-code (no cycle/timing shown), seg_idx re-numbered and aligned with CSV

Key points:
- FIFO timing logic matches your v6.1 Phase D/E.
- Cycle assignment uses boundary-crossing (tolerant to E rounding drift) to avoid "only 1 cycle".
- Manual detection: any line containing 'MANUAL' (case-insensitive) sets manual=True.
  CSV note will include 'manual' (and can co-exist with 'optimizable').

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
    if not line:
        return None
    raw = line.rstrip("\n")
    code_part, sep, comment_part = raw.partition(";")
    if not sep:
        return None
    m = SEGIDX_RE.search(comment_part)
    return int(m.group(1)) if m else None


def strip_segidx_from_comment(comment: str) -> str:
    if not comment:
        return ""
    c = SEGIDX_STRIP_RE.sub("", comment)
    c = re.sub(r"\s*\|\s*$", "", c).strip()
    return c


def ensure_comment_has_segidx(line: str, seg_idx: int) -> str:
    raw = line.rstrip("\n")
    code_part, sep, comment_part = raw.partition(";")
    code = code_part.rstrip()
    c = comment_part.strip() if sep else ""
    c_clean = strip_segidx_from_comment(c)
    new_c = (c_clean + " " if c_clean else "") + f"seg_idx={int(seg_idx)}"
    return f"{code} ; {new_c}\n"


# ---------------------------- Utilities ----------------------------

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


def distance_xy(p1, p2) -> float:
    return math.hypot(p2["X"] - p1["X"], p2["Y"] - p1["Y"])


def volume_from_E(dE: float, d_filament: float, e_mode: str) -> float:
    if abs(dE) < 1e-12:
        return 0.0
    if e_mode == "filament":
        return area_circle(d_filament) * dE
    if e_mode == "mm3":
        return dE
    raise ValueError(f"Unknown e_mode: {e_mode}")


def parse_g4_ts(parts):
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
    return "optimizable" in (line or "").lower()


def has_manual(line: str) -> bool:
    return "manual" in (line or "").lower()


def strip_segidx_anywhere(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"\s*\bseg_idx\s*=\s*\d+\b\s*", " ", text, flags=re.IGNORECASE)
    return " ".join(t.split()).strip()


def split_code_and_comment(line: str):
    if line is None:
        return "", ""
    if ";" in line:
        a, b = line.split(";", 1)
        return a.rstrip(), b.strip()
    return line.rstrip(), ""


# ---------------- FIFO helper: EXACTLY v6.1 ----------------

def fifo_backfill_times(pieces_in_cycle, prev_out_log, fallback_time):
    """
    EXACT v6.1 behavior:
    - Backfill current cycle t_in by volume-weighted sampling of previous cycle *t_out* parcels.
    - prev_out_log entries contain {"V": remaining_volume, "t_out": absolute_t_out}
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
):
    # -------- Phase A: Parse segmented gcode into expanded timeline --------
    expanded = []
    last = {"X": 0.0, "Y": 0.0, "Z": 0.0, "F": 1000.0}
    seen_first_seg_piece = False   # ✅ once we see the first "; SEG" piece, pre_infill ends


    with open(gcode_in, "r", encoding="utf-8", errors="ignore") as fin:
        for raw in fin:
            line0 = raw.rstrip("\n")
            opt = has_optimizable(line0)
            manu = has_manual(line0)

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
                })
                continue

            # other non-motion
            if cmd not in ("G0", "G1"):
                expanded.append({
                    "type": "raw",
                    "orig_line": orig_line,
                    "optimizable": opt,
                    "manual": manu,
                    "t_s": 0.0
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
            is_piece = ("SEG" in c_upper)
            if is_piece:
                seen_first_seg_piece = True  # ✅ lock: after first piece, no more implicit pre_infill

            is_pre_infill = ("PRE_INFILL" in c_upper)

            # ---------------------------------------------------------
            # ✅ NEW: implicit pre_infill detection
            # Before the first "; SEG" piece appears:
            #   G1 with E>0 and NO X/Y/Z => treat as pre_infill
            # Example: "G1 F800 E0.3804 ; optimizable"
            # ---------------------------------------------------------
            implicit_pre_infill = (
                (not seen_first_seg_piece) and
                (cmd == "G1") and has_E and (dE > 1e-12) and (d_xy <= 1e-12)
            )


            if is_piece:
                V = volume_from_E(dE, d_filament, e_mode)
                expanded.append({
                    "type": "piece",
                    "orig_line": orig_line,
                    "optimizable": opt,
                    "manual": manu,
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
                })

            last = move.copy()

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

    # -------- Phase D (EXACT v6.1): time origin + inj_end_time --------
    nonextr_delay_up_to = []
    delay = 0.0
    time_started = False
    inj_end_time = 0.0

    for it in expanded:
        # v6.1: time origin starts at first optimizable (any type)
        if (not time_started) and it.get("optimizable", False):
            time_started = True
            delay = 0.0
            inj_end_time = 0.0

        if it["type"] == "piece":
            nonextr_delay_up_to.append(delay if time_started else 0.0)
        else:
            if time_started:
                delay += it.get("t_s", 0.0)
                # v6.1: track "optimizable tail time"
                if it.get("optimizable", False):
                    inj_end_time = delay

    # -------- Phase E (EXACT v6.1): FIFO timing --------
    max_cycle = max(p["cycle"] for p in pieces)
    cycles = [[p for p in pieces if p["cycle"] == c] for c in range(1, max_cycle + 1)]

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

            # v6.1: cycle 1 t_in = linearly increasing using inj_end_time and N
            T_inj = max(0.0, float(inj_end_time))
            step = (T_inj / N) if N > 0 else 0.0
            base_t_in = [i * step for i in range(N)]

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

    def build_note(base: str, manual_flag: bool, opt_flag: bool) -> str:
        tags = []
        if manual_flag:
            tags.append("manual")
        if opt_flag:
            tags.append("optimizable")
        if tags:
            return ",".join(tags)
        return base

    rows = []
    seg_idx = 0

    SEGIDX_ANY_RE = re.compile(r"\bseg_idx\s*=\s*\d+\b", re.IGNORECASE)

    def strip_all_segidx(line: str) -> str:
        """Remove ALL seg_idx tokens from a full line (code/comment), keep everything else."""
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
        """
        Append seg_idx with consistent rules:
        - comment line (starts with ';'): append as ' ... seg_idx=N'
        - otherwise: ensure comment exists; append 'seg_idx=N' at end of comment
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
                add_csv_row(
                    seg_idx,
                    cmd="G1",
                    cycle=it["cycle"],
                    cycle_local=it["cycle_local"],
                    d_mm=it["d_len"],
                    F_mm_per_min=it["F"],
                    V_mm3=it["V"],
                    t_s=it["t_s"],
                    t_in=fmt_fixed(tinfo["t_in"], 5) if tinfo["t_in"] is not None else "",
                    t_out=fmt_fixed(tinfo["t_out"], 5) if tinfo["t_out"] is not None else "",
                    tau_s=fmt_fixed(tinfo["tau_s"], 5) if tinfo["tau_s"] is not None else "",
                    note=build_note("extrusion", manu, opt),
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
    ap = argparse.ArgumentParser(description="GLOBAL FIFO rebuild (v6.1) + MANUAL detection -> CSV note.")
    ap.add_argument("gcode", help="Input segmented gcode (must contain '; SEG')")
    ap.add_argument("--out", default="rebuilt_segments.csv", help="Output CSV filename")
    ap.add_argument("--gcode_out", default="rebuilt_segmented.gcode", help="Output rebuilt gcode filename")
    ap.add_argument("--vnozzle", type=float, required=True, help="V_nozzle [mm^3]")
    ap.add_argument("--d_filament", type=float, required=True, help="d_filament for E->V")
    ap.add_argument("--emode", choices=["filament", "mm3"], required=True, help="E mode")
    args = ap.parse_args()

    rebuild_global(
        gcode_in=args.gcode,
        csv_out=args.out,
        gcode_out=args.gcode_out,
        V_nozzle=args.vnozzle,
        d_filament=args.d_filament,
        e_mode=args.emode,
    )


if __name__ == "__main__":
    main()
