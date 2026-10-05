#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI:
python residence_optimizer_trueonly.py \
  --input house_withoutdoor_xy30h5_segmented.csv \
  --gcode_in house_withoutdoor_xy30h5_segmented.gcode \
  --gcode_out house_withoutdoor_xy30h5_optimized_final.gcode \
  --tau 940 \
  --buffer 0.05 \
  --vnozzle 723 \
  --dnozzle 0.58 \
  --decision_vcap_ratio 1.0 \
  --guard_vcap_ratio 1.0 \
  --Fmin 350 \
  --Fmax 1200 \
  --dFmax 0.25 \
  --eta 0.40 \
  --eta_up 1.10 \
  --eta_down 0.50 \
  --eta_min 0.05 \
  --eta_max 0.80 \
  --Pmax_ms 20000 \
  --dPmax_ms 2000 \
  --score_mode cvar95 \
  --cvar_alpha 0.95 \
  --max_iter 30 \
  --out_dir optimization_outputs_trueonly \
  --save_intermediate

Terminal:
python residence_optimizer_v9.py --input house_withoutdoor_xy30h5_segmented.csv --gcode_in house_withoutdoor_xy30h5_segmented.gcode --gcode_out house_withoutdoor_xy30h5_segmented_optimized.gcode --tau 940 --buffer 0.05 --vnozzle 723 --dnozzle 0.84 --decision_vcap_ratio 1.0 --guard_vcap_ratio 1.0 --Fmin 600 --Fmax 1200 --dFmax 0.25 --eta 0.40 --eta_up 1.10 --eta_down 0.50 --eta_min 0.05 --eta_max 0.08 --Pmax_ms 20000 --dPmax_ms 2000 --score_mode cvar95 --cvar_alpha 0.95 --max_iter 30 --out_dir optimization_outputs_trueonly --save_intermediate

"""

import csv
import math
import time
import argparse
import os
import sys
import re
import shutil
from typing import List, Dict, Tuple, Optional

import pandas as pd
from rebuild_csv_and_gcode_fifo_v1_1 import rebuild_global as true_fifo_rebuild

EPS = 1e-12


# ---------- Data structures ----------

class Row:
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
def _dt_bounds_for_dwell_ms(P_curr_ms: float, Pmax_ms: float, dPmax_ms: float) -> Tuple[float, float]:
    # returns (dt_min, dt_max) in seconds for dwell actuator
    # dwell only allows add time
    P_hi_iter = min(float(Pmax_ms), float(P_curr_ms) + float(dPmax_ms))
    dt_max_s = max(0.0, (P_hi_iter - float(P_curr_ms)) / 1000.0)
    return 0.0, dt_max_s

def _apply_dt_to_P_ms(P_curr_ms: float, dt_s: float, Pmax_ms: float, dPmax_ms: float) -> float:
    # only add
    dt_s = max(0.0, float(dt_s))
    P_hi_iter = min(float(Pmax_ms), float(P_curr_ms) + float(dPmax_ms))
    P_new = float(P_curr_ms) + 1000.0 * dt_s
    return float(min(P_new, P_hi_iter))


def _is_travel_row(rr: Row) -> bool:
    # travel: G0/G1, 非 extrusion/optimizable, 有距離
    if (rr.cmd or "").upper() not in ("G0","G1"):
        return False
    if rr.is_extrusion or rr.is_optimizable:
        return False
    return rr.d_mm > EPS

def r2(x: float) -> float:
    return round(float(x), 2)

def area_circle(d_mm: float) -> float:
    r = 0.5 * d_mm
    return math.pi * r * r

def read_segments_csv(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(Row(d))
    rows.sort(key=lambda x: x.seg_idx)
    return rows

def infer_tau_from_rows(rows: List[Row], fallback: float) -> List[float]:
    # baseline tau vector from CSV (tau_s column)
    out = []
    for r in rows:
        v = float(getattr(r, "tau_s", float("nan")))
        if v <= EPS:
            v = float("nan")
        out.append(v)
    # if nothing valid, caller will handle
    return out

def tau_stats(tau_list, rows):
    extr = []
    for i, r in enumerate(rows):
        if not r.is_extrusion:
            continue
        v = tau_list[i] if i < len(tau_list) else float("nan")
        if v is None or math.isnan(v):
            continue
        extr.append(v)
    if not extr:
        return (float("nan"), float("nan"), float("nan"))
    return (min(extr), max(extr), sum(extr)/len(extr))

def within_percentage(tau_list, rows, tau_star, buffer_ratio) -> float:
    lo = tau_star * (1.0 - buffer_ratio)
    hi = tau_star * (1.0 + buffer_ratio)
    extr = []
    for i, r in enumerate(rows):
        if not r.is_extrusion:
            continue
        v = tau_list[i] if i < len(tau_list) else float("nan")
        if v is None or math.isnan(v):
            continue
        extr.append(v)
    if not extr:
        return float("nan")
    ok = sum(1 for v in extr if (lo - EPS) <= v <= (hi + EPS))
    return 100.0 * ok / len(extr)

def global_error_abs_sum(tau_list, rows, tau_star: float) -> float:
    s = 0.0
    for i, r in enumerate(rows):
        if not r.is_extrusion:
            continue
        v = tau_list[i] if i < len(tau_list) else float("nan")
        if v is None or math.isnan(v):
            continue
        s += abs(v - tau_star)
    return s

def risk_metrics(tau_list, rows, tau_lo: float, tau_hi: float):
    dist = []
    for i, r in enumerate(rows):
        if not r.is_extrusion:
            continue
        t = tau_list[i] if i < len(tau_list) else float("nan")
        if t is None or math.isnan(t):
            continue
        if t < tau_lo:
            dist.append(tau_lo - t)
        elif t > tau_hi:
            dist.append(t - tau_hi)
        else:
            dist.append(0.0)

    if not dist:
        return 0, 0.0, 0.0, 0.0, 0.0

    violations_count = sum(1 for d in dist if d > EPS)
    v_sum = float(sum(dist))
    v_max = float(max(dist))

    dist_sorted = sorted(dist)
    n = len(dist_sorted)
    k95 = min(n - 1, int(math.floor(0.95 * (n - 1))))
    p95_abs = float(dist_sorted[k95])

    tail_start = int(math.floor(0.95 * n))
    tail = dist_sorted[tail_start:] if tail_start < n else [dist_sorted[-1]]
    cvar95 = float(sum(tail) / max(1, len(tail)))
    return violations_count, v_sum, v_max, p95_abs, cvar95


# ---------- Boundary maps (cycle-local windows) ----------

def compute_boundaries_and_prev_map(rows: List[Row], V_nozzle: float, d_nozzle: float) -> Tuple[List[int], List[int]]:
    A = area_circle(d_nozzle)
    boundaries: List[int] = []
    S = 0.0
    for i, r in enumerate(rows):
        if not (r.is_extrusion or r.is_optimizable):
            continue
        V = r.V_mm3 if r.V_mm3 > EPS else A * r.d_mm
        if S + V >= V_nozzle + EPS:
            boundaries.append(i)
            S = (S + V) - V_nozzle
        else:
            S += V

    N = len(rows)
    prev_of = [-1] * N
    bptr = 0
    last = -1
    for i in range(N):
        while bptr < len(boundaries) and boundaries[bptr] <= i:
            last = boundaries[bptr]
            bptr += 1
        prev_of[i] = last
    return boundaries, prev_of

def _is_piece(rr: Row) -> bool:
    return bool(rr.is_extrusion or rr.is_optimizable)

def _piece_volume_mm3(rr: Row, A_nozzle: float) -> float:
    # 用 CSV 的 V_mm3 為主，沒有就用 A*d_mm fallback
    if rr.V_mm3 > EPS:
        return float(rr.V_mm3)
    if rr.d_mm > EPS:
        return float(A_nozzle * rr.d_mm)
    return 0.0

def get_volume_capped_backward_window(
    rows: List[Row],
    target_i: int,
    V_cap: float,
    d_nozzle: float,
) -> List[int]:
    """上游 decision window：回溯累積體積到 V_cap（<=V_nozzle），可跨 boundary。"""
    N = len(rows)
    if target_i < 0 or target_i >= N:
        return []

    A = area_circle(d_nozzle)
    win = []
    cumV = 0.0

    j = target_i
    while j >= 0 and cumV < V_cap - EPS:
        rr = rows[j]
        if _is_piece(rr):
            V = _piece_volume_mm3(rr, A)
            # 若單一 piece 就超過 cap，也至少收進來
            if (cumV <= EPS) or (cumV + V <= V_cap + EPS):
                win.append(j)
                cumV += V
            else:
                # 已經接近 cap，停止（不硬塞超過）
                break
        j -= 1

    win.reverse()
    return win

def get_volume_capped_forward_window(
    rows: List[Row],
    target_i: int,
    V_cap: float,
    d_nozzle: float,
) -> List[int]:
    """下游 guard window：前推累積體積到 V_cap（<=V_nozzle），可跨 boundary。"""
    N = len(rows)
    if target_i < 0 or target_i >= N:
        return []

    A = area_circle(d_nozzle)
    win = []
    cumV = 0.0

    j = target_i
    while j < N and cumV < V_cap - EPS:
        rr = rows[j]
        if _is_piece(rr):
            V = _piece_volume_mm3(rr, A)
            if (cumV <= EPS) or (cumV + V <= V_cap + EPS):
                win.append(j)
                cumV += V
            else:
                break
        j += 1

    return win

# ---------- G-code patching by seg_idx (kept) ----------

SEGIDX_RE = re.compile(r"\bseg_idx\s*=\s*(?P<id>\d+)\b", re.IGNORECASE)

def _extract_seg_idx_from_comment(comment: str):
    if not comment:
        return None
    m = SEGIDX_RE.search(comment)
    return int(m.group("id")) if m else None

def _gcode_cmd_token(s: str) -> str:
    toks = s.strip().split()
    if not toks:
        return ""
    if toks[0].upper().startswith("N") and toks[0][1:].isdigit():
        toks = toks[1:]
        if not toks:
            return ""
    t = toks[0].upper()
    if t.startswith("G") and t[1:].isdigit():
        t = "G" + str(int(t[1:]))
    return t

def patch_gcode_by_segidx(
    gcode_template_path: str,
    gcode_out_path: str,
    rows: List[Row],
    F_vec: List[float],
    P_vec_ms: List[float],
) -> Tuple[int, int]:
    segidx_to_rowi = {r.seg_idx: i for i, r in enumerate(rows)}

    # --- helper: is travel gcode line (G0/G1 without E) ---
    def _is_travel_cmd(cmd: str, has_E: bool) -> bool:
        return (cmd in ("G0","G1")) and (not has_E)

    # PASS1 expected
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
            if seg_idx is None or seg_idx not in segidx_to_rowi:
                continue
            i = segidx_to_rowi[seg_idx]
            rr = rows[i]
            cmd = _gcode_cmd_token(s)
            parts = s.split()
            has_E = any(p.upper().startswith("E") for p in parts)
            has_P = any(p.upper().startswith("P") for p in parts)
            has_S = any(p.upper().startswith("S") for p in parts)

            if cmd == "G1" and has_E and ((rr.cmd or "").upper() == "G1") and (rr.is_extrusion or rr.is_optimizable):
                expected += 1
            elif cmd == "G4" and (has_P or has_S) and ((rr.cmd or "").upper() == "G4") and rr.is_optimizable:
                expected += 1
            elif _is_travel_cmd(cmd, has_E) and ((rr.cmd or "").upper() in ("G0","G1")):
                # travel rows 不一定是 optimizable，但你要能 patch F
                expected += 1

    # PASS2 patch
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
            has_S = any(p.upper().startswith("S") for p in parts)

            if seg_idx is not None and seg_idx in segidx_to_rowi:
                i = segidx_to_rowi[seg_idx]
                rr = rows[i]

                # G1+E -> patch F
                if cmd == "G1" and has_E and ((rr.cmd or "").upper() == "G1") and (rr.is_extrusion or rr.is_optimizable):
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
                    fout.write(f"{new_code} ; {c}\n" if sep else f"{new_code}\n")
                    patched += 1
                    continue

                # G4 -> patch P/S
                if cmd == "G4" and (has_P or has_S) and ((rr.cmd or "").upper() == "G4") and rr.is_optimizable:
                    P_new = int(round(float(P_vec_ms[i])))
                    new_parts = []
                    changed = False
                    for p in parts:
                        up = p.upper()
                        if up.startswith("P"):
                            new_parts.append(f"P{P_new}")
                            changed = True
                        elif up.startswith("S"):
                            new_parts.append(f"S{round(P_new/1000.0, 6)}")
                            changed = True
                        else:
                            new_parts.append(p)
                    if not changed:
                        new_parts.append(f"P{P_new}")
                    new_code = " ".join(new_parts)
                    fout.write(f"{new_code} ; {c}\n" if sep else f"{new_code}\n")
                    patched += 1
                    continue
                    
                if _is_travel_cmd(cmd, has_E) and ((rr.cmd or "").upper() in ("G0","G1")):
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
                    fout.write(f"{new_code} ; {c}\n" if sep else f"{new_code}\n")
                    patched += 1
                    continue
                                    
            fout.write(line)

    return patched, expected


# ---------- TRUE physical FIFO wrapper (kept, simplified) ----------

def simulate_true_fifo(
    gcode_template_path: str,
    rows: List[Row],
    F_vec: List[float],
    P_vec_ms: List[float],
    V_nozzle: float,
    d_nozzle: float,
    iteration: int,
    out_root: str,
    save_intermediate: bool = False,
) -> Tuple[bool, List[float], List[float], List[float], str, str, str]:

    tmp_dir = os.path.join(out_root, "true_fifo_results", f"iter_{iteration:04d}")
    os.makedirs(tmp_dir, exist_ok=True)

    tmp_gcode = os.path.join(tmp_dir, "temp_in.gcode")
    tmp_csv   = os.path.join(tmp_dir, "temp_out.csv")
    tmp_outgcode = os.path.join(tmp_dir, "temp_out.gcode")

    # patch gcode
    try:
        patched, expected = patch_gcode_by_segidx(
            gcode_template_path=gcode_template_path,
            gcode_out_path=tmp_gcode,
            rows=rows,
            F_vec=F_vec,
            P_vec_ms=P_vec_ms,
        )
        if patched < max(1, int(0.98 * expected)):
            print(f"⛔ TRUE patch coverage too low: {patched}/{expected}.")
            return False, [float("nan")]*len(rows), [float("nan")]*len(rows), [float("nan")]*len(rows), tmp_csv, tmp_gcode
    except Exception as e:
        print(f"⛔ TRUE patch failed: {e}")
        return False, [float("nan")]*len(rows), [float("nan")]*len(rows), [float("nan")]*len(rows), tmp_csv, tmp_gcode

    # physical rebuild
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
        print(f"⛔ TRUE physical engine failed: {e}")
        return False, [float("nan")]*len(rows), [float("nan")]*len(rows), [float("nan")]*len(rows), tmp_csv, tmp_gcode

    if not os.path.exists(tmp_csv):
        print("⛔ TRUE physical engine produced no CSV.")
        return False, [float("nan")]*len(rows), [float("nan")]*len(rows), [float("nan")]*len(rows), tmp_csv, tmp_gcode

    # align by seg_idx
    try:
        df = pd.read_csv(tmp_csv)
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

        hit = 0
        for i, r in enumerate(rows):
            if r.seg_idx in m:
                t_in_list[i], t_out_list[i], tau_list[i] = m[r.seg_idx]
                hit += 1

        coverage = hit / max(1, len(rows))
        if coverage < 0.98:
            print(f"⛔ TRUE CSV seg_idx coverage too low: {hit}/{len(rows)} ({coverage:.3f}).")
            return False, [float("nan")]*len(rows), [float("nan")]*len(rows), [float("nan")]*len(rows), tmp_csv, tmp_gcode

        if save_intermediate:
            dst = os.path.join(out_root, f"true_fifo_iter_{iteration:04d}")
            if not os.path.exists(dst):
                shutil.copytree(tmp_dir, dst)

        return True, t_in_list, t_out_list, tau_list, tmp_csv, tmp_gcode, tmp_outgcode

    except Exception as e:
        print(f"⛔ TRUE read/align failed: {e}")
        return False, [float("nan")]*len(rows), [float("nan")]*len(rows), [float("nan")]*len(rows), tmp_csv, tmp_gcode, tmp_outgcode

# ---------- Selection + Update hooks (YOU will implement) ----------

def _dist_to_window(t: float, tau_lo: float, tau_hi: float) -> float:
    if t < tau_lo: return tau_lo - t
    if t > tau_hi: return t - tau_hi
    return 0.0

def _guard_score_from_dists(dists: List[float], mode: str, alpha: float = 0.95) -> float:
    """Aggregate guard-window distances into a single score."""
    if not dists:
        return 0.0

    if mode == "max":
        return float(max(dists))

    if mode == "sum":
        return float(sum(dists))

    if mode == "cvar95":
        # general CVaR_alpha (default 0.95)
        a = float(alpha)
        a = min(max(a, 0.0), 0.999999)  # avoid alpha=1 edge
        ds = sorted(dists)
        n = len(ds)
        tail_start = int(math.floor(a * n))
        if tail_start >= n:
            tail_start = n - 1
        tail = ds[tail_start:]
        return float(sum(tail) / max(1, len(tail)))

    # fallback
    return float(max(dists))

def select_worst_target(
    rows: List[Row],
    tau: List[float],
    boundaries: List[int],   # 這個可以先留著不用
    tau_lo: float,
    tau_hi: float,
    V_guard: float,
    d_nozzle: float,
    score_mode: str = "max",
    cvar_alpha: float = 0.95,
) -> Tuple[Optional[int], Optional[int], float]:
    """
    用 guard window 的 aggregated risk 選 i_star：
      score(i) = agg(dist in guard window) where agg ∈ {max, sum, cvar95}
    回傳: (worst_cycle_id(None), i_star, worst_dist_of_i_star)
    """
    best_i = None
    best_score = -1.0
    best_worst_dist = 0.0

    for i, r in enumerate(rows):
        if not r.is_extrusion:
            continue
        t = tau[i] if i < len(tau) else float("nan")
        if t is None or math.isnan(t):
            continue

        # 只考慮「本身有違規」的點（更快、更符合 worst-first）
        d0 = _dist_to_window(t, tau_lo, tau_hi)
        if d0 <= EPS:
            continue

        guard = get_volume_capped_forward_window(rows, i, V_guard, d_nozzle)
        if not guard:
            continue

        dists = []
        for j in guard:
            if not rows[j].is_extrusion:
                continue
            tj = tau[j] if j < len(tau) else float("nan")
            if tj is None or math.isnan(tj):
                continue
            dists.append(_dist_to_window(tj, tau_lo, tau_hi))

        score = _guard_score_from_dists(dists, mode=score_mode, alpha=cvar_alpha)

        # tie-break：同 score 時，優先選本身違規更大的點（更穩）
        if (score > best_score + EPS) or (abs(score - best_score) <= EPS and d0 > best_worst_dist + EPS):
            best_score = score
            best_i = i
            best_worst_dist = d0

    return None, best_i, float(best_worst_dist)

def _signed_dist(t: float, tau_lo: float, tau_hi: float) -> float:
    """
    Return signed distance to window:
      + : need increase tau (tau too small)  -> add time (slow down)
      - : need decrease tau (tau too large)  -> remove time (speed up)
      0 : inside
    """
    if t < tau_lo:
        return float(tau_lo - t)     # need +ΔT
    if t > tau_hi:
        return -float(t - tau_hi)   # need -ΔT
    return 0.0


def _ts_from_F_and_dmm(F_mm_per_min: float, d_mm: float) -> float:
    """Extrusion segment time from distance and feedrate."""
    if F_mm_per_min <= EPS or d_mm <= EPS:
        return 0.0
    return 60.0 * float(d_mm) / float(F_mm_per_min)


def _dt_bounds_for_extrusion(
    F_curr: float,
    d_mm: float,
    Fmin: float,
    Fmax: float,
    dFmax_ratio: float,
) -> Tuple[float, float, float]:
    """
    For an extrusion segment with current F and distance d_mm:
      returns (t_curr, dt_min, dt_max)
    where:
      dt_max >= 0 is max time increase allowed this iter (slow down)
      dt_min <= 0 is max time decrease allowed this iter (speed up)
    considering BOTH absolute bounds and per-iter step limit (dFmax_ratio).
    """
    t_curr = _ts_from_F_and_dmm(F_curr, d_mm)
    if t_curr <= EPS:
        return 0.0, 0.0, 0.0

    # per-iter F bounds (trust-region in F space)
    F_lo_iter = max(float(Fmin), float(F_curr) * (1.0 - float(dFmax_ratio)))
    F_hi_iter = min(float(Fmax), float(F_curr) * (1.0 + float(dFmax_ratio)))

    # slow down => F decreases => time increases (use F_lo_iter)
    t_slow = _ts_from_F_and_dmm(F_lo_iter, d_mm)
    dt_max = max(0.0, t_slow - t_curr)

    # speed up => F increases => time decreases (use F_hi_iter)
    t_fast = _ts_from_F_and_dmm(F_hi_iter, d_mm)
    dt_min = min(0.0, t_fast - t_curr)  # negative (or 0)

    return t_curr, dt_min, dt_max


def _apply_dt_to_F(
    F_curr: float,
    d_mm: float,
    dt: float,
    Fmin: float,
    Fmax: float,
    dFmax_ratio: float,
) -> float:
    """
    Apply dt in time domain, then project back to feasible F (bounds + per-iter step limit).
    """
    t_curr = _ts_from_F_and_dmm(F_curr, d_mm)
    if t_curr <= EPS:
        return float(F_curr)

    t_new = t_curr + float(dt)
    t_new = max(EPS, t_new)

    F_raw = 60.0 * float(d_mm) / t_new

    # per-iter bounds
    F_lo_iter = max(float(Fmin), float(F_curr) * (1.0 - float(dFmax_ratio)))
    F_hi_iter = min(float(Fmax), float(F_curr) * (1.0 + float(dFmax_ratio)))

    F_proj = min(max(float(F_raw), F_lo_iter), F_hi_iter)
    return float(F_proj)

def propose_update_prox(
    rows: List[Row],
    F_curr: List[float],
    P_curr_ms: List[float],
    tau_curr: List[float],
    i_star: int,
    worst_dist: float,
    tau_star: float,
    buffer_ratio: float,
    V_decision: float,
    V_guard: float,
    d_nozzle: float,
    Fmin: float,
    Fmax: float,
    dFmax_ratio: float,
    eta: float,
    Pmax_ms: float,
    dPmax_ms: float,
) -> Tuple[List[float], List[float]]:
    """
    Actuator order A:
      1) dwell (G4 optimizable)    : only +ΔT
      2) extrusion feedrate (F)    : ±ΔT
      3) travel feedrate (G0/G1 noE): ±ΔT

    Scheme C = water-filling (nearest-to-i_star backward first)
    Saturation: each stage clips by its capacity; leftover passes to next stage.
    """

    F_next = F_curr[:]
    P_next = P_curr_ms[:]

    tau_lo = tau_star * (1.0 - buffer_ratio)
    tau_hi = tau_star * (1.0 + buffer_ratio)

    t_star = tau_curr[i_star] if i_star < len(tau_curr) else float("nan")
    if t_star is None or math.isnan(t_star):
        return F_next, P_next

    sd = _signed_dist(float(t_star), tau_lo, tau_hi)
    if abs(sd) <= EPS:
        return F_next, P_next

    dT_des = float(eta) * float(sd)  # seconds

    # decision window (volume-based pieces) gives you an index span upstream
    decision = get_volume_capped_backward_window(rows, i_star, V_decision, d_nozzle)
    if not decision:
        return F_next, P_next

    # We'll also consider travel rows in the index span of decision window
    span_lo = decision[0]
    span_hi = i_star

    # ==========================================================
    # Stage 1) DWELL (only positive)
    # ==========================================================
    dwell_idxs = []
    total_plus_P = 0.0  # seconds

    if dT_des > EPS:
        for j in range(span_lo, span_hi + 1):
            rr = rows[j]
            if (rr.cmd or "").upper() == "G4" and rr.is_optimizable:
                dt_min, dt_max = _dt_bounds_for_dwell_ms(P_curr_ms[j], Pmax_ms, dPmax_ms)
                if dt_max > EPS:
                    dwell_idxs.append(j)
                    total_plus_P += dt_max

    dT_P = 0.0
    if dT_des > EPS and total_plus_P > EPS:
        dT_P = min(dT_des, total_plus_P)  # only add

        # water-filling: nearest first (backward from i_star)
        ordered = sorted(dwell_idxs, reverse=True)
        rem = dT_P
        for j in ordered:
            if rem <= EPS:
                break
            dt_min, dt_max = _dt_bounds_for_dwell_ms(P_curr_ms[j], Pmax_ms, dPmax_ms)
            take = min(dt_max, rem)
            if take > EPS:
                P_next[j] = _apply_dt_to_P_ms(P_curr_ms[j], take, Pmax_ms, dPmax_ms)
                rem -= take

    dT_rem = dT_des - dT_P  # seconds (can be + or -)

    # ==========================================================
    # Stage 2) EXTRUSION F (±)
    # ==========================================================
    extr_idxs = []
    total_plus_F = 0.0
    total_minus_F = 0.0  # negative

    for j in decision:
        rr = rows[j]
        if not rr.is_extrusion:
            continue
        if rr.d_mm <= EPS:
            continue
        extr_idxs.append(j)
        _, dt_min, dt_max = _dt_bounds_for_extrusion(F_curr[j], rr.d_mm, Fmin, Fmax, dFmax_ratio)
        total_plus_F += max(0.0, dt_max)
        total_minus_F += min(0.0, dt_min)

    dT_F = 0.0
    if extr_idxs and (abs(dT_rem) > EPS):
        # clip by F capacity
        dT_F = min(max(dT_rem, total_minus_F), total_plus_F)

        ordered = sorted(extr_idxs, reverse=True)
        rem = dT_F
        dt_alloc = {j: 0.0 for j in ordered}

        if rem > EPS:
            # add time => slow down
            for j in ordered:
                if rem <= EPS:
                    break
                rr = rows[j]
                _, _, dt_max = _dt_bounds_for_extrusion(F_curr[j], rr.d_mm, Fmin, Fmax, dFmax_ratio)
                cap = max(0.0, dt_max)
                if cap <= EPS:
                    continue
                take = min(cap, rem)
                dt_alloc[j] = take
                rem -= take

        elif rem < -EPS:
            # remove time => speed up
            for j in ordered:
                if rem >= -EPS:
                    break
                rr = rows[j]
                _, dt_min, _ = _dt_bounds_for_extrusion(F_curr[j], rr.d_mm, Fmin, Fmax, dFmax_ratio)
                cap = min(0.0, dt_min)  # negative
                if cap >= -EPS:
                    continue
                take = max(cap, rem)
                dt_alloc[j] = take
                rem -= take

        for j, dt in dt_alloc.items():
            if abs(dt) <= EPS:
                continue
            rr = rows[j]
            F_next[j] = _apply_dt_to_F(F_curr[j], rr.d_mm, dt, Fmin, Fmax, dFmax_ratio)

    dT_rem2 = dT_rem - dT_F  # leftover after dwell+extrusion-F

    # ==========================================================
    # Stage 3) TRAVEL F (±)  (最後兜底)
    # ==========================================================
    travel_idxs = []
    total_plus_T = 0.0
    total_minus_T = 0.0

    if abs(dT_rem2) > EPS:
        for j in range(span_lo, span_hi + 1):
            rr = rows[j]
            if not _is_travel_row(rr):
                continue
            if rr.d_mm <= EPS:
                continue
            travel_idxs.append(j)
            _, dt_min, dt_max = _dt_bounds_for_extrusion(F_curr[j], rr.d_mm, Fmin, Fmax, dFmax_ratio)
            total_plus_T += max(0.0, dt_max)
            total_minus_T += min(0.0, dt_min)

    if travel_idxs and abs(dT_rem2) > EPS:
        dT_T = min(max(dT_rem2, total_minus_T), total_plus_T)

        ordered = sorted(travel_idxs, reverse=True)
        rem = dT_T
        dt_alloc = {j: 0.0 for j in ordered}

        if rem > EPS:
            for j in ordered:
                if rem <= EPS:
                    break
                rr = rows[j]
                _, _, dt_max = _dt_bounds_for_extrusion(F_curr[j], rr.d_mm, Fmin, Fmax, dFmax_ratio)
                cap = max(0.0, dt_max)
                if cap <= EPS:
                    continue
                take = min(cap, rem)
                dt_alloc[j] = take
                rem -= take

        elif rem < -EPS:
            for j in ordered:
                if rem >= -EPS:
                    break
                rr = rows[j]
                _, dt_min, _ = _dt_bounds_for_extrusion(F_curr[j], rr.d_mm, Fmin, Fmax, dFmax_ratio)
                cap = min(0.0, dt_min)
                if cap >= -EPS:
                    continue
                take = max(cap, rem)
                dt_alloc[j] = take
                rem -= take

        for j, dt in dt_alloc.items():
            if abs(dt) <= EPS:
                continue
            rr = rows[j]
            F_next[j] = _apply_dt_to_F(F_curr[j], rr.d_mm, dt, Fmin, Fmax, dFmax_ratio)

    return F_next, P_next


def better_by_feasibility_first(
    cand: Dict[str, float],
    best: Dict[str, float],
) -> bool:
    """
    Lexicographic:
      1) v_cnt lower
      2) v_max lower
      3) cvar95 lower
      4) within higher
      5) ge lower
    """
    c = (
        -int(cand["v_cnt"]),
        -float(cand["v_max"]),
        -float(cand["cvar95"]),
        float(cand["within"]),
        -float(cand["ge"]),
    )
    b = (
        -int(best["v_cnt"]),
        -float(best["v_max"]),
        -float(best["cvar95"]),
        float(best["within"]),
        -float(best["ge"]),
    )
    return c > b


# ---------- Writers (kept minimal) ----------

def write_summary_csv(path: str, history: List[Dict[str,float]], cli_used: str = ""):
    header = ["iteration","ge","tau_min","tau_max","tau_mean","within","v_cnt","v_max","p95","cvar95","eta"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for rec in history:
            w.writerow([
                int(rec.get("it", -1)),
                r2(rec.get("ge", float("nan"))),
                r2(rec.get("tau_min", float("nan"))),
                r2(rec.get("tau_max", float("nan"))),
                r2(rec.get("tau_mean", float("nan"))),
                r2(rec.get("within", float("nan"))),
                int(rec.get("v_cnt", 0)),
                r2(rec.get("v_max", float("nan"))),
                r2(rec.get("p95", float("nan"))),
                r2(rec.get("cvar95", float("nan"))),
                r2(rec.get("eta", float("nan"))),
            ])
        if cli_used:
            w.writerow([])
            w.writerow(["CLI", cli_used])

def _win_info(win: List[int]) -> str:
    """Format: len [lo:hi] using row indices (not seg_idx)."""
    if not win:
        return "0 [-:-]"
    return f"{len(win)} [{win[0]}:{win[-1]}]"

def safe_copy(src: str, dst: str) -> bool:
    """Copy file if exists; create dst dir; overwrite allowed."""
    try:
        if not src or (not os.path.exists(src)):
            return False
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"⚠️ copy failed: {src} -> {dst} | {e}")
        return False

            
# ---------- Main (TRUE-only, single TRUE per iter) ----------
def main():
    ap = argparse.ArgumentParser(description="Residence-time optimizer (TRUE-only clean skeleton).")

    ap.add_argument("--input", required=True, help="Input segmented CSV (rebuilt/with seg_idx).")
    ap.add_argument("--gcode_in", required=True, help="Segmented G-code template with seg_idx tags.")
    ap.add_argument("--gcode_out", required=True, help="Final patched G-code output path.")

    ap.add_argument("--tau", type=float, required=True, help="Target tau* (s).")
    ap.add_argument("--buffer", type=float, default=0.05, help="Allowed window ratio ±.")
    ap.add_argument("--vnozzle", type=float, required=True, help="V_nozzle (mm^3).")
    ap.add_argument("--dnozzle", type=float, required=True, help="Nozzle diameter (mm).")

    ap.add_argument("--decision_vcap_ratio", type=float, default=1.0, help="Decision window cap = ratio * V_nozzle")
    ap.add_argument("--guard_vcap_ratio", type=float, default=1.0, help="Guard window cap = ratio * V_nozzle")

    ap.add_argument("--Fmin", type=float, default=350.0)
    ap.add_argument("--Fmax", type=float, default=1200.0)
    ap.add_argument("--dFmax", type=float, default=0.25, help="Per-iter relative step limit for F (TRUE-only recommend <=0.25).")

    ap.add_argument("--eta", type=float, default=0.40, help="Prox step scale (TRUE-only).")
    ap.add_argument("--eta_up", type=float, default=1.10)
    ap.add_argument("--eta_down", type=float, default=0.50)
    ap.add_argument("--eta_min", type=float, default=0.05)
    ap.add_argument("--eta_max", type=float, default=0.80)

    ap.add_argument("--max_iter", type=int, default=30)
    ap.add_argument("--out_dir", default="optimization_outputs_trueonly")
    ap.add_argument("--save_intermediate", action="store_true")
    
    ap.add_argument("--Pmax_ms", type=float, default=20000.0, help="Max dwell per G4 (ms).")
    ap.add_argument("--dPmax_ms", type=float, default=2000.0, help="Max dwell increase per iter per G4 (ms).")

    # -------- Score Mode --------
    ap.add_argument(
    "--score_mode",
    choices=["max", "sum", "cvar95"],
    default="max",
    help="Guard-window risk aggregation for selecting i_star: max | sum | cvar95"
    )
    ap.add_argument(
        "--cvar_alpha",
        type=float,
        default=0.95,
        help="Alpha for CVaR in guard score (used when --score_mode=cvar95). Typical: 0.95"
    )
    # ----------------------------

    args = ap.parse_args()
    cli_string = " ".join(sys.argv)
    os.makedirs(args.out_dir, exist_ok=True)

    V_decision = args.vnozzle * float(args.decision_vcap_ratio)
    V_guard    = args.vnozzle * float(args.guard_vcap_ratio)


    rows = read_segments_csv(args.input)
    if not rows:
        print("⛔ Empty CSV.")
        return

    boundaries, prev_boundary_of = compute_boundaries_and_prev_map(rows, args.vnozzle, args.dnozzle)

    # init F/P from CSV
    F_curr = [r.F if r.F > EPS else args.Fmin for r in rows]
    P_curr_ms = [0.0] * len(rows)
    for i, r in enumerate(rows):
        if (r.cmd or "").upper() == "G4" and r.is_optimizable:
            P_curr_ms[i] = max(0.0, float(r.t_s) * 1000.0)

    # Baseline tau from CSV (iter0)
    tau0 = infer_tau_from_rows(rows, fallback=args.tau)
    if all((t is None or math.isnan(t)) for t in tau0):
        print("⚠️ tau_s in CSV is empty/invalid. Running TRUE once to initialize tau0...")
        ok, _, _, tau_init, _, _ = simulate_true_fifo(
            gcode_template_path=args.gcode_in,
            rows=rows,
            F_vec=F_curr,
            P_vec_ms=P_curr_ms,
            V_nozzle=args.vnozzle,
            d_nozzle=args.dnozzle,
            iteration=0,
            out_root=args.out_dir,
            save_intermediate=args.save_intermediate,
        )
        if not ok:
            print("⛔ TRUE init failed; cannot start optimization.")
            return
        tau0 = tau_init

    tau_lo = args.tau * (1.0 - args.buffer)
    tau_hi = args.tau * (1.0 + args.buffer)

    ge0 = global_error_abs_sum(tau0, rows, args.tau)
    tau_min0, tau_max0, tau_mean0 = tau_stats(tau0, rows)
    within0 = within_percentage(tau0, rows, args.tau, args.buffer)
    v_cnt0, v_sum0, v_max0, p95_0, cvar95_0 = risk_metrics(tau0, rows, tau_lo, tau_hi)
    
    # debug: window size
    dec, grd = [], []
    i0 = next((k for k,r in enumerate(rows) if r.is_extrusion), None)
    if i0 is not None:
        dec = get_volume_capped_backward_window(rows, i0, V_decision, args.dnozzle)
        grd = get_volume_capped_forward_window(rows, i0, V_guard, args.dnozzle)
        print(f"windows: decision_len={len(dec)} guard_len={len(grd)} (Vcap ratio applied)")

    print(f"[Iter 000] CSV_INIT | τ_min={r2(tau_min0)}, τ_max={r2(tau_max0)}, τ_mean={r2(tau_mean0)} | "
          f"ge={r2(ge0)} | within={r2(within0)}% | v_cnt={v_cnt0} | v_max={r2(v_max0)} | cvar95={r2(cvar95_0)}")
    #
    best = {
        "it": 0,
        "F": F_curr[:],
        "P": P_curr_ms[:],
        "tau": tau0[:],
        "ge": ge0,
        "tau_min": tau_min0,
        "tau_max": tau_max0,
        "tau_mean": tau_mean0,
        "within": within0,
        "v_cnt": int(v_cnt0),
        "v_max": float(v_max0),
        "p95": float(p95_0),
        "cvar95": float(cvar95_0),
        "eta": float(args.eta),
    }
    history = [best.copy()]

    # ---- Fixed artifact paths (always overwritten) ----
    BEST_CSV   = os.path.join(args.out_dir, "best.csv")
    BEST_GCODE = os.path.join(args.out_dir, "best.gcode")
    FINAL_CSV   = os.path.join(args.out_dir, "final.csv")
    FINAL_GCODE = os.path.join(args.out_dir, "final.gcode")

    # Track last successful TRUE evaluation artifacts (final = last successful eval)
    last_ok_csv = None
    last_ok_gcode = None

    def _finalize_outputs(tag: str):
        # 1) summary
        summary_path = os.path.join(args.out_dir, "summary.csv")
        try:
            write_summary_csv(summary_path, history, cli_string)
            print(f"🧾 Summary({tag}) -> {summary_path}")
        except Exception as e:
            print(f"⚠️ Summary write failed({tag}): {e}")

        # 2) best gcode
        try:
            patch_out = args.gcode_out
            patched, expected = patch_gcode_by_segidx(
                gcode_template_path=args.gcode_in,
                gcode_out_path=patch_out,
                rows=rows,
                F_vec=best["F"],
                P_vec_ms=best["P"],
            )
            print(f"✅ Patched BEST gcode({tag}): {patched}/{expected} -> {patch_out}")
            # ---- Ensure BEST artifacts exist at end ----
            if not os.path.exists(BEST_CSV) and os.path.exists(FINAL_CSV):
                safe_copy(FINAL_CSV, BEST_CSV)
            if not os.path.exists(BEST_GCODE) and os.path.exists(FINAL_GCODE):
                safe_copy(FINAL_GCODE, BEST_GCODE)
            
        except Exception as e:
            print(f"⚠️ BEST gcode patch failed({tag}): {e}")


    # TRUE-only pipeline: cache last evaluated tau (so 1 TRUE per iter)
    tau_curr = tau0[:]  # NOTE: after iter1, tau_curr will be last tau_new

    eta = float(args.eta)

    interrupted = False
    try:
        for it in range(1, args.max_iter + 1):
            t0 = time.time()
            #
            worst_cycle, i_star, worst_dist = select_worst_target(
                rows, tau_curr, boundaries, tau_lo, tau_hi,
                V_guard=V_guard, d_nozzle=args.dnozzle,
                score_mode=args.score_mode, cvar_alpha=args.cvar_alpha,
            )
            if i_star is None or worst_dist <= EPS:
                print(f"[Iter {it:03d}] No violations found (or no valid target). Stop.")
                break

            # --- NEW: per-iter window info for i_star ---
            decision_win = get_volume_capped_backward_window(rows, i_star, V_decision, args.dnozzle)
            guard_win    = get_volume_capped_forward_window(rows, i_star, V_guard, args.dnozzle)

            print(
                f"[Iter {it:03d}] target seg_idx={rows[i_star].seg_idx} | worst_dist={r2(worst_dist)} | "
                f"decision_len={_win_info(decision_win)} , guard_len={_win_info(guard_win)}"
            )

            F_next, P_next_ms = propose_update_prox(
                rows=rows,
                F_curr=F_curr,
                P_curr_ms=P_curr_ms,
                tau_curr=tau_curr,
                i_star=i_star,
                worst_dist=worst_dist,
                tau_star=args.tau,
                buffer_ratio=args.buffer,
                V_decision=V_decision,
                V_guard=V_guard,
                d_nozzle=args.dnozzle,
                Fmin=args.Fmin,
                Fmax=args.Fmax,
                dFmax_ratio=args.dFmax,
                eta=eta,
                Pmax_ms=args.Pmax_ms,
                dPmax_ms=args.dPmax_ms,
            )

            ok, _, _, tau_new, tmp_csv, tmp_in_gcode, tmp_out_gcode = simulate_true_fifo(
                gcode_template_path=args.gcode_in,
                rows=rows,
                F_vec=F_next,
                P_vec_ms=P_next_ms,
                V_nozzle=args.vnozzle,
                d_nozzle=args.dnozzle,
                iteration=it,
                out_root=args.out_dir,
                save_intermediate=args.save_intermediate,
            )

            if not ok:
                print(f"⚠️ [Iter {it:03d}] TRUE failed. Shrink eta and retry next iter.")
                eta = max(args.eta_min, eta * args.eta_down)
                continue

            # ✅ 就放這裡：TRUE 成功後立刻把「本次評估」存成 final
            last_ok_csv = tmp_csv
            last_ok_gcode = tmp_out_gcode
            safe_copy(last_ok_csv, FINAL_CSV)
            safe_copy(last_ok_gcode, FINAL_GCODE)

            ge = global_error_abs_sum(tau_new, rows, args.tau)
            tau_min, tau_max, tau_mean = tau_stats(tau_new, rows)
            within = within_percentage(tau_new, rows, args.tau, args.buffer)
            v_cnt, v_sum, v_max, p95_abs, cvar95 = risk_metrics(tau_new, rows, tau_lo, tau_hi)

            cand = {
                "it": it, "F": F_next[:], "P": P_next_ms[:], "tau": tau_new[:],
                "ge": ge, "within": within, "v_cnt": int(v_cnt), "v_max": float(v_max),
                "p95": float(p95_abs), "cvar95": float(cvar95),
                "eta": eta,
            }

            if better_by_feasibility_first(cand, best):
                F_curr = F_next
                P_curr_ms = P_next_ms
                tau_curr = tau_new
                best = cand
                # ---- BEST artifacts: persist best TRUE CSV + TRUE gcode ----
                # tmp_csv/tmp_out_gcode correspond to the candidate we just ACCEPTed.
                safe_copy(tmp_csv, BEST_CSV)
                safe_copy(tmp_out_gcode, BEST_GCODE)
                eta = min(args.eta_max, eta * args.eta_up)
                verdict = "ACCEPT"
                if verdict == "ACCEPT":
                    _finalize_outputs(f"ACCEPT@{it:03d}")

            else:
                eta = max(args.eta_min, eta * args.eta_down)
                verdict = "REJECT"

            history.append({
                "it": it,
                "ge": ge, "tau_min": tau_min, "tau_max": tau_max, "tau_mean": tau_mean,
                "within": within, "v_cnt": int(v_cnt), "v_max": float(v_max),
                "p95": float(p95_abs), "cvar95": float(cvar95),
                "eta": eta,
            })

            dt = time.time() - t0
            print(f"[Iter {it:03d}] {verdict} | dt={dt:.2f}s | "
                  f"τ_min={r2(tau_min)}, τ_max={r2(tau_max)}, τ_mean={r2(tau_mean)} | "
                  f"within={r2(within)}% | v_cnt={v_cnt} | v_max={r2(v_max)} | cvar95={r2(cvar95)} | eta={r2(eta)}")

            if int(v_cnt) == 0:
                print(f"🎉 Feasible: v_cnt=0 at iter {it}. Stop.")
                break

    except KeyboardInterrupt:
        interrupted = True
        print("\n⛔ KeyboardInterrupt received. Finalizing BEST outputs...")

    finally:
        _finalize_outputs("INTERRUPT" if interrupted else "NORMAL")

if __name__ == "__main__":
    main()
