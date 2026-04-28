#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split the first (or Nth) 'G1 F... E...' line that contains ONLY F and E
(no X/Y/Z/I/J/K/R/etc) into user-defined number of segments.

- Detects E-mode from M82 (absolute) / M83 (relative). Defaults to relative if not found.
- Preserves comments after ';'
- Tries to preserve decimal precision of E token

CLI:
python split_g1_fe_only.py input.gcode --n 12 --out output.gcode
python split_g1_fe_only.py input.gcode --n 8 --occurrence 2 --out output.gcode
python split_g1_fe_only.py input.gcode --n 10 --head 500 --out output.gcode
"""

import argparse
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Tuple, List


FE_ONLY_RE = re.compile(
    r'^\s*G1\s+([^;]*?)\s*(?:;(?P<comment>.*))?$',
    re.IGNORECASE
)

TOKEN_RE = re.compile(r'([A-Za-z])\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)')


def detect_e_mode(lines: List[str], head: int = 4000) -> str:
    """
    Return 'absolute' if M82 appears before M83 in head area,
    'relative' if M83 appears before M82,
    else default 'relative'.
    """
    m82_idx = None
    m83_idx = None
    for i, raw in enumerate(lines[:head]):
        s = raw.strip().upper()
        # ignore commented full-line; but many gcode uses leading ';'
        if s.startswith(";"):
            continue
        if "M82" in s and m82_idx is None:
            m82_idx = i
        if "M83" in s and m83_idx is None:
            m83_idx = i
        if m82_idx is not None and m83_idx is not None:
            break

    if m82_idx is None and m83_idx is None:
        return "relative"
    if m82_idx is None:
        return "relative"
    if m83_idx is None:
        return "absolute"
    return "absolute" if m82_idx < m83_idx else "relative"


def parse_g1_fe_only(line: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    If line is a G1 line whose parameters contain ONLY F and E (in any order),
    return (F_str, E_str, comment). Otherwise return None.
    """
    m = FE_ONLY_RE.match(line)
    if not m:
        return None

    params = (m.group(1) or "").strip()
    comment = m.group("comment")

    # Extract tokens like F800, E1.23
    tokens = TOKEN_RE.findall(params)
    if not tokens:
        return None

    # Build dict and also ensure there are no other axes/params
    allowed = {"F", "E"}
    seen = {}
    for k, v in tokens:
        kk = k.upper()
        if kk not in allowed:
            return None
        seen[kk] = v

    # Ensure params text doesn't contain other letters (e.g. X/Y/Z) not captured
    # Quick conservative check: remove recognized tokens and whitespace, if leftovers include letters -> reject
    scrub = TOKEN_RE.sub("", params)
    if re.search(r"[A-Za-z]", scrub):
        return None

    if "E" not in seen:
        return None
    # F can be missing; but user said "G1 Fx Ex". We'll still allow missing F.
    f_str = seen.get("F", "")
    e_str = seen["E"]
    return f_str, e_str, comment


def count_decimals(num_str: str) -> int:
    s = num_str.strip()
    if "e" in s.lower():
        # scientific: keep a sane precision (we'll quantize later)
        return 6
    if "." in s:
        return len(s.split(".", 1)[1])
    return 0


def fmt_decimal(x: Decimal, decimals: int) -> str:
    if decimals <= 0:
        # integer output
        q = Decimal("1")
        return str(x.quantize(q, rounding=ROUND_HALF_UP))
    q = Decimal("1").scaleb(-decimals)  # 10^-decimals
    return str(x.quantize(q, rounding=ROUND_HALF_UP))


def split_line(
    original_line: str,
    f_str: str,
    e_str: str,
    comment: Optional[str],
    n: int,
    e_mode: str,
    last_abs_e: Optional[Decimal],
) -> Tuple[List[str], Optional[Decimal]]:
    """
    Return (new_lines, new_last_abs_e).

    - relative: replace with N lines of E = E_total / N
    - absolute: need last_abs_e; create monotonic absolute E values
    """
    decimals = count_decimals(e_str)
    E_total = Decimal(e_str)

    # keep F formatting if present
    f_part = f" F{f_str}" if f_str else ""
    c_part = f" ;{comment}" if (comment is not None and comment.strip() != "") else ""
    prefix_ws = re.match(r"^(\s*)", original_line).group(1)

    new_lines: List[str] = []

    if e_mode == "relative":
        e_piece = (E_total / Decimal(n))
        e_piece_s = fmt_decimal(e_piece, decimals)
        for _ in range(n):
            new_lines.append(f"{prefix_ws}G1{f_part} E{e_piece_s}{c_part}\n")
        return new_lines, last_abs_e

    # absolute mode
    if last_abs_e is None:
        raise ValueError("Absolute E mode (M82) detected, but could not determine last absolute E before target line.")

    target_abs = E_total
    delta = target_abs - last_abs_e
    e_piece = delta / Decimal(n)

    # build N absolute E values
    for i in range(1, n + 1):
        Ei = last_abs_e + e_piece * Decimal(i)
        Ei_s = fmt_decimal(Ei, decimals)
        new_lines.append(f"{prefix_ws}G1{f_part} E{Ei_s}{c_part}\n")

    return new_lines, target_abs


def extract_last_abs_e_before(lines: List[str], idx: int) -> Optional[Decimal]:
    """
    For absolute E mode: find last seen E value in any G1/G0 line before idx (very common).
    This is conservative but practical.
    """
    e_re = re.compile(r'(^|\s)E\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)', re.IGNORECASE)
    for j in range(idx - 1, -1, -1):
        s = lines[j]
        if s.lstrip().startswith(";"):
            continue
        m = e_re.search(s.split(";", 1)[0])
        if m:
            try:
                return Decimal(m.group(2))
            except Exception:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input .gcode file")
    ap.add_argument("--out", required=True, help="Output .gcode file")
    ap.add_argument("--n", type=int, required=True, help="Split the target E into N segments (N>=2)")
    ap.add_argument("--occurrence", type=int, default=1, help="Which FE-only G1 line to split (1=first match)")
    ap.add_argument("--head", type=int, default=50, help="Search only within first HEAD lines")
    args = ap.parse_args()

    if args.n < 2:
        raise SystemExit("--n must be >= 2")
    if args.occurrence < 1:
        raise SystemExit("--occurrence must be >= 1")
    if args.head < 1:
        raise SystemExit("--head must be >= 1")

    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    e_mode = detect_e_mode(lines, head=max(args.head, 200))
    # find target line
    match_count = 0
    target_idx = None
    target_parsed = None

    for i, line in enumerate(lines[:args.head]):
        parsed = parse_g1_fe_only(line)
        if parsed:
            match_count += 1
            if match_count == args.occurrence:
                target_idx = i
                target_parsed = parsed
                break

    if target_idx is None or target_parsed is None:
        raise SystemExit(f"Could not find occurrence #{args.occurrence} of a 'G1 F... E...' line with ONLY F and E within first {args.head} lines.")

    f_str, e_str, comment = target_parsed
    print(f"[OK] Detected E-mode: {e_mode}")
    print(f"[OK] Target line index: {target_idx} (1-based line {target_idx+1})")
    print(f"[OK] Detected target E: {e_str}")
    if f_str:
        print(f"[OK] Detected target F: {f_str}")
    else:
        print("[WARN] Target line has no F token; will keep it empty.")

    last_abs_e = None
    if e_mode == "absolute":
        last_abs_e = extract_last_abs_e_before(lines, target_idx)
        if last_abs_e is None:
            raise SystemExit("M82 (absolute E) detected but could not find a previous E value before the target line. "
                             "Either add an earlier extrusion with E, or run in relative mode (M83).")
        print(f"[OK] Last absolute E before target: {last_abs_e}")

    new_lines, _ = split_line(
        original_line=lines[target_idx],
        f_str=f_str,
        e_str=e_str,
        comment=comment,
        n=args.n,
        e_mode=e_mode,
        last_abs_e=last_abs_e
    )

    out = lines[:target_idx] + new_lines + lines[target_idx + 1:]

    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(out)

    print(f"[DONE] Wrote output: {args.out}")


if __name__ == "__main__":
    main()
