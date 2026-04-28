#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-shot G-code parameter editor

Features:
- Only accepts .gcode input file
- Edit only lines whose "G-code prefix" matches (e.g., G0, G1, G4)
  - Supports multiple prefixes in one run: e.g. "G0 G1 G4" or "G0,G1,G4" or "G0/G1/G4"
- Edit only one parameter type per run: X/Y/Z/E/F...
- Mode:
  - set   : replace value with given number
  - inc   : add delta to existing value (can be negative)
  - scale : multiply existing value by a factor (e.g., 1.1, 0.95)
- Preserve original line formatting (spacing/order/comments). Only the number token is replaced.
- Preserve original decimal precision per line.
"""

import os
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Tuple


def _count_decimals(num_str: str) -> int:
    """Return number of digits after decimal point in a numeric literal."""
    s = num_str.strip()
    if "e" in s.lower():
        # scientific notation: no clear decimal places; fallback to 6
        return 6
    if "." in s:
        return len(s.split(".", 1)[1])
    return 0


def _format_with_decimals(value: Decimal, decimals: int) -> str:
    """Format Decimal with fixed decimals, avoiding scientific notation."""
    if decimals <= 0:
        q = Decimal("1")
        vq = value.quantize(q, rounding=ROUND_HALF_UP)
        return f"{vq:f}".split(".", 1)[0]
    q = Decimal("1").scaleb(-decimals)  # 10^-decimals
    vq = value.quantize(q, rounding=ROUND_HALF_UP)
    return f"{vq:f}"  # fixed-point


def _is_gcode_file(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(".gcode")


def _parse_prefixes(prefix_input: str) -> List[str]:
    """
    Parse multi-prefix input like:
      "G0 G1 G4" or "G0,G1,G4" or "G0/G1/G4"
    """
    parts = re.split(r"[,\s/]+", prefix_input.strip())
    prefixes = [p.strip().upper() for p in parts if p.strip()]
    if not prefixes:
        raise ValueError("prefix 不能為空，請至少輸入一個，例如 G1 或 G0,G1。")
    return prefixes


def _compile_prefix_regex(prefixes: List[str]) -> re.Pattern:
    """
    Match any given prefix token at start of G-code (ignoring leading whitespace),
    allowing line numbers like 'N123 ' before it.

    Examples matched:
      "G1 X10"
      "  G1 X10"
      "N123 G1 X10"
    Not matched:
      ";G1 X10" (commented)
      "M104 ..."
    """
    clean = [re.escape(p.strip().upper()) for p in prefixes if p.strip()]
    if not clean:
        raise ValueError("prefix 清單無效。")

    # (?=\s|$) ensures the prefix ends as a token
    alt = "|".join(clean)
    return re.compile(rf"^\s*(?:N\d+\s+)?(?:{alt})(?=\s|$)", re.IGNORECASE)


def _compile_param_regex(param: str) -> re.Pattern:
    """
    Match the parameter token like X12.34, allowing +/- sign and decimals.
    We capture:
      group(1): the letter (e.g., X)
      group(2): the numeric literal
    """
    letter = param.strip().upper()
    if not letter:
        raise ValueError("參數類型不可為空，例如 X/Y/Z/E/F。")

    return re.compile(
        rf"({re.escape(letter)})([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    )


def process_gcode(
    input_path: str,
    prefixes: List[str],
    param: str,
    mode: str,
    value: Decimal,
    output_path: str,
    encoding: str = "utf-8",
) -> Tuple[int, int]:
    """
    Returns (lines_matched_prefix, params_modified)
    """
    if not _is_gcode_file(input_path):
        raise ValueError("Input must be an existing .gcode file.")

    if not output_path.lower().endswith(".gcode"):
        raise ValueError("Output file name must end with .gcode")

    prefix_re = _compile_prefix_regex(prefixes)
    param_re = _compile_param_regex(param)

    mode = mode.strip().lower()
    if mode not in ("set", "inc", "scale"):
        raise ValueError("Mode must be 'set', 'inc', or 'scale'.")

    lines_matched = 0
    mods = 0

    with open(input_path, "r", encoding=encoding, newline="") as fin, open(
        output_path, "w", encoding=encoding, newline=""
    ) as fout:
        for line in fin:
            original_line = line

            # Protect ';' comments (common in 3D printing gcode).
            if ";" in line:
                code_part, comment_part = line.split(";", 1)
                comment_part = ";" + comment_part
            else:
                code_part, comment_part = line, ""

            if prefix_re.search(code_part):
                lines_matched += 1

                def repl(m: re.Match) -> str:
                    nonlocal mods
                    letter = m.group(1)
                    num_str = m.group(2)

                    try:
                        old_val = Decimal(num_str)
                    except Exception:
                        return m.group(0)

                    decs = _count_decimals(num_str)

                    if mode == "set":
                        new_val = value
                    elif mode == "inc":
                        new_val = old_val + value
                    else:  # scale
                        new_val = old_val * value

                    new_num_str = _format_with_decimals(new_val, decs)
                    mods += 1
                    return f"{letter}{new_num_str}"

                new_code_part = param_re.sub(repl, code_part)
                fout.write(new_code_part + comment_part)
            else:
                fout.write(original_line)

    return lines_matched, mods


def main():
    print("=== G-code 全檔案一次性調整工具 ===")
    input_path = input("1) gcode檔案名稱（.gcode）: ").strip()

    prefix_in = input("2) gcode前綴 (可複數：例如 G1 或 G0,G1,G4 或 G0/G1/G4): ").strip()
    prefixes = _parse_prefixes(prefix_in)

    param = input("3) 修改的參數類型 (例如 X/Y/Z/E/F): ").strip().upper()

    mode_in = input("4) 模式：a=set(直接指定) / b=inc(增量 +/-) / c=scale(比例縮放): ").strip().lower()
    if mode_in in ("a", "set"):
        mode = "set"
        val_tip = "輸入新值"
    elif mode_in in ("b", "inc"):
        mode = "inc"
        val_tip = "輸入增量(可負數)"
    elif mode_in in ("c", "scale"):
        mode = "scale"
        val_tip = "輸入倍率(例如 1.1 / 0.95)"
    else:
        raise ValueError("模式輸入錯誤，請輸入 a / b / c。")

    val_str = input(f"   4-值：{val_tip}: ").strip()
    value = Decimal(val_str)

    output_path = input("5) 新的檔案名稱（.gcode）: ").strip()

    lines, mods = process_gcode(
        input_path=input_path,
        prefixes=prefixes,
        param=param,
        mode=mode,
        value=value,
        output_path=output_path,
    )

    print("\n=== 完成 ===")
    print(f"匹配前綴的行數: {lines}")
    print(f"實際修改參數次數: {mods}")
    print(f"輸出檔案: {output_path}")


if __name__ == "__main__":
    main()
