#!/usr/bin/env python3
"""
快速檢測 G-code 中的 E-only movement commands。

E-only 的定義：
1. 命令為 G0/G1，或省略 G0/G1 但沿用先前的 modal G0/G1。
2. 該行包含 E。
3. 該行不包含其他運動軸：X/Y/Z/A/B/C/U/V/W。

G92 E0、M200 等非移動命令不會被計入。
只使用 Python 標準函式庫，並逐行掃描，適合大型 G-code。

CLI:
python detect_e_only_commands.py LiQ5_testmodel_XL_2244s_Z+1_F0.1_T0.1_optimized_best.gcode
python detect_e_only_commands.py LiQ5_testmodel_XL_2244s_Z+1_F0.1_T0.1_optimized_best.gcode --csv e_only_report.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, TextIO


NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
WORD_RE = re.compile(rf"([A-Za-z])\s*({NUMBER_PATTERN})")
MOTION_AXES = frozenset("XYZABCUVW")
EPSILON = 1e-12


@dataclass(frozen=True)
class EOnlyCommand:
    line_number: int
    extrusion_mode: str
    direction: str
    e_value: float
    explicit_f: Optional[float]
    active_f: Optional[float]
    content: str


def strip_gcode_comments(line: str) -> str:
    """移除分號註解及括號註解，但保留括號外的 G-code。"""
    result: list[str] = []
    parentheses_depth = 0

    for char in line:
        if char == ";" and parentheses_depth == 0:
            break
        if char == "(":
            parentheses_depth += 1
            continue
        if char == ")" and parentheses_depth > 0:
            parentheses_depth -= 1
            continue
        if parentheses_depth == 0:
            result.append(char)

    return "".join(result)


def parse_words(code: str) -> list[tuple[str, float]]:
    """解析如 G1、E-0.2、F3e2，亦支援 G1E0.2F300 緊湊格式。"""
    return [
        (match.group(1).upper(), float(match.group(2)))
        for match in WORD_RE.finditer(code)
    ]


def integer_code(value: float) -> Optional[int]:
    """只接受整數型 G/M code，例如 G1、G01、M83。"""
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=EPSILON):
        return int(rounded)
    return None


def e_direction(value: float) -> str:
    if value > EPSILON:
        return "extrude"
    if value < -EPSILON:
        return "retract"
    return "zero"


def scan_e_only_commands(stream: Iterable[str]) -> Iterable[EOnlyCommand]:
    """
    逐行掃描並 yield E-only commands。

    同時追蹤：
    - G0/G1 modal motion
    - M82/M83 extrusion mode
    - modal feed rate F
    """
    modal_motion: Optional[int] = None
    extrusion_mode = "unknown"
    active_f: Optional[float] = None

    for line_number, raw_line in enumerate(stream, start=1):
        original = raw_line.rstrip("\r\n")
        code = strip_gcode_comments(original)
        words = parse_words(code)

        if not words:
            continue

        # M82/M83 依該行出現順序更新；最後一個設定生效。
        for letter, value in words:
            if letter != "M":
                continue
            m_code = integer_code(value)
            if m_code == 82:
                extrusion_mode = "M82"
            elif m_code == 83:
                extrusion_mode = "M83"

        g_codes = [
            code_number
            for letter, value in words
            if letter == "G"
            for code_number in [integer_code(value)]
            if code_number is not None
        ]
        explicit_motion = next(
            (code_number for code_number in reversed(g_codes) if code_number in (0, 1)),
            None,
        )

        # 有明確的非 G0/G1 G-code（如 G92 E0）時，不套用 modal motion。
        if explicit_motion is not None:
            effective_motion = explicit_motion
            modal_motion = explicit_motion
        elif g_codes:
            effective_motion = None
        else:
            effective_motion = modal_motion

        f_values = [value for letter, value in words if letter == "F"]
        explicit_f = f_values[-1] if f_values else None
        if explicit_f is not None:
            active_f = explicit_f

        if effective_motion not in (0, 1):
            continue

        letters = {letter for letter, _ in words}
        if "E" not in letters or letters.intersection(MOTION_AXES):
            continue

        e_values = [value for letter, value in words if letter == "E"]
        if not e_values:
            continue
        e_value = e_values[-1]

        yield EOnlyCommand(
            line_number=line_number,
            extrusion_mode=extrusion_mode,
            direction=e_direction(e_value),
            e_value=e_value,
            explicit_f=explicit_f,
            active_f=active_f,
            content=original,
        )


def format_number(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.12g}"


def write_csv_report(
    path: Path,
    source_gcode: Path,
    commands: list[EOnlyCommand],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "source_gcode",
                "line_number",
                "extrusion_mode",
                "direction",
                "E",
                "explicit_F",
                "active_F",
                "content",
            ]
        )
        for command in commands:
            writer.writerow(
                [
                    str(source_gcode),
                    command.line_number,
                    command.extrusion_mode,
                    command.direction,
                    format_number(command.e_value),
                    format_number(command.explicit_f),
                    format_number(command.active_f),
                    command.content,
                ]
            )


def print_report(
    source_gcode: Path,
    commands: list[EOnlyCommand],
    output: TextIO = sys.stdout,
) -> None:
    extrude_count = sum(command.direction == "extrude" for command in commands)
    retract_count = sum(command.direction == "retract" for command in commands)
    zero_count = sum(command.direction == "zero" for command in commands)
    explicit_f_count = sum(command.explicit_f is not None for command in commands)
    modal_f_count = sum(
        command.explicit_f is None and command.active_f is not None
        for command in commands
    )
    missing_f_count = sum(command.active_f is None for command in commands)

    print(f"File: {source_gcode}", file=output)
    print(f"E-only commands: {len(commands)}", file=output)
    print(
        "Direction: "
        f"extrude={extrude_count}, retract={retract_count}, zero={zero_count}",
        file=output,
    )
    print(
        "Feed rate: "
        f"explicit F={explicit_f_count}, inherited F={modal_f_count}, "
        f"F unavailable={missing_f_count}",
        file=output,
    )

    if not commands:
        return

    print(file=output)
    print(
        f"{'Line':>8}  {'Mode':<7}  {'Type':<8}  {'E':>14}  "
        f"{'F(active)':>14}  Content",
        file=output,
    )
    print("-" * 96, file=output)

    for command in commands:
        print(
            f"{command.line_number:>8}  "
            f"{command.extrusion_mode:<7}  "
            f"{command.direction:<8}  "
            f"{format_number(command.e_value):>14}  "
            f"{format_number(command.active_f):>14}  "
            f"{command.content}",
            file=output,
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "快速列出 G-code 中只有 E 軸移動、沒有其他運動軸的 G0/G1 指令。"
        )
    )
    parser.add_argument("gcode", type=Path, help="要檢查的 .gcode 檔案")
    parser.add_argument(
        "--csv",
        type=Path,
        metavar="REPORT.csv",
        help="另存包含所有結果的 CSV 報告",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    gcode_path: Path = args.gcode

    if not gcode_path.is_file():
        print(f"Error: 找不到 G-code 檔案：{gcode_path}", file=sys.stderr)
        return 2

    try:
        with gcode_path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as gcode_file:
            commands = list(scan_e_only_commands(gcode_file))
    except OSError as exc:
        print(f"Error: 無法讀取 {gcode_path}：{exc}", file=sys.stderr)
        return 2

    print_report(gcode_path, commands)

    if args.csv is not None:
        try:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            write_csv_report(args.csv, gcode_path, commands)
        except OSError as exc:
            print(f"Error: 無法寫入 CSV 報告 {args.csv}：{exc}", file=sys.stderr)
            return 2
        print(f"\nCSV report: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
