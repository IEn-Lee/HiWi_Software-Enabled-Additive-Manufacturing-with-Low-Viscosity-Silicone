import re
import argparse
from pathlib import Path

"""
python convert_e_by_segidx.py \
  --input input.gcode \
  --output output.gcode \
  --seg_start 1 \
  --seg_end 10
========================================
Start from E12.5

python convert_e_by_segidx.py \
  --input input.gcode \
  --output output.gcode \
  --seg_start 1 \
  --seg_end 10 \
  --initial_e 12.5
"""

SEG_RE = re.compile(r"\bseg_idx\s*=\s*(-?\d+)\b")

E_RE = re.compile(
    r"(?<![A-Za-z])E\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)

G92_RE = re.compile(r"^\s*G92\b", re.IGNORECASE)
MOVE_RE = re.compile(r"^\s*G[01]\b", re.IGNORECASE)
M83_RE = re.compile(r"^\s*M83\b", re.IGNORECASE)


def split_comment(line: str):
    """
    Split one G-code line into command part and comment part.
    The semicolon is kept in the comment part.
    """
    if ";" in line:
        code, comment = line.split(";", 1)
        return code, ";" + comment
    return line, ""


def get_seg_idx(line: str):
    """
    Extract seg_idx from the full line, including comments.
    Example:
        G1 E0.1 ; seg_idx=5
    """
    match = SEG_RE.search(line)
    if match:
        return int(match.group(1))
    return None


def replace_first_e_value(code: str, new_e: float, decimals: int = 6):
    """
    Replace only the first E value in the command part.
    Comments are not modified.
    """
    def repl(_match):
        return f"E{new_e:.{decimals}f}"

    return E_RE.sub(repl, code, count=1)


def convert_relative_e_to_cumulative_e_by_segidx(
    lines,
    seg_start: int,
    seg_end: int,
    initial_e: float = 0.0,
    decimals: int = 6,
):
    """
    Convert relative E values into cumulative E values only inside the selected seg_idx range.

    Rules:
    - Only G0/G1 lines with E and seg_idx inside [seg_start, seg_end] are modified.
    - G92 E... resets the internal E accumulator if it is inside the selected range.
    - M83 is not modified, but a warning is printed.
    - G0/G1 E... without seg_idx gives a warning and is kept unchanged.
    """

    e_abs = initial_e
    output_lines = []
    warnings = []

    found_any_seg_idx = False
    found_selected_seg_idx = False

    for line_no, line in enumerate(lines, start=1):
        raw_line = line.rstrip("\n")
        code, comment = split_comment(raw_line)

        seg_idx = get_seg_idx(raw_line)

        if seg_idx is not None:
            found_any_seg_idx = True

        has_e = E_RE.search(code) is not None
        is_g92 = G92_RE.search(code) is not None
        is_move = MOVE_RE.search(code) is not None
        is_m83 = M83_RE.search(code) is not None

        if is_m83:
            warnings.append(
                f"[WARNING] line {line_no}: M83 detected. "
                f"Line kept unchanged: {raw_line}"
            )

        in_selected_range = (
            seg_idx is not None and seg_start <= seg_idx <= seg_end
        )

        if in_selected_range:
            found_selected_seg_idx = True

        # ------------------------------------------------------------
        # Case 1: G92 E...
        # ------------------------------------------------------------
        if is_g92 and has_e:
            if seg_idx is None:
                warnings.append(
                    f"[WARNING] line {line_no}: G92 with E has no seg_idx comment. "
                    f"Line kept unchanged; accumulator reset is skipped: {raw_line}"
                )
                output_lines.append(raw_line + "\n")
                continue

            if in_selected_range:
                e_value = float(E_RE.search(code).group(1))
                e_abs = e_value

            output_lines.append(raw_line + "\n")
            continue

        # ------------------------------------------------------------
        # Case 2: G0/G1 E...
        # ------------------------------------------------------------
        if is_move and has_e:
            if seg_idx is None:
                warnings.append(
                    f"[WARNING] line {line_no}: movement line with E has no seg_idx comment. "
                    f"Line kept unchanged: {raw_line}"
                )
                output_lines.append(raw_line + "\n")
                continue

            if in_selected_range:
                e_rel = float(E_RE.search(code).group(1))
                e_abs += e_rel
                new_code = replace_first_e_value(code, e_abs, decimals)
                output_lines.append(new_code + comment + "\n")
            else:
                output_lines.append(raw_line + "\n")

            continue

        # ------------------------------------------------------------
        # Case 3: all other lines
        # ------------------------------------------------------------
        output_lines.append(raw_line + "\n")

    if not found_any_seg_idx:
        warnings.append(
            "[WARNING] No seg_idx comment was found anywhere in the input G-code."
        )

    if found_any_seg_idx and not found_selected_seg_idx:
        warnings.append(
            f"[WARNING] No line found in selected seg_idx range: "
            f"{seg_start} to {seg_end}."
        )

    return output_lines, warnings


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert relative E values to cumulative E values only within "
            "a selected seg_idx range. M83 is detected and warned but not modified."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input G-code file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output G-code file.",
    )

    parser.add_argument(
        "--seg_start",
        type=int,
        required=True,
        help="First seg_idx to process.",
    )

    parser.add_argument(
        "--seg_end",
        type=int,
        required=True,
        help="Last seg_idx to process.",
    )

    parser.add_argument(
        "--initial_e",
        type=float,
        default=0.0,
        help="Initial E accumulator value. Default: 0.0.",
    )

    parser.add_argument(
        "--decimals",
        type=int,
        default=6,
        help="Number of decimals for rewritten E values. Default: 6.",
    )

    args = parser.parse_args()

    if args.seg_end < args.seg_start:
        raise ValueError("seg_end must be greater than or equal to seg_start.")

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)

    converted_lines, warnings = convert_relative_e_to_cumulative_e_by_segidx(
        lines=lines,
        seg_start=args.seg_start,
        seg_end=args.seg_end,
        initial_e=args.initial_e,
        decimals=args.decimals,
    )

    output_path.write_text("".join(converted_lines), encoding="utf-8")

    print(f"Done: {output_path}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(warning)


if __name__ == "__main__":
    main()