#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Edit one parameter in selected G-code commands.

This script modifies a specified parameter, such as X, Y, Z, E, or F, only
within G-code lines whose command prefixes match the user-defined selection.

Multiple command prefixes can be processed in one run. For example, the user
may select G0, G1, and G4 simultaneously. The original spacing, parameter
order, comments, line endings, and decimal precision are preserved whenever
possible.

Supported modification modes are:

* ``set``: Replace the existing parameter value.
* ``inc``: Add a specified value to the existing parameter value.
* ``scale``: Multiply the existing parameter value by a specified factor.

Only existing parameter tokens are modified. Missing parameters are not added.
"""

import os
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Tuple


def _count_decimals(num_str: str) -> int:
    """Count the decimal places in a numeric literal.

    Scientific notation does not provide an unambiguous fixed decimal
    precision, so six decimal places are used as a fallback.

    Args:
        num_str: Numeric literal to inspect, such as ``"12.340"`` or
            ``"1.2e-3"``.

    Returns:
        The number of digits after the decimal point. Returns 6 when the
        input uses scientific notation.
    """
    s = num_str.strip()
    if "e" in s.lower():
        # scientific notation: no clear decimal places; fallback to 6
        return 6
    if "." in s:
        return len(s.split(".", 1)[1])
    return 0


def _format_with_decimals(value: Decimal, decimals: int) -> str:
    """Format a decimal value using a fixed number of decimal places.

    The value is rounded using ``ROUND_HALF_UP`` and formatted without
    scientific notation.

    Args:
        value: Decimal value to format.
        decimals: Number of digits to retain after the decimal point.

    Returns:
        The formatted numeric string.
    """
    if decimals <= 0:
        q = Decimal("1")
        vq = value.quantize(q, rounding=ROUND_HALF_UP)
        return f"{vq:f}".split(".", 1)[0]
    q = Decimal("1").scaleb(-decimals)  # 10^-decimals
    vq = value.quantize(q, rounding=ROUND_HALF_UP)
    return f"{vq:f}"  # fixed-point


def _is_gcode_file(path: str) -> bool:
    """Check whether a path points to an existing G-code file.

    Args:
        path: Path to the file to check.

    Returns:
        ``True`` if the path is an existing file with a ``.gcode``
        extension; otherwise, ``False``.
    """
    return os.path.isfile(path) and path.lower().endswith(".gcode")


def _parse_prefixes(prefix_input: str) -> List[str]:
    """Parse one or more G-code command prefixes.

    Prefixes may be separated by commas, whitespace, forward slashes, or
    any combination of these delimiters. Returned prefixes are converted
    to uppercase.

    Examples:
        ``"G0 G1 G4"``, ``"G0,G1,G4"``, and ``"G0/G1/G4"`` all produce
        ``["G0", "G1", "G4"]``.

    Args:
        prefix_input: User-provided string containing one or more G-code
            command prefixes.

    Returns:
        A list of normalized uppercase prefixes.

    Raises:
        ValueError: If no valid prefix is provided.
    """
    parts = re.split(r"[,\s/]+", prefix_input.strip())
    prefixes = [p.strip().upper() for p in parts if p.strip()]
    if not prefixes:
        raise ValueError(
            "The prefix cannot be empty. Enter at least one prefix, "
            "such as G1 or G0,G1."
        )
    return prefixes


def _compile_prefix_regex(prefixes: List[str]) -> re.Pattern:
    """Compile a regular expression for selected G-code commands.

    The resulting expression matches a command at the beginning of a
    G-code line while allowing leading whitespace and an optional line
    number such as ``N123``.

    For example, a prefix of ``G1`` matches ``"G1 X10"``,
    ``"  G1 X10"``, and ``"N123 G1 X10"``, but does not match a
    commented line such as ``";G1 X10"``.

    Args:
        prefixes: G-code command prefixes to match.

    Returns:
        A compiled case-insensitive regular expression.

    Raises:
        ValueError: If the prefix list contains no valid entries.
    """
    clean = [re.escape(p.strip().upper()) for p in prefixes if p.strip()]
    if not clean:
        raise ValueError("The prefix list in invalid")

    # (?=\s|$) ensures the prefix ends as a token
    alt = "|".join(clean)
    return re.compile(rf"^\s*(?:N\d+\s+)?(?:{alt})(?=\s|$)", re.IGNORECASE)


def _compile_param_regex(param: str) -> re.Pattern:
    """Compile a regular expression for a G-code parameter token.

    The expression matches a parameter letter followed by a signed or
    unsigned numeric value. Decimal and scientific notation are supported.

    The first capture group contains the parameter letter, and the second
    capture group contains the numeric literal.

    Args:
        param: Parameter letter to match, such as ``X``, ``Y``, ``Z``,
            ``E``, or ``F``.

    Returns:
        A compiled regular expression for the parameter token.

    Raises:
        ValueError: If the parameter string is empty.
    """
    letter = param.strip().upper()
    if not letter:
        raise ValueError("The parameter type cannot be empty. Enter X, Y, Z, E, or F.")

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
    """Modify one parameter in selected G-code command lines.

    Only lines beginning with one of the specified G-code prefixes are
    processed. Content following a semicolon is treated as a comment and
    is not modified.

    Existing parameter values can be replaced, incremented, or scaled.
    The original decimal precision of each parameter value is preserved.
    Parameters that are not already present in a matching line are not
    added.

    Args:
        input_path: Path to the input ``.gcode`` file.
        prefixes: G-code command prefixes to process, such as
            ``["G0", "G1"]``.
        param: Parameter letter to modify, such as ``X``, ``Y``, ``Z``,
            ``E``, or ``F``.
        mode: Modification mode. Supported values are ``"set"``,
            ``"inc"``, and ``"scale"``.
        value: Replacement value, increment, or scaling factor, depending
            on the selected mode.
        output_path: Path to the output ``.gcode`` file.
        encoding: Text encoding used to read and write the files.
            Defaults to ``"utf-8"``.

    Returns:
        A tuple containing:

        * The number of lines matching the selected G-code prefixes.
        * The number of parameter occurrences actually modified.

    Raises:
        ValueError: If the input is not an existing ``.gcode`` file.
        ValueError: If the output filename does not end with ``.gcode``.
        ValueError: If ``mode`` is not ``"set"``, ``"inc"``, or
            ``"scale"``.
        ValueError: If the prefix list or parameter is invalid.
        OSError: If the input file cannot be read or the output file
            cannot be written.
        UnicodeError: If the file cannot be decoded or encoded using the
            selected encoding.
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
                    """Calculate and format a replacement parameter token.

                    Args:
                        m: Regular expression match containing the parameter letter and
                            its numeric value.

                    Returns:
                        The parameter token containing the modified numeric value. If the
                        original value cannot be converted to ``Decimal``, the unchanged
                        matched token is returned.
                    """
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
    """Run the interactive command-line interface.

    The user is prompted to select an input file, G-code command prefixes,
    parameter type, modification mode, modification value, and output
    filename.

    Raises:
        ValueError: If a user-provided mode or numeric value is invalid.
        OSError: If the input file cannot be read or the output file
            cannot be written.
    """
    print("=== One-Shot G-code Parameer Editor ===")
    input_path = input("1) Input G-code filename（.gcode）: ").strip()

    prefix_in = input("2) G-code prefix or prefixes "
                      "(e.g., G1; G0,G1,G4; or G0/G1/G4): ").strip()
    prefixes = _parse_prefixes(prefix_in)

    param = input("3) Parameter to modify (e.g., X/Y/Z/E, or F): ").strip().upper()

    mode_in = input("4) Modification mode: "
                    "a = set value / b = increment(+/-) / c = scale: ").strip().lower()
    
    if mode_in in ("a", "set"):
        mode = "set"
        val_tip = "Enter the new value"
    elif mode_in in ("b", "inc"):
        mode = "inc"
        val_tip = "Enter the increment (may be negative)"
    elif mode_in in ("c", "scale"):
        mode = "scale"
        val_tip = "Enter the scaling factor(e.g., 1.1 / 0.95)"
    else:
        raise ValueError("Invalid modification mode. Enter a / b / c")

    val_str = input(f"   4-value：{val_tip}: ").strip()
    value = Decimal(val_str)

    output_path = input("5) Output G-code filename (.gcode): ").strip()

    lines, mods = process_gcode(
        input_path=input_path,
        prefixes=prefixes,
        param=param,
        mode=mode,
        value=value,
        output_path=output_path,
    )

    print("\n=== Processing Compplete ===")
    print(f"Number of lines matching the prefixes: {lines}")
    print(f"Number of parameter values modified: {mods}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
