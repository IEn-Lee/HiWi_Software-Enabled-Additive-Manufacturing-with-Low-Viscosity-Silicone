#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manual G-code Subdivider with Mode Selection

Mode 1: Single command mode
    Input:
        G1 F100 E3.083 ; precondition optimizable

    n = 2 output:
        G1 F100 E1.5415 ; precondition optimizable
        G1 F100 E1.5415 ; precondition optimizable


Mode 2: Compound G1 + G4 mode
    Input:
        G1 F100 E3.083 ; precondition optimizable
        G4 P2000000 ; optimizable

    n = 2 output:
        G1 F100 E1.5415 ; precondition optimizable
        G4 P1000000 ; optimizable
        G1 F100 E1.5415 ; precondition optimizable
        G4 P1000000 ; optimizable

Important:
    Mode descriptions are only shown in the Python terminal interface.
    They are NOT written into the exported .gcode file.
"""

import os
import re
from decimal import Decimal, getcontext


# Use high precision for subdivision calculations
getcontext().prec = 28


# -------------------------------------------------------------------------
# Basic G-code parsing functions
# -------------------------------------------------------------------------

def split_comment(line: str):
    """Split a G-code line into code and comment components.

    Args:
        line: G-code line that may contain a semicolon comment.

    Returns:
        A tuple containing the G-code portion and the normalized comment.
        The comment is an empty string if no comment is present.

    Example:
        ``"G1 E3.083 ; optimizable"`` returns
        ``("G1 E3.083", "; optimizable")``.
    """
    line = line.strip()

    if ";" in line:
        code, comment = line.split(";", 1)
        return code.rstrip(), "; " + comment.strip()
    else:
        return line.rstrip(), ""


def get_command(code: str):
    """Extract the command token from a G-code string.

    Args:
        code: G-code content without its semicolon comment.

    Returns:
        The uppercase command token, such as ``"G1"`` or ``"G4"``.
        Returns ``None`` if the input contains no tokens.
    """
    tokens = code.strip().split()

    if not tokens:
        return None

    return tokens[0].upper()


def extract_field_value(code: str, field: str):
    """Extract a numeric field value from a G-code string.

    Args:
        code: G-code content to search.
        field: Field letter to extract, such as ``"E"`` or ``"P"``.

    Returns:
        The field value as a ``Decimal``, or ``None`` if the field
        is not present.
    """
    pattern = rf"(?<![A-Za-z]){field}([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    match = re.search(pattern, code, flags=re.IGNORECASE)

    if not match:
        return None

    return Decimal(match.group(1))


def replace_field_value(code: str, field: str, new_value: str):
    """Replace the first matching field value in a G-code string.

    Args:
        code: G-code content containing the field to replace.
        field: Field letter to replace, such as ``"E"`` or ``"P"``.
        new_value: Replacement value formatted as a string.

    Returns:
        The updated G-code string. If the field is absent, the original
        string is returned unchanged.
    """
    pattern = rf"(?<![A-Za-z])({field})([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

    return re.sub(
        pattern,
        rf"\g<1>{new_value}",
        code,
        count=1,
        flags=re.IGNORECASE
    )


def rebuild_line(code: str, comment: str):
    """Recombine G-code content with its optional comment.

    Args:
        code: G-code content without a comment.
        comment: Semicolon comment, or an empty string.

    Returns:
        The complete G-code line.
    """
    if comment:
        return f"{code} {comment}"
    else:
        return code


# -------------------------------------------------------------------------
# Number formatting and subdivision functions
# -------------------------------------------------------------------------

def format_decimal(value: Decimal, digits: int = 6):
    """Format a decimal value without unnecessary trailing zeros.

    Args:
        value: Decimal value to format.
        digits: Maximum number of fractional digits. Defaults to 6.

    Returns:
        A fixed-point numeric string with trailing zeros removed.
    """
    quant = Decimal("1." + "0" * digits)
    value = value.quantize(quant)

    text = format(value, "f")

    if "." in text:
        text = text.rstrip("0").rstrip(".")

    return text


def split_decimal_preserve_total(
    value: Decimal,
    n: int,
    digits: int = 6,
):
    """Split a decimal value while preserving its total.

    The first ``n - 1`` parts use the requested decimal precision.
    The final part compensates for accumulated rounding error.

    Args:
        value: Decimal value to subdivide.
        n: Number of output parts.
        digits: Maximum number of fractional digits. Defaults to 6.

    Returns:
        A list of ``n`` decimal values whose sum equals ``value``.

    Raises:
        decimal.DivisionByZero: If ``n`` is zero.
    """
    part = value / Decimal(n)
    rounded_part = Decimal(format_decimal(part, digits))

    parts = [rounded_part for _ in range(n - 1)]
    last_part = value - sum(parts)

    parts.append(last_part)

    return parts


def split_integer_preserve_total(value: int, n: int):
    """Split an integer into parts while preserving its total.

    Any remainder is distributed one unit at a time from the beginning
    of the returned list.

    Args:
        value: Integer value to subdivide.
        n: Number of output parts.

    Returns:
        A list of ``n`` integers whose sum equals ``value``.

    Raises:
        ZeroDivisionError: If ``n`` is zero.
    """
    base = value // n
    remainder = value % n

    parts = []

    for i in range(n):
        parts.append(base + (1 if i < remainder else 0))

    return parts


# -------------------------------------------------------------------------
# G-code subdivision functions
# -------------------------------------------------------------------------

def subdivide_gcode_line(line: str, n: int, digits: int = 6):
    """Subdivide one supported G-code command.

    G0 and G1 commands are subdivided using their E values. G4 commands
    are subdivided using their integer P dwell values. The original
    comment is copied to every generated command.

    Args:
        line: Complete G-code line to subdivide.
        n: Number of commands to generate.
        digits: Maximum number of fractional digits for E values.
            Defaults to 6.

    Returns:
        A list containing the subdivided G-code lines.

    Raises:
        ValueError: If the command is unsupported or the required
            E or P field is missing.
    """
    code, comment = split_comment(line)
    command = get_command(code)

    if command in ["G0", "G1"]:
        target_field = "E"
    elif command == "G4":
        target_field = "P"
    else:
        raise ValueError(
            f"Unsupported command: {command}. "
            "Only G0, G1, and G4 are supported."
        )

    value = extract_field_value(code, target_field)

    if value is None:
        raise ValueError(f"Cannot find {target_field} value in line: {line}")

    output_lines = []

    if target_field == "E":
        parts = split_decimal_preserve_total(value, n=n, digits=digits)

        for part in parts:
            part_text = format_decimal(part, digits)
            new_code = replace_field_value(code, target_field, part_text)
            output_lines.append(rebuild_line(new_code, comment))

    elif target_field == "P":
        # G4 P is usually integer dwell time in milliseconds.
        p_value = int(value)
        parts = split_integer_preserve_total(p_value, n=n)

        for part in parts:
            new_code = replace_field_value(code, target_field, str(part))
            output_lines.append(rebuild_line(new_code, comment))

    return output_lines


def subdivide_single_mode(n: int, digits: int):
    """Prompt for and subdivide one G-code command.

    Args:
        n: Number of commands to generate.
        digits: Maximum number of fractional digits for E values.

    Returns:
        A list containing the subdivided G-code lines.

    Raises:
        ValueError: If the input line is empty or invalid.
    """
    print()
    print("Mode 1 selected: Single command mode")
    print("Please enter one G-code command.")
    print()
    print("Example:")
    print("    G1 F100 E3.083 ; precondition optimizable")
    print()

    line = input("Enter G-code line: ").strip()

    if not line:
        raise ValueError("Input line cannot be empty.")

    return subdivide_gcode_line(line, n=n, digits=digits)


def subdivide_compound_mode(n: int, digits: int):
    """Prompt for and subdivide a compound G1 and G4 block.

    The subdivided extrusion and dwell commands are interleaved as
    repeated G1-G4 pairs.

    Args:
        n: Number of command pairs to generate.
        digits: Maximum number of fractional digits for E values.

    Returns:
        A list containing the interleaved subdivided G-code lines.

    Raises:
        ValueError: If either input line is empty or invalid.
    """
    print()
    print("Mode 2 selected: Compound G1 + G4 mode")
    print("Please enter one G1 extrusion line and one G4 dwell line.")
    print()
    print("Example:")
    print("    G1 F100 E3.083 ; precondition optimizable")
    print("    G4 P2000000 ; optimizable")
    print()

    g1_line = input("Enter G1 line: ").strip()
    g4_line = input("Enter G4 line: ").strip()

    if not g1_line:
        raise ValueError("G1 line cannot be empty.")

    if not g4_line:
        raise ValueError("G4 line cannot be empty.")

    g1_parts = subdivide_gcode_line(g1_line, n=n, digits=digits)
    g4_parts = subdivide_gcode_line(g4_line, n=n, digits=digits)

    output_lines = []

    for i in range(n):
        output_lines.append(g1_parts[i])
        output_lines.append(g4_parts[i])

    return output_lines


# -------------------------------------------------------------------------
# Main program
# -------------------------------------------------------------------------

def main():
    """Run the interactive G-code subdivision workflow.

    The function collects the subdivision mode and parameters, generates
    the subdivided commands, exports them to a G-code file, and displays
    the result in the terminal.

    Raises:
        ValueError: If the selected mode, subdivision count, decimal
            precision, or G-code input is invalid.
        OSError: If the output file cannot be written.
    """
    print("Manual G-code Subdivider")
    print("=" * 70)
    print("Select subdivision mode:")
    print()
    print("1 = Single command mode")
    print("    Use this when only one G-code command should be subdivided.")
    print("    Example:")
    print("        G1 F100 E3.0831 ; precondition optimizable")
    print()
    print("2 = Compound G1 + G4 mode")
    print("    Use this when one extrusion command and one dwell command should be")
    print("    subdivided as a repeated pair.")
    print("    Example:")
    print("        G1 F100 E3.0831 ; precondition optimizable")
    print("        G4 P2000000 ; optimizable")
    print()
    print("Important:")
    print("    These mode descriptions are only shown here in the terminal.")
    print("    They will NOT be written into the exported .gcode file.")
    print("=" * 70)

    mode = input("Mode [1 = single command, 2 = compound G1+G4]: ").strip()

    if mode not in ["1", "2"]:
        raise ValueError("Mode must be 1 or 2.")

    n = int(input("Subdivision number n: ").strip())

    if n <= 0:
        raise ValueError("Subdivision number n must be larger than 0.")

    digits_input = input("Decimal digits for E values [default = 6]: ").strip()
    digits = int(digits_input) if digits_input else 6

    if digits < 0:
        raise ValueError("Decimal digits must be zero or larger.")

    output_file = input("Output file name [default = subdivided_output.gcode]: ").strip()

    if not output_file:
        output_file = "subdivided_n100_2000s.gcode"

    if not output_file.lower().endswith(".gcode"):
        output_file += ".gcode"

    if mode == "1":
        output_lines = subdivide_single_mode(n=n, digits=digits)
    else:
        output_lines = subdivide_compound_mode(n=n, digits=digits)

    # ---------------------------------------------------------------------
    # Export clean G-code only.
    # No mode descriptions, headers, or UI notes are written into the file.
    # ---------------------------------------------------------------------
    with open(output_file, "w", encoding="utf-8", newline="\n") as f:
        for line in output_lines:
            f.write(line + "\n")

    print()
    print("Generated G-code:")
    print("=" * 70)

    for line in output_lines:
        print(line)

    print("=" * 70)
    print(f"File exported to: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()