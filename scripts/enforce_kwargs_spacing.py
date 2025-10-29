#!/usr/bin/env python3
"""Ensure keyword arguments use spaces around '=' (e.g., foo = bar)."""

from __future__ import annotations

import argparse
import io
import sys
import tokenize
from collections import defaultdict
from pathlib import Path


def enforce_spacing(text: str) -> tuple[str, bool]:
    """Return updated text with keyword '=' padded by spaces, plus change flag."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return text, False

    offsets: dict[int, int] = defaultdict(int)
    changed = False

    reader = io.StringIO(text).readline
    for token in tokenize.generate_tokens(reader):
        if token.type != tokenize.OP or token.string != "=":
            continue

        line_index = token.start[0] - 1
        col = token.start[1] + offsets[line_index]

        if line_index < 0 or line_index >= len(lines):
            continue

        line = lines[line_index]
        if col >= len(line) or line[col] != "=":
            continue

        line_changed = False

        # Insert a space before '=' when missing and not preceded by whitespace.
        if col > 0 and line[col - 1] not in {" ", "\t"}:
            line = f"{line[:col]} {line[col:]}"
            offsets[line_index] += 1
            col += 1
            line_changed = True
            changed = True

        # Insert a space after '=' when missing and not followed by whitespace or newline.
        next_index = col + 1
        if next_index < len(line) and line[next_index] not in {" ", "\t", "\n", "\r"}:
            line = f"{line[:next_index]} {line[next_index:]}"
            offsets[line_index] += 1
            line_changed = True
            changed = True

        if line_changed:
            lines[line_index] = line

    if not changed:
        return text, False

    return "".join(lines), True


def process_file(path: Path) -> bool:
    try:
        with tokenize.open(path) as handle:
            original = handle.read()
            encoding = handle.encoding
    except (OSError, SyntaxError) as exc:  # SyntaxError from tokenize on invalid python
        print(f"Failed to read {path}: {exc}", file=sys.stderr)
        return False

    updated, changed = enforce_spacing(original)
    if changed:
        path.write_text(updated, encoding=encoding)
    return changed


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="Python files to fix")
    args = parser.parse_args(argv)

    touched: list[Path] = []
    self_path = Path(__file__).resolve()

    for entry in args.files:
        path = Path(entry)
        # Skip modifying this script to avoid self-edit loops.
        if path.resolve() == self_path:
            continue
        if not path.exists() or path.is_dir():
            continue
        if process_file(path):
            touched.append(path)

    if touched:
        for path in touched:
            print(f"Adjusted kwarg spacing in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
