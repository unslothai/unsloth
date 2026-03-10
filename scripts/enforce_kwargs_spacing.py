#!/usr/bin/env python3
"""Ensure keyword arguments use spaces around '=', prune redundant pass statements."""

from __future__ import annotations

import ast
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


def remove_redundant_passes(text: str) -> tuple[str, bool]:
    """Drop pass statements that share a block with other executable code."""

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text, False

    redundant: list[ast.Pass] = []

    def visit(node: ast.AST) -> None:
        for attr in ("body", "orelse", "finalbody"):
            value = getattr(node, attr, None)
            if not isinstance(value, list) or len(value) <= 1:
                continue
            for stmt in value:
                if isinstance(stmt, ast.Pass):
                    redundant.append(stmt)
            for stmt in value:
                if isinstance(stmt, ast.AST):
                    visit(stmt)
        handlers = getattr(node, "handlers", None)
        if handlers:
            for handler in handlers:
                visit(handler)

    visit(tree)

    if not redundant:
        return text, False

    lines = text.splitlines(keepends=True)
    changed = False

    for node in sorted(
        redundant, key=lambda item: (item.lineno, item.col_offset), reverse=True
    ):
        start = node.lineno - 1
        end = (node.end_lineno or node.lineno) - 1
        if start >= len(lines):
            continue
        changed = True
        if start == end:
            line = lines[start]
            col_start = node.col_offset
            col_end = node.end_col_offset or (col_start + 4)
            segment = line[:col_start] + line[col_end:]
            lines[start] = segment if segment.strip() else ""
            continue

        # Defensive fall-back for unexpected multi-line 'pass'.
        prefix = lines[start][: node.col_offset]
        lines[start] = prefix if prefix.strip() else ""
        for idx in range(start + 1, end):
            lines[idx] = ""
        suffix = lines[end][(node.end_col_offset or 0) :]
        lines[end] = suffix

    # Normalise to ensure lines end with newlines except at EOF.
    result_lines: list[str] = []
    for index, line in enumerate(lines):
        if not line:
            continue
        if index < len(lines) - 1 and not line.endswith("\n"):
            result_lines.append(f"{line}\n")
        else:
            result_lines.append(line)

    return "".join(result_lines), changed


def process_file(path: Path) -> bool:
    try:
        with tokenize.open(path) as handle:
            original = handle.read()
            encoding = handle.encoding
    except (OSError, SyntaxError) as exc:  # SyntaxError from tokenize on invalid python
        print(f"Failed to read {path}: {exc}", file=sys.stderr)
        return False

    updated, changed = enforce_spacing(original)
    updated, removed = remove_redundant_passes(updated)
    if changed or removed:
        path.write_text(updated, encoding=encoding)
        return True
    return False


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
