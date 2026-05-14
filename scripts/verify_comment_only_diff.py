# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""Deterministic comment / docstring-only verifier.

Compares a list of changed files between two git refs and reports whether
each diff is strictly comments / docstrings (Python) or comments
(YAML / GitHub Actions). Useful for gating a "comment trim" /
"docstring refactor" PR against accidental code drift.

Per .py file: parse both revs into AST, strip module / class / function
docstrings, then compare ast.unparse output. Pure Python comments are
discarded by the parser by construction, so any post-strip diff is real
code. Per .yml file: yaml.safe_load both sides and compare the parsed
Python object; if scalar values differ, also strip shell comments inside
``run: |`` block bodies before comparing. Exit code 0 = all OK, 1 = at
least one file has a real (non-comment) diff or an error.

Usage:
    python scripts/verify_comment_only_diff.py [--base REF] [--head REF] path ...

Defaults: --base origin/main, --head HEAD. Paths are repo-relative.

Example:
    git diff --name-only origin/main..HEAD \\
      | xargs python scripts/verify_comment_only_diff.py --base origin/main
"""

from __future__ import annotations

import argparse
import ast
import difflib
import subprocess
import sys
from typing import Any

import yaml


def _git_show(rev: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{rev}:{path}"],
        text = True,
        stderr = subprocess.DEVNULL,
    )


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove every string-literal docstring (Module / FunctionDef /
    AsyncFunctionDef / ClassDef). Empty body becomes ``pass`` so
    ast.unparse stays valid."""
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = body[1:]
                if not node.body:
                    node.body = [ast.Pass()]
    return tree


def _normalize_py(src: str) -> str:
    tree = ast.parse(src)
    tree = _strip_docstrings(tree)
    return ast.unparse(tree)


def _strip_shell_comments(s: str) -> str:
    """Strip pure-comment lines and inline trailing comments from a shell
    snippet, then collapse runs of blank lines. Heuristic only: leaves a
    line untouched if it has an odd quote count (open string)."""
    out = []
    for line in s.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        has_single = line.count("'") % 2 == 0
        has_double = line.count('"') % 2 == 0
        if has_single and has_double:
            idx = line.find(" #")
            if idx >= 0:
                line = line[:idx].rstrip()
        out.append(line)
    norm = []
    prev_blank = False
    for line in out:
        if line.strip() == "":
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        norm.append(line)
    return "\n".join(norm).strip()


def _normalize_yaml_run_strings(obj: Any) -> Any:
    """Walk the parsed YAML object; for any multi-line string (i.e. a
    ``run: |`` script body), strip shell comments. Returns a normalised
    copy."""
    if isinstance(obj, dict):
        return {k: _normalize_yaml_run_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_yaml_run_strings(x) for x in obj]
    if isinstance(obj, str) and "\n" in obj:
        return _strip_shell_comments(obj)
    return obj


def _walk_yaml_diff(b: Any, a: Any, prefix: str = "") -> None:
    """Print a path-keyed summary of the first structural / scalar diff."""
    if type(b) is not type(a):
        print(
            f"     type-diff at {prefix or '/'}: "
            f"{type(b).__name__} -> {type(a).__name__}",
        )
        return
    if isinstance(b, dict):
        keys = sorted((set(b.keys()) | set(a.keys())), key = lambda x: str(x))
        for k in keys:
            if k not in b:
                print(f"     added key {prefix}/{k}")
            elif k not in a:
                print(f"     removed key {prefix}/{k}")
            else:
                _walk_yaml_diff(b[k], a[k], f"{prefix}/{k}")
    elif isinstance(b, list):
        if len(b) != len(a):
            print(
                f"     list len at {prefix or '/'}: " f"{len(b)} -> {len(a)}",
            )
        for i, (bi, ai) in enumerate(zip(b, a)):
            _walk_yaml_diff(bi, ai, f"{prefix}[{i}]")
    elif b != a:
        bs = repr(b)[:300]
        as_ = repr(a)[:300]
        print(f"     scalar at {prefix or '/'}:")
        print(f"       before: {bs}")
        print(f"       after:  {as_}")


def _verify_python(path: str, before: str, after: str) -> bool:
    try:
        norm_before = _normalize_py(before)
        norm_after = _normalize_py(after)
    except SyntaxError as exc:
        print(f"FAIL {path}: SyntaxError parsing -- {exc}")
        return False
    if norm_before == norm_after:
        print(f"OK   {path}  (AST identical after docstring strip)")
        return True
    diff = list(
        difflib.unified_diff(
            norm_before.splitlines(),
            norm_after.splitlines(),
            fromfile = f"{path}@before",
            tofile = f"{path}@after",
            n = 2,
        )
    )
    print(f"FAIL {path}: AST differs after docstring strip:")
    for line in diff[:40]:
        print(f"     {line}")
    return False


def _verify_yaml(path: str, before: str, after: str) -> bool:
    try:
        raw_before = yaml.safe_load(before)
        raw_after = yaml.safe_load(after)
    except yaml.YAMLError as exc:
        print(f"FAIL {path}: YAML parse error -- {exc}")
        return False
    if raw_before == raw_after:
        print(f"OK   {path}  (YAML parsed object identical)")
        return True
    norm_before = _normalize_yaml_run_strings(raw_before)
    norm_after = _normalize_yaml_run_strings(raw_after)
    if norm_before == norm_after:
        print(
            f"OK   {path}  (YAML parsed object identical after "
            f"stripping shell comments from run: bodies)",
        )
        return True
    print(
        f"FAIL {path}: YAML parsed objects still differ after stripping "
        f"shell comments from `run:` bodies.",
    )
    _walk_yaml_diff(norm_before, norm_after)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description = "Verify each path's diff between BASE and HEAD is "
        "strictly comments / docstrings.",
    )
    parser.add_argument("--base", default = "origin/main", help = "base git ref")
    parser.add_argument("--head", default = "HEAD", help = "head git ref")
    parser.add_argument("paths", nargs = "+", help = "repo-relative paths")
    args = parser.parse_args(argv)

    rc = 0
    print(f"Comparing {len(args.paths)} files: {args.base} vs {args.head}\n")
    for path in args.paths:
        try:
            before = _git_show(args.base, path)
            after = _git_show(args.head, path)
        except subprocess.CalledProcessError as exc:
            print(f"SKIP {path}: {exc}")
            continue

        if path.endswith(".py"):
            if not _verify_python(path, before, after):
                rc = 1
        elif path.endswith((".yml", ".yaml")):
            if not _verify_yaml(path, before, after):
                rc = 1
        else:
            print(f"NOTE {path}: not .py or .yaml -- skipped automated check.")

    return rc


if __name__ == "__main__":
    sys.exit(main())
