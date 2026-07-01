# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Shared helpers for version-compat suites: GitHub raw fetch + AST symbol lookup with regex fallback."""

from __future__ import annotations

import ast
import functools
import os
import re
import urllib.error
import urllib.request

import pytest


@functools.lru_cache(maxsize = None)
def fetch_text(repo: str, ref: str, path: str) -> str | None:
    """Fetch a file from GitHub raw. None on 404; skips on transient network errors."""
    url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"
    req = urllib.request.Request(url)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout = 15) as r:
            return r.read().decode("utf-8", errors = "replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        pytest.skip(f"GitHub fetch failed ({e.code}) for {url}")
    except (urllib.error.URLError, TimeoutError) as e:
        pytest.skip(f"GitHub fetch failed ({e}) for {url}")


def first_match(repo: str, ref: str, paths: list[str]) -> tuple[str, str] | None:
    """Return (path, src) for the first existing candidate path, else None."""
    for p in paths:
        src = fetch_text(repo, ref, p)
        if src is not None:
            return (p, src)
    return None


def _parse(src: str) -> ast.Module | None:
    try:
        return ast.parse(src)
    except SyntaxError:
        return None


def _walk_module_stmts(stmts):
    # Descend into If/Try/With (gated bindings) but not into def/class (new scope).
    for node in stmts:
        yield node
        if isinstance(node, ast.If):
            yield from _walk_module_stmts(node.body)
            yield from _walk_module_stmts(node.orelse)
        elif isinstance(node, ast.Try):
            yield from _walk_module_stmts(node.body)
            for h in node.handlers:
                yield from _walk_module_stmts(h.body)
            yield from _walk_module_stmts(node.orelse)
            yield from _walk_module_stmts(node.finalbody)
        elif isinstance(node, ast.With):
            yield from _walk_module_stmts(node.body)


def _module_bindings(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in _walk_module_stmts(tree.body):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                out.add(a.asname or a.name.split(".")[0])
    return out


def _re_is_bound(src: str, name: str) -> bool:
    n = re.escape(name)
    patterns = (
        rf"^\s*class\s+{n}\b",
        rf"^\s*(?:async\s+)?def\s+{n}\b",
        rf"^\s*{n}\s*[:=]",
        rf"^\s*from\s+\S+\s+import\s+[^#\n]*\b{n}\b",
        rf"^\s*import\s+[^#\n]*\b{n}\b",
        rf"\bas\s+{n}\b",
    )
    return any(re.search(p, src, re.MULTILINE) for p in patterns)


def _re_has_def(src: str, name: str, kind: str) -> bool:
    n = re.escape(name)
    if kind in ("any", "class") and re.search(rf"^\s*class\s+{n}\b", src, re.MULTILINE):
        return True
    if kind in ("any", "func") and re.search(rf"^\s*(?:async\s+)?def\s+{n}\b", src, re.MULTILINE):
        return True
    return kind == "any" and _re_is_bound(src, name)


def _re_function_params(src: str, name: str) -> tuple[str, ...] | None:
    n = re.escape(name)
    m = re.search(rf"^\s*(?:async\s+)?def\s+{n}\s*\(([^)]*)\)", src, re.MULTILINE | re.DOTALL)
    if m is None:
        return None
    out: list[str] = []
    for part in m.group(1).split(","):
        p = part.strip().lstrip("*")
        for sep in (":", "="):
            i = p.find(sep)
            if i >= 0:
                p = p[:i]
        p = p.strip()
        if p and p != "/":
            out.append(p)
    return tuple(out)


def is_bound(src: str, name: str) -> bool:
    """True if `name` is bound at module scope (def/class/assign/import-as)."""
    tree = _parse(src)
    if tree is None:
        return _re_is_bound(src, name)
    return name in _module_bindings(tree)


def has_def(
    src: str,
    name: str,
    kind: str = "any",
) -> bool:
    """`class name` / `def name` at any depth; with kind="any" also any module binding."""
    tree = _parse(src)
    if tree is None:
        return _re_has_def(src, name, kind)
    want_class = kind in ("any", "class")
    want_func = kind in ("any", "func")
    for node in ast.walk(tree):
        if want_class and isinstance(node, ast.ClassDef) and node.name == name:
            return True
        if (
            want_func
            and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return True
    return kind == "any" and name in _module_bindings(tree)


def function_params(
    src: str,
    name: str,
    *,
    cls: str | None = None,
) -> tuple[str, ...] | None:
    """Param names of the first `def name(...)`; scoped to `class cls:` if given. None if absent."""
    tree = _parse(src)
    if tree is None:
        # cls scoping isn't reliable via regex; degrade to module-level lookup.
        return _re_function_params(src, name) if cls is None else None
    scope: ast.AST | None = tree
    if cls is not None:
        scope = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls),
            None,
        )
        if scope is None:
            return None
    for node in ast.walk(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            a = node.args
            return tuple(p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs))
    return None
