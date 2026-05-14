# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Shared helpers for the version-compat suites: fetch a file from
GitHub raw at a specific tag/branch, and grep for class / def / module
symbols without ast.parse so a single non-importable line doesn't
false-fail us. Mirrors tests/vllm_compat/test_vllm_pinned_symbols.py.

Used by:
  - tests/version_compat/test_trl_grpo_pinned_symbols.py
  - tests/version_compat/test_peft_pinned_symbols.py
  - tests/version_compat/test_sentence_transformers_pinned_symbols.py
  - tests/version_compat/test_bitsandbytes_pinned_symbols.py
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request

import pytest


def fetch_text(repo: str, ref: str, path: str) -> str | None:
    """Fetch a file from GitHub raw. None on 404 (the path was renamed
    or removed in this version, which is informational and the caller
    decides whether that's fatal). Skips the test on transient network
    errors so we don't make CI flaky."""
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


def has_def(src: str, name: str, kind: str = "any") -> bool:
    """Heuristic AST-equivalent grep for `class Name`, `def name`,
    or `Name = ...` — at any indent level. We avoid a full ast.parse
    so a single non-importable line (e.g. `# type: ignore` after an
    unresolved alias) doesn't false-fail us. Indented matches are
    accepted because most class methods we want to verify live four
    spaces in (and tests should pass for `class.method` definitions
    just as much as for module-level `def`)."""
    if kind in ("any", "class") and re.search(
        rf"^\s*class\s+{re.escape(name)}\b", src, re.MULTILINE
    ):
        return True
    if kind in ("any", "func") and re.search(
        rf"^\s*(?:async\s+)?def\s+{re.escape(name)}\b", src, re.MULTILINE
    ):
        return True
    if kind == "any" and re.search(rf"^\s*{re.escape(name)}\s*[:=]", src, re.MULTILINE):
        return True
    return False


def first_match(repo: str, ref: str, paths: list[str]) -> tuple[str, str] | None:
    """Try a list of candidate paths; return (path, src) for the first
    one that exists, or None if none do. Useful when upstream split or
    moved a module across versions."""
    for p in paths:
        src = fetch_text(repo, ref, p)
        if src is not None:
            return (p, src)
    return None
