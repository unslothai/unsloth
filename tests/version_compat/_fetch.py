# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Shared helpers for version-compat suites: GitHub raw fetch + regex symbol grep."""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request

import pytest


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


def has_def(
    src: str,
    name: str,
    kind: str = "any",
) -> bool:
    """Grep for `class Name`, `def name`, or `Name = ...` at any indent (no ast.parse)."""
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
    """Return (path, src) for the first existing candidate path, else None."""
    for p in paths:
        src = fetch_text(repo, ref, p)
        if src is not None:
            return (p, src)
    return None
