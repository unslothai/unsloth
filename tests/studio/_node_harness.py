# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared plumbing for the ``node --experimental-strip-types`` source harnesses.

``studio/frontend`` carries no JS test runner, so frontend behaviours are pinned by slicing
the real source VERBATIM into a harness module and running it under node; only the fixtures
the sliced code reads through are hand-written. Harness and runner go into a per-invocation
``mkdtemp(prefix = "run")``, so concurrent tests share no file.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]


def source_path(relative_path: str) -> Path:
    """Locate a studio source both in this repo and in a vendored checkout."""
    direct = WORKDIR / relative_path
    if direct.exists():
        return direct
    return WORKDIR / "unsloth_repo" / relative_path


def read(path: Path) -> str:
    return path.read_text(encoding = "utf-8")


def slice_between(text: str, start_marker: str, end_marker: str) -> str:
    """The source from ``start_marker`` up to (not including) ``end_marker``."""
    start = text.index(start_marker)
    end = text.index(end_marker, start + len(start_marker))
    return text[start:end]


def require_node(sources: Iterable[Path]) -> None:
    """Skip unless node can strip types and every sliced source is present."""
    if shutil.which("node") is None:
        pytest.skip("node not available")
    for path in sources:
        if not Path(path).exists():
            pytest.skip("studio chat sources not present")
    probe = subprocess.run(
        ["node", "--experimental-strip-types", "--version"],
        capture_output = True,
        text = True,
        timeout = 30,
    )
    if probe.returncode != 0:
        pytest.skip("node --experimental-strip-types not available")


def run_harness(temp_root: Path, harness_source: str, script: str) -> dict:
    """Run ``script`` against ``harness_source`` and parse its last stdout line."""
    temp_root.mkdir(parents = True, exist_ok = True)
    workdir = Path(tempfile.mkdtemp(prefix = "run", dir = str(temp_root)))
    (workdir / "harness.ts").write_text(harness_source, encoding = "utf-8")
    (workdir / "run.mts").write_text(script, encoding = "utf-8")
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--no-warnings", "run.mts"],
        cwd = str(workdir),
        capture_output = True,
        text = True,
        timeout = 60,
        env = dict(os.environ, NODE_NO_WARNINGS = "1"),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    return json.loads(lines[-1])
