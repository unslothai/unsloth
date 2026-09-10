# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/system/disk: one syscall, at the volume the downloads land on.

The low-disk notice used to poll /api/system every 60 s from the app shell. That route
enumerates GPUs, reads package metadata and samples CPU, so it ran that work forever in every
open tab to answer a question whose answer only changes when something writes to the disk. The
notice now asks for this route instead, once on mount and once on the way into a download.

Which means this route has to be cheap enough to sit in front of a download, and has to report
the volume the bytes are actually going to, which is not the filesystem root whenever the model
cache lives on another disk.
"""

from __future__ import annotations

import ast
import shutil
import time
from pathlib import Path

import pytest

ROUTE_SOURCE = Path(__file__).resolve().parents[1] / "main.py"


def _route_source() -> str:
    src = ROUTE_SOURCE.read_text(encoding = "utf-8")
    start = src.index('@app.get("/api/system/disk")')
    return src[start : src.index('@app.get("/api/system/gpu-visibility")', start)]


def _route_body() -> ast.FunctionDef:
    """The route function, parsed. Structure, not text: a comment naming psutil is fine, a call
    to it is not, and the two are indistinguishable to a grep."""
    module = ast.parse(_route_source().split("\n", 1)[1])
    function = module.body[0]
    assert isinstance(function, ast.FunctionDef)
    return function


def test_the_route_walks_nothing():
    """A directory walk here would put seconds in front of every download start.

    cache_inventory measures cache sizes by walking, and its own comments call that "seconds on
    a large uv or triton cache". None of that may be reachable from this route.
    """
    called = {
        node.func.attr
        for node in ast.walk(_route_body())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "disk_usage" in called
    for banned in ("walk", "scandir", "rglob", "iterdir", "glob"):
        assert banned not in called, f"the disk route calls {banned}"

    names = {node.id for node in ast.walk(_route_body()) if isinstance(node, ast.Name)}
    assert "psutil" not in names
    # Imported inside the function, so the resolver is not pulled in at module import time.
    imported = {
        alias.name
        for node in ast.walk(_route_body())
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert (
        "hf_default_cache_dir" in imported
    ), "the route reports the filesystem root, not the models volume"


def test_a_disk_reading_is_microseconds(tmp_path):
    """The claim the design rests on, measured rather than asserted.

    shutil.disk_usage is statvfs on Linux and macOS and GetDiskFreeSpaceExW on Windows. The
    budget is deliberately loose, three orders of magnitude above what it costs here, so this
    fails on a genuinely slow path (a stalled network mount) and not on a busy runner.
    """
    shutil.disk_usage(tmp_path)  # warm any lazy loading, so the first call is not the sample
    started = time.perf_counter()
    for _ in range(200):
        shutil.disk_usage(tmp_path)
    per_call_ms = (time.perf_counter() - started) / 200 * 1000
    assert per_call_ms < 5.0, f"disk_usage took {per_call_ms:.2f} ms per call"


def test_a_missing_cache_dir_falls_back_to_a_real_ancestor(tmp_path):
    """disk_usage raises on a path that does not exist, and the model cache legitimately does
    not exist yet on a fresh install. The route walks up to the first ancestor that does."""
    missing = tmp_path / "not" / "created" / "yet"
    with pytest.raises(OSError):
        shutil.disk_usage(missing)

    for candidate in (missing, *missing.parents):
        try:
            usage = shutil.disk_usage(candidate)
        except (OSError, ValueError):
            continue
        assert usage.total > 0
        assert candidate in missing.parents
        break
    else:
        pytest.fail("no ancestor of a tmp_path was readable")


def test_an_unreadable_host_reports_null_not_zero():
    """Zero is a different claim from unknown.

    The frontend's diskPressure() reads a zero total as the host having failed and a zero free
    as a full disk. A route that answered 0 for "I could not tell" would warn every user on a
    platform where the probe fails.
    """
    source = _route_source()
    assert '"total_gb": None' in source
    assert '"free_gb": None' in source
