# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the on-disk CUDA runtime-line detection.

The prebuilt CUDA bundles do not ship libcudart/libcublas, so selection must only
offer a cuda<major> line whose runtime libraries are actually present. This pins
the real filesystem scan (a controlled temp lib dir) so the glob actually checks
for a match rather than reporting every major present.
"""

from __future__ import annotations

import sys
from pathlib import Path

_studio = Path(__file__).resolve().parent.parent.parent
if str(_studio) not in sys.path:
    sys.path.insert(0, str(_studio))

from backend.utils.prebuilt import runtime_libs as rl  # noqa: E402


def _isolate(monkeypatch):
    """Neutralise every runtime-dir source except CUDA_RUNTIME_LIB_DIR so the scan
    sees only the test's temp dir (not the host's real CUDA install)."""
    monkeypatch.setattr(rl, "python_runtime_dirs", lambda: [])
    monkeypatch.setattr(rl, "ldconfig_runtime_dirs", lambda required: [])
    monkeypatch.setattr(rl, "glob_paths", lambda *patterns: [])
    for var in ("LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        monkeypatch.delenv(var, raising = False)


def test_detected_linux_runtime_lines_matches_only_present_major(tmp_path, monkeypatch):
    libdir = tmp_path / "cuda13"
    libdir.mkdir()
    (libdir / "libcudart.so.13").write_text("x")
    (libdir / "libcublas.so.13").write_text("x")
    monkeypatch.setenv("CUDA_RUNTIME_LIB_DIR", str(libdir))
    _isolate(monkeypatch)
    lines = rl.detected_linux_runtime_lines()
    assert lines == ["cuda13"]  # only cuda13 libs on disk -> not cuda12/14/...


def test_detected_linux_runtime_lines_requires_both_libs(tmp_path, monkeypatch):
    # libcudart present but libcublas missing -> the line is NOT usable.
    libdir = tmp_path / "partial"
    libdir.mkdir()
    (libdir / "libcudart.so.13").write_text("x")
    monkeypatch.setenv("CUDA_RUNTIME_LIB_DIR", str(libdir))
    _isolate(monkeypatch)
    assert rl.detected_linux_runtime_lines() == []


def test_detected_linux_runtime_lines_empty_when_nothing_present(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("CUDA_RUNTIME_LIB_DIR", str(empty))
    _isolate(monkeypatch)
    assert rl.detected_linux_runtime_lines() == []


def test_glob_hit_helper(tmp_path):
    (tmp_path / "libcudart.so.13").write_text("x")
    assert rl._glob_hit([str(tmp_path)], "libcudart.so.13*") is True
    assert rl._glob_hit([str(tmp_path)], "libcudart.so.12*") is False
    assert rl._glob_hit([], "libcudart.so.13*") is False
