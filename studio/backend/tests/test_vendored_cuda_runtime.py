# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for locating a CUDA runtime another application ships privately.

The prebuilt installer counts those directories when it picks a CUDA runtime
line (prebuilt_core.py: linux_runtime_dirs_for_required_libraries), so the
launchers have to put the matching one back on LD_LIBRARY_PATH. Pins the
match rule (exact CUDA major, complete runtime) and the refusals.
"""

from __future__ import annotations

import sys

import pytest

from utils.prebuilt.runtime_libs import vendored_cuda_runtime_dirs

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason = "the vendored roots are Linux paths"
)


def _make_runtime(root, name: str, *, cudart: bool = True, cublas: bool = True, major = "13"):
    directory = root / name
    directory.mkdir(parents = True)
    if cudart:
        (directory / f"libcudart.so.{major}.0.48").write_bytes(b"")
    if cublas:
        (directory / f"libcublas.so.{major}.0.1").write_bytes(b"")
    return directory


def _roots(tmp_path):
    return ((tmp_path, "cuda_v{major}"),)


def test_matches_the_marker_runtime_line(tmp_path):
    runtime_dir = _make_runtime(tmp_path, "cuda_v13")

    result = vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path))

    assert result == [str(runtime_dir.resolve())]


def test_ignores_a_different_cuda_major(tmp_path):
    _make_runtime(tmp_path, "cuda_v13")
    _make_runtime(tmp_path, "cuda_v12", major = "12")

    result = vendored_cuda_runtime_dirs({"runtime_line": "cuda12"}, roots = _roots(tmp_path))

    assert result == [str((tmp_path / "cuda_v12").resolve())]


def test_accepts_a_minor_qualified_directory(tmp_path):
    runtime_dir = _make_runtime(tmp_path, "cuda_v13.0")

    result = vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path))

    assert result == [str(runtime_dir.resolve())]


def test_rejects_a_longer_major_that_shares_the_prefix(tmp_path):
    # cuda_v130 answers a cuda_v13* glob but is not CUDA 13.
    _make_runtime(tmp_path, "cuda_v130", major = "130")

    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == []


@pytest.mark.parametrize("missing", ["cudart", "cublas"])
def test_requires_a_complete_runtime(tmp_path, missing):
    _make_runtime(tmp_path, "cuda_v13", cudart = missing != "cudart", cublas = missing != "cublas")

    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == []


def test_ignores_a_file_named_like_a_runtime_dir(tmp_path):
    (tmp_path / "cuda_v13").write_bytes(b"")

    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == []


@pytest.mark.parametrize(
    "marker",
    [
        None,  # source build / unreadable or corrupt marker
        [],  # valid JSON, wrong shape
        "cuda13",
        {},  # marker without a runtime line
        {"runtime_line": None},
        {"runtime_line": 13},  # not a string
        {"runtime_line": "cpu"},
        {"runtime_line": "vulkan"},
    ],
)
def test_yields_nothing_without_a_cuda_runtime_line(tmp_path, marker):
    _make_runtime(tmp_path, "cuda_v13")

    assert vendored_cuda_runtime_dirs(marker, roots = _roots(tmp_path)) == []


def test_missing_root_is_not_an_error(tmp_path):
    roots = ((tmp_path / "nonexistent", "cuda_v{major}"),)

    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = roots) == []
