# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic tests for the standalone diffusion benchmark report."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "diffusion_bench", _REPO_ROOT / "scripts" / "diffusion_bench.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_DIFFUSION_BENCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DIFFUSION_BENCH)


@pytest.mark.parametrize(
    ("psnr", "expected_exit"),
    ((math.inf, 0), (-math.inf, 1), (math.nan, 1)),
)
def test_compare_report_serializes_non_finite_psnr_as_strict_json(
    tmp_path, monkeypatch, psnr, expected_exit
):
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference")
    baseline = {
        "env": {
            "gpu_name": "test-gpu",
            "status": {"device": "cuda", "dtype": "bfloat16"},
        },
        "generate": {
            "median_latency_s": 1.0,
            "peak_vram_bytes": 100,
            "host_rss": {"post_warmup_growth_bytes": 0},
        },
        "accuracy": {"reference_png": str(reference)},
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))
    current = {
        "env": {"status": {"device": "cuda", "dtype": "bfloat16"}},
        "generate": {
            "median_latency_s": 1.0,
            "peak_vram_bytes": 100,
            "host_rss": {"post_warmup_growth_bytes": 0},
        },
    }
    monkeypatch.setattr(_DIFFUSION_BENCH, "_gpu_name", lambda: "test-gpu")
    monkeypatch.setattr(_DIFFUSION_BENCH, "_run", lambda args: current)
    monkeypatch.setattr(_DIFFUSION_BENCH, "_psnr", lambda ref, candidate: psnr)
    out_dir = tmp_path / "out"
    args = SimpleNamespace(
        compare = str(baseline_path),
        out_dir = str(out_dir),
        force_compare = False,
        max_latency_regression = 0.1,
        max_vram_regression = 0.1,
        max_host_rss_growth_mib = 1.0,
        min_psnr = 30.0,
    )

    assert _DIFFUSION_BENCH._compare(args) == expected_exit
    report_text = (out_dir / "compare.json").read_text()
    report = json.loads(
        report_text,
        parse_constant = lambda token: pytest.fail(f"non-standard JSON constant: {token}"),
    )
    assert report["comparison"]["psnr_db"] is None


def test_process_rss_is_plausible_or_absent():
    value = _DIFFUSION_BENCH._process_rss_bytes()
    assert value is None or (isinstance(value, int) and 2**20 < value < 2**44)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "/proc fallback is Linux only")
def test_process_rss_falls_back_to_proc_without_psutil(monkeypatch):
    """The /proc branch is unreachable wherever psutil is installed, which is everywhere the
    benchmark normally runs, so drive it directly rather than leaving it unexercised."""
    real_import = __import__

    def without_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("psutil unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", without_psutil)
    value = _DIFFUSION_BENCH._process_rss_bytes()
    assert isinstance(value, int) and value > 0


@pytest.mark.parametrize("name", ["compare.json", "compare.png"])
def test_compare_refuses_to_overwrite_its_own_baseline(tmp_path, monkeypatch, name):
    """--write-baseline accepts any path, so a baseline can legitimately be sitting on one of the
    names the compare run writes. Refuse instead of destroying the reference metrics, and refuse
    BEFORE the generation rather than after paying for it.

    _run and _psnr are stubbed so that without the guard this reaches the real write and
    clobbers the baseline; otherwise the case would pass for the wrong reason.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference")
    baseline_path = out_dir / name
    baseline_path.write_text(
        json.dumps(
            {
                "env": {"gpu_name": "test-gpu", "status": {"device": "cuda", "dtype": "bfloat16"}},
                "generate": {
                    "median_latency_s": 1.0,
                    "peak_vram_bytes": 100,
                    "host_rss": {"post_warmup_growth_bytes": 0},
                },
                "accuracy": {"reference_png": str(reference)},
            }
        )
    )
    before = baseline_path.read_text()

    monkeypatch.setattr(_DIFFUSION_BENCH, "_gpu_name", lambda: "test-gpu")
    monkeypatch.setattr(
        _DIFFUSION_BENCH,
        "_run",
        lambda args: {
            "env": {"status": {"device": "cuda", "dtype": "bfloat16"}},
            "generate": {
                "median_latency_s": 1.0,
                "peak_vram_bytes": 100,
                "host_rss": {"post_warmup_growth_bytes": 0},
            },
        },
    )
    monkeypatch.setattr(_DIFFUSION_BENCH, "_psnr", lambda ref, candidate: 99.0)

    args = SimpleNamespace(
        compare = str(baseline_path),
        out_dir = str(out_dir),
        force_compare = False,
        max_latency_regression = 0.1,
        max_vram_regression = 0.1,
        max_host_rss_growth_mib = None,
        min_psnr = 30.0,
    )
    assert _DIFFUSION_BENCH._compare(args) == 2
    assert baseline_path.read_text() == before, "the baseline was overwritten"


def test_host_rss_guard_is_off_unless_requested():
    """Back-compat for existing scripted callers: the new threshold only engages when asked
    for, so an invocation written before this flag existed behaves exactly as it did."""
    args = _DIFFUSION_BENCH._build_parser().parse_args(["--write-baseline", "out.json"])
    assert args.max_host_rss_growth_mib is None
