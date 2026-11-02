# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def _orchestrator_class():
    spec = importlib.util.spec_from_file_location(
        "inference_orchestrator_shortfall_under_test",
        Path(__file__).resolve().parent.parent / "core/inference/orchestrator.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InferenceOrchestrator


def _devices(monkeypatch, devices):
    import utils.hardware as hardware
    from utils.hardware import gpu_query

    monkeypatch.setattr(hardware, "get_visible_gpu_utilization", lambda: {"devices": devices})
    monkeypatch.setattr(gpu_query, "fresh_reads", lambda: nullcontext())


def test_shortfall_reports_free_and_needed(monkeypatch):
    _devices(monkeypatch, [{"index": 0, "vram_total_gb": 7.93, "vram_used_gb": 6.2}])

    text = _orchestrator_class()._describe_vram_shortfall({0: 3.9})

    assert "free 1.7 GiB" in text
    assert "3.9 GiB needed on cuda:0" in text


def test_shortfall_covers_every_gpu_that_was_expected(monkeypatch):
    _devices(
        monkeypatch,
        [
            {"index": 0, "vram_total_gb": 8.0, "vram_used_gb": 6.0},
            {"index": 1, "vram_total_gb": 8.0, "vram_used_gb": 1.0},
        ],
    )

    text = _orchestrator_class()._describe_vram_shortfall({0: 3.9, 1: 3.9})

    assert "cuda:0" in text and "cuda:1" in text
    assert text.count("GiB needed") == 2


def test_shortfall_is_empty_when_telemetry_raises(monkeypatch):
    import utils.hardware as hardware

    def boom():
        raise RuntimeError("no GPU telemetry here")

    monkeypatch.setattr(hardware, "get_visible_gpu_utilization", boom)

    assert _orchestrator_class()._describe_vram_shortfall({0: 3.9}) == ""


def test_shortfall_ignores_gpus_that_were_not_expected(monkeypatch):
    _devices(monkeypatch, [{"index": 2, "vram_total_gb": 8.0, "vram_used_gb": 7.0}])

    assert _orchestrator_class()._describe_vram_shortfall({0: 3.9}) == ""


def test_message_names_both_holders_and_carries_the_measurement(monkeypatch):
    _devices(monkeypatch, [{"index": 0, "vram_total_gb": 7.93, "vram_used_gb": 6.2}])

    text = _orchestrator_class()._vram_wait_message({0: 3.9})

    assert text.startswith("GPU memory has not freed up yet")
    assert "free 1.7 GiB of the 3.9 GiB needed on cuda:0" in text
    assert "A reply may still be finishing" in text
    assert "another app" in text
    assert "Retry shortly." in text
    assert "previous inference worker" not in text


def test_message_still_reads_well_without_telemetry(monkeypatch):
    import utils.hardware as hardware

    def boom():
        raise RuntimeError("no GPU telemetry here")

    monkeypatch.setattr(hardware, "get_visible_gpu_utilization", boom)

    text = _orchestrator_class()._vram_wait_message({0: 3.9})

    assert text.startswith("GPU memory has not freed up yet")
    assert "(" not in text.split("model")[0]
    assert "Retry shortly." in text
