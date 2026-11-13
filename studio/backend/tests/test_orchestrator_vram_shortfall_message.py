# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest


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


@pytest.mark.parametrize("settled", [False, True])
@pytest.mark.parametrize("telemetry_available", [False, True])
def test_load_uses_shortfall_message_only_after_failed_settle(
    monkeypatch, settled, telemetry_available
):
    from core.inference import orchestrator as module
    from utils import hardware, transformers_version
    from utils.hardware import gpu_query

    cls = module.InferenceOrchestrator
    backend = cls.__new__(cls)
    backend.__dict__.update(
        loading_models = set(),
        models = {},
        active_model_name = None,
        _proc = None,
        _subprocess_shutdown_lock = threading.RLock(),
        _load_download_keys = [],
    )
    events = []

    def telemetry():
        events.append("telemetry")
        if not telemetry_available:
            raise RuntimeError("telemetry unavailable")
        return {"devices": [{"index": 0, "vram_total_gb": 7.93, "vram_used_gb": 6.2}]}

    def wait(_since, **kwargs):
        assert kwargs["expected_free_gb"] == {0: 3.9}
        events.append("settle")
        return settled

    monkeypatch.setattr(hardware, "get_visible_gpu_utilization", telemetry)
    monkeypatch.setattr(gpu_query, "fresh_reads", nullcontext)
    monkeypatch.setattr(module, "get_device", lambda: SimpleNamespace(value = "cpu"))
    monkeypatch.setattr(module, "prepare_gpu_selection", lambda *a, **k: ([0], {}))
    monkeypatch.setattr(transformers_version, "needs_transformers_5", lambda _: False)
    monkeypatch.setattr(transformers_version, "sidecar_swap_kind", lambda: None)
    monkeypatch.setattr(backend, "_ensure_subprocess_alive", lambda: False)
    monkeypatch.setattr(backend, "_shutdown_subprocess", lambda **k: True)
    monkeypatch.setattr(backend, "_wait_for_worker_vram_settle", wait)
    monkeypatch.setattr(backend, "_spawn_subprocess", lambda _: events.append("spawn"))
    monkeypatch.setattr(backend, "_wait_response", lambda *a: {"success": True})

    def load():
        return backend.load_model(
            SimpleNamespace(identifier = "test/replacement"),
            post_handoff_expected_free_gb = {0: 3.9},
            on_prior_worker_released = lambda: events.append("released"),
        )

    if settled:
        assert load() is True
        assert events == ["settle", "released", "spawn"]
        assert backend.active_model_name == "test/replacement"
    else:
        with pytest.raises(RuntimeError) as error:
            load()
        text = str(error.value)
        assert "previous inference worker" not in text
        assert "A reply may still be finishing" in text
        if telemetry_available:
            assert "free 1.7 GiB of the 3.9 GiB needed on cuda:0" in text
        assert events == ["settle", "telemetry"]
        assert backend.active_model_name is None
        assert backend.models == {}
