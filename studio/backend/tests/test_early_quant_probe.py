# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The quant smoke probe's child starts during the torch warm (after dynamo is imported), not after diffusers."""

from __future__ import annotations

import threading

import pytest


@pytest.fixture
def warm(monkeypatch):
    from utils import torch_warmup

    monkeypatch.delenv(torch_warmup.EARLY_PROBE_ENV_VAR, raising = False)
    # _warm() publishes into the module-level status; keep that out of later tests.
    monkeypatch.setattr(
        torch_warmup, "_status", {"started": False, "finished": False, "stages": {}}
    )
    return torch_warmup


def test_probe_is_kicked_right_after_the_dynamo_import_stage(warm, monkeypatch):
    order: list[str] = []
    stages = tuple(
        (name, (lambda n = name: order.append(n)))
        for name in ("hardware", "inference_backend", "transformers", "datasets")
    )
    monkeypatch.setattr(warm, "_STAGES", stages)
    monkeypatch.setattr(warm, "_kick_early_quant_probe", lambda: order.append("probe"))
    monkeypatch.setattr(warm, "_detection_epoch", lambda: None)
    warm._warm(epoch = None)
    assert order == ["hardware", "inference_backend", "probe", "transformers", "datasets"]


def test_kick_runs_the_existing_prewarm_on_a_daemon_thread_only_for_diffusers_installs(
    warm, monkeypatch
):
    calls: list[str] = []
    ran = threading.Event()

    def prewarm() -> None:
        calls.append(threading.current_thread().name)
        ran.set()

    monkeypatch.setattr(warm, "_prewarm_quant_probe", prewarm)
    monkeypatch.setattr(warm, "_a_local_model_would_load_through_diffusers", lambda: True)
    from core.inference import diffusion_probe_cache

    monkeypatch.setattr(diffusion_probe_cache, "has_file", lambda: False)
    thread = warm._kick_early_quant_probe()
    assert thread is not None and thread.daemon
    thread.join(10)
    assert calls == ["early-quant-probe"]

    calls.clear()
    monkeypatch.setattr(warm, "_a_local_model_would_load_through_diffusers", lambda: False)
    warm._kick_early_quant_probe().join(10)
    assert calls == []  # chat-only / training-only installs pay nothing

    def broken() -> bool:
        raise RuntimeError("index unreadable")

    monkeypatch.setattr(warm, "_a_local_model_would_load_through_diffusers", broken)
    warm._kick_early_quant_probe().join(10)
    assert calls == []


def test_a_later_start_with_a_persisted_table_does_not_import_the_probe_during_the_warm(
    warm, monkeypatch
):
    from core.inference import diffusion_probe_cache

    monkeypatch.setattr(warm, "_a_local_model_would_load_through_diffusers", lambda: True)
    monkeypatch.setattr(
        warm, "_prewarm_quant_probe", lambda: pytest.fail("probe ran although its table is on disk")
    )
    monkeypatch.setattr(diffusion_probe_cache, "has_file", lambda: True)
    warm._kick_early_quant_probe().join(10)


def test_probe_cache_file_check(tmp_path, monkeypatch):
    from core.inference import diffusion_probe_cache

    path = tmp_path / "diffusion_quant_probe.json"
    monkeypatch.setattr(diffusion_probe_cache, "_cache_file", lambda: path)
    assert not diffusion_probe_cache.has_file()
    path.write_text("{}")
    assert diffusion_probe_cache.has_file()
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PROBE_CACHE", "0")
    assert not diffusion_probe_cache.has_file()


def test_kill_switch_keeps_the_old_timing(warm, monkeypatch):
    monkeypatch.setenv(warm.EARLY_PROBE_ENV_VAR, "0")
    monkeypatch.setattr(
        warm, "_prewarm_quant_probe", lambda: pytest.fail("probe ran under the kill switch")
    )
    assert warm._kick_early_quant_probe() is None
