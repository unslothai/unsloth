# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 streamed tiers: rotating VAEs park on render return; diffusers' evict-all warning is muted."""

from __future__ import annotations

import logging
import pathlib
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import video_minimax_h3_residency as residency  # noqa: E402

_GB = 1024**3


class _Sized(torch.nn.Module):
    def __init__(self, gb: float):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(4))
        self._gb = gb

    def get_memory_footprint(self):
        return int(self._gb * _GB)


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def cm_log():
    log = logging.getLogger(residency._COMPONENTS_MANAGER_LOGGER)
    handler = _Records()
    level = log.level
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        log.removeHandler(handler)
        log.setLevel(level)


def _real_manager(monkeypatch):
    cm = pytest.importorskip("diffusers.modular_pipelines.components_manager")
    pytest.importorskip("accelerate")
    manager = cm.ComponentsManager()
    manager.add("vae", _Sized(4.0))
    manager.add("audio_vae", _Sized(0.61))
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda *a, **k: (30 * _GB, 80 * _GB), raising = False
    )
    manager.enable_auto_cpu_offload(device = "cuda", memory_reserve_margin = "40GB")
    return manager


def _hook(manager, name):
    return next(
        h
        for h in manager.model_hooks
        if h.model_id.split("_")[0] == name.split("_")[0]
        and (("audio" in h.model_id) == ("audio" in name))
    )


def test_strategy_keeps_the_diffusers_decision_and_mutes_only_its_warning(monkeypatch, cm_log):
    manager = _real_manager(monkeypatch)
    pipe = types.SimpleNamespace(_components_manager = manager)
    vae, audio = _hook(manager, "vae"), _hook(manager, "audio_vae")
    stock = audio.hook.offload_strategy
    # 30 GB free under a 40 GB margin: no subset frees enough, so diffusers evicts every resident model.
    expected = stock(
        hooks = [vae],
        model_id = "audio_vae",
        model = audio.model,
        execution_device = torch.device("cuda:0"),
    )
    assert expected == [vae]
    assert any("no combination of models" in m for m in cm_log.messages)
    cm_log.messages.clear()

    assert residency.install_h3_rotation_park(pipe, park = True)
    quiet = audio.hook.offload_strategy
    assert isinstance(quiet, residency.H3QuietOffloadStrategy) and quiet.inner is stock
    assert manager._offload_strategy is quiet and vae.hook.offload_strategy is quiet
    assert quiet.memory_reserve_margin == stock.memory_reserve_margin
    got = quiet(
        hooks = [vae],
        model_id = "audio_vae",
        model = audio.model,
        execution_device = torch.device("cuda:0"),
    )
    assert got == expected
    assert not any("no combination of models" in m for m in cm_log.messages)
    assert any("search for models to offload" in m for m in cm_log.messages)
    assert not any(
        isinstance(f, residency._DropFallbackWarning)
        for f in logging.getLogger(residency._COMPONENTS_MANAGER_LOGGER).filters
    )
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda *a, **k: (60 * _GB, 80 * _GB), raising = False
    )
    assert (
        quiet(
            hooks = [vae],
            model_id = "audio_vae",
            model = audio.model,
            execution_device = torch.device("cuda:0"),
        )
        == []
    )
    assert residency.install_h3_rotation_park(pipe, park = True)
    assert audio.hook.offload_strategy is quiet


def test_kill_switch_keeps_stock(monkeypatch, cm_log):
    monkeypatch.setenv(residency.H3_ROTATION_PARK_ENV, "0")
    manager = _real_manager(monkeypatch)
    pipe = types.SimpleNamespace(_components_manager = manager)
    stock = _hook(manager, "vae").hook.offload_strategy
    assert not residency.install_h3_rotation_park(pipe, park = True)
    assert _hook(manager, "vae").hook.offload_strategy is stock
    assert not getattr(pipe, residency._PARK_ATTR, False)
    assert not residency.park_h3_rotation(pipe)


class _FakeHook:
    def __init__(self, name, device):
        self.model_id = name
        self.model = torch.nn.Linear(2, 2, device = device)
        self.offloads = 0
        self.hook = types.SimpleNamespace(offload_strategy = None)

    def offload(self):
        self.offloads += 1
        self.model.to_empty(device = "cpu")


def _fake_pipe(*, parked_device = "meta"):
    vae = _FakeHook("vae", "cpu")
    audio = _FakeHook("audio_vae", parked_device)  # still on the device after the last decode
    manager = types.SimpleNamespace(
        model_hooks = [vae, audio], _auto_offload_device = torch.device("meta"), _offload_strategy = None
    )
    return types.SimpleNamespace(_components_manager = manager), vae, audio


def test_render_return_parks_what_the_next_forward_would_evict():
    pipe, vae, audio = _fake_pipe()
    logger = types.SimpleNamespace(lines = [], info = lambda msg, *a: logger.lines.append(msg % a))
    assert residency.install_h3_rotation_park(pipe, park = True, logger = logger)
    assert len(logger.lines) == 1 and "video.h3_rotation" in logger.lines[0]
    assert residency.park_h3_rotation(pipe)
    assert audio.offloads == 1 and audio.model.weight.device.type == "cpu"
    assert vae.offloads == 0


def test_no_park_without_a_streamed_denoiser():
    pipe, _vae, audio = _fake_pipe()
    assert residency.install_h3_rotation_park(pipe, park = False)
    assert not residency.park_h3_rotation(pipe)
    assert audio.offloads == 0 and audio.model.weight.device.type == "meta"


def test_video_wires_install_and_render_end_park():
    src = (
        pathlib.Path(__file__).resolve().parents[1] / "core" / "inference" / "video.py"
    ).read_text(encoding = "utf-8")
    swap = src.index("install_pinned_swap(\n")
    install = src.index("install_h3_rotation_park(pipe, park = bool(denoiser_streamed)")
    assert swap < install < swap + 800
    settle = src.index("settle_h3_vae_fallback(state, pipe)")
    park = src.index("park_h3_rotation(pipe)")
    assert settle < park < settle + 400
