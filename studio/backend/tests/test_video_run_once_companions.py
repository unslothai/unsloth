# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Streaming offload streams LTX-2's run-once text connectors with the encoders, off the denoise floor."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_memory as dm

_ENV = "UNSLOTH_VIDEO_STREAM_RUN_ONCE_COMPANIONS"


class LTX2TextConnectors(torch.nn.Module):
    def __init__(self, mib):
        super().__init__()
        self.layers = torch.nn.ModuleList([_linear(2) for _ in range(mib // 2)])
        self.moved_to = []

    def to(self, *args, **kwargs):
        self.moved_to.append(args)
        return self


class _Recorded(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.inner = module
        self.moved_to = []

    def to(self, *args, **kwargs):
        self.moved_to.append(args)
        return self


def _linear(mib):
    return torch.nn.Linear(1024, mib * 512, bias = False).to(torch.bfloat16)


def _pipe(connector_mib = 6):
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([_linear(2) for _ in range(4)])
    pipe = types.SimpleNamespace()
    pipe.transformer = dit
    pipe.connectors = LTX2TextConnectors(connector_mib)
    pipe.vae = _Recorded(_linear(3))
    pipe.components = {
        "transformer": dit,
        "connectors": pipe.connectors,
        "vae": pipe.vae,
        "scheduler": object(),
    }
    return pipe


def _plan(policy, te = False):
    return types.SimpleNamespace(
        offload_policy = policy,
        stream_text_encoders = te,
        stream_transformer = True,
        device_memory = types.SimpleNamespace(backend = "cuda"),
    )


def test_streaming_floor_leaves_the_connectors_off_the_card(monkeypatch):
    from core.inference.video import _video_offload_vram_floor_mib

    monkeypatch.delenv(_ENV, raising = False)
    pipe = _pipe()
    # vae 3 + DiT window (2x2 MiB); 6 MiB connectors excluded
    assert _video_offload_vram_floor_mib(pipe, _plan("streaming")) == 3 + 4
    assert _video_offload_vram_floor_mib(pipe, _plan("streaming"), phase = "decode") == 3


def test_kill_switch_and_other_tiers_keep_the_connectors_resident(monkeypatch):
    from core.inference.video import _video_offload_vram_floor_mib

    pipe = _pipe()
    monkeypatch.delenv(_ENV, raising = False)
    assert _video_offload_vram_floor_mib(pipe, _plan("group")) == 3 + 6 + 4
    assert _video_offload_vram_floor_mib(pipe, _plan("group", te = True)) == 3 + 6 + 4
    monkeypatch.setenv(_ENV, "0")
    assert _video_offload_vram_floor_mib(pipe, _plan("streaming")) == 3 + 6 + 4


def _stub_streaming(monkeypatch, budget_mib):
    pytest.importorskip("diffusers.hooks")
    calls = []

    def apply_group_offloading(
        module,
        onload_device = None,
        offload_device = None,
        offload_type = None,
        num_blocks_per_group = None,
        use_stream = False,
        non_blocking = False,
        record_stream = False,
        low_cpu_mem_usage = False,
    ):
        calls.append((module, offload_type, low_cpu_mem_usage))

    monkeypatch.setattr(
        sys.modules["diffusers.hooks"], "apply_group_offloading", apply_group_offloading
    )
    for name in (
        dm.GROUP_OFFLOAD_PIN_ENV,
        dm.BACKGROUND_PIN_ENV,
        "UNSLOTH_DIFFUSION_STREAMING_PREFETCH",
    ):
        monkeypatch.delenv(name, raising = False)
    for name, value in (
        ("host_pinning_allowed", True),
        ("_streaming_prefetch_enabled", True),
        ("_pinned_memory_capped", False),
        ("_pin_top_level_group", True),
    ):
        monkeypatch.setattr(dm, name, lambda *a, _v = value, **k: _v)
    for name in (
        "install_group_offload_buffer_restore",
        "install_group_offload_hooks_eager",
        "install_group_pin_wait",
        "install_group_prefetch",
        "_skip_top_level_copy_back",
        "_pin_vision_embedding_device",
    ):
        monkeypatch.setattr(dm, name, lambda *a, **k: None)
    monkeypatch.setattr(
        dm,
        "_apply_group_offloading_packed",
        lambda apply, module, kwargs, logger: apply(module, **kwargs),
    )
    monkeypatch.setattr(dm, "_pin_budget_mib", lambda: budget_mib)
    return calls


def test_streaming_offload_streams_the_connectors_when_they_pin(monkeypatch):
    monkeypatch.delenv(_ENV, raising = False)
    calls = _stub_streaming(monkeypatch, 1 << 20)
    pipe = _pipe()
    dm._apply_streaming_offload(pipe, "cuda", None)
    hooked = {id(m): (t, low) for m, t, low in calls}
    assert hooked[id(pipe.connectors)] == ("leaf_level", False)
    assert pipe.connectors.moved_to == []
    assert pipe.vae.moved_to


@pytest.mark.parametrize("why", ["budget", "kill_switch"])
def test_connectors_stay_resident_unless_pinned_or_when_switched_off(monkeypatch, why):
    if why == "kill_switch":
        monkeypatch.setenv(_ENV, "0")
        calls = _stub_streaming(monkeypatch, 1 << 20)
    else:
        monkeypatch.delenv(_ENV, raising = False)
        calls = _stub_streaming(
            monkeypatch, 9
        )  # the 8 MiB DiT pins, the 6 MiB connectors would not
    pipe = _pipe()
    dm._apply_streaming_offload(pipe, "cuda", None)
    assert all(m is not pipe.connectors for m, _t, _low in calls)
    assert pipe.connectors.moved_to
