# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The pre-download auto denoiser planner settles the card once for both of its seed checks."""

import itertools
import time
import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_component_sizes as dcs
import core.inference.diffusion_memory as dm
import core.inference.video as vid
from core.inference.diffusion_memory import DeviceMemory
from core.inference.video_families import detect_video_family

MIB_PER_GIB = 1024
WAN5B = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


class _Card:
    """Fake device-memory provider under the real settle loop: counts reads and settle sleeps."""

    def __init__(self, monkeypatch, free_mib, total_mib):
        self.reads = 0
        self.sleeps = 0
        self.free_mib = free_mib
        self.total_mib = total_mib
        monkeypatch.setattr(dm, "snapshot_device_memory", self._read)
        monkeypatch.setattr(time, "sleep", self._sleep)

    def _read(self, _target):
        self.reads += 1
        return DeviceMemory("cuda", "cuda", "discrete_vram", self.free_mib, self.total_mib)

    def _sleep(self, _s):
        self.sleeps += 1


def _backend(monkeypatch, *, scheme, resident_mib):
    fam = detect_video_family(WAN5B)
    backend = vid.VideoBackend()
    resident_pipe = object() if resident_mib else None
    backend._state = types.SimpleNamespace(pipe = resident_pipe)
    monkeypatch.setattr(vid, "_video_auto_denoiser_scheme", lambda fam, **kw: scheme)
    monkeypatch.setattr(
        vid,
        "_pipeline_device_mib",
        lambda pipe, ordinal = None: resident_mib if pipe is resident_pipe and pipe else 0,
    )
    monkeypatch.setattr(
        vid,
        "resolve_diffusion_device_target",
        lambda **_k: types.SimpleNamespace(
            device = "cuda", dtype = torch.bfloat16, ordinal = None, supports_model_cpu_offload = True
        ),
    )
    monkeypatch.setattr(
        dcs, "video_components_gb", lambda fam, base, **kw: tuple(fam.bf16_components_gb)
    )
    return backend, fam


def _plan(backend, fam, requested):
    return backend._video_planned_auto_denoiser_scheme(
        fam, base = None, kind = "pipeline", transformer_quant = requested, speed_mode = None
    )


def _main_arm(monkeypatch):
    """The pre-fix planner: every seed check runs its own settle."""
    real = vid._video_seed_stays_resident

    def _fresh(fam, **kw):
        kw.pop("device_memory_reader", None)
        return real(fam, **kw)

    monkeypatch.setattr(vid, "_video_seed_stays_resident", _fresh)


def test_auto_seed_settles_once_per_planner_call(monkeypatch):
    # 16 GiB card with another tenant holding 6 GiB: free is below the settle reserve, so each settle sleeps twice.
    # An auto fp8 seed first asks whether bf16 stays resident (no), then whether fp8 does: two seed checks.
    card = _Card(monkeypatch, free_mib = 10 * MIB_PER_GIB, total_mib = 16 * MIB_PER_GIB)
    backend, fam = _backend(monkeypatch, scheme = "fp8", resident_mib = 0)
    _plan(backend, fam, "auto")
    head = (card.reads, card.sleeps)

    with monkeypatch.context() as m:
        _main_arm(m)
        card.reads = card.sleeps = 0
        _plan(backend, fam, "auto")
        main = (card.reads, card.sleeps)

    assert main == (6, 4)  # two settles, three reads and two 1 s sleeps each
    assert head == (3, 2)  # one settle shared by both checks


_FREE_GIB = (2, 6, 10, 14, 20, 30, 40, 78)
_TOTAL_GIB = (16, 24, 32, 48, 80)


@pytest.mark.parametrize(
    "scheme, requested, resident_mib",
    list(
        itertools.product(
            ("nvfp4", "fp8", "int8"), ("auto", None, "nvfp4", "fp8"), (0, 6_000, 20_000)
        )
    ),
)
def test_planner_decisions_match_the_two_settle_planner(
    monkeypatch, scheme, requested, resident_mib
):
    """Same free/total reading on every read: head (one settle) and main (two settles) choose alike."""
    card = _Card(monkeypatch, free_mib = 0, total_mib = 0)
    backend, fam = _backend(monkeypatch, scheme = scheme, resident_mib = resident_mib)
    head, main = [], []
    for total in _TOTAL_GIB:
        for free in _FREE_GIB:
            if free > total:
                continue
            card.free_mib, card.total_mib = free * MIB_PER_GIB, total * MIB_PER_GIB
            card.reads = 0
            head.append((total, free, _plan(backend, fam, requested)))
            assert card.reads <= 3
    with monkeypatch.context() as m:
        _main_arm(m)
        for total in _TOTAL_GIB:
            for free in _FREE_GIB:
                if free > total:
                    continue
                card.free_mib, card.total_mib = free * MIB_PER_GIB, total * MIB_PER_GIB
                main.append((total, free, _plan(backend, fam, requested)))
    assert head == main
    if requested in ("auto", None) and scheme == "fp8" and resident_mib == 0:
        outcomes = {d for _t, _f, d in head}
        assert outcomes == {None, vid.DENOISER_SEED_DECLINED, "fp8"}, outcomes


def test_reader_failure_still_answers_none(monkeypatch):
    """A settle that raises leaves the planner's contract unchanged: an unanswerable probe keeps the dense shards."""
    backend, fam = _backend(monkeypatch, scheme = "nvfp4", resident_mib = 0)

    def _boom(_target, **_kw):
        raise RuntimeError("no device")

    monkeypatch.setattr(vid, "settled_snapshot_device_memory", _boom)
    assert _plan(backend, fam, "auto") is None
