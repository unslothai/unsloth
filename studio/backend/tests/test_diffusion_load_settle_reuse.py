# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A load's re-plans reuse its settled CUDA reading while one fresh read still equals it, and settle as before
otherwise (another process moved, our weights landed, a deliberate re-snapshot, any device but CUDA)."""

import inspect
import types

import pytest

from core.inference import diffusion as dmod
from core.inference import diffusion_memory as mem
from core.inference.diffusion_memory import DeviceMemory

CUDA = types.SimpleNamespace(device = "cuda", backend = "cuda")


@pytest.fixture
def card(monkeypatch):
    """A CUDA card whose free MiB the test sets; counts reads and settle sleeps (the wait is real seconds in prod)."""
    state = types.SimpleNamespace(free = 10_000, total = 24_000, reads = 0, sleeps = 0, raise_on_read = False)

    def snapshot(target):
        if state.raise_on_read:
            raise RuntimeError("probe failed")
        state.reads += 1
        device = getattr(target, "device", "cpu")
        kind = "discrete_vram" if device in ("cuda", "xpu") else "unified_memory"
        return DeviceMemory(
            getattr(target, "backend", device), device, kind, state.free, state.total
        )

    def sleep(_s):
        state.sleeps += 1

    monkeypatch.setattr(mem, "snapshot_device_memory", snapshot)
    monkeypatch.setattr(mem, "_release_cuda_cache", lambda: None)
    monkeypatch.setattr("time.sleep", sleep)
    monkeypatch.setenv("UNSLOTH_SETTLE_DELAY_S", "1.0")
    return state


def _in_load(fn):
    return dmod._load_scoped_settle(fn)()


def test_unchanged_card_reuses_the_settled_reading(card):
    def load():
        first = dmod._load_settled_memory(CUDA)
        sleeps = card.sleeps
        second = dmod._load_settled_memory(CUDA)
        return first, sleeps, second

    first, sleeps_after_first, second = _in_load(load)
    assert (
        sleeps_after_first == 2
    )  # 14 GB foreign use: the first plan waits out two spaced re-reads
    assert card.sleeps == 2  # the re-plan did not wait again
    assert second == first == mem.settled_snapshot_device_memory(CUDA)


@pytest.mark.parametrize("delta", [-1, 1, -3000, 3000])
def test_any_change_settles_again(card, delta):
    def load():
        dmod._load_settled_memory(CUDA)
        card.free += delta
        return dmod._load_settled_memory(CUDA)

    got = _in_load(load)
    assert card.sleeps == 2 + (0 if card.free >= card.total - 2400 else 2)
    assert got.free_mib == card.free


def test_forget_forces_the_deliberate_re_snapshot(card):
    def load():
        dmod._load_settled_memory(CUDA)
        dmod._forget_load_settle()
        return dmod._load_settled_memory(CUDA)

    _in_load(load)
    assert card.sleeps == 4


def test_outside_a_load_every_plan_settles(card):
    dmod._load_settled_memory(CUDA)
    dmod._load_settled_memory(CUDA)
    assert card.sleeps == 4
    assert dmod._LOAD_SETTLED_MEMORY.get() is None


@pytest.mark.parametrize("device", ["mps", "xpu", "cpu"])
def test_other_devices_read_fresh_every_time(card, device, monkeypatch):
    target = types.SimpleNamespace(device = device, backend = device)
    calls = []
    real = dmod.settled_snapshot_device_memory
    monkeypatch.setattr(
        dmod, "settled_snapshot_device_memory", lambda t: calls.append(t) or real(t)
    )

    def load():
        dmod._load_settled_memory(target)
        dmod._load_settled_memory(target)

    _in_load(load)
    assert len(calls) == 2
    assert mem.settled_memory_unchanged(target, DeviceMemory(device, device, "x", 1, 2)) is False


def test_unknown_counts_and_failed_probes_never_reuse(card):
    assert mem.settled_memory_unchanged(CUDA, None) is False
    assert (
        mem.settled_memory_unchanged(
            CUDA, DeviceMemory("cuda", "cuda", "discrete_vram", None, 24_000)
        )
        is False
    )
    assert (
        mem.settled_memory_unchanged(
            CUDA, DeviceMemory("cuda", "cuda", "discrete_vram", 10_000, None)
        )
        is False
    )
    good = DeviceMemory("cuda", "cuda", "discrete_vram", card.free, card.total)
    assert mem.settled_memory_unchanged(CUDA, good) is True
    card.raise_on_read = True
    assert mem.settled_memory_unchanged(CUDA, good) is False


def test_the_settle_itself_is_unchanged_for_other_callers(card):
    mem.settled_snapshot_device_memory(CUDA)
    mem.settled_snapshot_device_memory(CUDA)
    assert card.sleeps == 4 and card.reads == 6
    card.free = card.total - 100
    card.sleeps = 0
    mem.settled_snapshot_device_memory(CUDA)
    assert card.sleeps == 0  # within the reserve: no wait, as before


def test_scope_is_per_load_and_survives_errors(card):
    def boom():
        dmod._load_settled_memory(CUDA)
        assert dmod._LOAD_SETTLED_MEMORY.get()
        raise ValueError("load failed")

    with pytest.raises(ValueError):
        _in_load(boom)
    assert dmod._LOAD_SETTLED_MEMORY.get() is None
    # A new load starts with no reading: its first plan settles in full.
    _in_load(lambda: dmod._load_settled_memory(CUDA))
    assert card.sleeps == 4


def test_load_pipeline_is_scoped_and_the_capacity_retry_re_snapshots():
    source = inspect.getsource(dmod.DiffusionBackend.load_pipeline)
    assert "@_load_scoped_settle" in source
    at = source.index("re-snapshot and replan once")
    assert source.index("_forget_load_settle()", at) < source.index("_replan_candidate()", at)
    plan_source = inspect.getsource(dmod.DiffusionBackend._plan_memory)
    assert "else _load_settled_memory(target)" in plan_source
