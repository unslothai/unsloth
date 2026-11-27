# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unified memory: two agreeing reads end the settle; a rise, discrete VRAM or the kill switch keep every read."""

from __future__ import annotations

import time

import pytest

from core.inference import diffusion_memory as dm

TOTAL = 65_536


def _run(
    monkeypatch,
    kind,
    frees,
    attempts = 3,
):
    reads, slept = [], []

    def snapshot(target):
        free = frees[min(len(reads), len(frees) - 1)]
        reads.append(free)
        return dm.DeviceMemory("cuda", "cuda:0", kind, free, TOTAL)

    monkeypatch.setattr(dm, "snapshot_device_memory", snapshot)
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    target = type("T", (), {"device": "cuda", "backend": "cuda"})()
    best = dm.settled_snapshot_device_memory(target, attempts = attempts, delay_s = 1.0)
    return best, reads, slept


@pytest.fixture(autouse = True)
def _production_spacing(monkeypatch):
    monkeypatch.delenv("UNSLOTH_SETTLE_DELAY_S", raising = False)
    monkeypatch.delenv(dm.SETTLE_STABLE_ENV, raising = False)


def test_unified_stable_reads_stop_after_one_wait(monkeypatch):
    best, reads, slept = _run(monkeypatch, "unified_memory", [50_000, 50_100])
    assert len(reads) == 2 and slept == [1.0]
    assert best.free_mib == 50_100  # still the max of the reads taken


def test_unified_rising_reads_keep_waiting(monkeypatch):
    best, reads, slept = _run(monkeypatch, "unified_memory", [40_000, 46_000, 50_000])
    assert len(reads) == 3 and slept == [1.0, 1.0]
    assert best.free_mib == 50_000


def test_unified_falling_reads_keep_waiting(monkeypatch):
    """A tenant that arrived between reads may leave again: not stable."""
    best, reads, _ = _run(monkeypatch, "unified_memory", [50_000, 44_000, 50_000])
    assert len(reads) == 3 and best.free_mib == 50_000


def test_discrete_vram_keeps_every_read(monkeypatch):
    """A busy discrete card is where a transient tenant on the card lives: spacing unchanged."""
    best, reads, slept = _run(monkeypatch, "discrete_vram", [20_000, 20_000, 20_000])
    assert len(reads) == 3 and slept == [1.0, 1.0]


def test_kill_switch_restores_every_read(monkeypatch):
    monkeypatch.setenv(dm.SETTLE_STABLE_ENV, "0")
    _, reads, _ = _run(monkeypatch, "unified_memory", [50_000, 50_000, 50_000])
    assert len(reads) == 3


def test_stable_tolerance_is_small_against_the_pool(monkeypatch):
    """A rise above the tolerance is a release in flight, not jitter."""
    tol = dm._settle_stable_tolerance_mib(TOTAL)
    assert 256 <= tol <= TOTAL // 100
    _, reads, _ = _run(monkeypatch, "unified_memory", [50_000, 50_000 + tol + 1, 50_000 + tol + 1])
    assert len(reads) == 3
