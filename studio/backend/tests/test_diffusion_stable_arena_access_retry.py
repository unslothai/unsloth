# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""VmmArena.map() against a fake CUDA driver: cuMemSetAccess returning CUDA_ERROR_NOT_READY is retried with a bounded
backoff on Windows only, and any persistent or other error still unmaps and releases the new allocation."""

from __future__ import annotations

import contextlib
import sys
import time

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_stable_arena as sa  # noqa: E402

_DEV = 97  # an ordinal no real test maps, so the mapped-bytes counter it touches is private to this file


class _FakeDriver:
    def __init__(self, access_results):
        self.access_results = list(access_results)
        self.calls = []

    def _log(
        self,
        name,
        rc = 0,
    ):
        self.calls.append(name)
        return rc

    def cuMemCreate(self, handle, size, prop, flags):
        handle._obj.value = 1234
        return self._log("create")

    def cuMemMap(self, va, size, offset, handle, flags):
        return self._log("map")

    def cuMemSetAccess(self, va, size, desc, count):
        rc = self.access_results.pop(0) if len(self.access_results) > 1 else self.access_results[0]
        return self._log("access", rc)

    def cuMemUnmap(self, va, size):
        return self._log("unmap")

    def cuMemRelease(self, handle):
        return self._log("release")


def _arena(monkeypatch, access_results, platform):
    drv = _FakeDriver(access_results)
    sleeps = []
    monkeypatch.setattr(sa, "_driver", lambda: drv)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.setattr(torch.cuda, "device", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(torch, "as_tensor", lambda *a, **k: "base")
    arena = object.__new__(sa.VmmArena)
    arena.device_index = _DEV
    arena.device = "cuda:97"
    arena.size = 2 << 20
    arena.granularity = 2 << 20
    arena.va = 1 << 40
    arena._handle = None
    arena._base = None
    arena.maps = arena.unmaps = 0
    arena.closed = False
    monkeypatch.setattr(arena, "_ensure_context", lambda: None)
    return arena, drv, sleeps


def test_not_ready_then_success_maps_on_windows(monkeypatch):
    arena, drv, sleeps = _arena(monkeypatch, [600, 0], "win32")
    assert arena.map() == "base"
    assert drv.calls == ["create", "map", "access", "access"]
    assert sleeps == [0.001]
    assert arena.mapped and arena.maps == 1
    arena._handle = None
    sa._count_mapped(_DEV, -arena.size)


def test_several_not_ready_back_off_doubling_to_16_ms(monkeypatch):
    arena, drv, sleeps = _arena(monkeypatch, [600] * 7 + [0], "win32")
    assert arena.map() == "base"
    assert drv.calls.count("access") == 8 and "unmap" not in drv.calls
    assert sleeps == [0.001, 0.002, 0.004, 0.008, 0.016, 0.016, 0.016]
    arena._handle = None
    sa._count_mapped(_DEV, -arena.size)


def test_persistent_not_ready_is_bounded_and_still_unmaps(monkeypatch):
    arena, drv, sleeps = _arena(monkeypatch, [600], "win32")
    with pytest.raises(RuntimeError, match = "cuMemSetAccess failed \\(CUresult 600\\)"):
        arena.map()
    assert max(sleeps) == pytest.approx(0.016)
    assert 2.0 <= sum(sleeps) < 2.0 + 0.016
    assert drv.calls[-2:] == ["unmap", "release"] and drv.calls.count("unmap") == 1
    assert not arena.mapped and arena.maps == 0


def test_other_errors_are_not_retried_on_windows(monkeypatch):
    arena, drv, sleeps = _arena(monkeypatch, [2], "win32")
    with pytest.raises(RuntimeError, match = "CUresult 2"):
        arena.map()
    assert drv.calls == ["create", "map", "access", "unmap", "release"]
    assert sleeps == []


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_not_ready_elsewhere_unmaps_at_once_as_before(monkeypatch, platform):
    arena, drv, sleeps = _arena(monkeypatch, [600, 0], platform)
    with pytest.raises(RuntimeError, match = "CUresult 600"):
        arena.map()
    assert drv.calls == ["create", "map", "access", "unmap", "release"]
    assert sleeps == []


def test_success_first_time_makes_one_call(monkeypatch):
    arena, drv, sleeps = _arena(monkeypatch, [0], "win32")
    assert arena.map() == "base"
    assert drv.calls == ["create", "map", "access"] and sleeps == []
    arena._handle = None
    sa._count_mapped(_DEV, -arena.size)
