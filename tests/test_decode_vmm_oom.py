# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Waking a decode cache after a memory-heavy training step: cuMemCreate cannot reuse torch's cached blocks."""

import importlib.util
import os

import pytest
import torch

_PATH = os.path.join(os.path.dirname(__file__), "..", "unsloth", "models", "_decode_vmm.py")


@pytest.fixture
def V():
    spec = importlib.util.spec_from_file_location("_u_decode_vmm_oom", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Lib:
    def __init__(self, creates):
        self.creates = list(creates)

    def cuDevicePrimaryCtxRetain(self, *a):
        return 0

    def cuCtxPushCurrent_v2(self, *a):
        return 0

    def cuCtxPopCurrent_v2(self, *a):
        return 0

    def cuMemCreate(self, *a):
        return self.creates.pop(0)

    def cuMemMap(self, *a):
        return 0

    def cuMemSetAccess(self, *a):
        return 0

    def cuMemUnmap(self, *a):
        return 0

    def cuMemRelease(self, *a):
        return 0


def _buffer(V, monkeypatch, creates):
    lib = _Lib(creates)
    emptied = []
    monkeypatch.setattr(V, "_lib", lambda: lib)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(1))
    buffer = V.VMMBuffer.__new__(V.VMMBuffer)
    buffer.index, buffer.size, buffer.ptr, buffer.handle = 0, 1 << 21, 1 << 40, None
    return buffer, emptied


def test_wake_releases_cached_blocks_and_retries(V, monkeypatch):
    buffer, emptied = _buffer(V, monkeypatch, [2, 0])
    buffer._map()
    assert buffer.awake and emptied == [1]


def test_wake_still_out_of_memory_raises_an_oom_error(V, monkeypatch):
    buffer, emptied = _buffer(V, monkeypatch, [2, 2])
    with pytest.raises(torch.cuda.OutOfMemoryError):
        buffer._map()
    assert not buffer.awake and emptied == [1]
