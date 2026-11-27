# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The T4 native W8A8 tile race (``native_prepare``) runs on a layer's first forward: a fresh process reads an
earlier race back from the disk cache instead of timing it again. CPU only, the race and the card are stubbed."""

from __future__ import annotations

import pytest

from core.inference import diffusion_int8_cutlass as cut
from core.inference import diffusion_int8_gemm as g8

torch = pytest.importorskip("torch")


class _Wq:
    """A CUDA-looking int8 weight: ``native_prepare`` reads the shape, card and draws its scales on the device."""

    is_cuda = True
    device = torch.device("cpu")

    def __init__(self, n, k):
        self.shape = (n, k)


@pytest.fixture
def env(monkeypatch):
    for env_ in (cut.CUTLASS_ENV, cut.CUTLASS_LIB_ENV, cut.SUSTAINED_ENV):
        monkeypatch.delenv(env_, raising = False)
    cut._NATIVE_PLAN.clear()
    cut._NATIVE_RACES.clear()
    table: dict = {}
    raced: list = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (7, 5))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(cut, "probed_tiles", lambda index: (1, 2, 3))
    monkeypatch.setattr(g8, "_cut_cache_key", lambda index: "T4|ck")
    monkeypatch.setattr(g8, "_cut_cache_read", lambda: {k_: dict(v) for k_, v in table.items()})
    monkeypatch.setattr(g8, "_cut_cache_write", lambda t: (table.clear(), table.update(t)))

    def fake_race(
        index,
        w,
        ws,
        bias,
        xs_dt,
        today,
        m,
        epilogue = cut.EPI_TORCHAO,
    ):
        raced.append((w.shape, m))
        return {"tile": {512: None, 4096: 2}[m], "us": {}}

    monkeypatch.setattr(cut, "race", fake_race)
    yield table, raced
    cut._NATIVE_PLAN.clear()
    cut._NATIVE_RACES.clear()


def _fresh_process():
    cut._NATIVE_PLAN.clear()
    cut._NATIVE_RACES.clear()


def test_a_fresh_process_reads_the_native_race_back(env):
    table, raced = env
    cut.native_prepare(_Wq(3072, 3072), torch.float16, False)
    cut.native_prepare(_Wq(9216, 3072), torch.float16, False)
    assert len(raced) == 4
    plan = dict(cut._NATIVE_PLAN)
    assert plan[(None, 3072, 3072)] == {512: None, 4096: 2}
    _fresh_process()
    del raced[:]
    cut.native_prepare(_Wq(3072, 3072), torch.float16, False)
    cut.native_prepare(_Wq(9216, 3072), torch.float16, False)
    assert not raced
    assert cut._NATIVE_PLAN == plan
    # a new shape still races, and only it
    cut.native_prepare(_Wq(3072, 12288), torch.float16, False)
    assert raced == [((3072, 12288), 512), ((3072, 12288), 4096)]


def test_a_cached_tile_the_library_no_longer_probes_races_again(env):
    table, raced = env
    cut.native_prepare(_Wq(3072, 3072), torch.float16, False)
    _fresh_process()
    del raced[:]
    for slot in table.values():
        for entry in slot.values():
            entry["4096"] = 7
    cut.native_prepare(_Wq(3072, 3072), torch.float16, False)
    assert len(raced) == 2 and cut._NATIVE_PLAN[(None, 3072, 3072)] == {512: None, 4096: 2}


def test_the_native_cache_follows_the_timing_mode_and_switch(env, monkeypatch):
    table, raced = env
    cut.native_prepare(_Wq(3072, 3072), torch.float16, False)
    _fresh_process()
    del raced[:]
    monkeypatch.setenv(cut.CUTLASS_ENV, "force")
    cut.native_prepare(_Wq(3072, 3072), torch.float16, False)
    assert len(raced) == 2  # a forced pick is its own plan, never the raced one
    assert len(table) == 2 and all("|native|" in k_ for k_ in table)


def test_a_failed_race_is_not_cached(env, monkeypatch):
    table, raced = env

    def boom(*a_, **k_):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(cut, "race", boom)
    cut.native_prepare(_Wq(3072, 3072), torch.float16, False)
    assert not table
