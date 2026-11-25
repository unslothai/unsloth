# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Large int8 GEMM calls take the race at their own row bucket once a load recorded it (CPU, fakes only).

A 1024^2 FLUX.3 joint call has 4176 rows: the 4096-row race picks a 256-row tile that leaves a 17th row tile 31% full
(L4: 700 vs 585 us per Q projection). The first load races 4096 rows as before and records the buckets its calls
ran; the next load races those, reusing every winner it already has."""

from __future__ import annotations

import pytest

from core.inference import diffusion_int8_cutlass as cut
from core.inference import diffusion_int8_gemm as g8

torch = pytest.importorskip("torch")


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(cut.ROW_RACE_ENV, raising = False)
    monkeypatch.delenv(cut.HUGE_RACE_ENV, raising = False)
    for d in (
        g8._CUT_PLAN,
        g8._CUT_FAMILY,
        g8._CUT_RACES,
        g8._CUT_DONE,
        g8._CUT_ROWS,
        g8._CUT_ROWS_PENDING,
        g8._CUT_CKEY,
    ):
        d.clear()
    yield
    for d in (
        g8._CUT_PLAN,
        g8._CUT_FAMILY,
        g8._CUT_RACES,
        g8._CUT_DONE,
        g8._CUT_ROWS,
        g8._CUT_ROWS_PENDING,
        g8._CUT_CKEY,
    ):
        d.clear()


def test_row_bucket_keeps_the_row_tile_count():
    assert (
        cut.row_bucket(4096) == 4096
        and cut.row_bucket(4176) == 4224
        and cut.row_bucket(4097) == 4160
    )
    for m in (1100, 4096, 4176, 4200, 5000, 8000):
        for bm in (64, 128, 256):
            assert -(-m // bm) == -(-cut.row_bucket(m) // bm)


def test_slot_prefers_the_calls_own_bucket():
    entry = {512: 1, 4096: 5, 4224: 8, 16384: 2}
    assert cut.slot(4176, entry) == 8  # joint call
    assert cut.slot(4096, entry) == 5  # image-only call
    assert cut.slot(2000, entry) == 5  # unraced bucket: the 4096-row winner, as before
    assert cut.slot(300, entry) == 1 and cut.slot(9000, entry) == 2
    # a plan raced only at recorded buckets: the nearest large race
    assert cut.slot(4096, {512: 1, 4224: 8}) == 8
    assert cut.slot(5000, {512: 1, 4224: 8, 6144: None}) == 8


def test_race_points_follow_the_recorded_buckets(monkeypatch):
    assert cut.race_points() == (512, 4096, 16384)
    assert cut.race_points([4096, 4224]) == (512, 4096, 4224, 16384)
    assert cut.race_points([2048, 4096, 4224]) == (512, 4096, 4224, 16384)  # the two most recent
    assert cut.race_points([300, 20000]) == (512, 4096, 16384)  # outside the large range
    monkeypatch.setenv(cut.ROW_RACE_ENV, "0")
    assert cut.race_points([4096, 4224]) == (512, 4096, 16384)


def test_large_calls_record_their_buckets_once(monkeypatch):
    table = {}
    writes = []
    monkeypatch.setattr(g8, "_cut_cache_read", lambda: dict(table))
    monkeypatch.setattr(g8, "_cut_cache_write", lambda t: writes.append(t) or table.update(t))
    g8._CUT_CKEY[0] = "ck"
    key = g8._cut_key(3072, 3072, torch.float32, torch.float32, False)
    for m in (4096, 4176, 4176, 300, 4176, 4096):
        g8._note_rows(0, key, m)
    assert len(writes) == 2  # one per new bucket, none for repeats or small calls
    assert table["ck|rows"] == {repr(key): [4096, 4224]}
    # the kill switch records nothing
    monkeypatch.setenv(cut.ROW_RACE_ENV, "0")
    g8._note_rows(0, key, 5000)
    assert len(writes) == 2


def _plan_env(monkeypatch, table):
    class _W:
        def __init__(self, n, k):
            self.qdata = torch.zeros(n, k, dtype = torch.int8)
            self.scale = torch.ones(n, 1, dtype = torch.bfloat16)

    rec = (torch.nn.Linear(32, 64, bias = False), ("v2", None, None, _W(64, 32)))
    monkeypatch.setattr(g8, "cutlass_wanted", lambda cap: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (8, 9))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a_: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(cut, "probed_tiles", lambda index: (1, 2, 8))
    monkeypatch.setattr(g8, "_cut_cache_key", lambda index: "ck")
    monkeypatch.setattr(g8, "_cut_cache_read", lambda: table)
    monkeypatch.setattr(g8, "_cut_cache_write", lambda t: None)
    monkeypatch.setattr(g8, "_xs_dtype", lambda *a_: torch.float32)
    monkeypatch.setattr(g8, "_huge_race_fits", lambda *a_: True)
    to = torch.Tensor.to
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, *a_, **k_: self
        if a_ and isinstance(a_[0], torch.device)
        else to(self, *a_, **k_),
    )
    raced = []

    def fake_race(index, w, ws, bias, xs_dt, today, m, *a_, **k_):
        raced.append(m)
        return {"tile": {512: None, 4096: 1, 4224: 8, 16384: 2}[m], "us": {}}

    monkeypatch.setattr(cut, "race", fake_race)
    key = g8._cut_key(64, 32, torch.float32, torch.bfloat16, False)
    return rec, key, raced


def test_a_later_load_races_only_the_recorded_bucket(monkeypatch):
    key = g8._cut_key(64, 32, torch.float32, torch.bfloat16, False)
    table = {
        "ck": {repr(key): {"512": None, "4096": 1, "16384": 2}},
        "ck|rows": {repr(key): [4096, 4224]},
    }
    rec, key, raced = _plan_env(monkeypatch, table)
    g8._cutlass_plan(0, [rec], True)
    assert raced == [4224]
    assert g8._CUT_PLAN[0][key] == {512: None, 4096: 1, 4224: 8, 16384: 2}


def test_first_load_races_as_before(monkeypatch):
    rec, key, raced = _plan_env(monkeypatch, {})
    g8._cutlass_plan(0, [rec], True)
    assert raced == [512, 4096, 16384]
    assert g8._CUT_CKEY[0] == "ck"


def test_op_body_runs_the_joint_buckets_winner(monkeypatch):
    g = torch.Generator().manual_seed(0)
    a = torch.randint(-127, 128, (4176, 64), generator = g, dtype = torch.int8)
    w = torch.randint(-127, 128, (64, 64), generator = g, dtype = torch.int8)
    xs = torch.rand(4176, generator = g).to(torch.bfloat16)
    ws = torch.rand(64, generator = g).to(torch.bfloat16)
    key = g8._cut_key(64, 64, xs.dtype, ws.dtype, False)
    g8._CUT_PLAN[None] = {key: {512: None, 4096: 3, 4224: 7}}
    g8._CUT_FAMILY[None] = 80
    monkeypatch.setattr(cut, "eligible", lambda *a_: True)
    calls = []
    monkeypatch.setattr(
        cut,
        "launch",
        lambda a_, w_, xs_, ws_, b_, fam, tile, out: calls.append((tile, a_.shape[0]))
        or torch.zeros(a_.shape[0], w_.shape[0], dtype = out),
    )
    g8._run(a, w, xs, ws, None)
    g8._run(a[:4096], w, xs[:4096], ws, None)
    assert calls == [(7, 4176), (3, 4096)]
