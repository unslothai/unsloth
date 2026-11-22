# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A large int8 GEMM call's unrecorded row bucket is raced at its first call (CPU, fakes only).

Before, only the next load raced a recorded bucket, so a first launch ran a 1024^2 FLUX.3 joint call (4176 rows) on
the 4096-row winner, which may be a 256-row tile that leaves a 17th row tile 31% full."""

from __future__ import annotations

import pytest

from core.inference import diffusion_int8_cutlass as cut
from core.inference import diffusion_int8_gemm as g8

torch = pytest.importorskip("torch")

_STATE = (
    g8._CUT_PLAN,
    g8._CUT_FAMILY,
    g8._CUT_RACES,
    g8._CUT_DONE,
    g8._CUT_ROWS,
    g8._CUT_ROWS_PENDING,
    g8._CUT_CKEY,
    g8._CUT_LIVE,
    g8._CUT_TRITON_ON,
)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for env in (cut.ROW_RACE_ENV, cut.HUGE_RACE_ENV, cut.LIVE_RACE_ENV, cut.CUTLASS_ENV):
        monkeypatch.delenv(env, raising = False)
    for d in _STATE:
        d.clear()
    yield
    for d in _STATE:
        d.clear()


def _operands(
    m,
    n = 64,
    k = 64,
):
    g = torch.Generator().manual_seed(0)
    a = torch.randint(-127, 128, (m, k), generator = g, dtype = torch.int8)
    w = torch.randint(-127, 128, (n, k), generator = g, dtype = torch.int8)
    xs = torch.rand(m, generator = g).to(torch.bfloat16)
    ws = torch.rand(n, generator = g).to(torch.bfloat16)
    return a, w, xs, ws


def _env(
    monkeypatch,
    winners = None,
    fits = True,
):
    table: dict = {}
    monkeypatch.setattr(
        g8,
        "_cut_cache_read",
        lambda: {k_: dict(v) if isinstance(v, dict) else v for k_, v in table.items()},
    )
    monkeypatch.setattr(g8, "_cut_cache_write", lambda t: table.clear() or table.update(t))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(g8, "_huge_race_fits", lambda *a_: fits)
    monkeypatch.setattr(cut, "eligible", lambda *a_: True)
    raced, calls = [], []

    def fake_race(index, w, ws, bias, xs_dt, today, m, *a_, **k_):
        raced.append((m, today is None))
        return {"tile": (winners or {}).get(m, 8), "us": {"8": 1.0, "today": 2.0}, "m": m}

    monkeypatch.setattr(cut, "race", fake_race)
    monkeypatch.setattr(
        cut,
        "launch",
        lambda a_, w_, xs_, ws_, b_, fam, tile, out: calls.append((tile, a_.shape[0]))
        or torch.zeros(a_.shape[0], w_.shape[0], dtype = out),
    )
    g8._CUT_FAMILY[None] = 80
    g8._CUT_CKEY[None] = "ck"
    return table, raced, calls


def test_first_call_at_a_new_bucket_races_it_and_runs_its_winner(monkeypatch):
    table, raced, calls = _env(monkeypatch)
    a, w, xs, ws = _operands(4176)
    key = g8._cut_key(64, 64, xs.dtype, ws.dtype, False)
    g8._CUT_PLAN[None] = {key: {512: None, 4096: 3, 16384: None}}
    g8._run(a, w, xs, ws, None)
    g8._run(a, w, xs, ws, None)
    g8._run(a[:4096], w, xs[:4096], ws, None)
    assert raced == [
        (4224, True)
    ]  # once, at the call's bucket; today = cuBLAS when the Triton path is off
    assert calls == [(8, 4176), (8, 4176), (3, 4096)]
    assert g8._CUT_PLAN[None][key][4224] == 8
    # the next load reads the winner and the recorded bucket back: it races nothing
    assert table["ck"][repr(key)]["4224"] == 8 and 4224 in table["ck|rows"][repr(key)]
    assert g8._CUT_RACES[None][repr(key)][4224]["live"] is True


def test_a_later_load_races_nothing_after_a_live_race(monkeypatch):
    table, raced, _calls = _env(monkeypatch)
    a, w, xs, ws = _operands(4176)
    key = g8._cut_key(64, 64, xs.dtype, ws.dtype, False)
    g8._CUT_PLAN[None] = {key: {512: None, 4096: 3, 16384: None}}
    table["ck"] = {repr(key): {"512": None, "4096": 3, "16384": None}}
    g8._run(a, w, xs, ws, None)
    stored = table["ck"][repr(key)]
    points = cut.race_points(table["ck|rows"][repr(key)])
    assert points == (512, 4224, 16384) and all(str(p) in stored for p in points)


def test_triton_on_races_todays_triton_path(monkeypatch):
    _table, raced, _calls = _env(monkeypatch)
    a, w, xs, ws = _operands(4176)
    key = g8._cut_key(64, 64, xs.dtype, ws.dtype, False)
    g8._CUT_PLAN[None] = {key: {512: None, 4096: 3}}
    g8._CUT_TRITON_ON[None] = True
    g8._run(a, w, xs, ws, None)
    assert raced == [(4224, False)]


def test_kill_switches_keep_the_4096_winner(monkeypatch):
    for env in (cut.LIVE_RACE_ENV, cut.ROW_RACE_ENV):
        g8._CUT_LIVE.clear()
        _table, raced, calls = _env(monkeypatch)
        monkeypatch.setenv(env, "0")
        a, w, xs, ws = _operands(4176)
        key = g8._cut_key(64, 64, xs.dtype, ws.dtype, False)
        g8._CUT_PLAN[None] = {key: {512: None, 4096: 3}}
        g8._run(a, w, xs, ws, None)
        assert raced == [] and calls == [(3, 4176)]
        monkeypatch.delenv(env)


def test_no_race_under_capture_compile_or_a_capture_elsewhere(monkeypatch):
    from core.inference import diffusion_cuda_graph as cg

    _table, raced, calls = _env(monkeypatch)
    a, w, xs, ws = _operands(4176)
    key = g8._cut_key(64, 64, xs.dtype, ws.dtype, False)
    g8._CUT_PLAN[None] = {key: {512: None, 4096: 3}}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    g8._run(a, w, xs, ws, None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    g8._run(a, w, xs, ws, None)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    import contextlib

    @contextlib.contextmanager
    def busy():
        yield False

    monkeypatch.setattr(cg, "hold_off_capture", busy)
    g8._run(a, w, xs, ws, None)
    assert raced == [] and calls == [(3, 4176)] * 3
    # none of those counted as tried: the first safe call races
    monkeypatch.undo()
    _table, raced, calls = _env(monkeypatch)
    g8._CUT_PLAN[None] = {key: {512: None, 4096: 3}}
    g8._run(a, w, xs, ws, None)
    assert raced == [(4224, True)] and calls == [(8, 4176)]


def test_no_room_keeps_the_4096_winner_and_does_not_retry(monkeypatch):
    _table, raced, calls = _env(monkeypatch, fits = False)
    a, w, xs, ws = _operands(4176)
    key = g8._cut_key(64, 64, xs.dtype, ws.dtype, False)
    g8._CUT_PLAN[None] = {key: {512: None, 4096: 3}}
    g8._run(a, w, xs, ws, None)
    g8._run(a, w, xs, ws, None)
    assert raced == [] and calls == [(3, 4176), (3, 4176)]


def test_a_failed_race_keeps_the_slot(monkeypatch):
    _table, _raced, calls = _env(monkeypatch)

    def boom(*a_, **k_):
        raise RuntimeError("launch failed")

    monkeypatch.setattr(cut, "race", boom)
    a, w, xs, ws = _operands(4176)
    key = g8._cut_key(64, 64, xs.dtype, ws.dtype, False)
    g8._CUT_PLAN[None] = {key: {512: None, 4096: 3}}
    g8._run(a, w, xs, ws, None)
    assert calls == [(3, 4176)] and 4224 not in g8._CUT_PLAN[None][key]


def test_live_races_per_shape_are_capped(monkeypatch):
    _table, raced, _calls = _env(monkeypatch)
    key = g8._cut_key(64, 64, torch.bfloat16, torch.bfloat16, False)
    g8._CUT_PLAN[None] = {key: {512: None, 4096: 3}}
    for m in (4176, 2100, 3000, 5000):
        a, w, xs, ws = _operands(m)
        g8._run(a, w, xs, ws, None)
    assert [r[0] for r in raced] == [4224, 2112][: cut._MAX_ROW_RACES]


def test_small_huge_and_unplanned_calls_never_race(monkeypatch):
    _table, raced, _calls = _env(monkeypatch)
    key = g8._cut_key(64, 64, torch.bfloat16, torch.bfloat16, False)
    g8._CUT_PLAN[None] = {key: {512: None, 4096: 3, 16384: None}}
    for m in (300, 9000):
        a, w, xs, ws = _operands(m)
        g8._run(a, w, xs, ws, None)
    a, w, xs, ws = _operands(4176, n = 128)  # a shape the plan has no entry for
    g8._run(a, w, xs, ws, None)
    assert raced == []


def test_today_wins_the_live_race(monkeypatch):
    _table, raced, calls = _env(monkeypatch, winners = {4224: None})
    monkeypatch.setattr(
        g8, "reference", lambda a_, w_, xs_, ws_, b_: torch.zeros(a_.shape[0], w_.shape[0])
    )
    a, w, xs, ws = _operands(4176)
    key = g8._cut_key(64, 64, xs.dtype, ws.dtype, False)
    g8._CUT_PLAN[None] = {key: {512: None, 4096: 3}}
    g8._run(a, w, xs, ws, None)
    assert raced == [(4224, True)] and calls == [] and g8._CUT_PLAN[None][key][4224] is None


def test_an_ineligible_call_or_an_untimed_race_records_nothing(monkeypatch):
    """A call the prebuilt kernel cannot take (here: eligible() refuses it) never races its shape's bucket, and a
    race that timed nothing leaves the bucket unrecorded: a later eligible call of the shape still races it."""
    table, raced, calls = _env(monkeypatch)
    a, w, xs, ws = _operands(4176)
    key = g8._cut_key(64, 64, xs.dtype, ws.dtype, False)
    g8._CUT_PLAN[None] = {key: {512: None, 4096: 3}}
    monkeypatch.setattr(cut, "eligible", lambda *a_: False)
    monkeypatch.setattr(
        g8, "reference", lambda a_, w_, xs_, ws_, b_: torch.zeros(a_.shape[0], w_.shape[0])
    )
    g8._run(a, w, xs, ws, None)
    assert raced == [] and 4224 not in g8._CUT_PLAN[None][key] and not g8._CUT_LIVE.get(None)
    monkeypatch.setattr(cut, "eligible", lambda *a_: True)
    monkeypatch.setattr(cut, "race", lambda *a_, **k_: {"tile": None, "us": {}})
    g8._run(a, w, xs, ws, None)
    assert 4224 not in g8._CUT_PLAN[None][key] and "ck" not in table
