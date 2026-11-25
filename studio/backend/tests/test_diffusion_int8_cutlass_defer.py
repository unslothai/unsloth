# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CUTLASS int8 race of shapes with no stored verdict runs after the first image (CPU, fakes only).

Before, a fresh install raced every int8 weight shape inside the load (22-26 s on an L4) and again inside the first
render (7-8 s), all before the first image. Those shapes now run the probed Triton kernel (same output, bit for bit)
until the race has run between renders; its verdicts persist as before."""

from __future__ import annotations

import contextlib
import threading
import time
import types

import pytest

from core.inference import diffusion_int8_cutlass as cut
from core.inference import diffusion_int8_cutlass_defer as defer
from core.inference import diffusion_int8_gemm as g8

torch = pytest.importorskip("torch")

_DICTS = (
    g8._CUT_PLAN,
    g8._CUT_FAMILY,
    g8._CUT_RACES,
    g8._CUT_DONE,
    g8._CUT_ROWS,
    g8._CUT_ROWS_PENDING,
    g8._CUT_CKEY,
    g8._CUT_LIVE,
    g8._CUT_TRITON_ON,
    g8._CUT_DEFER_KEYS,
    cut._PROBED,
)
_WINNERS = {512: None, 4096: 1, 4224: 8, 16384: 2}


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for env in (
        cut.ROW_RACE_ENV,
        cut.HUGE_RACE_ENV,
        cut.LIVE_RACE_ENV,
        cut.CUTLASS_ENV,
        defer.DEFER_ENV,
    ):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setenv(defer.IDLE_ENV, "0")
    for d in _DICTS:
        d.clear()
    defer._reset_for_tests()
    yield
    defer._reset_for_tests()
    for d in _DICTS:
        d.clear()


class _W:
    def __init__(self, n, k):
        self.qdata = torch.zeros(n, k, dtype = torch.int8)
        self.scale = torch.ones(n, 1, dtype = torch.bfloat16)


def _rec(n = 64, k = 32):
    return (torch.nn.Linear(k, n, bias = False), ("v2", None, None, _W(n, k), True))


def _env(
    monkeypatch,
    table,
    stored = None,
    gate = None,
):
    monkeypatch.setattr(g8, "cutlass_wanted", lambda cap: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (8, 9))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a_: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "device", lambda *a_: contextlib.nullcontext())
    monkeypatch.setattr(cut, "probed_tiles", lambda index: (1, 2, 8))
    monkeypatch.setattr(cut, "eligible", lambda *a_: True)
    monkeypatch.setattr(cut, "_stored_tiles", lambda index, fam: stored)
    monkeypatch.setattr(g8, "_cut_cache_key", lambda index: "ck")
    monkeypatch.setattr(
        g8,
        "_cut_cache_read",
        lambda: {k_: dict(v) if isinstance(v, dict) else v for k_, v in table.items()},
    )
    monkeypatch.setattr(g8, "_cut_cache_write", lambda t: table.clear() or table.update(t))
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
        if gate is not None:
            gate(int(w.shape[0]), m)
        raced.append((int(w.shape[0]), m, today is None))
        return {"tile": _WINNERS[m], "us": {"8": 1.0, "today": 2.0}, "m": m}

    monkeypatch.setattr(cut, "race", fake_race)
    return raced


def _key(n = 64, k = 32):
    return g8._cut_key(n, k, torch.float32, torch.bfloat16, False)


def _call(
    m,
    n = 64,
    k = 32,
):
    """The operands one int8 GEMM call of ``m`` rows hands the op body (key = ``_key(n, k)``)."""
    a = torch.zeros(m, k, dtype = torch.int8)
    w = torch.zeros(n, k, dtype = torch.int8)
    xs = torch.ones(m, dtype = torch.float32)
    ws = torch.ones(n, dtype = torch.bfloat16)
    return a, w, xs, ws, None


def _wait_state(*states, timeout = 10.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        st = defer.status()
        if st["state"] in states and not st["running"]:
            return st
        time.sleep(0.01)
    raise AssertionError(f"deferred race stuck: {defer.status()}")


def _image_load(
    index,
    recs,
    triton = True,
):
    g8._CUT_TRITON_ON[index] = triton
    with defer.allow():
        return g8._plan_or_defer(index, recs, triton, False)


def test_first_render_does_not_wait_on_the_race(monkeypatch):
    table: dict = {}
    raced = _env(monkeypatch, table)
    plan = _image_load(0, [_rec()])
    # nothing raced in the load: the shape runs the Triton kernel (no plan entry) until the race has run
    assert raced == [] and plan == {} and _key() not in g8._CUT_PLAN[0]
    assert defer.status()["state"] == "pending"
    t0 = time.perf_counter()
    assert defer.render_begin() is False
    assert time.perf_counter() - t0 < 0.5
    # the render's large calls record their row bucket for the race (a 1024^2 FLUX.3 joint call: 4176 rows)
    g8._defer_note_rows(0, *_call(4176))
    g8._defer_note_rows(0, *_call(300))
    assert table["ck|rows"] == {repr(_key()): [4224]}
    assert g8._cutlass_tile(0, *_call(4176)) is None  # the call stays on the Triton path
    time.sleep(0.2)
    assert (
        raced == [] and defer.status()["state"] == "pending"
    )  # nothing races while the render runs
    defer.render_end()
    _wait_state("done")


def test_race_runs_after_the_first_image_and_persists_its_verdict(monkeypatch):
    table: dict = {}
    raced = _env(monkeypatch, table)
    _image_load(0, [_rec()])
    defer.render_begin()
    g8._defer_note_rows(0, *_call(4176))
    defer.render_end()
    st = _wait_state("done")
    # the race the load used to run, at the first image's real row bucket, against the Triton kernel
    assert raced == [(64, 512, False), (64, 4224, False), (64, 16384, False)]
    assert st["shapes"] == 1 and st["raced"] == 1 and st["won"] == 1
    assert g8._CUT_PLAN[0][_key()] == {512: None, 4224: 8, 16384: 2}
    assert g8._cutlass_tile(0, *_call(4176)) == 8
    # persisted like a load-time race: the next load reads it back
    assert table["ck"][repr(_key())] == {"512": None, "4224": 8, "16384": 2}
    assert table["ck|rows"] == {repr(_key()): [4224]}
    # the next render re-records its CUDA graphs on the winner, once; a nested pass never does
    assert defer.render_begin() is True
    defer.render_end()
    assert defer.render_begin() is False
    defer.render_end()
    defer._STATE["refresh"] = True
    assert defer.render_begin() is True
    defer._STATE["refresh"] = True
    assert defer.render_begin() is False  # hires / Detailer pass inside the render
    defer.render_end()
    defer.render_end()
    assert defer.render_begin() is True
    defer.render_end()


def test_a_stored_verdict_races_and_defers_nothing(monkeypatch):
    skey = repr(_key())
    table = {"ck": {skey: {"512": None, "4224": 8, "16384": 2}}, "ck|rows": {skey: [4224]}}
    raced = _env(monkeypatch, table, stored = (1, 2, 8))
    plan = _image_load(0, [_rec()])
    assert raced == [] and plan[_key()] == {512: None, 4224: 8, 16384: 2}
    assert defer.status()["state"] == "none" and not g8._CUT_DEFER_KEYS
    assert (
        g8._cutlass_tile(0, *_call(4176)) == 8
    )  # the first render already runs the winner, as before


def test_only_the_unstored_shapes_wait(monkeypatch):
    skey = repr(_key())
    table = {"ck": {skey: {"512": None, "4096": 1, "16384": 2}}}
    raced = _env(monkeypatch, table, stored = (1, 2, 8))
    plan = _image_load(0, [_rec(), _rec(96, 32), _rec(96, 32)])
    assert raced == [] and plan == {_key(): {512: None, 4096: 1, 16384: 2}}
    assert defer.status()["shapes"] == 1  # one unit per shape, not per Linear
    defer.render_begin()
    defer.render_end()
    _wait_state("done")
    assert raced == [(96, 512, False), (96, 4096, False), (96, 16384, False)]


def test_opt_out_races_at_load(monkeypatch):
    monkeypatch.setenv(defer.DEFER_ENV, "0")
    raced = _env(monkeypatch, {})
    plan = _image_load(0, [_rec()])
    assert raced == [(64, 512, False), (64, 4096, False), (64, 16384, False)]
    assert plan[_key()] == {512: None, 4096: 1, 16384: 2}
    assert defer.status()["state"] == "none"


@pytest.mark.parametrize("case", ["not_an_image_load", "force", "no_triton"])
def test_loads_that_keep_racing_at_load(monkeypatch, case):
    raced = _env(monkeypatch, {})
    if case == "not_an_image_load":  # a video load has no render hooks to run it later
        g8._plan_or_defer(0, [_rec()], True, False)
    elif case == "force":
        monkeypatch.setenv(cut.CUTLASS_ENV, "1")
        _image_load(0, [_rec()])
    else:  # CUTLASS is the only fused path: deferring would change the module swap
        _image_load(0, [_rec()], triton = False)
    assert len(raced) == 3 and defer.status()["state"] == "none"


def test_a_render_waits_only_for_the_shape_in_flight(monkeypatch):
    release = threading.Event()
    racing = threading.Event()

    def gate(n, m):
        if n == 64 and m == 512:
            racing.set()
            assert release.wait(10)

    raced = _env(monkeypatch, {}, gate = gate)
    _image_load(0, [_rec(), _rec(96, 32)])
    defer.render_begin()
    defer.render_end()
    assert racing.wait(10)
    began = threading.Event()
    t = threading.Thread(target = lambda: (defer.render_begin(), began.set()))
    t.start()
    time.sleep(0.2)
    assert not began.is_set()  # the shape in flight finishes first (its timings stay clean)
    release.set()
    assert began.wait(10)
    t.join()
    time.sleep(0.3)
    assert [r[0] for r in raced] == [64, 64, 64]  # the next shape waits for the render to end
    assert defer.status()["waits"] == 1
    defer.render_end()
    _wait_state("done")
    assert [r[0] for r in raced] == [64, 64, 64, 96, 96, 96]


def test_unload_drops_the_queued_race(monkeypatch):
    raced = _env(monkeypatch, {})
    _image_load(0, [_rec()])
    g8.cancel_deferred_race()
    assert defer.status()["state"] == "cancelled" and not g8._CUT_DEFER_KEYS
    defer.render_begin()
    defer.render_end()
    time.sleep(0.3)
    assert raced == []


def test_image_backend_render_hook(monkeypatch):
    from core.inference.diffusion import DiffusionBackend

    resets = []
    handle = types.SimpleNamespace(reset = lambda: resets.append(1), cache = {})
    backend = object.__new__(DiffusionBackend)
    backend._lock = threading.RLock()
    backend._state = types.SimpleNamespace(cuda_graphs = (handle,), bg_compile = None)
    with backend._int8_defer_render():
        assert defer.allowed()  # a deferred speed install inside the render defers too
    assert resets == [] and not defer.allowed()
    defer._STATE["refresh"] = True  # winners landed between renders
    with backend._int8_defer_render():
        pass
    assert resets == [1]
    with backend._int8_defer_render():
        pass
    assert resets == [1]
    monkeypatch.setenv(defer.DEFER_ENV, "0")
    with backend._int8_defer_render():
        assert not defer.allowed()


def test_a_waiting_render_re_records_on_the_shape_that_just_won(monkeypatch):
    """Each shape's winner is published with the refresh before a render waiting on it is let in, not only once the
    whole card's job ends: else a graph recorded on that shape keeps replaying the Triton launches."""
    release = threading.Event()
    racing = threading.Event()

    def gate(n, m):
        if n == 64 and m == 512:
            racing.set()
            assert release.wait(10)

    _env(monkeypatch, {}, gate = gate)
    _image_load(0, [_rec(), _rec(96, 32)])
    defer.render_begin()
    defer.render_end()
    assert racing.wait(10)
    got = []
    t = threading.Thread(target = lambda: got.append(defer.render_begin()))
    t.start()
    time.sleep(0.2)
    release.set()
    t.join(10)
    assert _key() in g8._CUT_PLAN[0]
    assert got == [True]
    assert (
        defer.status()["pending_units"] == 1
    )  # the 96-row shape still waits for the next idle spell
    defer.render_end()
    _wait_state("done")
    assert defer.render_begin() is True  # the 96-row shape won too
    defer.render_end()


def test_a_shape_that_lost_asks_no_re_record(monkeypatch):
    monkeypatch.setitem(_WINNERS, 4096, None)
    monkeypatch.setitem(_WINNERS, 16384, None)
    _env(monkeypatch, {})
    _image_load(0, [_rec()])
    defer.render_begin()
    defer.render_end()
    st = _wait_state("done")
    assert st["raced"] == 1 and st["won"] == 0
    assert defer.render_begin() is False
    defer.render_end()


def test_a_capture_on_another_card_never_waits_on_the_race(monkeypatch):
    """The deferred unit holds only its own card off captures across its sustained timings: a video capture on another
    card records meanwhile, one on the race's card waits (then runs eager), and a recording on the race's card makes
    the unit retry later."""
    from core.inference import diffusion_cuda_graph as cg

    racing = threading.Event()
    release = threading.Event()

    def plan(
        index,
        recs,
        triton_on,
        logger = None,
    ):
        racing.set()
        assert release.wait(10)
        return {}

    monkeypatch.setattr(g8, "_cutlass_plan", plan)
    monkeypatch.setattr(torch.cuda, "device", lambda *a_: contextlib.nullcontext())
    monkeypatch.setattr(cg, "OUTSIDE_WAIT_S", 0.3)
    out = []
    race = threading.Thread(target = lambda: out.append(g8._defer_unit(0, None, (), _key(), None)))
    race.start()
    try:
        assert racing.wait(10)
        entered = threading.Event()

        def capture_other_card():
            with cg._capturing(1):
                entered.set()

        t = threading.Thread(target = capture_other_card)
        t.start()
        assert entered.wait(2)  # no wait on the race of card 0
        t.join(5)
        with pytest.raises(cg.CaptureRefused):
            with cg._capturing(0):
                pass
        # a capture whose card is unknown waits too (it may be the race's card)
        monkeypatch.setattr(
            torch.cuda, "current_device", lambda: (_ for _ in ()).throw(RuntimeError("no card"))
        )
        with pytest.raises(cg.CaptureRefused):
            with cg._capturing():
                pass
    finally:
        release.set()
        race.join(10)
    assert out == [0]
    assert not cg._DEVICE_HOLDS and not cg._DEVICE_DEPTH
    with cg._capturing(0):
        pass  # the hold is gone
    with cg._capturing(1):
        assert g8._defer_unit(0, None, (), _key(), None) == 0  # another card records: the race runs
    with cg._capturing(0):
        assert g8._defer_unit(0, None, (), _key(), None) is False  # its own card records: retried
    assert not cg._DEVICE_HOLDS and not cg._DEVICE_DEPTH


def test_the_race_frees_memory_only_while_no_card_records(monkeypatch):
    """``torch.cuda.empty_cache`` frees cached blocks on every card: inside the deferred unit it never runs beside
    another card's recording, and other threads keep the real one."""
    from core.inference import diffusion_cuda_graph as cg

    freed = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: freed.append(threading.get_ident()))
    real = torch.cuda.empty_cache
    seen = {}

    def plan(
        index,
        recs,
        triton_on,
        logger = None,
    ):
        torch.cuda.empty_cache()
        seen["inside"] = torch.cuda.empty_cache
        return {}

    monkeypatch.setattr(g8, "_cutlass_plan", plan)
    monkeypatch.setattr(torch.cuda, "device", lambda *a_: contextlib.nullcontext())
    recording, done = threading.Event(), threading.Event()

    def record_on_card_1():
        with cg._capturing(1):
            recording.set()
            assert done.wait(10)

    t = threading.Thread(target = record_on_card_1)
    t.start()
    try:
        assert recording.wait(5)
        assert g8._defer_unit(0, None, (), _key(), None) == 0
    finally:
        done.set()
        t.join(5)
    assert freed == []  # skipped beside card 1's recording
    assert g8._defer_unit(0, None, (), _key(), None) == 0
    assert freed == [threading.get_ident()]
    assert torch.cuda.empty_cache is real and seen["inside"] is not real


def test_a_failed_load_drops_only_its_own_shapes(monkeypatch):
    """The pre-publication rollback cancels the shapes its own arm deferred (they hold its modules), not another's."""
    from core.inference import diffusion as diff

    raced = _env(monkeypatch, {})
    with defer.allow():  # an earlier owner (a render) with a shape still queued
        g8._plan_or_defer(0, [_rec(96, 32)], True, False)
    token = defer.arm()  # the failing load
    try:
        g8._CUT_TRITON_ON[0] = True
        g8._plan_or_defer(0, [_rec()], True, False)
        assert defer.status()["pending_units"] == 4  # (before + one shape) per owner
        diff._cancel_int8_defer_of_this_load()
    finally:
        defer.disarm(token)
    assert defer.status()["pending_units"] == 2
    assert list(g8._CUT_DEFER_KEYS[0]) == [
        g8._cut_key(96, 32, torch.float32, torch.bfloat16, False)
    ]
    diff._cancel_int8_defer_of_this_load()  # not armed here: cancels nothing
    assert defer.status()["pending_units"] == 2
    defer.render_begin()
    defer.render_end()
    _wait_state("done")
    assert [r[0] for r in raced] == [96, 96, 96]


def test_cancel_waits_for_a_finalizer_already_past_job_removal(monkeypatch):
    """A finalizer runs after the worker removed its job: cancel() must wait for it, and it must neither remove a
    replacement load's shape keys nor publish for it."""
    table: dict = {}
    _env(monkeypatch, table)
    finishing = threading.Event()
    release = threading.Event()
    after = g8._defer_after

    def slow_after(index, owner = None):
        finishing.set()
        assert release.wait(10)
        return after(index, owner)

    monkeypatch.setattr(g8, "_defer_after", slow_after)
    _image_load(0, [_rec()])
    defer.render_begin()
    defer.render_end()
    assert finishing.wait(10)
    assert defer._STATE["refresh"] is True  # the shape's win was published with its unit
    cancelled = threading.Event()
    t = threading.Thread(target = lambda: (g8.cancel_deferred_race(), cancelled.set()))
    t.start()
    assert not cancelled.wait(0.3)  # the finalizer in flight belongs to the load being dropped
    release.set()
    assert cancelled.wait(10)
    t.join(5)
    # the replacement load on the same card (a fresh one: nothing stored): nothing of the old one touches it
    monkeypatch.setattr(g8, "_defer_after", after)
    g8._CUT_PLAN.clear()
    g8._CUT_DONE.clear()
    table.clear()
    _image_load(0, [_rec()])
    assert _key() in g8._CUT_DEFER_KEYS[0]
    assert defer.status()["state"] == "pending" and defer.status()["pending_units"] == 2
    assert defer.render_begin() is False
    defer.render_end()
    _wait_state("done")


def test_a_finalizer_outliving_the_cancel_wait_touches_only_its_own_load(monkeypatch):
    table: dict = {}
    _env(monkeypatch, table)
    finishing = threading.Event()
    release = threading.Event()
    after = g8._defer_after

    def slow_after(index, owner = None):
        finishing.set()
        assert release.wait(10)
        return after(index, owner)

    monkeypatch.setattr(g8, "_defer_after", slow_after)
    _image_load(0, [_rec()])
    defer.render_begin()
    defer.render_end()
    assert finishing.wait(10)
    monkeypatch.setattr(defer.cancel, "__defaults__", (0.2, None))
    g8.cancel_deferred_race()  # gives up waiting
    monkeypatch.setattr(g8, "_defer_after", after)
    g8._CUT_PLAN.clear()
    g8._CUT_DONE.clear()
    table.clear()
    blocked = threading.Event()

    def gate(n, m):
        blocked.wait(10)

    monkeypatch.setattr(
        cut, "race", lambda *a_, **k_: (gate(0, 0), {"tile": 1, "us": {}, "m": 0})[1]
    )
    _image_load(0, [_rec()])
    release.set()
    deadline = time.perf_counter() + 5
    while defer.status()["running"] and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert _key() in g8._CUT_DEFER_KEYS[0]  # the replacement's key survives the old finalizer
    assert defer.status()["state"] == "pending"
    assert defer.render_begin() is False
    blocked.set()
    defer.render_end()
    _wait_state("done")


def test_a_complete_stored_race_is_served_without_a_stored_tile_verdict(monkeypatch):
    """An upgrade that moves the tile-probe key (no stored tile verdict) keeps a card's complete stored races: the
    load probes the tiles, as before the deferral, and the first image already runs the stored winner."""
    skey = repr(_key())
    table = {"ck": {skey: {"512": None, "4224": 8, "16384": 2}}, "ck|rows": {skey: [4224]}}
    raced = _env(monkeypatch, table, stored = None)
    plan = _image_load(0, [_rec()])
    assert raced == [] and plan[_key()] == {512: None, 4224: 8, 16384: 2}
    assert defer.status()["state"] == "none" and not g8._CUT_DEFER_KEYS
    assert g8._cutlass_tile(0, *_call(4176)) == 8


def test_an_incomplete_race_without_a_tile_verdict_still_waits(monkeypatch):
    skey = repr(_key())
    table = {"ck": {skey: {"512": None, "4224": 8}}, "ck|rows": {skey: [4224]}}
    raced = _env(monkeypatch, table, stored = None)
    plan = _image_load(0, [_rec()])
    assert raced == [] and plan == {} and defer.status()["state"] == "pending"


def test_an_unrelated_gemm_edit_keeps_the_stored_tile_verdicts(monkeypatch):
    """The tile-probe key covers the probe's own code, not every byte of diffusion_int8_gemm.py: an edit elsewhere
    there (this deferral itself) must not drop every card's tile verdict on upgrade."""
    import pathlib

    from core.inference import diffusion_probe_cache as pc

    def digest():
        return pc.Namespace("int8_cutlass_tiles", cut.PROBE_PERSIST_SOURCES)._source_digest()

    before = digest()
    read = pathlib.Path.read_bytes

    def edited(self):
        data = read(self)
        return (
            data + b"\n\ndef _unrelated_new_helper():\n    return 1\n"
            if self.name == "diffusion_int8_gemm.py"
            else data
        )

    monkeypatch.setattr(pathlib.Path, "read_bytes", edited)
    assert digest() == before
    # the probe's diffusion_int8_gemm input still keys it, through the race code digest in _probe_extra
    assert "tie_operands" in g8._RACE_SOURCES["diffusion_int8_gemm"]
