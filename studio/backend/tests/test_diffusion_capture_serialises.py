# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A CUDA graph recording on one Studio thread survives the CUDA work every other Studio thread does meanwhile.

Measured (torch 2.14, CUDA 13): while a global-mode capture records, another thread's event query, host read, pinned
alloc, ``empty_cache``, stream sync or first cuBLAS / cuDNN call invalidates it (and raises in that thread); in
thread_local mode only a device-wide synchronize does, so captures record thread_local, exclude each other, and
Studio's device syncs wait for another thread's recording."""

from __future__ import annotations

import threading
import time

import pytest
import torch

import core.inference.diffusion_cuda_graph as cg

_CUDA = torch.cuda.is_available()
# Long enough that the other thread's call lands inside the recording; short, because the fixed code holds it out.
WINDOW_S = 1.0


@pytest.fixture(autouse = True)
def _fresh_gate(monkeypatch):
    monkeypatch.setattr(cg, "_FAILED_GRAPHS", [])
    monkeypatch.setattr(cg, "_POOL_BOX", [None])
    monkeypatch.setattr(cg, "_OUTSIDE_STUCK", [False])
    yield
    assert cg._CAPTURE_DEPTH == 0 and cg._OUTSIDE_JOBS == 0


class _Step(torch.nn.Module):
    """A denoiser step; while it records it opens ``window`` and holds the recording open until ``done`` (or 1 s)."""

    def __init__(
        self,
        window = None,
        done = None,
    ) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(64, 64)
        self.window, self.done = window, done

    def forward(
        self,
        x,
        return_dict = True,
    ):
        y = self.lin(x)
        if self.window is not None and torch.cuda.is_current_stream_capturing():
            self.window.set()
            self.done.wait(WINDOW_S)
        y = torch.nn.functional.gelu(y) * 2
        return (y,) if return_dict is False else y


def _run_in_thread(fn):
    box = {}

    def target():
        try:
            torch.cuda.set_device(0)
            with torch.inference_mode():
                box["out"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target = target, name = "studio-other-thread")
    return thread, box


def _record_while(other):
    """Record ``_Step`` on this thread while ``other`` runs on a fresh thread inside the recording window."""
    window, done = threading.Event(), threading.Event()
    step = _Step(window, done).cuda()
    handle = cg.GraphedForward(step).enable()
    x = torch.randn(8, 64, device = "cuda")

    def gated():
        assert window.wait(300)  # first CUDA use can take minutes on a loaded host
        try:
            return other()
        finally:
            done.set()

    thread, box = _run_in_thread(gated)
    thread.start()
    try:
        with torch.inference_mode():
            first = step(x, return_dict = False)[0].clone()
            again = step(x, return_dict = False)[0].clone()
    finally:
        done.set()
        thread.join(60)
    return step, handle, x, first, again, box


def _expect_graph_kept(step, handle, x, first, again):
    try:
        assert handle.capture_error is None, handle.capture_error
        assert not handle.poisoned
        assert handle.stats["captures"] == 1 and handle.stats["replays"] >= 1, handle.stats
        with torch.inference_mode():
            ref = torch.nn.functional.gelu(step.lin(x)) * 2
        assert torch.equal(first, ref) and torch.equal(again, ref)
    finally:
        handle.free()


@pytest.mark.skipif(not _CUDA, reason = "needs CUDA")
def test_a_recording_survives_another_threads_decode_tail():
    """A recording must survive another thread's decode (first cuBLAS/cuDNN use, host read, event
    query, pinned alloc); its noise uses an explicit generator."""

    def decode_tail():
        gen = torch.Generator(device = "cuda").manual_seed(0)
        w = torch.randn(64, 64, device = "cuda", generator = gen)
        img = torch.nn.functional.conv2d(
            torch.randn(1, 4, 32, 32, device = "cuda", generator = gen),
            torch.randn(3, 4, 3, 3, device = "cuda", generator = gen),
        )
        host = (img.float().clamp(-1, 1) * 127.5 + 127.5).to(torch.uint8).cpu()
        torch.cuda.Event().query()
        torch.empty(1 << 16).pin_memory()
        return (w @ w).sum().item(), host.shape

    step, handle, x, first, again, box = _record_while(decode_tail)
    assert "error" not in box, box["error"]
    _expect_graph_kept(step, handle, x, first, again)


def _torch_at_least(major: int, minor: int) -> bool:
    return tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2]) >= (major, minor)


@pytest.mark.skipif(not _CUDA, reason = "needs CUDA")
def test_another_threads_default_generator_draw_during_a_recording():
    """Up to torch 2.11 a capture flags the device's DEFAULT generator as capturing for every thread, so another
    thread's ``torch.randn(device = "cuda")`` raises "Offset increment outside graph capture encountered
    unexpectedly" in THAT thread; the recording itself is kept. torch 2.12 keys the RNG capture state by capture id
    (pytorch#176752), so the draw just runs. Studio draws its noise from explicit generators for that reason."""

    def default_draw():
        return torch.randn(64, device = "cuda").sum().item()

    step, handle, x, first, again, box = _record_while(default_draw)
    if _torch_at_least(2, 12):
        assert "error" not in box, box["error"]
    else:
        assert "Offset increment outside graph capture" in box.get("error", ""), box
    _expect_graph_kept(step, handle, x, first, again)
    # The default generator still draws once the recording ended.
    torch.randn(4, device = "cuda").sum().item()


@pytest.mark.skipif(not _CUDA, reason = "needs CUDA")
def test_two_threads_recording_at_once_both_keep_their_graphs():
    """A second capture's entry waits out the first's recording instead of breaking it."""
    other = _Step().cuda()
    other_handle = cg.GraphedForward(other).enable()
    y = torch.randn(8, 64, device = "cuda")

    def second_capture():
        out = other(y, return_dict = False)[0].clone()
        return out, other(y, return_dict = False)[0].clone()

    step, handle, x, first, again, box = _record_while(second_capture)
    assert "error" not in box, box["error"]
    try:
        assert other_handle.capture_error is None, other_handle.capture_error
        assert other_handle.stats["captures"] == 1 and other_handle.stats["replays"] >= 1
        with torch.inference_mode():
            ref = torch.nn.functional.gelu(other.lin(y)) * 2
        assert all(torch.equal(t, ref) for t in box["out"])
    finally:
        other_handle.free()
    _expect_graph_kept(step, handle, x, first, again)


@pytest.mark.skipif(not _CUDA, reason = "needs CUDA")
def test_a_studio_device_sync_on_another_thread_waits_for_the_recording():
    """A device-wide synchronize breaks a recording in any capture mode: Studio's sync waits the recording out."""
    seen = {}

    def studio_sync():
        t0 = time.perf_counter()
        cg.synchronize_outside_capture(torch.device("cuda", 0))
        seen["waited_s"] = time.perf_counter() - t0

    step, handle, x, first, again, box = _record_while(studio_sync)
    assert "error" not in box, box["error"]
    _expect_graph_kept(step, handle, x, first, again)
    assert seen["waited_s"] >= WINDOW_S * 0.5  # held out until the recording ended


def test_the_whole_step_records_thread_local():
    assert cg.CAPTURE_MODE == "thread_local"


def _hold_capture_open(entered, release):
    def target():
        with cg._capturing():
            entered.set()
            release.wait(10)

    return threading.Thread(target = target)


def test_a_capture_waits_for_another_threads_recording():
    entered, release = threading.Event(), threading.Event()
    first = _hold_capture_open(entered, release)
    first.start()
    assert entered.wait(10)
    second_in = threading.Event()

    def second():
        with cg._capturing():
            second_in.set()

    t = threading.Thread(target = second)
    t.start()
    assert not second_in.wait(0.3)  # held out while the first records
    release.set()
    first.join(10)
    assert second_in.wait(10)
    t.join(10)


def test_a_threads_own_nested_capture_does_not_wait_on_itself():
    with cg._capturing():
        assert cg.recording_here()
        with cg._capturing():
            assert cg._CAPTURE_DEPTH == 2
    assert not cg.recording_here()


def test_a_capture_held_out_too_long_is_refused_not_recorded(monkeypatch):
    monkeypatch.setattr(cg, "RETIRED_FLUSH_WAIT_S", 0.2)
    entered, release = threading.Event(), threading.Event()
    first = _hold_capture_open(entered, release)
    first.start()
    assert entered.wait(10)
    try:
        with pytest.raises(cg.CaptureRefused):
            with cg._capturing():
                pass
    finally:
        release.set()
        first.join(10)


def test_synchronize_outside_capture_waits_and_holds_captures_off(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device = None: calls.append((device, cg.capture_in_progress(), cg._OUTSIDE_JOBS)),
    )
    entered, release = threading.Event(), threading.Event()
    first = _hold_capture_open(entered, release)
    first.start()
    assert entered.wait(10)
    t = threading.Thread(target = cg.synchronize_outside_capture, args = ("cuda:0",))
    t.start()
    time.sleep(0.3)
    assert calls == []  # waiting for the recording to end
    release.set()
    first.join(10)
    t.join(10)
    # ran with no recording, and counted, so no capture could enter during it
    assert calls == [("cuda:0", False, 1)]


def test_synchronize_inside_this_threads_recording_does_not_wait_on_itself(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device = None: calls.append(device))
    monkeypatch.setattr(cg, "RETIRED_FLUSH_WAIT_S", 30.0)
    t0 = time.perf_counter()
    with cg._capturing():
        cg.synchronize_outside_capture()
    assert calls == [None] and time.perf_counter() - t0 < 5


def test_synchronize_under_hold_off_capture_does_not_retake_the_lock(monkeypatch):
    """The ring drop syncs under ``hold_off_capture`` (the lock is not reentrant): that sync runs straight away."""
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device = None: calls.append(device))
    done = threading.Event()

    def ring_drop():
        with cg.hold_off_capture() as safe:
            assert safe
            cg.synchronize_outside_capture("cuda:0")
        done.set()

    t = threading.Thread(target = ring_drop, daemon = True)
    t.start()
    assert done.wait(5), "deadlocked on the capture lock"
    t.join(5)
    assert calls == ["cuda:0"]


def test_a_captures_own_sync_is_refused_but_a_plain_sync_still_runs_after_the_wait(monkeypatch):
    """Past the wait a capture's pre-record sync refuses; any other sync still runs."""
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device = None: calls.append(device))
    monkeypatch.setattr(cg, "RETIRED_FLUSH_WAIT_S", 0.2)
    entered, release = threading.Event(), threading.Event()
    first = _hold_capture_open(entered, release)
    first.start()
    assert entered.wait(10)
    try:
        with pytest.raises(cg.CaptureRefused):
            cg.synchronize_outside_capture(refuse = True)
        assert calls == []
        cg.synchronize_outside_capture("cuda:0")
        assert calls == ["cuda:0"]
    finally:
        release.set()
        first.join(10)


class _FakeDriver:
    def __init__(self, calls):
        self.calls = calls

    def cuMemUnmap(self, va, size):
        self.calls.append("unmap")
        return 0

    def cuMemRelease(self, handle):
        self.calls.append("release")
        return 0

    def cuMemAddressFree(self, va, size):
        self.calls.append("free")
        return 0


def _fake_arena(monkeypatch, calls):
    """A mapped ``VmmArena`` over a fake driver, so the finalizer path runs on any host."""
    import contextlib

    import core.inference.diffusion_stable_arena as sa

    monkeypatch.setattr(sa, "_driver", lambda: _FakeDriver(calls))
    monkeypatch.setattr(torch.cuda, "device", lambda _index: contextlib.nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device = None: calls.append(
            ("sync", cg._CAPTURE_DEPTH > 0)
        ),  # lock-free: may run under it
    )
    arena = object.__new__(sa.VmmArena)
    arena.size, arena.device_index, arena.va = 1 << 21, 0, 1 << 40
    arena._handle, arena._base = 7, object()
    arena.maps, arena.unmaps, arena.closed = 1, 0, False
    sa._count_mapped(0, arena.size)
    return sa, arena


def _on_thread_with_timeout(fn, timeout = 5.0):
    """Run ``fn`` on a fresh thread: a deadlock fails the test instead of hanging it."""
    box = {}

    def target():
        try:
            box["out"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target = target, daemon = True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "deadlocked on the capture lock"
    if "error" in box:
        raise box["error"]
    return box.get("out")


def test_an_arena_finalizer_under_this_threads_capture_lock_defers_not_deadlocks(monkeypatch):
    """GC can run an arena finalizer on a thread inside the gate's own ``Condition`` setup (lock held, no flag set):
    it must queue the release, not re-take the non-reentrant lock; the next gated sync frees it after a device sync."""
    monkeypatch.setattr(cg, "_DEFERRED_RELEASES", __import__("collections").deque(), raising = False)
    calls = []
    sa, arena = _fake_arena(monkeypatch, calls)

    def finalizer_under_lock():
        with cg._CAPTURE_IDLE:  # what synchronize_outside_capture holds around wait_for
            sa.ModuleArena._finalize({"arena": arena})
            assert calls == [] and not arena.closed and arena.mapped
            assert len(cg._DEFERRED_RELEASES) == 1

    _on_thread_with_timeout(finalizer_under_lock)
    _on_thread_with_timeout(lambda: cg.synchronize_outside_capture("cuda:0"))
    assert arena.closed and not arena.mapped and len(cg._DEFERRED_RELEASES) == 0
    # the gated sync, then the deferred close: its own sync with no capture recording, before the memory goes
    assert calls == [("sync", False), ("sync", False), "unmap", "release", "free"]


def test_an_arena_finalizer_during_another_threads_recording_waits_for_the_capture_exit(
    monkeypatch,
):
    monkeypatch.setattr(cg, "_DEFERRED_RELEASES", __import__("collections").deque(), raising = False)
    calls = []
    sa, arena = _fake_arena(monkeypatch, calls)
    entered, release = threading.Event(), threading.Event()
    first = _hold_capture_open(entered, release)
    first.start()
    assert entered.wait(10)
    try:
        _on_thread_with_timeout(lambda: sa._close_ranges([types_ns(arena = arena)]))
        assert calls == [] and arena.mapped  # never freed under a recording
    finally:
        release.set()
        first.join(10)
    assert not first.is_alive()
    # drained by the capture's exit, with no capture recording
    assert arena.closed and not arena.mapped
    assert calls == [("sync", False), "unmap", "release", "free"]


def test_an_arena_finalizer_with_the_gate_free_closes_at_once(monkeypatch):
    monkeypatch.setattr(cg, "_DEFERRED_RELEASES", __import__("collections").deque(), raising = False)
    calls = []
    sa, arena = _fake_arena(monkeypatch, calls)
    _on_thread_with_timeout(lambda: sa.ModuleArena._finalize({"arena": arena}))
    assert arena.closed and len(cg._DEFERRED_RELEASES) == 0
    assert calls == [("sync", False), "unmap", "release", "free"]


def types_ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


@pytest.mark.skipif(not _CUDA, reason = "needs CUDA")
def test_a_real_arena_finalizer_under_the_capture_lock_frees_on_the_next_gated_sync():
    import core.inference.diffusion_stable_arena as sa

    torch.cuda.set_device(0)
    module = torch.nn.Linear(8, 8)
    marena = sa.ModuleArena(module, "cuda:0")
    module.to("cuda:0")
    arena = marena.arena
    assert arena is not None and arena.mapped

    def finalizer_under_lock():
        with cg._CAPTURE_IDLE:
            sa.ModuleArena._finalize({"arena": arena})
        return arena.mapped

    assert _on_thread_with_timeout(finalizer_under_lock) is True
    _on_thread_with_timeout(cg.synchronize_outside_capture)
    assert arena.closed and not arena.mapped
    del module, marena
