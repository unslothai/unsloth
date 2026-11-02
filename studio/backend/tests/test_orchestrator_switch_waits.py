# SPDX-License-Identifier: AGPL-3.0-only
"""Worker teardown on a model switch waits on worker exit, not on fixed sleeps."""

import inspect
import multiprocessing as mp
import os
import queue
import threading
import time

import pytest

import core.inference.orchestrator as orch_mod
from core.inference.orchestrator import InferenceOrchestrator


def _worker(cmd_queue, resp_queue, cancel_event, unload_s, ignore_cancel_s, tok_bytes):
    while True:
        try:
            cmd = cmd_queue.get(timeout = 0.2)
        except queue.Empty:
            continue
        kind = cmd.get("type")
        if kind == "generate":
            cancel_event.clear()
            rid = cmd.get("request_id")
            started = time.monotonic()
            sent = 0
            while not cancel_event.is_set() or time.monotonic() - started < ignore_cancel_s:
                resp_queue.put({"type": "token", "request_id": rid, "text": "x" * tok_bytes})
                sent += 1
                time.sleep(0.005)
            resp_queue.put({"type": "gen_done", "request_id": rid, "stats": {"tokens": sent}})
        elif kind == "shutdown":
            time.sleep(unload_s)
            resp_queue.put({"type": "shutdown_ack"})
            return


def _live(
    unload_s = 0.0,
    ignore_cancel_s = 0.0,
    tok_bytes = 8,
    generating = False,
):
    ctx = mp.get_context("spawn")
    o = InferenceOrchestrator()
    o._cmd_queue, o._resp_queue = ctx.Queue(), ctx.Queue()
    o._cancel_event, o._drain_event = ctx.Event(), ctx.Event()
    o._proc = ctx.Process(
        target = _worker,
        args = (o._cmd_queue, o._resp_queue, o._cancel_event, unload_s, ignore_cancel_s, tok_bytes),
        daemon = True,
    )
    o._proc.start()
    if generating:
        o._cmd_queue.put({"type": "generate"})
        o._resp_queue.get(timeout = 30)  # streaming has begun
    return o


class _Recorder:
    def __init__(self, monkeypatch):
        self.terminated = []
        self.sleeps = []
        import utils.process_lifetime as lifetime

        monkeypatch.setattr(
            lifetime, "terminate_pid", lambda pid, timeout = 5: self.terminated.append(pid)
        )
        real_sleep = time.sleep
        main = threading.get_ident()

        def sleep(seconds):
            if threading.get_ident() == main:
                self.sleeps.append(seconds)
            real_sleep(seconds)

        monkeypatch.setattr(orch_mod.time, "sleep", sleep)


def test_idle_worker_shutdown_has_no_fixed_sleep(monkeypatch):
    rec = _Recorder(monkeypatch)
    o = _live()
    proc = o._proc
    assert o._shutdown_subprocess() is True
    assert not proc.is_alive() and o._proc is None
    assert rec.sleeps == [] and rec.terminated == []


def test_slow_teardown_is_waited_out_not_terminated(monkeypatch):
    rec = _Recorder(monkeypatch)
    o = _live(unload_s = 1.0)
    proc = o._proc
    started = time.monotonic()
    assert o._shutdown_subprocess() is True
    assert time.monotonic() - started >= 1.0
    assert not proc.is_alive()
    assert rec.terminated == []


def test_cancelled_generation_finishes_before_exit(monkeypatch):
    rec = _Recorder(monkeypatch)
    o = _live(generating = True)
    proc = o._proc
    assert o._shutdown_subprocess() is True
    assert not proc.is_alive() and rec.terminated == []


def test_late_cancel_with_full_pipe_still_exits_cleanly(monkeypatch):
    # 0.8 s of ~4 KB tokens overfills a 64 KB pipe: the worker exits only if the parent keeps draining.
    rec = _Recorder(monkeypatch)
    o = _live(ignore_cancel_s = 0.8, tok_bytes = 4096, generating = True)
    proc = o._proc
    started = time.monotonic()
    assert o._shutdown_subprocess(timeout = 5) is True
    assert time.monotonic() - started < 5.0
    assert not proc.is_alive()
    assert rec.terminated == [], "a full reply pipe forced the terminate fallback"


class _NeverExits:
    pid = 999999
    exitcode = None

    def __init__(self, with_sentinel):
        if with_sentinel:
            self.sentinel = object()
        self.joins = []
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout = None):
        self.joins.append(timeout)
        if timeout and self.alive:
            time.sleep(timeout)  # module time, unpatched here

    def terminate(self):
        self.alive = False

    def kill(self):
        self.alive = False


def _fake(proc):
    o = InferenceOrchestrator()
    o._proc = proc
    o._cmd_queue = queue.Queue()
    o._resp_queue = queue.Queue()
    o._cancel_event = threading.Event()
    return o


def test_condition_never_met_keeps_old_budget_then_escalates(monkeypatch):
    import utils.process_lifetime as lifetime

    terminated = []
    proc = _NeverExits(with_sentinel = True)

    def terminate_pid(pid, timeout = 5):
        terminated.append(pid)
        proc.alive = False

    monkeypatch.setattr(lifetime, "terminate_pid", terminate_pid)
    o = _fake(proc)
    cmds = o._cmd_queue
    started = time.monotonic()
    assert o._shutdown_subprocess(timeout = 0.3) is True
    waited = time.monotonic() - started
    assert 0.3 + orch_mod._SHUTDOWN_CANCEL_GRACE_S - 0.05 <= waited < 2.0, waited
    assert terminated == [proc.pid]
    assert proc.joins and all(t <= orch_mod._SHUTDOWN_JOIN_SLICE_S + 1e-9 for t in proc.joins[:-1])
    assert cmds.get_nowait() == {"type": "shutdown"}


def test_handle_without_sentinel_gets_one_join_with_the_full_budget(monkeypatch):
    import utils.process_lifetime as lifetime

    monkeypatch.setattr(lifetime, "terminate_pid", lambda pid, timeout = 5: None)
    proc = _NeverExits(with_sentinel = False)
    proc.join = lambda timeout = None: proc.joins.append(timeout)
    o = _fake(proc)
    o._shutdown_subprocess(timeout = 2)
    assert proc.joins[0] == 2 + orch_mod._SHUTDOWN_CANCEL_GRACE_S


def test_source_has_no_fixed_switch_sleeps():
    shutdown_src = inspect.getsource(InferenceOrchestrator._shutdown_subprocess_locked)
    assert "time.sleep" not in shutdown_src
    load_src = inspect.getsource(InferenceOrchestrator.load_model)
    assert "time.sleep(0.3)" not in load_src


@pytest.mark.parametrize("reader", ["direct", "dispatched"])
def test_streaming_reader_keeps_its_tail_across_a_switch(monkeypatch, reader):
    _Recorder(monkeypatch)
    o = _live()
    proc = o._proc
    cancel_event = threading.Event()
    if reader == "direct":
        read_one, drain, release = o._direct_reader("r1", cancel_event)
    else:
        o._start_dispatcher()
        mailbox = orch_mod._WorkerMailbox(proc)
        with o._mailbox_lock:
            o._mailboxes["r1"] = mailbox

        def read_one(timeout):
            try:
                return o._read_mailbox(mailbox, timeout)
            except queue.Empty:
                return None

        def drain():
            o._drain_mailbox(mailbox, timeout = 5.0)

        def release():
            with o._mailbox_lock:
                o._mailboxes.pop("r1", None)

    o._cmd_queue.put({"type": "generate", "request_id": "r1"})
    seen = {"tokens": 0, "error": None, "finished": False}
    stats = {}

    def consume():
        try:
            for item in o._consume_token_stream(
                read_one,
                drain,
                crash_context = "generation",
                request_id = "r1",
                cancel_event = cancel_event,
                stats_holder = stats,
                read_timeout = 2.0,
            ):
                if isinstance(item, orch_mod.GenStreamError):
                    seen["error"] = str(item)
                    return
                seen["tokens"] += 1
                time.sleep(0.02)  # real thread sleep: slower than the worker's 5 ms tokens
            seen["finished"] = True
        finally:
            release()

    reader = threading.Thread(target = consume, daemon = True)
    reader.start()
    deadline = time.monotonic() + 30
    while seen["tokens"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)  # worker spawned and streaming: cancel must not land before it clears
    time.sleep(0.2)
    assert o._shutdown_subprocess() is True
    reader.join(5)
    assert not proc.is_alive()
    assert seen["error"] is None, seen
    assert seen["finished"] and seen["tokens"] > 0, seen
    assert stats["stats"]["tokens"] == seen["tokens"], (stats, seen)


def test_idle_switch_does_not_wait_for_readers(monkeypatch):
    rec = _Recorder(monkeypatch)
    o = _live()
    assert o._shutdown_subprocess() is True
    assert rec.sleeps == [] and rec.terminated == []


def test_stalled_reader_is_abandoned_after_the_idle_grace(monkeypatch):
    rec = _Recorder(monkeypatch)
    o = _live()
    with o._mailbox_lock:
        o._mailboxes["gone"] = orch_mod._WorkerMailbox(o._proc)
    waits = []
    real = o._let_readers_finish

    def timed():
        started = time.monotonic()
        real()
        waits.append(time.monotonic() - started)

    o._let_readers_finish = timed
    assert o._shutdown_subprocess() is True
    assert orch_mod._SHUTDOWN_READER_IDLE_S <= waits[0] < orch_mod._SHUTDOWN_READER_MAX_S, waits
    assert rec.terminated == []


def _send_interrupted_frame(replies, send):
    send.wait()
    replies.put({"type": "diffusion_frame", "request_id": "r1", "image": "x" * 1048576})
    time.sleep(60)


def _shutdown_interrupted_writer():
    ctx = mp.get_context("spawn")
    o = InferenceOrchestrator()
    o._cmd_queue, o._resp_queue = ctx.Queue(), ctx.Queue()
    o._cancel_event = ctx.Event()
    send = ctx.Event()
    o._proc = ctx.Process(target = _send_interrupted_frame, args = (o._resp_queue, send))
    o._proc.start()
    proc = o._proc
    o._mailboxes["r1"] = orch_mod._WorkerMailbox(proc)
    drain = o._drain_unread_replies
    calls = 0

    def enqueue_after_drain():
        nonlocal calls
        replies = drain()
        calls += 1
        if calls == 2:
            send.set()
            assert o._resp_queue._reader.poll(30)
            time.sleep(orch_mod._SHUTDOWN_CANCEL_GRACE_S + 0.1)
        return replies

    o._drain_unread_replies = enqueue_after_drain
    try:
        assert o._shutdown_subprocess(timeout = 0.01) is True
        assert o._proc is None and not proc.is_alive()
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(5)


def test_forced_shutdown_does_not_read_an_interrupted_queue_frame():
    runner = mp.get_context("spawn").Process(target = _shutdown_interrupted_writer)
    runner.start()
    try:
        runner.join(30)
        assert not runner.is_alive(), "shutdown blocked reading the killed writer's partial frame"
        assert runner.exitcode == 0
    finally:
        if runner.is_alive():
            runner.kill()
        runner.join(5)


def _crash_during_frame(replies, send):
    send.wait()
    replies.put({"type": "diffusion_frame", "request_id": "r1", "image": "x" * 1048576})
    assert replies._reader.poll(30)
    os._exit(1)


def _shutdown_crashed_writer():
    ctx = mp.get_context("spawn")
    o = InferenceOrchestrator()
    o._cmd_queue, o._resp_queue = ctx.Queue(), ctx.Queue()
    o._cancel_event = ctx.Event()
    send = ctx.Event()
    proc = ctx.Process(target = _crash_during_frame, args = (o._resp_queue, send))
    o._proc = proc
    proc.start()
    o._mailboxes["r1"] = orch_mod._WorkerMailbox(proc)
    drain = o._drain_unread_replies
    calls = 0

    def crash_after_drain():
        nonlocal calls
        replies = drain()
        calls += 1
        if calls == 1:
            send.set()
            proc.join(30)
            assert proc.exitcode == 1
        return replies

    o._drain_unread_replies = crash_after_drain
    try:
        assert o._shutdown_subprocess(timeout = 0.01) is True
        assert o._proc is None and not proc.is_alive()
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(5)


def test_worker_crash_mid_frame_cannot_block_the_shutdown_deadline():
    runner = mp.get_context("spawn").Process(target = _shutdown_crashed_writer)
    runner.start()
    try:
        runner.join(40)
        assert not runner.is_alive(), "join loop blocked on the crashed writer's partial frame"
        assert runner.exitcode == 0
    finally:
        if runner.is_alive():
            runner.kill()
        runner.join(5)


def _speech_then_shutdown(commands, replies):
    replies.put({"type": "speech", "request_id": "r1", "speech": "x" * 1048576})
    replies.put({"type": "gen_done", "request_id": "r1"})
    assert commands.get(timeout = 30) == {"type": "shutdown"}


def test_paused_direct_reader_cannot_force_termination_of_a_finished_worker(monkeypatch):
    rec = _Recorder(monkeypatch)
    ctx = mp.get_context("spawn")
    o = InferenceOrchestrator()
    o._cmd_queue, o._resp_queue = ctx.Queue(), ctx.Queue()
    o._cancel_event = ctx.Event()
    proc = ctx.Process(target = _speech_then_shutdown, args = (o._cmd_queue, o._resp_queue))
    o._proc = proc
    owner = threading.Event()
    _, _, release = o._direct_reader("r1", owner)
    proc.start()
    try:
        assert o._resp_queue._reader.poll(30)
        assert o._shutdown_subprocess(timeout = 2) is True
        assert proc.exitcode == 0, "the paused reader prevented a clean worker exit"
        assert rec.terminated == []
        assert owner.is_set()
    finally:
        release()
        if proc.is_alive():
            proc.kill()
        proc.join(5)
