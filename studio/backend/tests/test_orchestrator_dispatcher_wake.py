# SPDX-License-Identifier: AGPL-3.0-only
"""Stopping the dispatcher wakes it at once instead of waiting out its response-queue poll."""

import multiprocessing as mp
import inspect
import json
import queue
import subprocess
import sys
import textwrap
import threading
import time

import pytest

import core.inference.orchestrator as orch_mod
from core.inference.orchestrator import InferenceOrchestrator

_WAKE = {"type": "_dispatcher_wake"}


def _with_queue(resp_queue):
    o = InferenceOrchestrator()
    o._resp_queue = resp_queue
    return o


def test_stop_does_not_wait_for_the_poll_timeout(monkeypatch):
    monkeypatch.setattr(orch_mod, "_DISPATCH_POLL_INTERVAL", 60.0)
    monkeypatch.setattr(orch_mod, "_DISPATCH_STOP_TIMEOUT", 10.0)
    o = _with_queue(mp.get_context("spawn").Queue())
    entered = threading.Event()
    get = o._resp_queue.get

    def parked_get(*args, **kwargs):
        entered.set()
        return get(*args, **kwargs)

    monkeypatch.setattr(o._resp_queue, "get", parked_get)
    thread = o._start_dispatcher()
    assert entered.wait(5)
    started = time.monotonic()
    o._stop_dispatcher()
    assert not thread.is_alive()
    assert time.monotonic() - started < 5.0


def test_failed_wake_still_stops_on_the_poll(monkeypatch):
    monkeypatch.setattr(orch_mod, "_DISPATCH_POLL_INTERVAL", 0.1)

    class _NoPut(queue.Queue):
        def put_nowait(self, item):
            raise OSError("closed")

    o = _with_queue(_NoPut())
    thread = o._start_dispatcher()
    o._stop_dispatcher()
    assert not thread.is_alive()


def test_stray_wake_is_invisible_to_every_reader(monkeypatch):
    q = queue.Queue()
    o = _with_queue(q)
    observed = []
    monkeypatch.setattr(o, "_observe_response", lambda resp, worker: observed.append(resp) or resp)
    q.put(_WAKE)
    q.put({"type": "token", "request_id": "r1", "text": "a"})
    q.put(_WAKE)
    assert o._read_resp(timeout = 1.0) == {"type": "token", "request_id": "r1", "text": "a"}
    assert o._read_resp(timeout = 0.05) is None  # only a wake left: reads as empty
    assert observed == [{"type": "token", "request_id": "r1", "text": "a"}]
    q.put(_WAKE)
    q.put({"type": "gen_done", "request_id": "r1"})
    assert o._drain_queue() == [{"type": "gen_done", "request_id": "r1"}]


def test_restarted_dispatcher_skips_a_stray_wake_and_keeps_order(monkeypatch):
    monkeypatch.setattr(orch_mod, "_DISPATCH_POLL_INTERVAL", 0.05)
    q = queue.Queue()
    o = _with_queue(q)
    mailbox = queue.Queue()
    with o._mailbox_lock:
        o._mailboxes["r1"] = mailbox
    q.put(_WAKE)
    q.put({"type": "token", "request_id": "r1", "text": "a"})
    q.put(_WAKE)
    q.put({"type": "gen_done", "request_id": "r1"})
    thread = o._start_dispatcher()
    got = [mailbox.get(timeout = 5) for _ in range(2)]
    assert [r["type"] for r in got] == ["token", "gen_done"]
    assert thread.is_alive()
    o._stop_dispatcher()
    assert not thread.is_alive() and mailbox.empty()


def test_wake_is_sent_only_while_stopping_a_live_dispatcher():
    q = queue.Queue()
    o = _with_queue(q)
    o._stop_dispatcher()  # none running
    assert q.empty()
    stopped = threading.Thread(target = lambda: None)
    stopped.start()
    stopped.join()
    o._dispatcher_thread = stopped
    o._stop_dispatcher()
    assert q.empty()


@pytest.mark.skipif(sys.platform == "win32", reason = "Windows queues have no shared write lock")
def test_pending_frame_from_dead_writer_does_not_start_a_parent_feeder(tmp_path):
    script = tmp_path / "dead_writer_wake.py"
    method = textwrap.dedent(inspect.getsource(InferenceOrchestrator._wake_dispatcher))
    script.write_text(
        "import json, logging, multiprocessing as mp, os, types\n"
        f"METHOD = {method!r}\n"
        "def worker(q):\n"
        "    q.put({'type': 'speech', 'request_id': 'r1', 'speech': 'x' * 1048576})\n"
        "    if not q._reader.poll(10):\n"
        "        os._exit(2)\n"
        "    os._exit(1)\n"
        "if __name__ == '__main__':\n"
        "    q = mp.get_context('spawn').Queue()\n"
        "    proc = mp.get_context('spawn').Process(target=worker, args=(q,))\n"
        "    proc.start()\n"
        "    try:\n"
        "        proc.join(15)\n"
        "        assert proc.exitcode == 1, proc.exitcode\n"
        "        assert not q.empty()\n"
        "        scope = {'logger': logging.getLogger('test'), '_DISPATCHER_WAKE': '_dispatcher_wake'}\n"
        "        exec(METHOD, scope)\n"
        "        scope['_wake_dispatcher'](types.SimpleNamespace(_resp_queue=q))\n"
        "        feeder = q._thread\n"
        "        q._writer.close()\n"
        "        q._reader.close()\n"
        "        q.close()\n"
        "        if feeder is not None:\n"
        "            feeder.join(0.2)\n"
        "        print(json.dumps({'feeder_started': feeder is not None,\n"
        "                          'feeder_still_alive': feeder is not None and feeder.is_alive()}))\n"
        "    finally:\n"
        "        if proc.is_alive():\n"
        "            proc.kill()\n"
        "        proc.join(5)\n",
        encoding = "utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script)], capture_output = True, text = True, timeout = 30
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"feeder_started": False, "feeder_still_alive": False}
