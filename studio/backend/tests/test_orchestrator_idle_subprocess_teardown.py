# SPDX-License-Identifier: AGPL-3.0-only
"""Unloading the last model tears the inference subprocess down.

empty_cache in the child cannot return the accelerator context, so an idle
worker sat on its high-water mark, and gpu_arbiter never evicts between the
transformers and GGUF backends (both are chat-owned): the memory stayed
stranded for the rest of the session.
"""

import multiprocessing as mp
import threading
import time

from core.inference.orchestrator import InferenceOrchestrator


def _idle_orchestrator(models, loading = ()):
    """An orchestrator whose unload round-trip is stubbed to succeed."""
    o = InferenceOrchestrator.__new__(InferenceOrchestrator)
    o._gen_lock = threading.Lock()
    o._dispatcher_lifecycle_lock = threading.Lock()
    o._drain_event = threading.Event()
    o._unload_pending = False
    o.models = dict(models)
    o.loading_models = set(loading)
    o.active_model_name = next(iter(models), None)

    o.shutdowns = []
    o.cancel_load = lambda _name: False
    o._ensure_subprocess_alive = lambda: True
    o._cancel_generation = lambda: None
    o._wait_dispatcher_idle = lambda: True
    o._drain_queue = lambda: None
    o._send_cmd = lambda _cmd: None
    o._wait_response = lambda _token: None
    o._shutdown_subprocess = lambda timeout = 10.0: o.shutdowns.append(timeout)
    return o


def test_last_unload_shuts_the_subprocess_down():
    o = _idle_orchestrator({"m": {}})

    assert o.unload_model("m") is True
    assert o.models == {}
    assert o.shutdowns, "idle worker kept its VRAM after the last model unloaded"


def test_unload_keeps_the_worker_while_a_model_is_still_resident():
    o = _idle_orchestrator({"m": {}, "other": {}})

    assert o.unload_model("m") is True
    assert o.models == {"other": {}}
    assert o.shutdowns == []


def test_unload_keeps_the_worker_while_a_load_is_in_flight():
    # Tearing down here would kill the load that is about to reuse the worker.
    o = _idle_orchestrator({"m": {}}, loading = ("incoming",))

    assert o.unload_model("m") is True
    assert o.shutdowns == []


# The stubs above cannot see a teardown that deadlocks or raises: the real
# _shutdown_subprocess runs while unload_model holds _gen_lock and has set
# _unload_pending, and it nulls the very _drain_event the unload clears in its
# finally. These drive it for real against a live worker.


def _sleeper(cmd_queue, resp_queue):
    while True:
        try:
            msg = cmd_queue.get(timeout = 0.2)
        except Exception:
            continue
        if isinstance(msg, dict) and msg.get("type") == "shutdown":
            return


def _live_orchestrator(models, loading = ()):
    """A real orchestrator with a live dummy worker; only the round-trip is stubbed."""
    ctx = mp.get_context("spawn")
    o = InferenceOrchestrator()
    o._cmd_queue, o._resp_queue = ctx.Queue(), ctx.Queue()
    o._cancel_event, o._drain_event = ctx.Event(), ctx.Event()
    o._proc = ctx.Process(target = _sleeper, args = (o._cmd_queue, o._resp_queue), daemon = True)
    o._proc.start()
    o.models = dict(models)
    o.loading_models = set(loading)
    o.active_model_name = next(iter(models), None)
    o._send_cmd = lambda _cmd: None
    o._wait_response = lambda _token: None
    return o


def test_real_teardown_kills_the_worker_and_leaves_clean_state():
    o = _live_orchestrator({"m": {}})
    proc, done, result = o._proc, threading.Event(), {}

    def run():
        result["ok"] = o.unload_model("m")
        done.set()

    threading.Thread(target = run, daemon = True).start()
    assert done.wait(60), "unload_model deadlocked on the real teardown"
    assert result["ok"] is True
    assert o.models == {} and o.active_model_name is None
    time.sleep(0.5)
    assert not proc.is_alive(), "worker survived the teardown"
    assert o._proc is None and o._unload_pending is False
    # _gen_lock must be free for the next load.
    assert o._gen_lock.acquire(timeout = 1)
    o._gen_lock.release()


def test_real_teardown_failure_still_reports_the_unload_as_succeeded():
    o = _live_orchestrator({"m": {}})
    proc = o._proc
    try:
        def boom(timeout = 10.0):
            raise RuntimeError("teardown exploded")

        o._shutdown_subprocess = boom
        assert o.unload_model("m") is True
        assert o.models == {}
    finally:
        del o._shutdown_subprocess
        o._shutdown_subprocess(timeout = 5)
        proc.join(timeout = 5)
