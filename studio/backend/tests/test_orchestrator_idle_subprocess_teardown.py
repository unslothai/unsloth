# SPDX-License-Identifier: AGPL-3.0-only
"""Unloading the last model tears the inference subprocess down.

The child's unload frees the model and calls empty_cache, but the caching
allocator only returns blocks nothing still references and the accelerator
context outlives every model, so an idle worker sat on its high-water mark.
gpu_arbiter never evicts between the transformers and GGUF backends (both are
chat-owned), so that memory stayed stranded for the rest of the session.
"""

import threading

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
