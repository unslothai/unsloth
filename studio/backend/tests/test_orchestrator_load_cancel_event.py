# SPDX-License-Identifier: AGPL-3.0-only

import threading
import time
from types import SimpleNamespace

from core.inference import orchestrator as orch_mod
from core.inference.orchestrator import InferenceOrchestrator


def test_safetensors_load_event_interrupts_worker_wait(monkeypatch):
    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orchestrator.__dict__.update(
        loading_models = set(), active_model_name = None, models = {}, _proc = None, _resp_queue = object()
    )
    cancel_event = threading.Event()
    shutdowns = []
    spawned = {"value": False}
    monkeypatch.setattr(orch_mod, "prepare_gpu_selection", lambda *_a, **_k: ([], "auto"))
    monkeypatch.setattr(orch_mod, "get_device", lambda: SimpleNamespace(value = "cpu"))
    monkeypatch.setattr("utils.transformers_version.needs_transformers_5", lambda _model: False)
    monkeypatch.setattr("utils.transformers_version.sidecar_swap_kind", lambda: None)
    monkeypatch.setattr(orchestrator, "_ensure_subprocess_alive", lambda: spawned["value"])
    monkeypatch.setattr(
        orchestrator, "_spawn_subprocess", lambda _config: spawned.__setitem__("value", True)
    )
    monkeypatch.setattr(orchestrator, "_read_resp", lambda timeout = 1: time.sleep(timeout) or None)
    monkeypatch.setattr(
        orchestrator, "_shutdown_subprocess", lambda timeout = 5: shutdowns.append(timeout) or True
    )
    timer = threading.Timer(0.05, cancel_event.set)
    timer.start()
    try:
        started = time.monotonic()
        loaded = orchestrator.load_model(
            SimpleNamespace(identifier = "local.safetensors"), load_cancel_event = cancel_event
        )
    finally:
        timer.cancel()
    assert loaded is False
    assert time.monotonic() - started < 0.5
    assert shutdowns == [5]
    assert orchestrator.loading_models == set()
    assert orchestrator.models == {}
