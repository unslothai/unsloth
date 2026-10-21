# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import multiprocessing as mp
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import routes.inference as inference_route  # noqa: E402
from core.inference.orchestrator import InferenceOrchestrator  # noqa: E402

MODEL = "unsloth/Qwen3-0.6B"


def _orchestrator(killed = True):
    backend = InferenceOrchestrator()
    ctx = mp.get_context("spawn")
    backend._proc = ctx.Process(target = time.sleep, args = (60,), daemon = True)
    backend._proc.start()
    backend.active_model_name = MODEL
    backend.models = {MODEL: {"engine": "auto"}}
    if killed:
        backend._proc.kill()
        backend._proc.join(timeout = 10)
    return backend


@pytest.fixture
def released(monkeypatch):
    calls = []
    monkeypatch.setattr(inference_route, "release_chat_gpu_claim", lambda: calls.append("release"))
    monkeypatch.setattr(inference_route.account_access, "clear_resident", calls.append)
    return calls


def _status(monkeypatch, backend):
    monkeypatch.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_probe_llama_cpp_status", lambda _b: (False, {}))
    monkeypatch.setattr(inference_route, "_running_load_attempt", None)
    monkeypatch.setattr(inference_route, "_pending_load_attempts", {})
    monkeypatch.setattr(inference_route, "load_inference_config", lambda _model: None)
    return asyncio.run(inference_route.get_status(current_subject = "test"))


def test_status_stops_reporting_a_model_whose_worker_was_killed(monkeypatch, released):
    backend = _orchestrator()

    response = _status(monkeypatch, backend)

    assert response.active_model is None
    assert response.loaded == []
    assert backend.active_model_name is None and backend._proc is None
    assert released == ["release", "chat"]


@pytest.mark.parametrize("state", ["alive", "loading", "replying"])
def test_status_keeps_a_model_whose_worker_may_still_answer(monkeypatch, released, state):
    backend = _orchestrator(killed = state != "alive")
    if state == "loading":
        backend.loading_models = {"org/incoming-model"}
    if state == "replying":
        backend._direct_mailboxes["request"] = object()
    try:
        response = _status(monkeypatch, backend)
    finally:
        backend._proc.kill()
        backend._proc.join(timeout = 10)

    assert response.active_model == MODEL
    assert released == []


def test_loading_the_same_model_again_restarts_a_killed_worker(monkeypatch, released):
    from hub.services.models import account_access
    from models.inference import LoadRequest

    backend = _orchestrator()
    calls = []

    def _record(step):
        def _stop(*args, **kwargs):
            calls.append(step)
            raise RuntimeError(step)

        return _stop

    backend.set_parallel_slots = _record("reuse")
    backend.load_model = _record("load")
    llama = SimpleNamespace(is_loaded = False, is_active = False, model_identifier = None)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: backend)
    monkeypatch.setattr(
        inference_route,
        "_resolve_model_identifier_for_request",
        lambda *a, **k: (MODEL, MODEL, False),
    )
    monkeypatch.setattr(account_access, "require_model_access", lambda *a, **k: None)
    fastapi_request = SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))
    )
    with pytest.raises(Exception):
        asyncio.run(
            inference_route._load_model_impl(
                LoadRequest(model_path = MODEL), fastapi_request, current_subject = "user"
            )
        )

    assert calls[:1] == ["load"]
