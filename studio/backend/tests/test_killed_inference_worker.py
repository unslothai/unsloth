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
    monkeypatch.setattr(
        inference_route, "release_chat_gpu_claim", lambda: calls.append("release") or True
    )
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


def test_an_api_request_clears_a_model_whose_worker_was_killed(monkeypatch, released):
    from starlette.requests import Request

    backend = _orchestrator()
    monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: backend)
    request = Request(
        {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}
    )

    asyncio.run(inference_route._maybe_auto_switch_model(None, request, "user"))

    assert backend.active_model_name is None and backend._proc is None


@pytest.mark.parametrize("owner", ["user", "another-account"])
def test_a_request_is_not_routed_to_a_kept_model_whose_worker_was_killed(
    monkeypatch, released, owner
):
    from core.inference import model_slots

    # A managed installation hides another account's slot from routing, not from the reap.
    monkeypatch.setattr(model_slots.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(model_slots, "current_account_id", lambda: "user")
    kept = _orchestrator()
    llama = SimpleNamespace(is_active = False, is_loaded = False, model_identifier = None)
    slot = model_slots.ExtraSlot(llama = llama, orchestrator = kept, account = owner)
    monkeypatch.setattr(model_slots, "slots", [slot])

    routed = asyncio.run(inference_route._route_to_extra_slot(MODEL))

    assert routed is None
    assert kept.active_model_name is None and kept._proc is None


def test_reaping_the_last_gpu_kept_model_beside_a_cpu_primary_releases_chat(monkeypatch):
    from core.inference import gpu_arbiter, model_slots

    for name, value in (
        ("_owner", gpu_arbiter.CHAT),
        ("_owner_account", None),
        ("_prior_account", None),
        ("_residents", {gpu_arbiter.CHAT: None}),
        ("_accounts", {}),
        ("_media_needs", {}),
    ):
        monkeypatch.setattr(gpu_arbiter, name, value)
    cleared = []
    monkeypatch.setattr(inference_route.account_access, "clear_resident", cleared.append)
    cpu_primary = SimpleNamespace(is_active = True, is_loaded = True, holds_no_vram = True)
    monkeypatch.setattr(inference_route, "_llama_cpp_backend", cpu_primary)
    kept = _orchestrator()
    idle_llama = SimpleNamespace(is_active = False, is_loaded = False)
    slot = model_slots.ExtraSlot(llama = idle_llama, orchestrator = kept, account = "user")
    monkeypatch.setattr(model_slots, "slots", [slot])

    model_slots.in_slot(slot, lambda: inference_route.reap_dead_worker(kept))

    assert not gpu_arbiter.holds(gpu_arbiter.CHAT)
    assert cleared == ["chat"]


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
    from core.inference import gpu_arbiter
    from hub.services.models import account_access
    from models.inference import LoadRequest

    # The load takes the real CHAT claim before its stubbed loader raises.
    for name, empty in (
        ("_owner", None),
        ("_owner_account", None),
        ("_prior_account", None),
        ("_residents", {}),
        ("_accounts", {}),
        ("_media_needs", {}),
    ):
        monkeypatch.setattr(gpu_arbiter, name, empty)
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


@pytest.mark.parametrize("engine", ["vllm", "npu"])
def test_a_managed_or_npu_load_reaps_a_killed_primary_first(monkeypatch, released, engine):
    from core.inference import npu_backend
    from models.inference import LoadRequest

    backend = _orchestrator()
    monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: backend)
    monkeypatch.setattr(npu_backend, "is_npu_model_path", lambda _path: engine == "npu")
    request = LoadRequest(model_path = "org/other", engine = "auto" if engine == "npu" else engine)

    assert asyncio.run(inference_route._select_load_slot(request)) is None
    assert backend.active_model_name is None and backend._proc is None


def test_a_preview_of_another_checkpoint_is_not_refused_by_a_killed_worker(monkeypatch, released):
    from fastapi import HTTPException

    from core.inference import llama_keepwarm
    from models.inference import LoadRequest

    backend = _orchestrator()
    monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: backend)
    monkeypatch.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    monkeypatch.setattr(inference_route, "_resident_npu_model", lambda: None)
    monkeypatch.setattr(inference_route.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(inference_route.account_access, "require_idle_other_accounts", lambda: None)
    # Stop at the busy check right after the ownership one: reaching it means the dead model did not refuse.
    monkeypatch.setattr(llama_keepwarm, "other_admitted_inference_count", lambda: 1)
    request = SimpleNamespace(scope = {"type": "http"})

    with pytest.raises(HTTPException) as refused:
        asyncio.run(
            inference_route.load_model_for_preview(
                LoadRequest(model_path = "org/other"), request, current_subject = "user"
            )
        )

    assert "busy" in refused.value.detail
    assert backend.active_model_name is None and backend._proc is None
