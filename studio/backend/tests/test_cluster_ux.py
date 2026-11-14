# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a user reads about a model served across paired cluster workers."""

from __future__ import annotations

import httpx
import pytest
from fastapi import HTTPException

from core.inference import cluster_serving as cs
from core.inference import cluster_ux
from core.inference import spark_serving as ss
from routes import inference as routes
from .test_cluster_serving import GIB, _cluster_env, _request, clean, run  # noqa: F401

_WORKERS = "10.0.0.2, 10.0.0.3 and 10.0.0.4"
_SPLIT = f"Split across this machine, {_WORKERS}."


def _loaded_split(monkeypatch, serve = True):
    _cluster_env(monkeypatch)
    out = run(ss.before_load(_request(), 4))
    backend = type("B", (), {})()
    backend._process = type("P", (), {"args": ["/b/llama-server"] + out.llama_extra_args})()
    backend.is_loaded = True
    if serve:
        run(ss.after_load(backend, 4))
    return backend


def test_load_progress_says_what_a_split_load_is_doing(monkeypatch):
    st = cs.state()
    st.in_flight, st.plan = True, None
    assert "planning" in cluster_ux.load_progress_message(llama_started = False)
    st.in_flight = False

    backend = _loaded_split(monkeypatch, serve = False)
    assert cs.state().in_flight
    starting = cluster_ux.load_progress_message(llama_started = False)
    assert starting == f"Starting the model server, split across this machine, {_WORKERS}."
    sending = cluster_ux.load_progress_message(llama_started = True)
    assert sending.startswith(f"Sending the model's layers to {_WORKERS} over the network")
    assert routes._cluster_load_message({"phase": "mmap"}).startswith("Sending the model's layers")
    assert routes._cluster_load_message({"phase": "ready"}) is None

    run(ss.after_load(backend, 4))
    assert cluster_ux.load_progress_message(llama_started = True) is None
    assert cluster_ux.split_summary() == _SPLIT
    assert routes.LoadProgressResponse(message = "x").message == "x"
    assert routes.LoadProgressResponse().message is None


@pytest.mark.parametrize("bypass", [{"alongside": True}, {"engine": "vllm"}])
def test_a_pass_through_load_does_not_claim_the_served_split(monkeypatch, bypass):
    _loaded_split(monkeypatch)
    assert cluster_ux.split_summary() == _SPLIT
    # The next load bypasses planning: the served split's plan stays, but this model is not on it.
    run(ss.before_load(_request(model_path = "/models/other.gguf", **bypass), 4))
    assert cs.state()._passthrough
    assert cluster_ux.split_summary() is None


def test_lost_connection_names_the_workers_of_the_split(monkeypatch):
    assert routes._friendly_error(httpx.ReadError("gone")) == routes._LOST_CONNECTION_MSG
    _loaded_split(monkeypatch)
    for exc in (httpx.ReadError("gone"), RuntimeError("Lost connection to llama-server")):
        text = routes._friendly_error(exc)
        assert text.startswith("Lost connection to the model, which is split across this machine")
        assert _WORKERS in text and "`unsloth worker` is still running on each worker" in text
        assert "reload the model" in text
        assert "tls://" not in text and "RPC0" not in text
        assert len(text) <= 200  # the chat error bubble shows two lines


def test_a_non_streaming_chat_names_the_workers_when_the_split_dies(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from models.inference import ChatCompletionRequest
    from state.tool_policy import reset_tool_policy

    def _generate(**kwargs):
        raise RuntimeError("Lost connection to llama-server")
        yield

    async def _no_auto_switch(*_args, **_kwargs):
        return None

    class _Request:
        state = SimpleNamespace(skip_api_monitor = True)
        url = SimpleNamespace(path = "/v1/chat/completions")
        method = "POST"
        scope = {}

        async def is_disconnected(self):
            return False

    reset_tool_policy()
    llama = SimpleNamespace(
        is_loaded = True,
        is_vision = False,
        supports_tools = False,
        supports_reasoning = False,
        reasoning_always_on = False,
        _is_audio = False,
        model_identifier = "unsloth/M-GGUF",
        context_length = 4096,
        generate_chat_completion = _generate,
        _maybe_recover_from_mtp_crash = lambda exc: False,
    )
    monkeypatch.setattr(routes, "_automatic_model_load_may_run", lambda: False)
    monkeypatch.setattr(routes, "_maybe_auto_switch_model", _no_auto_switch)
    monkeypatch.setattr(routes, "get_llama_cpp_backend", lambda: llama)
    payload = ChatCompletionRequest(
        model = "local-model", messages = [{"role": "user", "content": "hi"}]
    )

    def _detail():
        with pytest.raises(HTTPException) as caught:
            asyncio.run(routes.openai_chat_completions(payload, _Request(), "test-user"))
        assert caught.value.status_code == 500
        return caught.value.detail

    assert _detail() == "Could not reach an upstream service. Please try again."
    _loaded_split(monkeypatch)
    detail = _detail()
    assert detail.startswith("Lost connection to the model, which is split across this machine")
    assert _WORKERS in detail
    assert routes._cluster_lost_connection_detail(httpx.ReadError("gone")) == detail
    assert routes._cluster_lost_connection_detail(httpx.ReadTimeout("slow")) is None
    assert routes._cluster_lost_connection_detail(ValueError("bad")) is None


def test_a_model_too_big_for_the_cluster_is_refused_in_plain_words(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 900 * GIB)
    with pytest.raises(ss.SparkLoadDoesNotFit) as caught:
        run(ss.before_load(_request(), 4))
    detail = routes._cluster_refusal_detail(caught.value)
    assert detail.startswith(
        f"This model does not fit, even split across this machine, {_WORKERS}:"
    )
    assert "it needs 900.0 GiB" in detail and "unsloth cluster pair" in detail
    assert "KV" not in detail and "devices" not in detail
    # Any other refusal keeps its own text.
    assert routes._cluster_refusal_detail(ss.SparkLoadDoesNotFit("no peer")) == "no peer"


def test_a_local_load_gets_no_cluster_wording_but_names_a_left_out_worker(monkeypatch):
    _cluster_env(monkeypatch, model_bytes = 10 * GIB)
    run(ss.before_load(_request(), 4))
    assert cluster_ux.load_progress_message(llama_started = True) is None
    assert cluster_ux.split_summary() is None
    assert cluster_ux.lost_connection_message() is None
    assert cluster_ux.dropped_worker_notice() is None
    cs.LINK_DROPPED["studio-2"] = (
        "cannot reach 6bef at 10.0.0.9:50070: [Errno 111] Connect call failed"
    )
    notice = cluster_ux.dropped_worker_notice()
    assert notice == (
        "Paired worker studio-2 was not reachable, so this model is running on this machine only. "
        "Start `unsloth worker` on studio-2 (`unsloth cluster list` shows which workers are "
        "ready), then reload the model to use it."
    )
    assert routes._cluster_dropped_notice() == notice

    failed = routes._with_cluster_notice_detail(
        HTTPException(500, "Failed to load the model."), notice
    )
    assert failed.detail == f"Failed to load the model. {notice}"
    cancelled = routes._with_cluster_notice_detail(
        HTTPException(409, "Model load cancelled"), notice
    )
    assert cancelled.detail == "Model load cancelled"
    response = routes.LoadResponse.model_construct(memory_warning = "Low memory.")
    assert routes._with_load_warning(response, notice).memory_warning == f"Low memory. {notice}"
    assert routes._with_load_warning(response, None) is response
    # A pass-through load reports no dropped worker.
    cs.state()._passthrough = True
    assert cluster_ux.dropped_worker_notice() is None


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_a_mac_master_with_no_route_is_pointed_at_local_network_access(monkeypatch, platform):
    _cluster_env(monkeypatch, model_bytes = 10 * GIB)
    run(ss.before_load(_request(), 4))
    monkeypatch.setattr(cluster_ux.sys, "platform", platform)
    cs.LINK_DROPPED["studio-2"] = "OSError: [Errno 65] No route to host"
    notice = cluster_ux.dropped_worker_notice()
    hint = (
        " On macOS, allow Local Network access for the app running Unsloth in "
        "System Settings > Privacy & Security > Local Network."
    )
    assert notice.endswith(
        "then reload the model to use it." + (hint if platform == "darwin" else "")
    )
    cs.LINK_DROPPED["studio-2"] = "[Errno 61] Connection refused"
    assert "Local Network" not in cluster_ux.dropped_worker_notice()


def test_status_keeps_the_dropped_worker_notice_until_the_next_load():
    # The client re-shows /status's warning right after a load; a missing notice dismisses the toast.
    from core.inference.llama_cpp import LlamaCppBackend
    from routes import inference as routes

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._variant_fallback_warning = backend._last_load_warning = None
    backend.hub_access_warning = backend._last_carveout_advice = None
    notice = (
        "Paired worker studio-2 was not reachable, so this model is running on this machine only."
    )
    backend.cluster_notice = notice
    assert routes._status_load_warning(backend) == notice
    backend._begin_load_warnings()
    assert routes._status_load_warning(backend) is None
