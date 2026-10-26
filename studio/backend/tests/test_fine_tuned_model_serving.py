# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException

import routes.inference as inference_route
from core.inference import llama_keepwarm
from utils.paths import outputs_root

BASE = "unsloth/SmolLM2-135M-Instruct"
RUN = "unsloth_SmolLM2-135M-Instruct_1792969744"


class _Url:
    path = "/v1/chat/completions"


class _Req:
    url = _Url()
    headers = {}


def _run_dir():
    run_dir = outputs_root() / RUN
    run_dir.mkdir(parents = True)
    (run_dir / "adapter_config.json").write_text("{}")
    return run_dir


def _reject(model, active, monkeypatch):
    idle = type("B", (), {"is_loaded": False, "model_identifier": None, "hf_variant": None})()
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: idle)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": active})(),
    )
    monkeypatch.setattr("core.inference.local_model_resolver.index_is_built", lambda: True)
    monkeypatch.setattr("core.inference.local_model_resolver.warm_index_soon", lambda: None)
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_local_gguf", lambda _n, **_kw: None
    )

    async def _unavailable(model):
        return f"The model '{model}' is not downloaded on this server."

    monkeypatch.setattr(inference_route, "_unavailable_model_message", _unavailable)
    return asyncio.run(inference_route._reject_unservable_model(model, _Req()))


def test_a_fine_tuned_model_is_not_answered_by_the_resident_base_model(monkeypatch):
    _run_dir()
    with pytest.raises(HTTPException) as excinfo:
        _reject(RUN, BASE, monkeypatch)
    assert excinfo.value.status_code == 404
    assert RUN in excinfo.value.detail["error"]["message"]


def test_a_loaded_fine_tuned_model_answers_under_its_name(monkeypatch):
    run_dir = _run_dir()
    assert _reject(RUN, str(run_dir), monkeypatch) is None


def test_an_unknown_bare_name_still_uses_the_resident_model(monkeypatch):
    _run_dir()
    assert _reject("some-label", BASE, monkeypatch) is None


def test_loading_a_fine_tuned_model_by_name_loads_its_run_folder(monkeypatch):
    run_dir = _run_dir()

    @asynccontextmanager
    async def _gate():
        yield

    loaded = []

    async def _impl(request, *_args, **_kwargs):
        loaded.append(request.model_path)
        return {"status": "loaded", "model": request.model_path}

    monkeypatch.setattr(inference_route, "_raise_if_sidecar_swap_in_progress", lambda: None)
    monkeypatch.setattr(inference_route, "_pending_load_attempts", {})
    monkeypatch.setattr(llama_keepwarm, "inference_lifecycle_gate", _gate)
    monkeypatch.setattr(inference_route, "_run_tracked_load_model_impl", _impl)

    answers = [
        asyncio.run(
            inference_route.load_model(
                inference_route.LoadRequest(model_path = model_path), object(), "test"
            )
        )
        for model_path in (RUN, BASE)
    ]

    assert loaded == [str(run_dir), BASE]
    assert [answer["model"] for answer in answers] == [RUN, BASE]
