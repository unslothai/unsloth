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


def _wire(active, monkeypatch):
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


def _reject(model, active, monkeypatch):
    _wire(active, monkeypatch)
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
        loaded.append((request.model_path, request._advertised_id))
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

    assert loaded == [(str(run_dir), RUN), (BASE, None)]
    assert [answer["model"] for answer in answers] == [RUN, BASE]


CHECKPOINT = f"{RUN}/checkpoint-5"


def test_a_checkpoint_name_is_refused_until_that_checkpoint_is_loaded(monkeypatch):
    checkpoint_dir = _run_dir() / "checkpoint-5"
    checkpoint_dir.mkdir()
    with pytest.raises(HTTPException) as excinfo:
        _reject(CHECKPOINT, BASE, monkeypatch)
    assert excinfo.value.status_code == 404
    assert CHECKPOINT in excinfo.value.detail["error"]["message"]
    assert _reject(CHECKPOINT, str(checkpoint_dir), monkeypatch) is None


@pytest.mark.parametrize(
    "name",
    ["..", ".", "", f"{RUN}/..", f"{RUN}/other", f"../{RUN}", f"{RUN}\\checkpoint-5", "a/b/c"],
)
def test_only_run_and_checkpoint_names_map_into_outputs(name):
    run_dir = _run_dir()
    (run_dir / "other").mkdir()
    assert inference_route._trained_run_dir(name) is None
    assert inference_route._trained_run_dir(RUN) == str(run_dir)


async def _never_held_back(_request):
    return False


def test_switch_by_request_never_fetches_a_checkpoint_name_from_the_hub(monkeypatch):
    from utils import openai_auto_switch_settings as settings
    from core.inference import local_model_resolver as resolver

    checkpoint_dir = _run_dir() / "checkpoint-5"
    checkpoint_dir.mkdir()
    downloads = []

    async def _download(model, *_args, **_kwargs):
        downloads.append(model)

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(resolver, "resolve_local_gguf_for_switch", lambda *_a, **_kw: None)
    monkeypatch.setattr(resolver, "resolve_trusted_cached_local_gguf", lambda *_a, **_kw: None)
    monkeypatch.setattr(inference_route, "_audio_cpp_switch_target", lambda _m: None)
    monkeypatch.setattr(inference_route, "_maybe_auto_download_model", _download)
    monkeypatch.setattr(inference_route, "_keyless_caller_held_back", _never_held_back)

    def _switch(active):
        _wire(active, monkeypatch)
        return asyncio.run(
            inference_route._maybe_auto_switch_model(
                CHECKPOINT, _Req(), "test", claim_resident = False
            )
        )

    with pytest.raises(HTTPException) as excinfo:
        _switch(BASE)
    assert excinfo.value.status_code == 404
    assert _switch(str(checkpoint_dir)) is None
    assert downloads == []


def test_a_managed_account_may_name_its_own_run(monkeypatch):
    run_dir = _run_dir()
    checked = []

    def _require(model):
        # an allowlist such as unsloth/* refuses a bare run name as a Hub id
        checked.append(model)
        if model != str(run_dir):
            raise HTTPException(status_code = 403, detail = "Model not allowed")

    monkeypatch.setattr(inference_route.account_access, "require_model_access", _require)

    asyncio.run(inference_route._require_named_model_access(RUN))
    assert checked == [str(run_dir)]
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route._require_named_model_access("someone-elses-run"))
    assert excinfo.value.status_code == 403


def test_unloading_by_checkpoint_name_targets_the_folder_it_loads(monkeypatch):
    from models.inference import UnloadRequest, UnloadResponse

    checkpoint_dir = _run_dir() / "checkpoint-5"
    checkpoint_dir.mkdir()
    seen = []

    async def _impl(request, _subject):
        seen.append((request.model_path, request.cancel_load_request_id))
        return UnloadResponse(status = "unloaded", model = request.model_path)

    monkeypatch.setattr(inference_route, "_unload_model_impl", _impl)
    answers = [
        asyncio.run(
            inference_route.unload_model(
                UnloadRequest(model_path = CHECKPOINT, cancel_load_request_id = request_id), "test"
            )
        )
        for request_id in (None, "attempt-1")
    ]

    assert seen == [(str(checkpoint_dir), None), (str(checkpoint_dir), "attempt-1")]
    assert [answer["model"] for answer in answers] == [CHECKPOINT, CHECKPOINT]


def test_a_checkpoint_loaded_by_path_answers_to_its_job_name(monkeypatch):
    # what the extra-slot router asks each slot, so a checkpoint kept alongside is found too
    checkpoint_dir = _run_dir() / "checkpoint-5"
    checkpoint_dir.mkdir()
    _wire(str(checkpoint_dir), monkeypatch)
    assert inference_route._loaded_satisfies(CHECKPOINT)
    assert not inference_route._loaded_satisfies(f"{RUN}/checkpoint-6")
    _wire(str(outputs_root() / RUN), monkeypatch)
    assert not inference_route._loaded_satisfies(CHECKPOINT)


def test_a_checkpoint_loaded_by_a_forward_slash_drive_path_is_recognized(monkeypatch):
    windows_dir = "C:/Studio/outputs/run/checkpoint-5"
    monkeypatch.setattr(inference_route, "_trained_run_dir", lambda _m: windows_dir)
    _wire(windows_dir, monkeypatch)
    assert inference_route._loaded_satisfies(CHECKPOINT)
