# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inference account boundaries with real route serialization and generation registrations."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import allow_ambient_hf_token, get_current_subject
from core.inference import gpu_arbiter
from hub.services.models import account_access as access
from models.inference import LoadRequest, UnloadRequest
from routes import inference, models, video
from state import active_generations
from utils.account_context import (
    OWNER,
    AccountContext,
    arun_as,
    bind_account,
    reset_account,
    run_as,
)

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(inference, "_managed_catalogs", {})
    monkeypatch.setattr(inference, "_CATALOG_CACHE", {"at": 0.0, "models": []})
    monkeypatch.setattr(inference, "_ADVERTISED_CACHE", {"at": None, "paths": {}})
    monkeypatch.setattr(access, "repo_is_public", lambda *a, **k: False)
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", None)
    monkeypatch.setattr(inference, "_CANCEL_REGISTRY", {})
    monkeypatch.setattr(inference, "_PENDING_CANCELS", {})
    active_generations.reset_for_tests()
    yield
    active_generations.reset_for_tests()


def client_for(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[allow_ambient_hf_token] = lambda: False
    app.include_router(inference.router, prefix = "/api/inference")
    app.include_router(inference.studio_router, prefix = "/api/inference")
    app.include_router(video.router, prefix = "/api/inference")
    app.include_router(models.router, prefix = "/api/models")
    return TestClient(app)


@pytest.mark.parametrize(
    "modality,paths",
    [
        ("chat", ["status", "load-progress"]),
        ("diffusion", ["images/status", "images/load-progress", "images/generate-progress"]),
        ("video", ["video/status", "video/load-progress", "video/generate-progress"]),
    ],
)
def test_resident_identity_and_progress_are_hidden_from_other_accounts(
    monkeypatch, modality, paths
):
    monkeypatch.setattr(gpu_arbiter, "_owner", modality)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", ALICE.account_id)
    with client_for(BOB) as client:
        for path in paths:
            response = client.get(f"/api/inference/{path}")
            assert response.status_code == 200
            assert response.json() == {"loaded": True, "yours": False}
    assert not run_as(ALICE, access.resident_hidden, modality)
    assert not run_as(OWNER, access.resident_hidden, modality)


@pytest.mark.parametrize(
    "path,body",
    [
        ("images/load", {"model_path": "org/public"}),
        ("video/load", {"model_path": "org/public"}),
        ("unload", {"model_path": "org/public", "force_cancel_active": True}),
        ("images/unload", None),
        ("video/unload", None),
    ],
)
def test_foreign_generation_returns_retryable_409_without_cancelling_it(path, body):
    event = threading.Event()
    with run_as(BOB, active_generations.ActiveGeneration, event, model = "org/bob-private"):
        with client_for(ALICE) as client:
            response = client.post(f"/api/inference/{path}", json = body)
        assert response.status_code == 409
        body = response.json()
        assert body["error"] == "gpu_busy"
        assert body["retry_after"] == int(response.headers["retry-after"])
        assert not event.is_set()


def test_chat_load_and_raw_arbiter_error_use_the_same_retry_response(monkeypatch):
    async def fail(*args, **kwargs):
        raise gpu_arbiter.GpuBusyForAnotherAccountError("video", 1)

    monkeypatch.setattr(inference, "load_model_gated", fail)
    with client_for(ALICE) as client:
        response = client.post("/api/inference/load", json = {"model_path": "org/public"})
    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "gpu_busy"
    assert body["retry_after"] == int(response.headers["retry-after"])


def test_forced_swap_cancels_only_callers_registrations():
    mine, theirs = threading.Event(), threading.Event()
    with (
        run_as(ALICE, active_generations.ActiveGeneration, mine),
        run_as(BOB, active_generations.ActiveGeneration, theirs),
    ):
        with pytest.raises(HTTPException) as exc:
            run_as(ALICE, inference._raise_or_cancel_active_generations, force = True, action = "Load")
        assert exc.value.detail["error"] == "gpu_busy"
        assert not theirs.is_set()
    with run_as(ALICE, active_generations.ActiveGeneration, mine):
        assert (
            run_as(ALICE, inference._raise_or_cancel_active_generations, force = True, action = "Load")
            == 1
        )
        assert mine.is_set()


def test_active_generation_and_cancel_id_routes_are_scoped(monkeypatch):
    monkeypatch.setattr(inference, "_openai_llama_admission_capacity", lambda *a: 2)
    mine, theirs = threading.Event(), threading.Event()
    with (
        run_as(
            ALICE,
            inference._TrackedCancel,
            mine,
            "same-id",
            thread_id = "alice-thread",
            model = "alice/private",
        ),
        run_as(
            BOB,
            inference._TrackedCancel,
            theirs,
            "same-id",
            thread_id = "bob-thread",
            model = "bob/private",
        ),
    ):
        with client_for(ALICE) as client:
            response = client.get("/api/inference/active-generations").json()
            assert response["thread_ids"] == ["alice-thread"]
            assert response["count"] == 1
            assert "bob/private" not in str(response)
            assert client.post("/api/inference/cancel", json = {"cancel_id": "same-id"}).json() == {
                "cancelled": 1
            }
        assert mine.is_set() and not theirs.is_set()


def test_pending_cancel_and_recreated_username_do_not_cross_accounts():
    run_as(ALICE, inference._cancel_by_cancel_id_or_stash, "pending")
    event = threading.Event()
    with run_as(BOB, inference._TrackedCancel, event, "pending"):
        assert not event.is_set()
    with run_as(AccountContext("c" * 32, "alice"), inference._TrackedCancel, event, "pending"):
        assert not event.is_set()
    with run_as(ALICE, inference._TrackedCancel, event, "pending"):
        assert event.is_set()


def test_scoped_load_request_ids_use_immutable_accounts():
    request = LoadRequest(model_path = "org/private", load_request_id = "same-request")
    alice = run_as(ALICE, inference._begin_load_attempt, request, "alice")
    replacement = AccountContext("c" * 32, "alice")
    second = run_as(replacement, inference._begin_load_attempt, request, "alice")
    try:
        cancel = UnloadRequest(model_path = "org/private", cancel_load_request_id = "same-request")
        run_as(replacement, inference._cancel_scoped_load_attempt, cancel, "alice")
        assert second.cancel_event.is_set()
        assert not alice.cancel_event.is_set()
    finally:
        inference._finish_load_attempt(alice)
        inference._finish_load_attempt(second)


@pytest.mark.parametrize("path", ["images/generate/cancel", "video/generate/cancel"])
def test_media_cancel_cannot_stop_foreign_resident(monkeypatch, path):
    monkeypatch.setattr(
        gpu_arbiter, "_owner", "diffusion" if path.startswith("images") else "video"
    )
    monkeypatch.setattr(gpu_arbiter, "_owner_account", BOB.account_id)
    with client_for(ALICE) as client:
        assert client.post(f"/api/inference/{path}").json() == {"cancelled": False}


@pytest.mark.parametrize(
    "path",
    [
        "config/org/private",
        "check-vision/org/private",
        "check-embedding/org/private",
        "loras/org/private/base-model",
        "cached-model-path?repo_id=org/private",
        "gguf-variants?repo_id=org/private",
        "download-progress?repo_id=org/private",
        "gguf-download-progress?repo_id=org/private&variant=Q4_K_M",
        "kv-cache-estimate?repo_id=org/private&quant=Q4_K_M",
    ],
)
def test_private_model_object_routes_refuse_unguarded_cache_reads(path):
    with client_for(BOB) as client:
        response = client.get(f"/api/models/{path}")
    assert response.status_code == 404, response.text


def test_preview_load_refuses_private_foreign_target_before_gpu_work():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            arun_as(
                BOB,
                inference.load_model_for_preview(
                    LoadRequest(model_path = "org/private"), SimpleNamespace(scope = {}), "bob"
                ),
            )
        )
    assert exc.value.status_code == 404


def test_same_public_resident_is_available_without_loading_or_cancelling(monkeypatch):
    monkeypatch.setattr(access, "repo_is_public", lambda *a, **k: True)
    # A caller already naming the resident public repo can pass the shared-model
    # fast path even while another account has a registered stream.
    from utils import openai_auto_switch_settings

    monkeypatch.setattr(
        openai_auto_switch_settings, "get_openai_auto_switch_enabled", lambda: False
    )
    monkeypatch.setattr(openai_auto_switch_settings, "idle_unload_is_configured", lambda: False)
    monkeypatch.setattr(inference, "_loaded_satisfies", lambda model: True)
    monkeypatch.setattr(inference, "_claim_slot_for_non_preview", lambda request: None)
    event = threading.Event()
    with run_as(BOB, active_generations.ActiveGeneration, event):
        asyncio.run(
            arun_as(
                ALICE,
                inference._maybe_auto_switch_model(
                    "org/public", SimpleNamespace(scope = {}, state = SimpleNamespace()), "alice"
                ),
            )
        )
    assert not event.is_set()


@pytest.mark.parametrize("path", ["validate", "estimate-memory", "audio/download-plan"])
def test_model_probe_posts_cannot_read_private_cache(path):
    with client_for(BOB) as client:
        response = client.post(f"/api/inference/{path}", json = {"model_path": "org/private"})
    assert response.status_code == 404, response.text


def test_cpu_resident_provenance_does_not_depend_on_a_gpu_lease(monkeypatch):
    monkeypatch.setattr(access, "_resident_accounts", {})
    run_as(ALICE, access.note_resident_account, "chat", "org/private")
    assert run_as(BOB, access.resident_hidden, "chat", "org/private")
    assert not run_as(ALICE, access.resident_hidden, "chat", "org/private")
    assert not run_as(OWNER, access.resident_hidden, "chat", "org/private")
    # A stale entry for a CPU model which has been unloaded never invents a resident.
    assert not run_as(BOB, access.resident_hidden, "chat")


def test_stt_private_status_and_download_state_are_filtered(monkeypatch):
    monkeypatch.setattr(inference, "_stt_download_accounts", {"transformers": ALICE.account_id})
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(inference, "_stt_repo_reference", lambda model, engine: model)
    run_as(ALICE, access.note_resident_account, "stt:transformers", "org/private")
    status = {"loaded_model": "org/private"}
    for engine in ["transformers", "gguf", "mtmd"]:
        status[engine] = {
            "loaded_model": "org/private",
            "downloaded_models": ["org/private"],
            "download": {"model": "org/private", "error": "private error"},
        }
    result = run_as(BOB, inference._account_stt_status, status)
    assert result["loaded_model"] is None
    assert "org/private" not in str(result)
    assert "private error" not in str(result)


def test_stt_downloads_grant_only_the_initiator_and_cancel_is_scoped(monkeypatch):
    monkeypatch.setattr(inference, "_stt_download_accounts", {})
    monkeypatch.setattr(inference, "_stt_grant_pending", {})
    monkeypatch.setattr(inference, "_stt_repo_reference", lambda model, engine: model)
    monkeypatch.setattr(access, "authorize_download", lambda *a: None)
    calls = []
    module = SimpleNamespace(
        start_model_download = lambda *a: calls.append(a),
        download_status = lambda: {"downloading": False},
        is_model_downloaded = lambda model: True,
        cancel_model_download = lambda: calls.append("cancel") or True,
    )
    run_as(
        ALICE,
        inference._start_account_stt_download,
        module,
        "transformers",
        "org/private",
        "alice-token",
    )
    assert inference._stt_grant_pending["transformers"].wait(5)
    assert run_as(ALICE, access.repo_visible, "org/private")
    assert not run_as(BOB, access.repo_visible, "org/private")
    assert run_as(BOB, inference._cancel_account_stt_download, module, "transformers") == {
        "downloading": False,
        "cancelled": False,
    }
    assert "cancel" not in calls
    assert run_as(ALICE, inference._cancel_account_stt_download, module, "transformers")[
        "cancelled"
    ]


def test_stt_download_cannot_adopt_another_accounts_job(monkeypatch):
    monkeypatch.setattr(inference, "_stt_download_accounts", {"transformers": ALICE.account_id})
    module = SimpleNamespace(download_status = lambda: {"downloading": True})
    with pytest.raises(HTTPException) as exc:
        run_as(
            BOB, inference._start_account_stt_download, module, "transformers", "org/private", False
        )
    assert exc.value.status_code == 409


def test_openai_catalog_and_advertised_paths_are_account_scoped(monkeypatch):
    from utils.account_context import current_account_id

    monkeypatch.setattr(inference, "_classified_catalog", lambda rows: rows)
    monkeypatch.setattr(
        models,
        "collect_local_models",
        lambda path: [SimpleNamespace(model_id = "same/name", path = current_account_id())],
    )

    async def read(account):
        return await arun_as(account, inference._cached_local_catalog())

    for account in [ALICE, BOB, ALICE, OWNER]:
        rows = asyncio.run(read(account))
        assert rows[0].path == account.account_id
        assert run_as(account, inference._advertised_local_path, "same/name") == account.account_id


def test_private_media_index_rows_cannot_bypass_filtered_openai_catalog(monkeypatch):
    picks = {
        "text-to-image": [
            ("org/private", SimpleNamespace(model_path = "org/private", gguf_filename = None), True)
        ]
    }
    monkeypatch.setattr(inference, "_validated_media_picks", lambda at: picks)
    monkeypatch.setattr(inference, "_resident_media_status", lambda task: None)
    run_as(ALICE, access.record_model_grant, "org/private")
    assert run_as(BOB, inference._media_model_objects, [], 1, 1) == []
    assert run_as(ALICE, inference._media_model_objects, [], 1, 1)[0]["id"] == "org/private"


def test_custom_stt_models_are_filtered_before_openai_listing(monkeypatch):
    from core.inference import stt_mtmd_sidecar, stt_sidecar

    monkeypatch.setattr(stt_sidecar, "is_available", lambda: True)
    monkeypatch.setattr(stt_sidecar, "is_model_downloaded", lambda model: False)
    monkeypatch.setattr(
        stt_sidecar, "get_stt_sidecar", lambda: SimpleNamespace(loaded_model = "org/private")
    )
    monkeypatch.setattr(stt_mtmd_sidecar, "is_available", lambda: False)
    monkeypatch.setattr(inference, "_downloaded_custom_stt_ids", lambda at: ("org/private",))
    run_as(ALICE, access.record_model_grant, "org/private")
    run_as(ALICE, access.note_resident_account, "stt:transformers", "org/private")
    assert run_as(BOB, inference._stt_model_objects, 1, 1) == []
    assert run_as(ALICE, inference._stt_model_objects, 1, 1)[0]["loaded"]


@pytest.mark.parametrize("kind", ["images", "video"])
def test_private_cpu_media_residents_hide_progress_and_refuse_generation_and_unload(
    monkeypatch, kind
):
    from core.inference import diffusion_engine_router, media_auto_switch
    from core.inference import video as video_engine

    backend = SimpleNamespace(status = lambda: {"loaded": True, "repo_id": "org/private"})
    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: backend)
    monkeypatch.setattr(video_engine, "get_video_backend", lambda: backend)

    async def no_switch(*a, **k):
        return None

    monkeypatch.setattr(media_auto_switch, "maybe_auto_switch_media_model", no_switch)
    run_as(
        ALICE,
        access.note_resident_account,
        "diffusion" if kind == "images" else "video",
        "org/private",
    )
    with client_for(BOB) as client:
        for path in ["load-progress", "generate-progress"]:
            response = client.get(f"/api/inference/{kind}/{path}")
            assert response.json() == {"loaded": True, "yours": False}
        response = client.post(f"/api/inference/{kind}/generate", json = {"prompt": "hello"})
        assert response.status_code == 404, response.text
        assert client.post(f"/api/inference/{kind}/unload").status_code == 404


@pytest.mark.parametrize("path", ["images/generate/cancel", "video/generate/cancel"])
def test_model_loader_cannot_cancel_a_different_accounts_generation(path):
    event = threading.Event()
    with run_as(BOB, active_generations.ActiveGeneration, event):
        with client_for(OWNER) as client:
            assert client.post(f"/api/inference/{path}").json() == {"cancelled": False}
        assert not event.is_set()


def test_authorized_private_download_progress_is_visible_before_grant_completion(monkeypatch):
    from hub.services import download_lifecycle, snapshot_progress
    from hub.services.models import downloads

    registry = SimpleNamespace(active_job_refs = lambda repo: [SimpleNamespace(key = "org/private::")])
    monkeypatch.setattr(downloads, "_registry", registry)
    monkeypatch.setattr(
        download_lifecycle, "_job_accounts", {(id(registry), "org/private::"): ALICE.account_id}
    )

    async def progress(**kwargs):
        return {"downloaded_bytes": 12}

    monkeypatch.setattr(snapshot_progress, "snapshot_progress_response", progress)
    with client_for(ALICE) as client:
        assert client.get("/api/models/download-progress?repo_id=org/private").json() == {
            "downloaded_bytes": 12
        }
    with client_for(BOB) as client:
        assert client.get("/api/models/download-progress?repo_id=org/private").status_code == 404


@pytest.mark.parametrize("kind", ["loras", "controlnets"])
def test_media_adapter_catalogs_and_selection_do_not_reveal_owner_files(
    monkeypatch, tmp_path, kind
):
    from core.inference import diffusion_lora, diffusion_controlnet

    module = diffusion_lora if kind == "loras" else diffusion_controlnet
    entry = SimpleNamespace(
        id = "private-adapter", local_path = str(tmp_path / "private-weights"), repo_id = None
    )
    monkeypatch.setattr(module, "list_" + kind, lambda **kwargs: [entry])
    with client_for(BOB) as client:
        response = client.get("/api/models/diffusion-" + kind)
        assert response.status_code == 200, response.text
        assert response.json()[kind] == []
        assert BOB.account_id in response.json()[kind + "_dir"]
    request = SimpleNamespace(
        loras = [SimpleNamespace(id = entry.id)] if kind == "loras" else None,
        controlnet = SimpleNamespace(id = entry.id) if kind == "controlnets" else None,
    )
    with pytest.raises(HTTPException) as exc:
        run_as(BOB, access.require_media_adapters, request)
    assert exc.value.status_code == 404
