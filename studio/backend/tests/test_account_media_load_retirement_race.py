# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An image load caught by retirement must not run for the deleted account.

Retirement tombstones the account and scans for its media work. A /images/load still inside a
pre-admission await owns nothing the scan can see, so it used to start its load afterwards; and
a load admitted just before the tombstone kept running. Now admission and the scan share one
lock: the late request is refused, the early load is torn down."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from auth import policy
from core.inference import diffusion, diffusion_device, diffusion_engine_router, video
from core.inference import diffusion_compat
from core.training import account_jobs as jobs
from hub.services.models import account_access as access
from models.inference import DiffusionLoadRequest
from routes import accounts, inference
from utils.account_context import AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")


class FakeEngine:
    def __init__(self):
        self.started = []
        self.unloaded = 0
        self.loading = False

    def begin_load(self, repo_id, **kwargs):
        self.started.append(repo_id)
        self.loading = True
        return {"loaded": False, "repo_id": repo_id}

    def loading_repo_ids(self):
        return ("org/model",) if self.loading else ()

    def unload(self):
        self.unloaded += 1
        self.loading = False
        return {"loaded": False}

    def preflight_base_access(self, *args, **kwargs):
        return None


@pytest.fixture
def engine(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)
    monkeypatch.setattr(jobs, "_retired", set())
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(access, "_resident_components", {})
    monkeypatch.setattr(access, "_uncommitted_resident", {})
    monkeypatch.setattr(access, "_uncommitted_components", {})
    fake = FakeEngine()
    backend = SimpleNamespace(
        validate_load_request = lambda *a, **k: None,
        assert_precision_available = lambda *a, **k: None,
    )
    monkeypatch.setattr(diffusion, "get_diffusion_backend", lambda: backend)
    monkeypatch.setattr(diffusion, "resolve_model_kind", lambda *a, **k: "pipeline")
    monkeypatch.setattr(diffusion, "resolve_local_single_file", lambda path: None)
    monkeypatch.setattr(diffusion, "_diffusion_backend", fake)
    monkeypatch.setattr(
        diffusion_device,
        "resolve_diffusion_device_target",
        lambda: SimpleNamespace(device = "cpu"),
    )
    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: fake)
    monkeypatch.setattr(diffusion_engine_router, "select_and_activate_engine", lambda *a, **k: fake)
    monkeypatch.setattr(diffusion_engine_router, "predict_engine", lambda *a, **k: None)
    monkeypatch.setattr(diffusion_engine_router, "engine_for", lambda name: fake)
    monkeypatch.setattr(diffusion_engine_router, "annotate_status", lambda status: status)
    monkeypatch.setattr(diffusion_compat, "assert_pick_is_not_speech", lambda *a, **k: None)
    monkeypatch.setattr(inference, "_guard_diffusion_load_against_training", lambda: None)

    async def no_ordinal(gpu_ids):
        return None

    monkeypatch.setattr(inference, "_selected_gpu_ordinal", no_ordinal)
    monkeypatch.setattr(inference, "reset_media_load_progress", lambda kind: None)
    monkeypatch.setattr(access, "require_media_references", lambda request: None)
    monkeypatch.setattr(access, "require_idle_other_accounts", lambda *a, **k: None)
    monkeypatch.setattr(video, "_backend", None)
    return fake


def _retire():
    jobs._retired.add(ALICE.account_id)
    diffusion_engine_router.retire_load_for_account(ALICE.account_id)
    video.retire_load_for_account(ALICE.account_id)


def _load():
    request = DiffusionLoadRequest(model_path = "org/model")
    return asyncio.run(inference.load_diffusion_model_gated(request, "alice", user_initiated = True))


def test_load_parked_before_admission_is_refused_after_retirement(monkeypatch, engine):
    blocked, release, result = threading.Event(), threading.Event(), {}

    def access_check(model_name):
        blocked.set()
        release.wait(timeout = 30)

    monkeypatch.setattr(access, "require_model_access", access_check)

    def request_thread():
        try:
            result["value"] = run_as(ALICE, _load)
        except HTTPException as exc:
            result["error"] = (exc.status_code, exc.detail)

    thread = threading.Thread(target = request_thread)
    thread.start()
    assert blocked.wait(timeout = 30)
    _retire()
    release.set()
    thread.join(timeout = 30)
    assert not thread.is_alive()
    print(f"late load: {result}, started={engine.started}")
    assert engine.started == []
    assert result["error"] == (403, "Account is retired")


def test_load_admitted_before_retirement_is_torn_down(monkeypatch, engine):
    monkeypatch.setattr(access, "require_model_access", lambda model_name: None)
    status = run_as(ALICE, _load)
    assert status.loaded is False and engine.loading
    _retire()
    print(f"early load: unloaded={engine.unloaded}, loading={engine.loading}")
    assert engine.unloaded == 1 and not engine.loading


def test_retirement_leaves_another_accounts_load_alone(monkeypatch, engine):
    monkeypatch.setattr(access, "require_model_access", lambda model_name: None)
    run_as(AccountContext("b" * 32, "bob"), _load)
    _retire()
    assert engine.unloaded == 0 and engine.loading
