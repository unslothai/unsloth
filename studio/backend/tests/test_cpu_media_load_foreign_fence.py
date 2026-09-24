# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A CPU-only media load must re-check for foreign generations before it touches the backend.

The GPU path re-asks ``require_no_foreign_generations`` right before engine activation, and the
arbiter re-asks it again under its lock. The CPU path takes neither, so a generation another
account starts while this load is still validating (a multi-second window of Hub / header reads)
is cancelled or replaced by ``begin_load``, which signals the active generation with no regard
for who owns it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from auth import policy
from core.inference import gpu_arbiter as arb
from hub.services.models import account_access
from state import active_generations as generations
from utils.account_context import AccountContext, arun_as, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture(autouse = True)
def isolated(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setattr(arb, "_owner_account", None)
    monkeypatch.setattr(arb, "_owner_epoch", 0)
    monkeypatch.setattr(account_access, "_generation_accounts", {})
    monkeypatch.setattr(account_access, "_generation_holders", {})
    generations.reset_for_tests()
    yield
    generations.reset_for_tests()


@pytest.fixture
def foreign_generation():
    """Start ALICE's image/video generation on demand, mid-load."""
    state = {}

    def start(modality):
        token = bind_account(ALICE)
        try:
            ctx = account_access.media_generation_slot(modality)
            ctx.__enter__()
        finally:
            reset_account(token)
        state["ctx"] = ctx

    yield start
    ctx = state.pop("ctx", None)
    if ctx is not None:
        ctx.__exit__(None, None, None)


@pytest.mark.parametrize("modality", ["image", "video"])
def test_cpu_media_load_refuses_a_generation_started_mid_load(
    monkeypatch, foreign_generation, modality
):
    from core.inference import (
        diffusion,
        diffusion_compat,
        diffusion_device,
        diffusion_engine_router,
        video,
    )
    from models.inference import DiffusionLoadRequest, VideoLoadRequest
    from routes import inference as route
    from routes import video as video_route

    touched = []
    backend = SimpleNamespace(
        validate_load_request = lambda *_a, **_k: SimpleNamespace(name = "ltx-2", base_repo = None),
        preflight_base_access = lambda *_a, **_k: None,
        assert_precision_available = lambda *_a, **_k: None,
        begin_load = lambda *_a, **_k: touched.append("load") or {},
    )

    async def ordinal(*_args):
        return None

    def start_foreign(*_a, **_k):
        # The window: another account's generation registers while this load is still validating.
        foreign_generation("diffusion" if modality == "image" else "video")

    monkeypatch.setattr(diffusion, "get_diffusion_backend", lambda: backend)
    monkeypatch.setattr(diffusion, "resolve_local_single_file", lambda *_a, **_k: None)
    monkeypatch.setattr(video, "get_video_backend", lambda: backend)
    monkeypatch.setattr(video, "assert_video_precision_available", lambda *_a, **_k: None)
    monkeypatch.setattr(
        diffusion_device, "resolve_diffusion_device_target", lambda: SimpleNamespace(device = "cpu")
    )
    monkeypatch.setattr(diffusion_engine_router, "predict_engine", lambda *_a, **_k: "sd_cpp")
    monkeypatch.setattr(diffusion_engine_router, "engine_for", lambda *_a: backend)
    monkeypatch.setattr(diffusion_engine_router, "active_engine_name", lambda: "sd_cpp")
    monkeypatch.setattr(
        diffusion_engine_router,
        "select_and_activate_engine",
        lambda *_a, **_k: touched.append("activate") or backend,
    )
    monkeypatch.setattr(diffusion_engine_router, "begin_load_on", lambda _e, start: start())
    monkeypatch.setattr(diffusion_compat, "assert_pick_is_not_speech", start_foreign)
    monkeypatch.setattr(route, "_guard_diffusion_load_against_training", lambda: None)
    monkeypatch.setattr(route, "_selected_gpu_ordinal", ordinal)
    monkeypatch.setattr(route, "_assert_native_precision_unset", lambda **_k: None)
    monkeypatch.setattr(video_route, "_guard_video_load_against_training", lambda: None)
    monkeypatch.setattr(video_route, "_selected_gpu_ordinal", ordinal)
    monkeypatch.setattr(account_access, "require_media_references", lambda *_a, **_k: None)
    monkeypatch.setattr(account_access, "require_model_access", lambda *_a, **_k: None)
    monkeypatch.setattr(account_access, "account_hf_token", lambda token = None: token)
    monkeypatch.setattr(account_access, "note_resident_account", lambda *_a, **_k: None)
    monkeypatch.setattr(account_access, "note_resident_components", lambda *_a, **_k: None)

    if modality == "image":
        request = DiffusionLoadRequest(model_path = "org/image", gguf_filename = "model.gguf")
        load = route.load_diffusion_model_gated
    else:
        request = VideoLoadRequest(model_path = "org/video", gguf_filename = "model.gguf")
        load = video_route.load_video_model_gated

    with pytest.raises(HTTPException) as refused:
        asyncio.run(arun_as(BOB, load(request, BOB.username)))
    assert refused.value.status_code == 409
    assert refused.value.detail["error"] == "gpu_busy"
    # Nothing was activated and nothing was loaded: ALICE's generation still owns the backend.
    assert touched == []
