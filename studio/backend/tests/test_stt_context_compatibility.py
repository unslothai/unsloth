# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The public transcription API remains usable with supported pre-counting runtimes."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from routes import inference as ri
from .test_stt_mtmd_context import MODEL, runtime  # noqa: F401 - shared protocol fixture


@pytest.mark.parametrize("preloaded_manual", [False, True])
def test_public_transcription_without_native_count_keeps_legacy_capacity(
    runtime, monkeypatch, preloaded_manual
):
    # The supported upstream b9415 macOS fallback can transcribe Qwen3-ASR and has /props,
    # but predates /v1/chat/completions/input_tokens. Public clients cannot send a UI setting.
    runtime.wire.count_status = 404
    if preloaded_manual:
        runtime.sidecar.load(MODEL, context_length = 4096)
    monkeypatch.setattr(ri.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(ri, "_stt_engine_for_model", lambda model: "mtmd")
    monkeypatch.setattr(ri, "_resolve_serving_stt_engine", lambda engine: "mtmd")
    monkeypatch.setattr(ri, "_prepare_runtime_fallback_checkpoint", lambda *args: None)
    monkeypatch.setattr(ri, "_stt_sidecar_for", lambda engine: runtime.sidecar)

    def load(
        model,
        engine,
        cancel_event = None,
        **kwargs,
    ):
        runtime.sidecar.load(model, request_cancel_event = cancel_event, **kwargs)

    monkeypatch.setattr(ri, "_stt_lifecycle", lambda: (load, lambda *args: []))
    app = FastAPI()
    app.include_router(ri.router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            data = {"model": MODEL},
            files = {"file": ("clip.wav", b"encoded", "audio/wav")},
        )
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "Complete sentence."
    assert runtime.sidecar.context_length == 65536
    assert runtime.sidecar.context_length_preference == 0
    assert runtime.wire.generation_requests()
