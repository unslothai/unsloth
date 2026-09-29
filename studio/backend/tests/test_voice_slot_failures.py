# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A failed voice initialization must not retain a half-started server."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

import routes.inference as routes
import core.inference.llama_cpp as llama
from utils.models import ModelConfig


@pytest.mark.parametrize(
    "failure,status", [("exception", 500), ("false", 500), ("wrong_type", 400)]
)
def test_failed_voice_initialization_unloads_process(monkeypatch, failure, status):
    backend = SimpleNamespace(
        is_loaded = False,
        model_identifier = None,
        _is_audio = False,
        _audio_type = None,
        unload_model = Mock(),
    )

    def load(intent):
        if failure == "exception":
            raise RuntimeError("codec initialization failed")
        return failure == "wrong_type"

    backend.load_model = load
    config = SimpleNamespace(
        is_gguf = True,
        identifier = "org/voice",
        gguf_variant = "Q4_K_M",
        gguf_hf_repo = "org/voice",
        gguf_file = None,
        base_model = None,
    )

    async def placement(*args):
        return None

    monkeypatch.setattr(routes, "get_voice_llama_backend", lambda: backend)
    monkeypatch.setattr(routes.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(ModelConfig, "from_identifier", lambda **kwargs: config)
    monkeypatch.setattr(llama, "_hf_offline_if_unreachable_for", lambda *a: nullcontext())
    monkeypatch.setattr(routes, "_prepare_load_placement", placement)
    monkeypatch.setattr(routes, "_offline_guarded", lambda *a, **kw: None)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            routes._voice_load_model_impl(routes._VoiceLoadRequest(model_path = "org/voice"), "owner")
        )
    assert error.value.status_code == status
    backend.unload_model.assert_called_once_with()
