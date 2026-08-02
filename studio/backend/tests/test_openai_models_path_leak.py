# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GET /v1/models must report a clean public id, never the on-disk .gguf path."""

import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import routes.inference as inf  # noqa: E402


class _FakeLlama:
    is_loaded = True
    model_identifier = "/srv/models/Qwen3-30B-A3B-Q4_K_M.gguf"
    context_length = 4096
    max_context_length = None
    native_context_length = None


class _FakeUnsloth:
    active_model_name = None
    models: dict = {}
    context_length = None
    max_seq_length = None


def test_openai_models_returns_clean_id_without_path(monkeypatch):
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _FakeLlama())
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _FakeUnsloth())

    objs = inf._openai_model_objects()

    assert len(objs) == 1
    assert objs[0]["id"] == "Qwen3-30B-A3B-Q4_K_M"
    # The serialized payload must not leak the absolute path or the .gguf suffix.
    blob = json.dumps(objs)
    assert "/srv/models" not in blob
    assert ".gguf" not in blob
    # Context fields still flow through.
    assert objs[0]["context_length"] == 4096
