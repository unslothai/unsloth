# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A named resident reload changes only the load settings the caller supplied."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

import unsloth_cli.commands.start as start

MODEL = "owner/model-GGUF"
BASE = "http://127.0.0.1:8888"


@pytest.mark.parametrize("kept", [False, True])
@pytest.mark.parametrize("manual", [False, True])
def test_named_resident_context_change_keeps_other_runtime_settings(monkeypatch, kept, manual):
    models = [{"id": MODEL, "loaded": True}]
    if kept:
        models.insert(0, {"id": "owner/primary-GGUF", "loaded": True})
    status = {
        "active_model": MODEL,
        "is_gguf": True,
        "gguf_variant": "Q8_0",
        "requested_context_length": 4096,
        "requested_n_batch": 128,
        "requested_n_ubatch": 64,
        "cache_type_kv": "q8_0",
        "gpu_memory_mode": "manual",
        "gpu_layers": 7,
    }
    payloads = []
    monkeypatch.setattr(start, "_loaded_models", lambda *a: models)

    def model_status(
        base,
        key,
        model_id = None,
    ):
        assert model_id == MODEL, "must read the requested slot, not the primary"
        return status

    monkeypatch.setattr(start, "_model_status", model_status)
    monkeypatch.setattr(
        start,
        "_load_model_with_progress",
        lambda base, key, model, load, payload: payloads.append(payload)
        or {"status": "loaded", "model": model},
    )
    options = start.LoadOptions(max_seq_length = 8192, gpu_memory_mode = "manual" if manual else None)
    start._resolve_model(BASE, "key", MODEL, options)
    assert payloads == [
        {
            "model_path": MODEL,
            "gguf_variant": "Q8_0",
            "max_seq_length": 8192,
            "n_batch": 128,
            "n_ubatch": 64,
            "cache_type_kv": "q8_0",
            "gpu_memory_mode": "manual",
            "gpu_layers": 7,
        }
    ]


@pytest.mark.parametrize("requested", ["owner/new-GGUF", "/new/model.gguf"])
def test_new_model_never_inherits_previous_resident_settings(monkeypatch, requested):
    models = [{"id": MODEL, "loaded": True}]
    monkeypatch.setattr(start, "_loaded_models", lambda *a: models)
    monkeypatch.setattr(start, "_model_status", lambda *a: pytest.fail("unrelated resident status"))
    payloads = []
    monkeypatch.setattr(
        start,
        "_load_model_with_progress",
        lambda base, key, model, load, payload: payloads.append(payload)
        or {"status": "loaded", "model": MODEL},
    )
    start._resolve_model(BASE, "key", requested, start.LoadOptions(max_seq_length = 8192))
    assert payloads == [{"model_path": requested, "max_seq_length": 8192}]


def test_named_resident_explicit_automatic_context_wins(monkeypatch):
    monkeypatch.setattr(start, "_loaded_models", lambda *a: [{"id": MODEL, "loaded": True}])
    monkeypatch.setattr(
        start,
        "_model_status",
        lambda *a: {
            "requested_context_length": 4096,
            "requested_n_batch": 128,
        },
    )
    payloads = []
    monkeypatch.setattr(
        start,
        "_load_model_with_progress",
        lambda base, key, model, load, payload: payloads.append(payload)
        or {"status": "loaded", "model": model},
    )
    start._resolve_model(
        BASE, "key", MODEL, start.LoadOptions(supplied = frozenset({"max_seq_length"}))
    )
    assert payloads == [{"model_path": MODEL, "max_seq_length": 0, "n_batch": 128}]
