# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

import unsloth_cli.commands.start as start


@pytest.mark.parametrize(("budget", "message"), [(256, "Budget exhausted"), (0, ""), (-1, "")])
def test_inferred_context_reload_keeps_requested_reasoning_settings(monkeypatch, budget, message):
    model = "owner/model-GGUF"
    status = {
        "active_model": model,
        "model_identifier": model,
        "is_gguf": True,
        "gguf_variant": "Q8_0",
        "requested_context_length": 4096,
        "requested_reasoning_budget": budget,
        "requested_reasoning_budget_message": message,
        "requested_llama_extra_args": [],
    }
    payloads = []
    monkeypatch.setattr(start, "_loaded_models", lambda *a: [{"id": model, "loaded": True}])
    monkeypatch.setattr(start, "_model_status", lambda *a: status)
    monkeypatch.setattr(
        start,
        "_load_model_with_progress",
        lambda base, key, requested, load, payload: payloads.append(payload)
        or {"status": "loaded", "model": model},
    )
    start._resolve_model(
        "http://127.0.0.1:8888", "key", None, start.LoadOptions(max_seq_length = 8192)
    )
    assert payloads[0]["reasoning_budget"] == budget
    assert payloads[0]["reasoning_budget_message"] == message
    assert payloads[0]["max_seq_length"] == 8192
    assert payloads[0]["llama_extra_args"] == []


def test_preserved_reasoning_settings_never_replace_explicit_values():
    status = {"requested_reasoning_budget": 256, "requested_reasoning_budget_message": "old"}
    payload = {"reasoning_budget": 0, "reasoning_budget_message": ""}
    assert start._resident_runtime_payload(status, payload) == {}
    assert (
        start._resident_runtime_payload(
            {
                "requested_reasoning_budget": None,
                "requested_reasoning_budget_message": None,
            },
            {},
        )
        == {}
    )
