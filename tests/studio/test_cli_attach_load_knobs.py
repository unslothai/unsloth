# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


from __future__ import annotations

import pytest

import unsloth_cli.commands.start as start_cli


BASE = "http://127.0.0.1:8888"
KEY = "k"
RESIDENT = {"id": "unsloth/Qwen3-8B", "loaded": True}


@pytest.fixture
def loads(monkeypatch):
    calls = []

    def _load(base, key, requested, load, payload):
        calls.append(payload)
        return {"status": "already_loaded", "model": requested}

    monkeypatch.setattr(start_cli, "_loaded_models", lambda base, key: [dict(RESIDENT)])
    monkeypatch.setattr(start_cli, "_load_model_with_progress", _load)
    monkeypatch.setattr(start_cli, "_http_json", lambda *a, **k: {})
    return calls


def test_context_length_without_model_reloads_the_resident_model(loads):
    entry = start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(max_seq_length = 32768),
    )
    assert loads == [{"model_path": RESIDENT["id"], "max_seq_length": 32768}]
    assert entry["id"] == RESIDENT["id"]


def test_gguf_variant_without_model_is_forwarded(loads):
    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(gguf_variant = "UD-Q8_K_XL", max_seq_length = 32768),
    )
    assert loads == [
        {
            "model_path": RESIDENT["id"],
            "gguf_variant": "UD-Q8_K_XL",
            "max_seq_length": 32768,
        }
    ]


def test_bare_attach_does_not_load(loads):
    entry = start_cli._resolve_model(BASE, KEY, None, start_cli.LoadOptions())
    assert loads == []
    assert entry["id"] == RESIDENT["id"]


def test_attach_knobs_skip_the_preload_check(loads):
    checked = []
    start_cli._resolve_model(
        BASE,
        KEY,
        None,
        start_cli.LoadOptions(max_seq_length = 4096),
        preload_check = lambda *a: checked.append(a),
    )
    assert checked == []
    assert len(loads) == 1
