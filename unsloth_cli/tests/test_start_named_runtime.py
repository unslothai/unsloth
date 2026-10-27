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
            "active_model": MODEL,
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


def _named_reload_payload(monkeypatch, status, options):
    status = {"active_model": MODEL, **status}
    monkeypatch.setattr(start, "_loaded_models", lambda *a: [{"id": MODEL, "loaded": True}])
    monkeypatch.setattr(start, "_model_status", lambda *a: status)
    payloads = []
    monkeypatch.setattr(
        start,
        "_load_model_with_progress",
        lambda base, key, model, load, payload: payloads.append(payload)
        or {"status": "loaded", "model": model},
    )
    start._resolve_model(BASE, "key", MODEL, options)
    return payloads[0]


@pytest.mark.parametrize(
    "options, sent",
    [
        (start.LoadOptions(max_seq_length = 32768), {"max_seq_length": 32768}),
        (
            start.LoadOptions(gpu_memory_mode = "manual"),
            {"gpu_memory_mode": "manual", "gpu_layers": -1},
        ),
    ],
)
def test_named_resident_with_extras_left_to_server_inheritance(monkeypatch, options, sent):
    status = {
        "is_gguf": True,
        "gguf_variant": "Q8_0",
        "requested_context_length": 0,
        "requested_llama_extra_args": ["--ctx-size", "8192", "--top-k", "20"],
        "requested_n_batch": 128,
    }
    payload = _named_reload_payload(monkeypatch, status, options)
    assert payload == {"model_path": MODEL, "gguf_variant": "Q8_0", **sent}


@pytest.mark.parametrize("effective", [2, 4])
def test_named_resident_slot_count_stays_defaulted(monkeypatch, effective):
    status = {
        "is_gguf": True,
        "gguf_variant": "Q4_K_M",
        "requested_parallel_slots": 4,
        "parallel_slots": effective,
    }
    payload = _named_reload_payload(monkeypatch, status, start.LoadOptions(gguf_variant = "Q4_K_M"))
    assert "n_parallel" not in payload


@pytest.mark.parametrize("mode, carried", [("auto", False), ("manual", True)])
def test_named_resident_auto_tensor_split_not_pinned(monkeypatch, mode, carried):
    status = {
        "is_gguf": True,
        "gguf_variant": "Q8_0",
        "gpu_memory_mode": mode,
        "tensor_parallel": True,
        "tensor_split": [0.6, 0.4],
    }
    payload = _named_reload_payload(
        monkeypatch, status, start.LoadOptions(gguf_variant = "Q8_0", tensor_parallel = True)
    )
    assert ("tensor_split" in payload) is carried


@pytest.mark.parametrize("resident, inherits", [("/old/foo", False), (None, True)])
def test_named_local_checkpoint_inherits_only_from_itself(
    monkeypatch, tmp_path, resident, inherits
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "foo").mkdir()
    status = {
        "active_model": "foo",
        "model_identifier": resident or str(tmp_path / "foo"),
        "load_in_4bit": False,
        "requested_gpu_ids": [1],
    }
    monkeypatch.setattr(start, "_loaded_models", lambda *a: [{"id": "foo", "loaded": True}])
    monkeypatch.setattr(start, "_model_status", lambda *a: status)
    payloads = []
    monkeypatch.setattr(
        start,
        "_load_model_with_progress",
        lambda base, key, model, load, payload: payloads.append(payload)
        or {"status": "loaded", "model": model},
    )
    start._resolve_model(BASE, "key", "foo", start.LoadOptions(max_seq_length = 8192))
    carried = {"load_in_4bit": False, "gpu_ids": [1]} if inherits else {}
    assert payloads == [{"model_path": "foo", "max_seq_length": 8192, **carried}]


def test_named_resident_status_for_another_model_is_not_inherited(monkeypatch):
    status = {"active_model": "owner/other-GGUF", "load_in_4bit": False, "requested_gpu_ids": [1]}
    payload = _named_reload_payload(monkeypatch, status, start.LoadOptions(max_seq_length = 8192))
    assert payload == {"model_path": MODEL, "max_seq_length": 8192}
