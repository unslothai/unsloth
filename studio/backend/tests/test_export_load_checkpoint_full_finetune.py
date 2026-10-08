# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import allow_ambient_hf_token, get_current_subject
from routes import export as export_routes
from utils.models import checkpoints


async def _fake_ensure_export_supported():
    pass


_FULL_FINETUNE = {"config.json": {"model_type": "llama"}}
_LORA_ADAPTER = {
    "config.json": {"model_type": "llama"},
    "adapter_config.json": {"base_model_name_or_path": "unsloth/Llama-3.2-1B"},
}
_BNB_QUANTIZED = {
    "config.json": {
        "model_type": "llama",
        "quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": True},
    }
}
_MALFORMED_CONFIG = {"config.json": "{not json"}


def _load_via_orchestrator(
    monkeypatch,
    checkpoint_path,
    load_in_4bit,
    hf_token = None,
):
    from core.export.orchestrator import ExportOrchestrator

    backend = ExportOrchestrator()
    spawned = {}
    monkeypatch.setattr(backend, "_ensure_subprocess_alive", lambda: False)
    monkeypatch.setattr(backend, "_spawn_subprocess", spawned.update)
    monkeypatch.setattr(
        backend, "_wait_response", lambda *a, **k: {"success": True, "message": "loaded"}
    )
    assert backend.load_checkpoint(checkpoint_path, load_in_4bit = load_in_4bit, hf_token = hf_token)[0]
    return spawned["load_in_4bit"]


@pytest.mark.parametrize(
    "files,load_in_4bit,expected",
    [
        (_FULL_FINETUNE, None, False),
        (_FULL_FINETUNE, True, True),
        (_LORA_ADAPTER, None, True),
        (_LORA_ADAPTER, False, False),
        (_BNB_QUANTIZED, None, True),
        (_MALFORMED_CONFIG, None, True),
    ],
    ids = [
        "full_finetune",
        "full_finetune_explicit_4bit",
        "lora_adapter",
        "lora_adapter_explicit_16bit",
        "bnb_quantized",
        "malformed_config",
    ],
)
def test_load_checkpoint_uses_16bit_for_unquantized_full_finetunes(
    monkeypatch, tmp_path, files, load_in_4bit, expected
):
    for name, content in files.items():
        text = content if isinstance(content, str) else json.dumps(content)
        (tmp_path / name).write_text(text, encoding = "utf-8")

    assert _load_via_orchestrator(monkeypatch, str(tmp_path), load_in_4bit) is expected


@pytest.mark.parametrize(
    "adapter_on_hub,hub_config,expected",
    [
        (False, {"model_type": "llama"}, False),
        (
            False,
            {"model_type": "llama", "quantization_config": {"quant_method": "bitsandbytes"}},
            True,
        ),
        (True, {"model_type": "llama"}, True),
    ],
    ids = ["remote_full_finetune", "remote_bnb_quantized", "remote_lora_adapter"],
)
def test_load_checkpoint_resolves_hub_ids_before_choosing_4bit(
    monkeypatch, tmp_path, adapter_on_hub, hub_config, expected
):
    """A Hub id never touches the local filesystem, so its config.json has to be fetched."""
    import huggingface_hub

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(hub_config), encoding = "utf-8")
    tokens = []

    def _file_exists(
        repo_id,
        filename,
        token = None,
    ):
        tokens.append(token)
        return adapter_on_hub and filename == "adapter_config.json"

    def _hf_hub_download(
        repo_id,
        filename,
        token = None,
    ):
        tokens.append(token)
        return str(config_file)

    monkeypatch.setattr(huggingface_hub, "file_exists", _file_exists)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _hf_hub_download)

    # False is the "ambient token denied" sentinel; it must reach the Hub unchanged.
    load_in_4bit = _load_via_orchestrator(
        monkeypatch, "unsloth/Llama-3.2-1B-Instruct", None, hf_token = False
    )
    assert load_in_4bit is expected
    assert tokens and all(token is False for token in tokens)


@pytest.mark.parametrize(
    "extra,expected",
    [({}, None), ({"load_in_4bit": True}, True), ({"load_in_4bit": False}, False)],
    ids = ["unset", "explicit_4bit", "explicit_16bit"],
)
def test_route_leaves_unset_load_in_4bit_to_the_backend(monkeypatch, tmp_path, extra, expected):
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    backend = MagicMock()
    backend.load_checkpoint.return_value = (True, "loaded")
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: backend)

    app = FastAPI()
    app.include_router(export_routes.router, prefix = "/api/export")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[allow_ambient_hf_token] = lambda: True

    response = TestClient(app).post(
        "/api/export/load-checkpoint",
        json = {"checkpoint_path": str(tmp_path), "hf_token": None, **extra},
    )

    assert response.status_code == 200
    assert backend.load_checkpoint.call_args.kwargs["load_in_4bit"] is expected


def test_hub_lookup_failure_keeps_the_old_default(monkeypatch):
    """Offline, gated without a token, or no such repo: fail OPEN to the historical 4-bit
    default rather than guessing 16-bit and OOMing a load that used to fit."""

    def _boom(
        repo_id,
        filename,
        token = None,
    ):
        raise OSError("no network")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "file_exists", _boom, raising = False)
    assert checkpoints._hub_model_config("org/model", None) is None
    assert checkpoints.is_unquantized_full_finetune("org/model", None) is False
