# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import allow_ambient_hf_token, get_current_subject
from routes import export as export_routes


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


@pytest.mark.parametrize(
    "files,extra,expected",
    [
        (_FULL_FINETUNE, {}, False),
        (_FULL_FINETUNE, {"load_in_4bit": True}, True),
        (_LORA_ADAPTER, {}, True),
        (_BNB_QUANTIZED, {}, True),
        (_MALFORMED_CONFIG, {}, True),
    ],
    ids=[
        "full_finetune",
        "full_finetune_explicit_4bit",
        "lora_adapter",
        "bnb_quantized",
        "malformed_config",
    ],
)
def test_load_checkpoint_uses_16bit_for_unquantized_full_finetunes(
    monkeypatch, tmp_path, files, extra, expected
):
    for name, content in files.items():
        text = content if isinstance(content, str) else json.dumps(content)
        (tmp_path / name).write_text(text, encoding="utf-8")

    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    backend = MagicMock()
    backend.load_checkpoint.return_value = (True, "loaded")
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: backend)

    app = FastAPI()
    app.include_router(export_routes.router, prefix="/api/export")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[allow_ambient_hf_token] = lambda: True

    response = TestClient(app).post(
        "/api/export/load-checkpoint",
        json={"checkpoint_path": str(tmp_path), "hf_token": None, **extra},
    )

    assert response.status_code == 200
    assert backend.load_checkpoint.call_args.kwargs["load_in_4bit"] is expected


def _hub_client(monkeypatch, backend):
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: backend)
    app = FastAPI()
    app.include_router(export_routes.router, prefix="/api/export")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[allow_ambient_hf_token] = lambda: True
    return TestClient(app)


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
    ids=["remote_full_finetune", "remote_bnb_quantized", "remote_lora_adapter"],
)
def test_load_checkpoint_resolves_hub_ids_before_choosing_4bit(
    monkeypatch, tmp_path, adapter_on_hub, hub_config, expected
):
    """A Hub id never touches the local filesystem, so without this the request keeps the
    request model's True default and a remote full fine-tune exports as 4-bit anyway."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(hub_config), encoding="utf-8")

    monkeypatch.setattr(
        export_routes,
        "_hub_config",
        lambda repo_id, hf_token: None if adapter_on_hub else hub_config,
    )
    backend = MagicMock()
    backend.load_checkpoint.return_value = (True, "loaded")

    response = _hub_client(monkeypatch, backend).post(
        "/api/export/load-checkpoint",
        json={"checkpoint_path": "unsloth/Llama-3.2-1B-Instruct", "hf_token": None},
    )

    assert response.status_code == 200
    assert backend.load_checkpoint.call_args.kwargs["load_in_4bit"] is expected


def test_hub_lookup_failure_keeps_the_old_default(monkeypatch):
    """Offline, gated without a token, or no such repo: fail OPEN to the historical 4-bit
    default rather than guessing 16-bit and OOMing a load that used to fit."""

    def _boom(
        repo_id,
        filename,
        token=None,
    ):
        raise OSError("no network")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "file_exists", _boom, raising=False)
    assert export_routes._hub_config("org/model", None) is None
    assert export_routes._is_unquantized_full_finetune("org/model", None) is False
