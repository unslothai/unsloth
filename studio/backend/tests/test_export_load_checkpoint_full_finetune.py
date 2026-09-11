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
    ids = [
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
        (tmp_path / name).write_text(text, encoding = "utf-8")

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
