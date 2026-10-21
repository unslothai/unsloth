# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Remote code reaches a load only through the consent gate.

- GGUF export: llama.cpp's converter loads some tokenizers with trust_remote_code=True, so
  unless the checkpoint load was approved for remote code the converter gets no usable
  modules cache to import a tokenizer auto_map from.
- Managed accounts: the export and training routes downgrade a requested
  trust_remote_code to False (the Export tab always sends true, so no refusal).
- A legacy list-form tokenizer auto_map counts as remote code for the scan.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

_KEY = "HF_MODULES_CACHE"


# ---- GGUF export: the converter follows the load's consent-gated decision ----------------


@pytest.mark.parametrize("mlx", [False, True])
@pytest.mark.parametrize("approved", [None, False, True])
def test_gguf_export_withholds_the_modules_cache_unless_approved(
    tmp_path, monkeypatch, mlx, approved
):
    from test_export_converter_pin import _export_mod, _zoo

    mod = _export_mod(monkeypatch, mlx = mlx)
    _zoo(monkeypatch)
    monkeypatch.setenv(_KEY, "/mine")
    save_dir = tmp_path / "export"
    monkeypatch.setattr(mod, "resolve_export_write_dir", lambda _value: save_dir)
    seen = []

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, **kwargs):
            seen.append(os.environ.get(_KEY))
            output_dir = Path(f"{model_save_path}_gguf")
            output_dir.mkdir(parents = True, exist_ok = True)
            (output_dir / "converted.gguf").write_bytes(b"gguf")

        def push_to_hub_gguf(self, repo_id, tokenizer, **kwargs):
            seen.append(os.environ.get(_KEY))

    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.current_model = _Model()
    backend.current_tokenizer = object()
    backend.current_checkpoint = None
    if approved is not None:
        backend.trust_remote_code = approved

    success, message, _ = backend.export_gguf(str(save_dir), "Q4_K_M")
    assert success is True, message
    success, message, _ = backend.export_gguf(
        "", "Q4_K_M", push_to_hub = True, repo_id = "me/model", hf_token = "hf_x"
    )
    assert success is True, message

    assert len(seen) == 2
    if approved:
        assert seen == ["/mine", "/mine"]
    else:
        for value in seen:
            assert value != "/mine"
            with pytest.raises(OSError):
                os.makedirs(value, exist_ok = True)
    assert os.environ[_KEY] == "/mine"


# ---- managed accounts never forward trust_remote_code ------------------------------------


def _as_managed(
    monkeypatch,
    managed,
    override = False,
):
    from utils.security import consent
    monkeypatch.setattr(consent, "is_owner_context", lambda: not managed)
    if override:
        monkeypatch.setenv(consent.MANAGED_REMOTE_CODE_OVERRIDE, "1")
    else:
        monkeypatch.delenv(consent.MANAGED_REMOTE_CODE_OVERRIDE, raising = False)


@pytest.mark.parametrize(
    "managed, override, expected",
    [(True, False, False), (False, False, True), (True, True, True)],
    ids = ["managed", "owner", "managed_with_owner_override"],
)
def test_export_load_downgrades_trust_for_managed_accounts(
    monkeypatch, tmp_path, managed, override, expected
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import allow_ambient_hf_token, get_current_subject
    from routes import export as export_routes

    async def _supported():
        return None

    _as_managed(monkeypatch, managed, override)
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _supported)
    monkeypatch.setattr(export_routes, "_authorized_adapter_base", lambda _path: None)
    backend = MagicMock()
    backend.load_checkpoint.return_value = (True, "loaded")
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: backend)

    app = FastAPI()
    app.include_router(export_routes.router, prefix = "/api/export")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[allow_ambient_hf_token] = lambda: True

    response = TestClient(app).post(
        "/api/export/load-checkpoint",
        json = {"checkpoint_path": str(tmp_path), "hf_token": None, "trust_remote_code": True},
    )

    assert response.status_code == 200, response.text
    assert backend.load_checkpoint.call_args.kwargs["trust_remote_code"] is expected


@pytest.fixture
def training_route():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "remote_code_gate_training_route", _BACKEND_DIR / "routes" / "training.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._hub_unreachable = lambda: False
    return module


async def _inline_to_thread(function, *args, **kwargs):
    return function(*args, **kwargs)


@pytest.mark.parametrize(
    "managed, expected", [(True, False), (False, True)], ids = ["managed", "owner"]
)
def test_training_start_downgrades_trust_for_managed_accounts(
    monkeypatch, tmp_path, training_route, managed, expected
):
    from models.training import TrainingStartRequest

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding = "utf-8")
    (model / "model.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
    request = TrainingStartRequest(
        model_name = str(model),
        training_type = "LoRA/QLoRA",
        format_type = "auto",
        hf_dataset = "org/data",
        load_in_4bit = False,
        trust_remote_code = True,
    )
    _as_managed(monkeypatch, managed)
    captured = {}
    backend = SimpleNamespace(
        current_job_id = None,
        is_training_active = lambda: False,
        start_training = lambda **kwargs: captured.update(kwargs) or True,
    )
    with (
        patch.object(training_route, "get_training_backend", return_value = backend),
        patch.object(training_route, "_preflight_hf_dataset_request", return_value = None),
        patch.object(training_route.asyncio, "to_thread", _inline_to_thread),
    ):
        response = asyncio.run(training_route.start_training(request, current_subject = "alice"))
    assert response.status == "queued", response
    assert "trust_remote_code" in captured, "training backend was never reached"
    assert captured["trust_remote_code"] is expected


# ---- a list-form tokenizer auto_map is remote code ---------------------------------------


def test_a_list_form_tokenizer_auto_map_is_declared():
    from utils.security.remote_code_scan import _auto_map_refs, config_declares_auto_map

    config = {"auto_map": ["tokenization_evil.EvilTok", None]}
    assert config_declares_auto_map(config) is True
    assert _auto_map_refs(config) == {(None, "tokenization_evil.py")}
    hub = {"auto_map": ["acme/evil--tokenization_evil.EvilTok", None]}
    assert _auto_map_refs(hub) == {("acme/evil", "tokenization_evil.py")}
    # Dict form and no auto_map are unchanged.
    assert config_declares_auto_map({"auto_map": {"AutoTokenizer": [None, "t.T"]}}) is True
    assert config_declares_auto_map({"model_type": "llama"}) is False
    assert config_declares_auto_map({"auto_map": []}) is False


def test_the_gate_scans_a_list_form_tokenizer_auto_map(tmp_path):
    from utils.security import evaluate_remote_code_consent

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding = "utf-8")
    (model / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "auto_map": [None, "tokenization_evil.EvilTok"],
            }
        ),
        encoding = "utf-8",
    )
    (model / "tokenization_evil.py").write_text(
        "import subprocess\nsubprocess.Popen(['id'])\n", encoding = "utf-8"
    )

    decision = evaluate_remote_code_consent(str(model), trust_remote_code = True)

    assert decision.has_remote_code is True
    assert decision.fingerprint
    assert "no-op" not in decision.reason
