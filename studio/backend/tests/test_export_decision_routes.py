# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export routes for decision models: /api/export/decision-info and the GGUF result details."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import allow_ambient_hf_token, get_current_subject
from core.systemone.gguf_export_contract import fingerprint
from routes import export as export_routes

_QWEN35 = {"architectures": ["Qwen3_5ForConditionalGeneration"]}


def _write(
    folder: Path,
    files: dict,
    dirs = (),
) -> Path:
    folder.mkdir(parents = True, exist_ok = True)
    for name, content in files.items():
        (folder / name).write_text(json.dumps(content), encoding = "utf-8")
    for name in dirs:
        (folder / name).mkdir(parents = True, exist_ok = True)
    return folder


def _clef(folder: Path, config = _QWEN35) -> Path:
    return _write(
        folder,
        {"config.json": config, "joint_head.safetensors": {}, "joint_head_config.json": {}},
    )


def _client(monkeypatch, backend = None) -> TestClient:
    async def _supported():
        pass

    monkeypatch.setattr(export_routes, "_ensure_export_supported", _supported)
    if backend is not None:
        monkeypatch.setattr(export_routes, "get_export_backend", lambda: backend)
    app = FastAPI()
    app.include_router(export_routes.router, prefix = "/api/export")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[allow_ambient_hf_token] = lambda: True
    return TestClient(app)


def _info(client, path):
    response = client.get("/api/export/decision-info", params = {"checkpoint_path": str(path)})
    assert response.status_code == 200, response.text
    return response.json()["decision"]


def test_decision_info_for_an_eligible_merged_clef(monkeypatch, tmp_path):
    folder = _clef(tmp_path / "run")
    info = _info(_client(monkeypatch), folder)
    assert info["layout"] == "clef" and info["adapter_only"] is False
    assert info["eligible"] is True and info["reason"] is None
    assert info["quantizations"][0] == info["default_quantization"] == "q8_0"
    assert info["output_dir"] == str(folder / "gguf")
    assert info["existing_export"] is None


def test_decision_info_names_why_a_non_qwen35_clef_is_ineligible(monkeypatch, tmp_path):
    info = _info(
        _client(monkeypatch), _clef(tmp_path / "run", {"architectures": ["LlamaForCausalLM"]})
    )
    assert info["eligible"] is False
    assert "Qwen3.5" in info["reason"] and "LlamaForCausalLM" in info["reason"]


def test_decision_info_for_adapter_only_clef_reads_a_local_base(monkeypatch, tmp_path):
    base = _write(tmp_path / "base", {"config.json": _QWEN35})
    folder = _write(
        tmp_path / "run",
        {
            "adapter_config.json": {"base_model_name_or_path": str(base)},
            "joint_head.safetensors": {},
            "joint_head_config.json": {},
        },
    )
    info = _info(_client(monkeypatch), folder)
    assert info["adapter_only"] is True and info["eligible"] is True


def test_decision_info_for_laya(monkeypatch, tmp_path):
    folder = _write(
        tmp_path / "run",
        {"rl_agent_config.json": {}, "model.safetensors": {}},
        dirs = ("encoder", "tokenizer"),
    )
    _write(folder / "encoder", {"config.json": {"model_type": "modernbert"}})
    info = _info(_client(monkeypatch), folder)
    assert info["layout"] == "laya" and info["eligible"] is True


def _export(folder: Path, quants, fingerprint: str) -> dict:
    export = {
        "format": "unsloth-decision-gguf",
        "version": 1,
        "layout": "clef",
        "quantizations": list(quants),
        "files": {q: {"model": f"model-{q}.gguf", "mmproj": f"mmproj-{q}.gguf"} for q in quants},
        "source_fingerprint": fingerprint,
    }
    _write(folder / "gguf", {"export.json": export})
    return export


def test_decision_info_reports_a_current_export_with_only_the_files_on_disk(monkeypatch, tmp_path):
    folder = _clef(tmp_path / "run")
    export = _export(folder, ["Q8_0", "Q4_K_M"], fingerprint(folder, "clef"))
    for name in ("model-Q8_0.gguf", "mmproj-Q8_0.gguf", "model-Q4_K_M.gguf"):
        (folder / "gguf" / name).write_bytes(b"gguf")
    client = _client(monkeypatch)
    assert _info(client, folder)["existing_export"] == {
        **export,
        "quantizations": ["Q8_0"],
        "files": {"Q8_0": export["files"]["Q8_0"]},
    }
    (folder / "gguf" / "model-Q8_0.gguf").unlink()
    assert _info(client, folder)["existing_export"] is None


def test_decision_info_ignores_a_stale_export(monkeypatch, tmp_path):
    folder = _clef(tmp_path / "run")
    _export(folder, ["Q8_0"], fingerprint(folder, "clef"))
    for name in ("model-Q8_0.gguf", "mmproj-Q8_0.gguf"):
        (folder / "gguf" / name).write_bytes(b"gguf")
    client = _client(monkeypatch)
    assert _info(client, folder)["existing_export"]["quantizations"] == ["Q8_0"]
    # Retrained since the export.
    (folder / "joint_head.safetensors").write_bytes(b"retrained")
    assert _info(client, folder)["existing_export"] is None
    _export(folder, ["Q8_0"], "abc")
    assert _info(client, folder)["existing_export"] is None


def test_decision_info_is_null_for_other_models(monkeypatch, tmp_path):
    client = _client(monkeypatch)
    assert (
        _info(client, _write(tmp_path / "llama", {"config.json": {"model_type": "llama"}})) is None
    )
    assert _info(client, "unsloth/Llama-3.2-1B") is None


def test_ineligible_load_is_a_400_with_the_reason(monkeypatch, tmp_path):
    backend = MagicMock()
    backend.load_checkpoint.return_value = (False, "only Qwen3.5 backbones")
    response = _client(monkeypatch, backend).post(
        "/api/export/load-checkpoint", json = {"checkpoint_path": str(tmp_path)}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "only Qwen3.5 backbones"


def test_decision_gguf_details_point_at_the_run_folder_without_registering_it(
    monkeypatch, tmp_path
):
    folder = _clef(tmp_path / "run")
    export = {
        "format": "unsloth-decision-gguf",
        "version": 1,
        "layout": "clef",
        "quantizations": ["Q8_0", "Q4_K_M"],
        "files": {
            "Q8_0": {"model": "model-Q8_0.gguf", "mmproj": None},
            "Q4_K_M": {"model": "model-Q4_K_M.gguf", "mmproj": None},
        },
        "source_fingerprint": "abc",
    }
    _write(folder / "gguf", {"export.json": export})
    backend = MagicMock()
    backend.decision = {"layout": "clef", "adapter_only": False}
    backend.export_gguf.return_value = (True, "done", str(folder / "gguf"))
    registered = []
    monkeypatch.setattr(
        export_routes, "_try_register_external_export", lambda *a, **k: registered.append(a)
    )

    response = _client(monkeypatch, backend).post(
        "/api/export/export/gguf",
        json = {"save_directory": "ignored", "quantization_method": ["q8_0", "q4_k_m"]},
    )
    assert response.status_code == 200, response.text
    details = response.json()["details"]
    assert details["output_path"] == str(folder / "gguf")
    assert details["quantizations"] == ["Q8_0", "Q4_K_M"]
    assert details["decision_export"] == export
    assert registered == []
    assert backend.export_gguf.call_args.kwargs["quantization_method"] == ["q8_0", "q4_k_m"]


def test_non_decision_gguf_details_unchanged(monkeypatch, tmp_path):
    backend = MagicMock()
    backend.decision = None
    backend.export_gguf.return_value = (True, "done", str(tmp_path / "out"))
    seen = []
    monkeypatch.setattr(
        export_routes,
        "_export_details",
        lambda path, refresh_index = False: seen.append((path, refresh_index))
        or {"output_path": "out"},
    )
    response = _client(monkeypatch, backend).post(
        "/api/export/export/gguf", json = {"save_directory": "out", "quantization_method": "Q4_K_M"}
    )
    assert response.status_code == 200
    assert seen == [(str(tmp_path / "out"), True)]
    assert response.json()["details"] == {"output_path": "out"}
