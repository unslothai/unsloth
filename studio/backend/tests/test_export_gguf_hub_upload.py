# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.util
from pathlib import Path


_HELPERS_SPEC = importlib.util.spec_from_file_location(
    "export_gguf_hub_upload_helpers",
    Path(__file__).with_name("test_export_absolute_paths.py"),
)
assert _HELPERS_SPEC is not None and _HELPERS_SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_HELPERS_SPEC)
_HELPERS_SPEC.loader.exec_module(_HELPERS)
_install_export_backend_stubs = _HELPERS._install_export_backend_stubs
_load_module = _HELPERS._load_module


class _RepoUrl(str):
    repo_id = "owner/model"


def _hub_doubles(calls, seen):
    class _HfApi:
        def __init__(self, token = None):
            seen["token"] = token

        def create_repo(
            self,
            repo_id,
            private = False,
            exist_ok = False,
        ):
            seen["repo"] = {"repo_id": repo_id, "private": private, "exist_ok": exist_ok}
            return _RepoUrl("https://huggingface.co/owner/model")

        def upload_folder(
            self,
            folder_path,
            repo_id,
            repo_type,
            ignore_patterns = None,
        ):
            calls.append("upload_folder")
            seen["folder"] = folder_path
            ignored = set(ignore_patterns or ())
            seen["uploaded"] = sorted(
                p.name for p in Path(folder_path).iterdir() if p.name not in ignored
            )

    class _ModelCard:
        def __init__(self, content):
            seen["card"] = content

        def push_to_hub(
            self,
            repo_id,
            token = None,
            commit_message = None,
        ):
            seen["card_repo"] = repo_id

    return _HfApi, _ModelCard


def test_gguf_hub_export_uploads_the_built_files_instead_of_reconverting(tmp_path, monkeypatch):
    _install_export_backend_stubs(monkeypatch)
    export_module = _load_module(
        "test_export_gguf_hub_upload_backend",
        "core/export/export.py",
        monkeypatch,
    )

    calls: list[str] = []
    seen: dict = {}

    class Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            calls.append("convert")
            output = Path(f"{model_save_path}_gguf")
            output.mkdir(parents = True)
            (output / "model.Q4_K_M.gguf").write_bytes(b"GGUF")
            (output / "Modelfile").write_text("FROM model.Q4_K_M.gguf")

        def push_to_hub_gguf(self, *args, **kwargs):
            calls.append("push_to_hub_gguf")

    hf_api, model_card = _hub_doubles(calls, seen)
    monkeypatch.setattr(export_module, "HfApi", hf_api)
    monkeypatch.setattr(export_module, "ModelCard", model_card)
    monkeypatch.setattr(export_module, "resolve_export_write_dir", lambda value: Path(value))

    backend = export_module.ExportBackend.__new__(export_module.ExportBackend)
    backend.current_model = Model()
    backend.current_tokenizer = object()
    backend.current_checkpoint = None

    success, message, output_path = backend.export_gguf(
        str(tmp_path / "export"),
        "Q4_K_M",
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "token",
        private = True,
    )

    assert success is True, message
    assert calls == ["convert", "upload_folder"]
    assert seen["folder"] == output_path
    assert seen["repo"] == {"repo_id": "owner/model", "private": True, "exist_ok": True}
    assert "model.Q4_K_M.gguf" in seen["uploaded"]
    assert "Modelfile" in seen["uploaded"]
    # Studio-local, and push_to_hub_gguf never published it.
    assert "export_metadata.json" not in seen["uploaded"]
    assert Path(output_path, "export_metadata.json").is_file()
    assert "`model.Q4_K_M.gguf`" in seen["card"]
    assert seen["card_repo"] == "owner/model"
