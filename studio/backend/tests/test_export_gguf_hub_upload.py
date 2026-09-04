# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.util
from fnmatch import fnmatch
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
            allow_patterns = None,
            ignore_patterns = None,
        ):
            calls.append("upload_folder")
            seen["folder"] = folder_path
            # Mirror huggingface_hub: fnmatch over repo-relative paths, whole tree.
            root = Path(folder_path)
            paths = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]
            if allow_patterns is not None:
                paths = [
                    p for p in paths
                    if any(fnmatch(p, pattern) for pattern in allow_patterns)
                ]
            for pattern in ignore_patterns or ():
                paths = [p for p in paths if not fnmatch(p, pattern)]
            seen["uploaded"] = sorted(paths)

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


def test_gguf_hub_export_uploads_only_the_export_artifacts(tmp_path, monkeypatch):
    """The save directory is user-chosen, so its other contents must stay off the Hub."""
    _install_export_backend_stubs(monkeypatch)
    export_module = _load_module(
        "test_export_gguf_hub_upload_scoped_backend",
        "core/export/export.py",
        monkeypatch,
    )

    save_dir = tmp_path / "picked-in-the-folder-browser"
    save_dir.mkdir()
    (save_dir / "notes.txt").write_text("unrelated")
    (save_dir / "dataset.jsonl").write_text('{"a": 1}')
    # A relocation failure keeps the merged checkpoint behind, by design.
    leftover = save_dir / "_tmp_model_earlier" / "model"
    leftover.mkdir(parents = True)
    (leftover / "model-00001-of-00002.safetensors").write_bytes(b"weights")

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
        str(save_dir),
        "Q4_K_M",
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "token",
        private = False,
    )

    assert success is True, message
    assert seen["uploaded"] == ["Modelfile", "model.Q4_K_M.gguf"]
    # Still on disk where the user put them, just not published.
    assert (Path(output_path) / "notes.txt").is_file()
    assert (Path(output_path) / "dataset.jsonl").is_file()
    assert (leftover / "model-00001-of-00002.safetensors").is_file()
