# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from models.export import ExportGGUFRequest


_HELPERS_SPEC = importlib.util.spec_from_file_location(
    "export_absolute_path_helpers",
    Path(__file__).with_name("test_export_absolute_paths.py"),
)
assert _HELPERS_SPEC is not None and _HELPERS_SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_HELPERS_SPEC)
_HELPERS_SPEC.loader.exec_module(_HELPERS)
_install_export_backend_stubs = _HELPERS._install_export_backend_stubs
_load_module = _HELPERS._load_module


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("", "0"),
        ("none", "0"),
        ("0", "0"),
        ("500m", "500MB"),
        (" 4 GB ", "4GB"),
    ],
)
def test_gguf_request_normalizes_shard_size(value, expected):
    request = ExportGGUFRequest(save_directory = "/tmp/export", gguf_shard_size = value)
    assert request.gguf_shard_size == expected


@pytest.mark.parametrize(
    "value",
    ["0MB", "0GB", "1.5GB", "512", "64KB", "-2GB", "4TB", "4GBx"],
)
def test_gguf_request_rejects_invalid_shard_size(value):
    with pytest.raises(ValidationError, match = "gguf_shard_size"):
        ExportGGUFRequest(save_directory = "/tmp/export", gguf_shard_size = value)


def test_orchestrator_preserves_shard_size_in_command():
    from core.export.orchestrator import ExportOrchestrator

    orchestrator = ExportOrchestrator.__new__(ExportOrchestrator)
    seen = {}

    def run_export(kind, params):
        seen.update(kind = kind, params = params)
        return True, "ok", None

    orchestrator._run_export = run_export

    result = orchestrator.export_gguf("output", gguf_shard_size = "2GB", private = True)

    assert result == (True, "ok", None)
    assert seen["kind"] == "gguf"
    assert seen["params"]["gguf_shard_size"] == "2GB"
    assert seen["params"]["private"] is True


def test_worker_passes_shard_size_to_backend():
    from core.export import worker

    seen = {}

    class Backend:
        def export_gguf(self, **kwargs):
            seen.update(kwargs)
            return True, "ok", "/output"

    class Queue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    queue = Queue()
    worker._handle_export(
        Backend(),
        {
            "export_type": "gguf",
            "save_directory": "/output",
            "gguf_shard_size": "512MB",
            "private": True,
        },
        queue,
    )

    assert seen["gguf_shard_size"] == "512MB"
    assert seen["private"] is True
    assert queue.items[-1]["success"] is True


def test_backend_forwards_shard_size_to_the_local_export_it_then_uploads(tmp_path, monkeypatch):
    _install_export_backend_stubs(monkeypatch)
    export_module = _load_module(
        "test_export_gguf_shard_backend",
        "core/export/export.py",
        monkeypatch,
    )
    save_directory = tmp_path / "export with spaces ü"
    seen = {}

    class Model:
        def save_pretrained_gguf(
            self,
            model_save_path,
            tokenizer,
            quantization_method,
            gguf_shard_size = None,
        ):
            seen["local"] = gguf_shard_size
            output = Path(f"{model_save_path}_gguf")
            output.mkdir(parents = True)
            shard = output / "model.F16-00001-of-00002.gguf"
            shard.write_bytes(b"GGUF")
            return {"gguf_files": [str(shard)]}

        def push_to_hub_gguf(self, *args, **kwargs):
            seen["hub"] = kwargs

    class _RepoUrl(str):
        repo_id = "owner/model"

    class _HfApi:
        def __init__(self, token = None):
            seen["token"] = token

        def create_repo(
            self,
            repo_id,
            private = False,
            exist_ok = False,
        ):
            seen["repo"] = {"repo_id": repo_id, "private": private}
            return _RepoUrl("https://huggingface.co/owner/model")

        def upload_folder(
            self,
            folder_path,
            repo_id,
            repo_type,
            allow_patterns = None,
            ignore_patterns = None,
        ):
            seen["upload"] = folder_path

    class _ModelCard:
        def __init__(self, content):
            pass

        def push_to_hub(
            self,
            repo_id,
            token = None,
            commit_message = None,
        ):
            pass

    monkeypatch.setattr(export_module, "HfApi", _HfApi)
    monkeypatch.setattr(export_module, "ModelCard", _ModelCard)
    monkeypatch.setattr(export_module, "resolve_export_write_dir", lambda value: Path(value))
    backend = export_module.ExportBackend.__new__(export_module.ExportBackend)
    backend.current_model = Model()
    backend.current_tokenizer = object()
    backend.current_checkpoint = None

    success, message, output_path = backend.export_gguf(
        str(save_directory),
        "F16",
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "token",
        private = True,
        gguf_shard_size = "512MB",
    )

    assert success is True, message
    assert output_path == str(save_directory.resolve())
    assert seen["local"] == "512MB"
    assert "hub" not in seen
    assert seen["upload"] == output_path
    assert seen["repo"] == {"repo_id": "owner/model", "private": True}


def test_backend_rejects_old_exporter_only_when_option_is_set(tmp_path, monkeypatch):
    _install_export_backend_stubs(monkeypatch)
    export_module = _load_module(
        "test_export_gguf_old_backend",
        "core/export/export.py",
        monkeypatch,
    )

    class OldModel:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            output = Path(f"{model_save_path}_gguf")
            output.mkdir(parents = True)
            (output / "model.Q4_K_M.gguf").write_bytes(b"GGUF")

    monkeypatch.setattr(export_module, "resolve_export_write_dir", lambda value: Path(value))
    backend = export_module.ExportBackend.__new__(export_module.ExportBackend)
    backend.current_model = OldModel()
    backend.current_tokenizer = object()
    backend.current_checkpoint = None

    unsupported = backend.export_gguf(
        str(tmp_path / "unsupported"),
        gguf_shard_size = "0",
    )
    compatible = backend.export_gguf(str(tmp_path / "compatible"))

    assert unsupported[0] is False
    assert "does not support GGUF shard-size control" in unsupported[1]
    assert compatible[0] is True
