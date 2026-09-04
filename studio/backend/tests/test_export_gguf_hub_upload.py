# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.util
from fnmatch import fnmatch
from pathlib import Path

import pytest


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

        def upload_file(
            self,
            path_or_fileobj,
            path_in_repo,
            repo_id,
            repo_type = None,
            commit_message = None,
        ):
            calls.append(f"upload_file:{path_in_repo}")
            seen[path_in_repo] = path_or_fileobj

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
                paths = [p for p in paths if any(fnmatch(p, pattern) for pattern in allow_patterns)]
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
            calls.append("model_card")
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
    # One conversion, and the card lands after the files it advertises.
    assert calls == ["convert", "upload_folder", "model_card"]
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


def test_gguf_hub_export_allow_list_treats_gguf_names_literally(tmp_path, monkeypatch):
    """A glob character in the model name must not skip its file or match another."""
    _install_export_backend_stubs(monkeypatch)
    export_module = _load_module(
        "test_export_gguf_hub_upload_glob_backend",
        "core/export/export.py",
        monkeypatch,
    )

    save_dir = tmp_path / "llama-3[8b]"
    save_dir.mkdir()
    # An earlier export's file, named so a bare "a*.gguf" pattern would sweep it in.
    (save_dir / "a-previous-quant.gguf").write_bytes(b"GGUF")

    calls: list[str] = []
    seen: dict = {}

    class Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            calls.append("convert")
            output = Path(f"{model_save_path}_gguf")
            output.mkdir(parents = True)
            (output / "a*.gguf").write_bytes(b"GGUF")
            (output / "llama-3[8b].Q4_K_M.gguf").write_bytes(b"GGUF")

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

    success, message, _path = backend.export_gguf(
        str(save_dir),
        "Q4_K_M",
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "token",
        private = False,
    )

    assert success is True, message
    # The bracketed name is published rather than skipped, and "a*.gguf" is matched as a
    # literal, so it neither misses itself nor sweeps in the earlier export's file.
    assert seen["uploaded"] == ["a*.gguf", "llama-3[8b].Q4_K_M.gguf"]


def test_gguf_hub_export_skips_an_earlier_export_left_in_the_folder(tmp_path, monkeypatch):
    """Only this run's GGUFs are published, even when the folder holds an older one."""
    _install_export_backend_stubs(monkeypatch)
    export_module = _load_module(
        "test_export_gguf_hub_upload_stale_backend",
        "core/export/export.py",
        monkeypatch,
    )

    save_dir = tmp_path / "shared-export-folder"
    save_dir.mkdir()
    # A different model, exported here earlier. It must not ride along.
    (save_dir / "some-other-model.Q8_0.gguf").write_bytes(b"GGUF")

    calls: list[str] = []
    seen: dict = {}

    class Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            merged = Path(model_save_path)
            merged.mkdir(parents = True)
            (merged / "config.json").write_text('{"model_type": "llama"}')
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
    assert "some-other-model.Q8_0.gguf" not in seen["card"]
    # Still on disk, just not republished under this repo.
    assert (Path(output_path) / "some-other-model.Q8_0.gguf").is_file()
    # config.json comes from the merged directory, which is deleted before the upload,
    # and it is never written into the export folder.
    assert seen["config.json"] == b'{"model_type": "llama"}'
    assert not (Path(output_path) / "config.json").exists()
    # The card advertises the files, so it must land after them.
    assert calls.index("model_card") > calls.index("upload_folder")


def test_gguf_hub_export_leaves_a_stale_modelfile_behind(tmp_path, monkeypatch):
    """A Modelfile from an earlier export points at another model; only a fresh one ships."""
    _install_export_backend_stubs(monkeypatch)
    export_module = _load_module(
        "test_export_gguf_hub_upload_stale_modelfile_backend",
        "core/export/export.py",
        monkeypatch,
    )

    save_dir = tmp_path / "shared-export-folder"
    save_dir.mkdir()
    (save_dir / "Modelfile").write_text("FROM some-other-model.Q8_0.gguf")

    calls: list[str] = []
    seen: dict = {}

    class Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            output = Path(f"{model_save_path}_gguf")
            output.mkdir(parents = True)
            (output / "model.Q4_K_M.gguf").write_bytes(b"GGUF")
            # This conversion produces no Modelfile.

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
    assert seen["uploaded"] == ["model.Q4_K_M.gguf"]
    assert (Path(output_path) / "Modelfile").is_file()


def test_gguf_hub_export_does_not_publish_appledouble_companions(tmp_path, monkeypatch):
    """A ._ companion beside a GGUF is Finder metadata, not a model file."""
    _install_export_backend_stubs(monkeypatch)
    export_module = _load_module(
        "test_export_gguf_hub_upload_appledouble_backend",
        "core/export/export.py",
        monkeypatch,
    )

    save_dir = tmp_path / "export"
    calls: list[str] = []
    seen: dict = {}

    class Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            output = Path(f"{model_save_path}_gguf")
            output.mkdir(parents = True)
            (output / "model.Q4_K_M.gguf").write_bytes(b"GGUF")
            (output / "._model.Q4_K_M.gguf").write_bytes(b"\x00\x05\x16\x07")

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

    success, message, _path = backend.export_gguf(
        str(save_dir),
        "Q4_K_M",
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "token",
        private = False,
    )

    assert success is True, message
    assert seen["uploaded"] == ["model.Q4_K_M.gguf"]
    assert "._model.Q4_K_M.gguf" not in seen["card"]


@pytest.mark.parametrize("is_vlm", [True, False])
def test_gguf_hub_export_card_carries_the_vlm_tag(tmp_path, monkeypatch, is_vlm):
    """The Hub filters on vision-language-model, and only the exporter knows."""
    _install_export_backend_stubs(monkeypatch)
    export_module = _load_module(
        f"test_export_gguf_hub_upload_vlm_{is_vlm}_backend",
        "core/export/export.py",
        monkeypatch,
    )

    save_dir = tmp_path / "export"
    calls: list[str] = []
    seen: dict = {}

    class Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            output = Path(f"{model_save_path}_gguf")
            output.mkdir(parents = True)
            gguf = output / "model.Q4_K_M.gguf"
            gguf.write_bytes(b"GGUF")
            return {"gguf_files": [str(gguf)], "is_vlm": is_vlm}

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

    success, message, _path = backend.export_gguf(
        str(save_dir),
        "Q4_K_M",
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "token",
        private = False,
    )

    assert success is True, message
    assert ("- vision-language-model" in seen["card"]) is is_vlm
    # The front matter stays valid either way.
    assert seen["card"].startswith("---\ntags:\n")
    assert (
        seen["card"]
        .split("---")[1]
        .strip()
        .endswith("vision-language-model" if is_vlm else "unsloth")
    )
