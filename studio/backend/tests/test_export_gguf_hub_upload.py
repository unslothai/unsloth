# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.util
import os
import types
from fnmatch import fnmatchcase
from pathlib import Path

import pytest


# "*" is a reserved character in a Windows filename, so only POSIX can hold one.
_STAR_IN_NAME_IS_LEGAL = os.name != "nt"


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
            calls.append("create_repo")
            seen["repo"] = {"repo_id": repo_id, "private": private, "exist_ok": exist_ok}
            return _RepoUrl("https://huggingface.co/owner/model")

        def update_repo_settings(
            self,
            repo_id,
            private = None,
            repo_type = None,
        ):
            calls.append("update_repo_settings")
            seen["visibility"] = {"repo_id": repo_id, "private": private}

        def repo_info(
            self,
            repo_id,
            repo_type = None,
        ):
            calls.append("repo_info")
            return seen.get("repo_info_result")

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
            # Mirror filter_repo_objects: case-sensitive fnmatchcase over repo-relative paths.
            root = Path(folder_path)
            paths = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
            if allow_patterns is not None:
                paths = [
                    p for p in paths if any(fnmatchcase(p, pattern) for pattern in allow_patterns)
                ]
            for pattern in ignore_patterns or ():
                paths = [p for p in paths if not fnmatchcase(p, pattern)]
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
    assert calls == [
        "convert",
        "create_repo",
        "update_repo_settings",
        "upload_folder",
        "model_card",
    ]
    assert seen["folder"] == output_path
    assert seen["repo"] == {"repo_id": "owner/model", "private": True, "exist_ok": True}
    assert seen["visibility"] == {"repo_id": "owner/model", "private": True}
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
            if _STAR_IN_NAME_IS_LEGAL:
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
    expected = ["llama-3[8b].Q4_K_M.gguf"]
    if _STAR_IN_NAME_IS_LEGAL:
        expected = ["a*.gguf"] + expected
    assert seen["uploaded"] == expected


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
    assert (Path(output_path) / "some-other-model.Q8_0.gguf").is_file()
    # config.json comes from the merged directory, deleted before the upload.
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
    assert seen["card"].startswith("---\ntags:\n")
    assert (
        seen["card"]
        .split("---")[1]
        .strip()
        .endswith("vision-language-model" if is_vlm else "unsloth")
    )
    # The Hub filters on the exact string, so both spellings must ship.
    tags = [line[2:] for line in seen["card"].split("---")[1].strip().splitlines()[1:]]
    assert "llama.cpp" in tags
    assert "llama-cpp" in tags


def _visibility_backend(
    tmp_path,
    monkeypatch,
    name,
    gguf_names = ("model.Q4_K_M.gguf",),
):
    """A backend wired to the Hub doubles, for the visibility and failure-path tests."""
    _install_export_backend_stubs(monkeypatch)
    export_module = _load_module(name, "core/export/export.py", monkeypatch)

    calls: list[str] = []
    seen: dict = {}

    class Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            calls.append("convert")
            output = Path(f"{model_save_path}_gguf")
            output.mkdir(parents = True)
            produced = []
            for gguf_name in gguf_names:
                gguf = output / gguf_name
                # A "._" name is only Finder metadata if it carries the magic too.
                gguf.write_bytes(b"\x00\x05\x16\x07" if gguf_name.startswith("._") else b"GGUF")
                produced.append(str(gguf))
            return {"gguf_files": produced}

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
    return export_module, backend, calls, seen


def test_gguf_hub_export_makes_an_existing_repo_private_before_uploading(tmp_path, monkeypatch):
    """create_repo ignores private on an existing repo, so a "private" export would land public."""
    _module, backend, calls, seen = _visibility_backend(
        tmp_path, monkeypatch, "test_export_gguf_hub_upload_visibility_backend"
    )

    success, message, _path = backend.export_gguf(
        str(tmp_path / "export"),
        "Q4_K_M",
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "token",
        private = True,
    )

    assert success is True, message
    assert seen["visibility"] == {"repo_id": "owner/model", "private": True}
    assert calls.index("update_repo_settings") < calls.index("upload_folder")


def test_gguf_hub_export_refuses_to_upload_when_privacy_cannot_be_confirmed(tmp_path, monkeypatch):
    """A private=True export must fail rather than publish to a repo it cannot close."""
    module, backend, calls, seen = _visibility_backend(
        tmp_path, monkeypatch, "test_export_gguf_hub_upload_visibility_denied_backend"
    )

    def _denied(
        self,
        repo_id,
        private = None,
        repo_type = None,
    ):
        raise RuntimeError("403 Forbidden: write:repo_settings missing")

    monkeypatch.setattr(module.HfApi, "update_repo_settings", _denied)
    # repo_info reports a public repo, so the refusal stands.
    seen["repo_info_result"] = types.SimpleNamespace(private = False)

    success, message, output_path = backend.export_gguf(
        str(tmp_path / "export"),
        "Q4_K_M",
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "token",
        private = True,
    )

    assert success is False
    assert "could not be confirmed private" in message
    assert "upload_folder" not in calls
    assert output_path is not None
    assert Path(output_path, "model.Q4_K_M.gguf").is_file()


def test_gguf_hub_export_uploads_when_the_repo_is_already_private(tmp_path, monkeypatch):
    """A token without write:repo_settings still works against an already-private repo."""
    module, backend, calls, seen = _visibility_backend(
        tmp_path, monkeypatch, "test_export_gguf_hub_upload_visibility_ok_backend"
    )

    def _denied(
        self,
        repo_id,
        private = None,
        repo_type = None,
    ):
        raise RuntimeError("403 Forbidden: write:repo_settings missing")

    monkeypatch.setattr(module.HfApi, "update_repo_settings", _denied)
    seen["repo_info_result"] = types.SimpleNamespace(private = True)

    success, message, _path = backend.export_gguf(
        str(tmp_path / "export"),
        "Q4_K_M",
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "token",
        private = True,
    )

    assert success is True, message
    assert seen["uploaded"] == ["model.Q4_K_M.gguf"]


def test_gguf_hub_export_leaves_an_existing_private_repo_alone(tmp_path, monkeypatch):
    """private=False only means "do not create it private", never "publish this repo"."""
    _module, backend, calls, seen = _visibility_backend(
        tmp_path, monkeypatch, "test_export_gguf_hub_upload_visibility_public_backend"
    )

    success, message, _path = backend.export_gguf(
        str(tmp_path / "export"),
        "Q4_K_M",
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "token",
        private = False,
    )

    assert success is True, message
    assert "update_repo_settings" not in calls
    assert "visibility" not in seen


def test_gguf_hub_export_survives_a_model_card_failure(tmp_path, monkeypatch):
    """The GGUFs are already committed by then, so losing the card must not fail the export."""
    module, backend, calls, seen = _visibility_backend(
        tmp_path, monkeypatch, "test_export_gguf_hub_upload_card_fails_backend"
    )

    def _boom(
        self,
        repo_id,
        token = None,
        commit_message = None,
    ):
        raise RuntimeError("connection refused: api/validate-yaml")

    monkeypatch.setattr(module.ModelCard, "push_to_hub", _boom)

    success, message, output_path = backend.export_gguf(
        str(tmp_path / "export"),
        "Q4_K_M",
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "token",
        private = False,
    )

    assert success is True, message
    assert seen["uploaded"] == ["model.Q4_K_M.gguf"]
    assert output_path is not None


def test_gguf_hub_export_reports_the_local_path_when_the_upload_fails(tmp_path, monkeypatch):
    """A failed upload must not read as "re-run the hour-long conversion"."""
    module, backend, calls, seen = _visibility_backend(
        tmp_path, monkeypatch, "test_export_gguf_hub_upload_upload_fails_backend"
    )

    def _boom(
        self,
        folder_path,
        repo_id,
        repo_type,
        allow_patterns = None,
        ignore_patterns = None,
    ):
        raise RuntimeError("504 Gateway Timeout")

    monkeypatch.setattr(module.HfApi, "upload_folder", _boom)

    success, message, output_path = backend.export_gguf(
        str(tmp_path / "export"),
        "Q4_K_M",
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "token",
        private = False,
    )

    assert success is False
    assert "504 Gateway Timeout" in message
    assert "saved to" in message
    assert output_path is not None
    assert Path(output_path, "model.Q4_K_M.gguf").is_file()


def test_gguf_hub_export_fails_when_only_appledouble_companions_were_produced(
    tmp_path, monkeypatch
):
    """A run that produced only Finder metadata has produced nothing publishable."""
    save_dir = tmp_path / "export"
    save_dir.mkdir(parents = True)
    # An earlier export of a different model, which the directory-wide gate would pass on.
    (save_dir / "some-other-model.Q8_0.gguf").write_bytes(b"GGUF")

    _module, backend, calls, seen = _visibility_backend(
        tmp_path,
        monkeypatch,
        "test_export_gguf_hub_upload_only_appledouble_backend",
        gguf_names = ("._model.Q4_K_M.gguf",),
    )

    success, message, _path = backend.export_gguf(
        str(save_dir),
        "Q4_K_M",
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "token",
        private = False,
    )

    assert success is False
    assert "AppleDouble" in message
    assert "upload_folder" not in calls
    assert (save_dir / "some-other-model.Q8_0.gguf").is_file()


def test_gguf_hub_export_uses_the_canonical_repo_id_the_hub_returns(tmp_path, monkeypatch):
    """The card must use the Hub-resolved repo id, else it advertises `llama-cli -hf model-gguf`."""
    _install_export_backend_stubs(monkeypatch)
    export_module = _load_module(
        "test_export_gguf_hub_upload_canonical_backend", "core/export/export.py", monkeypatch
    )

    calls: list[str] = []
    seen: dict = {}

    class _CanonicalRepoUrl(str):
        repo_id = "resolved-owner/model-gguf"

    class Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            output = Path(f"{model_save_path}_gguf")
            output.mkdir(parents = True)
            (output / "model.Q4_K_M.gguf").write_bytes(b"GGUF")

        def push_to_hub_gguf(self, *args, **kwargs):
            calls.append("push_to_hub_gguf")

    hf_api, model_card = _hub_doubles(calls, seen)

    def _create_repo(
        self,
        repo_id,
        private = False,
        exist_ok = False,
    ):
        calls.append("create_repo")
        seen["repo"] = {"repo_id": repo_id, "private": private, "exist_ok": exist_ok}
        return _CanonicalRepoUrl("https://huggingface.co/resolved-owner/model-gguf")

    hf_api.create_repo = _create_repo
    monkeypatch.setattr(export_module, "HfApi", hf_api)
    monkeypatch.setattr(export_module, "ModelCard", model_card)
    monkeypatch.setattr(export_module, "resolve_export_write_dir", lambda value: Path(value))

    backend = export_module.ExportBackend.__new__(export_module.ExportBackend)
    backend.current_model = Model()
    backend.current_tokenizer = object()
    backend.current_checkpoint = None

    success, message, _path = backend.export_gguf(
        str(tmp_path / "export"),
        "Q4_K_M",
        push_to_hub = True,
        repo_id = "model-gguf",
        hf_token = "token",
        private = False,
    )

    assert success is True, message
    assert seen["repo"]["repo_id"] == "model-gguf"
    assert seen["card_repo"] == "resolved-owner/model-gguf"
    assert "# model-gguf : GGUF" in seen["card"]
    assert "llama-cli -hf resolved-owner/model-gguf --jinja" in seen["card"]
    assert "llama-mtmd-cli -hf resolved-owner/model-gguf --jinja" in seen["card"]


def test_gguf_hub_export_reads_config_from_the_directory_the_exporter_reports(
    tmp_path, monkeypatch
):
    """The merged checkpoint is not always under our temp root; unsloth reports where it went."""
    _install_export_backend_stubs(monkeypatch)
    export_module = _load_module(
        "test_export_gguf_hub_upload_reported_dir_backend", "core/export/export.py", monkeypatch
    )

    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    (elsewhere / "config.json").write_bytes(b'{"model_type": "qwen2"}')

    calls: list[str] = []
    seen: dict = {}

    class Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            output = Path(f"{model_save_path}_gguf")
            output.mkdir(parents = True)
            gguf = output / "model.Q4_K_M.gguf"
            gguf.write_bytes(b"GGUF")
            # Deliberately no config.json under model_save_path.
            return {"gguf_files": [str(gguf)], "save_directory": str(elsewhere)}

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
        private = False,
    )

    assert success is True, message
    assert seen["config.json"] == b'{"model_type": "qwen2"}'
    # Still not written into the export folder: _is_model_dir would read it as a checkpoint.
    assert not Path(output_path, "config.json").exists()


def test_gguf_hub_export_uploads_a_modelfile_it_could_not_place_locally(tmp_path, monkeypatch):
    """A read-only destination must not silently drop an artifact this run produced."""
    _install_export_backend_stubs(monkeypatch)
    export_module = _load_module(
        "test_export_gguf_hub_upload_modelfile_move_fails_backend",
        "core/export/export.py",
        monkeypatch,
    )

    calls: list[str] = []
    seen: dict = {}

    class Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            output = Path(f"{model_save_path}_gguf")
            output.mkdir(parents = True)
            (output / "model.Q4_K_M.gguf").write_bytes(b"GGUF")
            # write_bytes, not write_text: text mode turns "\n" into "\r\n" on Windows.
            (output / "Modelfile").write_bytes(b"FROM ./model.Q4_K_M.gguf\n")

        def push_to_hub_gguf(self, *args, **kwargs):
            calls.append("push_to_hub_gguf")

    hf_api, model_card = _hub_doubles(calls, seen)
    monkeypatch.setattr(export_module, "HfApi", hf_api)
    monkeypatch.setattr(export_module, "ModelCard", model_card)
    monkeypatch.setattr(export_module, "resolve_export_write_dir", lambda value: Path(value))

    real_move = export_module.shutil.move

    def _refuse_the_modelfile(src, dst, *args, **kwargs):
        if Path(str(dst)).name == "Modelfile":
            raise OSError("read-only file system")
        return real_move(src, dst, *args, **kwargs)

    monkeypatch.setattr(export_module.shutil, "move", _refuse_the_modelfile)

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
        private = False,
    )

    assert success is True, message
    assert not Path(output_path, "Modelfile").exists()
    assert "Modelfile" not in seen["uploaded"]
    assert seen["Modelfile"] == b"FROM ./model.Q4_K_M.gguf\n"


def test_push_only_gguf_export_still_delegates_to_push_to_hub_gguf(tmp_path, monkeypatch):
    """Nothing built to upload, so push_to_hub_gguf stands and must get the token exactly once."""
    _install_export_backend_stubs(monkeypatch)
    export_module = _load_module(
        "test_export_gguf_hub_upload_push_only_backend", "core/export/export.py", monkeypatch
    )

    calls: list[str] = []
    seen: dict = {}

    class Model:
        def save_pretrained_gguf(
            self,
            model_save_path,
            tokenizer,
            quantization_method,
            imatrix_file = None,
            token = None,
            gguf_shard_size = None,
        ):
            calls.append("save_pretrained_gguf")

        def push_to_hub_gguf(
            self,
            repo_id,
            tokenizer,
            quantization_method = None,
            token = None,
            private = None,
            imatrix_file = None,
            gguf_shard_size = None,
        ):
            calls.append("push_to_hub_gguf")
            seen["push"] = {
                "repo_id": repo_id,
                "quantization_method": quantization_method,
                "token": token,
                "private": private,
                "imatrix_file": imatrix_file,
                "gguf_shard_size": gguf_shard_size,
            }

    hf_api, model_card = _hub_doubles(calls, seen)
    monkeypatch.setattr(export_module, "HfApi", hf_api)
    monkeypatch.setattr(export_module, "ModelCard", model_card)

    backend = export_module.ExportBackend.__new__(export_module.ExportBackend)
    backend.current_model = Model()
    backend.current_tokenizer = object()
    backend.current_checkpoint = None

    success, message, output_path = backend.export_gguf(
        "",
        "iq2_xxs",
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "token",
        private = True,
        imatrix_file = True,
        gguf_shard_size = "512MB",
    )

    assert success is True, message
    assert output_path is None
    assert "save_pretrained_gguf" not in calls
    assert "upload_folder" not in calls
    assert calls == ["push_to_hub_gguf"]
    assert seen["push"] == {
        "repo_id": "owner/model",
        "quantization_method": "iq2_xxs",
        "token": "token",
        "private": True,
        "imatrix_file": True,
        "gguf_shard_size": "512MB",
    }


@pytest.mark.parametrize("is_vision", [True, False])
def test_gguf_hub_export_falls_back_to_studios_own_vlm_detection(tmp_path, monkeypatch, is_vision):
    """The MLX binding returns None, so without a fallback vision exports publish untagged."""
    _install_export_backend_stubs(monkeypatch)
    export_module = _load_module(
        f"test_export_gguf_hub_upload_vlm_fallback_{is_vision}_backend",
        "core/export/export.py",
        monkeypatch,
    )

    calls: list[str] = []
    seen: dict = {}

    class Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            output = Path(f"{model_save_path}_gguf")
            output.mkdir(parents = True)
            (output / "model.Q4_K_M.gguf").write_bytes(b"GGUF")
            return None  # what the MLX exporter returns

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
    backend.is_vision = is_vision

    success, message, _path = backend.export_gguf(
        str(tmp_path / "export"),
        "Q4_K_M",
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "token",
        private = False,
    )

    assert success is True, message
    assert ("- vision-language-model" in seen["card"]) is is_vision


def test_gguf_hub_export_trusts_the_exporter_over_studios_guess(tmp_path, monkeypatch):
    """A CUDA exporter that reports is_vlm=False wins, even if Studio loaded it as vision."""
    _install_export_backend_stubs(monkeypatch)
    export_module = _load_module(
        "test_export_gguf_hub_upload_vlm_reported_wins_backend",
        "core/export/export.py",
        monkeypatch,
    )

    calls: list[str] = []
    seen: dict = {}

    class Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            output = Path(f"{model_save_path}_gguf")
            output.mkdir(parents = True)
            gguf = output / "model.Q4_K_M.gguf"
            gguf.write_bytes(b"GGUF")
            return {"gguf_files": [str(gguf)], "is_vlm": False}

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
    backend.is_vision = True

    success, message, _path = backend.export_gguf(
        str(tmp_path / "export"),
        "Q4_K_M",
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "token",
        private = False,
    )

    assert success is True, message
    assert "- vision-language-model" not in seen["card"]
