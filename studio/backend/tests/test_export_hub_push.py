# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import weakref

import functools
import json
import os
import shutil
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import pytest
from huggingface_hub import constants as hf_constants
from huggingface_hub import get_token as hf_get_token


_HELPERS_SPEC = importlib.util.spec_from_file_location(
    "export_hub_push_helpers",
    Path(__file__).with_name("test_export_gguf_hub_upload.py"),
)
assert _HELPERS_SPEC is not None and _HELPERS_SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_HELPERS_SPEC)
_HELPERS_SPEC.loader.exec_module(_HELPERS)


class _Config:
    _name_or_path = "unsloth/Qwen2.5-0.5B-Instruct"
    model_type = "qwen2"


class _Tokenizer:
    def save_pretrained(self, save_directory):
        Path(save_directory, "tokenizer.json").write_bytes(b"{}")


class _Model:
    config = _Config()

    def __init__(self):
        self.merges = []

    def save_pretrained(self, save_directory):
        Path(save_directory, "model.safetensors").write_bytes(b"weights")

    def save_pretrained_merged(
        self,
        save_directory,
        tokenizer,
        save_method = None,
        token = None,
    ):
        self.merges.append(save_method)
        suffix = "-torchao-fp8" if save_method == "torchao_fp8" else ""
        output = Path(f"{save_directory}{suffix}")
        output.mkdir(parents = True, exist_ok = True)
        (output / "model.safetensors").write_bytes(b"weights")

    def push_to_hub_merged(self, *args, **kwargs):
        self.merges.append("push_to_hub_merged")


def _non_mlx_backend(monkeypatch, name, calls, seen):
    _HELPERS._install_export_backend_stubs(monkeypatch)
    unsloth = sys.modules["unsloth"]
    monkeypatch.setattr(unsloth, "_IS_MLX", False)

    unsloth_save = types.ModuleType("unsloth.save")
    unsloth_save._normalize_torchao_method = lambda method: (
        ("fp8", "torchao-fp8") if method == "torchao_fp8" else None
    )
    monkeypatch.setattr(unsloth, "save", unsloth_save, raising = False)
    monkeypatch.setitem(sys.modules, "unsloth.save", unsloth_save)

    peft = types.ModuleType("peft")
    peft.PeftModel = object
    peft.PeftModelForCausalLM = object
    transformers = types.ModuleType("transformers")
    transformers.__path__ = []
    modeling_utils = types.ModuleType("transformers.modeling_utils")
    modeling_utils.PushToHubMixin = type("PushToHubMixin", (), {})
    transformers.modeling_utils = modeling_utils
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_utils)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    export_module = _HELPERS._load_module(name, "core/export/export.py", monkeypatch)
    _HELPERS._patch_hub(monkeypatch, export_module, calls, seen)

    backend = export_module.ExportBackend.__new__(export_module.ExportBackend)
    backend.current_model = _Model()
    backend.current_tokenizer = _Tokenizer()
    backend.current_checkpoint = None
    backend.is_peft = False
    backend._audio_type = None
    return backend


def _expected_calls(private):
    visibility = ["update_repo_settings"] if private else []
    return ["create_repo", *visibility, "model_card", "upload_folder"]


def _expected_merged_calls(private):
    visibility = ["update_repo_settings"] if private else []
    return ["create_repo", *visibility, "upload_folder", "model_card"]


_OLD_NS = 1_600_000_000_000_000_000


def _age(directory):
    """Leftovers of an earlier export: written long before this one (a fresh write is staged)."""
    for path in Path(directory).rglob("*"):
        if path.is_file():
            os.utime(path, ns = (_OLD_NS, _OLD_NS))


@pytest.mark.parametrize("private", [True, False])
def test_base_export_push_creates_the_repo_without_push_to_hub_mixin(
    tmp_path, monkeypatch, private
):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_base_backend", calls, seen)

    success, message, output_path = backend.export_base_model(
        str(tmp_path / "export"),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
        private = private,
    )

    assert success is True, message
    assert seen["token"] == "hf_fake"
    assert seen["repo"] == {"repo_id": "model", "private": private, "exist_ok": True}
    assert calls == _expected_calls(private)
    assert seen["card_repo"] == "owner/model"
    assert seen["folder"] == output_path
    assert "model.safetensors" in seen["uploaded"]


@pytest.mark.parametrize("private", [True, False])
def test_merged_torchao_export_push_creates_the_repo_without_push_to_hub_mixin(
    tmp_path, monkeypatch, private
):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)

    success, message, output_path = backend.export_merged_model(
        str(tmp_path / "export"),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
        private = private,
        compressed_method = "torchao_fp8",
    )

    assert success is True, message
    assert backend.current_model.merges == ["torchao_fp8"]
    assert output_path == str(Path(f"{tmp_path / 'export'}-torchao-fp8").resolve())
    assert seen["token"] == "hf_fake"
    assert seen["repo"] == {"repo_id": "model", "private": private, "exist_ok": True}
    assert calls == _expected_merged_calls(private)
    assert seen["card_repo"] == "owner/model"
    assert seen["folder"] == output_path
    assert "model.safetensors" in seen["uploaded"]


@pytest.mark.parametrize(
    "format_type, save_method",
    [("16-bit (FP16)", "merged_16bit"), ("4-bit (FP4)", "merged_4bit_forced")],
)
@pytest.mark.parametrize("private", [True, False])
def test_merged_export_push_uploads_the_saved_folder_instead_of_merging_again(
    tmp_path, monkeypatch, format_type, save_method, private
):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)

    success, message, output_path = backend.export_merged_model(
        str(tmp_path / "export"),
        format_type = format_type,
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
        private = private,
    )

    assert success is True, message
    assert backend.current_model.merges == [save_method]
    assert output_path == str((tmp_path / "export").resolve())
    assert seen["repo"] == {"repo_id": "model", "private": private, "exist_ok": True}
    assert calls == _expected_merged_calls(private)
    assert f"# Uploaded finetuned {format_type} model" in seen["card"]
    assert "base_model: unsloth/Qwen2.5-0.5B-Instruct" in seen["card"]
    assert seen["folder"] == output_path
    assert seen["uploaded"] == ["model.safetensors"]


def test_merged_export_push_to_a_reused_folder_does_not_upload_its_leftovers(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    leftover = export_dir / "adapter_config.json"
    leftover.write_text("{}")
    os.utime(leftover, ns = (1_600_000_000_000_000_000, 1_600_000_000_123_456_789))
    # The value the filesystem kept (NTFS stores 100 ns ticks).
    kept_ns = leftover.stat().st_mtime_ns

    success, message, output_path = backend.export_merged_model(
        str(export_dir),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
    )

    assert success is True, message
    assert output_path == str(export_dir.resolve())
    # One merge: the reused folder is uploaded as is, limited to the files this save wrote.
    assert backend.current_model.merges == ["merged_16bit"]
    assert seen["folder"] == output_path
    assert seen["uploaded"] == ["model.safetensors"]
    assert leftover.read_text() == "{}"
    assert leftover.stat().st_mtime_ns == kept_ns


def test_merged_export_push_to_a_reused_folder_uploads_overwritten_files(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)
    export_dir = tmp_path / "export"
    (export_dir / "eagle3").mkdir(parents = True)
    (export_dir / "model.safetensors").write_bytes(b"stale")
    (export_dir / "eagle3" / "old.json").write_text("{}")
    (export_dir / "weird[v2].txt").write_text("leftover")
    _age(export_dir)
    save = backend.current_model.save_pretrained_merged

    def save_with_extras(
        save_directory,
        tokenizer,
        save_method = None,
        token = None,
    ):
        save(save_directory, tokenizer, save_method = save_method, token = token)
        shutil.copy2(export_dir / "eagle3" / "old.json", Path(save_directory, "copied.json"))
        Path(save_directory, "eagle3", "config.json").write_text("{}")
        Path(save_directory, "model[v2].bin").write_bytes(b"w")

    backend.current_model.save_pretrained_merged = save_with_extras

    success, message, output_path = backend.export_merged_model(
        str(export_dir), push_to_hub = True, repo_id = "model", hf_token = "hf_fake"
    )

    assert success is True, message
    assert backend.current_model.merges == ["merged_16bit"]
    assert seen["folder"] == output_path
    assert seen["uploaded"] == [
        "copied.json",
        "eagle3/config.json",
        "model.safetensors",
        "model[v2].bin",
    ]
    assert Path(output_path, "model.safetensors").read_bytes() == b"weights"


@pytest.mark.parametrize("why", ["hard_link", "base_model_dir"])
def test_merged_export_push_stages_a_clean_save_when_the_folder_cannot_be_tracked(
    tmp_path, monkeypatch, why
):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    leftover = export_dir / "adapter_config.json"
    leftover.write_text("{}")
    if why == "hard_link":
        os.link(leftover, tmp_path / "elsewhere.json")
    else:
        config = types.SimpleNamespace(_name_or_path = str(export_dir), model_type = "qwen2")
        monkeypatch.setattr(backend.current_model, "config", config, raising = False)
    before = leftover.stat().st_mtime_ns

    success, message, output_path = backend.export_merged_model(
        str(export_dir), push_to_hub = True, repo_id = "model", hf_token = "hf_fake"
    )

    assert success is True, message
    assert backend.current_model.merges == ["merged_16bit", "merged_16bit"]
    assert seen["folder"] != output_path
    assert seen["uploaded"] == ["model.safetensors"]
    assert leftover.stat().st_mtime_ns == before


def test_reused_folder_tracking_leaves_leftover_times_alone_when_the_save_fails(
    tmp_path, monkeypatch
):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    leftover = export_dir / "adapter_config.json"
    leftover.write_text("{}")
    os.utime(leftover, ns = (1_600_000_000_000_000_000, 1_600_000_000_000_000_007))
    kept_ns = leftover.stat().st_mtime_ns  # NTFS rounds to 100 ns ticks

    def fail(*args, **kwargs):
        raise OSError(28, "No space left on device")

    backend.current_model.save_pretrained_merged = fail
    success, message, _ = backend.export_merged_model(
        str(export_dir), push_to_hub = True, repo_id = "model", hf_token = "hf_fake"
    )
    assert success is False and "No space left on device" in message
    assert leftover.stat().st_mtime_ns == kept_ns
    assert calls == []


def test_merged_export_push_into_the_checkpoint_folder_stages_a_clean_save(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)
    export_dir = tmp_path / "checkpoint"
    export_dir.mkdir()
    (export_dir / "adapter_config.json").write_text("{}")
    backend.current_checkpoint = str(export_dir)

    success, message, output_path = backend.export_merged_model(
        str(export_dir / "merged"), push_to_hub = True, repo_id = "model", hf_token = "hf_fake"
    )
    assert success is True, message
    assert backend.current_model.merges == ["merged_16bit"]  # fresh subfolder, no staging

    (export_dir / "merged" / "leftover.json").write_text("{}")
    success, message, output_path = backend.export_merged_model(
        str(export_dir / "merged"), push_to_hub = True, repo_id = "model", hf_token = "hf_fake"
    )
    assert success is True, message
    assert backend.current_model.merges == ["merged_16bit"] * 3
    assert seen["folder"] != output_path
    assert seen["uploaded"] == ["model.safetensors"]


def test_whisper_merged_export_is_a_16bit_merge(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)
    backend._audio_type = "whisper"
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "adapter_config.json").write_text("{}")
    _age(export_dir)

    success, message, _ = backend.export_merged_model(
        str(export_dir),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
    )

    assert success is True, message
    assert backend.current_model.merges == ["merged_16bit"]
    assert seen["uploaded"] == ["model.safetensors"]


@pytest.mark.parametrize("roomier", ["temp", "export", "export_but_unwritable"])
def test_merged_export_push_stages_the_clean_save_where_there_is_room(
    tmp_path, monkeypatch, roomier
):
    import shutil

    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "adapter_config.json").write_text("{}")
    export_parent = export_dir.resolve().parent
    # Only an untracked reused folder is staged.
    backend._export_dir_overlaps_checkpoint = lambda directory: True

    def fake_disk_usage(path):
        under_export = Path(path).resolve() == export_parent
        roomy = under_export if roomier != "temp" else not under_export
        return types.SimpleNamespace(total = 1 << 40, used = 0, free = (1 << 40) if roomy else 1)

    monkeypatch.setattr(shutil, "disk_usage", fake_disk_usage)

    if roomier == "export_but_unwritable":
        real_temporary_directory = tempfile.TemporaryDirectory

        def refusing_temporary_directory(
            *args,
            dir = None,
            **kwargs,
        ):
            if dir is not None and Path(dir).resolve() == export_parent:
                raise PermissionError(13, "Permission denied", str(dir))
            return real_temporary_directory(*args, dir = dir, **kwargs)

        monkeypatch.setattr(tempfile, "TemporaryDirectory", refusing_temporary_directory)

    success, message, _ = backend.export_merged_model(
        str(export_dir),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
    )

    assert success is True, message
    staging_parent = Path(seen["folder"]).resolve().parent
    if roomier == "export":
        assert staging_parent == export_parent
    else:
        assert staging_parent != export_parent


def test_merged_export_push_treats_a_folder_of_finder_metadata_as_fresh(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "._model.safetensors").write_bytes(b"\x00\x05\x16\x07rsrc")

    success, message, output_path = backend.export_merged_model(
        str(export_dir),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
    )

    assert success is True, message
    assert backend.current_model.merges == ["merged_16bit"]
    assert seen["folder"] == output_path
    assert seen["uploaded"] == ["model.safetensors"]


def test_merged_export_push_keeps_the_card_of_an_existing_repo(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {"existing_files": {"README.md": True}}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)

    success, message, _ = backend.export_merged_model(
        str(tmp_path / "export"),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
    )

    assert success is True, message
    assert calls == ["create_repo", "upload_folder"]
    assert "card" not in seen
    assert seen["uploaded"] == ["model.safetensors"]


def test_merged_export_push_still_uploads_when_the_card_fails(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {"card_error": RuntimeError("validate-yaml unreachable")}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)

    success, message, output_path = backend.export_merged_model(
        str(tmp_path / "export"),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
    )

    assert success is True, message
    assert seen["folder"] == output_path
    assert seen["uploaded"] == ["model.safetensors"]


# a stale weight file makes the folder look reused, so emptiness shows only on staging
@pytest.mark.parametrize("stale_weights", [False, True])
def test_merged_export_push_merges_again_when_the_save_left_no_weights(
    tmp_path, monkeypatch, stale_weights
):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)
    export_dir = tmp_path / "export"
    if stale_weights:
        export_dir.mkdir()
        (export_dir / "model.safetensors").write_bytes(b"stale")

    def save_nothing(
        save_directory,
        tokenizer,
        save_method = None,
        token = None,
    ):
        Path(save_directory).mkdir(parents = True, exist_ok = True)

    backend.current_model.save_pretrained_merged = save_nothing

    success, message, _ = backend.export_merged_model(
        str(export_dir),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
    )

    assert success is True, message
    assert backend.current_model.merges == ["push_to_hub_merged"]
    assert calls == []


def test_merged_export_push_card_does_not_name_a_local_base_model(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)
    backend.current_model.config = types.SimpleNamespace(
        _name_or_path = str(tmp_path), model_type = "qwen2"
    )

    success, message, _ = backend.export_merged_model(
        str(tmp_path / "export"),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
    )

    assert success is True, message
    assert "base_model: owner/model" in seen["card"]
    assert str(tmp_path) not in seen["card"]


_MLX_MERGED_FILES = ["README.md", "model.safetensors", "model.safetensors.index.json"]


class _MlxMergeModel(_Model):
    def __init__(self, hf_api):
        super().__init__()
        self.hf_api = hf_api

    def save_pretrained_merged(
        self,
        save_directory,
        tokenizer,
        save_method = None,
        token = None,
    ):
        self.merges.append(save_method)
        output = Path(save_directory)
        output.mkdir(parents = True, exist_ok = True)
        for name in _MLX_MERGED_FILES:
            (output / name).write_bytes(b"merged")

    def push_to_hub_merged(
        self,
        repo_id,
        tokenizer,
        save_directory = None,
        token = None,
        private = None,
    ):
        api = self.hf_api(token = token)
        api.create_repo(repo_id, private = private, exist_ok = True)
        if private is not None:
            api.update_repo_settings(repo_id = repo_id, private = private, repo_type = "model")
        api.upload_folder(folder_path = str(save_directory), repo_id = repo_id, repo_type = "model")


# export -> (ExportBackend method, is_peft, extra kwargs, save_method)
_MLX_EXPORTS = {
    "merged": ("export_merged_model", True, {}, "merged_16bit"),
    "merged_4bit": ("export_merged_model", True, {"format_type": "4-bit (FP4)"}, "merged_4bit"),
    "base": ("export_base_model", False, {}, "merged_16bit"),
}


def _mlx_backend(monkeypatch, name, calls, seen, export):
    method, is_peft, kwargs, _save_method = _MLX_EXPORTS[export]
    backend = _non_mlx_backend(monkeypatch, name, calls, seen)
    export_module = sys.modules[type(backend).__module__]
    monkeypatch.setattr(export_module, "_IS_MLX", True)
    monkeypatch.setattr(export_module, "_export_runtime_available", lambda: True)
    backend.current_model = _MlxMergeModel(export_module.HfApi)
    backend.is_peft = is_peft
    return backend, functools.partial(getattr(backend, method), **kwargs)


@pytest.mark.parametrize("export", _MLX_EXPORTS)
def test_mlx_export_push_keeps_the_export_metadata_out_of_the_repo(tmp_path, monkeypatch, export):
    calls: list[str] = []
    seen: dict = {}
    backend, push = _mlx_backend(
        monkeypatch, "test_export_hub_push_mlx_fresh_backend", calls, seen, export
    )

    success, message, output_path = push(
        str(tmp_path / "export"),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
    )

    assert success is True, message
    assert seen["uploaded"] == _MLX_MERGED_FILES
    assert seen["upload_repo"] == "owner/model"
    assert Path(output_path, "export_metadata.json").is_file()
    assert backend.current_model.merges == [_MLX_EXPORTS[export][3]]
    assert seen["folder"] == output_path


@pytest.mark.parametrize("export", _MLX_EXPORTS)
def test_mlx_export_push_to_a_reused_folder_does_not_upload_its_leftovers(
    tmp_path, monkeypatch, export
):
    calls: list[str] = []
    seen: dict = {}
    backend, push = _mlx_backend(
        monkeypatch, "test_export_hub_push_mlx_reused_backend", calls, seen, export
    )
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "adapter_config.json").write_text("{}")
    (export_dir / "adapters.safetensors").write_bytes(b"lora")

    success, message, output_path = push(
        str(export_dir),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
    )

    assert success is True, message
    assert seen["uploaded"] == _MLX_MERGED_FILES
    assert output_path == str(export_dir.resolve())
    assert backend.current_model.merges == [_MLX_EXPORTS[export][3]] * 2
    # A REAP report an earlier export pushed to this repo is deleted in the same commit.
    assert seen["delete_patterns"] == ["reap_report.json"]
    assert seen["folder"] != output_path
    assert not Path(seen["folder"]).exists()


@pytest.mark.parametrize("export", _MLX_EXPORTS)
def test_mlx_export_staging_failure_leaves_the_hub_untouched(tmp_path, monkeypatch, export):
    calls: list[str] = []
    seen: dict = {}
    backend, push = _mlx_backend(
        monkeypatch, "test_export_hub_push_mlx_stage_failure", calls, seen, export
    )
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "old.gguf").write_bytes(b"GGUF")
    save = backend.current_model.save_pretrained_merged

    def fail_staging(directory, *args, **kwargs):
        save(directory, *args, **kwargs)
        if len(backend.current_model.merges) == 2:
            raise OSError(28, "No space left on device")

    monkeypatch.setattr(backend.current_model, "save_pretrained_merged", fail_staging)
    success, message, _path = push(
        str(export_dir), push_to_hub = True, repo_id = "model", hf_token = "hf_fake"
    )

    assert success is False
    assert "No space left on device" in message
    assert Path(export_dir, "model.safetensors").is_file()
    assert calls == []


@pytest.mark.parametrize("export", _MLX_EXPORTS)
@pytest.mark.parametrize("private", [True, False])
def test_mlx_export_push_never_makes_an_existing_private_repo_public(
    tmp_path, monkeypatch, private, export
):
    calls: list[str] = []
    seen: dict = {}
    _backend, push = _mlx_backend(
        monkeypatch, "test_export_hub_push_mlx_private_backend", calls, seen, export
    )

    success, message, _path = push(
        str(tmp_path / "export"),
        push_to_hub = True,
        repo_id = "owner/model",
        hf_token = "hf_fake",
        private = private,
    )

    assert success is True, message
    visibility = ["update_repo_settings"] if private else []
    assert calls == ["create_repo", *visibility, "upload_folder"]
    assert seen.get("visibility") == (
        {"repo_id": "owner/model", "private": True} if private else None
    )
    assert seen["uploaded"] == _MLX_MERGED_FILES


def test_base_export_push_keeps_the_export_metadata_out_of_the_repo(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_base_backend", calls, seen)

    success, message, output_path = backend.export_base_model(
        str(tmp_path / "export"),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
    )

    assert success is True, message
    assert Path(output_path, "export_metadata.json").is_file()
    assert seen["folder"] == output_path
    assert seen["uploaded"] == ["model.safetensors", "tokenizer.json"]


def test_base_export_push_to_a_reused_folder_does_not_upload_its_leftovers(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_base_backend", calls, seen)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "old.Q4_K_M.gguf").write_bytes(b"GGUF")
    (export_dir / "export_metadata.json").write_text("{}")
    _age(export_dir)

    success, message, output_path = backend.export_base_model(
        str(export_dir),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
    )

    assert success is True, message
    assert output_path == str(export_dir.resolve())
    assert Path(output_path, "model.safetensors").is_file()
    assert Path(output_path, "old.Q4_K_M.gguf").is_file()
    assert seen["folder"] == output_path
    assert seen["uploaded"] == ["model.safetensors", "tokenizer.json"]


def test_base_export_staging_failure_leaves_the_hub_untouched(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_stage_failure", calls, seen)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "old.gguf").write_bytes(b"GGUF")
    # Only an untracked reused folder is staged.
    backend._export_dir_overlaps_checkpoint = lambda directory: True
    save = backend.current_model.save_pretrained
    saved_to = []

    def fail_staging(directory):
        saved_to.append(directory)
        if len(saved_to) == 2:
            raise OSError(28, "No space left on device")
        save(directory)

    monkeypatch.setattr(backend.current_model, "save_pretrained", fail_staging)
    success, message, _output_path = backend.export_base_model(
        str(export_dir), push_to_hub = True, repo_id = "model", hf_token = "hf_fake"
    )

    assert success is False
    assert "No space left on device" in message
    assert len(saved_to) == 2
    assert Path(export_dir, "model.safetensors").is_file()
    assert not Path(saved_to[1]).exists()
    assert calls == []


class _LoraTokenizer(_Tokenizer):
    def __init__(self, calls):
        self.calls = calls

    def push_to_hub(
        self,
        repo_id,
        token = None,
        private = None,
    ):
        self.calls.append(f"tokenizer_push:{repo_id}")


class _LoraModel:
    config = _Config()
    peft_config: dict = {}

    def __init__(self, calls):
        self.calls = calls
        self.conversions = []

    def save_pretrained(self, save_directory):
        Path(save_directory, "adapter_model.safetensors").write_bytes(b"weights")

    def save_lora_adapters(self, save_directory):
        Path(save_directory, "adapters.safetensors").write_bytes(b"weights")

    def save_pretrained_gguf(
        self,
        save_directory,
        tokenizer,
        save_method = None,
        quantization_method = None,
        token = None,
    ):
        self.conversions.append(save_directory)
        output = Path(save_directory)
        output.mkdir(parents = True, exist_ok = True)
        (output / "adapter_config.json").write_text("{}")
        (output / "adapter_model.safetensors").write_bytes(b"lora")
        (output / f"model-lora-{quantization_method}.gguf").write_bytes(b"GGUF")

    def push_to_hub(
        self,
        repo_id,
        token = None,
        private = None,
    ):
        self.calls.append(f"model_push:{repo_id}")


# leg -> (gguf, is_mlx, repo contents); each leg opens the repo itself so it cannot go public
_LORA_LEGS = {
    "adapter": (
        False,
        False,
        ["model_card", "model_push:owner/model", "tokenizer_push:owner/model"],
    ),
    "gguf": (True, False, ["upload_folder"]),
    "mlx": (False, True, ["upload_folder"]),
}


def _lora_backend(monkeypatch, name, calls, seen, leg):
    gguf, is_mlx, uploads = _LORA_LEGS[leg]
    backend = _non_mlx_backend(monkeypatch, name, calls, seen)
    export_module = sys.modules[type(backend).__module__]
    monkeypatch.setattr(export_module, "_IS_MLX", is_mlx)
    monkeypatch.setattr(export_module, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(export_module, "_apply_wsl_sudo_patch", lambda: None)
    backend.current_model = _LoraModel(calls)
    backend.current_tokenizer = _LoraTokenizer(calls)
    backend.is_peft = True
    return export_module, backend, gguf, uploads


def _push_lora(backend, save_directory, gguf, private):
    return backend.export_lora_adapter(
        save_directory,
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
        private = private,
        gguf = gguf,
    )


@pytest.mark.parametrize("leg", list(_LORA_LEGS))
@pytest.mark.parametrize("private", [True, False])
def test_lora_export_push_makes_an_existing_repo_private_before_uploading(
    tmp_path, monkeypatch, leg, private
):
    calls: list[str] = []
    seen: dict = {}
    _module, backend, gguf, uploads = _lora_backend(
        monkeypatch, f"test_export_hub_push_lora_{leg}_backend", calls, seen, leg
    )

    success, message, _path = _push_lora(backend, str(tmp_path / "export"), gguf, private)

    assert success is True, message
    assert seen["token"] == "hf_fake"
    assert seen["repo"] == {"repo_id": "model", "private": private, "exist_ok": True}
    visibility = ["update_repo_settings"] if private else []
    assert calls == ["create_repo", *visibility, *uploads]
    if private:
        assert seen["visibility"] == {"repo_id": "owner/model", "private": True}
    else:
        assert "visibility" not in seen


@pytest.mark.parametrize("leg", list(_LORA_LEGS))
def test_lora_export_push_refuses_to_upload_when_privacy_cannot_be_confirmed(
    tmp_path, monkeypatch, leg
):
    calls: list[str] = []
    seen: dict = {}
    module, backend, gguf, _uploads = _lora_backend(
        monkeypatch, f"test_export_hub_push_lora_{leg}_denied_backend", calls, seen, leg
    )

    def _denied(
        self,
        repo_id,
        private = None,
        repo_type = None,
    ):
        raise RuntimeError("403 Forbidden: write:repo_settings missing")

    monkeypatch.setattr(module.HfApi, "update_repo_settings", _denied)
    seen["repo_info_result"] = types.SimpleNamespace(private = False)

    success, message, output_path = _push_lora(backend, str(tmp_path / "export"), gguf, True)

    assert success is False
    assert "could not be confirmed private" in message
    assert output_path is None
    assert calls == ["create_repo", "repo_info"]


@pytest.mark.parametrize("leg", list(_LORA_LEGS))
def test_lora_export_push_uploads_when_the_repo_is_already_private(tmp_path, monkeypatch, leg):
    calls: list[str] = []
    seen: dict = {}
    module, backend, gguf, uploads = _lora_backend(
        monkeypatch, f"test_export_hub_push_lora_{leg}_ok_backend", calls, seen, leg
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

    success, message, _path = _push_lora(backend, str(tmp_path / "export"), gguf, True)

    assert success is True, message
    assert calls == ["create_repo", "repo_info", *uploads]


def test_lora_adapter_push_writes_the_card_the_delegated_push_can_no_longer_write(
    tmp_path, monkeypatch
):
    """Opening the repo first makes the delegated push skip MODEL_CARD, so this path writes it."""
    calls: list[str] = []
    seen: dict = {}
    _module, backend, gguf, _uploads = _lora_backend(
        monkeypatch, "test_export_hub_push_lora_card_backend", calls, seen, "adapter"
    )

    success, message, _path = _push_lora(backend, str(tmp_path / "export"), gguf, True)

    assert success is True, message
    assert seen["card_repo"] == "owner/model"
    assert "base_model: unsloth/Qwen2.5-0.5B-Instruct" in seen["card"]
    # unsloth not duplicated: the template already carries it
    tags = seen["card"].split("tags:", 1)[1].split("license:", 1)[0]
    assert sorted(line.strip("- ").strip() for line in tags.strip().splitlines()) == [
        "qwen2",
        "text-generation-inference",
        "transformers",
        "trl",
        "unsloth",
    ]
    # empty `method` leaves a double space from the template; markdown collapses it
    assert "# Uploaded finetuned  model" in seen["card"]
    assert calls.index("model_card") < calls.index("model_push:owner/model")


def test_lora_adapter_push_keeps_the_card_of_an_existing_repo(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {"existing_files": {"README.md": True}}
    _module, backend, gguf, _uploads = _lora_backend(
        monkeypatch, "test_export_hub_push_lora_keepcard_backend", calls, seen, "adapter"
    )

    success, message, _path = _push_lora(backend, str(tmp_path / "export"), gguf, True)

    assert success is True, message
    assert "card" not in seen
    assert calls == [
        "create_repo",
        "update_repo_settings",
        "model_push:owner/model",
        "tokenizer_push:owner/model",
    ]


def test_lora_mlx_push_that_cannot_serialise_leaves_no_repo_behind(tmp_path, monkeypatch):
    """The repo is opened after serialisation, so a failed save cannot orphan one."""
    calls: list[str] = []
    seen: dict = {}
    _module, backend, gguf, _uploads = _lora_backend(
        monkeypatch, "test_export_hub_push_lora_mlx_orphan_backend", calls, seen, "mlx"
    )

    saves = {"n": 0}
    real_save = backend.current_model.save_lora_adapters

    def _fail_the_upload_save(save_directory):
        saves["n"] += 1
        if saves["n"] >= 2:  # 1st is the local save, 2nd is the temp dir to upload
            raise RuntimeError("MLX serialization failed")
        real_save(save_directory)

    backend.current_model.save_lora_adapters = _fail_the_upload_save

    success, message, _path = _push_lora(backend, str(tmp_path / "export"), gguf, True)

    assert success is False
    assert "MLX serialization failed" in message
    assert calls == []
    assert "repo" not in seen


_LORA_GGUF_FILES = ["adapter_config.json", "adapter_model.safetensors", "model-lora-q8_0.gguf"]


def _gguf_backend(monkeypatch, name, calls, seen):
    _module, backend, _gguf, _uploads = _lora_backend(monkeypatch, name, calls, seen, "gguf")
    return backend


def test_lora_gguf_export_push_uploads_the_saved_folder(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {}
    backend = _gguf_backend(
        monkeypatch, "test_export_hub_push_lora_gguf_fresh_backend", calls, seen
    )
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "._old.gguf").write_bytes(b"\x00\x05\x16\x07rsrc")

    success, message, output_path = _push_lora(backend, str(export_dir), True, False)

    assert success is True, message
    assert backend.current_model.conversions == [str(export_dir)]
    assert calls == ["create_repo", "upload_folder"]
    assert seen["folder"] == output_path
    assert seen["uploaded"] == _LORA_GGUF_FILES


def test_lora_gguf_export_push_to_a_reused_folder_does_not_upload_its_leftovers(
    tmp_path, monkeypatch
):
    calls: list[str] = []
    seen: dict = {}
    backend = _gguf_backend(
        monkeypatch, "test_export_hub_push_lora_gguf_reused_backend", calls, seen
    )
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "model.Q4_K_M.gguf").write_bytes(b"GGUF")
    (export_dir / "Modelfile").write_text("FROM model.Q4_K_M.gguf")
    (export_dir / "export_metadata.json").write_text('{"base_model": "/home/me/models/base"}')

    success, message, output_path = _push_lora(backend, str(export_dir), True, False)

    assert success is True, message
    assert output_path == str(export_dir.resolve())
    assert Path(output_path, "model-lora-q8_0.gguf").is_file()
    assert len(backend.current_model.conversions) == 2
    assert seen["folder"] != output_path
    assert Path(seen["folder"]).name == export_dir.name
    assert not Path(seen["folder"]).exists()
    assert seen["uploaded"] == _LORA_GGUF_FILES


def _reap_pruned(backend, monkeypatch):
    # As apply_reap leaves it: the pruning key and the kept experts on the loaded model. The stub
    # core.export package of this harness gets the real reap module the export imports.
    spec = importlib.util.spec_from_file_location(
        "core.export.reap", Path(__file__).resolve().parents[1] / "core/export/reap.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "core.export.reap", module)
    backend._reap_key = (0.5, 512, None)
    backend._reap_model = weakref.ref(backend.current_model)
    backend.current_model._unsloth_reap_report = {"model.layers.0.mlp": [0, 2]}
    return backend


@pytest.mark.parametrize("reused", [False, True])
def test_merged_push_of_a_reap_pruned_model_uploads_its_report(tmp_path, monkeypatch, reused):
    calls: list[str] = []
    seen: dict = {}
    backend = _reap_pruned(
        _non_mlx_backend(monkeypatch, "test_export_hub_push_reap_merged", calls, seen), monkeypatch
    )
    export_dir = tmp_path / "export"
    if reused:
        export_dir.mkdir()
        (export_dir / "adapter_config.json").write_text("{}")
    success, message, output_path = backend.export_merged_model(
        str(export_dir), push_to_hub = True, repo_id = "model", hf_token = "hf_fake"
    )
    assert success is True, message
    # Written before the upload, into whichever folder was uploaded.
    assert "reap_report.json" in seen["uploaded"]
    assert Path(output_path, "reap_report.json").is_file()


def test_merged_push_into_a_folder_with_a_stale_report_uploads_none(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_reap_stale", calls, seen)
    _reap_pruned(backend, monkeypatch)
    backend._reap_key = None  # not pruned this time
    export_dir = tmp_path / "export-torchao-fp8"
    export_dir.mkdir(parents = True)
    (export_dir / "reap_report.json").write_text('{"kept_experts": {}}')
    success, message, output_path = backend.export_merged_model(
        str(tmp_path / "export"),
        compressed_method = "torchao_fp8",
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
    )
    assert success is True, message
    assert seen["folder"] == output_path
    assert "reap_report.json" not in seen["uploaded"]
    assert not Path(output_path, "reap_report.json").exists()


def test_base_push_of_a_reap_pruned_model_uploads_its_report(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {}
    backend = _reap_pruned(
        _non_mlx_backend(monkeypatch, "test_export_hub_push_reap_base", calls, seen), monkeypatch
    )
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "old.Q4_K_M.gguf").write_bytes(b"GGUF")
    success, message, output_path = backend.export_base_model(
        str(export_dir), push_to_hub = True, repo_id = "model", hf_token = "hf_fake"
    )
    assert success is True, message
    assert "reap_report.json" in seen["uploaded"]


def test_base_and_adapter_exports_after_a_reap_merge_of_a_lora_load(tmp_path, monkeypatch):
    # REAP merged the adapters in memory (is_peft is False now); the load is still a LoRA one.
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_reap_merged_peft", calls, seen)
    backend._reap_merged_peft = True
    success, message, _ = backend.export_base_model(str(tmp_path / "base"))
    assert success is False and "PEFT model" in message
    assert not (tmp_path / "base").exists()
    success, message, _ = backend.export_lora_adapter(str(tmp_path / "lora"))
    assert success is False and "merged into the model for a REAP export" in message


@pytest.mark.parametrize("method", ["merged", "base"])
def test_hub_uploads_delete_a_reap_report_left_in_a_reused_repo(tmp_path, monkeypatch, method):
    # upload_folder never removes remote files; one it uploads in the same commit is kept, so a
    # pruned export keeps its new report and any other export drops an old one.
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(
        monkeypatch, f"test_export_hub_push_reap_delete_{method}", calls, seen
    )
    export = backend.export_merged_model if method == "merged" else backend.export_base_model
    success, message, _ = export(
        str(tmp_path / "export"), push_to_hub = True, repo_id = "model", hf_token = "hf_fake"
    )
    assert success is True, message
    assert seen["delete_patterns"] == ["reap_report.json"]


def test_cleanup_clears_reap_state_even_when_an_unload_fails(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {}
    backend = _reap_pruned(
        _non_mlx_backend(monkeypatch, "test_export_hub_push_reap_cleanup", calls, seen), monkeypatch
    )
    backend._reap_model = weakref.ref(backend.current_model)

    def unload(name):
        raise RuntimeError("inference engine is busy")

    backend.inference_backend = types.SimpleNamespace(
        models = {"chat": object()}, unload_model = unload
    )
    assert backend.cleanup_memory() is False
    assert backend._reap_key is None and backend._reap_model is None


class _GgufModel:
    def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
        output = Path(f"{model_save_path}_gguf")
        output.mkdir(parents = True)
        (output / f"model.{quantization_method}.gguf").write_bytes(b"GGUF")


def _gguf_model_backend(monkeypatch, calls, seen):
    _HELPERS._install_export_backend_stubs(monkeypatch)
    module = _HELPERS._load_module(
        "test_export_hub_push_cli_login_gguf_backend", "core/export/export.py", monkeypatch
    )
    _HELPERS._patch_hub(monkeypatch, module, calls, seen)
    return _HELPERS._export_backend(module, _GgufModel)


def _push_without_a_token(monkeypatch, export, save_directory, calls, seen):
    if export == "gguf":
        backend = _gguf_model_backend(monkeypatch, calls, seen)
        push = functools.partial(backend.export_gguf, quantization_method = "Q4_K_M")
    elif export == "lora":
        _module, backend, _gguf, _uploads = _lora_backend(
            monkeypatch, "test_export_hub_push_cli_login_lora_backend", calls, seen, "adapter"
        )
        push = backend.export_lora_adapter
    else:
        backend = _non_mlx_backend(
            monkeypatch, f"test_export_hub_push_cli_login_{export}_backend", calls, seen
        )
        push = getattr(backend, f"export_{export}_model")
    monkeypatch.setattr(sys.modules["huggingface_hub"], "get_token", hf_get_token, raising = False)
    return push(save_directory, push_to_hub = True, repo_id = "model", hf_token = None)


@pytest.mark.parametrize("export", ["base", "merged", "lora", "gguf"])
def test_hub_push_with_no_token_uses_the_cli_login(tmp_path, monkeypatch, export):
    login = tmp_path / "hf_token"
    login.write_text("hf_cli_login")
    monkeypatch.setattr(hf_constants, "HF_TOKEN_PATH", str(login))
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_OIDC_RESOURCE"):
        monkeypatch.delenv(key, raising = False)
    calls: list[str] = []
    seen: dict = {}

    success, message, _path = _push_without_a_token(
        monkeypatch, export, str(tmp_path / "export"), calls, seen
    )

    assert success is True, message
    assert seen["token"] == "hf_cli_login"
    assert calls[0] == "create_repo"


@pytest.mark.parametrize("export", ["base", "merged", "lora", "gguf"])
def test_hub_push_with_no_token_and_no_cli_login_is_refused(tmp_path, monkeypatch, export):
    monkeypatch.setattr(hf_constants, "HF_TOKEN_PATH", str(tmp_path / "missing"))
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_OIDC_RESOURCE"):
        monkeypatch.delenv(key, raising = False)
    calls: list[str] = []
    seen: dict = {}

    success, message, _path = _push_without_a_token(
        monkeypatch, export, str(tmp_path / "export"), calls, seen
    )

    assert success is False
    assert "Hugging Face token required" in message
    assert calls == []


def _reexport(backend, export_dir, extra_save):
    save = backend.current_model.save_pretrained_merged

    def save_like_the_merge(
        save_directory,
        tokenizer,
        save_method = None,
        token = None,
    ):
        save(save_directory, tokenizer, save_method = save_method, token = token)
        extra_save(Path(save_directory))

    backend.current_model.save_pretrained_merged = save_like_the_merge
    return backend.export_merged_model(
        str(export_dir), push_to_hub = True, repo_id = "model", hf_token = "hf_fake"
    )


def test_reexport_uploads_the_shard_index_huggingface_hub_kept(tmp_path, monkeypatch):
    # unsloth_zoo fetches a Hub base's index with hf_hub_download(local_dir = export folder):
    # one an earlier export fetched at this commit is kept as is, yet belongs to this save.
    from huggingface_hub import hf_hub_download
    from huggingface_hub._local_folder import write_download_metadata

    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    shards = [f"model-0000{i}-of-00002.safetensors" for i in (1, 2)]
    index = {"weight_map": {"a": shards[0], "b": shards[1]}}
    (export_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    commit = "c" * 40
    write_download_metadata(
        local_dir = export_dir, filename = "model.safetensors.index.json", commit_hash = commit, etag = "e"
    )
    for shard in shards:
        (export_dir / shard).write_bytes(b"old")
    _age(export_dir)
    before = (export_dir / "model.safetensors.index.json").stat()

    def merge(folder):
        (folder / "model.safetensors").unlink()
        for shard in shards:
            (folder / shard).write_bytes(b"merged")
        hf_hub_download(
            "org/base", "model.safetensors.index.json", revision = commit, local_dir = folder
        )

    success, message, output_path = _reexport(backend, export_dir, merge)

    assert success is True, message
    assert backend.current_model.merges == ["merged_16bit"]
    assert seen["folder"] == output_path
    kept = Path(output_path, "model.safetensors.index.json").stat()
    assert (kept.st_mtime_ns, kept.st_ino) == (before.st_mtime_ns, before.st_ino)
    assert seen["uploaded"] == sorted(["model.safetensors.index.json", *shards])


def test_reexport_with_a_new_shard_layout_does_not_upload_the_old_shards(tmp_path, monkeypatch):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    old = [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]
    for shard in old:
        (export_dir / shard).write_bytes(b"old")
    (export_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {str(i): shard for i, shard in enumerate(old)}})
    )
    (export_dir / "README.md").write_text("old card")
    (export_dir / ".huggingface").mkdir()
    (export_dir / ".huggingface" / "upload.lock").write_text("")
    _age(export_dir)

    success, message, output_path = _reexport(backend, export_dir, lambda folder: None)

    assert success is True, message
    assert backend.current_model.merges == ["merged_16bit"]
    assert seen["folder"] == output_path
    assert seen["uploaded"] == ["model.safetensors"]


@pytest.mark.parametrize("leftover", ["tokenizer.model", "single_file_index", "fresh"])
def test_reexport_stages_a_clean_save_when_a_kept_file_may_belong_to_it(
    tmp_path, monkeypatch, leftover
):
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    if leftover == "tokenizer.model":
        # unsloth_zoo downloads tokenizer.model only when it is missing.
        (export_dir / "tokenizer.model").write_bytes(b"spm")
    elif leftover == "single_file_index":
        (export_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "model.safetensors"}})
        )
    else:
        (export_dir / "adapter_config.json").write_text("{}")
    if leftover != "fresh":
        _age(export_dir)

    success, message, output_path = _reexport(backend, export_dir, lambda folder: None)

    assert success is True, message
    assert backend.current_model.merges == ["merged_16bit", "merged_16bit"]
    assert seen["folder"] != output_path
    assert seen["uploaded"] == ["model.safetensors"]


def test_reexport_uploads_a_file_copied_with_the_leftovers_times(tmp_path, monkeypatch):
    # copy2 / copystat give a rewritten file its source's times, however old.
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "config.json").write_text('{"a": 1}')
    _age(export_dir)
    source = tmp_path / "base" / "config.json"
    source.parent.mkdir()
    source.write_text('{"a": 2}')
    os.utime(source, ns = (_OLD_NS, _OLD_NS))

    def copy_from_base(folder):
        staged = folder / ".unsloth-copy"
        shutil.copy2(source, staged)
        os.replace(staged, folder / "config.json")

    success, message, output_path = _reexport(backend, export_dir, copy_from_base)

    assert success is True, message
    assert backend.current_model.merges == ["merged_16bit"]
    assert Path(output_path, "config.json").stat().st_mtime_ns == _OLD_NS
    assert seen["uploaded"] == ["config.json", "model.safetensors"]


def test_reexport_never_touches_the_leftovers_while_saving(tmp_path, monkeypatch):
    # Cancel terminates the export worker mid-save, so nothing may be changed to undo later.
    calls: list[str] = []
    seen: dict = {}
    backend = _non_mlx_backend(monkeypatch, "test_export_hub_push_merged_backend", calls, seen)
    export_dir = tmp_path / "export"
    (export_dir / "sub").mkdir(parents = True)
    (export_dir / "notes.txt").write_text("user")
    (export_dir / "sub" / "data.bin").write_bytes(b"x")
    _age(export_dir)

    def stat_all():
        return {
            path: (path.stat().st_mtime_ns, path.stat().st_atime_ns, path.stat().st_ctime_ns)
            for path in export_dir.rglob("*")
            if path.is_file()
        }

    before = stat_all()
    during = {}
    success, message, _ = _reexport(backend, export_dir, lambda folder: during.update(stat_all()))

    assert success is True, message
    assert {path: during[path] for path in before} == before
    assert {path: stat_all()[path] for path in before} == before
    assert seen["uploaded"] == ["model.safetensors"]
