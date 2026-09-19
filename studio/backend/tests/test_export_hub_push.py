# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import pytest


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
    (export_dir / "adapter_config.json").write_text("{}")

    success, message, output_path = backend.export_merged_model(
        str(export_dir),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
    )

    assert success is True, message
    assert output_path == str(export_dir.resolve())
    assert backend.current_model.merges == ["merged_16bit", "merged_16bit"]
    assert seen["folder"] != output_path
    assert not Path(seen["folder"]).exists()
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

    def fake_disk_usage(path):
        under_export = Path(path).resolve() == export_parent
        roomy = under_export if roomier != "temp" else not under_export
        return types.SimpleNamespace(total = 1 << 40, used = 0, free = (1 << 40) if roomy else 1)

    monkeypatch.setattr(shutil, "disk_usage", fake_disk_usage)

    if roomier == "export_but_unwritable":
        # Only the export directory has to be writable; its parent need not be.
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


# A stale weight file makes the folder look reused, so the emptiness is only visible on the staging
# copy; a fresh folder shows it on the export directory itself.
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

    success, message, output_path = backend.export_base_model(
        str(export_dir),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
    )

    assert success is True, message
    assert output_path == str(export_dir.resolve())
    assert Path(output_path, "model.safetensors").is_file()
    assert seen["folder"] != output_path
    assert not Path(seen["folder"]).exists()
    assert seen["uploaded"] == ["model.safetensors", "tokenizer.json"]


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


# leg -> (gguf, is_mlx, what lands in the repo)
# Every leg opens the repo itself: leaving a fresh one to push_to_hub would let another
# client create it public in the gap. The adapter leg writes the model card that the
# delegated push can then no longer write.
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
    """Opening the repo first makes Unsloth's wrapper skip its own card, so we write it.

    `upload_to_huggingface` writes MODEL_CARD only when its `create_repo(exist_ok=False)`
    finds the repo absent, which opening it here makes impossible.
    """
    calls: list[str] = []
    seen: dict = {}
    _module, backend, gguf, _uploads = _lora_backend(
        monkeypatch, "test_export_hub_push_lora_card_backend", calls, seen, "adapter"
    )

    success, message, _path = _push_lora(backend, str(tmp_path / "export"), gguf, True)

    assert success is True, message
    assert seen["card_repo"] == "owner/model"
    assert "base_model: unsloth/Qwen2.5-0.5B-Instruct" in seen["card"]
    # Same tags upload_to_huggingface produced for this path, trl included and unsloth
    # not duplicated (the template already carries it).
    tags = seen["card"].split("tags:", 1)[1].split("license:", 1)[0]
    assert sorted(line.strip("- ").strip() for line in tags.strip().splitlines()) == [
        "qwen2",
        "text-generation-inference",
        "transformers",
        "trl",
        "unsloth",
    ]
    # empty `method` leaves a double space the template supplies; markdown collapses it
    assert "# Uploaded finetuned  model" in seen["card"]
    # written before the weights, never after
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
    assert calls == []  # no create_repo, no tightening, no upload
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
