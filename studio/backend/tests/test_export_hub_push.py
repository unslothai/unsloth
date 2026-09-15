# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.util
import sys
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

    def save_pretrained(self, save_directory):
        Path(save_directory, "model.safetensors").write_bytes(b"weights")

    def save_pretrained_merged(
        self,
        save_directory,
        tokenizer,
        save_method = None,
        token = None,
    ):
        output = Path(f"{save_directory}-torchao-fp8")
        output.mkdir(parents = True)
        (output / "model.safetensors").write_bytes(b"fp8")


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
    assert output_path == str(Path(f"{tmp_path / 'export'}-torchao-fp8").resolve())
    assert seen["token"] == "hf_fake"
    assert seen["repo"] == {"repo_id": "model", "private": private, "exist_ok": True}
    assert calls == _expected_calls(private)
    assert seen["card_repo"] == "owner/model"
    assert seen["folder"] == output_path
    assert "model.safetensors" in seen["uploaded"]
