# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Adapter-format export: platform default, conversion routing, GGUF rejects, feature metadata."""

import json
import os
from unittest.mock import MagicMock

import pytest

from core.export import export as export_mod
from utils.models.checkpoints import parse_adapter_features


def _backend(monkeypatch, is_mlx, tmp_path):
    monkeypatch.setattr(export_mod, "_IS_MLX", is_mlx)
    monkeypatch.setattr(export_mod, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(export_mod, "resolve_export_write_dir", lambda p: tmp_path / "out")
    monkeypatch.setattr(export_mod, "ensure_dir", lambda p: os.makedirs(p, exist_ok = True))
    backend = export_mod.ExportBackend.__new__(export_mod.ExportBackend)
    backend.current_model = MagicMock()
    backend.current_tokenizer = MagicMock()
    backend.is_peft = True
    return backend


def _peft_writer(model):
    # The real zoo converter refuses an existing destination and publishes a fresh one.
    def _save(
        path,
        adapter_config = None,
        adapter_format = "mlx",
    ):
        assert not os.path.lexists(path)
        os.makedirs(path)
        with open(os.path.join(path, "adapter_model.safetensors"), "w") as f:
            f.write(adapter_format)
        with open(os.path.join(path, "adapter_config.json"), "w") as f:
            json.dump({"r": 8}, f)

    model.save_lora_adapters = MagicMock(side_effect = _save)


# Omission stays the pre-existing native call on both platforms; explicit native values match.
@pytest.mark.parametrize(
    "is_mlx,requested,expect",
    [
        (True, None, "mlx"),
        (True, "mlx", "mlx"),
        (True, "peft", "peft"),
        (False, None, "peft"),
        (False, "peft", "peft"),
        (False, "mlx", "error"),
    ],
)
def test_six_cell_matrix(monkeypatch, tmp_path, is_mlx, requested, expect):
    backend = _backend(monkeypatch, is_mlx, tmp_path)
    if expect == "peft" and is_mlx:
        _peft_writer(backend.current_model)
    ok, message, _path = backend.export_lora_adapter(
        str(tmp_path / "dst"),
        adapter_format = requested,
    )
    if expect == "error":
        assert not ok and "MLX" in message
        backend.current_model.save_pretrained.assert_not_called()
        return
    assert ok, message
    if is_mlx:
        args, kwargs = backend.current_model.save_lora_adapters.call_args
        backend.current_model.save_pretrained.assert_not_called()
        if expect == "mlx":
            assert args == (str(tmp_path / "out"),) and kwargs == {}
        else:
            assert kwargs == {"adapter_format": "peft"}
            assert (tmp_path / "out" / "adapter_model.safetensors").read_text() == "peft"
    else:
        backend.current_model.save_pretrained.assert_called_once()
        backend.current_model.save_lora_adapters.assert_not_called()


def test_repeat_peft_export_overwrites(monkeypatch, tmp_path):
    backend = _backend(monkeypatch, True, tmp_path)
    _peft_writer(backend.current_model)
    out = tmp_path / "out"
    out.mkdir()
    (out / "adapter_model.safetensors").write_text("stale")
    for _ in range(2):
        ok, message, _ = backend.export_lora_adapter(str(out), adapter_format = "peft")
        assert ok, message
    assert (out / "adapter_model.safetensors").read_text() == "peft"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_cuda_hub_push_unchanged(monkeypatch, tmp_path):
    backend = _backend(monkeypatch, False, tmp_path)
    monkeypatch.setattr(export_mod, "HfApi", lambda token = None: MagicMock())
    monkeypatch.setattr(export_mod, "_open_hub_repo", lambda hf_api, repo_id, private: repo_id)
    monkeypatch.setattr(export_mod, "_publish_unsloth_model_card", lambda *a: None)
    ok, message, _ = backend.export_lora_adapter("", push_to_hub = True, repo_id = "u/r", hf_token = "t")
    assert ok, message
    backend.current_model.push_to_hub.assert_called_once_with("u/r", token = "t", private = False)


def test_outdated_zoo_hard_error(monkeypatch, tmp_path):
    backend = _backend(monkeypatch, True, tmp_path)

    def _old_zoo_saver(path, adapter_config = None):  # no adapter_format kwarg
        raise AssertionError("an outdated saver must not be invoked")

    backend.current_model.save_lora_adapters = _old_zoo_saver
    ok, message, _ = backend.export_lora_adapter(
        str(tmp_path / "dst"),
        adapter_format = "peft",
    )
    assert not ok and "unsloth-zoo" in message

    def _modern_saver_with_internal_bug(
        path,
        adapter_config = None,
        adapter_format = "mlx",
    ):
        raise TypeError("scale must be a float")

    backend.current_model.save_lora_adapters = _modern_saver_with_internal_bug
    ok, message, _ = backend.export_lora_adapter(
        str(tmp_path / "dst2"),
        adapter_format = "peft",
    )
    assert not ok and "unsloth-zoo" not in message and "scale" in message


@pytest.mark.parametrize(
    "cfg,fs_attr,reason",
    [
        ({"alpha_pattern": {"^q_proj": 32}}, None, "alpha"),
        ({"use_rslora": True, "rank_pattern": {"^q_proj": 4}}, None, "rsLoRA"),
        ({"use_dora": True}, None, "DoRA"),
        ({"modules_to_save": ["lm_head"]}, None, "full-module state"),
        ({}, {"model.embed_tokens": "embedding_auto"}, "full-module state"),
        ({"target_parameters": ["experts.gate_up_proj"]}, None, "expert"),
    ],
)
def test_gguf_rejects(monkeypatch, tmp_path, cfg, fs_attr, reason):
    backend = _backend(monkeypatch, True, tmp_path)
    backend.current_model._unsloth_full_state_modules = fs_attr
    (tmp_path / "adapter_config.json").write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError, match = reason):
        backend._convert_peft_dir_to_gguf(str(tmp_path), "q8_0", None)


@pytest.mark.parametrize("token,expect", [(False, None), ("hf_x", "hf_x")])
def test_gguf_converter_token_env(monkeypatch, tmp_path, token, expect):
    import subprocess
    import sys
    import types

    llama = tmp_path / "llama.cpp"
    (llama / "gguf-py").mkdir(parents = True)
    (llama / "convert_lora_to_gguf.py").write_text("")
    monkeypatch.setitem(
        sys.modules,
        "unsloth_zoo.llama_cpp",
        types.SimpleNamespace(LLAMA_CPP_DEFAULT_DIR = str(llama), install_llama_cpp = lambda **k: None),
    )
    seen = {}

    def _run(cmd, env, **kwargs):
        seen["env"] = env
        return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")

    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setenv("HF_TOKEN", "host-token")
    backend = _backend(monkeypatch, True, tmp_path)
    backend.current_model._unsloth_full_state_modules = None
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "o/m"}))
    backend._convert_peft_dir_to_gguf(str(tmp_path), "q8_0", token)
    assert seen["env"].get("HF_TOKEN") == expect
    assert (seen["env"].get("HF_HUB_DISABLE_IMPLICIT_TOKEN") == "1") is (token is False)


def test_parse_adapter_features(tmp_path):
    def _dir(cfg):
        d = tmp_path / f"a{len(list(tmp_path.iterdir()))}"
        d.mkdir()
        (d / "adapter_config.json").write_text(json.dumps(cfg))
        return str(d)

    assert parse_adapter_features(str(tmp_path)) is None  # no config
    # A PEFT config without markers and no readable weights is UNVERIFIED,
    # never a false "verified plain".
    base = parse_adapter_features(_dir({"r": 8}))
    assert base == {
        "dora": False,
        "full_state": None,
        "moe_target_parameters": False,
        "non_uniform": False,
    }
    # MLX artifacts verify through their weight header too: pure LoRA is a
    # verified negative, extra trainable state is positive, and no readable
    # file stays unverified (trainer checkpoints may carry unmarked state).
    assert parse_adapter_features(_dir({"fine_tune_type": "lora"}))["full_state"] is None
    import numpy as np
    from safetensors.numpy import save_file

    d_mlx = _dir({"fine_tune_type": "lora"})
    save_file(
        {"model.layers.0.self_attn.q_proj.lora_a": np.zeros((2, 2), dtype = "float32")},
        os.path.join(d_mlx, "adapters.safetensors"),
    )
    assert parse_adapter_features(d_mlx)["full_state"] is False
    save_file(
        {
            "model.layers.0.self_attn.q_proj.lora_a": np.zeros((2, 2), dtype = "float32"),
            "lm_head.bias": np.zeros((2,), dtype = "float32"),
        },
        os.path.join(d_mlx, "adapters.safetensors"),
    )
    assert parse_adapter_features(d_mlx)["full_state"] is True
    assert parse_adapter_features(_dir({"use_dora": True}))["dora"] is True
    assert parse_adapter_features(_dir({"fine_tune_type": "dora"}))["dora"] is True
    assert parse_adapter_features(_dir({"modules_to_save": ["lm_head"]}))["full_state"] is True
    assert (
        parse_adapter_features(_dir({"full_state_modules": {"lm_head": "modules_to_save"}}))[
            "full_state"
        ]
        is True
    )
    assert (
        parse_adapter_features(_dir({"target_parameters": ["experts.g"]}))["moe_target_parameters"]
        is True
    )
    assert parse_adapter_features(_dir({"rank_pattern": {"q": 4}}))["non_uniform"] is True
    assert (
        parse_adapter_features(_dir({"unsloth_mlx_lora_module_scales": {"q": 2.0}}))["non_uniform"]
        is True
    )


def test_local_dir_never_format_mixed(monkeypatch, tmp_path):
    backend = _backend(monkeypatch, True, tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "adapter_model.safetensors").write_bytes(b"x")  # other format
    ok, message, _ = backend.export_lora_adapter(str(out))
    assert not ok and "mix" in message
    backend.current_model.save_lora_adapters.assert_not_called()
    # Nested layouts (peft named adapters) refuse too.
    (out / "adapter_model.safetensors").unlink()
    (out / "named").mkdir()
    (out / "named" / "adapter_model.safetensors").write_bytes(b"x")
    ok, message, _ = backend.export_lora_adapter(str(out))
    assert not ok and "mix" in message
    # The legacy PEFT weight spelling counts as PEFT too.
    (out / "named" / "adapter_model.safetensors").unlink()
    (out / "adapter_model.bin").write_bytes(b"x")
    ok, message, _ = backend.export_lora_adapter(str(out))
    assert not ok and "mix" in message
