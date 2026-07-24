# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Adapter-format export contracts: platform-resolved default, conversion
routing, GGUF staging rejects, Hub compare-and-swap, feature metadata."""

import json
import os
from unittest.mock import MagicMock

import pytest

from core.export import export as export_mod
from utils.models.checkpoints import parse_adapter_features


def _backend(
    monkeypatch,
    is_mlx,
    tmp_path,
    created = None,
):
    monkeypatch.setattr(export_mod, "_IS_MLX", is_mlx)
    monkeypatch.setattr(export_mod, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(export_mod, "resolve_export_write_dir", lambda p: tmp_path / "out")
    monkeypatch.setattr(
        export_mod,
        "ensure_dir",
        (lambda p: created.append(str(p))) if created is not None else (lambda p: None),
    )
    backend = export_mod.ExportBackend.__new__(export_mod.ExportBackend)
    backend.current_model = MagicMock()
    backend.current_tokenizer = MagicMock()
    backend.is_peft = True
    return backend


# The six-cell platform x field matrix: omission stays byte-identical to the
# pre-existing native call on BOTH platforms, and explicit native values match.
@pytest.mark.parametrize(
    "is_mlx,requested,expect",
    [
        (True, None, "mlx"),  # Mac + omitted -> native MLX passthrough
        (True, "mlx", "mlx"),  # Mac + explicit mlx (frontend default path)
        (True, "peft", "peft"),  # Mac + explicit peft -> converts
        (False, None, "peft"),  # CUDA + omitted -> native PEFT passthrough
        (False, "peft", "peft"),  # CUDA + explicit peft -> same native result
        (False, "mlx", "error"),  # CUDA + explicit mlx -> specified error
    ],
)
def test_six_cell_matrix(monkeypatch, tmp_path, is_mlx, requested, expect):
    created = []
    backend = _backend(monkeypatch, is_mlx, tmp_path, created)
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
        if expect == "mlx":
            # Exactly the pre-existing native call: no adapter_format kwarg.
            assert kwargs == {}
        else:
            assert kwargs == {"adapter_format": "peft"}
        backend.current_model.save_pretrained.assert_not_called()
        # Conversions get a fresh destination (parent created only); the
        # native path keeps its pre-created directory.
        expected_dir = tmp_path / ("out" if expect == "mlx" else "")
        assert created == [str(expected_dir)]
    else:
        backend.current_model.save_pretrained.assert_called_once()
        backend.current_model.save_lora_adapters.assert_not_called()


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


def _gguf_setup(
    monkeypatch,
    tmp_path,
    peft_cfg,
    fs_attr = None,
):
    backend = _backend(monkeypatch, True, tmp_path)
    out = tmp_path / "gguf"
    out.mkdir()

    def _fake_save(destination, resolved_format):
        assert resolved_format == "peft"
        os.makedirs(destination, exist_ok = True)  # the real zoo publishes the dir
        with open(os.path.join(destination, "adapter_config.json"), "w") as f:
            json.dump(peft_cfg, f)

    backend._save_mlx_adapter = _fake_save
    backend.current_model._unsloth_full_state_modules = fs_attr
    return backend, str(out)


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
def test_gguf_staging_rejects(monkeypatch, tmp_path, cfg, fs_attr, reason):
    backend, out = _gguf_setup(monkeypatch, tmp_path, cfg, fs_attr)
    with pytest.raises(RuntimeError, match = reason):
        backend._export_mlx_lora_gguf(out, "q8_0", None)


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
