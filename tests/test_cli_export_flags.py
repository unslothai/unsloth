# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth-cli.py: --save_gguf, --push_gguf and --push_model export without --save_model."""

import argparse
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "unsloth_legacy_cli_export_flags", Path(__file__).resolve().parents[1] / "unsloth-cli.py"
)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


class _RecordingModel:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name not in {
            "save_pretrained_gguf",
            "push_to_hub_gguf",
            "save_pretrained_merged",
            "push_to_hub_merged",
        }:
            raise AttributeError(name)
        return lambda *a, **k: self.calls.append((name, a, k))


def _export(is_mlx = False, **flags):
    args = dict(
        save_model = False,
        save_gguf = False,
        push_model = False,
        push_gguf = False,
        save_method = "merged_16bit",
        save_path = "model",
        quantization = ["q8_0"],
        hub_path = "hf/model",
        hub_token = "tok",
    )
    args.update(flags)
    model = _RecordingModel()
    cli._save_or_push_model(model, "tokenizer", argparse.Namespace(**args), is_mlx)
    return model.calls


@pytest.mark.parametrize(
    "flags, expected",
    [
        (dict(save_gguf = True), ["save_pretrained_gguf"]),
        (dict(push_gguf = True), ["push_to_hub_gguf"]),
        (dict(push_model = True), ["push_to_hub_merged"]),
        (dict(save_model = True), ["save_pretrained_merged"]),
        (dict(save_model = True, push_model = True), ["save_pretrained_merged", "push_to_hub_merged"]),
        (dict(save_gguf = True, push_gguf = True), ["save_pretrained_gguf", "push_to_hub_gguf"]),
        (dict(save_gguf = True, quantization = ["f16", "q8_0"]), ["save_pretrained_gguf"] * 2),
    ],
)
def test_each_export_flag_exports_on_its_own(flags, expected):
    assert [name for name, _, _ in _export(**flags)] == expected


def test_push_only_routes_hub_path_token_and_method():
    [(_, gguf_args, gguf_kwargs)] = _export(push_gguf = True)
    assert gguf_args[0] == "hf/model"
    assert gguf_kwargs == {"quantization_method": "q8_0", "token": "tok"}
    [(_, merged_args, merged_kwargs)] = _export(push_model = True, save_method = "lora")
    assert merged_args == ("hf/model", "tokenizer", "lora")
    assert merged_kwargs == {"token": "tok"}


def test_mlx_push_model_alone_pushes():
    [(name, _, kwargs)] = _export(is_mlx = True, push_model = True)
    assert name == "save_pretrained_merged"
    assert kwargs["push_to_hub"] is True
    assert kwargs["repo_id"] == "hf/model"


def test_no_export_flag_warns_and_exports_nothing(capsys):
    assert _export() == []
    assert "not saved" in capsys.readouterr().out
