# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""#3726: a ModelScope load must record the repo id, not the local snapshot, and the
16bit merge must fetch its base from ModelScope.

unsloth cannot be imported on GPU-less hosts, so the helpers are extracted via ast and
exec'd against fakes, like the other GPU-free saving tests."""

from __future__ import annotations

import ast
import os
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent


def _load(relpath, name, namespace):
    source = (_ROOT / relpath).read_text(encoding = "utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(compile(ast.get_source_segment(source, node), relpath, "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"{name} not found in {relpath}")


def _tiny_model(name_or_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size = 32,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
    )
    config._name_or_path = name_or_path
    model = LlamaForCausalLM(config)
    model.name_or_path = name_or_path
    return model


@pytest.fixture
def record():
    return _load("unsloth/models/loader.py", "_record_modelscope_repo_id", {"os": os})


def test_snapshot_path_replaced_and_peft_records_repo_id(record, tmp_path):
    from peft import LoraConfig, get_peft_model

    snapshot = tmp_path / "models" / "unsloth--Qwen3-0.6B-unsloth-bnb-4bit" / "snapshots" / "master"
    snapshot.mkdir(parents = True)
    model = _tiny_model(str(snapshot))
    record(model, "unsloth/Qwen3-0.6B-unsloth-bnb-4bit", str(snapshot))

    assert model.config._name_or_path == "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
    assert model.name_or_path == "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
    peft_model = get_peft_model(model, LoraConfig(r = 2, target_modules = ["q_proj"]))
    assert (
        peft_model.peft_config["default"].base_model_name_or_path
        == "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
    )


@pytest.mark.parametrize("repo_id, local_dir", [(None, None), ("unsloth/x", None), (None, "/a")])
def test_noop_without_modelscope_load(record, repo_id, local_dir):
    model = _tiny_model("unsloth/x-bnb-4bit")
    record(model, repo_id, local_dir)
    assert model.config._name_or_path == "unsloth/x-bnb-4bit"


def test_other_names_untouched(record, tmp_path):
    model = _tiny_model(str(tmp_path / "elsewhere"))
    record(model, "unsloth/x", str(tmp_path / "snapshot"))
    assert model.config._name_or_path == str(tmp_path / "elsewhere")


class _Logger:
    def __init__(self):
        self.messages = []

    def warning_once(self, message):
        self.messages.append(message)


@pytest.fixture
def merge_name(monkeypatch, tmp_path):
    calls = []

    def snapshot_download(name):
        calls.append(name)
        if name == "unsloth/missing":
            raise ValueError("not on ModelScope")
        return str(tmp_path / name.replace("/", "--"))

    monkeypatch.setitem(
        sys.modules, "modelscope", types.SimpleNamespace(snapshot_download = snapshot_download)
    )
    logger = _Logger()
    fn = _load(
        "unsloth/save.py",
        "_modelscope_base_model_name",
        {
            "os": os,
            "logger": logger,
            "get_model_name": lambda name, load_in_4bit: name.removesuffix("-unsloth-bnb-4bit"),
        },
    )
    return fn, calls, logger


def test_modelscope_merge_fetches_16bit_base(merge_name, tmp_path):
    fn, calls, _ = merge_name
    got = fn("unsloth/Qwen3-14B-unsloth-bnb-4bit", load_in_4bit = False)
    assert got == str(tmp_path / "unsloth--Qwen3-14B")
    assert calls == ["unsloth/Qwen3-14B"]


def test_modelscope_merge_keeps_local_dir_and_falls_back(merge_name, tmp_path):
    fn, calls, logger = merge_name
    assert fn(str(tmp_path), load_in_4bit = False) == str(tmp_path)
    assert fn("unsloth/missing", load_in_4bit = False) == "unsloth/missing"
    assert calls == ["unsloth/missing"]
    assert "trying Hugging Face" in logger.messages[0]
