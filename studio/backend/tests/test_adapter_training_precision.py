# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Studio adapter must load on the base precision it was trained on.

A 16-bit Continued Pretraining run is tagged "CPT", which the resolvers treated as
neither lora nor qlora, so Chat's load_in_4bit=True loaded the base in 4-bit.
"""

import ast
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference.worker import _resolve_lora_4bit

_TRAINER = Path(__file__).resolve().parents[1] / "core/training/trainer.py"


def _patch_adapter_config(bnb_usable = True):
    tree = ast.parse(_TRAINER.read_text(encoding = "utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnslothTrainer")
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_patch_adapter_config"
    )
    namespace = {
        "os": os,
        "json": json,
        "logger": logging.getLogger(__name__),
        "_bitsandbytes_allows_4bit": lambda: bnb_usable,
    }
    exec(compile(ast.Module(body = [fn], type_ignores = []), str(_TRAINER), "exec"), namespace)
    return namespace["_patch_adapter_config"]


def _adapter(tmp_path, payload, base_model):
    (tmp_path / "adapter_config.json").write_text(json.dumps(payload))
    return SimpleNamespace(is_lora = True, path = str(tmp_path), base_model = base_model)


@pytest.mark.parametrize(
    "is_cpt, load_in_4bit, method",
    [(True, False, "CPT"), (True, True, "CPT"), (False, False, "lora"), (False, True, "qlora")],
)
def test_trainer_records_training_precision(tmp_path, is_cpt, load_in_4bit, method):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    trainer = SimpleNamespace(is_cpt = is_cpt, load_in_4bit = load_in_4bit)
    _patch_adapter_config()(trainer, str(tmp_path))
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert cfg["unsloth_training_method"] == method
    assert cfg["unsloth_load_in_4bit"] is load_in_4bit


def test_trainer_records_16bit_when_bitsandbytes_unusable(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    trainer = SimpleNamespace(is_cpt = True, load_in_4bit = True)
    _patch_adapter_config(bnb_usable = False)(trainer, str(tmp_path))
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert cfg["unsloth_load_in_4bit"] is False


@pytest.mark.parametrize("requested", [True, False])
def test_worker_legacy_cpt_keeps_request(tmp_path, requested):
    # A pre-flag CPT adapter may have been quantized on the fly from a non-bnb repo.
    mc = _adapter(tmp_path, {"unsloth_training_method": "CPT"}, "unsloth/Qwen3-4B")
    assert _resolve_lora_4bit(mc, requested) is requested


def test_worker_cpt_on_bnb_base_keeps_4bit(tmp_path):
    mc = _adapter(tmp_path, {"unsloth_training_method": "CPT"}, "unsloth/Qwen3-4B-unsloth-bnb-4bit")
    assert _resolve_lora_4bit(mc, True) is True


@pytest.mark.parametrize("recorded", [True, False])
@pytest.mark.parametrize("requested", [True, False])
def test_worker_recorded_precision_wins(tmp_path, recorded, requested):
    payload = {"unsloth_training_method": "CPT", "unsloth_load_in_4bit": recorded}
    mc = _adapter(tmp_path, payload, "unsloth/Qwen3-4B")
    assert _resolve_lora_4bit(mc, requested) is recorded


@pytest.mark.parametrize(
    "method, requested, expected",
    [("lora", True, False), ("lora", False, False), ("qlora", True, True), ("qlora", False, True)],
)
def test_worker_lora_qlora_unchanged(tmp_path, method, requested, expected):
    mc = _adapter(tmp_path, {"unsloth_training_method": method}, "unsloth/Qwen3-4B")
    assert _resolve_lora_4bit(mc, requested) is expected
