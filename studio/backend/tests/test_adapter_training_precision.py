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


def test_method_and_precision_never_disagree(tmp_path):
    # No bitsandbytes: trained on a 16-bit base, so it is a lora, not qlora.
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    _patch_adapter_config(bnb_usable = False)(
        SimpleNamespace(is_cpt = False, load_in_4bit = True), str(tmp_path)
    )
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert cfg["unsloth_training_method"] == "lora"
    assert cfg["unsloth_load_in_4bit"] is False


def _forced_16bit_branches_in_load_model():
    """Audio branches of load_model that pass a literal load_in_4bit=False to from_pretrained."""
    tree = ast.parse(_TRAINER.read_text(encoding = "utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnslothTrainer")
    load_model = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_model"
    )
    forced = set()
    for node in ast.walk(load_model):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)):
            continue
        left, ops, comparators = node.test.left, node.test.ops, node.test.comparators
        if not (
            isinstance(left, ast.Attribute)
            and left.attr == "_audio_type"
            and len(ops) == 1
            and isinstance(ops[0], ast.Eq)
            and isinstance(comparators[0], ast.Constant)
            and isinstance(comparators[0].value, str)
        ):
            continue
        for call in (
            c
            for c in ast.walk(ast.Module(body = node.body, type_ignores = []))
            if isinstance(c, ast.Call)
        ):
            if not (isinstance(call.func, ast.Attribute) and call.func.attr == "from_pretrained"):
                continue
            for kw in call.keywords:
                if (
                    kw.arg == "load_in_4bit"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is False
                ):
                    forced.add(comparators[0].value)
    return forced


def _declared_forced_16bit_audio_types():
    """_FORCED_16BIT_AUDIO_TYPES read out of the source.

    Read rather than imported: importing core.training.trainer pulls in torch and unsloth,
    and a runner without them turns this assertion into a skip instead of a failure.
    """
    for node in ast.parse(_TRAINER.read_text(encoding = "utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "_FORCED_16BIT_AUDIO_TYPES" for t in node.targets
        ):
            value = node.value
            if isinstance(value, ast.Call) and value.args:
                value = value.args[0]
            return set(ast.literal_eval(value))
    raise AssertionError("_FORCED_16BIT_AUDIO_TYPES is not defined in trainer.py")


def test_forced_16bit_audio_types_match_the_loader():
    # Read the branches, not the list: an unlisted forced-16-bit type would record 4-bit.
    declared = _declared_forced_16bit_audio_types()
    assert declared == _forced_16bit_branches_in_load_model()
    assert "snac" not in declared


def test_load_model_records_the_forced_16bit_precision():
    # load_model must correct load_in_4bit before the forced-16-bit branches run.
    tree = ast.parse(_TRAINER.read_text(encoding = "utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnslothTrainer")
    load_model = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_model"
    )
    fixups = [
        node
        for node in ast.walk(load_model)
        if isinstance(node, ast.If)
        and any(
            isinstance(n, ast.Name) and n.id == "_FORCED_16BIT_AUDIO_TYPES"
            for n in ast.walk(node.test)
        )
        and any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "load_in_4bit" for t in n.targets)
            and isinstance(n.value, ast.Constant)
            and n.value.value is False
            for n in node.body
        )
    ]
    assert len(fixups) == 1, "load_model must set self.load_in_4bit = False for the forced types"
    csm = next(
        node
        for node in ast.walk(load_model)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Attribute)
        and node.test.left.attr == "_audio_type"
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "csm"
    )
    assert fixups[0].lineno < csm.lineno


def test_forced_16bit_audio_run_reloads_in_16bit(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    trainer = SimpleNamespace(is_cpt = False, load_in_4bit = False)
    _patch_adapter_config()(trainer, str(tmp_path))
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert cfg["unsloth_load_in_4bit"] is False
    mc = SimpleNamespace(is_lora = True, path = str(tmp_path), base_model = "unsloth/csm-1b")
    assert _resolve_lora_4bit(mc, True) is False
