# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL 0.22 DPO sends image-text-to-text models (Gemma3) down its text path unless Unsloth widens the check."""

import ast
import inspect
import os
import types

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RL_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")
TRL_GATE = (
    "self.is_vision_model = model.config.model_type in MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.keys()"
)


def _lift():
    src = open(RL_PATH, encoding = "utf-8").read()
    ns = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_unsloth_dpo_is_vision_model",
            "dpo_trainer_vision_model_detection",
        ):
            exec(ast.get_source_segment(src, node), ns)
    return ns


def _patched_init():
    dpo_trainer = pytest.importorskip("trl.trainer.dpo_trainer")
    source = inspect.getsource(dpo_trainer.DPOTrainer.__init__)
    transform = _lift().get("dpo_trainer_vision_model_detection", lambda name, function: function)
    return source, transform("__init__", source), dpo_trainer


def _is_vision(patched, dpo_trainer, model_type, processing_class):
    line = next(l.strip() for l in patched.splitlines() if "self.is_vision_model =" in l)
    expression = line.split("=", 1)[1]
    ns = dict(vars(dpo_trainer))
    ns.update(_lift())
    ns["model"] = types.SimpleNamespace(config = types.SimpleNamespace(model_type = model_type))
    ns["processing_class"] = processing_class
    return eval(expression, ns)


class _Tokenizer:
    pass


class _Processor:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.image_processor = object()


def test_image_text_to_text_models_take_the_vision_path():
    source, patched, dpo_trainer = _patched_init()
    if TRL_GATE not in source:
        # This TRL picks its vision path without the model-type table: nothing to widen.
        assert TRL_GATE not in patched
        return
    from transformers.models.auto import modeling_auto

    assert "gemma3" in modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    assert _is_vision(patched, dpo_trainer, "gemma3", _Processor()) is True
    # Already covered by TRL's own table.
    assert _is_vision(patched, dpo_trainer, "qwen2_5_vl", _Processor()) is True
    # The helper the rewritten line calls is emitted into the generated trainer module.
    src = open(RL_PATH, encoding = "utf-8").read()
    assert (
        'RL_PRE_ITEMS["dpo_trainer"].append(inspect.getsource(_unsloth_dpo_is_vision_model))' in src
    )


def test_text_only_use_keeps_the_text_path():
    source, patched, dpo_trainer = _patched_init()
    if TRL_GATE not in source:
        assert TRL_GATE not in patched
        return
    # A VLM trained from its tokenizer alone, and a text model with a processor, stay on tokenize_row.
    assert not _is_vision(patched, dpo_trainer, "gemma3", _Tokenizer())
    assert not _is_vision(patched, dpo_trainer, "llama", _Processor())
