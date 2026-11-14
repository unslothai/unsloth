# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A forced 4-bit -> 16-bit load must surface as a run warning, not only a log line."""

import ast
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def test_worker_config_carries_the_forced_16bit_flag():
    from core.training.training import _build_training_worker_config
    assert (
        _build_training_worker_config(
            {"model_name": "x", "load_in_4bit": False, "load_in_4bit_forced_16bit": True}
        )["load_in_4bit_forced_16bit"]
        is True
    )
    assert _build_training_worker_config({"model_name": "x"})["load_in_4bit_forced_16bit"] is False


def test_route_marks_the_flip_it_makes():
    source = (BACKEND / "routes" / "training.py").read_text()
    flip = source.index('training_kwargs["load_in_4bit"] = False')
    assert 'training_kwargs["load_in_4bit_forced_16bit"] = True' in source[flip : flip + 400]


def test_worker_emits_a_warning_event_for_the_flip():
    source = (BACKEND / "core" / "training" / "worker.py").read_text()
    tree = ast.parse(source)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "load_in_4bit_forced_16bit" in ast.unparse(node.test):
            body = ast.unparse(node)
            found = '"type": "warning"' in body.replace("'", '"') and "FORCED_16BIT_WARNING" in body
            break
    assert found
    import core.training.worker as worker

    assert "16-bit" in worker.FORCED_16BIT_WARNING and "4-bit" in worker.FORCED_16BIT_WARNING


class _Trainer:
    def __init__(self, is_vlm = False):
        self.is_vlm = is_vlm
        self.is_audio_vlm = False


def test_text_run_with_mlp_off_is_told_the_selector_is_ignored():
    from core.training.worker import _ignored_selector_warning

    config = {
        "finetune_mlp_modules": False,
        "target_modules": ["q_proj", "gate_proj", "up_proj", "down_proj"],
    }
    message = _ignored_selector_warning(_Trainer(), config)
    assert message and "finetune_mlp_modules" in message and "gate_proj" in message
    assert _ignored_selector_warning(_Trainer(), {"target_modules": ["q_proj"]}) is None
    assert _ignored_selector_warning(_Trainer(is_vlm = True), config) is None
    assert (
        _ignored_selector_warning(_Trainer(), {**config, "training_type": "Full Finetuning"})
        is None
    )


def test_worker_emits_the_selector_warning():
    source = (BACKEND / "core" / "training" / "worker.py").read_text()
    call = source.index("_ignored_selector_warning(trainer, config)")
    assert '"type": "warning"' in source[call : call + 300]
