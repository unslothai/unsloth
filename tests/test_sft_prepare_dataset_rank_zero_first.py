# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Zoo's sft_prepare_dataset replaces the body of TRL's SFTTrainer._prepare_dataset, which dropped
TRL's main_process_first window: under DDP every rank tokenized the train split at once. The
generated trainer wraps _prepare_dataset in Zoo's rank_zero_first_prepare when the installed Zoo
has it. Read from source (CPU only): the rewrite is applied to TRL's own sft_trainer module."""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import re
import types

import pytest

RL_PATH = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"
NAMES = ("_SFT_PREPARE_DATASET_PATTERN", "_SFT_PREPARE_DATASET_REPLACEMENT")


def _constants():
    tree = ast.parse(RL_PATH.read_text(encoding = "utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in NAMES:
                    found[target.id] = ast.literal_eval(node.value)
    assert set(found) == set(NAMES), f"missing {set(NAMES) - set(found)}"
    return found["_SFT_PREPARE_DATASET_PATTERN"], found["_SFT_PREPARE_DATASET_REPLACEMENT"]


def _rewritten_init():
    # The file, not the class: once Unsloth is imported, the module's SFTTrainer is the generated one.
    spec = importlib.util.find_spec("trl.trainer.sft_trainer")
    if spec is None or spec.origin is None:
        pytest.skip("trl is not installed")
    pattern, replacement = _constants()
    source = pathlib.Path(spec.origin).read_text(encoding = "utf-8")
    rewritten, n = re.subn(pattern, replacement, source, count = 1)
    assert n == 1, "TRL's train_dataset = self._prepare_dataset( call was not found"
    return rewritten


def test_generated_init_still_parses_and_wraps_prepare_dataset():
    rewritten = _rewritten_init()
    ast.parse(rewritten)
    assert "self._unsloth_model_ref = model" in rewritten
    wrap = rewritten.index(
        "self._prepare_dataset = _unsloth_rank_zero_first_prepare(self._prepare_dataset)"
    )
    assert wrap < rewritten.index("train_dataset = self._prepare_dataset(")


def _run_injected_block(dataset_utils):
    """Execute the injected lines on their own, with a stand-in Zoo module."""
    _, replacement = _constants()
    block = re.sub(r"\\1", "", replacement).replace("\\n", "\n")
    block = block[: block.index("train_dataset = self._prepare_dataset(")]
    calls = []

    class Trainer:
        def _prepare_dataset(self, dataset, *args):
            calls.append(dataset)
            return dataset

    trainer = Trainer()
    import sys

    saved = sys.modules.get("unsloth_zoo.dataset_utils")
    sys.modules["unsloth_zoo.dataset_utils"] = dataset_utils
    try:
        exec(block, {}, {"self": trainer, "model": object()})
    finally:
        if saved is None:
            sys.modules.pop("unsloth_zoo.dataset_utils", None)
        else:
            sys.modules["unsloth_zoo.dataset_utils"] = saved
    return trainer, calls


def test_injected_block_installs_the_zoo_wrapper():
    seen = []

    def rank_zero_first_prepare(prepare):
        def wrapped(dataset, *args):
            seen.append(dataset)
            return prepare(dataset, *args)

        return wrapped

    zoo = types.ModuleType("unsloth_zoo.dataset_utils")
    zoo.rank_zero_first_prepare = rank_zero_first_prepare
    trainer, calls = _run_injected_block(zoo)
    assert trainer._prepare_dataset("train_split", None) == "train_split"
    assert seen == calls == ["train_split"]


def test_injected_block_keeps_the_method_on_an_older_zoo():
    trainer, calls = _run_injected_block(types.ModuleType("unsloth_zoo.dataset_utils"))
    assert "_prepare_dataset" not in vars(trainer)
    assert trainer._prepare_dataset("train_split") == "train_split"
