# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO compute_loss files metrics under TRL's bucket: "train" if model.training else "eval".

It read control.should_evaluate instead, which is False for eval_on_start and for a standalone
trainer.evaluate(), so those eval passes logged kl / completion_length / clip_ratio into "train".
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
HELPERS = {
    "grpo_trainer_compute_loss",
    "_unsloth_grpo_accumulation_steps",
    "_unsloth_grpo_is_metric_values",
    "_unsloth_get_model_config",
    "_unsloth_text_configs",
    "_unsloth_get_final_logit_softcapping",
    "_unsloth_resolve_logit_scales",
}
BATCH, PROMPT, COMPLETION = 2, 3, 4


def _compute_loss(completion_length):
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"), filename = str(SOURCE_PATH))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS]
    namespace = {"torch": torch, "inspect": inspect}
    module = ast.Module(body = nodes, type_ignores = [])
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    source = namespace["grpo_trainer_compute_loss"]("compute_loss", "")

    def grpo_compute_loss_slow(*args, **kwargs):
        # Stands in for unsloth_zoo's loss; returns the tuple compute_loss unpacks.
        ones = torch.ones(BATCH, COMPLETION)
        return (
            torch.tensor(1.0),
            torch.tensor(completion_length),
            torch.tensor(0.25),
            None,
            None,
            ones,
            ones,
        )

    namespace.update(
        grpo_compute_loss_slow = grpo_compute_loss_slow,
        grpo_accumulated_loss = None,
        detect_logit_transforms = None,
        _unsloth_grpo_vision_inputs = lambda source: {},
        _unsloth_fix_mm_token_type_ids = lambda *args, **kwargs: None,
        nanmin = torch.min,
        nanmax = torch.max,
    )
    exec(textwrap.dedent(source), namespace)
    return namespace["compute_loss"]


def _trainer(training, should_evaluate):
    return SimpleNamespace(
        args = SimpleNamespace(
            loss_type = "bnpo",
            max_completion_length = COMPLETION,
            delta = None,
            temperature = 1.0,
            unsloth_num_chunks = 1,
        ),
        accelerator = SimpleNamespace(num_processes = 1, gather = lambda tensor: tensor),
        beta = 0.0,
        epsilon_low = 0.2,
        epsilon_high = 0.2,
        loss_type = "bnpo",
        use_vllm = False,
        model = SimpleNamespace(training = training, config = None),
        control = SimpleNamespace(should_evaluate = should_evaluate),
        _metrics = {"train": defaultdict(list), "eval": defaultdict(list)},
        _get_per_token_logps = lambda *args, **kwargs: torch.zeros(BATCH, COMPLETION),
        processing_class = None,
    )


def _inputs():
    ids = torch.ones(BATCH, PROMPT, dtype = torch.long)
    completion = torch.ones(BATCH, COMPLETION, dtype = torch.long)
    return dict(
        prompt_ids = ids,
        prompt_mask = ids,
        completion_ids = completion,
        completion_mask = completion,
        advantages = torch.ones(BATCH),
        old_per_token_logps = None,
    )


@pytest.mark.parametrize(
    ("training", "should_evaluate", "bucket"),
    [
        (True, False, "train"),
        # eval_on_start=True or a standalone trainer.evaluate(): flag still False.
        (False, False, "eval"),
        # An evaluation the schedule triggered: both agree.
        (False, True, "eval"),
    ],
)
def test_metrics_follow_model_training(training, should_evaluate, bucket):
    trainer = _trainer(training, should_evaluate)
    _compute_loss(4.0)(trainer, trainer.model, _inputs())
    other = "eval" if bucket == "train" else "train"
    assert trainer._metrics[bucket]["completion_length"] == [4.0]
    assert trainer._metrics[bucket]["kl"] == [0.25]
    assert not trainer._metrics[other]


def test_eval_pass_does_not_change_the_logged_train_length():
    trainer = _trainer(training = False, should_evaluate = False)
    _compute_loss(4.0)(trainer, trainer.model, _inputs())
    trainer.model.training = True
    _compute_loss(8.0)(trainer, trainer.model, _inputs())
    lengths = trainer._metrics["train"]["completion_length"]
    assert sum(lengths) / len(lengths) == 8.0
