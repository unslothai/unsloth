# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`eval_steps` without `eval_strategy="steps"` is silently ignored by transformers (#3177).

CPU-only: the generated-config snippet and `UnslothTrainingArguments` are lifted
from source with `ast`, so `unsloth` is never imported.
"""

import ast
import contextlib
import io
import warnings
from pathlib import Path
from typing import Optional

import pytest
from transformers import TrainingArguments
from transformers.trainer_utils import IntervalStrategy

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    (10, "no", True),
    (10, "epoch", True),
    (10, IntervalStrategy.EPOCH, True),
    (10, "steps", False),
    (10, IntervalStrategy.STEPS, False),
    (None, "no", False),
    (None, "steps", False),
]


def _rl_config_snippet():
    tree = ast.parse((ROOT / "unsloth" / "models" / "rl.py").read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", None) == "check_eval_steps"
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("check_eval_steps not found in rl.py")


def _unsloth_training_arguments():
    tree = ast.parse((ROOT / "unsloth" / "trainer.py").read_text(encoding = "utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnslothTrainingArguments"
    )
    ns = {
        "TrainingArguments": TrainingArguments,
        "Optional": Optional,
        "QGaloreConfig": object,
        "warnings": warnings,
    }
    exec(compile(ast.Module([cls], []), "trainer.py", "exec"), ns)
    return ns["UnslothTrainingArguments"]


@pytest.mark.parametrize("eval_steps,eval_strategy,warns", CASES)
def test_rl_config_warns_on_ignored_eval_steps(eval_steps, eval_strategy, warns):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(_rl_config_snippet(), {"eval_steps": eval_steps, "eval_strategy": eval_strategy})
    assert ("is ignored because" in buf.getvalue()) is warns
    if warns:
        assert "IntervalStrategy" not in buf.getvalue()


@pytest.mark.parametrize("eval_steps,eval_strategy,warns", CASES)
def test_training_arguments_warns_on_ignored_eval_steps(tmp_path, eval_steps, eval_strategy, warns):
    cls = _unsloth_training_arguments()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        cls(
            output_dir = str(tmp_path),
            report_to = "none",
            eval_steps = eval_steps,
            eval_strategy = eval_strategy,
        )
    hits = [str(w.message) for w in caught if "is ignored because" in str(w.message)]
    assert bool(hits) is warns
