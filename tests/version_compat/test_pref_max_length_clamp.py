# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Preference-trainer rows longer than the model max_seq_length must not crash the log-prob gather.

The fast forward cuts input_ids to model.max_seq_length while TRL builds labels at args.max_length.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


RL_PY = Path(importlib.util.find_spec("unsloth").origin).parent / "models" / "rl.py"


def _clamp_snippet():
    src = RL_PY.read_text(encoding = "utf-8")
    m = re.search(
        r'elif trainer_file in \("dpo_trainer", "kto_trainer", "orpo_trainer", "cpo_trainer"\):\n(.*?)\n        \)\n',
        src,
        re.S,
    )
    assert m, "no preference-trainer max_length clamp in rl.py"
    body = m.group(1).split("extra_args += (", 1)[1]
    return "".join(
        ast.literal_eval(line.strip()) for line in body.splitlines() if line.strip().startswith('"')
    )


def _run(
    max_seq_length,
    max_length,
    max_prompt_length = None,
):
    model = SimpleNamespace(max_seq_length = max_seq_length)
    args = SimpleNamespace(max_length = max_length, max_prompt_length = max_prompt_length)
    exec(_clamp_snippet(), {"model": model, "args": args, "print": lambda *a: None})
    return args


@pytest.mark.parametrize("max_length", [1024, None])
def test_max_length_is_capped_at_the_model_limit(max_length):
    assert _run(48, max_length).max_length == 48


def test_a_shorter_max_length_is_kept():
    assert _run(2048, 1024).max_length == 1024


def test_prompt_length_stays_below_max_length():
    args = _run(48, 1024, max_prompt_length = 512)
    assert args.max_length == 48 and args.max_prompt_length == 24
    assert _run(2048, 1024, max_prompt_length = 512).max_prompt_length == 512


def test_prompt_length_is_left_alone_without_a_clamp():
    args = _run(2048, 1024, max_prompt_length = 1024)
    assert args.max_length == 1024 and args.max_prompt_length == 1024


def test_unset_prompt_length_stays_below_a_small_clamp():
    # ORPO / CPO / KTO resolve None to 128, which must stay below max_length.
    assert _run(64, 1024, max_prompt_length = None).max_prompt_length == 32
    assert _run(512, 1024, max_prompt_length = None).max_prompt_length is None


def test_model_without_a_limit_is_untouched():
    assert _run(None, 1024).max_length == 1024


@pytest.mark.parametrize("trainer", ["DPOTrainer", "KTOTrainer"])
def test_patched_trainer_carries_the_clamp(trainer):
    import unsloth  # noqa: F401
    import trl

    cls = getattr(trl, trainer, None)
    patched = [k for k in getattr(cls, "__mro__", ()) if "Unsloth" in k.__module__ + k.__name__]
    if not patched:
        pytest.skip(f"Unsloth does not patch {trainer} on this TRL")
    assert any("_unsloth_model_msl" in inspect.getsource(k) for k in patched)
