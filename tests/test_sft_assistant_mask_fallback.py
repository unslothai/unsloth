# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""assistant_only_loss with chat templates TRL has no training template for.

TRL's SFTTrainer swaps in a training template only for templates it knows by exact
text and raises ValueError for the rest, which includes every Unsloth template.
``sft_trainer_assistant_mask_fallback`` turns that raise into a flag that
unsloth_zoo's ``sft_prepare_dataset`` answers with train_on_responses_only masks.
Loaded with ``ast`` from the source so the test stays CPU-only and import-free.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
NAMES = ("_SFT_TRAINING_TEMPLATE", "_SFT_STOP_TOKEN_CHECK", "sft_trainer_assistant_mask_fallback")


def _load():
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in NAMES:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in NAMES for t in node.targets
        ):
            nodes.append(node)
    namespace = {"re": re}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace["sft_trainer_assistant_mask_fallback"]


TRL_SHAPED_INIT = """
def __init__(self, processing_class, args, get_training_chat_template, is_chat_template_stop_token_trained, warn):
    if args.assistant_only_loss and not has_generation_markers(processing_class.chat_template):
        self.chat_template = get_training_chat_template(processing_class)
    else:
        self.chat_template = None
    if args.assistant_only_loss and not is_chat_template_stop_token_trained(
        processing_class, chat_template=self.chat_template
    ):
        warn()
"""


class _Args:
    assistant_only_loss = True


class _Tok:
    chat_template = "{{ messages }}"


def _run(source, template_fn):
    namespace = {"has_generation_markers": lambda template: False}
    exec(source, namespace)
    trainer = type("T", (), {})()
    warned = []
    namespace["__init__"](
        trainer, _Tok(), _Args(), template_fn, lambda *a, **k: False, lambda: warned.append(1)
    )
    return trainer, warned


def _unsupported(processing_class):
    raise ValueError("The chat template is not training-compatible")


def test_unsupported_template_falls_back_instead_of_raising():
    fallback = _load()
    with pytest.raises(ValueError):
        _run(TRL_SHAPED_INIT, _unsupported)
    trainer, warned = _run(fallback("__init__", TRL_SHAPED_INIT), _unsupported)
    assert trainer.chat_template is None
    assert trainer._unsloth_assistant_mask_fallback is True
    assert warned == []  # the stop-token warning describes TRL's masks, not the marker masks


def test_supported_template_keeps_trl_training_template():
    fallback = _load()
    trainer, warned = _run(fallback("__init__", TRL_SHAPED_INIT), lambda pc: "TRAINING_TEMPLATE")
    assert trainer.chat_template == "TRAINING_TEMPLATE"
    assert not hasattr(trainer, "_unsloth_assistant_mask_fallback")
    assert warned == [1]


def test_other_functions_and_idempotent():
    fallback = _load()
    assert fallback("compute_loss", TRL_SHAPED_INIT) == TRL_SHAPED_INIT
    once = fallback("__init__", TRL_SHAPED_INIT)
    assert fallback("__init__", once) == once


def test_applies_to_installed_trl():
    sft_trainer = pytest.importorskip("trl.trainer.sft_trainer")
    # The class, not __init__: other tests may leave __init__ wrapped.
    source = Path(sft_trainer.__file__).read_text(encoding = "utf-8")
    if "get_training_chat_template(processing_class)" not in source:
        pytest.skip("installed TRL predates the training-template swap")
    patched = _load()("__init__", source)
    assert "_unsloth_assistant_mask_fallback = True" in patched
    assert "not getattr(self, '_unsloth_assistant_mask_fallback', False)" in patched
    compile(patched, "<patched SFTTrainer.__init__>", "exec")
