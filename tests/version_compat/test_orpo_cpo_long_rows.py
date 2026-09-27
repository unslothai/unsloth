# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""ORPO / CPO rows must fit max_length after the Unsloth tokenize_row patch.

TRL 0.29+ never truncates the prompt and cuts answers to `max_length - longer_response_length`,
so prompt + answer can exceed max_length; the fast forward then cuts input_ids but not labels.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import re
import textwrap
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest


RL_REPLACEMENTS = (
    Path(importlib.util.find_spec("unsloth").origin).parent / "models" / "rl_replacements.py"
)


def _row_cap():
    tree = ast.parse(RL_REPLACEMENTS.read_text(encoding = "utf-8"))
    wanted = {"_ORPO_ROW_CAP", "orpo_trainer_row_cap"}
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in wanted)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in wanted for t in n.targets))
    ]
    assert len(nodes) == 2, "no ORPO/CPO tokenize_row row cap in rl_replacements.py"
    ns = {"re": re}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(RL_REPLACEMENTS), "exec"), ns)
    return ns["orpo_trainer_row_cap"]


def _trainer(name):
    # Importing unsloth (the test conftest does) swaps the exported class for the patched copy, so read TRL's module source.
    for mod in (f"trl.experimental.{name}.{name}_trainer", f"trl.trainer.{name}_trainer"):
        try:
            module = importlib.import_module(mod)
        except Exception:
            continue
        tree = ast.parse(inspect.getsource(module))
        cls = next(
            (
                n
                for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == f"{name.upper()}Trainer"
            ),
            None,
        )
        methods = {n.name: n for n in getattr(cls, "body", ()) if isinstance(n, ast.FunctionDef)}
        if {"tokenize_row", "build_tokenized_answer"} <= methods.keys():
            return module, {
                k: ast.get_source_segment(inspect.getsource(module), methods[k]) for k in methods
            }
    pytest.skip(f"{name.upper()}Trainer.tokenize_row not available on this TRL")


class _Tok:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(
        self,
        text,
        add_special_tokens = False,
        **kwargs,
    ):
        ids = [3 + sum(map(ord, w)) % 997 for w in text.split()]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _tokenize_row(trainer, patched):
    module, methods = trainer
    src = textwrap.dedent(methods["tokenize_row"])
    if patched:
        new = _row_cap()("tokenize_row", src)
        assert new != src, "row cap anchor not found in this TRL tokenize_row"
        src = new
    ns = dict(vars(module))
    exec(src, ns)
    exec(textwrap.dedent(methods["build_tokenized_answer"]), ns)
    fake = SimpleNamespace(
        processing_class = _Tok(),
        max_length = 32,
        max_prompt_length = 16,
        max_completion_length = None,
        truncation_mode = "keep_end",
        is_encoder_decoder = False,
        label_pad_token_id = -100,
        padding_value = 0,
    )
    fake.build_tokenized_answer = MethodType(ns["build_tokenized_answer"], fake)
    return lambda feature: ns["tokenize_row"](fake, feature)


def _words(n, tag):
    return " ".join(f"{tag}{i}" for i in range(n))


LONG_ROWS = {
    "long_prompt": (_words(60, "p"), " " + _words(3, "c"), " " + _words(2, "r")),
    "long_answer": (_words(4, "p"), " " + _words(60, "c"), " " + _words(50, "r")),
    "both_long": (_words(40, "p"), " " + _words(40, "c"), " " + _words(45, "r")),
    "just_over": (_words(15, "p"), " " + _words(15, "c"), " " + _words(14, "r")),
}


@pytest.mark.parametrize("name", ["orpo", "cpo"])
@pytest.mark.parametrize("row", sorted(LONG_ROWS))
def test_long_rows_fit_max_length(name, row):
    prompt, chosen, rejected = LONG_ROWS[row]
    out = _tokenize_row(_trainer(name), True)(
        {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    )
    for side in ("chosen", "rejected"):
        ids, labels = out[f"{side}_input_ids"], out[f"{side}_labels"]
        assert len(ids) == len(labels) == len(out[f"{side}_attention_mask"])
        assert len(ids) <= 32, f"{side} row of {len(ids)} tokens exceeds max_length 32"
        assert any(label != -100 for label in labels), f"{side} lost every answer token"


@pytest.mark.parametrize("name", ["orpo", "cpo"])
def test_rows_that_fit_are_unchanged(name):
    trainer = _trainer(name)
    feature = {
        "prompt": _words(6, "p"),
        "chosen": " " + _words(5, "c"),
        "rejected": " " + _words(7, "r"),
    }
    assert _tokenize_row(trainer, True)(dict(feature)) == _tokenize_row(trainer, False)(
        dict(feature)
    )


@pytest.mark.parametrize("name", ["orpo", "cpo"])
def test_unpatched_trl_row_overflows(name):
    # Control: without the cap, TRL 0.29+ really emits rows past max_length for these inputs.
    trainer = _trainer(name)
    if "max_prompt_length" in trainer[1]["tokenize_row"]:
        pytest.skip("this TRL truncates the prompt itself")
    prompt, chosen, rejected = LONG_ROWS["long_prompt"]
    out = _tokenize_row(trainer, False)({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    assert len(out["chosen_input_ids"]) > 32
