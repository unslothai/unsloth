# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Continuing from a saved adapter keeps the adapter's own LoRA shape (#13140).

The run's requested rank, alpha and CPT embedding modules cannot be applied to an
existing adapter, so the trainer must name what it ignored instead of dropping it.
"""

import ast
import os
from types import SimpleNamespace

TRAINER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "core",
    "training",
    "trainer.py",
)


def _helper():
    # trainer.py imports torch and unsloth; load only the pure helper.
    tree = ast.parse(open(TRAINER, encoding = "utf-8").read())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_ignored_adapter_settings"
    )
    namespace = {}
    exec(compile(ast.Module(body = [fn], type_ignores = []), TRAINER, "exec"), namespace)
    return namespace["_ignored_adapter_settings"]


def _adapter(
    r = 8,
    lora_alpha = 16,
    modules_to_save = None,
):
    return {"default": SimpleNamespace(r = r, lora_alpha = lora_alpha, modules_to_save = modules_to_save)}


def test_matching_request_ignores_nothing():
    ignored = _helper()
    assert ignored(_adapter(), 8, 16, None) == []
    assert ignored(_adapter(modules_to_save = ["embed_tokens", "lm_head"]), 8, 16, ["lm_head"]) == []


def test_reports_rank_and_alpha_kept_from_adapter():
    ignored = _helper()(_adapter(r = 8, lora_alpha = 16), 16, 32, None)
    assert ignored == ["rank 16 (adapter has 8)", "alpha 32 (adapter has 16)"]


def test_reports_cpt_embedding_modules_the_adapter_does_not_train():
    ignored = _helper()(_adapter(), 8, 16, ["embed_tokens", "model.lm_head"])
    assert ignored == ["training embed_tokens, model.lm_head (adapter does not train them)"]


def test_unknown_peft_config_shape_is_ignored():
    ignored = _helper()
    assert ignored(None, 8, 16, None) == []
    assert ignored({}, 8, 16, None) == []
