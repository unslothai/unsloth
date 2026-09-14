# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Unsloth's CPT path routes embed_tokens/lm_head through `modules_to_save`.

Every branch of `Trainer.prepare_model_for_training` that builds an adapter therefore
has to forward that argument. The audio branches used to omit it, which silently froze
the matrices the run was configured to train while still applying an embedding LR.
"""

import ast
import os

TRAINER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "core",
    "training",
    "trainer.py",
)


def _peft_calls():
    """Every get_peft_model call, plus the peft_kwargs dict that feeds one."""
    tree = ast.parse(open(TRAINER, encoding = "utf-8").read())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "prepare_model_for_training"
    )
    calls = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None)
        if name == "get_peft_model" or (name is None and getattr(node.func, "id", None) == "dict"):
            kwargs = {k.arg for k in node.keywords if k.arg}
            # Only the dicts that actually build an adapter.
            if name == "get_peft_model" or "target_modules" in kwargs:
                calls.append((node.lineno, kwargs))
    return calls


def test_every_adapter_call_forwards_modules_to_save():
    calls = _peft_calls()
    assert len(calls) >= 5, f"expected every branch, found {len(calls)}"
    missing = [
        lineno
        for lineno, kwargs in calls
        if "target_modules" in kwargs and "modules_to_save" not in kwargs
    ]
    assert not missing, (
        f"trainer.py lines {missing} build an adapter without modules_to_save; "
        "a CPT run there would silently not train the embeddings it asked for"
    )
