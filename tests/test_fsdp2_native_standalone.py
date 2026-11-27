# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unsloth/models/fsdp2_native.py runs without `unsloth` imported: the gloo FSDP2 suites load it by path on CPU.
Importing unsloth.models._utils from it pulls unsloth_zoo in, whose accelerator check raises on a CPU runner
("Unsloth cannot find any torch accelerator?"), so every rank of those suites died at shard_model."""

import ast
import os

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "unsloth", "models")


def _tree(name):
    with open(os.path.join(_MODELS, name), encoding = "utf-8") as f:
        return ast.parse(f.read())


def _function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_fsdp2_native_never_imports_utils():
    offenders = []
    for node in ast.walk(_tree("fsdp2_native.py")):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            if node.module == "_utils" or (
                node.module is None and any(a.name == "_utils" for a in node.names)
            ):
                offenders.append(node.lineno)
    assert offenders == [], f"fsdp2_native.py imports ._utils at lines {offenders}"


def test_its_expert_parallel_check_matches_the_one_in_utils():
    """Two copies of one walk: a change to either must be made to both."""
    ours = _function(_tree("fsdp2_native.py"), "_is_unsloth_expert_parallel")
    theirs = _function(_tree("_utils.py"), "_is_unsloth_expert_parallel")
    strip = lambda fn: [
        ast.dump(n)
        for n in fn.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    ]
    assert strip(ours) == strip(theirs)
