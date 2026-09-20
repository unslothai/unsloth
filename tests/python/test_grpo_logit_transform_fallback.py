# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The GRPO fallback must answer the same as ``detect_logit_transforms``.

``rl_replacements`` resolves the logit transforms twice: through the planner when
unsloth_zoo provides it, and through an inline fallback when it does not. Only the
planner arm is pinned by ``test_grpo_logit_transform_families``; the fallback arm is
inlined into two 700-line GRPO bodies by ``inspect.getsource``, so it cannot be
imported and has never been executed by a test.

Three families are what the fallback gets wrong if it reads ``logits_scaling`` alone:
Falcon-H1 carries its scale under a different name, HyperCLOVA X multiplies by the
same name Granite divides by, and MiniCPM3's scales the hidden states rather than the
logits. A wrong factor here does not raise. It shifts every log-prob, and with it the
importance ratio the policy update is computed from.

So: lift the ``else:`` block out of each function and run it.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import unsloth.models.rl_replacements as rl


SOURCE = Path(inspect.getfile(rl)).read_text(encoding = "utf-8")
GUARD = "detect_logit_transforms is not None"


def _fallback_blocks():
    """Every `else:` arm of an `if detect_logit_transforms is not None:` in the file."""
    tree = ast.parse(SOURCE)
    blocks = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not node.orelse:
            continue
        if ast.unparse(node.test).strip() != GUARD:
            continue
        enclosing = next(
            fn.name
            for fn in ast.walk(tree)
            if isinstance(fn, ast.FunctionDef)
            and fn.lineno <= node.lineno <= (fn.end_lineno or node.lineno)
        )
        blocks[f"{enclosing}:{node.lineno}"] = ast.Module(body = node.orelse, type_ignores = [])
    return blocks


BLOCKS = _fallback_blocks()


def test_both_grpo_call_sites_were_found():
    """A rename would otherwise make this whole file silently vacuous."""
    assert len(BLOCKS) == 2, f"expected two fallback arms, found {sorted(BLOCKS)}"


class _Cfg:
    def __init__(self, **fields):
        for key, value in fields.items():
            setattr(self, key, value)


def _run(block, config):
    namespace = {
        "model_config": config,
        "model": None,
        # The fallback's soft-cap reader needs the model, which this harness does not
        # build; the scales are what is under test.
        "_unsloth_get_final_logit_softcapping": lambda _model: 0,
    }
    exec(compile(block, "<fallback>", "exec"), namespace)
    return (
        namespace["logit_softcapping"],
        namespace["logit_scale_multiply"],
        namespace["logit_scale_divide"],
    )


# (name, config, (softcapping, multiply, divide), why)
_CASES = [
    (
        "granite",
        _Cfg(model_type = "granite", logits_scaling = 8.0),
        (0, 0, 8.0),
        "modeling_granite.py divides by logits_scaling",
    ),
    (
        "cohere",
        _Cfg(model_type = "cohere", logit_scale = 0.0625),
        (0, 0.0625, 0),
        "modeling_cohere.py multiplies by logit_scale",
    ),
    (
        "falcon_h1",
        _Cfg(model_type = "falcon_h1", lm_head_multiplier = 0.01953125),
        (0, 0.01953125, 0),
        "the scale is spelled lm_head_multiplier, and was dropped entirely",
    ),
    (
        "hyperclovax",
        _Cfg(model_type = "hyperclovax", logits_scaling = 8.0),
        (0, 8.0, 0),
        "MuP multiplies by logits_scaling, unlike Granite which divides",
    ),
    (
        "minicpm3",
        _Cfg(model_type = "minicpm3", logits_scaling = 8.0),
        (0, 0, 0),
        "logits_scaling scales the hidden states before the head, not the logits",
    ),
    (
        "llama",
        _Cfg(model_type = "llama"),
        (0, 0, 0),
        "no transform, and no attribute to read either",
    ),
    (
        "null fields",
        _Cfg(model_type = None, logit_scale = None, logits_scaling = None),
        (0, 0, 0),
        "None must read as off rather than reach the loss",
    ),
]


@pytest.mark.parametrize("site", sorted(BLOCKS))
@pytest.mark.parametrize("name, config, expected, why", _CASES)
def test_fallback_matches_the_planner(site, name, config, expected, why):
    assert _run(BLOCKS[site], config) == expected, f"{site}: {why}"


@pytest.mark.parametrize("name, config, expected, why", _CASES)
def test_fallback_agrees_with_detect_logit_transforms(name, config, expected, why):
    """The two arms must not disagree, or the answer depends on the zoo version."""
    detect = pytest.importorskip(
        "unsloth_zoo.device_map_planner",
        reason = "unsloth_zoo without the planner",
    ).__dict__.get("detect_logit_transforms")
    if detect is None:
        pytest.skip("installed unsloth_zoo predates detect_logit_transforms")
    transforms = detect(config)
    planner = (
        transforms["logit_softcapping"],
        transforms["logit_scale_multiply"],
        transforms["logit_scale_divide"],
    )
    if planner != expected:
        pytest.skip(f"installed unsloth_zoo predates {name} coverage")
    assert _run(BLOCKS[sorted(BLOCKS)[0]], config) == planner, why
