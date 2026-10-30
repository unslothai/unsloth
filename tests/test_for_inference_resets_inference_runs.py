# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""for_inference must reset the inference forward count, on new and older unsloth_zoo."""

from __future__ import annotations

import ast
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_return_logits_survives_for_training import MODELS, _fast_base_model, _Model  # noqa: E402

COMPILED = "unsloth_compiled_module_inference_runs_fake"


def _reset_definition():
    tree = ast.parse((MODELS / "vision.py").read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Try) and any(
            isinstance(n, ast.ImportFrom)
            and any(a.name == "unsloth_reset_inference_runs" for a in n.names)
            for n in node.body
        ):
            namespace = {"sys": sys}
            exec(
                compile(
                    ast.Module(body = [node], type_ignores = []), str(MODELS / "vision.py"), "exec"
                ),
                namespace,
            )
            return namespace["unsloth_reset_inference_runs"]
    return None


class _Stance:
    def __init__(self):
        self.stance = "default"

    def set(
        self,
        stance,
        skip_guard_eval_unsafe = False,
    ):
        self.stance = stance


def _install_zoo(monkeypatch, new_zoo):
    zoo = types.ModuleType("unsloth_zoo")
    training_utils = types.ModuleType("unsloth_zoo.training_utils")
    training_utils.restore_use_cache = lambda model: None
    training_utils.disable_use_cache = lambda model: None
    patches = types.ModuleType("unsloth_zoo.temporary_patches")
    utils = types.ModuleType("unsloth_zoo.temporary_patches.utils")
    if new_zoo:
        utils.UNSLOTH_INFERENCE_RUNS = [0]

        def unsloth_reset_inference_runs():
            utils.UNSLOTH_INFERENCE_RUNS[0] = 0

        utils.unsloth_reset_inference_runs = unsloth_reset_inference_runs
    for name, module in (
        ("unsloth_zoo", zoo),
        ("unsloth_zoo.training_utils", training_utils),
        ("unsloth_zoo.temporary_patches", patches),
        ("unsloth_zoo.temporary_patches.utils", utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return utils


def _compiled_module(monkeypatch, stance, utils, new_zoo):
    """Mimics unsloth_zoo compiler.__DYNAMO__RECOMPILING__."""
    module = types.ModuleType(COMPILED)
    module.INFERENCE_RUNS = 0
    monkeypatch.setitem(sys.modules, COMPILED, module)
    runs = utils.UNSLOTH_INFERENCE_RUNS if new_zoo else None
    stances = []

    def forward():
        stances.append(stance.stance)
        count = runs[0] if new_zoo else module.INFERENCE_RUNS
        if count == 1:
            if stance.stance != "eager_on_recompile":
                stance.set("eager_on_recompile")
        elif stance.stance == "default" and count > 1:
            count = 0
        count += 1
        if new_zoo:
            runs[0] = count
        else:
            module.INFERENCE_RUNS = count

    return forward, stances


@pytest.fixture(params = ["new_zoo", "older_zoo"])
def setup(request, monkeypatch):
    new_zoo = request.param == "new_zoo"
    utils = _install_zoo(monkeypatch, new_zoo)
    stance = _Stance()
    forward, stances = _compiled_module(monkeypatch, stance, utils, new_zoo)
    fbm = _fast_base_model()
    namespace = fbm.for_inference.__globals__
    namespace["torch_compiler_set_stance"] = stance.set
    reset = _reset_definition()
    if reset is not None:
        namespace["unsloth_reset_inference_runs"] = reset

    def generate(max_new_tokens):
        fbm.for_inference(_Model())
        stances.clear()
        for _ in range(max_new_tokens):
            forward()
        return list(stances)

    return generate


def test_decode_step_one_runs_under_default_after_a_one_token_generate(setup):
    generate = setup
    assert generate(1) == ["default"]
    assert generate(4) == ["default", "default", "eager_on_recompile", "eager_on_recompile"]


def test_any_number_of_one_token_generates(setup):
    generate = setup
    for _ in range(3):
        generate(1)
    assert generate(3)[1] == "default"


def test_repeated_generates_still_go_eager_from_decode_step_two(setup):
    generate = setup
    for _ in range(3):
        assert generate(4)[2:] == ["eager_on_recompile", "eager_on_recompile"]


def test_older_zoo_fallback_leaves_other_modules_alone(monkeypatch):
    _install_zoo(monkeypatch, new_zoo = False)
    reset = _reset_definition()
    assert reset is not None, "vision.py no longer binds unsloth_reset_inference_runs"
    module = types.ModuleType(COMPILED)
    module.INFERENCE_RUNS = 1
    other = types.ModuleType("not_compiled_inference_runs")
    other.INFERENCE_RUNS = 1
    monkeypatch.setitem(sys.modules, COMPILED, module)
    monkeypatch.setitem(sys.modules, other.__name__, other)
    reset()
    assert module.INFERENCE_RUNS == 0 and other.INFERENCE_RUNS == 1


def test_new_zoo_helper_is_used(monkeypatch):
    utils = _install_zoo(monkeypatch, new_zoo = True)
    assert _reset_definition() is utils.unsloth_reset_inference_runs
