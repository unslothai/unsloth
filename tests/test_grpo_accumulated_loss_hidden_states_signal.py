# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The gradient GRPO path must share the no-grad hidden-state signal.

The generated trainer embeds ``unsloth_zoo.grpo_accumulated_loss`` as source.
Zoo's raw-logits branches dispatch on width, which cannot distinguish logits
from hidden states when ``vocab_size == hidden_size``. These tests exercise the
source patch that replaces those comparisons before the function is embedded.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
PATCH_HELPER = "_patch_grpo_accumulated_loss_hidden_states_dispatch"
DISPATCH_HELPER = "_unsloth_grpo_returns_hidden_states"
SIGNAL_HELPER = "_unsloth_grpo_hidden_states_signal"
PATTERN_NAME = "_GRPO_HIDDEN_STATES_WIDTH_DISPATCH"
CANDIDATE_PATTERN_NAME = "_GRPO_HIDDEN_STATES_WIDTH_DISPATCH_CANDIDATE"
MARKER = "__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__"
WRAPPED = "_unsloth_grpo_hidden_states_forward_wrapped"
DEGRADED = "_unsloth_grpo_hidden_states_warning_issued"


def _load_helpers():
    text = SOURCE_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(text, filename = str(SOURCE_PATH))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {PATTERN_NAME, CANDIDATE_PATTERN_NAME}
            for target in node.targets
        ):
            body.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            PATCH_HELPER,
            DISPATCH_HELPER,
            SIGNAL_HELPER,
        }:
            body.append(node)
    names = {
        target.id
        for node in body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    names |= {
        node.name for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    expected = {
        PATTERN_NAME,
        CANDIDATE_PATTERN_NAME,
        PATCH_HELPER,
        DISPATCH_HELPER,
        SIGNAL_HELPER,
    }
    assert names == expected, (names, expected)
    namespace = {"inspect": inspect, "re": re}
    exec(compile(ast.Module(body = body, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace


_HELPERS = _load_helpers()
patch_gradient_source = _HELPERS[PATCH_HELPER]
returns_hidden_states = _HELPERS[DISPATCH_HELPER]


_FOUR_SITE_SOURCE = """
def grpo_accumulated_loss(unwrapped_model, lm_head, tensors):
    _pack_h, _pack_rh, _h, new_hidden_states_chunk = tensors
    paths = []
    if _pack_h.shape[-1] == lm_head.shape[1]:
        paths.append("hidden")
    else:
        paths.append("raw")
    if _pack_rh.shape[-1] == lm_head.shape[1]:
        paths.append("hidden")
    else:
        paths.append("raw")
    _pg_fn = "hidden"
    if _h.shape[-1] != lm_head.shape[1]:
        _pg_fn = "raw"
    paths.append(_pg_fn)
    if new_hidden_states_chunk.shape[-1] == lm_head.shape[1]:
        paths.append("hidden")
    else:
        paths.append("raw")
    return paths
""".lstrip()


_NUMERIC_SOURCE = """
def grpo_accumulated_loss(unwrapped_model, lm_head, output, index):
    new_hidden_states_chunk = output
    if new_hidden_states_chunk.shape[-1] == lm_head.shape[1]:
        logits = new_hidden_states_chunk @ lm_head.t()
    else:
        logits = new_hidden_states_chunk
    return torch.gather(
        torch.log_softmax(logits.float(), dim = -1),
        dim = -1,
        index = index.unsqueeze(-1),
    ).squeeze(-1)
""".lstrip()


def _compile_gradient_function(source):
    namespace = {"torch": torch, DISPATCH_HELPER: returns_hidden_states}
    exec(
        compile(patch_gradient_source(source), "<patched grpo_accumulated_loss>", "exec"), namespace
    )
    return namespace["grpo_accumulated_loss"]


def _degraded_model():
    return SimpleNamespace(
        forward = lambda *args, **kwargs: None,
        **{WRAPPED: True, DEGRADED: True},
    )


class _CompiledModel:
    __UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__ = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError


def test_all_four_gradient_dispatches_use_the_explicit_signal():
    patched = patch_gradient_source(_FOUR_SITE_SOURCE)
    assert patched.count(f"{DISPATCH_HELPER}(unwrapped_model,") == 4
    assert ".shape[-1] == lm_head.shape[1]" not in patched
    assert ".shape[-1] != lm_head.shape[1]" not in patched


def test_degraded_square_logits_take_the_raw_path_at_every_gradient_site():
    function = _compile_gradient_function(_FOUR_SITE_SOURCE)
    square = torch.zeros(1, 2, 8)
    head = torch.zeros(8, 8)
    assert function(_degraded_model(), head, (square,) * 4) == ["raw"] * 4


def test_compiled_square_hidden_states_stay_on_the_hidden_path():
    function = _compile_gradient_function(_FOUR_SITE_SOURCE)
    square = torch.zeros(1, 2, 8)
    head = torch.zeros(8, 8)
    assert function(_CompiledModel(), head, (square,) * 4) == ["hidden"] * 4


def test_degraded_square_logits_are_not_projected_twice_in_the_gradient_path():
    function = _compile_gradient_function(_NUMERIC_SOURCE)
    generator = torch.Generator().manual_seed(20260810)
    head = torch.randn(8, 8, generator = generator)
    logits = torch.randn(2, 4, 8, generator = generator)
    index = torch.randint(0, 8, (2, 4), generator = generator)

    actual = function(_degraded_model(), head, logits, index)
    expected = torch.gather(
        torch.log_softmax(logits.float(), dim = -1),
        dim = -1,
        index = index.unsqueeze(-1),
    ).squeeze(-1)
    doubled = torch.gather(
        torch.log_softmax((logits @ head.t()).float(), dim = -1),
        dim = -1,
        index = index.unsqueeze(-1),
    ).squeeze(-1)

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, doubled, rtol = 1e-3, atol = 1e-3)


def test_source_patch_fails_loudly_if_zoo_removes_the_width_dispatch_contract():
    source = "def grpo_accumulated_loss():\n    return None\n"
    with pytest.raises(RuntimeError, match = "could not find the GRPO gradient"):
        patch_gradient_source(source)


def test_source_patch_rejects_a_partially_patched_zoo():
    """One surviving match must not license the others to stay width-only.

    A zoo that respells some of its dispatches still leaves at least one the
    strict pattern recognises, which is enough to make `replacements == 0`
    false. The respelled sites would then keep deciding on width alone, back to
    silently wrong gradients for a square lm_head.
    """
    source = _FOUR_SITE_SOURCE.replace(
        "    if _pack_h.shape[-1] == lm_head.shape[1]:",
        "    if _pack_h.shape[-1] == lm_head.shape[-1]:",
    ).replace(
        "    if _pack_rh.shape[-1] != lm_head.shape[1]:",
        "    if lm_head.shape[-1] != _pack_rh.shape[-1]:",
    )
    assert source != _FOUR_SITE_SOURCE
    with pytest.raises(RuntimeError, match = r"patched only \d+ of \d+"):
        patch_gradient_source(source)


def test_source_patch_still_accepts_the_installed_zoo():
    """The stricter guard must not reject the zoo actually installed here."""
    zoo = pytest.importorskip("importlib.util")
    spec = zoo.find_spec("unsloth_zoo")
    if spec is None or spec.origin is None:
        pytest.skip("unsloth_zoo is not installed")
    zoo_source = (Path(spec.origin).parent / "rl_replacements.py").read_text(encoding = "utf-8")
    lines = zoo_source.splitlines()
    functions = [
        node
        for node in ast.walk(ast.parse(zoo_source))
        if isinstance(node, ast.FunctionDef) and node.name == "grpo_accumulated_loss"
    ]
    if len(functions) != 1:
        pytest.skip("this unsloth_zoo has no single grpo_accumulated_loss")
    body = "\n".join(lines[functions[0].lineno - 1 : functions[0].end_lineno])
    patched = patch_gradient_source(body)
    assert DISPATCH_HELPER in patched


def test_generated_trainer_embeds_the_patched_gradient_function():
    text = SOURCE_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(text, filename = str(SOURCE_PATH))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and ast.unparse(node.func.value) == "RL_PRE_ITEMS['grpo_trainer']"
        and node.args
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == PATCH_HELPER
    ]
    assert len(calls) == 1
    assert ast.unparse(calls[0].args[0].args[0]) == "grpo_accumulated_loss"
