# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Every leaf loader must resolve the unsloth device_map sentinel before transformers sees it."""

import ast
import os

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "unsloth", "models")


def _source(name):
    return open(os.path.join(MODELS, name), encoding = "utf-8").read()


def _resolve_calls(source):
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "resolve_unsloth_device_map"
    ]


def test_unsloth_is_not_a_device_map_transformers_accepts():
    """The premise. If transformers ever learns the string, the rest of this file is moot."""
    import torch
    with pytest.raises(RuntimeError):
        torch.device("unsloth")


@pytest.mark.parametrize("name", ["llama.py", "vision.py", "diffusion.py"])
def test_every_leaf_loader_resolves_before_it_loads(name):
    """loader.py only routes; these three are what actually call transformers, and each
    one is reachable holding "unsloth" (diffusion via `_dispatch_diffusion`)."""
    assert _resolve_calls(_source(name)), f"{name} forwards device_map unresolved"


def test_the_diffusion_dispatch_hands_over_the_planner_hints():
    """`_dispatch_diffusion` forwards **kwargs, but `device_map_planner_kwargs` is a named
    parameter of `FastModel.from_pretrained`, so it is not in **kwargs and would be lost."""
    source = _source("loader.py")
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if ast.unparse(node.func) != "FastDiffusionModel.from_pretrained":
            continue
        passed = {kw.arg for kw in node.keywords}
        assert "device_map_planner_kwargs" in passed
        return
    raise AssertionError("no FastDiffusionModel.from_pretrained call in loader.py")


@pytest.mark.parametrize(
    "name,expected",
    [("llama.py", "revision"), ("vision.py", "_revision"), ("diffusion.py", "revision")],
)
def test_the_planner_gets_the_same_ref_the_weights_do(name, expected):
    """The planner must read the same revision as the weights, or the map names modules missing there."""
    for call in _resolve_calls(_source(name)):
        revisions = [kw for kw in call.keywords if kw.arg == "revision"]
        assert revisions, f"{name}:{call.lineno} plans without a revision"
        for keyword in revisions:
            assert ast.unparse(keyword.value) == expected, (
                f"{name}:{call.lineno} passes "
                f"{ast.unparse(keyword.value)}, not the ref the load uses"
            )


def test_sentence_transformer_never_hands_the_sentinel_to_sentence_transformers():
    """Spend the sentinel before st_device reads it; .to() would pull a split model back onto one card."""
    tree = ast.parse(_source("sentence_transformer.py"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "from_pretrained"
    )
    assert any(
        kw.arg == "device_map" for kw in function.args.kwonlyargs + function.args.args
    ), "from_pretrained no longer takes device_map"

    spends = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Compare)
        and ast.unparse(node.left) == "device_map"
        and any(
            ast.unparse(c) in ("UNSLOTH_DEVICE_MAP", "_PLANNED_DEVICE_MAPS")
            for c in node.comparators
        )
    ]
    assert spends, "the 'unsloth' sentinel reaches SentenceTransformer(device = ...) unresolved"

    first_st_device = min(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "st_device" for t in node.targets)
    )
    assert (
        min(node.lineno for node in spends) < first_st_device
    ), "the sentinel is spent after st_device is derived from device_map"


def test_sentence_transformer_decline_survives_the_env_var():
    """Declining must survive the nested FastModel call: strip only the marker, never pin os.environ."""
    source = _source("sentence_transformer.py")
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "from_pretrained"
    )

    assert any(
        isinstance(node, ast.Call) and getattr(node.func, "id", None) == "requested_device_map"
        for node in ast.walk(function)
    ), "the decline reads device_map raw, so UNSLOTH_AUTO_DEVICE_MAP=1 walks past it"

    strips = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "device_map" for t in node.targets)
        and ast.unparse(node.value) == "unmarked_device_map(device_map)"
    ]
    assert strips, "the nested load still gets the marked default, which it will re-upgrade"

    fastmodel_call = min(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "FastModel.from_pretrained"
    )
    assert (
        min(node.lineno for node in strips) < fastmodel_call
    ), "the marker is stripped after FastModel has already planned"

    assert "os.environ['UNSLOTH_AUTO_DEVICE_MAP']" not in ast.unparse(
        function
    ), "the process-wide pin is back; every other thread sees it"


def test_every_planned_map_membership_test_is_guarded_against_a_dict():
    """Dicts are unhashable, so each _PLANNED_DEVICE_MAPS membership test must check isinstance first."""
    for name in os.listdir(MODELS):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(_source(name))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Compare)
                and any(ast.unparse(c) == "_PLANNED_DEVICE_MAPS" for c in node.comparators)
            ):
                continue
            # One tree walked twice: a reparse gives new nodes and the identity test passes vacuously.
            parents = [
                ast.unparse(outer)
                for outer in ast.walk(tree)
                if isinstance(outer, ast.BoolOp) and node in ast.walk(outer)
            ]
            assert any("isinstance(" in text and ", str)" in text for text in parents), (
                f"{name}: a membership test on _PLANNED_DEVICE_MAPS with no isinstance "
                f"guard beside it -- an explicit dict device_map raises TypeError here"
            )


def test_sentence_transformer_declines_to_a_value_st_device_normalises():
    """Declines to sequential, a value st_device normalises; nothing here is sharded."""
    tree = ast.parse(_source("sentence_transformer.py"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "from_pretrained"
    )

    declines = [
        ast.unparse(node.value)
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "device_map" for t in node.targets)
        and ast.unparse(node.value) != "requested_device_map(device_map)"
    ]
    assert "'sequential'" in declines, "the decline is not a literal 'sequential'"
    assert (
        "_PLANNED_DEVICE_MAPS[device_map]" not in declines
    ), "declining to the sharding fallback sends 'balanced' to SentenceTransformer(device=)"

    whitelists = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.List)
        and [getattr(e, "value", None) for e in node.elts] == ["auto", "sequential"]
    ]
    assert whitelists, "the st_device whitelist changed; re-check what the decline may be"
