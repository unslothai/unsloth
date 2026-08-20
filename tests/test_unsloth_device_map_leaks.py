# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Every leaf loader must resolve `device_map = "unsloth"` before transformers sees it.

`"unsloth"` is not a placement strategy transformers knows: `modeling_utils.py` turns any
string outside {auto, balanced, balanced_low_0, sequential} into `torch.device(...)`, so
an unresolved one raises:

    ValueError: When passing device_map as a string, the value needs to be a device name
    (e.g. cpu, cuda:0) or 'auto', 'balanced', 'balanced_low_0', 'sequential' but found unsloth

`FastModel.from_pretrained` converts the default to "unsloth" under
`UNSLOTH_AUTO_DEVICE_MAP=1` and then returns through `_dispatch_diffusion()` before
`FastBaseModel` can resolve it, so the text-diffusion slow path needs its own call. And
the planner needs the same repository ref as the real load, or it plans the default branch.

Extracted with ast so nothing has to import torch's CUDA stack.
"""

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
    """A plan built from the default branch's config can name modules the pinned revision
    does not have, and accelerate then refuses the map outright:

        ValueError: The device_map provided does not give any device for the following
        parameters: ...
    """
    for call in _resolve_calls(_source(name)):
        revisions = [kw for kw in call.keywords if kw.arg == "revision"]
        assert revisions, f"{name}:{call.lineno} plans without a revision"
        for keyword in revisions:
            assert ast.unparse(keyword.value) == expected, (
                f"{name}:{call.lineno} passes "
                f"{ast.unparse(keyword.value)}, not the ref the load uses"
            )


def test_sentence_transformer_never_hands_the_sentinel_to_sentence_transformers():
    """`FastSentenceTransformer.from_pretrained` has its own public `device_map`, and its
    `st_device` blocks pass it to `SentenceTransformer(device = ...)` -> `self.to(device)`:

        RuntimeError: Expected one of cpu, cuda, ... device type at start of device string:
        unsloth

    It cannot plan either -- that same `.to()` would pull a split model back onto one card
    -- so the sentinel has to be spent before the `st_device` blocks read it.
    """
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
        and any(ast.unparse(c) == "UNSLOTH_DEVICE_MAP" for c in node.comparators)
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
    """The decline has to outlive the re-entry into `FastModel.from_pretrained`.

    The sentinel is spent on our own `device_map`, but the resulting "sequential" goes to
    `FastModel.from_pretrained`, which runs `requested_device_map` again -- so
    `UNSLOTH_AUTO_DEVICE_MAP=1` upgrades it back to "unsloth" and plans a split map while
    the `st_device` blocks still read "sequential" and pull that model onto one card.
    Without the pin, the env var makes the decline a no-op.
    """
    source = _source("sentence_transformer.py")
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "from_pretrained"
    )

    # The decline itself must read the env var, not the raw argument.
    assert any(
        isinstance(node, ast.Call) and getattr(node.func, "id", None) == "requested_device_map"
        for node in ast.walk(function)
    ), "the decline reads device_map raw, so UNSLOTH_AUTO_DEVICE_MAP=1 walks past it"

    # And the switch must be pinned off across the FastModel load, or FastModel re-upgrades.
    pins = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Subscript)
        and ast.unparse(node.targets[0]) == "os.environ['UNSLOTH_AUTO_DEVICE_MAP']"
        and getattr(node.value, "value", None) == "0"
    ]
    assert pins, "UNSLOTH_AUTO_DEVICE_MAP is not pinned off across FastModel.from_pretrained"

    fastmodel_call = min(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "FastModel.from_pretrained"
    )
    assert (
        min(node.lineno for node in pins) < fastmodel_call
    ), "the pin lands after FastModel has already planned"

    # The pin is restored, so one embedding load does not disable planning process-wide.
    restores = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Try)
        and any("UNSLOTH_AUTO_DEVICE_MAP" in ast.unparse(stmt) for stmt in node.finalbody)
    ]
    assert restores, "UNSLOTH_AUTO_DEVICE_MAP is never restored"
