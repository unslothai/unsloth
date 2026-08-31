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
    """The decline has to outlive the re-entry into `FastModel.from_pretrained`.

    That nested call runs `requested_device_map` again, so a still-marked default is
    upgraded back to "unsloth" and planned as a split while `st_device` reads "sequential"
    and pulls the model onto one card. The guard is stripping the marker -- and only the
    marker, since `str()` over everything flattens an explicit dict placement into text.

    The absence of the process-wide pin is asserted too: `os.environ` is shared, so pinning
    it around the call reached unrelated loads on other threads.
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
    """`device_map` is a dict as often as it is a string, and dicts are unhashable.

    `{"": 0, "model.vision_tower": 1} in _PLANNED_DEVICE_MAPS` raises TypeError, so an
    explicit placement -- the one shape a user hand-wrote and most wants honoured -- would
    fail the load outright. Both call sites take the `isinstance` first for that reason,
    and there is no way to notice from reading either one alone.
    """
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
            # One tree, walked twice: a second parse gives different node objects, so
            # the identity test below would find no parent and pass on anything.
            parents = [
                ast.unparse(outer)
                for outer in ast.walk(tree)
                if isinstance(outer, ast.BoolOp) and node in ast.walk(outer)
            ]
            assert any("isinstance(" in text and ", str)" in text for text in parents), (
                f"{name}: a membership test on _PLANNED_DEVICE_MAPS with no isinstance "
                f"guard beside it -- an explicit dict device_map raises TypeError here"
            )
