# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Four review findings on the opt-in device map, each pinned by the failure it caused.

Kept apart from the other two device-map files because these are regressions, not the
feature's own contract: every test here fails on the code as it was reviewed.

Extracted with ast so nothing has to import torch's CUDA stack.
"""

import ast
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "unsloth", "models")
LOADER_UTILS = os.path.join(MODELS, "loader_utils.py")
_SRC = open(LOADER_UTILS, encoding = "utf-8").read()


class _FakeCuda:
    def __init__(self, count, free):
        self._count = count
        self._free = free

    def device_count(self):
        return self._count

    def mem_get_info(self, index):
        return (self._free.get(index, 8 * 2**30), 16 * 2**30)


class _Recorder:
    def __init__(self, plan = None):
        self.calls = []
        self._plan = plan

    def __call__(self, model_name, **kwargs):
        self.calls.append((model_name, kwargs))
        return self._plan


class _Plan:
    device_map = {"model.embed_tokens": 0, "lm_head": 1}

    def describe(self):
        return "<plan>"


def _build(
    *,
    devices = 2,
    free = None,
    planner = None,
):
    ns = {
        "os": os,
        "torch": types.SimpleNamespace(cuda = _FakeCuda(devices, free or {})),
        "DEVICE_TYPE_TORCH": "cuda",
        "is_distributed": lambda: False,
    }
    for node in ast.parse(_SRC).body:
        keep = (
            (
                isinstance(node, ast.FunctionDef)
                and node.name in ("requested_device_map", "resolve_unsloth_device_map", "_as_bytes")
            )
            or (isinstance(node, ast.ClassDef) and node.name == "_DefaultDeviceMap")
            or (
                isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", None)
                in ("UNSLOTH_DEVICE_MAP", "DEFAULT_DEVICE_MAP", "_SIZE_UNITS")
            )
        )
        if keep:
            exec(ast.get_source_segment(_SRC, node), ns)
    module = types.ModuleType("unsloth_zoo.device_map_planner")
    module.plan_device_map_for_pretrained = planner
    sys.modules["unsloth_zoo.device_map_planner"] = module
    return ns


# --------------------------------------------------------------------------------------
# 1. An explicit "sequential" is a placement, not the default.
# --------------------------------------------------------------------------------------


def test_the_env_opt_in_leaves_an_explicitly_requested_sequential_alone(monkeypatch):
    """`UNSLOTH_AUTO_DEVICE_MAP=1` upgraded every "sequential", including one the caller
    typed out, so a caller who needs accelerate's greedy fill got a head-aware split."""
    monkeypatch.setenv("UNSLOTH_AUTO_DEVICE_MAP", "1")
    ns = _build()
    assert ns["requested_device_map"]("sequential") == "sequential"
    assert ns["requested_device_map"](ns["DEFAULT_DEVICE_MAP"]) == "unsloth"


def test_the_default_is_indistinguishable_from_sequential_to_everyone_else():
    """The marker may not change what the value IS: it is the documented default, it is
    handed to transformers, and it is printed in signatures and docs."""
    ns = _build()
    default = ns["DEFAULT_DEVICE_MAP"]
    assert default == "sequential"
    assert str(default) == "sequential"
    assert isinstance(default, str)
    assert hash(default) == hash("sequential")
    assert {default: 1}["sequential"] == 1
    assert f"{default}" == "sequential"


@pytest.mark.parametrize("name", ["loader.py", "llama.py", "vision.py", "sentence_transformer.py"])
def test_every_entry_point_defaults_to_the_marked_value(name):
    """A signature left on the bare string cannot be told from an explicit request, so the
    fix above would silently not apply to whichever loader was missed."""
    source = open(os.path.join(MODELS, name), encoding = "utf-8").read()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef) or node.name != "from_pretrained":
            continue
        args = node.args
        defaults = (
            dict(zip([a.arg for a in args.args][-len(args.defaults) :], args.defaults))
            if args.defaults
            else {}
        )
        defaults.update(
            {a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None}
        )
        if "device_map" not in defaults:
            continue
        rendered = ast.unparse(defaults["device_map"])
        assert rendered != "'sequential'", (
            f"{name}:{node.lineno} defaults device_map to the bare string, so the env "
            f"opt-in cannot tell it from a caller who asked for sequential"
        )


def test_sentence_transformers_hands_the_nested_load_a_plain_value():
    """It declines planning for itself, then calls FastModel. Passing the marked default on
    would let that nested load re-upgrade it and split a model ST then pulls onto one card.

    Asserted as the absence of the old process-wide pin as well: os.environ is shared, so
    that fix reached unrelated loads on other threads.
    """
    source = open(os.path.join(MODELS, "sentence_transformer.py"), encoding = "utf-8").read()
    assert "device_map = str(device_map)" in source
    assert (
        'os.environ["UNSLOTH_AUTO_DEVICE_MAP"]' not in source
    ), "the process-wide pin is back; it is visible to every other thread"


# --------------------------------------------------------------------------------------
# 2. max_memory arrived twice.
# --------------------------------------------------------------------------------------


def test_a_caller_supplied_max_memory_does_not_collide_with_the_measured_one():
    """`max_memory` is a named parameter of the planner, so leaving the caller's copy in
    the forwarded kwargs raised `TypeError: got multiple values for keyword argument
    'max_memory'` -- caught by the handler and turned into a silent "sequential", losing
    both the cap and the plan."""
    planner = _Recorder(plan = _Plan())
    ns = _build(free = {0: 10 * 2**30, 1: 10 * 2**30}, planner = planner)
    resolved = ns["resolve_unsloth_device_map"](
        "unsloth",
        "unsloth/Qwen3-0.6B",
        planner_kwargs = {"max_memory": {0: 4 * 2**30}, "retained_rows": 128},
    )
    assert resolved == _Plan.device_map, "the plan was lost to a TypeError"
    assert len(planner.calls) == 1
    _, kwargs = planner.calls[0]
    assert kwargs["retained_rows"] == 128
    # Theirs on the device they capped, ours everywhere else.
    assert kwargs["max_memory"][0] == 4 * 2**30
    assert kwargs["max_memory"][1] == 10 * 2**30


def test_a_cap_above_free_memory_does_not_raise_the_budget():
    """A caller can reserve room we cannot measure, but cannot conjure memory the card has
    not got, and planning above free is how a plan OOMs on dispatch."""
    planner = _Recorder(plan = _Plan())
    ns = _build(free = {0: 2 * 2**30, 1: 2 * 2**30}, planner = planner)
    ns["resolve_unsloth_device_map"](
        "unsloth",
        "unsloth/Qwen3-0.6B",
        planner_kwargs = {"max_memory": {0: 99 * 2**30}},
    )
    assert planner.calls[0][1]["max_memory"][0] == 2 * 2**30


@pytest.mark.parametrize(
    "written,expected",
    [(4 * 2**30, 4 * 2**30), ("4GiB", 4 * 2**30), ("2MiB", 2 * 2**20)],
    ids = ["int", "GiB", "MiB"],
)
def test_the_cap_is_read_the_way_accelerate_reads_it(written, expected):
    """accelerate takes `"10GiB"` as readily as an int, so a caller writes what the loader
    would have taken. Comparing a string against measured bytes would be meaningless."""
    planner = _Recorder(plan = _Plan())
    ns = _build(free = {0: 8 * 2**30, 1: 8 * 2**30}, planner = planner)
    ns["resolve_unsloth_device_map"](
        "unsloth",
        "unsloth/Qwen3-0.6B",
        planner_kwargs = {"max_memory": {0: written}},
    )
    assert planner.calls[0][1]["max_memory"][0] == min(expected, 8 * 2**30)


def test_the_cap_is_read_without_needing_accelerate_importable():
    """Reading the budget through `accelerate.utils.modeling.convert_file_size_to_int` made
    the cap conditional on an import that runs while placement is still being decided: on an
    install without accelerate, or one that moves the symbol, every budget came back
    unreadable and the caller's cap was dropped in silence. Found by the cross-platform run,
    whose runners carry pytest and the Studio requirements but no accelerate."""
    import builtins

    as_bytes = _build()["_as_bytes"]
    real_import = builtins.__import__

    def no_accelerate(name, *args, **kwargs):
        if name.split(".")[0] == "accelerate":
            raise ImportError("No module named 'accelerate'")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = no_accelerate
    try:
        assert as_bytes("4GiB") == 4 * 2**30
        assert as_bytes(4 * 2**30) == 4 * 2**30
    finally:
        builtins.__import__ = real_import


@pytest.mark.parametrize(
    "written",
    [
        0,
        1,
        4 * 2**30,
        "0GiB",
        "4GiB",
        "2MiB",
        "512KiB",
        "1.5GiB",
        "0.5MiB",
        "4gib",
        "4GIB",
        "4Gib",
        "10GB",
        "10gb",
        "10Gb",
        "8MB",
        "8Mb",
        "900KB",
        "900Kb",
        "1.5GB",
        ".5GB",
        "1e3MB",
        "not a size",
        "",
        "GiB",
        "-4GiB",
        -1,
        "4 GiB",
        "4GiBs",
        "4G",
        "4B",
        "4",
        None,
        3.5,
        (),
        {"0": 1},
    ],
)
def test_the_local_size_parser_agrees_with_accelerate(written):
    """The rules are reproduced rather than imported, so something has to hold the copy in
    step with the original wherever the original is in fact installed. accelerate raises on
    what it cannot read and we return None, which is the same answer to the one caller."""
    accelerate_modeling = pytest.importorskip("accelerate.utils.modeling")
    try:
        theirs = accelerate_modeling.convert_file_size_to_int(written)
    except Exception:
        theirs = None
    assert _build()["_as_bytes"](written) == theirs


def test_an_unreadable_cap_leaves_the_measured_value_rather_than_dropping_the_device():
    """A device missing from `max_memory` is a device the planner may not use at all, which
    is a worse answer than ignoring one unparseable entry."""
    planner = _Recorder(plan = _Plan())
    ns = _build(free = {0: 8 * 2**30, 1: 8 * 2**30}, planner = planner)
    ns["resolve_unsloth_device_map"](
        "unsloth",
        "unsloth/Qwen3-0.6B",
        planner_kwargs = {"max_memory": {0: "not a size"}},
    )
    assert planner.calls[0][1]["max_memory"][0] == 8 * 2**30


def test_the_callers_kwargs_dict_is_not_mutated():
    """`device_map_planner_kwargs` is the caller's object, and a loader that empties it
    would change what a second load in the same script asks for."""
    planner = _Recorder(plan = _Plan())
    ns = _build(free = {0: 8 * 2**30, 1: 8 * 2**30}, planner = planner)
    caller_kwargs = {"max_memory": {0: 4 * 2**30}, "retained_rows": 8}
    ns["resolve_unsloth_device_map"](
        "unsloth",
        "unsloth/Qwen3-0.6B",
        planner_kwargs = caller_kwargs,
    )
    assert caller_kwargs == {"max_memory": {0: 4 * 2**30}, "retained_rows": 8}


# --------------------------------------------------------------------------------------
# 3. The legacy diffusion checkpoint the planner cannot rebuild.
# --------------------------------------------------------------------------------------


def test_the_legacy_diffusion_alias_declines_planning_with_its_own_reason():
    """`diffusion_gemma` loads only because `_load_diffusion_config` catches AutoConfig's
    unknown-model error and rewrites the type in memory. The planner is given a name, not a
    config, so it rebuilds from the checkpoint and hits the same error -- reported as a
    generic planning failure. It has to say what actually happened."""
    source = open(os.path.join(MODELS, "diffusion.py"), encoding = "utf-8").read()
    tree = ast.parse(source)

    assert (
        "_unsloth_legacy_alias = True" in source
    ), "nothing records that the alias was applied, so the planner call cannot know"

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "resolve_unsloth_device_map":
            continue
        reasons = [kw for kw in node.keywords if kw.arg == "skip_reason"]
        assert reasons, f"diffusion.py:{node.lineno} plans without vetoing the legacy alias"
        rendered = ast.unparse(reasons[0].value)
        assert "_unsloth_legacy_alias" in rendered
        assert "diffusion_gemma" in rendered
        return
    raise AssertionError("no resolve_unsloth_device_map call in diffusion.py")
