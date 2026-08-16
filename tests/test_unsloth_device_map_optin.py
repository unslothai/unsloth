# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests the opt-in multi-GPU planning in loader_utils.py. No GPU needed.

`device_map = "unsloth"` asks unsloth_zoo's planner for a head-aware placement instead of
accelerate's `"sequential"`. The Muse Glimmer GRPO notebook does this by hand today, in
about 25 lines of mem_get_info arithmetic.

It is opt-in precisely because the alternative is not safe: an existing multi-GPU caller
who never asked for planning must keep the placement they have. So most of this file is
about what must NOT change, and only the last group is about planning working.

Extracted with ast so nothing has to import torch's CUDA stack.
"""

import ast
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOADER_UTILS = os.path.join(HERE, "unsloth", "models", "loader_utils.py")
_SRC = open(LOADER_UTILS, encoding = "utf-8").read()


class _FakeCuda:
    def __init__(
        self,
        count,
        free = None,
    ):
        self._count = count
        self._free = free or {}

    def device_count(self):
        return self._count

    def mem_get_info(self, index):
        # (free, total). The planner must read the first, not the second.
        return (self._free.get(index, 8 * 2**30), 16 * 2**30)


def _load(
    *,
    devices = 2,
    device_type = "cuda",
    free = None,
    planner = None,
    distributed = False,
):
    """Rebuild the two functions over a fabricated CUDA and unsloth_zoo."""
    ns = {
        "os": os,
        "torch": types.SimpleNamespace(cuda = _FakeCuda(devices, free)),
        "DEVICE_TYPE_TORCH": device_type,
        "is_distributed": lambda: distributed,
    }
    for node in ast.parse(_SRC).body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "requested_device_map",
            "resolve_unsloth_device_map",
        ):
            exec(ast.get_source_segment(_SRC, node), ns)
        elif (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", None) == "UNSLOTH_DEVICE_MAP"
        ):
            exec(ast.get_source_segment(_SRC, node), ns)

    if planner is not None:
        module = types.ModuleType("unsloth_zoo.device_map_planner")
        module.plan_device_map_for_pretrained = planner
        sys.modules["unsloth_zoo.device_map_planner"] = module
    return ns


class _Plan:
    def __init__(self, device_map):
        self.device_map = device_map

    def describe(self):
        return "  (fabricated plan)"


# ------------------------------------------------- what an existing caller still gets


@pytest.mark.parametrize(
    "device_map",
    [
        "sequential",  # today's default
        "auto",  # accelerate's, which this must never reinterpret
        "balanced",  # what Unsloth Studio passes
        "balanced_low_0",
        None,  # a single device
    ],
)
def test_every_existing_device_map_is_returned_untouched(device_map):
    """The whole opt-in claim in one test. If any of these changed, every multi-GPU user
    who never asked for planning would silently get a different placement."""
    ns = _load(planner = lambda *a, **k: pytest.fail("the planner must not run"))
    assert ns["resolve_unsloth_device_map"](device_map, "some/model") is device_map


def test_an_explicit_dict_is_returned_untouched():
    explicit = {"": 0, "model.vision_tower": 1}
    ns = _load(planner = lambda *a, **k: pytest.fail("the planner must not run"))
    assert ns["resolve_unsloth_device_map"](explicit, "some/model") is explicit


def test_the_env_var_only_upgrades_the_default(monkeypatch):
    """UNSLOTH_AUTO_DEVICE_MAP is an operator switch, not a licence to override a
    placement the caller chose. "auto" and a dict must survive it."""
    ns = _load()
    monkeypatch.setenv("UNSLOTH_AUTO_DEVICE_MAP", "1")
    assert ns["requested_device_map"]("sequential") == "unsloth"
    assert ns["requested_device_map"]("auto") == "auto"
    assert ns["requested_device_map"]("balanced") == "balanced"
    assert ns["requested_device_map"]({"": 0}) == {"": 0}
    monkeypatch.delenv("UNSLOTH_AUTO_DEVICE_MAP")
    assert ns["requested_device_map"]("sequential") == "sequential"


# ------------------------------------------------------- where planning cannot apply


@pytest.mark.parametrize(
    "kwargs,why",
    [
        ({"fast_inference": True}, "vLLM places its own weights"),
        ({"full_finetuning": True}, "no bnb layout to plan"),
    ],
)
def test_planning_is_declined_where_something_else_owns_placement(kwargs, why):
    ns = _load(planner = lambda *a, **k: pytest.fail(f"must not plan: {why}"))
    assert ns["resolve_unsloth_device_map"]("unsloth", "m", **kwargs) == "sequential"


def test_a_distributed_launch_never_gets_an_intra_model_split():
    """torchrun/DDP/FSDP already put one whole model per rank. Splitting a model across
    the cards on top of that puts every rank on every card, which OOMs rather than fits.

    prepare_device_map() in loader.py converts the string to a rank-local dict first, but
    only when the load is quantized, so a 16-bit distributed run would still arrive here
    holding "unsloth". Hence the gate lives here too.
    """
    ns = _load(
        distributed = True, planner = lambda *a, **k: pytest.fail("planned inside a distributed launch")
    )
    assert ns["resolve_unsloth_device_map"]("unsloth", "m") == "sequential"


@pytest.mark.parametrize("device_type", ["xpu", "mps", "cpu", "hpu"])
def test_a_non_cuda_backend_never_reaches_the_cuda_planner(device_type):
    """The planner sizes cards through torch.cuda. On XPU or MPS that is either absent or
    lying, so falling back beats planning against numbers from the wrong device."""
    ns = _load(
        device_type = device_type,
        planner = lambda *a, **k: pytest.fail("CUDA planner on a non-CUDA backend"),
    )
    assert ns["resolve_unsloth_device_map"]("unsloth", "m") == "sequential"


@pytest.mark.parametrize("devices", [0, 1])
def test_one_gpu_or_none_falls_back_silently(devices):
    """Not a failure, just nothing to split across, so it prints nothing."""
    ns = _load(devices = devices, planner = lambda *a, **k: pytest.fail("nothing to plan across"))
    assert ns["resolve_unsloth_device_map"]("unsloth", "m") == "sequential"


def test_a_planner_that_declines_falls_back():
    ns = _load(planner = lambda *a, **k: None)
    assert ns["resolve_unsloth_device_map"]("unsloth", "m") == "sequential"


def test_a_planner_that_raises_falls_back_rather_than_failing_the_load():
    """A model that loads the old way beats one that will not load at all."""

    def _boom(*a, **k):
        raise RuntimeError("hub unreachable")

    ns = _load(planner = _boom)
    assert ns["resolve_unsloth_device_map"]("unsloth", "m") == "sequential"


def test_an_infeasible_plan_is_raised_not_swallowed():
    """The planner raises this instead of spilling a bitsandbytes model to CPU. Turning it
    back into "sequential" would hand the user an OOM in place of the diagnosis."""

    class DeviceMapInfeasible(RuntimeError):
        pass

    def _infeasible(*a, **k):
        raise DeviceMapInfeasible("needs 7.57 GiB free on cuda:0, has 4.10 GiB")

    ns = _load(planner = _infeasible)
    with pytest.raises(DeviceMapInfeasible):
        ns["resolve_unsloth_device_map"]("unsloth", "m")


# ------------------------------------------------------------- when it does plan


def test_the_plan_is_returned_and_the_model_name_reaches_the_planner():
    seen = {}

    def _planner(model_name, **kwargs):
        seen.update(kwargs, model_name = model_name)
        return _Plan({"": 0, "model.vision_tower": 1})

    ns = _load(planner = _planner)
    result = ns["resolve_unsloth_device_map"](
        "unsloth",
        "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit",
        load_in_4bit = True,
    )
    assert result == {"": 0, "model.vision_tower": 1}
    assert seen["model_name"] == "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit"
    assert seen["load_in_4bit"] is True


def test_free_memory_is_planned_against_not_total():
    """Two 16 GB cards with a CUDA context already resident have far less than 16 GB to
    give. Planning against total is what made the notebook's first attempt OOM
    (unsloth-zoo#1048)."""
    seen = {}
    ns = _load(
        free = {0: 4 * 2**30, 1: 15 * 2**30},
        planner = lambda name, **kw: seen.update(kw) or _Plan({"": 0}),
    )
    ns["resolve_unsloth_device_map"]("unsloth", "m")
    assert seen["max_memory"] == {0: 4 * 2**30, 1: 15 * 2**30}
    assert 16 * 2**30 not in seen["max_memory"].values(), "that is the total, not the free"


def test_planning_happens_only_where_the_model_name_is_final():
    """loader.py remaps model_name (a -bnb-4bit repo can resolve to its 16-bit twin, and
    BAD_MAPPINGS rewrites several Qwen3 repos outright) well after its device_map block.
    A plan built up there is sized for a repo that is not the one loaded, so the call
    belongs in llama.py and vision.py, where the name has stopped moving.
    """
    models = os.path.join(HERE, "unsloth", "models")
    loader = open(os.path.join(models, "loader.py"), encoding = "utf-8").read()
    assert (
        "resolve_unsloth_device_map(" not in loader
    ), "loader.py plans before get_model_name has had its say"
    for name in ("llama.py", "vision.py"):
        source = open(os.path.join(models, name), encoding = "utf-8").read()
        assert "resolve_unsloth_device_map(" in source, name


def test_every_entry_point_accepts_the_planner_kwargs():
    """FastLanguageModel and FastModel only forward what they were given, so a signature
    that quietly lacks the parameter turns the notebook's hint into a TypeError."""
    import ast as _ast

    models = os.path.join(HERE, "unsloth", "models")
    for name in ("loader.py", "llama.py", "vision.py"):
        source = open(os.path.join(models, name), encoding = "utf-8").read()
        found = [
            node
            for node in _ast.walk(_ast.parse(source))
            if isinstance(node, _ast.FunctionDef)
            and node.name == "from_pretrained"
            and any(a.arg == "device_map" for a in node.args.args + node.args.kwonlyargs)
        ]
        assert found, f"no from_pretrained taking device_map in {name}"
        for node in found:
            args = {a.arg for a in node.args.args + node.args.kwonlyargs}
            assert "device_map_planner_kwargs" in args, f"{name}:{node.lineno}"


def test_planner_kwargs_reach_the_planner():
    """A GRPO backward retains rows the planner's inference-shaped default (0) does not
    reserve for, so the notebook has to be able to say so."""
    seen = {}
    ns = _load(planner = lambda name, **kw: seen.update(kw) or _Plan({"": 0}))
    ns["resolve_unsloth_device_map"](
        "unsloth",
        "m",
        planner_kwargs = {"rows_per_chunk": 128, "retained_rows": 6144, "softcapped": True},
    )
    assert seen["rows_per_chunk"] == 128
    assert seen["retained_rows"] == 6144
    assert seen["softcapped"] is True
