# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The opt-in device map, over the whole product of host and accelerator.

test_unsloth_device_map_optin.py checks each decline path once. This file checks that the
three properties that make the change safe hold across every combination of them at once,
because the risk is not one path being wrong, it is one *combination* being wrong on a
machine none of us has:

  1. Nothing that is not the sentinel is touched. Every device_map an existing caller can
     pass comes back identical, whatever the host and whatever the accelerator.
  2. The sentinel never escapes. `resolve_unsloth_device_map` never returns "unsloth" --
     transformers turns an unknown device_map string into `torch.device("unsloth")` and
     raises, so a leak is a hard load failure rather than a bad placement.
  3. The planner is called only where a plan can apply, and never otherwise.

The host axis is Linux / Windows / WSL / macOS and the accelerator axis is NVIDIA (cuda),
AMD (cuda, since torch's ROCm build reports itself as cuda), Intel (xpu), Apple (mps) and
CPU. `resolve_unsloth_device_map` does not read the platform itself, which is the point:
these spoofs exist to prove no platform-specific branch grew in underneath it.

Extracted with ast so nothing has to import torch's CUDA stack.
"""

import ast
import itertools
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOADER_UTILS = os.path.join(HERE, "unsloth", "models", "loader_utils.py")
_SRC = open(LOADER_UTILS, encoding = "utf-8").read()

# (label, sys.platform, os.name, an /proc/version marker for the WSL case)
HOSTS = [
    ("linux", "linux", "posix", "Linux version 6.8.0-generic"),
    ("windows", "win32", "nt", None),
    ("wsl", "linux", "posix", "Linux version 5.15.0-microsoft-standard-WSL2"),
    ("macos", "darwin", "posix", None),
]

# (label, DEVICE_TYPE_TORCH). A ROCm torch build reports "cuda", so AMD is not a separate
# branch in the resolver -- it is here so a future one cannot be added unnoticed.
ACCELERATORS = [
    ("nvidia", "cuda"),
    ("amd", "cuda"),
    ("intel", "xpu"),
    ("apple", "mps"),
    ("cpu", "cpu"),
]

DEVICE_COUNTS = [0, 1, 2, 8]

# Everything a caller can hand the loader today.
# None is included because `FastDiffusionModel.from_pretrained` lets the caller clear it.
UNTOUCHED_DEVICE_MAPS = [
    "sequential",
    "auto",
    "balanced",
    "balanced_low_0",
    "cuda:0",
    "cuda:1",
    "cpu",
    "mps",
    "xpu:0",
    None,
    0,
    {"": 0},
    {"": "cuda:0"},
    {"model.embed_tokens": 0, "lm_head": 1},
]


class _FakeCuda:
    def __init__(
        self,
        count,
        count_raises = None,
        mem_raises = None,
    ):
        self._count = count
        self._count_raises = count_raises
        self._mem_raises = mem_raises

    def device_count(self):
        if self._count_raises is not None:
            raise self._count_raises
        return self._count

    def mem_get_info(self, index):
        if self._mem_raises is not None:
            raise self._mem_raises
        return (8 * 2**30, 16 * 2**30)


class _Recorder:
    """Stands in for unsloth_zoo's planner and records whether it was consulted."""

    def __init__(
        self,
        plan = None,
        raises = None,
    ):
        self.calls = []
        self._plan = plan
        self._raises = raises

    def __call__(self, model_name, **kwargs):
        self.calls.append((model_name, kwargs))
        if self._raises is not None:
            raise self._raises
        return self._plan


class _Plan:
    device_map = {"model.embed_tokens": 0, "lm_head": 1}

    def describe(self):
        return "<plan>"


def _build(
    *,
    device_type,
    devices,
    distributed,
    planner,
    planner_available = True,
    count_raises = None,
    mem_raises = None,
):
    """Rebuild the resolver over a fabricated torch, unsloth_zoo and host."""
    ns = {
        "os": os,
        "torch": types.SimpleNamespace(
            cuda = _FakeCuda(devices, count_raises = count_raises, mem_raises = mem_raises)
        ),
        "DEVICE_TYPE_TORCH": device_type,
        "is_distributed": lambda: distributed,
    }
    for node in ast.parse(_SRC).body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "requested_device_map",
            "resolve_unsloth_device_map",
            "_as_bytes",
        ):
            exec(ast.get_source_segment(_SRC, node), ns)
        elif isinstance(node, ast.ClassDef) and node.name == "_DefaultDeviceMap":
            exec(ast.get_source_segment(_SRC, node), ns)
        elif isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in (
            "UNSLOTH_DEVICE_MAP",
            "UNSLOTH_BALANCED_DEVICE_MAP",
            "_PLANNED_DEVICE_MAPS",
            "DEFAULT_DEVICE_MAP",
            "_SIZE_UNITS",
        ):
            exec(ast.get_source_segment(_SRC, node), ns)

    planner_module = types.ModuleType("unsloth_zoo.device_map_planner")
    if planner_available:
        planner_module.plan_device_map_for_pretrained = planner
    sys.modules["unsloth_zoo.device_map_planner"] = planner_module
    return ns


@pytest.fixture
def host(request, monkeypatch):
    """Spoof the operating system around a case, so a platform branch cannot hide."""
    label, platform_name, os_name, proc_version = request.param
    monkeypatch.setattr(sys, "platform", platform_name, raising = False)
    monkeypatch.setattr(os, "name", os_name, raising = False)
    monkeypatch.setenv("UNSLOTH_TEST_HOST", label)
    if proc_version is not None:
        monkeypatch.setenv("UNSLOTH_TEST_PROC_VERSION", proc_version)
    return label


_HOST_IDS = [h[0] for h in HOSTS]


@pytest.mark.parametrize("host", HOSTS, ids = _HOST_IDS, indirect = True)
@pytest.mark.parametrize("accelerator,device_type", ACCELERATORS, ids = [a[0] for a in ACCELERATORS])
@pytest.mark.parametrize("devices", DEVICE_COUNTS)
@pytest.mark.parametrize("device_map", UNTOUCHED_DEVICE_MAPS, ids = repr)
def test_an_existing_device_map_is_identical_on_every_host(
    host, accelerator, device_type, devices, device_map, monkeypatch
):
    """Property 1. 4 hosts x 5 accelerators x 4 GPU counts x 14 placements = 1120 cases,
    none of which may differ by so much as an object identity from what main returns."""
    monkeypatch.delenv("UNSLOTH_AUTO_DEVICE_MAP", raising = False)
    planner = _Recorder(plan = _Plan())
    ns = _build(device_type = device_type, devices = devices, distributed = False, planner = planner)
    resolved = ns["resolve_unsloth_device_map"](
        ns["requested_device_map"](device_map), "unsloth/Qwen3-0.6B"
    )
    assert resolved is device_map
    assert planner.calls == []


@pytest.mark.parametrize("host", HOSTS, ids = _HOST_IDS, indirect = True)
@pytest.mark.parametrize("accelerator,device_type", ACCELERATORS, ids = [a[0] for a in ACCELERATORS])
@pytest.mark.parametrize("devices", DEVICE_COUNTS)
@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("fast_inference", [False, True])
@pytest.mark.parametrize("full_finetuning", [False, True])
@pytest.mark.parametrize("planner_available", [False, True])
def test_the_sentinel_never_reaches_transformers(
    host,
    accelerator,
    device_type,
    devices,
    distributed,
    fast_inference,
    full_finetuning,
    planner_available,
    monkeypatch,
):
    """Property 2, the one that decides whether this can break a load anywhere.

    Whatever the host, the accelerator, the GPU count, the launcher, the vLLM/full-finetune
    flags, and whether unsloth_zoo is new enough to have a planner at all, the resolved
    value is either a placement transformers understands or a plan dict. Never "unsloth".
    """
    monkeypatch.delenv("UNSLOTH_AUTO_DEVICE_MAP", raising = False)
    planner = _Recorder(plan = _Plan())
    ns = _build(
        device_type = device_type,
        devices = devices,
        distributed = distributed,
        planner = planner,
        planner_available = planner_available,
    )
    resolved = ns["resolve_unsloth_device_map"](
        "unsloth",
        "unsloth/Qwen3-0.6B",
        fast_inference = fast_inference,
        full_finetuning = full_finetuning,
    )
    assert resolved != "unsloth"
    assert resolved == "sequential" or isinstance(resolved, dict)


@pytest.mark.parametrize("host", HOSTS, ids = _HOST_IDS, indirect = True)
@pytest.mark.parametrize("accelerator,device_type", ACCELERATORS, ids = [a[0] for a in ACCELERATORS])
@pytest.mark.parametrize("devices", DEVICE_COUNTS)
@pytest.mark.parametrize("distributed", [False, True])
def test_the_planner_runs_exactly_where_a_plan_can_apply(
    host, accelerator, device_type, devices, distributed, monkeypatch
):
    """Property 3, stated as the whole truth table rather than one path at a time."""
    monkeypatch.delenv("UNSLOTH_AUTO_DEVICE_MAP", raising = False)
    planner = _Recorder(plan = _Plan())
    ns = _build(
        device_type = device_type,
        devices = devices,
        distributed = distributed,
        planner = planner,
    )
    resolved = ns["resolve_unsloth_device_map"]("unsloth", "unsloth/Qwen3-0.6B")

    should_plan = device_type == "cuda" and devices >= 2 and not distributed
    assert bool(planner.calls) is should_plan
    if should_plan:
        assert resolved == _Plan.device_map
    else:
        assert resolved == "sequential"


@pytest.mark.parametrize("host", HOSTS, ids = _HOST_IDS, indirect = True)
@pytest.mark.parametrize("accelerator,device_type", ACCELERATORS, ids = [a[0] for a in ACCELERATORS])
@pytest.mark.parametrize("devices", DEVICE_COUNTS)
@pytest.mark.parametrize("value", ["0", "", "false", "no", "true", "1"])
def test_the_env_var_opts_in_on_1_and_nothing_else(
    host, accelerator, device_type, devices, value, monkeypatch
):
    """`UNSLOTH_AUTO_DEVICE_MAP` is an operator switch, so a half-set one must be off, not
    ambiguous. Only the literal "1" upgrades, on every host -- and only the default, never
    the same string handed over by a caller who meant it."""
    monkeypatch.setenv("UNSLOTH_AUTO_DEVICE_MAP", value)
    planner = _Recorder(plan = _Plan())
    ns = _build(device_type = device_type, devices = devices, distributed = False, planner = planner)
    requested = ns["requested_device_map"](ns["DEFAULT_DEVICE_MAP"])
    assert requested == ("unsloth" if value == "1" else "sequential")
    assert ns["resolve_unsloth_device_map"](requested, "unsloth/Qwen3-0.6B") != "unsloth"
    assert ns["requested_device_map"]("sequential") == "sequential"


@pytest.mark.parametrize("host", HOSTS, ids = _HOST_IDS, indirect = True)
def test_an_old_unsloth_zoo_without_a_planner_still_loads(host, monkeypatch):
    """An install that predates unsloth_zoo's planner must degrade, not fail: the whole
    point of the fallback is that a model which loads the old way beats one that will not
    load. Two shapes of old: the module is missing, and the module exists without the
    entry point."""
    monkeypatch.delenv("UNSLOTH_AUTO_DEVICE_MAP", raising = False)

    # Shape one: the module is there but predates the entry point.
    ns = _build(
        device_type = "cuda", devices = 4, distributed = False, planner = None, planner_available = False
    )
    assert ns["resolve_unsloth_device_map"]("unsloth", "unsloth/Qwen3-0.6B") == "sequential"

    # Shape two: no such module.
    # Block the import rather than deleting it from sys.modules, which on a machine that has the real planner installed
    # just imports it again.
    class _Blocked:
        def find_module(
            self,
            name,
            path = None,
        ):
            return None

        def find_spec(
            self,
            name,
            path = None,
            target = None,
        ):
            if name == "unsloth_zoo.device_map_planner":
                raise ModuleNotFoundError(f"No module named {name!r}")
            return None

    ns = _build(device_type = "cuda", devices = 4, distributed = False, planner = None)
    sys.modules.pop("unsloth_zoo.device_map_planner", None)
    blocker = _Blocked()
    sys.meta_path.insert(0, blocker)
    try:
        assert ns["resolve_unsloth_device_map"]("unsloth", "unsloth/Qwen3-0.6B") == "sequential"
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.pop("unsloth_zoo.device_map_planner", None)


@pytest.mark.parametrize("host", HOSTS, ids = _HOST_IDS, indirect = True)
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("planner blew up"),
        ValueError("bad config"),
        OSError("no network"),
        KeyError("lm_head"),
    ],
    ids = ["runtime", "value", "os", "key"],
)
def test_a_planner_that_raises_anything_but_infeasible_falls_back(host, error, monkeypatch):
    """Everything except the deliberate refusal degrades to the old placement."""
    monkeypatch.delenv("UNSLOTH_AUTO_DEVICE_MAP", raising = False)
    planner = _Recorder(raises = error)
    ns = _build(device_type = "cuda", devices = 2, distributed = False, planner = planner)
    assert ns["resolve_unsloth_device_map"]("unsloth", "unsloth/Qwen3-0.6B") == "sequential"


@pytest.mark.parametrize("host", HOSTS, ids = _HOST_IDS, indirect = True)
def test_the_deliberate_refusal_is_not_swallowed(host, monkeypatch):
    """`DeviceMapInfeasible` is the planner declining to place a model that would OOM.
    Turning it into "sequential" would hand the user the OOM instead of the diagnosis.
    Matched by name, because an old unsloth_zoo may not export the class."""
    monkeypatch.delenv("UNSLOTH_AUTO_DEVICE_MAP", raising = False)

    class DeviceMapInfeasible(RuntimeError):
        pass

    planner = _Recorder(raises = DeviceMapInfeasible("2 x 8 GiB is not enough"))
    ns = _build(device_type = "cuda", devices = 2, distributed = False, planner = planner)
    with pytest.raises(DeviceMapInfeasible):
        ns["resolve_unsloth_device_map"]("unsloth", "unsloth/Qwen3-0.6B")


@pytest.mark.parametrize("host", HOSTS, ids = _HOST_IDS, indirect = True)
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("CUDA error: uncorrectable ECC error encountered"),
        RuntimeError("CUDA error: all CUDA-capable devices are busy or unavailable"),
        RuntimeError("CUDA driver initialization failed"),
    ],
    ids = ["ecc", "exclusive-process", "driver"],
)
def test_a_card_that_refuses_to_report_memory_does_not_fail_the_load(host, error, monkeypatch):
    """Reading free memory is itself a CUDA call on every visible device, and it is the
    first thing this function does that can touch a broken one: an ECC-fenced card, a MIG
    parent handle, or a GPU another process holds in Exclusive_Process mode. That must
    degrade to the placement the caller would have had anyway, for the same reason a
    planner exception does. Nothing here is the deliberate refusal, which still raises."""
    monkeypatch.delenv("UNSLOTH_AUTO_DEVICE_MAP", raising = False)
    planner = _Recorder(plan = _Plan())
    ns = _build(
        device_type = "cuda",
        devices = 4,
        distributed = False,
        planner = planner,
        mem_raises = error,
    )
    assert ns["resolve_unsloth_device_map"]("unsloth", "unsloth/Qwen3-0.6B") == "sequential"
    assert planner.calls == []


@pytest.mark.parametrize("host", HOSTS, ids = _HOST_IDS, indirect = True)
def test_a_device_count_that_raises_does_not_fail_the_load(host, monkeypatch):
    """Same reasoning one call earlier. `device_count()` swallows most driver faults and
    answers 0, but not all of them, and a load with a working `sequential` placement should
    not die because the count could not be taken."""
    monkeypatch.delenv("UNSLOTH_AUTO_DEVICE_MAP", raising = False)
    planner = _Recorder(plan = _Plan())
    ns = _build(
        device_type = "cuda",
        devices = 4,
        distributed = False,
        planner = planner,
        count_raises = RuntimeError("CUDA unknown error"),
    )
    assert ns["resolve_unsloth_device_map"]("unsloth", "unsloth/Qwen3-0.6B") == "sequential"
    assert planner.calls == []


def _planner_quantization_kwargs():
    ns = {"os": os}
    for node in ast.parse(_SRC).body:
        if isinstance(node, ast.FunctionDef) and node.name == "planner_quantization_kwargs":
            exec(ast.get_source_segment(_SRC, node), ns)
    return ns["planner_quantization_kwargs"]


@pytest.mark.parametrize("host", HOSTS, ids = _HOST_IDS, indirect = True)
@pytest.mark.parametrize("four_bit,eight_bit", [(True, False), (False, True)], ids = ["4bit", "8bit"])
def test_a_zoo_without_the_shared_skip_list_still_loads_in_4bit(host, four_bit, eight_bit):
    """The leaf loaders evaluate these arguments on every quantized load, whether or not
    anything is going to be planned. So the one import in here is on the hot path of every
    4bit load in the library, and an unsloth_zoo below our pin has to degrade to "no skip
    list" rather than take the load down with an ImportError."""
    build = _planner_quantization_kwargs()

    peft_utils = types.ModuleType("unsloth_zoo.peft_utils")  # no SKIP_QUANTIZATION_MODULES
    saved = sys.modules.get("unsloth_zoo.peft_utils")
    sys.modules["unsloth_zoo.peft_utils"] = peft_utils
    try:
        kwargs = build(load_in_4bit = four_bit, load_in_8bit = eight_bit)
    finally:
        if saved is None:
            sys.modules.pop("unsloth_zoo.peft_utils", None)
        else:
            sys.modules["unsloth_zoo.peft_utils"] = saved

    assert kwargs == {"load_in_4bit": four_bit, "load_in_8bit": eight_bit}


def test_the_matrix_is_actually_the_product_we_claim():
    """A guard on the guard: if someone trims a list above, the coverage claim in the
    docstrings should stop being true loudly rather than quietly."""
    assert len(list(itertools.product(HOSTS, ACCELERATORS, DEVICE_COUNTS))) == 80
    assert len(UNTOUCHED_DEVICE_MAPS) == 14
