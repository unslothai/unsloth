# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests the opt-in multi-GPU planning in loader_utils.py. No GPU needed.

`device_map = "unsloth"` asks unsloth_zoo's planner for a head-aware placement instead of
accelerate's `"sequential"`. The Muse Glimmer GRPO notebook does this by hand today, in
about 25 lines of mem_get_info arithmetic.

It is opt-in because the alternative is not safe: an existing multi-GPU caller who never
asked for planning must keep the placement they have. So most of this file is about what
must NOT change, and only the last group is about planning working.

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
_SKIP_MODULES = ["lm_head", "vision_tower", "audio_tower"]


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
            "planner_quantization_kwargs",
            "planner_class_mismatch_reason",
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

    # planner_quantization_kwargs reads the shared skip list;
    # stub it so the test never imports the real unsloth_zoo (and so the assertions do not track its contents).
    peft_utils = types.ModuleType("unsloth_zoo.peft_utils")
    peft_utils.SKIP_QUANTIZATION_MODULES = list(_SKIP_MODULES)
    sys.modules["unsloth_zoo.peft_utils"] = peft_utils

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


@pytest.mark.parametrize("switch", [None, "1"])
def test_only_the_default_is_ever_upgraded(monkeypatch, switch):
    """Planning is what a caller who chose nothing gets, never a licence to override one
    they did choose. "auto", a dict, and a "sequential" they typed out all survive it --
    hence the marker, since the last of those is the same string as the default.
    """
    ns = _load()
    if switch is None:
        monkeypatch.delenv("UNSLOTH_AUTO_DEVICE_MAP", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_AUTO_DEVICE_MAP", switch)
    assert ns["requested_device_map"](ns["DEFAULT_DEVICE_MAP"]) == "unsloth"
    assert ns["requested_device_map"]("sequential") == "sequential"
    assert ns["requested_device_map"]("auto") == "auto"
    assert ns["requested_device_map"]("balanced") == "balanced"
    assert ns["requested_device_map"]({"": 0}) == {"": 0}


def test_the_env_var_can_turn_planning_back_off(monkeypatch):
    """The multi-GPU operator who wants accelerate's greedy fill back needs a switch that
    does not require editing call sites, so `0` has to reach the default itself."""
    ns = _load()
    monkeypatch.setenv("UNSLOTH_AUTO_DEVICE_MAP", "0")
    assert ns["requested_device_map"](ns["DEFAULT_DEVICE_MAP"]) == "sequential"
    # Still a plain "sequential" downstream, marker and all.
    assert ns["requested_device_map"](ns["DEFAULT_DEVICE_MAP"]) == ns["DEFAULT_DEVICE_MAP"]


def test_an_unset_switch_plans_so_a_bare_from_pretrained_needs_no_device_map(monkeypatch):
    """The reason the default flipped: a notebook should not have to pass
    `device_map = "unsloth"` to get the placement that fits."""
    monkeypatch.delenv("UNSLOTH_AUTO_DEVICE_MAP", raising = False)
    planned = {"model.embed_tokens": 0, "lm_head": 1}
    calls = []
    ns = _load(
        devices = 2,
        free = {0: 16 * 2**30, 1: 16 * 2**30},
        planner = lambda name, **kw: calls.append(name) or _Plan(planned),
    )
    resolved = ns["resolve_unsloth_device_map"](
        ns["requested_device_map"](ns["DEFAULT_DEVICE_MAP"]),
        "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit",
    )
    assert resolved == planned
    assert calls == ["unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit"]


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


def test_a_caller_that_vetoes_planning_is_obeyed():
    """Only the leaf knows when the config it is about to load is not the one the planner
    would rebuild from the repo, so it needs a way to say so."""
    ns = _load(planner = lambda *a, **k: pytest.fail("planned despite the veto"))
    assert ns["resolve_unsloth_device_map"]("unsloth", "m", skip_reason = "text_only") == "sequential"
    # A veto is not a licence to reinterpret a placement the caller chose.
    assert ns["resolve_unsloth_device_map"]("auto", "m", skip_reason = "text_only") == "auto"


def test_a_text_only_decoder_is_never_planned_against_the_full_vlm():
    """`text_only = True` loads a VLM's standalone decoder, so Gemma 3 builds
    Gemma3ForCausalLM (`model.layers.0`). The planner only gets `model_name`, rebuilds the
    repo's multimodal config and plans Gemma3ForConditionalGeneration
    (`model.language_model.layers.0`, plus a vision tower this load never creates). Not one
    decoder weight matches a key of that map, and transformers raises
    "model.embed_tokens.weight doesn't have any device set" for the first of them.
    """
    models = os.path.join(HERE, "unsloth", "models")

    vision = open(os.path.join(models, "vision.py"), encoding = "utf-8").read()
    tree = ast.parse(vision)
    signature = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "from_pretrained"
        and any(a.arg == "text_only" for a in node.args.args + node.args.kwonlyargs)
    ]
    assert signature, "vision.py no longer takes text_only"
    for node in signature:
        args = {a.arg for a in node.args.args + node.args.kwonlyargs}
        assert "text_only_decoder" in args, f"vision.py:{node.lineno}"
    # The direct-call path resolves the text config itself, so it has to raise the flag.
    assert "text_only_decoder = True" in vision

    # The veto reaches the planner call, and whatever it is spelled as is decided by the flag.
    assignments = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = assignments.get(target.id, "") + ast.unparse(node.value)
    for call in _resolve_calls(vision):
        passed = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        assert "skip_reason" in passed, f"vision.py:{call.lineno} plans a text-only decoder"
        source = passed["skip_reason"] + assignments.get(passed["skip_reason"], "")
        assert "text_only_decoder" in source, f"vision.py:{call.lineno}"
        # The other way the load can diverge from the plan; see the task-head test.
        assert "planner_class_mismatch_reason" in source, f"vision.py:{call.lineno}"

    # loader.py does the swap for FastModel/FastLanguageModel, so it has to say so too.
    loader = open(os.path.join(models, "loader.py"), encoding = "utf-8").read()
    assert "text_only_decoder = True" in loader
    forwarded = [
        node
        for node in ast.walk(ast.parse(loader))
        if isinstance(node, ast.Call) and any(kw.arg == "text_only_decoder" for kw in node.keywords)
    ]
    assert forwarded, "loader.py swaps the config but never tells the leaf"


def test_a_task_head_the_planner_cannot_see_declines_planning():
    """`num_labels` makes the load AutoModelForSequenceClassification, whose `score`
    replaces `lm_head`. The planner sees only `model_name`, reads the repo's own
    `LlamaForCausalLM` and emits units ending in `lm_head`, so dispatch refuses the map:
    "does not give any device for ... score.weight".

    Compared as model classes, since AutoModelForVision2Seq and AutoModelForImageTextToText
    are different objects building the same VLM and would decline planning for every VLM.
    """
    ns = _load()
    mismatch = ns["planner_class_mismatch_reason"]

    class LlamaForCausalLM:
        pass

    class LlamaForSequenceClassification:
        pass

    reason = mismatch(LlamaForSequenceClassification, LlamaForCausalLM)
    assert reason and "LlamaForSequenceClassification" in reason
    assert mismatch(LlamaForCausalLM, LlamaForCausalLM) is None
    # Unknown is not mismatched: an unsloth_zoo too old to name the class has no planner.
    assert mismatch(LlamaForCausalLM, None) is None
    assert mismatch(None, LlamaForCausalLM) is None

    ns = _load(planner = lambda *a, **k: pytest.fail("planned a head the plan does not name"))
    assert ns["resolve_unsloth_device_map"]("unsloth", "m", skip_reason = reason) == "sequential"


def test_the_optimized_llama_path_also_declines_a_classification_load():
    """The same veto has to live on llama.py's own planner call, not just vision.py's.

    loader.py delegates to FastModel only for 8bit / full finetuning / QAT, so
    `FastLanguageModel.from_pretrained(..., num_labels = 2, device_map = "unsloth")` on a
    llama/mistral/gemma/qwen repo dispatches to FastLlamaModel, plans the repo's causal LM,
    then loads AutoModelForSequenceClassification a few lines later. That model has `score`
    and no `lm_head`, so `dispatch_model` -> `check_device_map` raises.
    """
    llama = open(os.path.join(HERE, "unsloth", "models", "llama.py"), encoding = "utf-8").read()
    tree = ast.parse(llama)

    assignments = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = assignments.get(target.id, "") + ast.unparse(node.value)

    calls = _resolve_calls(llama)
    assert calls, "llama.py no longer plans a device map"
    for call in calls:
        passed = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        assert "skip_reason" in passed, f"llama.py:{call.lineno} plans a classification load"
        source = passed["skip_reason"] + assignments.get(passed["skip_reason"], "")
        assert "num_labels" in source, f"llama.py:{call.lineno}"
        assert "planner_class_mismatch_reason" in source, f"llama.py:{call.lineno}"

    # The veto must be decided before the call, or it is a NameError on every load.
    veto_line = min(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "_planner_skip_reason" for t in node.targets)
    )
    assert veto_line < min(call.lineno for call in calls)


def test_a_distributed_launch_never_gets_an_intra_model_split():
    """torchrun/DDP/FSDP already put one whole model per rank; splitting a model across the
    cards on top of that puts every rank on every card, which OOMs rather than fits.

    prepare_device_map() in loader.py converts the string to a rank-local dict first, but
    only when the load is quantized, so a 16-bit distributed run still arrives here holding
    "unsloth". Hence the gate lives here too.
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


@pytest.mark.parametrize(
    "kwargs,devices,planner",
    [
        ({"skip_reason": "text_only"}, 2, None),
        ({"fast_inference": True}, 2, None),
        ({"full_finetuning": True}, 2, None),
        ({}, 2, lambda *a, **k: None),
        ({}, 2, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("hub unreachable"))),
    ],
)
def test_the_balanced_sentinel_declines_to_balanced_not_sequential(kwargs, devices, planner):
    """`"unsloth_balanced"` is the same plan with a different answer when it is declined.

    "sequential" is not a shard: `get_max_memory` gives cuda:0 its whole free budget, so
    `infer_auto_device_map` fills it first and a model that fits lands there whole. On
    `unsloth/Qwen2.5-7B-Instruct` in bf16 across two cards, 16 GiB each, "sequential"
    answers {'0': 1} where "balanced" answers {'0': 13, '1': 19}. A caller that asked to
    plan across several cards wants a split even when the planner declines, and it
    declines on more shapes than a caller can enumerate -- a full finetune, an
    `auto_model` with no `_model_mapping`, a prequantized Falcon-H1 checkpoint.
    """
    ns = _load(devices = devices, planner = planner)
    assert ns["resolve_unsloth_device_map"]("unsloth_balanced", "m", **kwargs) == "balanced"
    # The plain sentinel is unchanged: an existing caller keeps the answer it had.
    assert ns["resolve_unsloth_device_map"]("unsloth", "m", **kwargs) == "sequential"


@pytest.mark.parametrize("devices", [0, 1])
def test_the_balanced_sentinel_declines_to_balanced_on_one_device_too(devices):
    """The silent fallbacks are the easy ones to leave hardcoded, and both were."""
    ns = _load(devices = devices, planner = lambda *a, **k: pytest.fail("nothing to plan"))
    assert ns["resolve_unsloth_device_map"]("unsloth_balanced", "m") == "balanced"


def test_both_names_plan_identically_when_the_planner_answers():
    """The name chooses the fallback and nothing else, so a plan is not a second code
    path that could drift."""
    planned = {"": 0, "model.vision_tower": 1}
    for name in ("unsloth", "unsloth_balanced"):
        ns = _load(planner = lambda *a, **k: _Plan(dict(planned)))
        assert ns["resolve_unsloth_device_map"](name, "m") == planned


def test_an_infeasible_plan_is_still_raised_for_the_balanced_name():
    """The one deliberate raise must not be softened into a shard by the new name."""

    class DeviceMapInfeasible(RuntimeError):
        pass

    def _infeasible(*a, **k):
        raise DeviceMapInfeasible("needs 7.57 GiB free on cuda:0, has 4.10 GiB")

    ns = _load(planner = _infeasible)
    with pytest.raises(DeviceMapInfeasible):
        ns["resolve_unsloth_device_map"]("unsloth_balanced", "m")


def test_the_default_device_map_still_resolves_to_the_plain_sentinel():
    """An omitted device_map becomes the planner, and its decline has always been
    "sequential" -- which is also what an omitted device_map got before planning existed.
    Widening that to "balanced" would change what every single-card caller loads."""
    ns = _load()
    assert ns["requested_device_map"](ns["DEFAULT_DEVICE_MAP"]) == "unsloth"


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


def test_a_user_quantization_config_replaces_the_flags_for_the_planner():
    """loader.py forwards a caller's `quantization_config` through `**kwargs` and clears
    `load_in_4bit` / `load_in_8bit`, because transformers refuses both at once. The cleared
    flags describe a full-precision load, so a 70B QLoRA gets sized at bf16 and
    `DeviceMapInfeasible` kills a load that would have fit.
    """
    ns = _load()
    config = types.SimpleNamespace(load_in_4bit = True, load_in_8bit = False)
    kwargs = ns["planner_quantization_kwargs"](
        load_in_4bit = False,
        load_in_8bit = False,
        quantization_config = config,
    )
    assert kwargs == {"quantization_config": config}
    # Both at once is exactly what transformers and the planner reject.
    assert "load_in_4bit" not in kwargs
    assert "load_in_8bit" not in kwargs


@pytest.mark.parametrize("four_bit, eight_bit", [(True, False), (False, True)])
def test_the_flags_are_used_when_no_config_was_given(four_bit, eight_bit):
    ns = _load()
    assert ns["planner_quantization_kwargs"](
        load_in_4bit = four_bit,
        load_in_8bit = eight_bit,
    ) == {
        "load_in_4bit": four_bit,
        "load_in_8bit": eight_bit,
        "llm_int8_skip_modules": _SKIP_MODULES,
    }


def test_a_16bit_load_is_planned_without_a_skip_list():
    """Nothing is being quantized, so there is nothing to keep out of it. A stray
    llm_int8_skip_modules would reach AutoConfig as an attribute of the model config."""
    ns = _load()
    assert ns["planner_quantization_kwargs"]() == {"load_in_4bit": False, "load_in_8bit": False}


def test_the_modules_unsloth_keeps_in_compute_dtype_are_sized_that_way():
    """On-the-fly quantization keeps SKIP_QUANTIZATION_MODULES out of bnb, and transformers
    reads llm_int8_skip_modules as `modules_to_not_convert`. Planning them at 4bit
    understates the head device by GiBs on a large-vocab VLM (`lm_head` plus a whole
    `vision_tower`), the number this plan exists to get right.
    """
    seen = {}
    ns = _load(planner = lambda name, **kw: seen.update(kw) or _Plan({"": 0}))
    ns["resolve_unsloth_device_map"](
        "unsloth",
        "m",
        **ns["planner_quantization_kwargs"](
            load_in_4bit = True,
            extra_skip_modules = ["out_proj"],
        ),
    )
    assert seen["load_in_4bit"] is True
    assert seen["llm_int8_skip_modules"] == _SKIP_MODULES + ["out_proj"]


def test_the_skip_list_is_not_sent_alongside_a_user_config():
    """The config already carries its own; sending both is what transformers refuses."""
    ns = _load()
    kwargs = ns["planner_quantization_kwargs"](
        quantization_config = types.SimpleNamespace(load_in_4bit = True),
        extra_skip_modules = ["out_proj"],
    )
    assert kwargs == {"quantization_config": kwargs["quantization_config"]}


def test_the_config_is_what_reaches_the_planner():
    seen = {}
    ns = _load(planner = lambda name, **kw: seen.update(kw) or _Plan({"": 0}))
    config = types.SimpleNamespace(load_in_4bit = True, load_in_8bit = False)
    ns["resolve_unsloth_device_map"](
        "unsloth",
        "m",
        **ns["planner_quantization_kwargs"](quantization_config = config),
    )
    assert seen["quantization_config"] is config
    assert "load_in_4bit" not in seen


def test_the_leaf_loaders_derive_the_planner_quantization_from_the_config():
    """A bare `load_in_4bit = load_in_4bit` at the call site is the bug above: the leaf
    receives the flag already cleared by loader.py."""
    models = os.path.join(HERE, "unsloth", "models")
    for name in ("llama.py", "vision.py", "diffusion.py"):
        source = open(os.path.join(models, name), encoding = "utf-8").read()
        for node in ast.walk(ast.parse(source)):
            if not (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "resolve_unsloth_device_map"
            ):
                continue
            keywords = {kw.arg for kw in node.keywords}
            assert "load_in_4bit" not in keywords, f"{name}:{node.lineno} plans on the cleared flag"
            assert "load_in_8bit" not in keywords, f"{name}:{node.lineno} plans on the cleared flag"
            unpacked = [
                kw.value
                for kw in node.keywords
                if kw.arg is None and isinstance(kw.value, ast.Call)
            ]
            assert any(
                getattr(call.func, "id", None) == "planner_quantization_kwargs" for call in unpacked
            ), f"{name}:{node.lineno}"


def _resolve_calls(source):
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "resolve_unsloth_device_map"
    ]


@pytest.mark.parametrize("name", ["llama.py", "vision.py", "diffusion.py"])
def test_the_planner_sizes_the_dtype_the_load_will_really_use(name):
    """`from_pretrained`'s dtype overrides the one config.json declares, and the planner
    only ever sees the config. So a float32 load of a bfloat16 checkpoint is sized at half
    its real weight bytes, the map is accepted, and materializing it OOMs; the reverse is
    the same error the other way, raising DeviceMapInfeasible on a load that would have fit.

    `add_dtype_kwargs` rather than a literal keyword: transformers renamed `torch_dtype` to
    `dtype`, and the planner hands these straight to AutoConfig, which only honours the
    name its own version knows.
    """
    source = open(os.path.join(HERE, "unsloth", "models", name), encoding = "utf-8").read()
    calls = _resolve_calls(source)
    assert calls, f"{name} never resolves a device map"
    for call in calls:
        unpacked = [
            kw.value for kw in call.keywords if kw.arg is None and isinstance(kw.value, ast.Call)
        ]
        assert any(
            getattr(unpack.func, "id", None) == "add_dtype_kwargs" for unpack in unpacked
        ), f"{name}:{call.lineno} plans against the checkpoint's dtype, not the load's"


def test_the_diffusion_plan_is_sized_against_the_config_the_load_applies():
    """diffusion.py keeps `lm_head`, `embed_tokens`, `experts`, `self_conditioning` and
    `router` out of bnb, most of an MoE checkpoint's parameters. Planning on the bare flags
    sizes all of them at 4 bits while the load materializes them in compute dtype, so the
    one config object is built before the plan and reused by the load.
    """
    path = os.path.join(HERE, "unsloth", "models", "diffusion.py")
    source = open(path, encoding = "utf-8").read()
    tree = ast.parse(source)

    built = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and getattr(node.value, "func", None) is not None
        and getattr(node.value.func, "id", None) == "BitsAndBytesConfig"
        and any(getattr(target, "id", None) == "qcfg" for target in node.targets)
    ]
    assert built, "diffusion.py no longer builds its own BitsAndBytesConfig"

    calls = _resolve_calls(source)
    assert calls, "diffusion.py never resolves a device map"
    for call in calls:
        assert (
            max(built) < call.lineno
        ), f"diffusion.py:{call.lineno} plans before the quantization config exists"
        forwarded = [
            unpack
            for unpack in (
                kw.value
                for kw in call.keywords
                if kw.arg is None and isinstance(kw.value, ast.Call)
            )
            if getattr(unpack.func, "id", None) == "planner_quantization_kwargs"
        ]
        assert forwarded, f"diffusion.py:{call.lineno} plans without the load's quantization"
        for unpack in forwarded:
            passed = {kw.arg: ast.unparse(kw.value) for kw in unpack.keywords}
            assert (
                passed.get("quantization_config") == "qcfg"
            ), f"diffusion.py:{call.lineno} plans without the skip list the load applies"


# --------------------------------------------------------------------------------------
# Planning by default reaches paths the opt-in never did.
# --------------------------------------------------------------------------------------


def _helpers():
    """`planner_kwargs_with_max_memory` / `planner_hub_kwargs`, without importing torch."""
    src = open(LOADER_UTILS, encoding = "utf-8").read()
    ns = {"os": os}
    for node in ast.parse(src).body:
        keep = (
            isinstance(node, ast.FunctionDef)
            and node.name
            in (
                "planner_kwargs_with_max_memory",
                "planner_hub_kwargs",
                "planner_config_overrides",
                "_get_effective_local_files_only",
                "_env_says_offline",
            )
        ) or (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "").startswith("_OFFLINE_ENV_")
        )
        if keep:
            exec(ast.get_source_segment(src, node), ns)
    return ns


def test_a_transformers_max_memory_reaches_the_planner():
    """Before the default flipped, `max_memory` bounded placement because transformers saw
    a string device_map. It only consults it then -- `_get_device_map` gates the whole
    `infer_auto_device_map` branch on `isinstance(device_map, str)` -- so once a plan
    returns a dict the budget is dropped and the map can exceed the caps or use a card the
    caller withheld."""
    ns = _helpers()
    merged = ns["planner_kwargs_with_max_memory"](None, {"max_memory": {0: "12GiB"}})
    assert merged["max_memory"] == {0: "12GiB"}


def test_an_explicit_planner_max_memory_wins_over_the_loader_one():
    ns = _helpers()
    merged = ns["planner_kwargs_with_max_memory"](
        {"max_memory": {0: "4GiB"}}, {"max_memory": {0: "12GiB"}}
    )
    assert merged["max_memory"] == {0: "4GiB"}


def test_no_max_memory_leaves_the_planner_kwargs_untouched():
    ns = _helpers()
    assert ns["planner_kwargs_with_max_memory"](None, {}) is None
    same = {"retained_rows": 8}
    assert ns["planner_kwargs_with_max_memory"](same, {"token": "x"}) is same


def test_the_planner_is_told_where_the_hub_is(monkeypatch):
    """It resolves the config a second time from `model_name`. Without these it can reach
    the network behind `local_files_only`, or miss a model that only exists in the caller's
    cache and lose a plan the load needed."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    ns = _helpers()
    assert ns["planner_hub_kwargs"]({"cache_dir": "/models", "local_files_only": True}) == {
        "cache_dir": "/models",
        "local_files_only": True,
    }
    assert ns["planner_hub_kwargs"]({}) == {}
    assert ns["planner_hub_kwargs"]({"local_files_only": False}) == {}


@pytest.mark.parametrize("name", ["vision.py", "llama.py", "diffusion.py"])
def test_every_leaf_planner_call_forwards_the_budget_and_the_hub(name):
    """A leaf that misses either one silently plans against the wrong facts."""
    source = open(os.path.join(MODELS, name), encoding = "utf-8").read()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "resolve_unsloth_device_map":
            continue
        rendered = ast.unparse(node)
        assert (
            "planner_kwargs_with_max_memory" in rendered
        ), f"{name}:{node.lineno} plans without the caller's max_memory"
        assert (
            "planner_hub_kwargs" in rendered
        ), f"{name}:{node.lineno} plans without the caller's cache_dir/local_files_only"
        assert (
            "planner_config_overrides" in rendered
        ), f"{name}:{node.lineno} plans without the caller's config overrides"
        return
    raise AssertionError(f"no resolve_unsloth_device_map call in {name}")


def test_the_wrapper_tells_the_leaf_the_config_was_the_callers():
    """FastModel pops `config` out of kwargs at loader.py:1248 and forwards it as
    `auto_config`, so by the time FastBaseModel looks, its own `kwargs.pop("config")` is
    None and a veto keyed on that alone never fires on the path almost everyone uses.
    The flag travels explicitly, the way `text_only_decoder` already does for the same
    reason: `auto_config` no longer describing the repo cannot be inferred downstream."""
    loader = open(os.path.join(MODELS, "loader.py"), encoding = "utf-8").read()
    assert "auto_config_from_caller = user_config is not None" in loader

    vision = open(os.path.join(MODELS, "vision.py"), encoding = "utf-8").read()
    args = [
        a.arg
        for node in ast.walk(ast.parse(vision))
        if isinstance(node, ast.FunctionDef) and node.name == "from_pretrained"
        for a in list(node.args.args) + list(node.args.kwonlyargs)
    ]
    assert "auto_config_from_caller" in args, "the leaf cannot see that the config was theirs"
    assert "or auto_config_from_caller" in vision


def test_a_resize_declines_the_automatic_offload():
    """`resize_token_embeddings` replaces the embedding module, and forward hooks do not
    travel to the replacement, so an offload installed during the load would leave a CPU
    embedding feeding a GPU decoder. An explicit request is left alone."""
    loader = open(os.path.join(MODELS, "loader.py"), encoding = "utf-8").read()
    assert "resize_model_vocab is not None" in loader
    assert "and offload_embedding == OFFLOAD_EMBEDDING_AUTO" in loader


def test_a_caller_supplied_config_declines_planning():
    """The weights load against their config; the planner rebuilds the repo's. Same class,
    different `num_hidden_layers` or `vocab_size`, and the map omits blocks or under-budgets
    weights -- which the class comparison cannot see."""
    source = open(os.path.join(MODELS, "vision.py"), encoding = "utf-8").read()
    assert "user_config is not None" in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        rendered = ast.unparse(node)
        if "user_config is not None" in rendered and "_planner_skip_reason" in rendered:
            return
    raise AssertionError("vision.py plans without vetoing a caller-supplied config")


def test_the_optimized_path_says_so_when_it_drops_an_offload_request():
    """`FastLanguageModel` accepts `offload_embedding`, but the optimized architectures
    take a path that has never had the parameter, so the request went nowhere in silence.
    The `"auto"` default stays quiet, since off is a decision it is entitled to make."""
    source = open(os.path.join(MODELS, "loader.py"), encoding = "utf-8").read()
    assert "does not support it" in source
    assert "offload_embedding != OFFLOAD_EMBEDDING_AUTO" in source


def test_the_auto_mode_is_recognised_by_value_everywhere():
    """`_resolve_offload_embedding` asks `== OFFLOAD_EMBEDDING_AUTO`, so a caller who
    hands in an equal but non-interned `"auto"` (one read out of a JSON config, say) is
    in automatic mode as far as the resolver is concerned. Any guard elsewhere that asks
    `is` disagrees with it: the resize guard would leave the offload on and the optimized
    path would print a notice for a request nobody made. Same question, same operator."""
    loader = open(os.path.join(MODELS, "loader.py"), encoding = "utf-8").read()
    for node in ast.walk(ast.parse(loader)):
        if not isinstance(node, ast.Compare):
            continue
        rendered = ast.unparse(node)
        if "OFFLOAD_EMBEDDING_AUTO" not in rendered:
            continue
        assert not any(
            isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops
        ), f"loader.py:{node.lineno} compares the auto mode by identity: {rendered}"


def test_the_optimized_path_declines_a_caller_supplied_config():
    """FastLanguageModel leaves `config` in kwargs, so the optimized Llama leaf pops its
    own `user_config` and loads the weights against it while the planner rebuilds the
    repo's from `model_name`. A caller who changed `num_hidden_layers` or `vocab_size`
    would get a map for a different model, so the plan is declined rather than guessed."""
    llama = open(os.path.join(MODELS, "llama.py"), encoding = "utf-8").read()
    tree = ast.parse(llama)
    body = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and ast.unparse(node) == "_planner_skip_reason = None":
            body = node
    assert body is not None, "llama.py no longer starts a planner skip reason"

    assert "if user_config is not None:" in llama
    assert (
        "a caller-supplied config may not describe the repo the planner rebuilds" in llama
    ), "the optimized path plans against a config the load does not use"

    # The veto has to come first, or a later branch that finds no other reason overwrites it.
    veto = llama.index("a caller-supplied config may not describe the repo the planner rebuilds")
    num_labels = llama.index("num_labels loads a task head the repo config does not describe")
    assert veto < num_labels, "the caller-config veto is set after another branch clears it"
    assert "if _planner_skip_reason is None and num_labels is not None:" in llama


def test_the_diffusion_leaf_plans_with_the_locality_the_load_uses():
    """diffusion.py pops `local_files_only` off kwargs and resolves the offline env vars
    into it before the load, so handing the planner the raw kwargs would tell it nothing.
    It gets the resolved value, or an offline load reaches the Hub behind the caller's
    back and, when that lookup fails, silently loses the split the model needs to fit."""
    source = open(os.path.join(MODELS, "diffusion.py"), encoding = "utf-8").read()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "resolve_unsloth_device_map":
            continue
        rendered = ast.unparse(node)
        assert (
            "local_files_only" in rendered and "cache_dir" in rendered
        ), f"diffusion.py:{node.lineno} plans without the locality the load resolved"
        return
    raise AssertionError("no resolve_unsloth_device_map call in diffusion.py")


def test_a_code_revision_reaches_the_planner():
    """`trust_remote_code` makes resolving the model class a second Hub lookup, and the
    planner honours `code_revision` for it. The load already gets it through kwargs, so
    leaving it out plans one revision of the remote code and loads another."""
    ns = _helpers()
    assert ns["planner_hub_kwargs"]({"code_revision": "abc123"}) == {"code_revision": "abc123"}
    assert "code_revision" not in ns["planner_hub_kwargs"]({})
    assert ns["planner_hub_kwargs"]({"code_revision": None}) == {}


def test_a_max_position_embeddings_override_reaches_the_planner():
    """The planner rebuilds the repo config from a name, so an override that lives only in
    the caller's kwargs never reaches it. Raising it on an architecture with learned
    position embeddings makes the planned tensors smaller than the materialized ones, and
    a map that fitted on paper OOMs."""
    ns = _helpers()
    assert ns["planner_config_overrides"]({"max_position_embeddings": 8192}) == {
        "max_position_embeddings": 8192,
    }
    assert ns["planner_config_overrides"]({}) == {}
    assert ns["planner_config_overrides"](None) == {}
    assert ns["planner_config_overrides"]({"max_position_embeddings": None}) == {}


def test_the_diffusion_leaf_plans_with_the_code_revision_too():
    """Same reason as its locality: this leaf builds the helper's input itself, so a key
    added to the helper does not reach it unless it is named here."""
    source = open(os.path.join(MODELS, "diffusion.py"), encoding = "utf-8").read()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "planner_hub_kwargs":
            assert "code_revision" in ast.unparse(
                node
            ), "the diffusion leaf plans against a different revision of the remote code"
            return
    raise AssertionError("no planner_hub_kwargs call in diffusion.py")


def test_an_unresolvable_explicit_model_class_declines_planning():
    """`resolve_model_class` reads `auto_model._model_mapping`, which a concrete
    `PreTrainedModel` subclass does not have, so it returns None and
    `planner_class_mismatch_reason` reads unknown as compatible. The planner would then
    build whatever the repo config selects while the load builds the caller's class."""
    vision = open(os.path.join(MODELS, "vision.py"), encoding = "utf-8").read()
    assert 'getattr(auto_model, "_model_mapping", None) is None' in vision
    assert "an explicit model class has no auto mapping" in vision

    # Ahead of the class comparison it backstops, or that one returns None and the
    # caller-config branch below claims the slot with the wrong reason.
    veto = vision.index("an explicit model class has no auto mapping")
    caller = vision.index("a caller-supplied config may not describe the repo the planner")
    assert veto < caller, "the unresolvable-class veto never gets to run"


def test_an_auto_class_still_plans():
    """The veto is keyed on the absence of `_model_mapping`, which every Auto class has,
    so a remote-code checkpoint whose config simply is not in the mapping keeps its plan.
    Declining on `model_class is None` alone would have turned planning off for those."""
    import ast as _ast

    vision = open(os.path.join(MODELS, "vision.py"), encoding = "utf-8").read()
    idx = vision.index("an explicit model class has no auto mapping")
    guard = vision[vision.rindex("if (", 0, idx) : idx]
    assert "_model_mapping" in guard, "the veto is not keyed on the class being concrete"
