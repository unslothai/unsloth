# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""text_only VLM loads plan from the text config; an old planner without `config` declines."""

import ast
import inspect
import os
import sys
import types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "unsloth", "models")
_SRC = open(os.path.join(MODELS, "loader_utils.py"), encoding = "utf-8").read()
_REASON = "text_only loads a decoder the repo config does not describe"


class _FakeCuda:
    def device_count(self):
        return 2

    def mem_get_info(self, index):
        return (8 * 2**30, 16 * 2**30)


class _Plan:
    device_map = {"model.layers.0": 0, "lm_head": 1}

    def describe(self):
        return "  (fabricated plan)"


def _load(planner):
    ns = {
        "os": os,
        "inspect": inspect,
        "torch": types.SimpleNamespace(cuda = _FakeCuda()),
        "DEVICE_TYPE_TORCH": "cuda",
        "is_distributed": lambda: False,
    }
    for node in ast.parse(_SRC).body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "resolve_unsloth_device_map",
            "_as_bytes",
        ):
            exec(ast.get_source_segment(_SRC, node), ns)
        elif isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in (
            "UNSLOTH_DEVICE_MAP",
            "UNSLOTH_BALANCED_DEVICE_MAP",
            "_PLANNED_DEVICE_MAPS",
            "_SIZE_UNITS",
        ):
            exec(ast.get_source_segment(_SRC, node), ns)
    module = types.ModuleType("unsloth_zoo.device_map_planner")
    module.plan_device_map_for_pretrained = planner
    sys.modules["unsloth_zoo.device_map_planner"] = module
    return ns


def test_a_planner_that_takes_a_config_plans_the_text_decoder():
    seen = {}
    text_config = object()

    def planner(
        model_name,
        *,
        max_memory = None,
        config = None,
        **kwargs,
    ):
        seen.update(config = config, kwargs = kwargs)
        return _Plan()

    ns = _load(planner)
    device_map = ns["resolve_unsloth_device_map"](
        "unsloth",
        "org/vlm",
        planner_config = text_config,
        planner_config_reason = _REASON,
        dtype = "bfloat16",
    )
    assert device_map == _Plan.device_map
    assert seen["config"] is text_config
    assert seen["kwargs"] == {"dtype": "bfloat16"}


def test_an_older_planner_still_declines_with_the_old_reason(capsys):
    """A planner without a `config` parameter would plan the whole VLM, so it is not called."""

    def old_planner(
        model_name,
        *,
        max_memory = None,
        **config_kwargs,
    ):
        raise AssertionError("an older planner must not be handed the text config")

    ns = _load(old_planner)
    device_map = ns["resolve_unsloth_device_map"](
        "unsloth", "org/vlm", planner_config = object(), planner_config_reason = _REASON
    )
    assert device_map == "sequential"
    assert _REASON in capsys.readouterr().out


def test_no_planner_config_leaves_the_planner_call_unchanged():
    seen = {}

    def old_planner(
        model_name,
        *,
        max_memory = None,
        **config_kwargs,
    ):
        seen.update(config_kwargs)
        return _Plan()

    ns = _load(old_planner)
    assert ns["resolve_unsloth_device_map"]("unsloth", "org/model", dtype = "bfloat16") == (
        _Plan.device_map
    )
    assert seen == {"dtype": "bfloat16"}


def test_a_caller_veto_still_wins_over_the_text_config():
    def planner(
        model_name,
        *,
        config = None,
        **kwargs,
    ):
        raise AssertionError("vetoed loads must not plan")

    ns = _load(planner)
    assert (
        ns["resolve_unsloth_device_map"](
            "unsloth", "org/vlm", skip_reason = "caller config", planner_config = object()
        )
        == "sequential"
    )


def test_vision_loader_hands_the_text_config_to_the_planner():
    source = open(os.path.join(MODELS, "vision.py"), encoding = "utf-8").read()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "resolve_unsloth_device_map"
    ]
    assert calls, "vision.py no longer plans a device map"
    for call in calls:
        passed = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        assert passed.get("planner_config") == "_planner_config"
    assert "_planner_config = auto_config if text_only_decoder else None" in source
    assert "if text_only_decoder\n            else None" not in source


def test_a_config_in_the_planner_kwargs_does_not_collide_with_the_text_config():
    seen = {}
    text_config = object()

    def planner(
        model_name,
        *,
        max_memory = None,
        config = None,
        **kwargs,
    ):
        seen.update(config = config, kwargs = kwargs)
        return _Plan()

    ns = _load(planner)
    device_map = ns["resolve_unsloth_device_map"](
        "unsloth",
        "org/vlm",
        planner_kwargs = {"config": object()},
        planner_config = text_config,
        planner_config_reason = _REASON,
    )
    assert device_map == _Plan.device_map
    assert seen["config"] is text_config
