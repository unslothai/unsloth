# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""One-time warning for low-rank LoRA with lora_alpha == r, and its call sites in get_peft_model."""

import ast
import importlib.util
import pathlib
import warnings

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "unsloth" / "models" / "_lora_scale_warning.py"


def _fresh_module():
    spec = importlib.util.spec_from_file_location("_test_lora_scale_warning", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _warnings_for(
    r,
    lora_alpha,
    use_rslora = False,
):
    module = _fresh_module()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        returned = module.warn_low_rank_lora_alpha(r, lora_alpha, use_rslora)
    messages = [str(w.message) for w in caught if "lora_alpha" in str(w.message)]
    assert returned == bool(messages)
    return messages


@pytest.mark.parametrize("r, lora_alpha", [(1, 1), (8, 8), (4, 4.0)])
def test_warns_for_low_rank_alpha_equal_r(r, lora_alpha):
    messages = _warnings_for(r, lora_alpha)
    assert len(messages) == 1
    assert f"r = {r}" in messages[0]
    assert "use_rslora = True" in messages[0]
    assert "larger learning rate" in messages[0]


@pytest.mark.parametrize(
    "r, lora_alpha, use_rslora",
    [
        (16, 16, False),  # the default: rank above the threshold
        (64, 64, False),
        (1, 32, False),  # fixed alpha: scale 32
        (8, 32, False),
        (1, 1, True),  # rsLoRA scales by alpha / sqrt(r)
        (8, 8, True),
        (8, None, False),
        (True, True, False),
    ],
)
def test_silent_otherwise(r, lora_alpha, use_rslora):
    assert _warnings_for(r, lora_alpha, use_rslora) == []


def test_warns_only_once_per_process():
    module = _fresh_module()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert module.warn_low_rank_lora_alpha(1, 1) is True
        assert module.warn_low_rank_lora_alpha(8, 8) is False
        assert module.warn_low_rank_lora_alpha(2, 2) is False
    assert len([w for w in caught if "lora_alpha" in str(w.message)]) == 1


def _get_peft_model_calls(relative_path, class_name):
    tree = ast.parse((ROOT / relative_path).read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "get_peft_model":
                    defaults = {
                        arg.arg: ast.literal_eval(default)
                        for arg, default in zip(
                            item.args.args[-len(item.args.defaults) :], item.args.defaults
                        )
                        if isinstance(default, ast.Constant)
                    }
                    calls = [
                        call
                        for call in ast.walk(item)
                        if isinstance(call, ast.Call)
                        and getattr(call.func, "id", None) == "warn_low_rank_lora_alpha"
                    ]
                    return calls, defaults
    raise AssertionError(f"{class_name}.get_peft_model not found in {relative_path}")


@pytest.mark.parametrize(
    "relative_path, class_name",
    [
        ("unsloth/models/llama.py", "FastLlamaModel"),
        ("unsloth/models/vision.py", "FastBaseModel"),
    ],
)
def test_get_peft_model_calls_warning_without_changing_defaults(relative_path, class_name):
    calls, defaults = _get_peft_model_calls(relative_path, class_name)
    assert len(calls) == 1
    assert [ast.unparse(a) for a in calls[0].args] == ["r", "lora_alpha", "use_rslora"]
    # Defaults stay r = 16, lora_alpha = 16, use_rslora = False: the default config never warns.
    assert (defaults["r"], defaults["lora_alpha"], defaults["use_rslora"]) == (16, 16, False)
