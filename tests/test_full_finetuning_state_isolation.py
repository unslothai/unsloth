# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""full_finetuning is per model, but UNSLOTH_ENABLE_FULL_FINETUNING is shared by the whole process."""

import ast
import os
import textwrap
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

UTILS_SRC = (REPO_ROOT / "unsloth" / "models" / "_utils.py").read_text(encoding = "utf-8")
VISION_SRC = (REPO_ROOT / "unsloth" / "models" / "vision.py").read_text(encoding = "utf-8")
LLAMA_SRC = (REPO_ROOT / "unsloth" / "models" / "llama.py").read_text(encoding = "utf-8")


def _method(src, class_name, name):
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == name
                ):
                    return child
    raise AssertionError(f"{class_name}.{name} not found")


def _top_level_function(src, name):
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _is_full_finetuning_impl():
    """The real helper, exec'd out of _utils.py so the package import chain is not needed."""
    ns = {"os": os}
    exec(
        ast.get_source_segment(UTILS_SRC, _top_level_function(UTILS_SRC, "_is_full_finetuning")), ns
    )
    return ns["_is_full_finetuning"]


def _finetuning_guard(src, class_name):
    """The `if` in get_peft_model that decides PEFT is a no-op on a full finetuning model."""
    method = _method(src, class_name, "get_peft_model")
    for node in ast.walk(method):
        if not isinstance(node, ast.If):
            continue
        segment = ast.get_source_segment(src, node)
        if "UNSLOTH_ENABLE_FULL_FINETUNING" in segment or "_is_full_finetuning" in segment:
            return segment
    raise AssertionError(f"no full finetuning guard in {class_name}.get_peft_model")


def _guard_returns_early(guard_src, model, env, monkeypatch):
    """Run the extracted guard against one model and one environment."""
    monkeypatch.setenv("UNSLOTH_ENABLE_FULL_FINETUNING", env)
    body = (
        "def _guard(model):\n"
        + textwrap.indent(textwrap.dedent(guard_src), "    ")
        + "\n    return False"
    )
    ns = {
        "os": os,
        "_is_full_finetuning": _is_full_finetuning_impl(),
        "print": lambda *a, **k: None,
        "_unsloth_install_pretrain_detector": lambda *a, **k: None,
        "install_frozen_prefix_skip": lambda *a, **k: None,
        "skip_checkpointing": lambda *a, **k: None,
        "checkpoint_skip_layers": 0,
    }
    exec(body, ns)
    return bool(ns["_guard"](model))


def _post_patch_finetuning_answer(model, env, monkeypatch):
    """Run post_patch_model's `full_finetuning = ...` line against one model and one environment."""
    monkeypatch.setenv("UNSLOTH_ENABLE_FULL_FINETUNING", env)
    method = _method(VISION_SRC, "FastBaseModel", "post_patch_model")
    for node in ast.walk(method):
        if not isinstance(node, ast.Assign):
            continue
        segment = ast.get_source_segment(VISION_SRC, node)
        if "full_finetuning" not in segment:
            continue
        if "UNSLOTH_ENABLE_FULL_FINETUNING" in segment or "_is_full_finetuning" in segment:
            body = (
                "def _answer(model):\n"
                + textwrap.indent(textwrap.dedent(segment), "    ")
                + "\n    return full_finetuning"
            )
            ns = {"os": os, "_is_full_finetuning": _is_full_finetuning_impl()}
            exec(body, ns)
            return ns["_answer"](model)
    raise AssertionError("post_patch_model does not derive full_finetuning")


def _loaded_model(full_finetuning):
    model = types.SimpleNamespace()
    model._unsloth_full_finetuning = full_finetuning
    return model


@pytest.mark.parametrize(
    "class_name,src",
    [
        ("FastBaseModel", VISION_SRC),
        ("FastLlamaModel", LLAMA_SRC),
    ],
)
def test_peft_guard_follows_the_model_not_the_last_load(class_name, src, monkeypatch):
    """Load full_finetuning=True, then a LoRA load flips the env to "0": the first model's
    get_peft_model must still no-op, or it injects adapters into a model that trains in full."""
    guard = _finetuning_guard(src, class_name)
    full_model = _loaded_model(True)
    lora_model = _loaded_model(False)

    # The later LoRA load owns the environment now.
    assert _guard_returns_early(guard, full_model, "0", monkeypatch) is True
    assert _guard_returns_early(guard, lora_model, "0", monkeypatch) is False


@pytest.mark.parametrize(
    "class_name,src",
    [
        ("FastBaseModel", VISION_SRC),
        ("FastLlamaModel", LLAMA_SRC),
    ],
)
def test_peft_guard_survives_a_later_full_finetuning_load(class_name, src, monkeypatch):
    """The other order: a LoRA model loaded first must still get adapters after a later
    full_finetuning=True load sets the env to "1"."""
    guard = _finetuning_guard(src, class_name)
    lora_model = _loaded_model(False)
    full_model = _loaded_model(True)

    assert _guard_returns_early(guard, lora_model, "1", monkeypatch) is False
    assert _guard_returns_early(guard, full_model, "1", monkeypatch) is True


@pytest.mark.parametrize("env", ["0", "1"])
@pytest.mark.parametrize(
    "class_name,src",
    [
        ("FastBaseModel", VISION_SRC),
        ("FastLlamaModel", LLAMA_SRC),
    ],
)
def test_peft_guard_preserves_the_env_for_an_unstamped_model(class_name, src, env, monkeypatch):
    """A model this process never loaded keeps the explicit environment answer."""
    guard = _finetuning_guard(src, class_name)
    model = types.SimpleNamespace()
    assert _guard_returns_early(guard, model, env, monkeypatch) is (env == "1")


def test_post_patch_model_uses_the_models_own_answer(monkeypatch):
    """post_patch_model picks float32_mixed_precision from full_finetuning; a stale env
    would spend float32 memory on a bf16 LoRA model (or skip it for a full finetuning one)."""
    lora_model = _loaded_model(False)
    full_model = _loaded_model(True)
    assert _post_patch_finetuning_answer(lora_model, "1", monkeypatch) is False
    assert _post_patch_finetuning_answer(full_model, "0", monkeypatch) is True


def test_every_loader_writes_the_flag_for_its_own_load():
    """Each from_pretrained declares this load's mode so mid-load readers (packed expert /
    MXFP4 hooks, expert quantization) do not inherit the previous load's answer."""
    for node in ast.walk(_method(LLAMA_SRC, "FastLlamaModel", "from_pretrained")):
        if isinstance(
            node, ast.Assign
        ) and "UNSLOTH_ENABLE_FULL_FINETUNING" in ast.get_source_segment(LLAMA_SRC, node):
            assert '"0"' in ast.get_source_segment(
                LLAMA_SRC, node
            ), "the legacy path only loads adapters, so it must declare 0"
            break
    else:
        raise AssertionError("FastLlamaModel.from_pretrained never declares the flag")


def test_the_stamp_helper_is_exported_for_star_imports():
    """vision.py and llama.py reach _utils names through `from ._utils import *`."""
    tree = ast.parse(UTILS_SRC)
    exported = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            exported = ast.literal_eval(node.value)
    assert "_is_full_finetuning" in exported
