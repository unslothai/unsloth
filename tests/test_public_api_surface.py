# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""Drift detectors for unsloth's OWN public surface (top symbols/classmethods the
unslothai/notebooks tree calls), so a rename or dropped kwarg fires DRIFT DETECTED here.

Call-site counts measured against unslothai/notebooks @ main:
  FastLanguageModel.from_pretrained   506
  FastLanguageModel.for_inference     370
  FastLanguageModel.get_peft_model    304
  FastVisionModel.for_inference       183
  FastVisionModel.from_pretrained     176
  FastVisionModel.get_peft_model       99
  FastVisionModel.for_training         60
  FastModel.from_pretrained           103
  FastModel.get_peft_model             67
"""

from __future__ import annotations

import inspect

import pytest


def _signature_param_names(callable_obj) -> set[str]:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return set()
    return set(sig.parameters)


def _accepts(callable_obj, kwargs: set[str]) -> tuple[bool, set[str]]:
    """(ok, missing): True if every kwarg is a named param or the signature has **kwargs."""
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True, set()
    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return True, set()
    missing = kwargs - set(params)
    return (not missing), missing


# FastLanguageModel: headline class.


def test_fast_language_model_class_present():
    unsloth = pytest.importorskip("unsloth")
    if not hasattr(unsloth, "FastLanguageModel"):
        pytest.fail(
            "DRIFT DETECTED: unsloth.FastLanguageModel is missing; every "
            "LoRA notebook fails at the first import cell."
        )


def test_fast_language_model_from_pretrained_kwargs():
    """from_pretrained must accept the canonical kwargs the notebooks pass."""
    unsloth = pytest.importorskip("unsloth")
    required = {"model_name", "max_seq_length", "dtype", "load_in_4bit"}
    ok, missing = _accepts(unsloth.FastLanguageModel.from_pretrained, required)
    if not ok:
        pytest.fail(
            f"DRIFT DETECTED: FastLanguageModel.from_pretrained dropped "
            f"kwargs {sorted(missing)}; 506 notebook call sites would "
            f"crash with TypeError."
        )


def test_fast_language_model_get_peft_model_kwargs():
    unsloth = pytest.importorskip("unsloth")
    required = {
        "r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "bias",
        "use_gradient_checkpointing",
        "random_state",
    }
    ok, missing = _accepts(unsloth.FastLanguageModel.get_peft_model, required)
    if not ok:
        pytest.fail(
            f"DRIFT DETECTED: FastLanguageModel.get_peft_model dropped "
            f"kwargs {sorted(missing)}; 304 notebook call sites would crash."
        )


def test_fast_language_model_for_inference_callable():
    unsloth = pytest.importorskip("unsloth")
    if not callable(getattr(unsloth.FastLanguageModel, "for_inference", None)):
        pytest.fail(
            "DRIFT DETECTED: FastLanguageModel.for_inference is missing; "
            "370 inference-cell call sites would crash."
        )


# FastVisionModel.


def test_fast_vision_model_class_and_methods():
    unsloth = pytest.importorskip("unsloth")
    if not hasattr(unsloth, "FastVisionModel"):
        pytest.fail(
            "DRIFT DETECTED: unsloth.FastVisionModel is missing; every "
            "vision fine-tuning notebook fails at import."
        )
    cls = unsloth.FastVisionModel
    missing = [
        m
        for m in ("from_pretrained", "get_peft_model", "for_inference", "for_training")
        if not callable(getattr(cls, m, None))
    ]
    if missing:
        pytest.fail(f"DRIFT DETECTED: FastVisionModel is missing methods {missing}.")


def test_fast_vision_model_get_peft_model_vision_kwargs():
    """Vision-specific kwargs the notebooks pass on the vision LoRA path."""
    unsloth = pytest.importorskip("unsloth")
    required = {
        "finetune_vision_layers",
        "finetune_language_layers",
        "finetune_attention_modules",
        "finetune_mlp_modules",
    }
    ok, missing = _accepts(unsloth.FastVisionModel.get_peft_model, required)
    if not ok:
        pytest.fail(
            f"DRIFT DETECTED: FastVisionModel.get_peft_model dropped "
            f"vision kwargs {sorted(missing)}."
        )


# FastModel: modern unified entry point.


def test_fast_model_class_and_methods():
    unsloth = pytest.importorskip("unsloth")
    if not hasattr(unsloth, "FastModel"):
        pytest.fail(
            "DRIFT DETECTED: unsloth.FastModel is missing; the modern "
            "unified entry point used by 100+ notebooks would crash."
        )
    missing = [
        m
        for m in ("from_pretrained", "get_peft_model")
        if not callable(getattr(unsloth.FastModel, m, None))
    ]
    if missing:
        pytest.fail(f"DRIFT DETECTED: FastModel is missing methods {missing}.")


def test_fast_model_from_pretrained_kwargs():
    unsloth = pytest.importorskip("unsloth")
    required = {"model_name", "max_seq_length", "dtype", "load_in_4bit"}
    ok, missing = _accepts(unsloth.FastModel.from_pretrained, required)
    if not ok:
        pytest.fail(
            f"DRIFT DETECTED: FastModel.from_pretrained dropped kwargs "
            f"{sorted(missing)}; 103 notebook call sites would crash."
        )


# Bf16 helper alias (renamed once already; keep both accepted).


def test_is_bf16_supported_or_alias_callable():
    """is_bf16_supported or the legacy is_bfloat16_supported alias must remain importable."""
    unsloth = pytest.importorskip("unsloth")
    has_new = callable(getattr(unsloth, "is_bf16_supported", None))
    has_old = callable(getattr(unsloth, "is_bfloat16_supported", None))
    if not (has_new or has_old):
        pytest.fail(
            "DRIFT DETECTED: neither unsloth.is_bf16_supported nor "
            "unsloth.is_bfloat16_supported is callable; dtype probing "
            "in 50+ notebooks fails."
        )
