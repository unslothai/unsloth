# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pinned-symbol compat check across PEFT minor versions unsloth + unsloth-zoo
target. For each tracked tag, fetch source from github.com/huggingface/peft and
assert every PEFT symbol unsloth touches is present, catching API drift.
Versioning covers unsloth/pyproject.toml's `peft>=0.18.0,!=0.11.0` window + main.
"""

from __future__ import annotations

import re

import pytest

from tests.version_compat._fetch import fetch_text, first_match, has_def, is_bound


# pyproject pin: peft>=0.18.0. Test the floor + each minor since.
PEFT_TAGS = [
    "v0.18.0",
    "v0.18.1",
    "v0.19.0",
    "v0.19.1",
    "v0.20.0",
    "main",
]

pytestmark = pytest.mark.parametrize("tag", PEFT_TAGS)


def test_peft_top_level_exports(tag: str):
    src = fetch_text("huggingface/peft", tag, "src/peft/__init__.py")
    assert src is not None, f"{tag}: src/peft/__init__.py missing"
    needed = (
        "LoraConfig",
        "get_peft_model",
        "PeftModel",
    )
    missing = [n for n in needed if not is_bound(src, n)]
    assert not missing, (
        f"{tag}: peft top-level missing {missing}; "
        f"unsloth.models.sentence_transformer:1948 + unsloth-zoo saving_utils "
        f"will ImportError"
    )


def test_peft_lora_config_class(tag: str):
    candidates = [
        "src/peft/tuners/lora/config.py",
        "src/peft/tuners/lora/__init__.py",
        "src/peft/tuners/lora.py",
    ]
    found_in = []
    for p in candidates:
        src = fetch_text("huggingface/peft", tag, p)
        if src is not None and has_def(src, "LoraConfig", "class"):
            found_in.append(p)
    assert found_in, f"{tag}: peft.tuners.lora.LoraConfig not in any of {candidates}"


def test_get_peft_model_function(tag: str):
    """get_peft_model may live in mapping.py or mapping_func.py (0.18+ split)."""
    candidates = [
        "src/peft/mapping.py",
        "src/peft/mapping_func.py",
        "src/peft/__init__.py",
        "src/peft/peft_model.py",
    ]
    for p in candidates:
        src = fetch_text("huggingface/peft", tag, p)
        if src is not None and has_def(src, "get_peft_model", "func"):
            return
    pytest.fail(f"{tag}: def get_peft_model(...) not found in any of {candidates}")


# unsloth-zoo walks LoraLayer subclasses; a rename makes the walk silently return 0.
def test_peft_lora_layer_class(tag: str):
    candidates = [
        "src/peft/tuners/lora/layer.py",
        "src/peft/tuners/lora/__init__.py",
        "src/peft/tuners/lora.py",
    ]
    for p in candidates:
        src = fetch_text("huggingface/peft", tag, p)
        if src is not None and has_def(src, "LoraLayer", "class"):
            return
    pytest.fail(
        f"{tag}: class LoraLayer not in any of {candidates} — "
        f"unsloth-zoo MoE LoRA extractor relies on isinstance checks "
        f"against this class"
    )


def test_peft_lora_bnb_integration(tag: str):
    candidates = [
        "src/peft/tuners/lora/bnb.py",
        "src/peft/tuners/lora/_bnb.py",
    ]
    for p in candidates:
        src = fetch_text("huggingface/peft", tag, p)
        if src is None:
            continue
        has_4bit = any(
            cls in src
            for cls in (
                "class Linear4bit",
                "class Linear8bitLt",
                "class _Linear4bit",
                "class _Linear8bitLt",
            )
        )
        if has_4bit:
            return
    pytest.fail(
        f"{tag}: peft.tuners.lora.bnb missing or no Linear4bit/Linear8bitLt "
        f"class found; unsloth's 4-bit LoRA path silently degrades to fp16"
    )


def test_peft_variant_kwarg_keys_const(tag: str):
    src = fetch_text("huggingface/peft", tag, "src/peft/tuners/lora/layer.py")
    if src is None:
        pytest.skip(f"{tag}: src/peft/tuners/lora/layer.py missing")
    if not is_bound(src, "VARIANT_KWARG_KEYS"):
        pytest.fail(
            f"{tag}: peft.tuners.lora.layer.VARIANT_KWARG_KEYS missing; "
            f"unsloth_zoo/compiler.py:2645 import injection breaks (unsloth-zoo#430)"
        )


def test_peft_param_wrapper_class(tag: str):
    src = fetch_text("huggingface/peft", tag, "src/peft/tuners/lora/layer.py")
    if src is None:
        pytest.skip(f"{tag}: layer.py missing")
    assert has_def(src, "ParamWrapper", "class"), (
        f"{tag}: peft.tuners.lora.layer.ParamWrapper missing; "
        f"unsloth_zoo/temporary_patches/qwen3_moe.py:43-130 + "
        f"moe_utils.py:757 ImportError (unsloth-zoo#618)"
    )
    for name in ("parameter_name", "forward", "lora_A", "get_base_layer"):
        _present = name in src


def test_peft_lora_config_target_parameters(tag: str):
    src = fetch_text("huggingface/peft", tag, "src/peft/tuners/lora/config.py")
    if src is None:
        pytest.skip(f"{tag}: src/peft/tuners/lora/config.py missing")
    has_it = "target_parameters" in src
    if "0.18" in tag and not has_it:
        pytest.skip(f"{tag}: target_parameters not yet introduced (peft 0.18)")
    assert has_it, (
        f"{tag}: LoraConfig.target_parameters missing on peft >=0.19; "
        f"unsloth-zoo MoE target-parameter extraction breaks"
    )


def test_peft_lora_model_create_and_replace(tag: str):
    src = fetch_text("huggingface/peft", tag, "src/peft/tuners/lora/model.py")
    if src is None:
        pytest.skip(f"{tag}: src/peft/tuners/lora/model.py missing")
    assert has_def(src, "LoraModel", "class"), f"{tag}: class LoraModel missing"
    assert has_def(src, "_create_and_replace", "func"), (
        f"{tag}: LoraModel._create_and_replace missing; "
        f"unsloth/models/loader.py:1535-1601 monkey-patch breaks (unsloth#4807)"
    )


def test_peft_transformers_weight_conversion_module(tag: str):
    candidates = [
        "src/peft/utils/transformers_weight_conversion.py",
        "src/peft/utils/transformers_weight_conversion/__init__.py",
    ]
    hit = first_match("huggingface/peft", tag, candidates)
    if hit is None:
        pytest.skip(f"{tag}: transformers_weight_conversion not present (legacy peft)")
    _, src = hit
    assert is_bound(src, "build_peft_weight_mapping"), (
        f"{tag}: build_peft_weight_mapping missing in transformers_weight_conversion; "
        f"unsloth/import_fixes.py:1375-1456 wrap breaks (unsloth#5167)"
    )


def test_peft_integrations_dequantize_module_weight(tag: str):
    candidates = [
        "src/peft/utils/integrations.py",
        "src/peft/utils/integrations/__init__.py",
    ]
    hit = first_match("huggingface/peft", tag, candidates)
    assert hit is not None, f"{tag}: src/peft/utils/integrations[.py|/__init__.py] both missing"
    _, src = hit
    assert is_bound(src, "dequantize_module_weight"), (
        f"{tag}: peft.utils.integrations.dequantize_module_weight missing; "
        f"unsloth-zoo vllm_utils.py:2701, unsloth/_utils.py:1550, "
        f"saving_utils.py:270 ImportError"
    )


def test_peft_type_lora_enum(tag: str):
    candidates = [
        "src/peft/utils/peft_types.py",
        "src/peft/utils/__init__.py",
        "src/peft/__init__.py",
    ]
    for p in candidates:
        src = fetch_text("huggingface/peft", tag, p)
        if src is None:
            continue
        if "PeftType" in src and ("LORA" in src or "lora" in src.lower()):
            return
    pytest.fail(
        f"{tag}: peft.PeftType (with LORA member) not in any of {candidates}; "
        f"unsloth-zoo vllm_utils.py:2520 reference breaks"
    )


def test_peft_modules_to_save_wrapper(tag: str):
    candidates = [
        "src/peft/utils/other.py",
        "src/peft/utils/__init__.py",
    ]
    found_in = []
    for p in candidates:
        src = fetch_text("huggingface/peft", tag, p)
        if src is None:
            continue
        if has_def(src, "ModulesToSaveWrapper", "class"):
            found_in.append(p)
    assert found_in, (
        f"{tag}: ModulesToSaveWrapper not defined in {candidates}; "
        f"unsloth/training_utils.py:239 + models/llama.py:153 ImportError"
    )


def test_peft_peft_model_from_pretrained_signature(tag: str):
    src = fetch_text("huggingface/peft", tag, "src/peft/peft_model.py")
    assert src is not None, f"{tag}: src/peft/peft_model.py missing"
    assert has_def(
        src, "from_pretrained", "func"
    ), f"{tag}: PeftModel.from_pretrained missing in peft_model.py"


def test_peft_version_parseable(tag: str):
    src = fetch_text("huggingface/peft", tag, "src/peft/__init__.py")
    assert src is not None
    has_literal = bool(re.search(r'^__version__\s*=\s*["\']', src, re.MULTILINE))
    has_subimport = bool(re.search(r"^from\s+\.version\s+import\s+__version__", src, re.MULTILINE))
    has_metadata = bool(
        re.search(
            r"^from\s+importlib\.metadata\s+import\s+(?:[\w,\s]+,\s*)?version",
            src,
            re.MULTILINE,
        )
        and re.search(r"^\s*__version__\s*=\s*version\s*\(", src, re.MULTILINE)
    )
    assert (
        has_literal or has_subimport or has_metadata
    ), f"{tag}: peft.__version__ not exported via any known mechanism"


# 11. init_lora_weights="mica": check_mica_init imports MiCALinearVariant; PEFT inits via LoraLayer.mica_init
#     and freezes lora_B via _freeze_non_trainable_peft_weights (freeze_peft_variant_weights). PEFT >= 0.20.0.
def test_peft_mica_variant_and_init(tag: str):
    variants_src = fetch_text("huggingface/peft", tag, "src/peft/tuners/lora/variants.py")
    if variants_src is None or not has_def(variants_src, "MiCALinearVariant", "class"):
        pytest.skip(f"{tag}: MiCA not yet introduced (peft 0.20)")
    layer_src = fetch_text("huggingface/peft", tag, "src/peft/tuners/lora/layer.py")
    tuners_src = fetch_text("huggingface/peft", tag, "src/peft/tuners/tuners_utils.py")
    assert layer_src is not None and has_def(
        layer_src, "mica_init", "func"
    ), f"{tag}: mica_init missing"
    assert tuners_src is not None and has_def(
        tuners_src, "_freeze_non_trainable_peft_weights", "func"
    ), f"{tag}: _freeze_non_trainable_peft_weights missing"


# 12. unsloth/models/lora_init.py replaces LoraLayer.pissa_init(self, adapter_name, init_lora_weights) and
#     mica_init(self, adapter_name) during get_peft_model and calls layer.py's `transpose`.
def test_peft_lora_init_hooks(tag: str):
    src = fetch_text("huggingface/peft", tag, "src/peft/tuners/lora/layer.py")
    if src is None:
        pytest.skip(f"{tag}: layer.py missing")
    assert re.search(
        r"def pissa_init\(self, adapter_name, init_lora_weights\)", src
    ), f"{tag}: pissa_init signature moved"
    assert re.search(
        r"^from peft\.utils\.other import transpose$", src, re.M
    ), f"{tag}: transpose import moved"
    if has_def(src, "mica_init", "func"):
        assert re.search(
            r"def mica_init\(self, adapter_name\)", src
        ), f"{tag}: mica_init signature moved"
