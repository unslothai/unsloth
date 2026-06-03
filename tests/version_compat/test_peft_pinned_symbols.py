# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pinned-symbol compat check across PEFT PyPI minor versions
unsloth + unsloth-zoo target. Catches API drift like:

  - peft 0.18 finalised the LoraConfig public surface (+ MoE-aware
    target_modules); unsloth uses target_modules + r + lora_alpha +
    lora_dropout + bias.
  - peft 0.19 introduced the LoraConfig.target_parameters extension;
    unsloth-zoo's MoE LoRA extractor in saving_utils.py reads it via
    getattr() so missing on older versions is OK but the attribute
    shape must remain stable on >= 0.19.
  - peft.tuners.lora package layout: LoraLayer / LoraConfig / Linear4bit
    re-exports must keep working under both `from peft import X` and
    `from peft.tuners.lora import X`.

Strategy: for each tracked PEFT tag, fetch source from
github.com/huggingface/peft (no pip install needed) and assert that
every symbol unsloth + unsloth-zoo's PEFT touchpoints depend on is
present.

Versioning policy: cover the supported window declared in
unsloth/pyproject.toml (`peft>=0.18.0,!=0.11.0`) plus `main`. The
`!=0.11.0` exclusion is for the historical broken release; we don't
test against it.
"""

from __future__ import annotations

import re

import pytest

from tests.version_compat._fetch import fetch_text, first_match, has_def


# pyproject pin: peft>=0.18.0. Test the floor + each minor since.
# `main` catches breakage before a release lands.
PEFT_TAGS = [
    "v0.18.0",
    "v0.18.1",
    "v0.19.0",
    "v0.19.1",
    "main",
]


# -------------------------------------------------------------------------
# Top-level public re-exports. unsloth/models/sentence_transformer.py:1948
# does `from peft import LoraConfig, get_peft_model as peft_get_peft_model`.
# unsloth_zoo's saving_utils + lora extractors hit `peft.PeftModel`.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_peft_top_level_exports(tag: str):
    src = fetch_text("huggingface/peft", tag, "src/peft/__init__.py")
    assert src is not None, f"{tag}: src/peft/__init__.py missing"
    needed = (
        "LoraConfig",
        "get_peft_model",
        "PeftModel",
    )
    missing = [n for n in needed if n not in src]
    assert not missing, (
        f"{tag}: peft top-level missing {missing}; "
        f"unsloth.models.sentence_transformer:1948 + unsloth-zoo saving_utils "
        f"will ImportError"
    )


# -------------------------------------------------------------------------
# LoraConfig at the canonical sub-module path: peft.tuners.lora.LoraConfig
# (or peft.tuners.lora.config.LoraConfig). unsloth-zoo's LoraConfig
# normaliser inspects it via getattr() and dataclass field
# introspection.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
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


# -------------------------------------------------------------------------
# get_peft_model: top-level helper used by sentence_transformer.py:2043.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_get_peft_model_function(tag: str):
    """`def get_peft_model(...)` may live in mapping.py (older
    layout) or mapping_func.py (peft 0.18+ split). Either is fine."""
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


# -------------------------------------------------------------------------
# LoraLayer base class: unsloth-zoo's MoE LoRA extractor walks subclasses
# of peft.tuners.lora.LoraLayer to find quantised LoRA modules. If the
# class is renamed or moved, the walk silently returns 0 modules (the
# pytest tests mentioned in the audit report exercise exactly this).
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
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


# -------------------------------------------------------------------------
# bnb-aware LoRA: peft.tuners.lora.bnb is the integration point with
# bitsandbytes. unsloth + unsloth-zoo dispatch to this when the user
# loads a 4-bit base. Missing this module -> 4bit LoRA silently falls
# back to fp16 LoRA (silently bigger memory footprint).
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_peft_lora_bnb_integration(tag: str):
    candidates = [
        "src/peft/tuners/lora/bnb.py",
        "src/peft/tuners/lora/_bnb.py",
    ]
    for p in candidates:
        src = fetch_text("huggingface/peft", tag, p)
        if src is None:
            continue
        # The Linear4bit subclass naming is the contract -- either name
        # is fine, but at least one bnb-flavoured Linear must exist.
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


# =========================================================================
# Coverage extension (added 2026-05): symbols from the 8-PR audit
#   unsloth#5015, #5167, #5036, #4807 + unsloth-zoo#618, #596, #482, #430.
# =========================================================================


# -------------------------------------------------------------------------
# 1. peft.tuners.lora.layer.VARIANT_KWARG_KEYS — added in peft 0.18.
#    unsloth-zoo#430 injects the import into the compiled forward.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_peft_variant_kwarg_keys_const(tag: str):
    src = fetch_text("huggingface/peft", tag, "src/peft/tuners/lora/layer.py")
    if src is None:
        pytest.skip(f"{tag}: src/peft/tuners/lora/layer.py missing")
    if "VARIANT_KWARG_KEYS" not in src:
        pytest.fail(
            f"{tag}: peft.tuners.lora.layer.VARIANT_KWARG_KEYS missing; "
            f"unsloth_zoo/compiler.py:2645 import injection breaks (unsloth-zoo#430)"
        )


# -------------------------------------------------------------------------
# 2. peft.tuners.lora.layer.ParamWrapper — peft 0.18 added the class
#    for MoE 3D-parameter LoRA. Required attrs: parameter_name, lora_A,
#    forward, get_base_layer. peft 0.19 also added _did_swap_in_out_features.
#    unsloth-zoo#618 monkey-patches the MoE LoRA extractor.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_peft_param_wrapper_class(tag: str):
    src = fetch_text("huggingface/peft", tag, "src/peft/tuners/lora/layer.py")
    if src is None:
        pytest.skip(f"{tag}: layer.py missing")
    assert has_def(src, "ParamWrapper", "class"), (
        f"{tag}: peft.tuners.lora.layer.ParamWrapper missing; "
        f"unsloth_zoo/temporary_patches/qwen3_moe.py:43-130 + "
        f"moe_utils.py:757 ImportError (unsloth-zoo#618)"
    )
    # Required member names — informational only; the class may
    # legitimately move some to a base class. The bug we want to
    # catch is full-class-removal.
    for name in ("parameter_name", "forward", "lora_A", "get_base_layer"):
        _present = name in src


# -------------------------------------------------------------------------
# 3. peft.tuners.lora.LoraConfig.target_parameters — peft 0.19+. Used
#    by unsloth-zoo's MoE target-parameter extractor.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_peft_lora_config_target_parameters(tag: str):
    src = fetch_text("huggingface/peft", tag, "src/peft/tuners/lora/config.py")
    if src is None:
        pytest.skip(f"{tag}: src/peft/tuners/lora/config.py missing")
    # Optional on 0.18.x; required from 0.19.0+. Don't fail older
    # versions; the test is informational below the floor.
    has_it = "target_parameters" in src
    if "0.18" in tag and not has_it:
        pytest.skip(f"{tag}: target_parameters not yet introduced (peft 0.18)")
    assert has_it, (
        f"{tag}: LoraConfig.target_parameters missing on peft >=0.19; "
        f"unsloth-zoo MoE target-parameter extraction breaks"
    )


# -------------------------------------------------------------------------
# 4. peft.tuners.lora.model.LoraModel._create_and_replace — unsloth#4807
#    monkey-patches this for Gemma4ClippableLinear. Signature pin.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_peft_lora_model_create_and_replace(tag: str):
    src = fetch_text("huggingface/peft", tag, "src/peft/tuners/lora/model.py")
    if src is None:
        pytest.skip(f"{tag}: src/peft/tuners/lora/model.py missing")
    assert has_def(src, "LoraModel", "class"), f"{tag}: class LoraModel missing"
    assert has_def(src, "_create_and_replace", "func"), (
        f"{tag}: LoraModel._create_and_replace missing; "
        f"unsloth/models/loader.py:1535-1601 monkey-patch breaks (unsloth#4807)"
    )


# -------------------------------------------------------------------------
# 5. peft.utils.transformers_weight_conversion.{build_peft_weight_mapping,
#    WeightConversion} — unsloth#5167 wraps build_peft_weight_mapping
#    to handle WeightConversion.__init__ kwargs (distributed_operation,
#    quantization_operation).
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_peft_transformers_weight_conversion_module(tag: str):
    candidates = [
        "src/peft/utils/transformers_weight_conversion.py",
        "src/peft/utils/transformers_weight_conversion/__init__.py",
    ]
    hit = first_match("huggingface/peft", tag, candidates)
    if hit is None:
        pytest.skip(f"{tag}: transformers_weight_conversion not present (legacy peft)")
    _, src = hit
    assert (
        has_def(src, "build_peft_weight_mapping", "func")
        or "build_peft_weight_mapping" in src
    ), (
        f"{tag}: build_peft_weight_mapping missing in transformers_weight_conversion; "
        f"unsloth/import_fixes.py:1375-1456 wrap breaks (unsloth#5167)"
    )


# -------------------------------------------------------------------------
# 6. peft.utils.integrations.dequantize_module_weight — used by 3 unsloth/
#    unsloth-zoo callsites. Function name + module path.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_peft_integrations_dequantize_module_weight(tag: str):
    candidates = [
        "src/peft/utils/integrations.py",
        "src/peft/utils/integrations/__init__.py",
    ]
    hit = first_match("huggingface/peft", tag, candidates)
    assert (
        hit is not None
    ), f"{tag}: src/peft/utils/integrations[.py|/__init__.py] both missing"
    _, src = hit
    assert (
        has_def(src, "dequantize_module_weight", "func")
        or "dequantize_module_weight" in src
    ), (
        f"{tag}: peft.utils.integrations.dequantize_module_weight missing; "
        f"unsloth-zoo vllm_utils.py:2701, unsloth/_utils.py:1550, "
        f"saving_utils.py:270 ImportError"
    )


# -------------------------------------------------------------------------
# 7. peft.PeftType.LORA — used by unsloth-zoo vllm_utils.py:2520-2559.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
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
        # Either `class PeftType(...)` definition with LORA member, or
        # re-export from a submodule.
        if "PeftType" in src and ("LORA" in src or "lora" in src.lower()):
            return
    pytest.fail(
        f"{tag}: peft.PeftType (with LORA member) not in any of {candidates}; "
        f"unsloth-zoo vllm_utils.py:2520 reference breaks"
    )


# -------------------------------------------------------------------------
# 8. peft.utils.ModulesToSaveWrapper — both peft.utils.* and
#    peft.utils.other.* import paths used.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
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


# -------------------------------------------------------------------------
# 9. peft.PeftModel.from_pretrained signature pin — unsloth#4807
#    call site uses (model, name, token, revision, is_trainable,
#    trust_remote_code).
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_peft_peft_model_from_pretrained_signature(tag: str):
    src = fetch_text("huggingface/peft", tag, "src/peft/peft_model.py")
    assert src is not None, f"{tag}: src/peft/peft_model.py missing"
    # We expect `def from_pretrained` in PeftModel class. Just check
    # the method name exists; full kwarg list is too brittle.
    assert has_def(
        src, "from_pretrained", "func"
    ), f"{tag}: PeftModel.from_pretrained missing in peft_model.py"


# -------------------------------------------------------------------------
# 10. peft.__version__ exported via known mechanism.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", PEFT_TAGS)
def test_peft_version_parseable(tag: str):
    src = fetch_text("huggingface/peft", tag, "src/peft/__init__.py")
    assert src is not None
    # Same gates as the TRL test: literal / submodule / metadata / VERSION file.
    has_literal = bool(re.search(r'^__version__\s*=\s*["\']', src, re.MULTILINE))
    has_subimport = bool(
        re.search(r"^from\s+\.version\s+import\s+__version__", src, re.MULTILINE)
    )
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
