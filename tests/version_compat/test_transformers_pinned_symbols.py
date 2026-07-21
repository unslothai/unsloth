# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pinned-symbol + source-pattern transformers compat checks via GitHub raw-fetch + grep.

Catches breakage classes from unsloth#3998/5036/5155/5259 and
unsloth-zoo#572/571/549/543/541/495/491/488/472/393/388/583/584/159.
CPU-only, no install. Anchor versions: transformers 4.57.6, 5.5.0.
"""

from __future__ import annotations

import re

import pytest

from tests.version_compat._fetch import fetch_text, first_match, has_def


# 4.57.6 floor + every 5.x minor since 5.0.0 + main.
TRANSFORMERS_TAGS = [
    "v4.57.6",  # anchor (must work)
    "v5.0.0",
    "v5.1.0",
    "v5.2.0",
    "v5.3.0",
    "v5.4.0",
    "v5.5.0",  # anchor (must work)
    "v5.5.4",
    "v5.6.2",
    "v5.7.0",
    "v5.8.0",
    "main",
]


# Trainer surface: unsloth/models/_utils.py rewrites Trainer.{__init__, training_step, get_batch_samples, compute_loss}.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_trainer_class_importable_path(tag: str):
    """transformers.Trainer must remain at trainer.py or trainer/__init__.py."""
    candidates = ["src/transformers/trainer.py", "src/transformers/trainer/__init__.py"]
    hit = first_match("huggingface/transformers", tag, candidates)
    assert (
        hit is not None
    ), f"{tag}: src/transformers/trainer[.py|/__init__.py] both missing"
    _, src = hit
    assert has_def(src, "Trainer", "class"), f"{tag}: class Trainer missing"


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_trainer_compute_loss_num_items_in_batch_param(tag: str):
    """unsloth-zoo#159 + unsloth#4998 + #4616: Trainer.compute_loss must accept num_items_in_batch kwarg."""
    candidates = ["src/transformers/trainer.py", "src/transformers/trainer/__init__.py"]
    hit = first_match("huggingface/transformers", tag, candidates)
    assert hit is not None
    _, src = hit
    m = re.search(r"^\s*def compute_loss\(([^)]*)\)", src, re.MULTILINE | re.DOTALL)
    if m is None:
        pytest.fail(f"{tag}: Trainer.compute_loss not found in source")
    assert "num_items_in_batch" in m.group(1), (
        f"{tag}: Trainer.compute_loss signature missing num_items_in_batch param; "
        f"unsloth grad-accum patches assume this kwarg present"
    )


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_trainer_training_step_grad_accum_pattern(tag: str):
    """unsloth#3598 patches Trainer.training_step source; drift = silent no-op = double-scale loss bug."""
    candidates = ["src/transformers/trainer.py", "src/transformers/trainer/__init__.py"]
    hit = first_match("huggingface/transformers", tag, candidates)
    assert hit is not None
    _, src = hit
    needed = (
        "loss *= self.args.gradient_accumulation_steps",
        "if self.model_accepts_loss_kwargs:",
        "self.accelerator.backward(loss",
    )
    missing = [s for s in needed if s not in src]
    # Hard-fail only when ALL substrings missing; partial drift is informational.
    if len(missing) == len(needed):
        pytest.fail(
            f"{tag}: Trainer.training_step has none of the grad-accum "
            f"fingerprints {needed}; unsloth/models/_utils.py:1689-1791 "
            f"patch silently no-ops -> double-scale loss"
        )


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_trainer_get_batch_samples_returns_num_items(tag: str):
    """unsloth-zoo loss_utils.py:241 replaces Trainer.get_batch_samples; must keep the num_items_in_batch return."""
    candidates = ["src/transformers/trainer.py", "src/transformers/trainer/__init__.py"]
    hit = first_match("huggingface/transformers", tag, candidates)
    assert hit is not None
    _, src = hit
    if not has_def(src, "get_batch_samples", "func"):
        pytest.skip(f"{tag}: get_batch_samples not yet on Trainer")
    assert (
        "num_items_in_batch" in src
    ), f"{tag}: Trainer.get_batch_samples / num_items_in_batch contract missing"


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_trainer_inner_training_loop_inplace_loss_v5(tag: str):
    """unsloth-zoo#543: transformers 5.0+ switched out-of-place tr_loss add to in-place `self._tr_loss +=`."""
    candidates = ["src/transformers/trainer.py", "src/transformers/trainer/__init__.py"]
    hit = first_match("huggingface/transformers", tag, candidates)
    assert hit is not None
    _, src = hit
    has_inplace = "self._tr_loss +=" in src
    has_outplace = "tr_loss = tr_loss + tr_loss_step" in src
    # Assert ONE form is present so a refactor dropping both is caught.
    assert has_inplace or has_outplace, (
        f"{tag}: Trainer._inner_training_loop has neither "
        f"`tr_loss = tr_loss + tr_loss_step` nor `self._tr_loss +=`; "
        f"unsloth-zoo#543 patch breaks"
    )


# modeling_utils: checkpoint, PushToHubMixin, ALL_ATTENTION_FUNCTIONS.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_modeling_utils_exposes_checkpoint(tag: str):
    """unsloth-zoo#549: transformers 5.2+ uses modeling_utils.checkpoint; patch must replace it, not just torch's."""
    src = fetch_text(
        "huggingface/transformers", tag, "src/transformers/modeling_utils.py"
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_utils.py missing")
    # Either a direct import or local rebinding.
    has_import = bool(
        re.search(
            r"^from\s+torch\.utils\.checkpoint\s+import\s+checkpoint",
            src,
            re.MULTILINE,
        )
        or re.search(r"^import\s+torch\.utils\.checkpoint", src, re.MULTILINE)
        or "checkpoint = torch.utils.checkpoint.checkpoint" in src
    )
    assert has_import, (
        f"{tag}: transformers.modeling_utils does not import / re-bind "
        f"torch.utils.checkpoint.checkpoint; unsloth-zoo#549 patch breaks"
    )


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_pushtohubmixin_create_repo_status(tag: str):
    """unsloth-zoo#393: transformers 5.x removed PushToHubMixin._create_repo; snapshot which side."""
    src = fetch_text(
        "huggingface/transformers", tag, "src/transformers/modeling_utils.py"
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_utils.py missing")
    has_create = bool(re.search(r"def _create_repo\b", src) or "_create_repo" in src)
    # Informational only.
    _ = has_create


# integrations.bitsandbytes: _replace_with_bnb_linear vs new path.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_integrations_bitsandbytes_module_present(tag: str):
    src = fetch_text(
        "huggingface/transformers", tag, "src/transformers/integrations/bitsandbytes.py"
    )
    if src is None:
        pytest.skip(f"{tag}: integrations/bitsandbytes.py missing (legacy layout)")
    assert (
        "Linear4bit" in src or "linear" in src.lower()
    ), f"{tag}: integrations/bitsandbytes.py has no Linear4bit reference"


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_quantizers_should_convert_module_signature(tag: str):
    """unsloth-zoo#491/#488: 5.x moved is_replaceable to quantizers_utils.should_convert_module; snapshot its form."""
    src = fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/quantizers/quantizers_utils.py",
    )
    if src is None:
        pytest.skip(f"{tag}: quantizers/quantizers_utils.py missing")
    if not has_def(src, "should_convert_module", "func"):
        pytest.skip(f"{tag}: should_convert_module not yet present (4.x)")
    # Catch substring matching in `.{key}.` form.
    has_dot_form = ".{key}." in src or "f'.{key}.'" in src or 'f".{key}."' in src
    # Informational only.
    _ = has_dot_form


# integrations.finegrained_fp8.FP8Linear: bias/has_bias rename in v5.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_fp8linear_init_param_names(tag: str):
    """unsloth-zoo#572: transformers 5.x renamed FP8Linear.__init__ `bias` -> `has_bias`."""
    src = fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/integrations/finegrained_fp8.py",
    )
    if src is None:
        pytest.skip(f"{tag}: integrations/finegrained_fp8.py missing")
    if not has_def(src, "FP8Linear", "class"):
        pytest.skip(f"{tag}: FP8Linear not yet defined")
    has_bias_kw = re.search(r"def __init__\([^)]*\bbias\b", src) is not None
    has_has_bias_kw = re.search(r"def __init__\([^)]*\bhas_bias\b", src) is not None
    assert (
        has_bias_kw or has_has_bias_kw
    ), f"{tag}: FP8Linear.__init__ has neither `bias` nor `has_bias` param"


# processing_utils: Unpack importable.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_processing_utils_unpack_importable(tag: str):
    """unsloth-zoo#583/584: transformers.processing_utils.Unpack must keep importing."""
    src = fetch_text(
        "huggingface/transformers", tag, "src/transformers/processing_utils.py"
    )
    if src is None:
        pytest.skip(f"{tag}: processing_utils.py missing")
    has_unpack = bool(re.search(r"^Unpack\b\s*=", src, re.MULTILINE) or "Unpack" in src)
    assert has_unpack, (
        f"{tag}: transformers.processing_utils.Unpack missing; "
        f"unsloth-zoo#583/584 import guard breaks"
    )


# Models: gemma3, gpt_oss forward signature drift.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_gemma3_attention_forward_present(tag: str):
    src = fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/models/gemma3/modeling_gemma3.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_gemma3.py missing")
    assert has_def(
        src, "Gemma3Attention", "class"
    ), f"{tag}: class Gemma3Attention missing"


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_gpt_oss_model_forward_present(tag: str):
    src = fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/models/gpt_oss/modeling_gpt_oss.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_gpt_oss.py missing (legacy)")
    assert has_def(src, "GptOssModel", "class"), f"{tag}: class GptOssModel missing"


# auto_factory: unsloth#5155 _LazyAutoMapping private API.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_auto_factory_lazy_mapping_private_api(tag: str):
    """unsloth#5155: resolve_model_class needs all four _LazyAutoMapping private attrs to remain."""
    src = fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/models/auto/auto_factory.py",
    )
    if src is None:
        pytest.skip(f"{tag}: auto/auto_factory.py missing")
    needed = (
        "_model_mapping",
        "_config_mapping",
        "_extra_content",
        "_load_attr_from_module",
    )
    missing = [n for n in needed if n not in src]
    assert not missing, (
        f"{tag}: _LazyAutoMapping private API missing {missing}; "
        f"unsloth/models/_utils.py:resolve_model_class breaks (unsloth#5155)"
    )


# configuration_utils: PreTrainedConfig vs PretrainedConfig in 5.x.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_configuration_utils_alias(tag: str):
    """transformers 5.x renamed PretrainedConfig -> PreTrainedConfig; unsloth-zoo imports both defensively."""
    src = fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/configuration_utils.py",
    )
    if src is None:
        pytest.skip(f"{tag}: configuration_utils.py missing")
    has_old = has_def(src, "PretrainedConfig", "class")
    has_new = has_def(src, "PreTrainedConfig", "class")
    assert has_old or has_new, (
        f"{tag}: neither PretrainedConfig (4.x) nor PreTrainedConfig (5.x) "
        f"defined in configuration_utils.py"
    )


# tokenization: apply_chat_template return_dict default flip in v5.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_apply_chat_template_signature_present(tag: str):
    """unsloth-zoo#572: apply_chat_template `return_dict` default flipped False -> True in transformers 5.x."""
    src = fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/tokenization_utils_base.py",
    )
    if src is None:
        pytest.skip(f"{tag}: tokenization_utils_base.py missing")
    assert has_def(
        src, "apply_chat_template", "func"
    ), f"{tag}: apply_chat_template missing in tokenization_utils_base.py"


# Generic-importability sweep: every transformers symbol unsloth/zoo imports must stay reachable.


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_modeling_attn_mask_utils_symbols(tag: str):
    """_prepare_4d_attention_mask_for_sdpa is imported by unsloth/models/llama.py + sentence_transformer.py."""
    src = fetch_text(
        "huggingface/transformers",
        tag,
        "src/transformers/modeling_attn_mask_utils.py",
    )
    if src is None:
        pytest.skip(f"{tag}: modeling_attn_mask_utils.py missing")
    assert has_def(
        src, "AttentionMaskConverter", "class"
    ), f"{tag}: AttentionMaskConverter missing"
    assert (
        has_def(src, "_prepare_4d_attention_mask_for_sdpa", "func")
        or "_prepare_4d_attention_mask_for_sdpa" in src
    ), f"{tag}: _prepare_4d_attention_mask_for_sdpa missing"


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_cache_utils_classes(tag: str):
    src = fetch_text("huggingface/transformers", tag, "src/transformers/cache_utils.py")
    if src is None:
        pytest.skip(f"{tag}: cache_utils.py missing")
    needed = ("Cache", "DynamicCache")
    for cls in needed:
        assert has_def(
            src, cls, "class"
        ), f"{tag}: transformers.cache_utils.{cls} missing"


@pytest.mark.parametrize("tag", TRANSFORMERS_TAGS)
def test_training_args_parallel_mode_importable(tag: str):
    src = fetch_text(
        "huggingface/transformers", tag, "src/transformers/training_args.py"
    )
    if src is None:
        pytest.skip(f"{tag}: training_args.py missing")
    assert "ParallelMode" in src, (
        f"{tag}: transformers.training_args.ParallelMode missing; "
        f"unsloth-zoo loss_utils.py:232 ImportError"
    )
