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

"""Drift detectors for upstream pathologies that ``unsloth/import_fixes.py``
works around. One test per ``fix_*`` / ``patch_*`` function. Each asserts
the healthy upstream shape; if the pathology is active, fires
``pytest.fail("DRIFT DETECTED: ...")`` -- never ``pytest.skip`` -- so CI
goes red and the maintainer triages on the next PR. Runs under the
GPU-free harness in ``tests/conftest.py``."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import re
import sys
from importlib.metadata import version as importlib_version

import pytest


# Mirrors the local ``Version()`` in import_fixes.py (51-68): strip
# dev/alpha/beta/rc/local suffixes so packaging.Version doesn't choke.
from packaging.version import Version as _PkgVersion


def _safe_version(raw):
    raw_str = str(raw)
    base = raw_str.split("+", 1)[0]
    try:
        return _PkgVersion(base)
    except Exception:
        match = re.match(r"[0-9]+(?:\.[0-9]+)*", base)
        if not match:
            raise
        return _PkgVersion(match.group(0))


# ===========================================================================
# protobuf
# ===========================================================================


def test_protobuf_message_factory_get_prototype_or_get_message_class_present():
    """``fix_message_factory_issue`` (import_fixes.py 264-308)."""
    mf = pytest.importorskip("google.protobuf.message_factory")
    has_mf_class = hasattr(mf, "MessageFactory")
    has_get_prototype = has_mf_class and hasattr(mf.MessageFactory, "GetPrototype")
    has_get_message_class = hasattr(mf, "GetMessageClass")
    if not has_mf_class:
        pytest.fail(
            "DRIFT DETECTED: google.protobuf.message_factory.MessageFactory is "
            "missing entirely -- fix_message_factory_issue would inject a stub."
        )
    if not (has_get_prototype or has_get_message_class):
        pytest.fail(
            "DRIFT DETECTED: neither MessageFactory.GetPrototype nor "
            "module-level GetMessageClass is present; fix_message_factory_issue "
            "would inject the GetPrototype/GetMessageClass shim."
        )
    assert has_get_prototype or has_get_message_class


# ===========================================================================
# datasets
# ===========================================================================


def test_datasets_version_not_in_broken_recursion_range():
    """``patch_datasets`` (import_fixes.py 574-586). datasets 4.4.0-4.5.0
    inclusive trigger RLock recursion errors in the Arrow loader."""
    pytest.importorskip("datasets")
    ds_v = _safe_version(importlib_version("datasets"))
    lo = _PkgVersion("4.4.0")
    hi = _PkgVersion("4.5.0")
    assert not (lo <= ds_v <= hi), (
        f"datasets=={ds_v} lies in the 4.4.0-4.5.0 recursion-error "
        f"range that patch_datasets explicitly forbids. Downgrade to "
        f"datasets==4.3.0 or upgrade past 4.5.0."
    )


# ===========================================================================
# trl
# ===========================================================================


def test_trl_is_x_available_returns_bool_not_tuple():
    """``fix_trl_vllm_ascend`` (import_fixes.py 493-516). transformers >=4.48's
    ``_is_package_available`` returns ``(bool, version_or_None)``; TRL's
    ``is_*_available`` accessors must still return real bools."""
    pytest.importorskip("trl")
    try:
        import trl.import_utils as tiu
    except Exception as exc:
        pytest.skip(f"trl.import_utils not importable: {exc!r}")

    accessor_names = [
        n
        for n in dir(tiu)
        if n.startswith("is_")
        and n.endswith("_available")
        and callable(getattr(tiu, n, None))
    ]
    assert accessor_names, "trl.import_utils has no is_*_available accessors"

    bad = {}
    for name in accessor_names:
        accessor = getattr(tiu, name)
        try:
            sig = inspect.signature(accessor)
            required = [
                p
                for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            if required:
                continue
            result = accessor()
        except Exception:
            continue
        if not isinstance(result, bool):
            bad[name] = (type(result).__name__, result)

    if bad:
        pytest.fail(
            "DRIFT DETECTED: fix_trl_vllm_ascend coerces these accessors "
            f"from tuple-cached values to bool: {bad}"
        )


def test_trl_cached_available_flags_are_not_tuples():
    """``fix_trl_vllm_ascend`` (import_fixes.py 493-516). Same drift, checked
    on the module-level cached ``_*_available`` attributes directly."""
    pytest.importorskip("trl")
    try:
        import trl.import_utils as tiu
    except Exception as exc:
        pytest.skip(f"trl.import_utils not importable: {exc!r}")

    tuple_flags = {
        name: value
        for name, value in vars(tiu).items()
        if name.startswith("_")
        and name.endswith("_available")
        and isinstance(value, tuple)
    }
    if tuple_flags:
        pytest.fail(
            "DRIFT DETECTED: fix_trl_vllm_ascend needs to coerce these tuple-"
            f"cached flags to bool: {sorted(tuple_flags)}"
        )


# ===========================================================================
# transformers
# ===========================================================================


def test_pretrained_model_enable_input_require_grads_uses_old_pattern():
    """``patch_enable_input_require_grads`` (import_fixes.py 609-670). HF
    PR #41993 rewrote enable_input_require_grads to iterate
    ``self.modules()`` and call ``get_input_embeddings`` on every
    submodule; vision submodules then raise NotImplementedError. Healthy
    state: either the upstream rewrite isn't present (pre-HF#41993), OR
    the patch installed a NotImplementedError-tolerant replacement."""
    pytest.importorskip("transformers")
    from transformers import PreTrainedModel

    try:
        src = inspect.getsource(PreTrainedModel.enable_input_require_grads)
    except Exception as exc:
        pytest.skip(f"could not getsource(enable_input_require_grads): {exc!r}")

    if "for module in self.modules()" not in src:
        return  # healthy: pre-HF#41993 shape
    if "NotImplementedError" in src:
        return  # healthy: unsloth's tolerant replacement is installed

    pytest.fail(
        "DRIFT DETECTED: PreTrainedModel.enable_input_require_grads now "
        "iterates self.modules() (post HF#41993) and has NOT been "
        "wrapped by patch_enable_input_require_grads; vision submodules "
        "(e.g. GLM V4.6's self.visual) will raise NotImplementedError "
        "from get_input_embeddings and crash the whole call."
    )


def test_transformers_torchcodec_available_flag_is_present():
    """``disable_torchcodec_if_broken`` (import_fixes.py 1291-1317). Needs
    either the pre-5.x module-level ``_torchcodec_available`` flag, or
    the 5.x ``is_torchcodec_available`` public function; one of the two
    is the patch site the fix monkey-patches when FFmpeg is missing."""
    tf_iu = pytest.importorskip("transformers.utils.import_utils")
    has_flag = hasattr(tf_iu, "_torchcodec_available")
    has_func = callable(getattr(tf_iu, "is_torchcodec_available", None))
    assert has_flag or has_func, (
        "transformers.utils.import_utils dropped both "
        "``_torchcodec_available`` (pre-5.x) AND "
        "``is_torchcodec_available`` (>=5.x); "
        "disable_torchcodec_if_broken can no longer disable a broken "
        "torchcodec install."
    )


def test_transformers_is_causal_conv1d_available_symbol_present():
    """``_disable_transformers_causal_conv1d`` (import_fixes.py 1881-1895).
    Needs at least one of the causal_conv1d availability hooks."""
    tf_iu = pytest.importorskip("transformers.utils.import_utils")
    candidates = [
        "is_causal_conv1d_available",
        "_causal_conv1d_available",
        "_is_causal_conv1d_available",
    ]
    present = [name for name in candidates if hasattr(tf_iu, name)]
    if not present:
        pytest.fail(
            "DRIFT DETECTED: transformers.utils.import_utils dropped every "
            f"hook in {candidates}; _disable_transformers_causal_conv1d "
            "can no longer mask a broken causal_conv1d binary."
        )


# ===========================================================================
# transformers + accelerate (wandb checkers)
# ===========================================================================


def test_transformers_and_accelerate_is_wandb_available_callable():
    """``disable_broken_wandb`` (import_fixes.py 1320-1372). Patches
    is_wandb_available in transformers.integrations.integration_utils
    AND accelerate.utils.imports / accelerate.utils -- all three must
    keep existing."""
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from transformers.integrations import integration_utils as tf_integration
    import accelerate.utils.imports as acc_imports
    import accelerate.utils as acc_utils

    assert callable(getattr(tf_integration, "is_wandb_available", None)), (
        "transformers.integrations.integration_utils.is_wandb_available "
        "was removed/renamed; disable_broken_wandb can no longer mask a "
        "broken wandb install for trl trainers."
    )
    assert callable(getattr(acc_imports, "is_wandb_available", None)), (
        "accelerate.utils.imports.is_wandb_available removed; "
        "disable_broken_wandb cannot patch the source module."
    )
    assert callable(getattr(acc_utils, "is_wandb_available", None)), (
        "accelerate.utils.is_wandb_available removed; "
        "disable_broken_wandb cannot patch the re-export namespace "
        "consulted by trl/trainer/callbacks.py."
    )


# ===========================================================================
# peft
# ===========================================================================


def test_peft_transformers_weight_conversion_importable_and_signature():
    """``patch_peft_weight_converter_compatibility`` (import_fixes.py
    1375-1454). Wraps build_peft_weight_mapping to retrofit
    distributed_operation / quantization_operation kwargs; if the
    module is unimportable the wrap silently no-ops."""
    pytest.importorskip("peft")
    try:
        from peft.utils import transformers_weight_conversion as twc
    except Exception as exc:
        pytest.fail(
            "DRIFT DETECTED: peft.utils.transformers_weight_conversion "
            f"is unimportable on this stack ({exc!r}). "
            "patch_peft_weight_converter_compatibility will silently no-op."
        )

    assert hasattr(twc, "build_peft_weight_mapping"), (
        "build_peft_weight_mapping vanished from "
        "peft.utils.transformers_weight_conversion."
    )
    sig = inspect.signature(twc.build_peft_weight_mapping)
    expected_params = {"weight_conversions", "adapter_name"}
    actual_params = set(sig.parameters)
    assert expected_params.issubset(actual_params), (
        f"build_peft_weight_mapping signature drifted: expected at "
        f"least {sorted(expected_params)}, got {sorted(actual_params)}."
    )


# ===========================================================================
# triton
# ===========================================================================


def test_triton_compiled_kernel_has_num_ctas_and_cluster_dims():
    """``fix_triton_compiled_kernel_missing_attrs`` (import_fixes.py 923-968).
    triton 3.6+ dropped num_ctas / cluster_dims on CompiledKernel; torch
    2.9 Inductor's make_launcher still eagerly evaluates them."""
    pytest.importorskip("torch")
    triton_mod = pytest.importorskip("triton")  # noqa: F841
    tc = pytest.importorskip("triton.compiler.compiler")

    ck_cls = tc.CompiledKernel
    # Healthy if either: pre-3.6 class attr present, or unsloth wrapped
    # ``__init__`` to install num_ctas + cluster_dims per instance (the
    # post-3.6 shape ``fix_triton_compiled_kernel_missing_attrs`` lands).
    if hasattr(ck_cls, "num_ctas"):
        return
    init = getattr(ck_cls, "__init__", None)
    if init is not None:
        code = getattr(init, "__code__", None)
        freevars = set(getattr(code, "co_freevars", ()) or ())
        co_names = set(getattr(code, "co_names", ()) or ())
        if "_orig_init" in freevars or {"num_ctas", "cluster_dims"}.issubset(co_names):
            return

    pytest.fail(
        "DRIFT DETECTED: triton.CompiledKernel lacks the `num_ctas` "
        "class attribute AND ``__init__`` has not been wrapped by "
        "fix_triton_compiled_kernel_missing_attrs; torch Inductor's "
        "``make_launcher`` will crash on the eager "
        "``binary.metadata.num_ctas, *binary.metadata.cluster_dims`` "
        "unpack under torch.compile."
    )


# ===========================================================================
# torch + torchvision pairing table
# ===========================================================================


# Mirrors TORCH_TORCHVISION_COMPAT in torchvision_compatibility_check
# (import_fixes.py 708-798).
_TORCH_TORCHVISION_COMPAT = {
    (2, 9): (0, 24),
    (2, 8): (0, 23),
    (2, 7): (0, 22),
    (2, 6): (0, 21),
    (2, 5): (0, 20),
    (2, 4): (0, 19),
}


def _is_custom_torch_build(raw_version_str):
    if "+" not in raw_version_str:
        return False
    local = raw_version_str.split("+", 1)[1]
    if not local:
        return False
    return not re.fullmatch(r"cu\d[\d.]*|rocm\d[\d.]*|cpu|xpu", local, re.IGNORECASE)


def test_installed_torch_torchvision_pair_is_compatible():
    """``torchvision_compatibility_check`` (import_fixes.py 708-798).
    Raises ImportError when installed (torch, torchvision) pair fails
    the pinned compat table; custom / prerelease builds are warning-only."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    torch_raw = importlib_version("torch")
    tv_raw = importlib_version("torchvision")
    torch_v = _safe_version(torch_raw)
    tv_v = _safe_version(tv_raw)

    torch_major = torch_v.release[0]
    torch_minor = torch_v.release[1] if len(torch_v.release) > 1 else 0

    required = _TORCH_TORCHVISION_COMPAT.get((torch_major, torch_minor))
    if required is None:
        pytest.skip(
            f"torch=={torch_raw} is outside the pinned compatibility "
            f"table (entries cover 2.4-2.9). The formula fallback "
            f"in _infer_required_torchvision handles it at runtime."
        )

    pre_tags = (".dev", "a0", "b0", "rc", "alpha", "beta", "nightly")
    is_prerelease = any(t in torch_raw for t in pre_tags) or any(
        t in tv_raw for t in pre_tags
    )
    is_custom = _is_custom_torch_build(torch_raw) or _is_custom_torch_build(tv_raw)
    if is_prerelease or is_custom:
        pytest.skip(
            f"torch=={torch_raw} torchvision=={tv_raw} is a custom/"
            f"prerelease build; the runtime check downgrades to warning."
        )

    required_str = f"{required[0]}.{required[1]}.0"
    assert tv_v >= _PkgVersion(required_str), (
        f"DRIFT DETECTED: torch=={torch_raw} requires "
        f"torchvision>={required_str}, but torchvision=={tv_raw} is "
        f"installed. torchvision_compatibility_check would raise."
    )


# ===========================================================================
# vllm
# ===========================================================================


def test_vllm_guided_decoding_params_or_structured_outputs_present():
    """``fix_vllm_guided_decoding_params`` (import_fixes.py 446-490).
    vLLM PR #22772 renamed GuidedDecodingParams -> StructuredOutputsParams;
    trl still imports the old name so the fix re-aliases."""
    pytest.importorskip("vllm")
    try:
        sp = importlib.import_module("vllm.sampling_params")
    except Exception as exc:
        pytest.skip(f"vllm.sampling_params unimportable: {exc!r}")

    has_guided = hasattr(sp, "GuidedDecodingParams")
    has_structured = hasattr(sp, "StructuredOutputsParams")
    assert has_guided or has_structured, (
        "vllm.sampling_params has neither GuidedDecodingParams nor "
        "StructuredOutputsParams; fix_vllm_guided_decoding_params "
        "cannot re-alias. trl import path will break."
    )
    if not has_guided:
        pytest.fail(
            "DRIFT DETECTED: vllm.sampling_params only exposes "
            "StructuredOutputsParams (post PR #22772); "
            "fix_vllm_guided_decoding_params injects a GuidedDecodingParams "
            "alias so trl keeps importing."
        )


def test_vllm_aimv2_ovis_config_is_past_fix_version():
    """``fix_vllm_aimv2_issue`` (import_fixes.py 404-443). vLLM <0.10.1 has
    an Ovis config that unconditionally registers ``aimv2`` and trips a
    duplicate-key ValueError; the fix only touches old versions."""
    pytest.importorskip("vllm")
    vllm_v = _safe_version(importlib_version("vllm"))
    cutoff = _PkgVersion("0.10.1")
    if vllm_v < cutoff:
        pytest.fail(
            f"DRIFT DETECTED: vllm=={vllm_v} < {cutoff}; "
            "fix_vllm_aimv2_issue rewrites ovis.py to skip the duplicate "
            'AutoConfig.register("aimv2", ...) call.'
        )


# ===========================================================================
# huggingface_hub
# ===========================================================================


def test_huggingface_hub_is_offline_mode_or_hf_hub_offline_present():
    """``fix_huggingface_hub`` (import_fixes.py 913-920). huggingface_hub
    removed top-level ``is_offline_mode``; fix re-injects from
    ``huggingface_hub.constants.HF_HUB_OFFLINE``."""
    hub = pytest.importorskip("huggingface_hub")
    has_top_level = False
    try:
        has_top_level = callable(getattr(hub, "is_offline_mode", None))
    except Exception:
        has_top_level = False

    has_constant = False
    try:
        constants_mod = importlib.import_module("huggingface_hub.constants")
        has_constant = hasattr(constants_mod, "HF_HUB_OFFLINE")
    except Exception:
        has_constant = False

    assert has_top_level or has_constant, (
        "huggingface_hub dropped both ``is_offline_mode`` AND "
        "``huggingface_hub.constants.HF_HUB_OFFLINE``; "
        "fix_huggingface_hub can no longer re-inject the helper."
    )


# ===========================================================================
# torch
# ===========================================================================


def test_torch_nn_init_trunc_normal_exists():
    """``patch_trunc_normal_precision_issue`` (import_fixes.py 971-1050).
    fp16/bf16 stability wrapper monkey-patches torch.nn.init.trunc_normal_."""
    pytest.importorskip("torch")
    import torch.nn.init as init_mod

    assert callable(getattr(init_mod, "trunc_normal_", None)), (
        "torch.nn.init.trunc_normal_ removed/renamed; "
        "patch_trunc_normal_precision_issue cannot wrap it."
    )


# ===========================================================================
# xformers
# ===========================================================================


def test_xformers_is_post_num_splits_key_fix_or_not_installed():
    """``fix_xformers_performance_issue`` (import_fixes.py 312-341).
    xformers <0.0.29 has the ``num_splits_key=-1`` perf bug Unsloth
    rewrites at install time."""
    if importlib.util.find_spec("xformers") is None:
        pytest.skip("xformers not installed -- nothing to drift-check.")
    x_v = _safe_version(importlib_version("xformers"))
    cutoff = _PkgVersion("0.0.29")
    if x_v < cutoff:
        pytest.fail(
            f"DRIFT DETECTED: xformers=={x_v} < {cutoff}; "
            "fix_xformers_performance_issue rewrites "
            "ops/fmha/cutlass.py num_splits_key=-1 -> None."
        )


# ===========================================================================
# transformers (PreTrainedModel base import sanity)
# ===========================================================================


def test_transformers_pretrained_model_has_get_input_embeddings():
    """``patch_enable_input_require_grads`` (import_fixes.py 609-670).
    The replacement function calls ``get_input_embeddings`` on every
    submodule, so the accessor must still exist."""
    pytest.importorskip("transformers")
    from transformers import PreTrainedModel

    assert hasattr(PreTrainedModel, "get_input_embeddings"), (
        "PreTrainedModel.get_input_embeddings was renamed or removed; "
        "patch_enable_input_require_grads's replacement no longer compiles."
    )


# ===========================================================================
# accelerate -- ``is_X_available`` API stability used across the fixes
# ===========================================================================


def test_accelerate_utils_imports_module_present():
    """``disable_broken_wandb`` + ``fix_trl_vllm_ascend`` (import_fixes.py
    493-516, 1320-1372). Both reach into accelerate.utils.imports."""
    pytest.importorskip("accelerate")
    mod = pytest.importorskip("accelerate.utils.imports")
    # is_wandb_available is the canonical representative -- disable_broken_wandb
    # specifically targets it, so its absence breaks the patch.
    assert hasattr(mod, "is_wandb_available"), (
        "accelerate.utils.imports.is_wandb_available is gone; "
        "disable_broken_wandb cannot patch the source module."
    )
