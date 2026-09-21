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

"""Drift detectors for the upstream pathologies ``unsloth/import_fixes.py``
works around; one test per ``fix_*`` / ``patch_*``, each fails (never skips)
when the pathology is active. Runs under the GPU-free ``tests/conftest.py``."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import platform
import re
import sys
from pathlib import Path
from importlib.metadata import version as importlib_version

import pytest


# Mirrors import_fixes.py's local Version(): strip dev/alpha/beta/rc/local suffixes.
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


def test_protobuf_message_factory_get_prototype_or_get_message_class_present():
    """``fix_message_factory_issue``."""
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


def test_datasets_version_not_in_broken_recursion_range():
    """``patch_datasets``: datasets 4.4.0-4.5.0 hit RLock recursion in the Arrow loader."""
    pytest.importorskip("datasets")
    ds_v = _safe_version(importlib_version("datasets"))
    lo = _PkgVersion("4.4.0")
    hi = _PkgVersion("4.5.0")
    assert not (lo <= ds_v <= hi), (
        f"datasets=={ds_v} lies in the 4.4.0-4.5.0 recursion-error "
        f"range that patch_datasets explicitly forbids. Downgrade to "
        f"datasets==4.3.0 or upgrade past 4.5.0."
    )


def test_trl_is_x_available_returns_bool_not_tuple():
    """``fix_trl_vllm_ascend``: TRL's ``is_*_available`` must still return bools
    after transformers >=4.48 made ``_is_package_available`` return a tuple."""
    pytest.importorskip("trl")
    try:
        import trl.import_utils as tiu
    except Exception as exc:
        pytest.skip(f"trl.import_utils not importable: {exc!r}")

    accessor_names = [
        n
        for n in dir(tiu)
        if n.startswith("is_") and n.endswith("_available") and callable(getattr(tiu, n, None))
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
    """``fix_trl_vllm_ascend``: same drift on the module-level cached ``_*_available`` attrs."""
    pytest.importorskip("trl")
    try:
        import trl.import_utils as tiu
    except Exception as exc:
        pytest.skip(f"trl.import_utils not importable: {exc!r}")

    tuple_flags = {
        name: value
        for name, value in vars(tiu).items()
        if name.startswith("_") and name.endswith("_available") and isinstance(value, tuple)
    }
    if tuple_flags:
        pytest.fail(
            "DRIFT DETECTED: fix_trl_vllm_ascend needs to coerce these tuple-"
            f"cached flags to bool: {sorted(tuple_flags)}"
        )


def test_pretrained_model_enable_input_require_grads_uses_old_pattern():
    """``patch_enable_input_require_grads``: HF PR #41993 made
    enable_input_require_grads iterate ``self.modules()``, so vision submodules
    raise NotImplementedError unless the tolerant replacement is installed."""
    pytest.importorskip("transformers")
    from transformers import PreTrainedModel

    try:
        src = inspect.getsource(PreTrainedModel.enable_input_require_grads)
    except Exception as exc:
        pytest.skip(f"could not getsource(enable_input_require_grads): {exc!r}")

    if "for module in self.modules()" not in src:
        return  # pre-HF#41993 shape
    if "NotImplementedError" in src:
        return  # tolerant replacement installed

    pytest.fail(
        "DRIFT DETECTED: PreTrainedModel.enable_input_require_grads now "
        "iterates self.modules() (post HF#41993) and has NOT been "
        "wrapped by patch_enable_input_require_grads; vision submodules "
        "(e.g. GLM V4.6's self.visual) will raise NotImplementedError "
        "from get_input_embeddings and crash the whole call."
    )


def test_transformers_torchcodec_available_flag_is_present():
    """``disable_torchcodec_if_broken``: needs the pre-5.x ``_torchcodec_available``
    flag or 5.x ``is_torchcodec_available`` as its patch site when FFmpeg is missing."""
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
    """``_disable_transformers_causal_conv1d``: needs a causal_conv1d availability hook."""
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


def test_transformers_and_accelerate_is_wandb_available_callable():
    """``disable_broken_wandb``: patches is_wandb_available in three modules
    (transformers integration_utils + accelerate imports/utils); all must exist."""
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


def test_peft_transformers_weight_conversion_importable_and_signature():
    """``patch_peft_weight_converter_compatibility``: wraps build_peft_weight_mapping;
    silently no-ops if the module is unimportable."""
    pytest.importorskip("peft")
    # transformers_weight_conversion arrived in PEFT 0.19.0 (huggingface/peft 5356277d), but
    # both pyprojects allow >=0.18.0, where the patch returns early and there is no drift.
    peft_version = _safe_version(importlib_version("peft"))
    if peft_version < _PkgVersion("0.19.0"):
        pytest.skip(
            f"peft {peft_version} predates peft.utils.transformers_weight_conversion "
            "(added in 0.19.0); patch_peft_weight_converter_compatibility no-ops by "
            "design below it, so its absence is the declared floor, not drift."
        )
    try:
        from peft.utils import transformers_weight_conversion as twc
    except Exception as exc:
        pytest.fail(
            "DRIFT DETECTED: peft.utils.transformers_weight_conversion "
            f"is unimportable on this stack ({exc!r}). "
            "patch_peft_weight_converter_compatibility will silently no-op."
        )

    assert hasattr(
        twc, "build_peft_weight_mapping"
    ), "build_peft_weight_mapping vanished from peft.utils.transformers_weight_conversion."
    sig = inspect.signature(twc.build_peft_weight_mapping)
    expected_params = {"weight_conversions", "adapter_name"}
    actual_params = set(sig.parameters)
    assert expected_params.issubset(actual_params), (
        f"build_peft_weight_mapping signature drifted: expected at "
        f"least {sorted(expected_params)}, got {sorted(actual_params)}."
    )


def test_triton_compiled_kernel_has_num_ctas_and_cluster_dims():
    """``fix_triton_compiled_kernel_missing_attrs``: triton 3.6+ dropped
    num_ctas/cluster_dims on CompiledKernel, but Inductor's make_launcher needs them."""
    pytest.importorskip("torch")
    triton_mod = pytest.importorskip("triton")  # noqa: F841
    tc = pytest.importorskip("triton.compiler.compiler")

    ck_cls = tc.CompiledKernel
    # Healthy if the pre-3.6 class attr is present, or __init__ is wrapped to install num_ctas + cluster_dims per
    # instance (the post-3.6 fix).
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


# Mirrors TORCH_TORCHVISION_COMPAT in torchvision_compatibility_check.
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
    """``torchvision_compatibility_check``: raises when the (torch, torchvision)
    pair fails the pinned table; custom/prerelease builds are warning-only."""
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
    is_prerelease = any(t in torch_raw for t in pre_tags) or any(t in tv_raw for t in pre_tags)
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


def test_vllm_guided_decoding_params_or_structured_outputs_present():
    """``fix_vllm_guided_decoding_params``: vLLM PR #22772 renamed
    GuidedDecodingParams -> StructuredOutputsParams; the fix re-aliases for trl."""
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
    """``fix_vllm_aimv2_issue``: vLLM <0.10.1 double-registers ``aimv2`` (duplicate-key
    ValueError); the fix only touches old versions."""
    pytest.importorskip("vllm")
    vllm_v = _safe_version(importlib_version("vllm"))
    cutoff = _PkgVersion("0.10.1")
    if vllm_v < cutoff:
        pytest.fail(
            f"DRIFT DETECTED: vllm=={vllm_v} < {cutoff}; "
            "fix_vllm_aimv2_issue rewrites ovis.py to skip the duplicate "
            'AutoConfig.register("aimv2", ...) call.'
        )


def test_huggingface_hub_is_offline_mode_or_hf_hub_offline_present():
    """``fix_huggingface_hub``: re-injects top-level ``is_offline_mode`` from
    ``constants.HF_HUB_OFFLINE`` after huggingface_hub dropped it."""
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


def test_torch_nn_init_trunc_normal_exists():
    """``patch_trunc_normal_precision_issue``: fp16/bf16 wrapper monkey-patches
    torch.nn.init.trunc_normal_, which must still exist."""
    pytest.importorskip("torch")
    import torch.nn.init as init_mod

    assert callable(getattr(init_mod, "trunc_normal_", None)), (
        "torch.nn.init.trunc_normal_ removed/renamed; "
        "patch_trunc_normal_precision_issue cannot wrap it."
    )


def test_xformers_is_post_num_splits_key_fix_or_not_installed():
    """``fix_xformers_performance_issue``: xformers <0.0.29 has the
    ``num_splits_key=-1`` perf bug Unsloth rewrites at install time."""
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


def test_transformers_pretrained_model_has_get_input_embeddings():
    """``patch_enable_input_require_grads``: its replacement calls
    ``get_input_embeddings`` per submodule, so the accessor must still exist."""
    pytest.importorskip("transformers")
    from transformers import PreTrainedModel

    assert hasattr(PreTrainedModel, "get_input_embeddings"), (
        "PreTrainedModel.get_input_embeddings was renamed or removed; "
        "patch_enable_input_require_grads's replacement no longer compiles."
    )


# Regression for https://github.com/unslothai/unsloth/issues/4188: Qwen3_5ForConditionalGeneration uses
# loss_type='ForConditionalGeneration', a separate LOSS_MAPPING key left unpatched, falling back to stock
# ForCausalLMLoss whose logits.float() OOMs on <=24 GB GPUs.
def _reset_loss_mapping(mapping, saved):
    mapping.clear()
    mapping.update(saved)


def test_patch_loss_functions_covers_conditional_generation():
    """patch_loss_functions() must repoint every ForCausalLMLoss alias to the
    Unsloth kernel, not just LOSS_MAPPING['ForCausalLM']."""
    lu = pytest.importorskip("transformers.loss.loss_utils")
    cel = pytest.importorskip("unsloth.kernels.cross_entropy_loss")

    saved = dict(lu.LOSS_MAPPING)
    try:
        cel.patch_loss_functions(torch_compile = False)

        unsloth_loss = lu.LOSS_MAPPING.get("ForCausalLM")
        assert unsloth_loss is not None
        assert "Unsloth" in str(
            unsloth_loss
        ), f"LOSS_MAPPING['ForCausalLM'] was not replaced: {unsloth_loss}"

        cg_loss = lu.LOSS_MAPPING.get("ForConditionalGeneration")
        assert cg_loss is unsloth_loss, (
            f"LOSS_MAPPING['ForConditionalGeneration'] not patched: {cg_loss}. "
            f"Qwen3_5ForConditionalGeneration will silently use the stock "
            f"ForCausalLMLoss and OOM at large sequence lengths."
        )
    finally:
        _reset_loss_mapping(lu.LOSS_MAPPING, saved)


def test_patch_loss_functions_does_not_touch_other_loss_types():
    """patch_loss_functions() must not overwrite unrelated loss types with the causal-LM kernel."""
    lu = pytest.importorskip("transformers.loss.loss_utils")
    cel = pytest.importorskip("unsloth.kernels.cross_entropy_loss")

    non_causal_keys = {
        k for k, v in lu.LOSS_MAPPING.items() if getattr(v, "__name__", "") != "ForCausalLMLoss"
    }

    saved = dict(lu.LOSS_MAPPING)
    try:
        cel.patch_loss_functions(torch_compile = False)

        unsloth_loss = lu.LOSS_MAPPING.get("ForCausalLM")
        for key in non_causal_keys:
            assert lu.LOSS_MAPPING.get(key) is not unsloth_loss, (
                f"patch_loss_functions() incorrectly overwrote "
                f"LOSS_MAPPING['{key}'] with the Unsloth ForCausalLM kernel."
            )
    finally:
        _reset_loss_mapping(lu.LOSS_MAPPING, saved)


def test_accelerate_utils_imports_module_present():
    """``disable_broken_wandb`` + ``fix_trl_vllm_ascend`` both reach into
    accelerate.utils.imports."""
    pytest.importorskip("accelerate")
    mod = pytest.importorskip("accelerate.utils.imports")
    # is_wandb_available is the canonical target of disable_broken_wandb.
    assert hasattr(mod, "is_wandb_available"), (
        "accelerate.utils.imports.is_wandb_available is gone; "
        "disable_broken_wandb cannot patch the source module."
    )


def test_accelerate_recursively_apply_empty_logits_patch():
    """patch_accelerate_recursively_apply overrides recursively_apply to bypass EmptyLogits."""
    pytest.importorskip("accelerate")

    import accelerate.utils.operations as acc_ops
    from unsloth.import_fixes import patch_accelerate_recursively_apply

    class EmptyLogits:
        pass

    e = EmptyLogits()
    patch_accelerate_recursively_apply()

    res = acc_ops.recursively_apply(lambda x: x, e, error_on_other_type = True)
    assert res is e


def test_accelerate_gather_empty_logits_debug_mode_patch():
    """gather and broadcast bypass EmptyLogits when debug mode is enabled."""
    pytest.importorskip("accelerate")
    from accelerate.state import PartialState, DistributedType
    import accelerate.utils.operations as acc_ops
    from unsloth.import_fixes import patch_accelerate_recursively_apply
    import unittest.mock as mock
    import torch

    class EmptyLogits:
        pass

    e = EmptyLogits()
    patch_accelerate_recursively_apply()

    state = PartialState()
    orig_debug = state.debug
    orig_dist_type = state.distributed_type
    orig_num_processes = state.num_processes
    orig_device = state.device

    state.debug = True
    state.distributed_type = DistributedType.MULTI_GPU
    state.num_processes = 2

    def mock_gather_object(obj, *args, **kwargs):
        return [obj] * state.num_processes

    def mock_gpu_gather(tensor, *args, **kwargs):
        def _gather_one(t):
            if t.ndim == 0:
                t = t.clone()[None]
            return torch.cat([t] * state.num_processes, dim = 0)

        return acc_ops.recursively_apply(_gather_one, tensor, error_on_other_type = True)

    def mock_gpu_broadcast(data, *args, **kwargs):
        return data

    try:
        with (
            mock.patch(
                "accelerate.utils.operations.gather_object",
                side_effect = mock_gather_object,
            ),
            mock.patch("accelerate.utils.operations._gpu_gather", side_effect = mock_gpu_gather),
            mock.patch(
                "accelerate.utils.operations._gpu_broadcast",
                side_effect = mock_gpu_broadcast,
            ),
        ):
            state.device = torch.device("cpu")

            res = acc_ops.gather(e)
            assert res is e

            res_nested = acc_ops.gather([e])
            assert isinstance(res_nested, list) and res_nested[0] is e

            # Mixed payload: real tensor gets gathered, EmptyLogits passes through.
            # Tensor must live on state.device or debug-mode device check fails on GPUs.
            real_tensor = torch.tensor([42], device = state.device)
            payload = {"labels": real_tensor, "logits": e}
            res_mixed = acc_ops.gather(payload)

            assert isinstance(res_mixed, dict)
            assert res_mixed["logits"] is e
            # num_processes = 2 -> gathered to [42, 42]
            assert torch.equal(res_mixed["labels"], torch.tensor([42, 42], device = state.device))

            res_broadcast = acc_ops.broadcast(e)
            assert res_broadcast is e

            res_broadcast_mixed = acc_ops.broadcast(payload)
            assert isinstance(res_broadcast_mixed, dict)
            assert res_broadcast_mixed["logits"] is e
            assert torch.equal(res_broadcast_mixed["labels"], real_tensor)
    finally:
        state.debug = orig_debug
        state.distributed_type = orig_dist_type
        state.num_processes = orig_num_processes
        state.device = orig_device


def test_accelerate_patch_is_idempotent():
    """Calling patch_accelerate_recursively_apply twice must not stack wrappers."""
    pytest.importorskip("accelerate")
    import accelerate.utils.operations as acc_ops
    from unsloth.import_fixes import patch_accelerate_recursively_apply

    patch_accelerate_recursively_apply()
    recursively_apply = acc_ops.recursively_apply
    find_device = acc_ops.find_device
    patch_accelerate_recursively_apply()
    assert (
        acc_ops.recursively_apply is recursively_apply
    ), "DRIFT DETECTED: recursively_apply was wrapped twice."
    assert acc_ops.find_device is find_device, "DRIFT DETECTED: find_device was wrapped twice."


def test_accelerate_find_device_skips_empty_logits():
    """find_device must search past EmptyLogits and keep None for tensor-free data."""
    pytest.importorskip("accelerate")
    import torch
    import accelerate.utils.operations as acc_ops
    from accelerate.state import PartialState
    from unsloth.import_fixes import patch_accelerate_recursively_apply

    class EmptyLogits:
        pass

    patch_accelerate_recursively_apply()
    tensor = torch.tensor([1.0])
    # Leading sentinel must not stop the search before the real tensor
    assert acc_ops.find_device({"logits": EmptyLogits(), "labels": tensor}) == tensor.device
    # Tensor-free payloads keep returning None (AlignDevicesHook needs it to skip moves)
    assert acc_ops.find_device({"a": 1}) is None
    # Sentinel-only payloads fall back to current device so debug-mode find_device(...).type doesn't raise
    # AttributeError
    assert acc_ops.find_device(EmptyLogits()) == PartialState().device


def test_accelerate_patch_wired_into_gpu_init():
    """The patch must be installed at startup, not only importable."""
    source = Path(__file__).resolve().parent.parent / "unsloth" / "_gpu_init.py"
    source = source.read_text(encoding = "utf-8")
    assert "patch_accelerate_recursively_apply()" in source, (
        "DRIFT DETECTED: patch_accelerate_recursively_apply is defined but "
        "never called in _gpu_init.py, so real imports never install it."
    )


# ===========================================================================
# bitsandbytes -- ROCm arch / warp-size detection shape
# ===========================================================================


def test_bitsandbytes_rocm_detection_helpers_recognizable():
    """``fix_bitsandbytes_rocm_arch_detection``: the source sniff only patches
    bnb's ROCm helpers in recognized shapes; fail (don't import) when it drifts."""
    spec = importlib.util.find_spec("bitsandbytes")
    if spec is None:
        pytest.skip("bitsandbytes not installed -- nothing to drift-check.")
    cuda_specs_path = None
    for location in spec.submodule_search_locations or []:
        candidate = os.path.join(location, "cuda_specs.py")
        if os.path.isfile(candidate):
            cuda_specs_path = candidate
            break
    if cuda_specs_path is None:
        pytest.skip("bitsandbytes has no cuda_specs.py (pre-ROCm version).")

    import ast

    with open(cuda_specs_path, "r", encoding = "utf-8") as f:
        source = f.read()
    helpers = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name in ("get_rocm_gpu_arch", "get_rocm_warpsize")
    ]
    if not helpers:
        pytest.skip("bitsandbytes cuda_specs has no ROCm detection helpers.")
    for node in helpers:
        segment = ast.get_source_segment(source, node) or ""
        recognized = (
            "subprocess" in segment
            or "get_device_properties" in segment
            or "gcnArchName" in segment
        )
        if not recognized:
            pytest.fail(
                f"DRIFT DETECTED: bitsandbytes.cuda_specs.{node.name} uses "
                "neither subprocess nor torch device properties; "
                "fix_bitsandbytes_rocm_arch_detection's shape sniff will "
                "decline to patch it and Windows ROCm import-time noise / "
                "wrong ROCM_GPU_ARCH may return."
            )


# ===========================================================================
# psutil -- cpu_freq shape the Apple Silicon M4+ unit fix relies on
# ===========================================================================


def test_psutil_cpu_freq_shape_and_wiring():
    """``patch_psutil_cpu_freq``: the wrapper rebuilds psutil's scpufreq
    namedtuple, so fail if that surface moves or the patch is never called."""
    psutil = pytest.importorskip("psutil")

    if getattr(psutil, "cpu_freq", None) is None:
        # On macOS psutil decides at runtime whether to expose cpu_freq at all (an absent one is normal on virtualised
        # Apple Silicon), so its absence is only drift off that platform.
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            pytest.skip("this Apple Silicon host exposes no psutil.cpu_freq")
        pytest.fail(
            "DRIFT DETECTED: psutil.cpu_freq is gone -- patch_psutil_cpu_freq "
            "would silently stop correcting Apple Silicon M4+ readings."
        )
    assert callable(psutil.cpu_freq)
    namedtuple_type = None
    for module_name in ("_ntuples", "_common"):
        namedtuple_type = getattr(getattr(psutil, module_name, None), "scpufreq", None)
        if namedtuple_type is not None:
            break
    if namedtuple_type is None:
        pytest.fail(
            "DRIFT DETECTED: psutil no longer exposes scpufreq in _ntuples or "
            "_common, so the M5 fallback cannot build a return value."
        )
    assert hasattr(namedtuple_type, "_replace") and namedtuple_type._fields[:3] == (
        "current",
        "min",
        "max",
    ), (
        "DRIFT DETECTED: psutil.scpufreq changed shape; the Apple Silicon "
        "rescale in patch_psutil_cpu_freq assumes (current, min, max)."
    )

    source = Path(__file__).resolve().parent.parent / "unsloth" / "_gpu_init.py"
    assert "patch_psutil_cpu_freq()" in source.read_text(encoding = "utf-8"), (
        "DRIFT DETECTED: patch_psutil_cpu_freq is defined but never called in "
        "_gpu_init.py, so real imports never install it."
    )


def _import_torchao_intmm_home():
    from unsloth.import_fixes import _TORCHAO_INTMM_MODULES
    for name in _TORCHAO_INTMM_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if callable(getattr(module, "safe_int_mm", None)):
            return module
    pytest.skip("torchao does not define safe_int_mm under any known module name")


def _torchao_intmm_original_source():
    """Once the fix has run, upstream's body is reachable only through ``__unsloth_original__``."""
    pytest.importorskip("torchao")
    intmm = _import_torchao_intmm_home()
    function = intmm.safe_int_mm
    if getattr(function, "__unsloth_patched__", False):
        function = function.__unsloth_original__
    return inspect.getsource(function)


def test_torchao_safe_int_mm_still_uses_the_repr_probe():
    """The pathology itself: a repr that formats the tensor's values, and so calls ``.item()``."""
    source = _torchao_intmm_original_source()
    if "__repr__" not in source:
        pytest.fail(
            "upstream fixed the probe, delete fix_torchao_safe_int_mm_repr_probe: "
            "torchao's safe_int_mm no longer reprs its input, so the replacement in "
            "unsloth/import_fixes.py (and its Studio copy in "
            "studio/backend/core/inference/diffusion_torchao_patches.py) is dead weight."
        )


def test_torchao_safe_int_mm_body_matches_the_verified_shape():
    from unsloth.import_fixes import _TORCHAO_SAFE_INT_MM_MARKERS

    source = _torchao_intmm_original_source()
    missing = [marker for marker in _TORCHAO_SAFE_INT_MM_MARKERS if marker not in source]
    if missing:
        pytest.fail(
            "DRIFT DETECTED: torchao's safe_int_mm body changed "
            f"({', '.join(missing)} missing); re-verify bit-identity before keeping the patch. "
            "_make_safe_int_mm copies the cuBLAS dimension guards, the contiguity fixes and the "
            "fp32 fallback, so a moved body is one the copy must not impersonate."
        )


def _patched_torchao_safe_int_mm():
    pytest.importorskip("torchao")
    intmm = _import_torchao_intmm_home()
    from unsloth.import_fixes import (
        _TORCHAO_SAFE_INT_MM_MARKERS,
        fix_torchao_safe_int_mm_repr_probe,
    )

    fix_torchao_safe_int_mm_repr_probe()
    function = intmm.safe_int_mm
    if not getattr(function, "__unsloth_patched__", False):
        source = inspect.getsource(function)
        if any(marker not in source for marker in _TORCHAO_SAFE_INT_MM_MARKERS):
            pytest.skip(
                "this torchao's safe_int_mm is not the verified body, so the fix declined to "
                "patch it (see test_torchao_safe_int_mm_body_matches_the_verified_shape)"
            )
        pytest.fail(
            "DRIFT DETECTED: fix_torchao_safe_int_mm_repr_probe left a recognised "
            "safe_int_mm unpatched."
        )
    return intmm, function


def test_torchao_intmm_patch_is_bit_identical_on_cpu():
    torch = pytest.importorskip("torch")
    intmm, patched = _patched_torchao_safe_int_mm()
    # int_scaled_matmul resolves the name through module globals, so the rebind must reach it
    assert intmm.int_scaled_matmul.__globals__["safe_int_mm"] is patched
    original = patched.__unsloth_original__

    generator = torch.Generator().manual_seed(0)

    def randint8(*shape):
        return torch.randint(-127, 127, shape, dtype = torch.int8, generator = generator)

    cases = [
        (randint8(64, 64), randint8(64, 64)),
        (randint8(40, 24), randint8(24, 72)),
        (randint8(64, 64), randint8(64, 64).t().contiguous().t()),
        (randint8(40, 20), randint8(20, 64)),
    ]
    for a, b in cases:
        assert torch.equal(patched(a, b), original(a, b)), (
            "DRIFT DETECTED: the patched safe_int_mm no longer matches torchao's on "
            f"{tuple(a.shape)} x {tuple(b.shape)}."
        )


def test_torchao_intmm_patch_is_idempotent():
    import types

    from unsloth.import_fixes import (
        _patch_torchao_intmm_module,
        fix_torchao_safe_int_mm_repr_probe,
    )

    intmm, patched = _patched_torchao_safe_int_mm()
    fix_torchao_safe_int_mm_repr_probe()
    assert intmm.safe_int_mm is patched, "DRIFT DETECTED: safe_int_mm was replaced twice."

    def already_patched(input, mat2):
        return None

    already_patched.__unsloth_patched__ = True
    stand_in = types.ModuleType("torchao_intmm_stand_in")
    stand_in.safe_int_mm = already_patched
    stand_in.out_dtype = lambda *args, **kwargs: None
    stand_in.dynamo_is_compiling = lambda: False
    assert _patch_torchao_intmm_module(stand_in) is False
    assert stand_in.safe_int_mm is already_patched


def test_torchao_intmm_patch_refuses_an_unrecognised_body():
    import types

    from unsloth.import_fixes import _patch_torchao_intmm_module

    def rewritten_upstream(input, mat2):
        # None of the markers the gate looks for
        return input @ mat2

    stand_in = types.ModuleType("torchao_intmm_stand_in")
    stand_in.safe_int_mm = rewritten_upstream
    stand_in.out_dtype = lambda *args, **kwargs: None
    stand_in.dynamo_is_compiling = lambda: False
    assert _patch_torchao_intmm_module(stand_in) is False
    assert stand_in.safe_int_mm is rewritten_upstream


def test_torchao_intmm_patch_covers_a_later_import():
    """The point of the finder: the prequant path imports torchao well after ``import unsloth``."""
    if importlib.util.find_spec("torchao") is None:
        pytest.skip("torchao not installed -- nothing to patch.")
    import subprocess

    import_fixes_path = Path(__file__).resolve().parent.parent / "unsloth" / "import_fixes.py"
    program = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('unsloth_import_fixes_under_test', {str(import_fixes_path)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert 'torchao' not in sys.modules, 'torchao was imported before the fix ran'\n"
        "module.fix_torchao_safe_int_mm_repr_probe()\n"
        "import torchao.quantization\n"
        "import importlib\n"
        "intmm = None\n"
        "for name in module._TORCHAO_INTMM_MODULES:\n"
        "    try:\n"
        "        candidate = importlib.import_module(name)\n"
        "    except ImportError:\n"
        "        continue\n"
        "    if callable(getattr(candidate, 'safe_int_mm', None)):\n"
        "        intmm = candidate\n"
        "        break\n"
        "assert intmm is not None, 'no torchao module defines safe_int_mm'\n"
        "print('PATCHED=' + str(bool(getattr(intmm.safe_int_mm, '__unsloth_patched__', False))))\n"
    )
    env = dict(os.environ)
    env.pop("UNSLOTH_TORCHAO_INT_MM_FIX", None)
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output = True,
        text = True,
        timeout = 600,
        env = env,
    )
    assert result.returncode == 0, f"child failed:\n{result.stdout}\n{result.stderr}"
    assert "PATCHED=True" in result.stdout, (
        "DRIFT DETECTED: a torchao imported AFTER fix_torchao_safe_int_mm_repr_probe ran was "
        f"left unpatched, so the meta path finder no longer fires.\n{result.stdout}"
    )


def test_torchao_intmm_finder_covers_both_module_homes(monkeypatch):
    """A finder covering only the old name loses the fix once torchao ships pytorch/ao#4718."""
    import importlib.machinery

    from unsloth.import_fixes import (
        _TORCHAO_INTMM_MODULES,
        _TorchaoIntmmLoader,
        _TorchaoIntmmPatchFinder,
    )

    assert "torchao.kernel.intmm" in _TORCHAO_INTMM_MODULES
    assert "torchao.quantization.quantize_.workflows.int8.kernels" in _TORCHAO_INTMM_MODULES

    class _Loader:
        def exec_module(self, module):
            pass

    def fake_find_spec(fullname, *args, **kwargs):
        return importlib.machinery.ModuleSpec(fullname, _Loader())

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    finder = _TorchaoIntmmPatchFinder()
    for name in _TORCHAO_INTMM_MODULES:
        spec = finder.find_spec(name)
        assert spec is not None and isinstance(spec.loader, _TorchaoIntmmLoader), name
    assert finder.find_spec("torchao.kernel.somewhere_else") is None
    assert finder.find_spec("torchao.quantization.quantize_.workflows.int8") is None


def test_torchao_intmm_installer_patches_the_new_home_when_already_imported(monkeypatch):
    """A module ALREADY in ``sys.modules`` must be patched too: the finder only sees later imports."""
    import types

    torch = pytest.importorskip("torch")
    from unsloth.import_fixes import (
        _TORCHAO_INTMM_MODULES,
        _TORCHAO_INTMM_SENTINEL,
        fix_torchao_safe_int_mm_repr_probe,
    )

    source = _torchao_intmm_original_source()
    module = types.ModuleType("torchao.quantization.quantize_.workflows.int8.kernels")
    module.__dict__["torch"] = torch
    from torch._dynamo import is_compiling as dynamo_is_compiling
    from torch._higher_order_ops.out_dtype import out_dtype

    module.out_dtype = out_dtype
    module.dynamo_is_compiling = dynamo_is_compiling
    import linecache
    import textwrap

    # The gate reads the body through inspect.getsource, so the copy needs a linecache entry.
    text = textwrap.dedent(source)
    filename = "<torchao safe_int_mm copy>"
    linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
    exec(compile(text, filename, "exec"), module.__dict__)
    assert callable(module.safe_int_mm)

    # Hide every real torchao home so only the stand-in is visible, and drop any finder.
    for name in _TORCHAO_INTMM_MODULES:
        monkeypatch.delitem(sys.modules, name, raising = False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        sys,
        "meta_path",
        [f for f in sys.meta_path if not getattr(f, _TORCHAO_INTMM_SENTINEL, False)],
    )
    monkeypatch.delenv("UNSLOTH_TORCHAO_INT_MM_FIX", raising = False)

    assert fix_torchao_safe_int_mm_repr_probe() is True
    assert getattr(
        module.safe_int_mm, "__unsloth_patched__", False
    ), "the installer ignored a safe_int_mm registered under torchao's new module name"
    a = torch.randint(-128, 127, (16, 32), dtype = torch.int8)
    b = torch.randint(-128, 127, (32, 24), dtype = torch.int8)
    assert torch.equal(module.safe_int_mm(a, b), module.safe_int_mm.__unsloth_original__(a, b))


def test_torchao_intmm_patch_wired_into_gpu_init():
    source = Path(__file__).resolve().parent.parent / "unsloth" / "_gpu_init.py"
    source = source.read_text(encoding = "utf-8")
    assert "fix_torchao_safe_int_mm_repr_probe()" in source, (
        "DRIFT DETECTED: fix_torchao_safe_int_mm_repr_probe is defined but never called in "
        "_gpu_init.py, so real imports never install it."
    )
    assert "del fix_torchao_safe_int_mm_repr_probe" in source, (
        "DRIFT DETECTED: fix_torchao_safe_int_mm_repr_probe is left bound on the unsloth "
        "namespace; _gpu_init.py deletes every fix it calls."
    )


# ===========================================================================
# transformers -- a replaced rope_scaling drops the RoPE base frequency
# ===========================================================================


def test_rope_scaling_replacement_keeps_the_base_frequency():
    """The pathology: transformers 5 moved ``rope_theta`` inside
    ``config.rope_parameters`` while keeping ``rope_scaling`` as an alias that replaces
    that whole dict, so assigning a normalized scaling dict leaves the base ``None``.
    Asked of the live build after the fix has run, so this fails whenever the fix
    stopped neutralising it, and passes on 4.57.6 where there is nothing to neutralise.
    """
    pytest.importorskip("transformers")
    from unsloth.import_fixes import (
        _rope_scaling_property_owner,
        _rope_scaling_setter_is_patched,
        _transformers_rope_scaling_assignment_drops_theta,
        fix_transformers_rope_scaling_drops_theta,
    )

    owner = _rope_scaling_property_owner()
    # Already installed by `import unsloth`; calling it again must be a no-op.
    fix_transformers_rope_scaling_drops_theta()
    assert not _transformers_rope_scaling_assignment_drops_theta(), (
        "DRIFT DETECTED: replacing config.rope_scaling still loses the RoPE base "
        "frequency, so the object-style delegation retry in models/llama.py falls back "
        "to unscaled RoPE (issue #2405)."
    )
    if owner is None:
        # transformers 4.x: rope_scaling is a plain attribute an assignment cannot
        # clobber, so there must be nothing installed.
        assert not _rope_scaling_setter_is_patched(owner), (
            "the rope_scaling setter is reported patched on a build that has no "
            "rope_scaling property to patch"
        )
    elif not _rope_scaling_setter_is_patched(owner):
        # Healthy, not drift: the fix installs nothing when the probe finds no loss, so a
        # transformers that keeps the alias and fixes the base lands here. Requiring the
        # wrapper would fail this hard gate on the release that makes it unnecessary.
        assert not _transformers_rope_scaling_assignment_drops_theta()


def test_rope_scaling_setter_patch_is_idempotent():
    """Calling the fix twice must not stack a wrapper on a wrapper."""
    pytest.importorskip("transformers")
    from unsloth.import_fixes import (
        _ROPE_SCALING_PATCH_FLAG,
        _rope_scaling_property_owner,
        _rope_scaling_setter_is_patched,
        fix_transformers_rope_scaling_drops_theta,
    )

    owner = _rope_scaling_property_owner()
    fix_transformers_rope_scaling_drops_theta()
    if owner is None:
        assert (
            _rope_scaling_property_owner() is None
        ), "the fix created a rope_scaling property on a build that had none"
        return
    if not _rope_scaling_setter_is_patched(owner):
        # Same healthy case as above: no wrapper was installed, so there is no stacking
        # to check. Idempotence of a no-op is that it stays a no-op.
        fix_transformers_rope_scaling_drops_theta()
        assert not _rope_scaling_setter_is_patched(owner)
        return
    before = owner.__dict__["rope_scaling"]
    fix_transformers_rope_scaling_drops_theta()
    after = owner.__dict__["rope_scaling"]
    assert after is before, "DRIFT DETECTED: the rope_scaling property was replaced twice."
    inner = getattr(after.fset, "__wrapped__", None)
    assert inner is not None, "the patched setter must keep the original reachable"
    assert not getattr(
        inner, _ROPE_SCALING_PATCH_FLAG, False
    ), "DRIFT DETECTED: the rope_scaling setter is wrapped twice."


def test_rope_theta_carry_only_writes_when_the_base_would_be_lost():
    """The carry helper, on every shape the parameters can arrive as.

    Cases three to five are the fix, case one is what keeps it self-neutralising on a
    transformers that keeps the base itself, and the last two are the shapes a naive
    carry would damage: a per-layer rope dict, and the Gemma local rotary, where a base
    the caller stated on purpose must survive untouched.
    """
    from types import SimpleNamespace

    from unsloth.import_fixes import _carry_rope_theta_across_assignment as carry

    # 1. The new parameters name their own base: write nothing at all.
    parameters = {"rope_type": "linear", "factor": 4.0, "rope_theta": 1000000.0}
    config = SimpleNamespace(rope_parameters = parameters)
    assert carry(config, 500000.0) == 1000000.0
    assert not hasattr(config, "rope_theta")
    assert parameters == {"rope_type": "linear", "factor": 4.0, "rope_theta": 1000000.0}

    # 2. An attribute that is already there is kept in step, never left stale.
    config = SimpleNamespace(
        rope_parameters = {"rope_type": "linear", "factor": 4.0, "rope_theta": 1000000.0},
        rope_theta = 500000.0,
    )
    assert carry(config, 500000.0) == 1000000.0
    assert config.rope_theta == 1000000.0

    # 3. The base would be lost: restore it inside rope_parameters, through a COPY, and
    #    leave the config without a top-level attribute it never had.
    parameters = {"rope_type": "linear", "factor": 4.0}
    config = SimpleNamespace(rope_parameters = parameters)
    assert carry(config, 500000.0) == 500000.0
    assert config.rope_parameters["rope_theta"] == 500000.0
    assert parameters == {"rope_type": "linear", "factor": 4.0}
    assert not hasattr(config, "rope_theta")

    # 4. Object-style replacement, #2405's own shape: no dict to write into, so the
    #    attribute is the only thing that carries the base to the retry.
    config = SimpleNamespace(rope_parameters = object())
    assert carry(config, 500000.0) == 500000.0
    assert config.rope_theta == 500000.0

    # 5. The retry: a dict again, and the base case 4 wrote lands back inside it.
    parameters = {"rope_type": "linear", "factor": 4.0}
    config = SimpleNamespace(rope_parameters = parameters, rope_theta = 500000.0)
    assert carry(config, None) == 500000.0
    assert config.rope_parameters["rope_theta"] == 500000.0
    assert parameters == {"rope_type": "linear", "factor": 4.0}

    # 6. The caller's dict is never written to: transformers 5 stores it verbatim, so
    #    one scaling dict reused across two configs would carry the first base into the
    #    second, silently wrong rather than an error.
    shared = {"rope_type": "linear", "factor": 4.0}
    first = SimpleNamespace(rope_parameters = shared)
    assert carry(first, 500000.0) == 500000.0
    assert shared == {"rope_type": "linear", "factor": 4.0}, shared
    second = SimpleNamespace(rope_parameters = shared, rope_theta = 10000.0)
    assert carry(second, None) == 10000.0
    assert second.rope_parameters["rope_theta"] == 10000.0
    assert first.rope_parameters["rope_theta"] == 500000.0

    # 7. Nothing to carry and nothing stated: untouched.
    parameters = {"rope_type": "linear", "factor": 4.0}
    config = SimpleNamespace(rope_parameters = parameters)
    assert carry(config, None) is None
    assert not hasattr(config, "rope_theta")
    assert "rope_theta" not in parameters

    # 8. Per-layer parameters: the base belongs one level down, so the top-level dict
    #    must not gain a key or transformers reads the whole thing as flat.
    parameters = {
        "full_attention": {"rope_type": "linear", "factor": 4.0},
        "sliding_attention": {"rope_type": "default"},
    }
    config = SimpleNamespace(
        rope_parameters = parameters,
        layer_types = ["full_attention", "sliding_attention"],
    )
    assert carry(config, 500000.0) == 500000.0
    assert set(parameters) == {"full_attention", "sliding_attention"}
    assert config.rope_theta == 500000.0, (
        "a per-layer dict has no global slot, so the attribute is where the base "
        "standardize_rope_params hands to each layer type has to live"
    )

    # 8. unsloth_zoo/empty_model.py's Gemma local rotary: rope_theta is set to the
    #    LOCAL base on purpose, then the scaling is replaced. Carrying the global base
    #    over it would give the local rotary the wrong base.
    parameters = {"rope_type": "default"}
    config = SimpleNamespace(rope_parameters = parameters, rope_theta = 10000.0)
    assert carry(config, 1000000.0) == 10000.0
    assert (
        config.rope_theta == 10000.0
    ), "the carry overwrote a base frequency the caller set deliberately"
    assert config.rope_parameters["rope_theta"] == 10000.0
    assert parameters == {"rope_type": "default"}


# The two shapes the carry got wrong when it first landed (#11037).

T5GEMMA2_LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention"]
T5GEMMA2_ROPE = {
    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
}
DEEPSEEK_V4_LABELS = ("main", "compress")
DEEPSEEK_V4_ROPE = {
    "main": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.125},
    "compress": {"rope_type": "default", "rope_theta": 160000.0, "partial_rotary_factor": 0.125},
}


def test_rope_theta_carry_never_puts_a_per_label_mapping_in_the_scalar_slot():
    """``_carry_per_layer_rope_theta`` returns ``None`` for the no-op too, and #11037 as merged
    read that as "fall through", carrying the mapping into the scalar ``config.rope_theta``."""
    from types import SimpleNamespace

    from unsloth.import_fixes import _carry_rope_theta_across_assignment as carry

    parameters = {k: dict(v) for k, v in T5GEMMA2_ROPE.items()}
    config = SimpleNamespace(
        rope_parameters = parameters,
        layer_types = list(T5GEMMA2_LAYER_TYPES),
    )
    carried = {"sliding_attention": 10000.0, "full_attention": 1000000.0}

    assert (
        carry(config, carried) is None
    ), "the carry reported it carried a base across an assignment that lost nothing"
    assert not hasattr(config, "rope_theta"), (
        f"a per-label MAPPING reached the scalar rope_theta slot: "
        f"{getattr(config, 'rope_theta', None)!r}. A base frequency is a number, and a "
        f"config that had no rope_theta must not gain one from a no-op assignment."
    )
    assert config.rope_parameters == {
        k: dict(v) for k, v in T5GEMMA2_ROPE.items()
    }, "the nested parameters were rewritten by an assignment that lost nothing"

    # Same refusal when the nested write is REFUSED rather than a no-op.
    stubborn = {
        "sliding_attention": {"rope_type": "default"},
        "full_attention": {"rope_type": "default"},
    }

    class _Frozen(SimpleNamespace):
        def __setattr__(self, name, value):
            if name == "rope_parameters" and getattr(self, "_locked", False):
                raise AttributeError("read-only")
            super().__setattr__(name, value)

    frozen = _Frozen(rope_parameters = stubborn, layer_types = list(T5GEMMA2_LAYER_TYPES))
    frozen._locked = True
    assert carry(frozen, carried) is None
    assert not isinstance(
        getattr(frozen, "rope_theta", None), dict
    ), "a refused nested write fell through and put the mapping in the scalar slot"


def test_rope_theta_carry_follows_rope_type_labels_not_only_layer_types():
    """``standardize_rope_params`` resolves the nesting axis as ``getattr(self,
    "_rope_type_labels", getattr(self, "layer_types", None))`` (transformers 5.17.0
    ``modeling_rope_utils.py``); on DeepseekV4 the two name different things."""
    from types import SimpleNamespace

    from unsloth.import_fixes import (
        _carry_rope_theta_across_assignment as carry,
        _rope_parameters_are_per_layer,
        _rope_theta_snapshot,
    )

    parameters = {k: dict(v) for k, v in DEEPSEEK_V4_ROPE.items()}
    config = SimpleNamespace(
        _rope_type_labels = DEEPSEEK_V4_LABELS,
        layer_types = ["heavily_compressed_attention", "compressed_sparse_attention"],
        rope_theta = 10000.0,
        rope_parameters = parameters,
    )

    assert _rope_parameters_are_per_layer(
        config, parameters
    ), "rope keyed by _rope_type_labels was read as a flat dict"
    assert _rope_theta_snapshot(config) == {
        "main": 10000.0,
        "compress": 160000.0,
    }, f"the per-label bases were not snapshotted: {_rope_theta_snapshot(config)!r}"
    carried = _rope_theta_snapshot(config)

    config.rope_parameters = {
        "main": {"rope_type": "linear", "factor": 4.0},
        "compress": {"rope_type": "default"},
    }
    carry(config, carried)

    restored = config.rope_parameters
    assert "rope_theta" not in restored, (
        f"a top-level rope_theta was written into a NESTED rope dict: {restored!r}. "
        f"transformers would hand that one global base to every label."
    )
    assert restored["compress"]["rope_theta"] == 160000.0, (
        f"the compression base was lost: {restored['compress'].get('rope_theta')!r} "
        f"(expected 160000.0). standardize_rope_params would setdefault the 10000.0 "
        f"global base into it instead."
    )
    assert restored["main"]["rope_theta"] == 10000.0
    assert restored["main"]["factor"] == 4.0, "the caller's scaling was damaged"
    assert config.rope_theta == 10000.0, "the stated global base went stale"

    # Flat replacement, same config: the two bases disagree, so the config's own base is used.
    flat = SimpleNamespace(
        _rope_type_labels = DEEPSEEK_V4_LABELS,
        rope_theta = 10000.0,
        rope_parameters = {"rope_type": "linear", "factor": 4.0},
    )
    assert carry(flat, {"main": 10000.0, "compress": 160000.0}) == 10000.0
    assert flat.rope_parameters["rope_theta"] == 10000.0


def test_rope_carry_handles_rope_labels_not_all_present_in_layer_types():
    """transformers asks ``isdisjoint``, not ``issubset``: laguna, mellum and zaya ship a rope
    dict naming a label their default ``layer_types`` omits. Under the subset test all their
    bases went to ``None`` -- the ``TypeError: ... 'NoneType' and 'Tensor'`` #11037 prevents."""
    from types import SimpleNamespace

    from unsloth.import_fixes import (
        _carry_rope_theta_across_assignment as carry,
        _rope_parameters_are_per_layer,
        _rope_theta_snapshot,
    )

    parameters = {
        "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    }
    config = SimpleNamespace(rope_parameters = parameters, layer_types = ["full_attention"])

    assert _rope_parameters_are_per_layer(config, parameters), (
        "a nested rope dict was read as flat because one of its labels is absent from "
        "layer_types; transformers asks isdisjoint, not issubset"
    )
    carried = _rope_theta_snapshot(config)
    assert carried == {
        "full_attention": 500000.0,
        "sliding_attention": 10000.0,
    }, f"the per-label bases were not snapshotted: {carried!r}"

    config.rope_parameters = {
        "full_attention": {"rope_type": "linear", "factor": 4.0},
        "sliding_attention": {"rope_type": "default"},
    }
    carry(config, carried)

    restored = config.rope_parameters
    assert (
        restored["full_attention"]["rope_theta"] == 500000.0
    ), f"the full-attention base was lost: {restored['full_attention'].get('rope_theta')!r}"
    assert restored["sliding_attention"]["rope_theta"] == 10000.0, (
        f"the sliding-attention base was lost: "
        f"{restored['sliding_attention'].get('rope_theta')!r}"
    )
    assert (
        "rope_theta" not in restored
    ), f"a top-level rope_theta was written into a NESTED rope dict: {restored!r}"
    assert not hasattr(
        config, "rope_theta"
    ), "a nested-only config gained a scalar rope_theta attribute it never had"


def test_rope_carry_on_the_real_nested_configs():
    """The stub shapes above on the real config classes, so a transformers change is caught."""
    pytest.importorskip("transformers")
    import copy as _copy

    from unsloth.import_fixes import fix_transformers_rope_scaling_drops_theta

    fix_transformers_rope_scaling_drops_theta()

    try:
        from transformers.models.t5gemma2.configuration_t5gemma2 import T5Gemma2DecoderConfig
    except Exception as exc:
        pytest.skip(f"T5Gemma2DecoderConfig unavailable: {exc!r}")

    config = T5Gemma2DecoderConfig()
    before = config.to_json_string()
    had_theta = hasattr(config, "rope_theta")
    config.rope_scaling = _copy.deepcopy(config.rope_parameters)
    assert not isinstance(
        getattr(config, "rope_theta", None), dict
    ), f"T5Gemma2 rope_theta became a mapping: {config.rope_theta!r}"
    assert (
        hasattr(config, "rope_theta") == had_theta
    ), "a no-op assignment gave the config a rope_theta attribute it never had"
    assert (
        config.to_json_string() == before
    ), "assigning a config its own rope_parameters back changed what it serializes"

    try:
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    except Exception as exc:
        pytest.skip(f"DeepseekV4Config unavailable: {exc!r}")

    deepseek = DeepseekV4Config()
    expected_compress = deepseek.rope_parameters["compress"]["rope_theta"]
    deepseek.rope_scaling = {
        "main": {"rope_type": "linear", "factor": 4.0},
        "compress": {"rope_type": "default"},
    }
    assert (
        "rope_theta" not in deepseek.rope_parameters
    ), f"stray top-level rope_theta in a nested dict: {deepseek.rope_parameters!r}"
    deepseek.standardize_rope_params()
    resolved = deepseek.rope_parameters
    assert resolved["compress"]["rope_theta"] == expected_compress, (
        f"the compression base resolved to {resolved['compress']['rope_theta']!r}, "
        f"expected {expected_compress!r}"
    )


def test_rope_carry_keeps_every_nested_base_on_every_real_config():
    """Swept, not enumerated, so a new nested-rope model is covered the day it lands. On 5.17.0:
    deepseek_v4 (``_rope_type_labels``) and laguna / mellum / zaya, all four broken by issubset."""
    pytest.importorskip("transformers")

    from unsloth.import_fixes import fix_transformers_rope_scaling_drops_theta

    fix_transformers_rope_scaling_drops_theta()

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    checked, damaged = [], {}
    # Told apart on purpose from `checked` below. A build with no `rope_parameters` at all is
    # transformers 4.x, where this whole carry does not exist and there is nothing to sweep; a
    # build that HAS the attribute but exposes no nested config is drift worth failing on.
    # Collapsing the two would either red the 4.x job forever or silently stop testing 5.x.
    saw_rope_parameters = False
    # keys(), then resolve inside the try. CONFIG_MAPPING is lazy: .items() imports every
    # config module to hand back the classes, so ONE model whose module needs an optional
    # dependency takes the whole sweep down before the loop body runs. Seen with
    # transformers.models.gemma3n, which imports timm.data.ImageNetInfo and raises ImportError
    # on a timm that does not export it. A config that cannot be built on this machine cannot
    # be the one that regressed, so it is skipped rather than allowed to end the sweep.
    for name in sorted(CONFIG_MAPPING.keys()):
        try:
            config = CONFIG_MAPPING[name]()
        except Exception:
            continue
        parameters = getattr(config, "rope_parameters", None)
        if not isinstance(parameters, dict):
            continue
        saw_rope_parameters = True
        labels = [k for k, v in parameters.items() if isinstance(v, dict)]
        if len(labels) < 2:
            continue
        expected = {k: parameters[k].get("rope_theta") for k in labels}
        if any(v is None for v in expected.values()):
            continue
        try:
            config.rope_scaling = {k: {"rope_type": "default"} for k in labels}
        except Exception:
            continue
        restored = config.rope_parameters
        actual = {k: (restored.get(k) or {}).get("rope_theta") for k in labels}
        checked.append(name)
        if actual != expected or "rope_theta" in restored:
            damaged[name] = {
                "expected": expected,
                "actual": actual,
                "stray_top_level_rope_theta": "rope_theta" in restored,
            }

    if not saw_rope_parameters:
        pytest.skip(
            "this transformers has no config.rope_parameters, so there is no per-label rope "
            "dict for a scaling replacement to damage (4.x keeps rope_scaling as a plain "
            "attribute; test_rope_scaling_replacement_keeps_the_base_frequency covers it)"
        )

    assert checked, "no config with a nested rope dict was found to check"
    assert not damaged, (
        f"a per-label scaling replacement lost bases on {sorted(damaged)} "
        f"(checked {len(checked)} nested configs): {damaged}"
    )


def test_rope_carry_leaves_the_zoo_gemma_local_base_alone_on_a_real_config():
    """``unsloth_zoo/empty_model.py`` sets Gemma's LOCAL base, then replaces the scaling."""
    pytest.importorskip("transformers")

    from unsloth.import_fixes import fix_transformers_rope_scaling_drops_theta

    fix_transformers_rope_scaling_drops_theta()

    try:
        from transformers import Gemma2Config
    except Exception as exc:
        pytest.skip(f"Gemma2Config unavailable: {exc!r}")

    config = Gemma2Config(num_hidden_layers = 2)
    config.rope_theta = 10000.0
    config.rope_scaling = {"rope_type": "default"}

    assert (
        config.rope_theta == 10000.0
    ), f"the carry overwrote the local rotary base with {config.rope_theta!r}"
    # getattr, not attribute access: the isinstance check below already says this is optional,
    # but transformers 4.x RAISES rather than returning None here, so reading it directly made
    # the tolerance unreachable and failed the test on the 4.x job.
    parameters = getattr(config, "rope_parameters", None)
    if isinstance(parameters, dict):
        assert (
            parameters.get("rope_theta") == 10000.0
        ), f"the local base did not reach rope_parameters: {parameters!r}"


def test_rope_scaling_patch_wired_into_gpu_init():
    source = Path(__file__).resolve().parent.parent / "unsloth" / "_gpu_init.py"
    source = source.read_text(encoding = "utf-8")
    assert "fix_transformers_rope_scaling_drops_theta()" in source, (
        "DRIFT DETECTED: fix_transformers_rope_scaling_drops_theta is defined but never "
        "called in _gpu_init.py, so real imports never install it."
    )
    assert "del fix_transformers_rope_scaling_drops_theta" in source, (
        "DRIFT DETECTED: fix_transformers_rope_scaling_drops_theta is left bound on the "
        "unsloth namespace; _gpu_init.py deletes every fix it calls."
    )


def test_a_reloaded_configuration_module_gets_the_new_base_class_patched():
    """A reload replaces the owner underneath the cached config the probe measures.

    `importlib.reload(transformers.configuration_utils)` re-runs the class body and
    produces a NEW, unpatched base class, while `transformers.LlamaConfig` stays in
    `sys.modules` with its old bases -- including the class we patched. The probe therefore
    reported the base frequency survives, the fix returned early, and the new base class
    stayed unpatched for every config module imported afterwards.
    """
    pytest.importorskip("transformers")
    from unsloth.import_fixes import (
        _rope_probe_inherits,
        _rope_scaling_property_owner,
    )

    owner = _rope_scaling_property_owner()
    if owner is None:
        pytest.skip("this transformers has no rope_scaling alias property to own")

    # On an ordinary build the probe measures a descendant of the live owner, so nothing
    # about the normal path changes.
    assert _rope_probe_inherits(owner) is True

    # A stand-in for the post-reload owner: a class the cached LlamaConfig does not
    # descend from. The probe's verdict cannot speak for it, so it must not veto.
    replacement = type("_ReloadedConfigBase", (object,), {})
    assert _rope_probe_inherits(replacement) is False


def test_the_reload_check_rejects_a_non_class_owner():
    """NEGATIVE CONTROL: the helper answers about classes, and anything else is 'no
    evidence' rather than an exception out of `issubclass`."""
    pytest.importorskip("transformers")
    from unsloth.import_fixes import _rope_probe_inherits

    for not_a_class in (None, object(), "PreTrainedConfig", 7):
        assert _rope_probe_inherits(not_a_class) is False


def test_rope_theta_carry_restores_each_layer_types_own_base():
    """Per-layer parameters hold one base PER LAYER TYPE, and a single scalar cannot
    describe them.

    transformers 5.5's T5Gemma2DecoderConfig starts at 10000.0 for sliding attention and
    1000000.0 for full attention. Reading `parameters["rope_theta"]` off the OUTER dict
    finds nothing, so the snapshot was None, the carry declined, and a later
    standardize_rope_params filled both nested bases with None: invalid RoPE
    initialisation with nothing raised.
    """
    from types import SimpleNamespace

    from unsloth.import_fixes import (
        _carry_rope_theta_across_assignment as carry,
        _rope_theta_snapshot,
    )

    before = SimpleNamespace(
        rope_parameters = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
        },
        layer_types = ["sliding_attention", "full_attention"],
    )
    carried = _rope_theta_snapshot(before)
    assert carried == {"sliding_attention": 10000.0, "full_attention": 1000000.0}

    # The replacement a caller assigns: per-layer scaling with no bases in it.
    replacement = {
        "sliding_attention": {"rope_type": "linear", "factor": 4.0},
        "full_attention": {"rope_type": "linear", "factor": 4.0},
    }
    config = SimpleNamespace(
        rope_parameters = replacement,
        layer_types = ["sliding_attention", "full_attention"],
    )
    assert carry(config, carried) == carried

    assert config.rope_parameters["sliding_attention"]["rope_theta"] == 10000.0
    assert config.rope_parameters["full_attention"]["rope_theta"] == 1000000.0
    # No global key: it would make standardize_rope_params read the whole dict as flat.
    assert "rope_theta" not in config.rope_parameters
    assert not hasattr(config, "rope_theta")
    # The caller's dicts are never written to, inner ones included.
    assert replacement["sliding_attention"] == {"rope_type": "linear", "factor": 4.0}
    assert replacement["full_attention"] == {"rope_type": "linear", "factor": 4.0}


def test_rope_theta_carry_leaves_a_per_layer_base_the_caller_stated():
    """NEGATIVE CONTROL: an entry that names its own base is a deliberate statement and
    must survive, exactly as the flat path leaves a stated base alone."""
    from types import SimpleNamespace

    from unsloth.import_fixes import _carry_rope_theta_across_assignment as carry

    config = SimpleNamespace(
        rope_parameters = {
            "sliding_attention": {"rope_type": "linear", "rope_theta": 50.0},
            "full_attention": {"rope_type": "linear"},
        },
        layer_types = ["sliding_attention", "full_attention"],
    )
    carry(config, {"sliding_attention": 10000.0, "full_attention": 1000000.0})

    assert config.rope_parameters["sliding_attention"]["rope_theta"] == 50.0
    assert config.rope_parameters["full_attention"]["rope_theta"] == 1000000.0


def test_rope_theta_snapshot_still_reads_a_flat_base():
    """The control that the snapshot did not change the ordinary shape: a flat dict has
    one base and the snapshot is that scalar, which is what every other case expects."""
    from types import SimpleNamespace

    from unsloth.import_fixes import _rope_theta_snapshot

    assert (
        _rope_theta_snapshot(
            SimpleNamespace(rope_parameters = {"rope_type": "linear", "rope_theta": 500000.0})
        )
        == 500000.0
    )
    assert _rope_theta_snapshot(SimpleNamespace(rope_parameters = {"rope_type": "linear"})) is None
    assert _rope_theta_snapshot(SimpleNamespace(rope_parameters = object())) is None
    assert _rope_theta_snapshot(SimpleNamespace()) is None
    # Per-layer with no bases anywhere is None, not an empty dict, so the global
    # attribute path below it still runs.
    assert (
        _rope_theta_snapshot(
            SimpleNamespace(
                rope_parameters = {"full_attention": {"rope_type": "linear"}},
                layer_types = ["full_attention"],
            )
        )
        is None
    )


def test_a_per_layer_snapshot_never_becomes_a_scalar_rope_theta():
    """Per-layer parameters replaced by a FLAT dict.

    The snapshot is a {layer_type: base} mapping and every slot below the per-layer branch
    holds a number, so passing the mapping through wrote a dict into
    `rope_parameters["rope_theta"]` and the first RoPE arithmetic on it would raise. One
    base can stand for the mapping only when every layer type agreed on it.
    """
    from types import SimpleNamespace

    from unsloth.import_fixes import _carry_rope_theta_across_assignment as carry

    # Disagreeing bases: there is no scalar that is true, so nothing is carried.
    flat = {"rope_type": "linear", "factor": 4.0}
    config = SimpleNamespace(rope_parameters = flat)
    assert carry(config, {"sliding_attention": 10000.0, "full_attention": 1000000.0}) is None
    assert config.rope_parameters == {"rope_type": "linear", "factor": 4.0}
    assert not hasattr(config, "rope_theta")

    # Agreeing bases: the one they agree on is a true answer, so it is carried as a number.
    config = SimpleNamespace(rope_parameters = {"rope_type": "linear", "factor": 4.0})
    assert carry(config, {"sliding_attention": 10000.0, "full_attention": 10000.0}) == 10000.0
    assert config.rope_parameters["rope_theta"] == 10000.0

    # And whatever is carried, it is never a dict.
    for snapshot in (
        {"a": 1.0, "b": 2.0},
        {"a": 10000.0, "b": 10000.0},
        {},
    ):
        config = SimpleNamespace(rope_parameters = {"rope_type": "linear"})
        carry(config, snapshot)
        written = config.rope_parameters.get("rope_theta", None)
        assert not isinstance(written, dict), written
        assert not isinstance(getattr(config, "rope_theta", None), dict)


# ===========================================================================
# transformers -- a submodule's prefix renaming leaks into the composite model
# ===========================================================================


def test_transformers_scopes_a_submodules_conversion_mapping():
    """``fix_transformers_composite_prefix_renaming``: transformers 5.4.0 to 5.5.4
    merge a submodule's own prefix renaming into the parent's conversion mapping
    verbatim, which renames a composite model's real weight names into names it does
    not have and throws away the bitsandbytes quant_state sidecars with them."""
    pytest.importorskip("transformers")
    from unsloth.import_fixes import _transformers_rescopes_submodule_prefix_renamings

    if not _transformers_rescopes_submodule_prefix_renamings():
        pytest.fail(
            "DRIFT DETECTED: this transformers recurses into submodules for "
            "conversion mappings without scoping them to where the submodule lives "
            "(no model_prefix argument, no PrefixChange.with_submodel_prefix, no "
            "scope_prefix field) -- fix_transformers_composite_prefix_renaming would "
            "wrap get_model_conversion_mapping. Pre-quantized multimodal checkpoints "
            "load with quant_state=None here; install transformers>=5.6.0."
        )


def test_composite_renaming_probe_agrees_with_the_real_mapping():
    """The install gate is a claim about behaviour, so check it against the behaviour.

    Builds a real composite Qwen3.5 on the meta device -- no weights, no download --
    and asks whether the mapping transformers really produces rewrites that model's
    own parameter names into names it does not have.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from unsloth.import_fixes import _transformers_rescopes_submodule_prefix_renamings

    try:
        import transformers
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import WeightRenaming
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except Exception as exc:
        pytest.skip(f"this transformers has no conversion mapping machinery: {exc!r}")
    if "qwen3_5" not in CONFIG_MAPPING:
        pytest.skip("this transformers has no qwen3_5 model type")

    config = CONFIG_MAPPING["qwen3_5"]()
    config.text_config.num_hidden_layers = 2
    config.text_config.layer_types = ["linear_attention", "full_attention"]
    if hasattr(config.text_config, "mtp_num_hidden_layers"):
        config.text_config.mtp_num_hidden_layers = 0
    config.vision_config.depth = 1
    try:
        with torch.device("meta"):
            model = transformers.AutoModelForImageTextToText.from_config(config)
    except Exception as exc:
        pytest.skip(f"cannot build a meta qwen3_5: {exc!r}")

    # Past EVERY wrapper, not just the first: unsloth_zoo patches the same function and
    # keeps its original in a closure cell, so stopping at `__wrapped__` would measure this
    # fix through this fix and report no pathology on a transformers that has one.
    mapping = get_model_conversion_mapping
    seen = set()
    while id(mapping) not in seen:
        seen.add(id(mapping))
        nxt = getattr(mapping, "__wrapped__", None)
        if nxt is None:
            for cell in getattr(mapping, "__closure__", None) or ():
                try:
                    candidate = cell.cell_contents
                except ValueError:
                    continue
                if callable(candidate) and getattr(candidate, "__name__", "") == (
                    "get_model_conversion_mapping"
                ):
                    nxt = candidate
                    break
        if nxt is None:
            break
        mapping = nxt
    keys = {name for name, _ in model.named_parameters(remove_duplicate = False)}
    keys |= {name for name, _ in model.named_buffers(remove_duplicate = False)}
    leaks = []
    for conversion in mapping(model):
        if not isinstance(conversion, WeightRenaming):
            continue
        for key in sorted(keys):
            renamed, matched = conversion.rename_source_key(key)
            if matched is not None and renamed != key and renamed not in keys:
                leaks.append((conversion.source_patterns, key, renamed))
                break

    rescopes = _transformers_rescopes_submodule_prefix_renamings()
    assert bool(leaks) != bool(rescopes), (
        f"DRIFT DETECTED: the probe says rescopes={rescopes}, but the mapping this "
        f"transformers builds for a composite Qwen3.5 {'does' if leaks else 'does not'} "
        f"rewrite the model's own weight names off the map: {leaks[:3]}"
    )


def test_composite_renaming_patch_wired_into_gpu_init():
    """The patch must be installed at startup, not only importable."""
    source = Path(__file__).resolve().parent.parent / "unsloth" / "_gpu_init.py"
    source = source.read_text(encoding = "utf-8")
    assert "fix_transformers_composite_prefix_renaming()" in source, (
        "DRIFT DETECTED: fix_transformers_composite_prefix_renaming is defined but "
        "never called in _gpu_init.py, so real imports never install it."
    )
