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


# ===========================================================================
# torchao -- the safe_int_mm __repr__ probe that syncs the device
# ===========================================================================


def _import_torchao_intmm_home():
    """The module that defines ``safe_int_mm`` in the installed torchao.

    Releases up to 0.18.0 keep it in ``torchao.kernel.intmm``; main moved it to the
    int8 workflow. The fix covers both, so the tests follow whichever one exists.
    """
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
    """Source of the ``safe_int_mm`` this process would run unpatched.

    Importing unsloth installs the fix, so once it has run the function bound
    on the module is ours; the body upstream ships is then reachable only
    through ``__unsloth_original__``.
    """
    pytest.importorskip("torchao")
    intmm = _import_torchao_intmm_home()
    function = intmm.safe_int_mm
    if getattr(function, "__unsloth_patched__", False):
        function = function.__unsloth_original__
    return inspect.getsource(function)


def test_torchao_safe_int_mm_still_uses_the_repr_probe():
    """``fix_torchao_safe_int_mm_repr_probe``: the pathology itself.

    torchao decides "am I being traced?" with ``"FakeTensor" in
    input.__repr__()``, and a real CUDA tensor's repr formats its values, so
    it calls ``.item()``: a device sync per eager int8 linear and an outright
    failure under ``torch.cuda.graph`` capture.
    """
    source = _torchao_intmm_original_source()
    if "__repr__" not in source:
        pytest.fail(
            "upstream fixed the probe, delete fix_torchao_safe_int_mm_repr_probe: "
            "torchao's safe_int_mm no longer reprs its input, so the replacement in "
            "unsloth/import_fixes.py (and its Studio copy in "
            "studio/backend/core/inference/diffusion_torchao_patches.py) is dead weight."
        )


def test_torchao_safe_int_mm_body_matches_the_verified_shape():
    """The replacement is a FULL copy of torchao 0.17.0's body, so it may only
    stand in for a body that still matches it. Every landmark that was checked
    when bit-identity was measured must still be there."""
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
    """Install the fix and hand back torchao's now-patched ``safe_int_mm``."""
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
    """The whole argument for shipping a copied body: same inputs, same bytes out.

    The shapes cover both cuBLAS guards (a good j and k, then a j that is not a nonzero
    multiple of 8) and the contiguity fix on mat2, which torchao needs because cuBLAS
    silently returns a wrong answer without it.
    """
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
    """Calling the fix twice, or letting Studio's copy of it run in the same process, must
    not stack replacements: both mark the function ``__unsloth_patched__`` and the other
    recognises it."""
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
    """A torchao whose GEMM has been rewritten is left alone rather than impersonated."""
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
    """The point of the meta path finder: the int8 prequant path only ``torch.load``s torchao
    subclasses, so nothing imports torchao until well after ``import unsloth``.

    Runs in a child that has no torchao imported yet, loads import_fixes.py by path (so the
    fix is exercised without ``import unsloth`` importing torchao first) and checks that
    ``import torchao.quantization`` comes out patched.
    """
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
    """``_TorchaoIntmmPatchFinder``: torchao main moved ``safe_int_mm`` from
    ``torchao.kernel.intmm`` to ``torchao.quantization.quantize_.workflows.int8.kernels``
    (pytorch/ao#4718) with the probe intact. The finder must wrap the loader for both
    names and stay silent for anything else, or the next torchao release loses the fix."""
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
    """``fix_torchao_safe_int_mm_repr_probe`` patches a module that is ALREADY in
    ``sys.modules`` under the new name, since the finder only sees later imports."""
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

    # Isolate: hide every real torchao home so only the stand-in is visible, and drop any finder.
    for name in _TORCHAO_INTMM_MODULES:
        monkeypatch.delitem(sys.modules, name, raising = False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        sys, "meta_path", [f for f in sys.meta_path if not getattr(f, _TORCHAO_INTMM_SENTINEL, False)]
    )
    monkeypatch.delenv("UNSLOTH_TORCHAO_INT_MM_FIX", raising = False)

    assert fix_torchao_safe_int_mm_repr_probe() is True
    assert getattr(module.safe_int_mm, "__unsloth_patched__", False), (
        "the installer ignored a safe_int_mm registered under torchao's new module name"
    )
    a = torch.randint(-128, 127, (16, 32), dtype = torch.int8)
    b = torch.randint(-128, 127, (32, 24), dtype = torch.int8)
    assert torch.equal(module.safe_int_mm(a, b), module.safe_int_mm.__unsloth_original__(a, b))

def test_torchao_intmm_patch_wired_into_gpu_init():
    """The patch must be installed at startup, not only importable."""
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
