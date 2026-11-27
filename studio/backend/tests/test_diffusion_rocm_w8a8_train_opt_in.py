# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only: UNSLOTH_DIFFUSION_ROCM_W8A8_TRAIN lets base_precision='int8' train the W8A8 base on ROCm, nothing else."""

from __future__ import annotations

import pytest

import core.training.diffusion_train_common as common
from core.training import diffusion_block_swap as swap
from core.training import diffusion_dit_trainer as dit
from core.training.diffusion_train_common import DiffusionLoraConfig

_ROCM_MSG = "ROCm/AMD GPU"


def _cfg(**kw) -> DiffusionLoraConfig:
    return DiffusionLoraConfig(base_model = "Qwen/Qwen-Image", data_dir = "d", output_dir = "o", **kw)


@pytest.fixture
def rocm(monkeypatch):
    for mod in (common, dit):
        monkeypatch.setattr(mod, "torch_is_rocm", lambda: True)
    monkeypatch.setattr(dit, "has_functional_torchao", lambda: False)
    monkeypatch.delenv(swap.ROCM_W8A8_TRAIN_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_TRAIN", raising = False)
    return monkeypatch


def _resolve(mode):
    cfg = _cfg(base_precision = mode)
    return dit._resolve_base_precision(cfg, None, "cuda")


@pytest.mark.parametrize("mode", ["int8", "fp8", "mxfp8"])
def test_rocm_refuses_without_the_opt_in(rocm, mode):
    with pytest.raises(ValueError, match = _ROCM_MSG):
        _resolve(mode)


def test_opt_in_trains_int8_w8a8_on_rocm(rocm):
    rocm.setenv(swap.ROCM_W8A8_TRAIN_ENV, "1")
    assert _resolve("int8") == "int8"


@pytest.mark.parametrize("mode", ["fp8", "mxfp8"])
def test_opt_in_leaves_fp8_refused(rocm, mode):
    rocm.setenv(swap.ROCM_W8A8_TRAIN_ENV, "1")
    with pytest.raises(ValueError, match = _ROCM_MSG):
        _resolve(mode)


def test_opt_in_never_lifts_weight_only_int8(rocm):
    rocm.setenv(swap.ROCM_W8A8_TRAIN_ENV, "1")
    rocm.setenv("UNSLOTH_DIFFUSION_INT8_TRAIN", "weight-only")
    with pytest.raises(ValueError, match = _ROCM_MSG):
        _resolve("int8")


@pytest.mark.parametrize("value, expected", [("1", True), ("on", True), ("0", False), ("", False)])
def test_opt_in_env_values(monkeypatch, value, expected):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_TRAIN", raising = False)
    monkeypatch.setenv(swap.ROCM_W8A8_TRAIN_ENV, value)
    assert swap.rocm_w8a8_train_opt_in("int8") is expected
    assert swap.rocm_w8a8_train_opt_in("bf16") is False


@pytest.fixture
def rocm_preflight(rocm):
    import torch

    for name in ("bf16_unsupported_reason", "dit_accelerator_missing_reason"):
        rocm.setattr(common, name, lambda *a, **k: None)
    rocm.setattr(common, "dit_fp16_fallback", lambda fam: False)
    rocm.setattr(common, "has_functional_torchao", lambda: False)
    rocm.setattr(torch.cuda, "is_available", lambda: True)
    rocm.setenv(swap.ROCM_W8A8_TRAIN_ENV, "1")
    return rocm


def _resume_kernel(monkeypatch, kernel):
    from core.training import diffusion_checkpoint as ckpt
    monkeypatch.setattr(ckpt, "recorded_int8_resume_kernel", lambda path: kernel)


def _both_guards(cfg):
    cfg.resolved_family = "qwen-image"
    pre = common.training_precision_preflight_error(
        cfg.resolved_family, cfg.base_precision, cfg = cfg
    )
    try:
        child = dit._resolve_base_precision(cfg, None, "cuda")
    except ValueError as e:
        child = e
    return pre, child


def test_preflight_refuses_a_weight_only_int8_resume_like_the_child(rocm_preflight):
    _resume_kernel(rocm_preflight, "weight-only")
    pre, child = _both_guards(_cfg(base_precision = "int8", resume_from_checkpoint = "ckpt"))
    assert pre is not None and _ROCM_MSG in pre
    assert isinstance(child, ValueError) and _ROCM_MSG in str(child)


def test_preflight_accepts_a_w8a8_resume_like_the_child(rocm_preflight):
    _resume_kernel(rocm_preflight, "w8a8")
    pre, child = _both_guards(_cfg(base_precision = "int8", resume_from_checkpoint = "ckpt"))
    assert pre is None and child == "int8"


def test_preflight_fresh_int8_run_unchanged(rocm_preflight):
    _resume_kernel(rocm_preflight, "weight-only")
    pre, child = _both_guards(_cfg(base_precision = "int8"))
    assert pre is None and child == "int8"
    assert common.training_precision_preflight_error("qwen-image", "int8") is None


def test_route_passes_the_resume_config_to_the_preflight():
    import ast
    import inspect

    import routes.training as tr

    tree = ast.parse(inspect.getsource(tr))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "training_precision_preflight_error"
    ]
    assert calls and all(any(k.arg == "cfg" for k in c.keywords) for c in calls)


@pytest.fixture
def int8_build(rocm):
    calls = []
    rocm.setattr(dit, "_int8_quantize_base", lambda *a, **k: calls.append(a))
    rocm.setenv(swap.ROCM_W8A8_TRAIN_ENV, "1")

    def run(converted, **kw):
        rocm.setattr(dit, "_apply_int8_w8a8", lambda *a, **k: converted)
        return dit._build_int8_base(
            object(), None, _cfg(base_precision = "int8"), device = "cuda", **kw
        )

    return run, calls


def test_rocm_opt_in_w8a8_conversion_failure_is_fatal(int8_build):
    run, calls = int8_build
    with pytest.raises(ValueError, match = "could not be built on this ROCm GPU"):
        run(False)
    assert calls == []


def test_rocm_opt_in_w8a8_conversion_success_trains_w8a8(int8_build):
    run, calls = int8_build
    assert run(True) is True and calls == []


def test_cuda_w8a8_failure_still_falls_back_to_weight_only(int8_build, monkeypatch):
    monkeypatch.setattr(dit, "torch_is_rocm", lambda: False)
    run, calls = int8_build
    assert run(False) is False and len(calls) == 1
    with pytest.raises(ValueError, match = "could not be built on this GPU"):
        run(False, fp16_amp = True)
