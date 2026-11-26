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
