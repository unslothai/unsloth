# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""A torchao stub (unsloth_zoo's Windows ROCm sentinel classes) is not a blockwise FP8 GEMM."""

import sys
import types

import pytest

F8 = pytest.importorskip("unsloth.kernels.fp8")


def _install(monkeypatch, attr):
    for name in (
        "torchao",
        "torchao.prototype",
        "torchao.prototype.blockwise_fp8_inference",
        "torchao.prototype.blockwise_fp8_inference.blockwise_quantization",
        "torchao.kernel",
        "torchao.kernel.blockwise_quantization",
    ):
        mod = types.ModuleType(name)
        mod.__path__ = []
        if name.endswith("blockwise_quantization"):
            mod.blockwise_fp8_gemm = attr
        monkeypatch.setitem(sys.modules, name, mod)


def test_a_sentinel_class_is_not_taken_for_the_kernel(monkeypatch):
    sentinel = type("blockwise_fp8_gemm", (), {})
    _install(monkeypatch, sentinel)
    assert F8._import_torchao_blockwise_gemm() is None


def test_the_real_function_is_still_found(monkeypatch):
    def blockwise_fp8_gemm(
        a,
        a_s,
        b,
        b_s,
        block_size = 128,
    ):
        return a

    _install(monkeypatch, blockwise_fp8_gemm)
    assert F8._import_torchao_blockwise_gemm() is blockwise_fp8_gemm


def test_the_zoo_sentinel_itself_is_refused(monkeypatch):
    utils = pytest.importorskip("unsloth_zoo.temporary_patches.utils")
    if not hasattr(utils, "_rocm_make_sentinel"):
        pytest.skip("unsloth_zoo without the Windows ROCm torchao stubs")
    _install(
        monkeypatch,
        utils._rocm_make_sentinel("blockwise_fp8_gemm", "torchao.kernel.blockwise_quantization"),
    )
    assert F8._import_torchao_blockwise_gemm() is None
