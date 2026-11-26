# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FlashAdamW's fp32 variance kernel must compile on Triton 3.2 / 3.3 (torch 2.6 / 2.7).

Triton before 3.4 looks up the names a `@triton.jit` body uses in the function's module globals
only. A kernel defined inside a function that reads a function-local `tl` compiles on Triton 3.4+
but raises `CompilationError: NameError('tl is not defined')` on 3.2 / 3.3, so the first
optimizer step of a bf16, quantize=False run fails there. Checked by the name lookup itself, so
it runs on CPU and on any Triton version."""

import contextlib
import importlib.util
import pathlib

import pytest
import torch
from real_accelerator import has_real_cuda

triton = pytest.importorskip("triton")

_PATH = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "optimizers" / "flash_adamw.py"


def _load_flash_adamw():
    spec = importlib.util.spec_from_file_location("_unsloth_flash_adamw_under_test", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_kernel(module):
    module._VARIANCE_UPDATE_KERNEL = None
    device = "cuda" if has_real_cuda() else "cpu"
    var = torch.zeros(8, dtype = torch.float32, device = device)
    grad = torch.ones(8, dtype = torch.bfloat16, device = device)
    # Without a GPU the launch fails after the kernel is built; only the kernel object is checked.
    with contextlib.suppress(Exception):
        module._variance_update(var, grad, 0.999)
    kernel = module._VARIANCE_UPDATE_KERNEL
    assert kernel is not None, "_variance_update did not build its kernel"
    return kernel


def test_variance_kernel_names_resolve_from_module_globals():
    import triton.language

    kernel = _build_kernel(_load_flash_adamw())
    fn = kernel.fn
    # Closure cells are invisible to Triton < 3.4's name lookup.
    assert fn.__code__.co_freevars == (), f"kernel closes over {fn.__code__.co_freevars}"
    assert fn.__globals__.get("tl") is triton.language


@pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU")
def test_variance_kernel_matches_fp32_reference():
    module = _load_flash_adamw()
    module._VARIANCE_UPDATE_KERNEL = None
    torch.manual_seed(0)
    var = torch.rand(5000, dtype = torch.float32, device = "cuda")
    grad = torch.randn(5000, dtype = torch.bfloat16, device = "cuda")
    out = module._variance_update(var, grad, 0.999)
    expected = var * 0.999 + grad.float() * grad.float() * (1.0 - 0.999)
    torch.testing.assert_close(out, expected, rtol = 1e-6, atol = 1e-7)
