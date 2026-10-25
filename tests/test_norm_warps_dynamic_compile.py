# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""LayerNorm / RMSNorm under torch.compile(dynamic=True): the per-GPU warps lookup sees a symbolic width
there, which must not be hashed (TypeError: unhashable type: non-nested SymInt)."""

import pytest
import torch
from real_accelerator import has_real_cuda

from unsloth.kernels import utils as U
from unsloth.kernels.layernorm import fast_layernorm
from unsloth.kernels.rms_layernorm import fast_rms_layernorm

pytestmark = pytest.mark.skipif(
    not has_real_cuda() or torch.version.hip is not None, reason = "needs NVIDIA CUDA"
)


class _RMS(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(d) * 0.1 + 1)
        self.variance_epsilon = 1e-6


@pytest.mark.parametrize("trainable", [False, True], ids = ["frozen_w", "train_w"])
@pytest.mark.parametrize("kind", ["rms", "ln"])
@pytest.mark.parametrize("tabled", [False, True], ids = ["untabled", "tabled"])
def test_norm_compiles_with_dynamic_width(kind, trainable, tabled, monkeypatch):
    torch.manual_seed(0)
    d = 4096
    dtype = torch.float16
    if kind == "rms":
        norm = _RMS(d).cuda().to(dtype)
        fn = lambda x: fast_rms_layernorm(norm, x)
    else:
        norm = torch.nn.LayerNorm(d).cuda().to(dtype)
        fn = lambda x: fast_layernorm(norm, x)
    norm.weight.requires_grad_(trainable)
    if tabled:
        # Make the lookup hit a table entry on this GPU, so the compiled call takes the tabled warps.
        cap = tuple(torch.cuda.get_device_capability())
        from unsloth.kernels import layernorm as LN, rms_layernorm as RMS

        for mod in (LN, RMS):
            for name in ("_NUM_WARPS", "_DW_NUM_WARPS"):
                monkeypatch.setitem(getattr(mod, name), (cap, d), 4)
            mod._cached_arch_settings.cache_clear()
    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph = True, dynamic = True)
    for rows in (3, 17):
        x = torch.randn(2, rows, d, device = "cuda", dtype = dtype, requires_grad = True)
        x_ref = x.detach().clone().requires_grad_()
        out = compiled(x)
        ref = fn(x_ref)
        # Same kernel and warps as eager; dynamic=True can still move an element by an ulp elsewhere.
        torch.testing.assert_close(out, ref)
        out.float().square().sum().backward()
        ref.float().square().sum().backward()
        torch.testing.assert_close(x.grad, x_ref.grad)
