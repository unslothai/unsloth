# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The per-width launch settings cache must not hash a symbolic width: a traced norm op's fake run
under dynamic=True passes one, outside Dynamo (is_compiling() is False there)."""

import pytest
import torch

pytest.importorskip("triton")

from torch._dynamo.source import ConstantSource
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from unsloth.kernels import layernorm, rms_layernorm
from unsloth.kernels import utils as kernel_utils
from unsloth.kernels.utils import calculate_settings


def _symbolic(n):
    env = ShapeEnv()
    return env.create_symintnode(env.create_symbol(n, ConstantSource("n_cols")), hint = n)


@pytest.mark.parametrize("module", [rms_layernorm, layernorm], ids = ["rms", "layernorm"])
@pytest.mark.parametrize("n", [64, 4096])
def test_a_symbolic_width_skips_the_cache(module, n):
    width = _symbolic(n)
    with pytest.raises(TypeError):
        hash(width)
    assert tuple(module._settings(width)) == tuple(calculate_settings(n))


@pytest.mark.parametrize("module", [rms_layernorm, layernorm], ids = ["rms", "layernorm"])
@pytest.mark.parametrize("dw", [False, True], ids = ["fwd", "dw"])
def test_a_symbolic_width_reads_the_cuda_warps_table(module, dw, monkeypatch):
    # On CUDA the warps come from a (capability, width) table, which must not hash the width either.
    table = module._DW_NUM_WARPS if dw else module._NUM_WARPS
    (capability, n), warps = next(iter(table.items()))
    monkeypatch.setattr(kernel_utils, "_NORM_ARCH_WARPS", True)
    monkeypatch.setattr(kernel_utils, "_cuda_capability", lambda index: capability)
    monkeypatch.setattr(torch.version, "hip", None)
    device = torch.device("cuda", 0)
    expected = module._arch_settings(n, device, dw)
    assert expected == (calculate_settings(n)[0], warps)
    assert tuple(module._settings(_symbolic(n), device, dw)) == expected
    # A width the table does not list keeps calculate_settings' warps.
    assert tuple(module._settings(_symbolic(n + 1), device, dw)) == tuple(calculate_settings(n + 1))


@pytest.mark.parametrize("module", [rms_layernorm, layernorm], ids = ["rms", "layernorm"])
def test_an_int_width_still_uses_the_cache(module):
    module._cached_arch_settings.cache_clear()
    for _ in range(3):
        assert module._settings(256) == calculate_settings(256)
    info = module._cached_arch_settings.cache_info()
    assert info.misses == 1 and info.hits == 2
