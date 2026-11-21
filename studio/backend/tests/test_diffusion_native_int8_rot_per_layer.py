# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Each rotated int8 Linear owns its ``rot_h``; a shared float32 alias broke group offload ("mat2 is on cpu")."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_native_quant as nq  # noqa: E402
from core.inference.diffusion_convrot import build_convrot_hadamard  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("from_codes", [False, True])
def test_each_layer_owns_its_rotation(dtype, from_codes):
    cls = nq.native_linear_class()
    torch.manual_seed(0)
    layers = []
    for _ in range(3):
        lin = torch.nn.Linear(512, 64).to(dtype)
        if from_codes:
            codes = torch.randint(-127, 128, (64, 512), dtype = torch.int8)
            scale = torch.rand(64) + 0.01
            layers.append(cls(lin, "int8", act_int8 = True, rot_group = 256, codes = codes, scale = scale))
        else:
            layers.append(cls(lin, "int8", act_int8 = True, rot_group = 256))
    cached = build_convrot_hadamard(256, device = "cpu", dtype = torch.float32)
    ptrs = {layer.rot_h.data_ptr() for layer in layers}
    assert len(ptrs) == len(layers) and cached.data_ptr() not in ptrs
    for layer in layers:
        assert layer.rot_h.dtype == layer.compute_dtype
        assert torch.equal(layer.rot_h.float(), cached.to(layer.compute_dtype).float())
        assert "rot_h" not in layer.state_dict()


def test_repointing_one_layers_buffer_leaves_the_others_alone():
    cls = nq.native_linear_class()
    a = cls(torch.nn.Linear(512, 64), "int8", act_int8 = True, rot_group = 256)
    b = cls(torch.nn.Linear(512, 64), "int8", act_int8 = True, rot_group = 256)
    before = b.rot_h.data_ptr()
    a.rot_h.data = torch.zeros_like(a.rot_h)
    assert b.rot_h.data_ptr() == before and b.rot_h.abs().sum() > 0
    x = torch.randn(4, 512)
    with torch.no_grad():
        ref = torch.nn.functional.linear(x, b.dequantized_weight(torch.float32), b.bias)
        got = b(x)
    assert ((got - ref).norm() / ref.norm()).item() < 0.05


def test_the_cached_hadamard_is_never_handed_out_at_float32():
    cached = build_convrot_hadamard(256, device = "cpu", dtype = torch.float32)
    assert cached.to(torch.float32) is cached
    own = nq._own_rotation(cached, torch.float32)
    assert own is not cached and own.data_ptr() != cached.data_ptr() and torch.equal(own, cached)
