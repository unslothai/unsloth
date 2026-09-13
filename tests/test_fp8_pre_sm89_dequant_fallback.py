# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The block dequant fallback must divert on exactly the GPUs triton cannot compile fp8e4nv for.

triton adds fp8e4nv (torch.float8_e4m3fn) to its supported dtypes at compute capability >= 89
(third_party/nvidia/backend/compiler.py), so sm80 has to take the torch path or it raises
instead of falling back. sm89 -- 4090, L40S, L4 -- is on the supported side of that line, and a
major-only "< 9" test silently drops all of Ada onto a path that costs several times the memory.

The capability is monkeypatched so the routing is pinned on any CUDA GPU, not just on the cards
being simulated.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")

BLOCK = [128, 128]


def _make(
    dtype = torch.float8_e4m3fn,
    m = 256,
    n = 256,
):
    torch.manual_seed(0)
    weight = (torch.randn(m, n, device = "cuda") * 0.4).to(dtype)
    scale = torch.rand(m // BLOCK[0], n // BLOCK[1], device = "cuda", dtype = torch.float32) + 0.5
    return weight, scale


def _route(
    monkeypatch,
    capability,
    weight,
    scale,
    hip = None,
):
    """Return ("triton"|"torch", output) for a simulated device."""
    from unsloth.kernels import fp8

    calls = []
    original = fp8.weight_dequant_block
    monkeypatch.setattr(
        fp8, "weight_dequant_block", lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(torch.version, "hip", hip)
    out = fp8._blockwise_weight_dequant_any_shape(weight, scale, BLOCK, torch.bfloat16)
    return ("triton" if calls else "torch"), out


@pytest.mark.parametrize(
    "capability, expected",
    [
        ((7, 0), "torch"),  # V100
        ((8, 0), "torch"),  # A100, the reported failure
        ((8, 6), "torch"),  # A10 / 3090
        ((8, 7), "torch"),  # Jetson Orin
        ((8, 9), "triton"),  # 4090 / L40S / L4: fp8e4nv IS supported here
        ((9, 0), "triton"),  # H100
        ((10, 0), "triton"),  # B200
        ((12, 0), "triton"),  # RTX 5090
    ],
)
def test_divert_boundary_is_sm89_not_sm90(monkeypatch, capability, expected):
    weight, scale = _make()
    assert _route(monkeypatch, capability, weight, scale)[0] == expected


@pytest.mark.parametrize("capability", [(7, 0), (8, 0), (8, 9), (9, 0)])
def test_float8_e5m2_is_never_diverted(monkeypatch, capability):
    # fp8e5 compiles on every arch triton supports, so the guard must not touch it.
    weight, scale = _make(dtype = torch.float8_e5m2)
    assert _route(monkeypatch, capability, weight, scale)[0] == "triton"


@pytest.mark.parametrize("capability", [(8, 0), (9, 0), (11, 5)])
def test_rocm_is_never_diverted(monkeypatch, capability):
    # get_device_capability is gfx-derived on ROCm, not an SM number, and AMD's triton backend
    # lists fp8e4nv unconditionally, so the NVIDIA-only guard must not fire there.
    weight, scale = _make()
    assert _route(monkeypatch, capability, weight, scale, hip = "6.2.0")[0] == "triton"


@pytest.mark.parametrize(
    "shape, block",
    [
        ((256, 256), [128, 128]),
        ((1024, 2048), [128, 128]),
        ((256, 256), [64, 64]),
        ((300, 256), [128, 128]),
        ((256, 300), [128, 128]),
        ((1, 128), [128, 128]),
    ],
)
def test_fallback_matches_the_triton_kernel_bit_for_bit(monkeypatch, shape, block):
    # Diverting must not change a single value, in either direction.
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("needs a GPU that can run the triton fp8e4nv path to compare against")
    m, n = shape
    torch.manual_seed(0)
    weight = (torch.randn(m, n, device = "cuda") * 0.4).to(torch.float8_e4m3fn)
    scale = (
        torch.rand(-(-m // block[0]), -(-n // block[1]), device = "cuda", dtype = torch.float32) + 0.5
    )

    from unsloth.kernels import fp8

    monkeypatch.setattr(fp8, "_DEQUANT_CHUNK_ELEMS", 4096, raising = False)  # force multi-chunk

    with monkeypatch.context() as mp:
        mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
        via_triton = fp8._blockwise_weight_dequant_any_shape(weight, scale, block, torch.bfloat16)
    with monkeypatch.context() as mp:
        mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
        via_torch = fp8._blockwise_weight_dequant_any_shape(weight, scale, block, torch.bfloat16)

    assert torch.equal(via_triton, via_torch)


def test_fallback_does_not_materialize_a_full_size_scale(monkeypatch):
    # The whole-weight expansion needed two m*n float32 temporaries; the chunked one must stay
    # well under that or a 40GB A100 trades a CompilationError for an OOM.
    from unsloth.kernels import fp8

    m, n = 4096, 8192
    weight = (torch.randn(m, n, device = "cuda") * 0.3).to(torch.float8_e4m3fn)
    scale = torch.rand(m // 128, n // 128, device = "cuda", dtype = torch.float32) + 0.5

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    monkeypatch.setattr(fp8, "_DEQUANT_CHUNK_ELEMS", 1 << 20)  # keep the test tensor small
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = fp8._blockwise_weight_dequant_any_shape(weight, scale, BLOCK, torch.bfloat16)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before

    assert out.shape == (m, n)
    # Two m*n float32 tensors plus the output is 10+ bytes/element; chunking holds it near the
    # 2-byte output itself.
    assert peak < 5 * m * n, f"fallback peaked at {peak / m / n:.1f} bytes/element"
