# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""unsloth/kernels/triton_launch.py launches a cached compiled Triton kernel directly. It must give
the same result as Triton's own launch for every specialization (alignment, integers equal to 1 or
divisible by 16), launch on the current stream, and step aside when a launch hook is registered."""

import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")

from unsloth.kernels import triton_launch


@triton.jit
def _axpy(x_ptr, y_ptr, out_ptr, n, scale, BLOCK: tl.constexpr, ADD: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(x_ptr + i, mask = m) * scale
    if ADD:
        v += tl.load(y_ptr + i, mask = m)
    tl.store(out_ptr + i, v, mask = m)


def _run(
    x,
    y,
    out,
    n,
    scale,
    add,
    block = 128,
):
    grid = (-(-n // block),)
    triton_launch.launch(
        _axpy,
        grid,
        (x, y, out, n, scale),
        3,
        dict(BLOCK = block, ADD = add),
        x.device.index,
        num_warps = 4,
    )
    return out


@pytest.fixture(autouse = True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(triton_launch, "_CACHE", {})
    monkeypatch.setattr(triton_launch, "_ENABLED", torch.version.hip is None)


@pytest.mark.parametrize("n", [1, 16, 17, 1000, 4096])
@pytest.mark.parametrize("offset", [0, 1, 4])
@pytest.mark.parametrize("add", [False, True])
def test_matches_reference_for_every_specialization(n, offset, add):
    base = torch.randn(n + 8, device = "cuda")
    x = base[offset : offset + n]
    y = torch.randn(n, device = "cuda")
    for scale in (1, 3, 16):
        # Reference: Triton's own launch of the same kernel (x * scale + y may be one FMA).
        ref = torch.empty(n, device = "cuda")
        _axpy[(-(-n // 128),)](x, y, ref, n, scale, BLOCK = 128, ADD = add, num_warps = 4)
        for _ in range(3):  # first call compiles, the others take the cached kernel
            out = _run(x, y, torch.empty(n, device = "cuda"), n, scale, add)
            torch.testing.assert_close(out, ref, rtol = 0, atol = 0)


def test_second_call_uses_the_cache_and_distinct_specializations_get_their_own_entry():
    if not triton_launch._ENABLED:
        pytest.skip("fast launch is CUDA only")
    x = torch.randn(64, device = "cuda")
    compiled = _axpy[(1,)](x, x, torch.empty(64, device = "cuda"), 64, 2, BLOCK = 128, ADD = False)
    if not hasattr(compiled, "packed_metadata"):
        pytest.skip(
            "this Triton returns no packed_metadata, so every launch takes its own path"
        )
    _run(x, x, torch.empty(64, device = "cuda"), 64, 2, False)
    assert len(triton_launch._CACHE) == 1
    _run(x, x, torch.empty(64, device = "cuda"), 64, 2, False)
    assert len(triton_launch._CACHE) == 1
    _run(x[1:], x[1:], torch.empty(63, device = "cuda"), 63, 1, False)
    assert len(triton_launch._CACHE) == 2


def test_runs_on_the_current_stream():
    x = torch.randn(1 << 20, device = "cuda")
    side = torch.cuda.Stream()
    for _ in range(2):
        out = torch.zeros_like(x)
        with torch.cuda.stream(side):
            torch.cuda._sleep(1_000_000)
            _run(x, x, out, x.numel(), 2, False)
        side.synchronize()
        torch.testing.assert_close(out, x * 2)


def test_registered_hook_keeps_tritons_own_launch(monkeypatch):
    knobs = getattr(triton, "knobs", None)
    if knobs is None or not hasattr(knobs.runtime.launch_enter_hook, "add"):
        pytest.skip("this Triton has no HookChain launch hooks")
    x = torch.randn(64, device = "cuda")
    _run(x, x, torch.empty(64, device = "cuda"), 64, 2, False)
    seen = []
    hook = lambda metadata: seen.append(1)
    knobs.runtime.launch_enter_hook.add(hook)
    try:
        out = _run(x, x, torch.empty(64, device = "cuda"), 64, 2, False)
    finally:
        knobs.runtime.launch_enter_hook.remove(hook)
    torch.testing.assert_close(out, x * 2)
    assert seen, "a registered launch hook must still be called"
