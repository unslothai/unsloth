# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Eager RMSNorm / LayerNorm launches go through triton_launch (no Triton dispatch once warm), keep
kernel[grid] under torch.compile, and stay correct under CUDA graph capture, on a side stream and on a
device that is not the current one."""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent / "_shared"))
try:
    from real_accelerator import has_real_cuda
except Exception:
    has_real_cuda = torch.cuda.is_available

if not has_real_cuda():
    pytest.skip("needs a CUDA GPU", allow_module_level = True)

import triton
from unsloth.kernels import layernorm, rms_layernorm, triton_launch

BF16 = torch.cuda.get_device_capability()[0] >= 8
DTYPE = torch.bfloat16 if BF16 else torch.float16
FAST = triton_launch._ENABLED  # off on HIP and with UNSLOTH_TRITON_FAST_LAUNCH=0
# (kind, width, gemma): 128 takes the multi-row kernels where enabled, 4096 / 1000 the one-row ones.
CASES = [
    ("rms", 128, False),
    ("rms", 128, True),
    ("rms", 4096, False),
    ("rms", 4096, True),
    ("ln", 1000, False),
    ("ln", 4096, False),
]


def _module(
    kind,
    width,
    train,
    device = "cuda",
    bias = True,
):
    g = torch.Generator().manual_seed(width)
    weight = torch.nn.Parameter(
        torch.randn(width, generator = g).to(device, DTYPE), requires_grad = train
    )
    b = torch.nn.Parameter(torch.randn(width, generator = g).to(device, DTYPE), requires_grad = train)
    return weight, (b if kind == "ln" and bias else None)


def _step(kind, gemma, x, weight, bias, dy):
    x = x.detach().requires_grad_()
    if kind == "rms":
        y = rms_layernorm.Fast_RMS_Layernorm.apply(x, weight, 1e-6, gemma)
    else:
        y = layernorm.Fast_Layernorm.apply(x, weight, bias, 1e-5)
    y.backward(dy.clone())  # non-Gemma RMSNorm writes dX over its dY
    grads = [p.grad for p in (weight, bias) if p is not None and p.requires_grad]
    return [y.detach(), x.grad] + grads


def _clear(weight, bias):
    for p in (weight, bias):
        if p is not None:
            p.grad = None


def _inputs(
    width,
    rows = 300,
    device = "cuda",
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(2, rows, width, generator = g).to(device, DTYPE),
        torch.randn(2, rows, width, generator = g).to(device, DTYPE),
    )


@pytest.mark.skipif(not FAST, reason = "fast launch disabled (HIP or UNSLOTH_TRITON_FAST_LAUNCH=0)")
@pytest.mark.parametrize("train", [False, True], ids = ["frozen", "trainable"])
@pytest.mark.parametrize("kind,width,gemma", CASES)
def test_warm_eager_call_skips_triton_dispatch(monkeypatch, kind, width, gemma, train):
    weight, bias = _module(kind, width, train)
    x, dy = _inputs(width)
    ref = _step(kind, gemma, x, weight, bias, dy)
    _clear(weight, bias)
    calls = []
    run = triton.runtime.jit.JITFunction.run
    monkeypatch.setattr(
        triton.runtime.jit.JITFunction,
        "run",
        lambda self, *a, **k: calls.append(self) or run(self, *a, **k),
    )
    got = _step(kind, gemma, x, weight, bias, dy)
    assert calls == [], [c.__name__ for c in calls]
    for a, b in zip(ref, got):
        assert torch.equal(a, b)


@pytest.mark.skipif(not getattr(torch.library, "triton_op", None), reason = "needs triton_op")
@pytest.mark.parametrize(
    "kind,width,gemma", [("rms", 128, False), ("rms", 4096, True), ("ln", 1000, False)]
)
def test_compiled_calls_never_use_fast_launch(monkeypatch, kind, width, gemma):
    calls = []
    for module in (rms_layernorm, layernorm):
        real = module.launch
        monkeypatch.setattr(
            module, "launch", lambda *a, real = real, **k: calls.append(a[0]) or real(*a, **k)
        )
    # fp32: T4 cannot trace bf16 Triton.
    weight = torch.nn.Parameter(torch.randn(width, device = "cuda"))
    bias = torch.nn.Parameter(torch.randn(width, device = "cuda"))
    norm = torch.nn.Module()
    norm.weight, norm.bias, norm.eps, norm.variance_epsilon = weight, bias, 1e-5, 1e-6
    norm.elementwise_affine = True
    if kind == "rms":
        fn = lambda x: rms_layernorm.fast_rms_layernorm(norm, x, gemma = gemma)
    else:
        fn = lambda x: layernorm.fast_layernorm(norm, x)
    torch._dynamo.reset()
    x = torch.randn(4, 64, width, device = "cuda", requires_grad = True)
    torch.compile(fn, fullgraph = True)(x).sum().backward()
    assert calls == []
    assert x.grad is not None and weight.grad is not None
    fn(x.detach().requires_grad_()).sum().backward()  # eager: same module, launch() taken
    assert calls


@pytest.mark.parametrize("train", [False, True], ids = ["frozen", "trainable"])
@pytest.mark.parametrize("kind,gemma", [("rms", False), ("rms", True), ("ln", False)])
def test_launcher_off_calls_triton_directly(monkeypatch, kind, gemma, train):
    # HIP and UNSLOTH_TRITON_FAST_LAUNCH=0: no detour through launch(), as before the fast path.
    calls = []
    for module in (rms_layernorm, layernorm):
        real = module.launch
        monkeypatch.setattr(
            module, "launch", lambda *a, real = real, **k: calls.append(a[0]) or real(*a, **k)
        )
    monkeypatch.setattr(triton_launch, "_ENABLED", False)
    weight, bias = _module(kind, 256, train)
    x, dy = _inputs(256)
    _step(kind, gemma, x, weight, bias, dy)
    assert calls == []


@pytest.mark.parametrize("train", [False, True], ids = ["frozen", "trainable"])
@pytest.mark.parametrize("kind,width,gemma", CASES)
def test_cuda_graph_capture_replays(kind, width, gemma, train):
    weight, bias = _module(kind, width, train)
    x, dy = _inputs(width, seed = 1)
    for _ in range(2):  # warm: multi-row self-check, compile, fast-launch entries
        _step(kind, gemma, x, weight, bias, dy)
        _clear(weight, bias)
    static_x, static_dy = x.clone(), dy.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _step(kind, gemma, static_x, weight, bias, static_dy)
        _clear(weight, bias)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = _step(kind, gemma, static_x, weight, bias, static_dy)
    for seed in (2, 3):
        new_x, new_dy = _inputs(width, seed = seed)
        static_x.copy_(new_x)
        static_dy.copy_(new_dy)
        graph.replay()
        torch.cuda.synchronize()
        _clear(weight, bias)
        ref = _step(kind, gemma, new_x, weight, bias, new_dy)
        _clear(weight, bias)
        for a, b in zip(ref, static_out):
            assert torch.equal(a, b)


@pytest.mark.parametrize("train", [False, True], ids = ["frozen", "trainable"])
@pytest.mark.parametrize("kind,width,gemma", CASES)
def test_launches_follow_the_current_stream(kind, width, gemma, train):
    weight, bias = _module(kind, width, train)
    x, dy = _inputs(width, seed = 4)
    ref = _step(kind, gemma, x, weight, bias, dy)
    _clear(weight, bias)
    _step(kind, gemma, x, weight, bias, dy)
    _clear(weight, bias)
    side = torch.cuda.Stream()
    late_x = torch.zeros_like(x)
    torch.cuda.synchronize()
    with torch.cuda.stream(side):
        # A launch on any other stream would read late_x before the copy lands.
        torch.cuda._sleep(200_000_000)
        late_x.copy_(x)
        got = _step(kind, gemma, late_x, weight, bias, dy)
    side.synchronize()
    for a, b in zip(ref, got):
        assert torch.equal(a, b)


@pytest.mark.skipif(not has_real_cuda() or torch.cuda.device_count() < 2, reason = "needs two GPUs")
@pytest.mark.parametrize("train", [False, True], ids = ["frozen", "trainable"])
@pytest.mark.parametrize("kind,width,gemma", CASES)
def test_second_device_with_first_current(kind, width, gemma, train):
    weight, bias = _module(kind, width, train, device = "cuda:1")
    x, dy = _inputs(width, device = "cuda:1", seed = 5)
    with torch.cuda.device(1):
        ref = [t.clone() for t in _step(kind, gemma, x, weight, bias, dy)]
    _clear(weight, bias)
    for _ in range(2):  # the second call is the warm fast-launch one
        with torch.cuda.device(0):
            got = _step(kind, gemma, x, weight, bias, dy)
        _clear(weight, bias)
        torch.cuda.synchronize(1)
        for a, b in zip(ref, got):
            assert a.device == b.device and torch.equal(a, b)
