# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Pre-sm89 block FP8 dequant decodes e4m3fn bytes in triton; it must match the torch fallback bit for bit."""

import os
import subprocess
import sys
import types

import pytest
import torch

from unsloth.kernels import fp8

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason = "needs CUDA")


@pytest.fixture
def pre_sm89(monkeypatch):
    # Real pre-sm89 cards take this path anyway; newer ones are told they are an A100.
    fp8._fp8_device_lacks_kernel.cache_clear()
    fp8._rocm_bytes_dequant_arch.cache_clear()
    if CUDA and torch.version.hip is None and torch.cuda.get_device_capability() >= (8, 9):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    yield
    fp8._fp8_device_lacks_kernel.cache_clear()
    fp8._rocm_bytes_dequant_arch.cache_clear()


def _weight(
    m,
    n,
    device,
    seed = 0,
):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    # Every byte value, incl. subnormals, -0.0 and both NaNs, then a realistic spread.
    raw = torch.randint(0, 256, (m, n), generator = g, dtype = torch.uint8)
    raw.view(-1)[: min(256, m * n)] = torch.arange(min(256, m * n), dtype = torch.uint8)
    return raw.view(torch.float8_e4m3fn).to(device)


def _scale(
    m,
    n,
    block,
    dtype,
    device,
    seed = 1,
):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    shape = (-(-m // block[0]), -(-n // block[1]))
    s = torch.rand(shape, generator = g, dtype = torch.float32) * 1e-2 + 1e-4
    return s.to(dtype).to(device)


def _same_bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    nan = torch.isnan(a)
    assert torch.equal(nan, torch.isnan(b))
    # NaN payloads may differ between the two casts; everything else (incl. -0.0) must match exactly.
    ai = a.masked_fill(nan, 0).contiguous().view(-1).view(torch.uint8)
    bi = b.masked_fill(nan, 0).contiguous().view(-1).view(torch.uint8)
    assert torch.equal(ai, bi)


SHAPES = [
    (128, 128),
    (256, 384),
    (384, 256),
    (128, 384),
    (1, 128),
    (130, 200),
    (576, 7168 // 4),
    (300, 129),
    (17, 3),
]


@needs_cuda
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("block", [[128, 128], [64, 128], [128, 64]])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_bytes_kernel_matches_torch_fallback(pre_sm89, shape, block, out_dtype):
    m, n = shape
    w = _weight(m, n, "cuda")
    s = _scale(m, n, block, torch.float32, "cuda")
    ref = fp8._torch_blockwise_dequant(w, s, block, out_dtype)
    out = fp8._blockwise_weight_dequant_any_shape(w, s, block, out_dtype)
    _same_bits(out, ref)


@needs_cuda
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.bfloat16, torch.float16, "e8m0"])
def test_scale_dtypes(pre_sm89, scale_dtype):
    m, n, block = 384, 256, [128, 128]
    w = _weight(m, n, "cuda")
    if scale_dtype == "e8m0":
        if not hasattr(torch, "float8_e8m0fnu"):
            pytest.skip("no float8_e8m0fnu in this torch")
        s = (2.0 ** torch.randint(-12, 2, (3, 2))).to("cuda").to(torch.float8_e8m0fnu)
        ref_scale = s.to(torch.float32)
    else:
        s = _scale(m, n, block, scale_dtype, "cuda")
        ref_scale = s
    ref = fp8._torch_blockwise_dequant(w, ref_scale, block, torch.bfloat16)
    _same_bits(fp8._blockwise_weight_dequant_any_shape(w, s, block, torch.bfloat16), ref)


@needs_cuda
@pytest.mark.parametrize("shape", [(256, 384), (130, 200), (384, 384)])
def test_transposed_and_strided(pre_sm89, shape):
    m, n = shape
    block = [128, 128]
    w = _weight(m, n, "cuda")
    s = _scale(m, n, block, torch.float32, "cuda")
    ref = fp8._torch_blockwise_dequant(w, s, block, torch.float16)
    # weight_dequant on a transposed view (fast_lora backward's W.t()).
    _same_bits(fp8.weight_dequant(w.t(), s, torch.float16), ref.t())
    # Transposed weight with transposed scale grid (FP8_fbgemm_block_linear's re-orientation).
    wt, st = w.t(), s.t()
    _same_bits(
        fp8._blockwise_weight_dequant_any_shape(wt, st, block, torch.float16),
        fp8._torch_blockwise_dequant(wt, st, block, torch.float16),
    )
    wide = _weight(m, n + 128, "cuda")[:, :n]
    _same_bits(
        fp8._blockwise_weight_dequant_any_shape(wide, s, block, torch.bfloat16),
        fp8._torch_blockwise_dequant(wide, s, block, torch.bfloat16),
    )


@needs_cuda
@pytest.mark.parametrize("shape", [(14336, 4096), (4096, 14336)])
def test_real_shapes_and_peak_memory(pre_sm89, shape):
    m, n = shape
    w = (torch.randn(m, n, device = "cuda") * 0.3).to(torch.float8_e4m3fn)
    s = _scale(m, n, [128, 128], torch.float32, "cuda")
    ref = fp8._torch_blockwise_dequant(w, s, [128, 128], torch.bfloat16)
    del_ref_bytes = ref.numel() * ref.element_size()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = fp8._blockwise_weight_dequant_any_shape(w, s, [128, 128], torch.bfloat16)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    _same_bits(out, ref)
    # Only the output is allocated (the chunked fallback adds float32 scratch on top).
    assert peak <= del_ref_bytes + (1 << 20), peak


def _no_props(*a, **k):
    raise RuntimeError("no device properties")


def _router(monkeypatch):
    calls = []
    orig = fp8._e4m3fn_bytes_dequant
    monkeypatch.setattr(
        fp8, "_e4m3fn_bytes_dequant", lambda *a, **k: calls.append(1) or orig(*a, **k)
    )
    monkeypatch.setattr(
        fp8,
        "weight_dequant_block",
        lambda x, s, block_size = 128, dtype = torch.bfloat16: calls.append(2)
        or torch.empty(x.shape, dtype = dtype, device = x.device),
    )
    w = _weight(256, 256, "cuda")

    def route(
        capability,
        hip = None,
        weight = w,
        arch = "gfx90a",
        triton_version = "3.2.0",
        block = (128, 128),
        out_dtype = torch.bfloat16,
    ):
        calls.clear()
        s = _scale(*weight.shape, list(block), torch.float32, "cuda")
        with monkeypatch.context() as mp:
            mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
            mp.setattr(torch.version, "hip", hip)
            if hip is not None:
                props = types.SimpleNamespace(gcnArchName = arch)
                mp.setattr(torch.cuda, "get_device_properties", lambda *a, **k: props)
                if arch is None:
                    mp.setattr(torch.cuda, "get_device_properties", _no_props)
                mp.setattr(fp8.triton, "__version__", triton_version)
            fp8._fp8_device_lacks_kernel.cache_clear()
            fp8._rocm_bytes_dequant_arch.cache_clear()
            fp8._blockwise_weight_dequant_any_shape(weight, s, list(block), out_dtype)
        fp8._fp8_device_lacks_kernel.cache_clear()
        fp8._rocm_bytes_dequant_arch.cache_clear()
        return {1: "bytes", 2: "triton_fp8"}.get(calls[0], "?") if calls else "torch"

    return route, w


@needs_cuda
def test_routes_to_bytes_kernel_on_nvidia(monkeypatch):
    route, w = _router(monkeypatch)
    assert route((7, 5)) == "bytes"
    assert route((8, 0)) == "bytes"
    assert route((8, 6)) == "bytes"
    # sm89+ keeps the native fp8 kernel for even tiles; ragged e4m3fn weights take the byte decoder
    # there too (bit-identical to the torch loop), ragged e5m2 keeps the torch loop.
    assert route((8, 9)) == "triton_fp8"
    assert route((9, 0)) == "triton_fp8"
    assert route((9, 0), weight = _weight(130, 256, "cuda")) == "bytes"
    assert route((10, 0), weight = _weight(256, 200, "cuda")) == "bytes"
    assert route((9, 0), weight = _weight(130, 256, "cuda").float().to(torch.float8_e5m2)) == "torch"
    assert route((9, 0), weight = _weight(130, 256, "cuda"), out_dtype = torch.float16) == "bytes"
    assert route((9, 4), hip = "6.4.0") == "torch"
    assert route((11, 5), hip = "6.4.0", arch = "gfx1151") == "bytes"
    # e5m2 weights are compiled natively before sm89 and never reach the byte decoder.
    assert route((8, 0), weight = w.float().to(torch.float8_e5m2)) == "triton_fp8"


ROCM_ROUTES = [
    # (gcnArchName, triton, route, route with UNSLOTH_FP8_HIP_BYTES_DEQUANT=0).
    ("gfx1151", "3.6.0", "triton_fp8", "triton_fp8"),  # Strix Halo
    ("gfx1151", "3.4.0", "bytes", "torch"),
    ("gfx1100", "3.3.1", "bytes", "torch"),
    ("gfx1030", "3.5.0", "triton_fp8", "triton_fp8"),
    ("gfx90a:sramecc+:xnack-", "3.5.0", "triton_fp8", "triton_fp8"),
    ("gfx942:sramecc+:xnack-", "3.5.0", "triton_fp8", "triton_fp8"),
    ("gfx942:sramecc+:xnack-", "3.4.0", "triton_fp8", "triton_fp8"),
    ("gfx942:sramecc+:xnack-", "3.2.0", "torch", "torch"),
    ("gfx950", "3.5.0", "triton_fp8", "triton_fp8"),
    ("gfx950", "3.2.0", "torch", "torch"),
    ("gfx1201", "3.4.0", "triton_fp8", "triton_fp8"),
    ("gfx1201", "3.3.1", "torch", "torch"),
]


@needs_cuda
@pytest.mark.parametrize("arch, triton_version, expected, old", ROCM_ROUTES)
def test_rocm_routes(monkeypatch, arch, triton_version, expected, old):
    route, w = _router(monkeypatch)
    hip = dict(hip = "6.4.0", arch = arch, triton_version = triton_version)
    assert route((11, 5), **hip) == expected
    ragged = "bytes" if arch.startswith("gfx11") else "torch"
    assert route((11, 5), weight = _weight(130, 200, "cuda"), **hip) == ragged
    assert route((11, 5), block = (64, 128), **hip) == ragged
    assert (
        route((11, 5), weight = _weight(130, 200, "cuda"), out_dtype = torch.float32, **hip) == ragged
    )
    # fp16 output keeps the old route (the torch loop rounds fp16 differently on gfx1151).
    assert (
        route((11, 5), weight = _weight(130, 200, "cuda"), out_dtype = torch.float16, **hip) == "torch"
    )
    assert route((11, 5), weight = w.float().to(torch.float8_e5m2), **hip) == "triton_fp8"
    if hasattr(torch, "float8_e4m3fnuz"):
        fnuz = w.float().nan_to_num(0).clamp(-240, 240).to(torch.float8_e4m3fnuz)
        assert route((11, 5), weight = fnuz, **hip) == "triton_fp8"


@needs_cuda
@pytest.mark.parametrize("arch, triton_version, expected, old", ROCM_ROUTES)
def test_rocm_kill_switch_restores_old_routes(monkeypatch, arch, triton_version, expected, old):
    monkeypatch.setattr(fp8, "_HIP_FP8_BYTES_DEQUANT", False)
    route, w = _router(monkeypatch)
    hip = dict(hip = "6.4.0", arch = arch, triton_version = triton_version)
    assert route((11, 5), **hip) == old
    assert route((11, 5), weight = _weight(130, 200, "cuda"), **hip) == "torch"


def test_kill_switch_env_is_read_at_import():
    # A fresh interpreter has no conftest spoof, so on a CPU-only box `import unsloth` raises
    # "cannot find any torch accelerator" before the switch is ever read; install the CI spoof.
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    code = (
        "import sys, torch\n"
        f"sys.path.insert(0, {tests_dir!r})\n"
        "if not torch.cuda.is_available():\n"
        "    import _zoo_aggressive_cuda_spoof\n"
        "    _zoo_aggressive_cuda_spoof.apply()\n"
        "from unsloth.kernels import fp8\n"
        "print(fp8._HIP_FP8_BYTES_DEQUANT)\n"
    )
    for value, expected in (("0", "False"), ("1", "True")):
        env = dict(os.environ, UNSLOTH_FP8_HIP_BYTES_DEQUANT = value)
        out = subprocess.run(
            [sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 600
        )
        lines = out.stdout.strip().splitlines()
        assert lines and lines[-1] == expected, (value, out.stderr[-2000:])


@needs_cuda
def test_rocm_unknown_arch_keeps_its_route(monkeypatch):
    route, w = _router(monkeypatch)
    with monkeypatch.context() as mp:
        mp.setattr(torch.cuda, "get_device_properties", _no_props)
        fp8._rocm_bytes_dequant_arch.cache_clear()
        assert not fp8._rocm_bytes_dequant_arch(torch.device("cuda", 0))
    fp8._rocm_bytes_dequant_arch.cache_clear()
    assert route((11, 5), hip = "6.4.0", arch = None, triton_version = "3.6.0") == "triton_fp8"
    assert (
        route(
            (11, 5),
            hip = "6.4.0",
            arch = None,
            triton_version = "3.6.0",
            weight = _weight(130, 200, "cuda"),
        )
        == "torch"
    )


HIP = CUDA and torch.version.hip is not None


@pytest.mark.skipif(not HIP, reason = "needs a real ROCm device")
@pytest.mark.parametrize("shape", [(4096, 4096), (14336, 4096), (4096, 14336), (130, 200)])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_real_rocm_device_takes_bytes_kernel_bit_exact(pre_sm89, monkeypatch, shape, out_dtype):
    m, n = shape
    w = _weight(m, n, "cuda")
    s = _scale(m, n, [128, 128], torch.float32, "cuda")
    calls = []
    orig = fp8._e4m3fn_bytes_dequant
    monkeypatch.setattr(
        fp8, "_e4m3fn_bytes_dequant", lambda *a, **k: calls.append(1) or orig(*a, **k)
    )
    out = fp8.weight_dequant(w, s, out_dtype)
    loop_case = m % 128 or n % 128 or fp8._fp8_kernel_unsupported(w)
    if not (fp8._rocm_bytes_dequant_arch(w.device) and loop_case and out_dtype != torch.float16):
        assert not calls
        return
    assert calls == [1]
    _same_bits(out, fp8._torch_blockwise_dequant(w, s, [128, 128], out_dtype))


@needs_cuda
def test_fp8_block_linear_forward_backward_unchanged(pre_sm89):
    torch.manual_seed(0)
    m, k = 384, 256
    weight = (torch.randn(m, k, device = "cuda") * 0.3).to(torch.float8_e4m3fn)
    scale = torch.rand(3, 2, device = "cuda") + 0.5
    X = (torch.randn(8, k, device = "cuda", dtype = torch.float16)).requires_grad_(True)
    out = fp8.FP8BlockQuantLinear.apply(X, weight, scale)
    out.float().pow(2).sum().backward()
    W = fp8._torch_blockwise_dequant(weight, scale, [128, 128], torch.float16)
    Xr = X.detach().clone().requires_grad_(True)
    ref = Xr @ W.T
    ref.float().pow(2).sum().backward()
    assert torch.equal(out, ref)
    assert torch.equal(X.grad, Xr.grad)


@needs_cuda
def test_under_torch_compile(pre_sm89):
    w = _weight(256, 384, "cuda")
    s = _scale(256, 384, [128, 128], torch.float32, "cuda")
    f = torch.compile(
        lambda w, s: fp8._e4m3fn_bytes_dequant(w, s, 128, 128, torch.bfloat16), fullgraph = True
    )
    _same_bits(f(w, s), fp8._torch_blockwise_dequant(w, s, [128, 128], torch.bfloat16))


def test_cpu_keeps_torch_path(monkeypatch):
    monkeypatch.setattr(
        fp8,
        "_e4m3fn_bytes_dequant",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("byte kernel on CPU")),
    )
    w = _weight(130, 200, "cpu")
    s = _scale(130, 200, [128, 128], torch.float32, "cpu")
    out = fp8._blockwise_weight_dequant_any_shape(w, s, [128, 128], torch.bfloat16)
    _same_bits(out, fp8._torch_blockwise_dequant(w, s, [128, 128], torch.bfloat16))


@needs_cuda
def test_past_int32_offsets(pre_sm89):
    # 2**31 + 128 * 512 elements: int32 offsets would wrap; spot-check the last rows against the torch loop.
    m, n = (2**31) // 4096 + 128, 4096
    if torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        pytest.skip("needs 8 GB free GPU memory")
    w = (
        torch.empty(m, n, dtype = torch.uint8, device = "cuda")
        .random_(0, 126)
        .view(torch.float8_e4m3fn)
    )
    s = _scale(m, n, [128, 128], torch.float32, "cuda")
    out = fp8._blockwise_weight_dequant_any_shape(w, s, [128, 128], torch.float16)
    tail = slice(m - 256, m)
    _same_bits(out[tail], fp8._torch_blockwise_dequant(w[tail], s[-2:], [128, 128], torch.float16))


@needs_cuda
@pytest.mark.skipif(not CUDA or torch.version.hip is not None, reason = "NVIDIA routing")
@pytest.mark.parametrize(
    "shape, block", [((130, 256), [128, 128]), ((300, 520), [128, 128]), ((256, 384), [128, 64])]
)
def test_real_device_ragged_or_rectangular_takes_bytes_bit_exact(monkeypatch, shape, block):
    # Unmocked: every NVIDIA arch sends ragged / rectangular e4m3fn blocks to the byte decoder.
    calls = []
    orig = fp8._e4m3fn_bytes_dequant
    monkeypatch.setattr(
        fp8, "_e4m3fn_bytes_dequant", lambda *a, **k: calls.append(1) or orig(*a, **k)
    )
    w = _weight(*shape, "cuda")
    s = _scale(*shape, block, torch.float32, "cuda")
    out = fp8._blockwise_weight_dequant_any_shape(w, s, block, torch.bfloat16)
    assert calls == [1]
    _same_bits(out, fp8._torch_blockwise_dequant(w, s, block, torch.bfloat16))
