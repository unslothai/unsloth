# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The block dequant fallback must divert on exactly the GPUs triton cannot compile fp8e4nv for.

triton adds fp8e4nv (torch.float8_e4m3fn) to its supported dtypes at compute capability >= 89
(third_party/nvidia/backend/compiler.py), so sm80 has to take the torch path or it raises
instead of falling back. sm89 -- 4090, L40S, L4 -- is on the supported side of that line, and a
major-only "< 9" test silently drops all of Ada onto a path that costs several times the memory.

The capability is monkeypatched and the triton entry point is stubbed, so the routing cases run
on any CUDA GPU including the pre-sm89 ones under discussion, which cannot compile the kernel
they are asserted to select. Only the value comparison needs real fp8 hardware, and it skips.
"""

import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")

BLOCK = [128, 128]


@pytest.fixture(autouse = True)
def _clear_capability_cache():
    # The guard caches per device, so a simulated capability would otherwise be ignored.
    from unsloth.kernels import fp8

    fp8._fp8_device_lacks_kernel.cache_clear()
    yield
    fp8._fp8_device_lacks_kernel.cache_clear()


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
    gcn_arch = None,
    triton_version = None,
):
    """Return "triton" or "torch" for a simulated device.

    The kernel is stubbed rather than wrapped: these cases assert which branch is chosen, and
    letting the real one run would need a GPU that can compile fp8e4nv, so every "routes to
    triton" case would fail on exactly the pre-sm89 cards this file is about. Values are
    covered separately by test_fallback_matches_the_triton_kernel_bit_for_bit, which skips
    when the hardware cannot run both sides.
    """
    from unsloth.kernels import fp8

    calls = []

    def _stub(
        x,
        s,
        block_size = 128,
        dtype = torch.bfloat16,
    ):
        calls.append(1)
        return torch.empty(x.shape, dtype = dtype, device = x.device)

    monkeypatch.setattr(fp8, "weight_dequant_block", _stub)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(torch.version, "hip", hip)
    if gcn_arch is not None:
        props = types.SimpleNamespace(gcnArchName = gcn_arch)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: props)
    if triton_version is not None:
        monkeypatch.setattr(fp8.triton, "__version__", triton_version)
    fp8._blockwise_weight_dequant_any_shape(weight, scale, BLOCK, torch.bfloat16)
    return "triton" if calls else "torch"


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
    assert _route(monkeypatch, capability, weight, scale) == expected


@pytest.mark.parametrize("capability", [(7, 0), (8, 0), (8, 9), (9, 0)])
def test_float8_e5m2_is_never_diverted(monkeypatch, capability):
    # fp8e5 compiles on every arch triton supports, so the guard must not touch it.
    weight, scale = _make(dtype = torch.float8_e5m2)
    assert _route(monkeypatch, capability, weight, scale) == "triton"


@pytest.mark.parametrize(
    "gcn_arch, capability, triton_version, expected",
    [
        # triton 3.5+ lists fp8e4nv on every AMD arch (emulated without OCP fp8 hardware).
        ("gfx950", (9, 5), "3.5.0", "triton"),  # MI350X / MI355X
        ("gfx942:sramecc+:xnack-", (9, 4), "3.5.0", "triton"),  # MI300X
        ("gfx90a:sramecc+:xnack-", (9, 0), "3.5.0", "triton"),  # MI210 / MI250X (CDNA2)
        ("gfx1151", (11, 5), "3.6.0", "triton"),  # Strix Halo: verified on real hardware
        ("gfx1030", (10, 3), "3.5.0", "triton"),  # RX 6900 XT (RDNA2)
        # triton 3.4: gfx942 / gfx950 / gfx12 only.
        ("gfx1201", (12, 0), "3.4.0", "triton"),  # RX 9070 XT (RDNA4)
        ("gfx942:sramecc+:xnack-", (9, 4), "3.4.0", "triton"),
        ("gfx1151", (11, 5), "3.4.0", "torch"),
        ("gfx90a:sramecc+:xnack-", (9, 0), "3.4.0", "torch"),
        # triton 3.3: CDNA3 / CDNA4 only.
        ("gfx950", (9, 5), "3.3.1", "triton"),
        ("gfx942:sramecc+:xnack-", (9, 4), "3.3.1", "triton"),
        ("gfx1201", (12, 0), "3.3.1", "torch"),
        ("gfx1100", (11, 0), "3.3.1", "torch"),
        # triton 3.2 (torch 2.6 ROCm): fp8e5 only.
        ("gfx942:sramecc+:xnack-", (9, 4), "3.2.0", "torch"),
        ("gfx950", (9, 5), "3.2.0", "torch"),
    ],
)
def test_rocm_routes_on_gfx_arch_not_capability(
    monkeypatch, gcn_arch, capability, triton_version, expected
):
    # get_device_capability is gfx-derived on ROCm (gfx1151 reports 11.5, above any sm89 cut), so
    # the guard keys on the arch: archs AMD triton cannot compile fp8e4nv for take the torch path.
    weight, scale = _make()
    route = _route(
        monkeypatch,
        capability,
        weight,
        scale,
        hip = "6.4.0",
        gcn_arch = gcn_arch,
        triton_version = triton_version,
    )
    assert route == expected


def test_rocm_unknown_arch_takes_the_torch_path(monkeypatch):
    # A missing or unreadable gcnArchName must fall back, not compile a kernel that may not exist.
    from unsloth.kernels import fp8

    def _boom(*a, **k):
        raise RuntimeError("no device properties")

    monkeypatch.setattr(torch.cuda, "get_device_properties", _boom)
    weight, scale = _make()
    # Only triton 3.3 / 3.4 need the arch; 3.5+ supports fp8e4nv everywhere without asking.
    assert (
        _route(monkeypatch, (9, 4), weight, scale, hip = "6.4.0", triton_version = "3.4.0") == "torch"
    )


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

    # Both capabilities are simulated in one test, so clear between them too: the cache is
    # keyed on the device, and a stale entry would quietly compare triton against itself.
    with monkeypatch.context() as mp:
        mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
        fp8._fp8_device_lacks_kernel.cache_clear()
        assert not fp8._fp8_kernel_unsupported(weight)
        via_triton = fp8._blockwise_weight_dequant_any_shape(weight, scale, block, torch.bfloat16)
    with monkeypatch.context() as mp:
        mp.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
        fp8._fp8_device_lacks_kernel.cache_clear()
        assert fp8._fp8_kernel_unsupported(weight)
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


@pytest.mark.parametrize("weight_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_pre_sm89_forward_never_reaches_the_fp8_only_kernels(monkeypatch, weight_dtype):
    # Fixing the weight dequant alone is not enough to run a model: the forward also calls
    # act_quant and the w8a8 gemm, and both take fp8e4nv pointers, so on a pre-sm89 card the
    # public path kept raising with the dequant helper already fixed. Assert the forward
    # reaches neither, rather than only that it returns. e5m2 weights are covered because
    # act_quant emits e4m3fn regardless, so keying the decision on the weight alone lets them
    # through to a kernel this device cannot compile.
    from unsloth.kernels import fp8

    called = []
    monkeypatch.setattr(
        fp8,
        "act_quant",
        lambda *a, **k: called.append("act_quant")
        or (_ for _ in ()).throw(AssertionError("act_quant reached on a pre-sm89 device")),
    )
    monkeypatch.setattr(
        fp8,
        "fp8_block_matmul",
        lambda *a, **k: called.append("fp8_block_matmul")
        or (_ for _ in ()).throw(AssertionError("fp8_block_matmul reached on a pre-sm89 device")),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))

    torch.manual_seed(0)
    bs, m, k = 128, 512, 1024
    weight = (torch.randn(m, k, device = "cuda") * 0.3).to(weight_dtype)
    scale = torch.rand(m // bs, k // bs, device = "cuda", dtype = torch.float32) + 0.5
    scale.block_size = [bs, bs]
    X = (torch.randn(32, k, device = "cuda", dtype = torch.bfloat16) * 0.5).requires_grad_(True)

    out = fp8.FP8BlockQuantLinear.apply(X, weight, scale)
    out.float().pow(2).mean().backward()
    torch.cuda.synchronize()

    assert called == [], f"pre-sm89 forward reached fp8-only kernels: {called}"
    assert out.shape == (32, m) and torch.isfinite(out.float()).all()
    assert X.grad is not None and torch.isfinite(X.grad).all()


def test_grouped_fp8_eval_is_guarded_in_source():
    # The grouped layer only exists in newer transformers, so the runtime test below skips on
    # most installs. Assert statically that eval does not bypass the guard, or admitting these
    # checkpoints at load time just moves the failure to the first grouped forward.
    import ast
    import inspect

    from unsloth.kernels import fp8

    tree = ast.parse(inspect.getsource(fp8))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_fp8_grouped_forward"
    )
    guard = ast.dump(fn.body[0].test)
    assert "_fp8_kernel_unsupported" in guard, "grouped eval no longer consults the guard"
    assert "training" in guard, "grouped forward no longer distinguishes training from eval"


def test_grouped_fp8_eval_uses_the_fallback_on_pre_sm89(monkeypatch):
    from unsloth.kernels import fp8

    if getattr(fp8, "FP8GroupedLinear", None) is None:
        pytest.skip("this transformers has no FP8GroupedLinear")

    calls = []
    monkeypatch.setattr(
        fp8,
        "_blockwise_weight_dequant_any_shape",
        lambda w, s, b, d, _o = fp8._blockwise_weight_dequant_any_shape: (
            calls.append(1) or _o(w, s, b, d)
        ),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))

    n_groups, out_per, hidden, bs = 2, 256, 256, 128
    layer = type("Stub", (), {})()
    layer.weight = (torch.randn(n_groups * out_per, hidden, device = "cuda") * 0.3).to(
        torch.float8_e4m3fn
    )
    layer.weight_scale_inv = (
        torch.rand(n_groups * out_per // bs, hidden // bs, device = "cuda", dtype = torch.float32) + 0.5
    )
    layer.n_groups, layer.block_size, layer.has_bias = n_groups, [bs, bs], False
    layer.bias, layer.training = None, False

    out = fp8.FP8GroupedLinear.forward(
        layer, torch.randn(4, n_groups, hidden, device = "cuda", dtype = torch.bfloat16)
    )

    assert calls, "grouped eval bypassed the guarded dequant on a pre-sm89 device"
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("gcn_arch", ["gfx1151", "gfx90a:sramecc+:xnack-"])
def test_rocm_forward_without_fp8e4nv_skips_act_quant(monkeypatch, gcn_arch):
    # The block forward quantizes the activation to e4m3fn in triton (act_quant) before the fp8 gemm;
    # on an arch without fp8e4nv both must be skipped for dequant plus a 16-bit matmul.
    from unsloth.kernels import fp8

    def _no_kernel(*a, **k):
        raise AssertionError("fp8e4nv triton kernel launched on an arch without fp8e4nv")

    monkeypatch.setattr(fp8, "act_quant", _no_kernel)
    monkeypatch.setattr(fp8, "fp8_block_matmul", _no_kernel)
    monkeypatch.setattr(fp8, "weight_dequant_block", _no_kernel)
    monkeypatch.setattr(torch.version, "hip", "6.4.0")
    props = types.SimpleNamespace(gcnArchName = gcn_arch)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: props)
    monkeypatch.setattr(fp8.triton, "__version__", "3.4.0")

    weight, scale = _make()
    X = torch.randn(4, 256, device = "cuda", dtype = torch.bfloat16)
    out = fp8.FP8BlockQuantLinear.apply(X, weight, scale)
    W = (weight.float() * scale.repeat_interleave(BLOCK[0], 0).repeat_interleave(BLOCK[1], 1)).to(
        torch.bfloat16
    )
    torch.testing.assert_close(out, X @ W.T)


def test_real_device_forward_matches_the_dequant_reference():
    # Unmocked: whatever this GPU is (sm80, sm89+, gfx1151, gfx942 ...), the public forward must run
    # and agree with a float32 dequant reference. On a card triton cannot compile fp8e4nv for, the
    # guard is the only thing between this call and a compile error.
    from unsloth.kernels import fp8

    torch.manual_seed(0)
    bs, m, k = 128, 256, 512
    weight = (torch.randn(m, k, device = "cuda") * 0.3).to(torch.float8_e4m3fn)
    scale = torch.rand(m // bs, k // bs, device = "cuda", dtype = torch.float32) + 0.5
    scale.block_size = [bs, bs]
    X = torch.randn(16, k, device = "cuda", dtype = torch.bfloat16) * 0.5
    out = fp8.FP8BlockQuantLinear.apply(X, weight, scale)
    torch.cuda.synchronize()
    full_scale = scale.repeat_interleave(bs, 0).repeat_interleave(bs, 1)
    ref = X.float() @ (weight.float() * full_scale).t()
    rel = (out.float() - ref).norm() / ref.norm()
    # The fp8 kernels also quantize the activation (a few % error); the dequant path is ~bf16 exact.
    assert torch.isfinite(out.float()).all() and rel < 0.08, f"relative error {rel:.4f}"
