# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm gfx11 Triton conv3d for video VAEs (vendored from ROCm/aiter): gating, MIOpen fallback, memory check,
uncovered shapes, license headers, and the vendored kernels' numerics under the Triton interpreter (no GPU)."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from core.inference import diffusion_rocm_triton_conv3d as R  # noqa: E402

try:  # imported before any fixture fakes a ROCm card (diffusers' import probes the device)
    from diffusers.models.autoencoders import autoencoder_kl_wan as _WAN
except Exception:  # noqa: BLE001
    _WAN = None

BACKEND = Path(__file__).resolve().parents[1]
PKG = BACKEND / "core" / "inference" / "rocm_triton_conv3d"
GiB = 1 << 30


class _Kernels:
    """Stand-in for the vendored module: the real route / size helpers when Triton imports, F.conv3d as the kernel."""

    def __init__(self, fail = None):
        self.calls = 0
        self.fail = fail
        try:
            from core.inference.rocm_triton_conv3d import conv3d as real
            real._is_amd_wave32 = lambda: True
            self._resolve_route, self._out_dhw = real._resolve_route, real._out_dhw
        except Exception:  # noqa: BLE001 - no Triton on this host: a fixed route
            self._resolve_route = lambda *a, **k: types.SimpleNamespace(name = "GENERAL")
            self._out_dhw = lambda D, H, W, T, Rr, S, st, p, d: tuple(
                (n + 2 * pp - dd * (k - 1) - 1) // s + 1
                for n, k, s, pp, dd in zip((D, H, W), (T, Rr, S), st, p, d)
            )

    def conv3d(self, x, w, b, stride, padding, dilation):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return F.conv3d(x, w, b, stride, padding, dilation) + 0  # same math, separate tensor


@pytest.fixture
def gpu(monkeypatch):
    """A gfx1151 ROCm card with plenty of free memory; CPU tensors stand in for device tensors."""
    for k in (R.ENABLE_ENV, R.LEGACY_ENABLE_ENV, R.WINOGRAD_ENV):
        monkeypatch.delenv(k, raising = False)
    monkeypatch.setenv(R.SELFCHECK_ENV, "0")
    monkeypatch.setattr(torch.version, "hip", "7.13.0-fake", raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_a, **_k: types.SimpleNamespace(gcnArchName = "gfx1151:xnack-"),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a: (64 * GiB, 128 * GiB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_a: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_a: 0)
    monkeypatch.setattr(R, "_on_gpu", lambda t: True)
    kernels = _Kernels()
    monkeypatch.setitem(R._STATE, "mod", kernels)
    R._DEVICE_OK.clear()
    R._FAILED.clear()
    R.reset_counts()
    yield kernels
    R._DEVICE_OK.clear()
    R._FAILED.clear()
    R.reset_counts()


def _vae(padding_mode = "zeros", groups = 1):
    vae = torch.nn.Module()
    vae.conv = torch.nn.Conv3d(8, 8, 3, padding = 1, padding_mode = padding_mode, groups = groups).to(
        torch.bfloat16
    )
    vae.pw = torch.nn.Conv3d(8, 4, 1).to(torch.bfloat16)
    vae.conv2d = torch.nn.Conv2d(8, 8, 3).to(torch.bfloat16)
    vae.requires_grad_(True)  # diffusers' default: inference weights still carry requires_grad
    return vae


def _x(shape = (1, 8, 3, 6, 6)):
    return torch.randn(shape).to(torch.bfloat16)


def test_default_on_for_gfx1151_and_measured_vaes_only(gpu, monkeypatch):
    assert R.DEFAULT_ON_ARCHES == ("gfx1151",)
    assert R.arch_enabled("gfx1151") and R.arch_enabled("gfx1151:xnack-")
    assert not R.arch_enabled("gfx1100") and not R.arch_enabled("gfx942")
    assert R.install(_vae()) == 0  # env unset: an unmeasured VAE class stays on MIOpen
    monkeypatch.setattr(R, "DEFAULT_ON_VAES", ("Module",))
    assert R.install(_vae()) == 2
    monkeypatch.setenv(R.ENABLE_ENV, "0")
    R._DEVICE_OK.clear()
    assert R.install(_vae()) == 0


def test_env_on_routes_any_vae_class(gpu, monkeypatch):
    monkeypatch.setenv(R.ENABLE_ENV, "1")
    assert R.install(_vae()) == 2


@pytest.mark.parametrize("env", [R.ENABLE_ENV, R.LEGACY_ENABLE_ENV])
def test_env_routes_and_matches_stock(gpu, monkeypatch, env):
    monkeypatch.setenv(env, "1")
    vae = _vae()
    assert R.install(vae) == 2  # the two Conv3d; Conv2d untouched
    assert R.install(vae) == 0  # idempotent
    x = _x()
    with torch.no_grad():
        y = vae.conv(x)
        ref = F.conv3d(x, vae.conv.weight, vae.conv.bias, padding = 1)
    assert torch.equal(y, ref)
    assert gpu.calls == 1 and R.COUNTS["routed"] == 1
    assert R.uninstall(vae) == 2 and "_conv_forward" not in vae.conv.__dict__


@pytest.mark.parametrize(
    "hip, arch, routed",
    [(None, "gfx1151", 0), ("7.13", "gfx942", 0), ("7.13", "gfx1201", 0), ("7.13", "gfx1100", 2)],
)
def test_gate_rocm_gfx11_only(gpu, monkeypatch, hip, arch, routed):
    monkeypatch.setenv(R.ENABLE_ENV, "1")
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    if hip is None:
        monkeypatch.setattr(torch, "__version__", "2.11.0+cu130")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_a, **_k: types.SimpleNamespace(gcnArchName = arch),
    )
    assert R.install(_vae()) == routed


def test_stock_when_autograd_records_or_dtype_or_compiling(gpu, monkeypatch):
    monkeypatch.setenv(R.ENABLE_ENV, "1")
    vae = _vae()
    R.install(vae)
    vae.conv(_x())  # grad enabled + weight requires grad: no Triton backward, so MIOpen
    with torch.no_grad():
        vae.float().conv(_x().float())  # fp32
        vae.bfloat16()
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        vae.conv(_x())
    assert gpu.calls == 0 and R.COUNTS["routed"] == 0


def test_non_zero_padding_mode_and_groups_are_never_wrapped(gpu, monkeypatch):
    monkeypatch.setenv(R.ENABLE_ENV, "1")
    assert R.install(_vae(padding_mode = "reflect")) == 1  # only the 1x1x1
    assert R.install(_vae(groups = 2)) == 1


def test_launch_failure_falls_back_and_is_not_retried(gpu, monkeypatch):
    monkeypatch.setenv(R.ENABLE_ENV, "1")
    gpu.fail = RuntimeError("triton compile failed")
    vae = _vae()
    R.install(vae)
    x = _x()
    with torch.no_grad():
        ref = F.conv3d(x, vae.conv.weight, vae.conv.bias, padding = 1)
        assert torch.equal(vae.conv(x), ref)
        assert torch.equal(vae.conv(x), ref)
    assert gpu.calls == 1  # the failing shape is cached
    assert R.COUNTS["fallback_error"] == 2 and R.COUNTS["routed"] == 0


def test_out_of_memory_is_reraised(gpu, monkeypatch):
    monkeypatch.setenv(R.ENABLE_ENV, "1")
    gpu.fail = torch.OutOfMemoryError("HIP out of memory")
    vae = _vae()
    R.install(vae)
    with torch.no_grad(), pytest.raises(torch.OutOfMemoryError):
        vae.conv(_x())


def test_low_free_memory_keeps_miopen(gpu, monkeypatch):
    monkeypatch.setenv(R.ENABLE_ENV, "1")
    monkeypatch.setattr(R, "MEM_CHECK_MIN_BYTES", 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a: (256 << 20, 128 * GiB))
    vae = _vae()
    R.install(vae)
    with torch.no_grad():
        vae.conv(_x())
    assert gpu.calls == 0 and R.COUNTS["fallback_memory"] == 1
    # reserved-but-unallocated allocator blocks are not counted as free
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_a: 16 * GiB)
    with torch.no_grad():
        vae.conv(_x())
    assert gpu.calls == 0 and R.COUNTS["fallback_memory"] == 2
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a: (64 * GiB, 128 * GiB))
    with torch.no_grad():
        vae.conv(_x())
    assert gpu.calls == 1 and R.COUNTS["routed"] == 1


def test_small_calls_skip_the_driver_memory_query(gpu, monkeypatch):
    monkeypatch.setenv(R.ENABLE_ENV, "1")
    monkeypatch.setattr(R, "MEM_CHECK_MIN_BYTES", 1 << 40)

    def boom(*_a):
        raise AssertionError("mem_get_info called for a small call")

    monkeypatch.setattr(torch.cuda, "mem_get_info", boom)
    vae = _vae()
    R.install(vae)
    with torch.no_grad():
        vae.conv(_x())
    assert gpu.calls == 1


def test_graph_capture_stays_on_miopen(gpu, monkeypatch):
    monkeypatch.setenv(R.ENABLE_ENV, "1")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    vae = _vae()
    R.install(vae)
    x = _x()
    with torch.no_grad():
        y = vae.conv(x)
        ref = F.conv3d(x, vae.conv.weight, vae.conv.bias, padding = 1)
    assert gpu.calls == 0 and torch.equal(y, ref)


def test_old_triton_keeps_miopen(monkeypatch):
    fake = types.SimpleNamespace(__version__ = "3.4.0")
    monkeypatch.setitem(sys.modules, "triton", fake)
    assert not R._triton_ok()
    fake.__version__ = "3.6.0"
    assert R._triton_ok()
    fake.__version__ = "not-a-version"
    assert not R._triton_ok()


_WAN_720P = ((1, 384, 4, 176, 320), (384, 384, 3, 3, 3), 2, (1, 1, 1), (0, 1, 1), (1, 1, 1))


def test_default_routes_3x3x3_to_direct_kernels_not_winograd(monkeypatch):
    pytest.importorskip("triton")
    from core.inference.rocm_triton_conv3d import conv3d as real

    k = _Kernels()
    monkeypatch.setattr(real, "ALLOW_WINOGRAD", False)
    route, extra, _ = R.plan(k, *_WAN_720P)
    assert route.name == "CBLOCKED_NCDHW"  # Winograd-eligible shape, direct kernel by default
    y, pack, wpack = 384 * 2 * 176 * 320 * 2, 384 * 4 * 176 * 320 * 2, 384 * 27 * 384 * 2
    assert (
        extra == y + pack + wpack
    )  # output (OD = 2) + NCDHWc input pack + weight pack, no Winograd temporaries
    monkeypatch.setattr(real, "ALLOW_WINOGRAD", True)
    assert R.plan(k, *_WAN_720P)[0].name.startswith("WINOGRAD_HW")


def test_winograd_only_on_explicit_opt_in(gpu, monkeypatch):
    monkeypatch.setattr(R, "DEFAULT_ON_VAES", ("Module",))
    assert not R.winograd_enabled()
    assert R.install(_vae()) == 2 and gpu.ALLOW_WINOGRAD is False
    for v in ("0", "", "maybe"):
        monkeypatch.setenv(R.WINOGRAD_ENV, v)
        assert not R.winograd_enabled()
    monkeypatch.setenv(R.ENABLE_ENV, "1")  # routing on everywhere still leaves Winograd off
    monkeypatch.delenv(R.WINOGRAD_ENV)
    assert R.install(_vae()) == 2 and gpu.ALLOW_WINOGRAD is False
    monkeypatch.setenv(R.WINOGRAD_ENV, "1")
    assert R.install(_vae()) == 2 and gpu.ALLOW_WINOGRAD is True
    monkeypatch.setenv(R.ENABLE_ENV, "0")  # the kill switch wins over the Winograd opt-in
    R._DEVICE_OK.clear()
    assert R.install(_vae()) == 0


def test_selfcheck_bound_and_coverage(monkeypatch):
    monkeypatch.setenv(R.SELFCHECK_ENV, "1")
    assert R.SELFCHECK_REL_L2 <= 1e-2
    seen = []

    class _Stub:
        ALLOW_WINOGRAD = False
        noise = 0.0

        def conv3d(self, x, w, b, stride, padding, dilation):
            seen.append((tuple(w.shape[2:]), tuple(stride)))
            y = F.conv3d(x.float(), w.float(), b.float(), stride = stride, padding = padding)
            return (y * (1 + self.noise)).to(x.dtype)

    def _forget():
        for k in [k for k in R._STATE if isinstance(k, tuple) and k[0] == "selfcheck"]:
            R._STATE.pop(k)
        R._STATE.pop("selfcheck_error", None)

    stub = _Stub()
    _forget()
    assert R._selfcheck(torch, stub, "cpu", torch.bfloat16)  # bf16 output rounding only
    assert ((3, 3, 3), (1, 2, 2)) in seen and ((3, 1, 1), (2, 1, 1)) in seen
    stub.noise = 0.02
    stub.ALLOW_WINOGRAD = True  # separate cache entry
    assert not R._selfcheck(torch, stub, "cpu", torch.bfloat16)
    assert "rel L2" in R._STATE["selfcheck_error"]
    _forget()


def test_direct_kernels_keep_spatial_sizes_out_of_the_compile_key():
    pytest.importorskip("triton")
    from core.inference.rocm_triton_conv3d import conv3d_1x1x1, conv3d_3x3x3, conv3d_general

    sizes = {"N", "D", "H", "W_in", "OD", "P", "Q", "M_total"}
    for kern in (
        conv3d_1x1x1._conv3d_1x1x1_kernel,
        conv3d_general._conv3d_general_kernel,
        conv3d_3x3x3._conv3d_3x3x3_cblocked_kernel,
    ):
        params = {p.name: p for p in kern.params}
        for name in sizes:
            assert not params[name].is_constexpr, (kern.fn.__name__, name)
            assert params[name].do_not_specialize, (kern.fn.__name__, name)
        for name in ("C", "K_out", "BLOCK_M", "BLOCK_N", "BLOCK_K"):
            assert params[name].is_constexpr, (kern.fn.__name__, name)


def test_memory_estimate_counts_winograd_buffers(monkeypatch):
    pytest.importorskip("triton")
    from core.inference.rocm_triton_conv3d import conv3d as real

    monkeypatch.setattr(real, "ALLOW_WINOGRAD", True)
    k = _Kernels()
    # Wan 2.2 decoder, 720p, 384 channels, 4 frames with cache: H/W Winograd (opt-in)
    route, extra, biggest = R.plan(k, *_WAN_720P)
    assert route.name.startswith("WINOGRAD_HW")
    tiles = 44 * 80
    assert extra >= 36 * 4 * tiles * 384 * 2 + 36 * 2 * tiles * 384 * 4  # V + M
    assert biggest >= 36 * 4 * tiles * 384


def test_int32_index_overflow_stays_on_miopen(gpu, monkeypatch):
    monkeypatch.setenv(R.ENABLE_ENV, "1")
    vae = _vae()
    R.install(vae)
    monkeypatch.setattr(R, "plan", lambda *a, **k: (None, 0, 2**31))
    with torch.no_grad():
        vae.conv(_x())
    assert gpu.calls == 0 and R.COUNTS["fallback_shape"] == 1


def test_wan_causal_conv_reaches_the_router(gpu, monkeypatch):
    if _WAN is None:
        pytest.skip("diffusers without AutoencoderKLWan")
    W = _WAN
    monkeypatch.setenv(R.ENABLE_ENV, "1")
    vae = torch.nn.Module()
    vae.c = W.WanCausalConv3d(8, 8, 3, padding = 1).to(torch.bfloat16)
    assert R.install(vae) == 1
    x = _x()
    with torch.no_grad():
        y = vae.c(x)
        ref = F.conv3d(F.pad(x, vae.c._padding), vae.c.weight, vae.c.bias)
    assert torch.equal(y, ref) and gpu.calls == 1


def test_install_wired_into_apply_speed_optims_on_rocm_only():
    src = (BACKEND / "core" / "inference" / "diffusion_speed.py").read_text(encoding = "utf-8")
    i = src.index("diffusion_rocm_triton_conv3d.install(")
    assert 'getattr(target, "backend", None) == "rocm"' in src[i - 400 : i]


def test_every_vendored_file_carries_both_licenses():
    files = sorted(PKG.glob("*.py"))
    assert len(files) >= 12
    for f in files:
        head = f.read_text(encoding = "utf-8")[:4000]
        assert head.startswith("# SPDX-License-Identifier: AGPL-3.0-only\n"), f.name
        assert "# SPDX-License-Identifier: MIT" in head, f.name
        assert (
            "MIT License" in head and "Permission is hereby granted, free of charge" in head
        ), f.name
        assert "Advanced Micro Devices, Inc. All rights reserved." in head, f.name
        assert "ROCm/aiter" in head and "9e5e03626fa119f2429fa1c1578d5a715a6c2401" in head, f.name


def test_vendored_package_needs_no_aiter():
    for f in PKG.glob("*.py"):
        for line in f.read_text(encoding = "utf-8").splitlines():
            s = line.strip()
            if s.startswith(("import ", "from ")):
                assert "aiter" not in s, (f.name, s)


_INTERP = r"""
import contextlib, os, sys
os.environ["TRITON_INTERPRET"] = "1"
import torch, torch.nn.functional as F
sys.path.insert(0, sys.argv[1])
assert "aiter" not in sys.modules
from core.inference.rocm_triton_conv3d import _config, conv3d as C
_config.get_arch = lambda: "gfx1151"
C._validate_conv_devices = lambda *a, **k: None
C._is_amd_wave32 = lambda: True
torch.cuda.device = lambda d: contextlib.nullcontext()
torch.manual_seed(0)
seen = set()
for (wino, N, c, D, H, W, K, k, s, p) in [
    (False, 1, 32, 2, 6, 7, 48, (1, 1, 1), 1, 0),
    (False, 1, 8, 3, 6, 6, 16, (3, 3, 3), 1, 1),
    (False, 1, 64, 3, 6, 6, 64, (3, 3, 3), 1, (0, 1, 1)),
    (False, 1, 64, 3, 9, 9, 64, (3, 3, 3), (1, 2, 2), (0, 1, 1)),
    (False, 1, 32, 5, 6, 6, 32, (3, 1, 1), (2, 1, 1), (1, 0, 0)),
    (False, 1, 160, 3, 28, 28, 96, (3, 3, 3), 1, (1, 1, 1)),
    (False, 1, 160, 2, 20, 12, 96, (3, 3, 3), 1, (1, 1, 1)),
    (True, 1, 160, 3, 28, 28, 96, (3, 3, 3), 1, (1, 1, 1)),
    (False, 1, 16, 3, 8, 8, 16, (1, 3, 3), (1, 2, 2), 0),
]:
    C.ALLOW_WINOGRAD = wino
    x = torch.randn(N, c, D, H, W, dtype = torch.float16)
    w = (torch.randn(K, c, *k) * 0.05).half()
    b = torch.randn(K).half()
    st, pd, dl = C._normalize_conv3d_params(s, p, 1)
    seen.add((wino, C._resolve_route(*k, st, dl, N, c, D, H, W, K, "ncdhw", padding = pd).name))
    y = C.conv3d(x, w, b, s, p, 1)
    ref = F.conv3d(x.float(), w.float(), b.float(), stride = s, padding = p)
    rel = float((y.float() - ref).norm() / ref.norm())
    assert y.dtype == x.dtype and y.shape == ref.shape and rel < 1e-2, (k, rel)
assert seen == {
    (False, "ONE_X_ONE_X_ONE"), (False, "GENERAL"), (False, "CBLOCKED_NCDHW"), (True, "WINOGRAD_HW")
}, seen
print("INTERP_OK")
"""


def test_vendored_kernels_match_conv3d_under_triton_interpreter():
    pytest.importorskip("triton")
    env = {k: v for k, v in os.environ.items() if "TOKEN" not in k}
    r = subprocess.run(
        [sys.executable, "-c", _INTERP, str(BACKEND)],
        capture_output = True,
        text = True,
        timeout = 900,
        env = env,
    )
    assert r.returncode == 0 and "INTERP_OK" in r.stdout, (r.stdout + r.stderr)[-3000:]
