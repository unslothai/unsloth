# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the cached FlashInfer dispatch (``diffusion_nvfp4_dispatch.py``): the version
allowlist, the all-or-nothing private import, and the per-device bit-identity check."""

from __future__ import annotations

import sys
import types

import pytest

from core.inference import diffusion_nvfp4_dispatch as dispatch
from core.inference import diffusion_nvfp4_ops as ops

_PRIVATE = {
    "flashinfer.autotuner": ("AutoTuner",),
    "flashinfer.fp4_quantization": ("get_fp4_quantization_module",),
    "flashinfer.gemm.gemm_base": (
        "DEFAULT_WORKSPACE_SIZE",
        "_MM_FP4_TUNING_CONFIG_128x4",
        "_get_cache_buf",
        "get_cutlass_fp4_gemm_module",
    ),
    "flashinfer.utils": ("device_support_pdl", "get_compute_capability"),
}


@pytest.fixture(autouse = True)
def _clean_dispatch():
    dispatch.reset()
    yield
    dispatch.reset()


def _fake_flashinfer(
    monkeypatch,
    *,
    version = "0.6.6",
    drop = (),
):
    """A ``flashinfer`` package tree with exactly the private symbols the module imports."""
    root = types.ModuleType("flashinfer")
    root.__version__ = version
    root.__path__ = []
    monkeypatch.setitem(sys.modules, "flashinfer", root)
    gemm = types.ModuleType("flashinfer.gemm")
    gemm.__path__ = []
    monkeypatch.setitem(sys.modules, "flashinfer.gemm", gemm)
    for name, symbols in _PRIVATE.items():
        module = types.ModuleType(name)
        for symbol in symbols:
            if symbol not in drop:
                setattr(module, symbol, object())
        monkeypatch.setitem(sys.modules, name, module)
    return root


def test_the_allowlisted_version_with_every_symbol_is_available(monkeypatch):
    _fake_flashinfer(monkeypatch)
    ok, reason = dispatch.available()
    assert ok is True, reason
    assert "0.6.6" in reason


@pytest.mark.parametrize("version", ["0.6.5", "0.6.7", "0.7.0", "unknown"])
def test_an_unlisted_version_refuses_even_when_every_symbol_is_there(monkeypatch, version):
    """Exact, not a minimum: a private symbol that moves in 0.6.7 is not a bug in 0.6.7."""
    _fake_flashinfer(monkeypatch, version = version)
    ok, reason = dispatch.available()
    assert ok is False
    assert version in reason and "public API" in reason


@pytest.mark.parametrize(
    "missing",
    ["AutoTuner", "_get_cache_buf", "_MM_FP4_TUNING_CONFIG_128x4", "get_cutlass_fp4_gemm_module"],
)
def test_one_missing_private_symbol_takes_the_whole_fast_path_down(monkeypatch, missing):
    _fake_flashinfer(monkeypatch, drop = (missing,))
    ok, reason = dispatch.available()
    assert ok is False
    assert "ImportError" in reason and missing in reason


def test_the_env_switch_refuses_before_it_imports_anything(monkeypatch):
    def _boom(*_a, **_kw):  # pragma: no cover - reached only on a regression
        raise AssertionError("the probe imported flashinfer under FAST_DISPATCH=0")

    monkeypatch.setitem(sys.modules, "flashinfer", property(_boom))
    monkeypatch.setenv(dispatch.NVFP4_FAST_DISPATCH_ENV, "0")
    ok, reason = dispatch.available()
    assert ok is False and reason.endswith("=0")


def test_env_one_skips_the_version_check_and_nothing_else(monkeypatch):
    monkeypatch.setenv(dispatch.NVFP4_FAST_DISPATCH_ENV, "1")
    _fake_flashinfer(monkeypatch, version = "0.7.0")
    assert dispatch.available()[0] is True

    dispatch.reset()
    _fake_flashinfer(monkeypatch, version = "0.7.0", drop = ("AutoTuner",))
    ok, reason = dispatch.available()
    assert ok is False and "AutoTuner" in reason


@pytest.mark.parametrize("value", ["", "auto", "AUTO", "yes", "2"])
def test_an_unrecognised_env_value_reads_as_auto(monkeypatch, value):
    monkeypatch.setenv(dispatch.NVFP4_FAST_DISPATCH_ENV, value)
    assert dispatch.fast_dispatch_env() == "auto"


def test_nothing_is_enabled_until_verify_has_passed_on_that_device(monkeypatch):
    _fake_flashinfer(monkeypatch)
    monkeypatch.setattr(ops, "_device_index", lambda device: int(device))
    assert dispatch.available()[0] is True
    assert dispatch.enabled(0) is False
    assert dispatch.quant_fn(0) is None


def test_a_failed_verify_is_remembered_and_never_retried(monkeypatch):
    _fake_flashinfer(monkeypatch)
    monkeypatch.setattr(ops, "_device_index", lambda device: int(device))
    calls = []

    def _fail(device):
        calls.append(device)
        return False, "stub failure"

    monkeypatch.setattr(dispatch, "_run_verify", _fail)
    assert dispatch.verify(0) == (False, "stub failure")
    assert dispatch.verify(0) == (False, "stub failure")
    assert len(calls) == 1
    assert dispatch.enabled(0) is False


def test_verify_does_not_run_the_gemm_when_the_library_is_wrong(monkeypatch):
    _fake_flashinfer(monkeypatch, version = "0.7.0")
    monkeypatch.setattr(ops, "_device_index", lambda device: int(device))

    def _boom(_device):  # pragma: no cover - reached only on a regression
        raise AssertionError("verify ran the GEMM on an unlisted version")

    monkeypatch.setattr(dispatch, "_run_verify", _boom)
    ok, reason = dispatch.verify(0)
    assert ok is False and "0.7.0" in reason


class _Ptr:
    """A stand-in for a weight buffer: a data_ptr, a shape and a ``.T``."""

    def __init__(
        self,
        pointer,
        shape = (8, 4),
    ):
        self._pointer = pointer
        self.shape = shape

    def data_ptr(self):
        return self._pointer

    def detach(self):
        return self

    @property
    def T(self):
        return ("view", self._pointer, self.shape)


def test_a_cold_plan_is_never_built_during_a_capture(monkeypatch):
    """``choose_one`` may PROFILE, and a profiling launch inside a capture is in the graph forever."""
    _fake_flashinfer(monkeypatch)
    monkeypatch.setattr(ops, "_device_index", lambda device: 0)
    monkeypatch.setattr(dispatch, "enabled", lambda device: True)
    monkeypatch.setattr(ops, "_is_capturing", lambda: True)

    def _boom(*_a, **_kw):  # pragma: no cover - reached only on a regression
        raise AssertionError("a plan was built inside a capture")

    monkeypatch.setattr(dispatch, "_build_plan", _boom)
    xq = types.SimpleNamespace(device = 0, shape = (512, 1536))
    assert dispatch.gemm_plan(xq, None, None, None, None, None, 3072, "cutlass") is None

    dispatch._GEMM_PLAN[(512, 1536, 3072, "cutlass", 0)] = ("runner", 7, "ws")
    assert dispatch.gemm_plan(xq, None, None, None, None, None, 3072, "cutlass") == (
        "runner",
        7,
        "ws",
    )


def test_a_backend_other_than_cutlass_has_no_cached_plan(monkeypatch):
    monkeypatch.setattr(ops, "_device_index", lambda device: 0)
    monkeypatch.setattr(dispatch, "enabled", lambda device: True)
    monkeypatch.setattr(ops, "_is_capturing", lambda: False)
    xq = types.SimpleNamespace(device = 0, shape = (512, 1536))
    assert dispatch.gemm_plan(xq, None, None, None, None, None, 3072, "trtllm") is None


def test_the_transpose_cache_holds_the_view_and_revalidates_pointer_and_shape():
    weight = _Ptr(1024)
    view = dispatch.transposed(weight)
    assert dispatch.transposed(weight) is view
    # A reallocated buffer at the same address with a different shape must not get the old view.
    other = _Ptr(1024, shape = (4, 8))
    assert dispatch.transposed(other) is not view
    # Nor may the SAME object whose storage was swapped under it.
    weight._pointer = 2048
    assert dispatch.transposed(weight) is not view


def test_the_transpose_cache_is_bounded():
    held = [_Ptr(pointer) for pointer in range(dispatch._TRANSPOSE_CACHE_MAX)]
    for weight in held:
        dispatch.transposed(weight)
    assert len(dispatch._TRANSPOSED) == dispatch._TRANSPOSE_CACHE_MAX
    last = _Ptr(10**9)
    dispatch.transposed(last)
    assert len(dispatch._TRANSPOSED) == 1


def test_the_transpose_cache_releases_a_weight_that_was_collected():
    """A load superseded between its prewarm and its commit returns without any reset, so the only
    thing that can free its weights is the cache letting go of them on its own."""
    import gc

    weight = _Ptr(4096)
    dispatch.transposed(weight)
    assert len(dispatch._TRANSPOSED) == 1
    del weight
    gc.collect()
    assert dispatch._TRANSPOSED == {}


def test_a_live_weight_survives_a_superseded_loads_collection():
    """The point of the weakref keying: dropping the stale load must not cost the replacement its
    own entries, which a blanket reset would."""
    import gc

    live = _Ptr(64)
    live_view = dispatch.transposed(live)
    stale = _Ptr(128)
    dispatch.transposed(stale)
    del stale
    gc.collect()
    assert len(dispatch._TRANSPOSED) == 1
    assert dispatch.transposed(live) is live_view


def test_reset_clears_every_cache(monkeypatch):
    _fake_flashinfer(monkeypatch)
    monkeypatch.setattr(ops, "_device_index", lambda device: int(device))
    dispatch.available()
    held = _Ptr(7)
    dispatch.transposed(held)
    dispatch._GEMM_PLAN[("k",)] = ("runner", 0, "ws")
    dispatch._QUANT_FN[0] = ("fn", True)
    dispatch._VERIFIED[0] = (True, "ok")
    assert dispatch.enabled(0) is True

    dispatch.reset()
    assert dispatch._AVAILABLE is None
    assert not dispatch._GEMM_PLAN and not dispatch._QUANT_FN
    assert not dispatch._TRANSPOSED and not dispatch._VERIFIED
    assert dispatch.enabled(0) is False


def test_describe_reports_what_is_cached(monkeypatch):
    _fake_flashinfer(monkeypatch)
    monkeypatch.setattr(ops, "_device_index", lambda device: int(device))
    dispatch._VERIFIED[1] = (True, "ok")
    dispatch._VERIFIED[2] = (False, "nope")
    held = _Ptr(3)
    dispatch.transposed(held)
    record = dispatch.describe()
    assert record["available"] is True
    assert record["verified_devices"] == [1]
    assert record["transposed"] == 1


def test_a_quantiser_built_for_a_failed_verify_is_never_handed_out(monkeypatch):
    """verify() builds the quantiser with force BEFORE it knows the answer, so a cache read ahead
    of the gate handed the failed device the very quantiser its verify rejected."""
    _fake_flashinfer(monkeypatch)
    monkeypatch.setattr(ops, "_device_index", lambda device: int(device))
    dispatch._QUANT_FN[0] = ("fn", True)
    dispatch._VERIFIED[0] = (False, "the cached quantiser is not bit-identical to nvfp4_quantize")
    assert dispatch.enabled(0) is False
    assert dispatch.quant_fn(0) is None
    x = types.SimpleNamespace(device = 0, is_contiguous = lambda: True)
    assert dispatch._fast_quantize(x, None) == (None, None)


def test_the_unload_reset_also_forgets_the_preflight_that_ran_verify():
    """verify() runs ONLY from inside the preflight, and the preflight is memoised per device: a
    reset that keeps it leaves the next load on flashinfer with the cached dispatch off."""
    from core.inference import diffusion_nvfp4_linear as nl
    try:
        ops._PREFLIGHT[0] = {"ok": True, "fast_dispatch": True}
        dispatch._VERIFIED[0] = (True, "ok")
        nl.reset_nvfp4_state()
        assert not ops._PREFLIGHT
        assert not dispatch._VERIFIED
    finally:
        ops.reset_preflight_cache()


def _block_after(source: str, marker: str, lines: int) -> str:
    return "\n".join(source[source.index(marker) :].splitlines()[:lines])


def test_an_aborted_load_drops_the_caches_that_pin_the_failed_model():
    """A load that reached nvfp4_prewarm and then failed leaves the transposed-weight cache holding
    a view of every warmed weight, and a view is not something clear_gpu_cache() can free."""
    import inspect

    from core.inference.diffusion import DiffusionBackend
    from core.inference.video import VideoBackend

    load = inspect.getsource(DiffusionBackend.load_pipeline)
    assert "reset_nvfp4_state()" in _block_after(load, "diffusion.transformer_quant_fallback", 30)
    assert "reset_nvfp4_state()" in _block_after(load, "if not state_committed:", 20)
    assert "reset_nvfp4_state()" in _block_after(
        inspect.getsource(VideoBackend._run_load), "video.load_failed", 15
    )


CUDA_SHAPES = (
    (32, 2560, 3840),
    (1056, 3840, 3840),
    (1056, 3840, 10240),
    (4128, 3840, 10240),
    (27280, 3072, 14336),
)


def _cuda_or_skip():
    torch = pytest.importorskip("torch")
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if tuple(torch.cuda.get_device_capability(0)) not in ops.NVFP4_FLASHINFER_CAPS:
        pytest.skip("this device has no flashinfer NVFP4 kernels")
    pytest.importorskip("flashinfer")
    return torch


def test_the_real_flashinfer_verifies_bit_identical():
    torch = _cuda_or_skip()
    import flashinfer

    if flashinfer.__version__ not in dispatch._SUPPORTED:
        pytest.skip(f"flashinfer {flashinfer.__version__} is outside the allowlist by design")
    ok, reason = dispatch.verify(torch.device("cuda", 0))
    assert ok is True, reason
    assert dispatch.enabled(torch.device("cuda", 0)) is True


@pytest.mark.parametrize("m,k,n", CUDA_SHAPES)
def test_the_cached_dispatch_matches_the_public_api_exactly(m, k, n):
    torch = _cuda_or_skip()
    import flashinfer

    if flashinfer.__version__ not in dispatch._SUPPORTED:
        pytest.skip(f"flashinfer {flashinfer.__version__} is outside the allowlist by design")

    device = torch.device("cuda", 0)
    torch.manual_seed(m + k + n)
    with torch.cuda.device(device), torch.inference_mode():
        x = torch.randn(m, k, device = device, dtype = torch.bfloat16) * 0.05
        w = torch.randn(n, k, device = device, dtype = torch.bfloat16) * 0.02
        a_gsf, w_gsf = ops.global_scale(x), ops.global_scale(w)
        wq, w_sf = flashinfer.nvfp4_quantize(w, w_gsf, do_shuffle = False)
        alpha = (1.0 / (a_gsf * w_gsf)).float()

        dispatch.reset()
        assert dispatch.enabled(device) is False
        want_q, want_sf = ops._quantize_impl(x, a_gsf)
        want = ops._mm_impl(want_q, wq, want_sf, w_sf, alpha, n, ops.DEFAULT_MM_BACKEND)

        assert dispatch.verify(device)[0] is True
        got_q, got_sf = ops._quantize_impl(x, a_gsf)
        got = ops._mm_impl(got_q, wq, got_sf, w_sf, alpha, n, ops.DEFAULT_MM_BACKEND)
        torch.cuda.synchronize(device)

    assert torch.equal(want_q, got_q) and torch.equal(want_sf, got_sf)
    assert torch.equal(want, got), float((want.float() - got.float()).abs().max())
    assert (m, k // 2, n, ops.DEFAULT_MM_BACKEND, 0) in dispatch._GEMM_PLAN


def test_the_preflight_unlocks_the_fast_dispatch_and_says_so():
    torch = _cuda_or_skip()
    import flashinfer

    if flashinfer.__version__ not in dispatch._SUPPORTED:
        pytest.skip(f"flashinfer {flashinfer.__version__} is outside the allowlist by design")

    ops.reset_preflight_cache()
    dispatch.reset()
    record = ops.nvfp4_preflight(0, refresh = True)
    assert record["ok"] is True, record["reason"]
    assert record["fast_dispatch"] is True, record["fast_dispatch_reason"]
    assert dispatch.enabled(torch.device("cuda", 0)) is True
    ops.reset_preflight_cache()


def test_a_collected_layer_takes_its_cached_weights_and_their_vram_with_it():
    """The superseded-load case: the worker returns at its token check without ever reaching a
    reset, so the cache has to release the dead layer by itself, VRAM included."""
    torch = _cuda_or_skip()
    import gc

    import flashinfer

    from core.inference import diffusion_nvfp4_linear as nl

    if flashinfer.__version__ not in dispatch._SUPPORTED:
        pytest.skip(f"flashinfer {flashinfer.__version__} is outside the allowlist by design")

    device = torch.device("cuda", 0)
    dispatch.reset()
    assert dispatch.verify(device)[0] is True
    ops.register_ops()

    in_features, out_features = 3072, 18432
    torch.manual_seed(0)
    with torch.cuda.device(device), torch.inference_mode():
        w = torch.randn(out_features, in_features, device = device, dtype = torch.bfloat16) * 0.02
        x = torch.randn(64, in_features, device = device, dtype = torch.bfloat16) * 0.05
        a_gsf, w_gsf = ops.global_scale(x), ops.global_scale(w)
        wq, w_sf = flashinfer.nvfp4_quantize(w, w_gsf, do_shuffle = False)
        layer = nl.nvfp4_linear_class()(
            in_features,
            out_features,
            wq = wq.clone(),
            w_sf = w_sf.clone(),
            alpha = (1.0 / (a_gsf * w_gsf)).float(),
            a_gsf = a_gsf,
        )
        del wq, w_sf, w
        gc.collect()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated(device)
        layer(x)
        torch.cuda.synchronize(device)

    # Both weight buffers are cached, and both entries are keyed on the buffer the layer holds.
    assert len(dispatch._TRANSPOSED) == 2
    assert {id(layer.wq), id(layer.w_sf)} == set(dispatch._TRANSPOSED)
    weight_bytes = layer.wq.numel() + layer.w_sf.numel()

    del layer
    gc.collect()
    assert dispatch._TRANSPOSED == {}
    torch.cuda.empty_cache()
    freed = before - torch.cuda.memory_allocated(device)
    assert freed >= weight_bytes, f"{freed} bytes released, expected at least {weight_bytes}"
