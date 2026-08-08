# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The probe child must answer "does this actually work", not "did it import".

Three packages can import cleanly while the thing they exist for is absent, and all three
rendered as "Working" in Settings:

* bitsandbytes >= 0.46 hands back a ``throw_on_call`` closure for every ctypes symbol when
  its native library did not load, so 4-bit dies mid-run instead of falling back;
* torchao imports with its C++ extension "cleanly skipped" on a torch it has no build for
  (the Python stack pins 0.17.0 on torch 2.10+cu130 for exactly that reason), leaving no
  optimized quantization kernels;
* xformers loads its library on a GPU it ships no attention kernel for, and every call is
  capability-rejected back to SDPA -- the degraded state this whole report exists to name.

Hermetic: the packages are stubbed through ``sys.modules``, so none of this needs a GPU, a
real xformers, or a working bitsandbytes.
"""

from __future__ import annotations

import sys
import types

import pytest

import utils.hardware.accelerator_probe as probe


@pytest.fixture(autouse = True)
def _no_real_capability(monkeypatch):
    # Never shell out to nvidia-smi from a unit test; each test says what the host is.
    monkeypatch.delenv("UNSLOTH_PROBE_DEVICE_CC", raising = False)
    monkeypatch.setattr(probe, "_device_compute_capability", lambda: None)


def _op(
    *,
    minimum = None,
    maximum = None,
    operator = object(),
):
    op = types.SimpleNamespace(OPERATOR = operator)
    if minimum is not None:
        op.CUDA_MINIMUM_COMPUTE_CAPABILITY = minimum
    if maximum is not None:
        op.CUDA_MAXIMUM_COMPUTE_CAPABILITY = maximum
    return op


def _install_xformers(monkeypatch, ops):
    xformers = types.ModuleType("xformers")
    cpp_lib = types.ModuleType("xformers._cpp_lib")
    cpp_lib._cpp_library_load_exception = None
    ops_mod = types.ModuleType("xformers.ops")
    fmha = types.ModuleType("xformers.ops.fmha")
    fmha.ALL_FW_OPS = ops
    ops_mod.fmha = fmha
    xformers._cpp_lib = cpp_lib
    xformers.ops = ops_mod
    for name, module in (
        ("xformers", xformers),
        ("xformers._cpp_lib", cpp_lib),
        ("xformers.ops", ops_mod),
        ("xformers.ops.fmha", fmha),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def test_a_loaded_xformers_with_no_kernel_for_this_gpu_is_not_working(monkeypatch):
    # sm_120 (RTX 50-series) against a build whose ops all cap at sm_90: the library loads,
    # _cpp_library_load_exception is None, and attention silently runs on SDPA.
    _install_xformers(monkeypatch, [_op(maximum = (9, 0)), _op(minimum = (10, 0), operator = None)])
    monkeypatch.setattr(probe, "_device_compute_capability", lambda: (12, 0))

    entry = probe.probe_xformers()
    assert entry["imports"] is True
    assert entry["runs"] is False
    assert "no memory-efficient attention kernel" in entry["error"]
    assert "12.0" in entry["error"]


def test_a_kernel_that_covers_this_gpu_still_reports_working(monkeypatch):
    _install_xformers(monkeypatch, [_op(maximum = (9, 0)), _op(minimum = (10, 0))])
    monkeypatch.setattr(probe, "_device_compute_capability", lambda: (12, 0))

    entry = probe.probe_xformers()
    assert entry["runs"] is True and entry["error"] is None


def test_an_unknown_capability_leaves_the_load_status_alone(monkeypatch):
    # No nvidia-smi, no answer. "Cannot be checked" must not become "broken".
    _install_xformers(monkeypatch, [_op(maximum = (9, 0))])

    entry = probe.probe_xformers()
    assert entry["runs"] is True and entry["error"] is None


def test_an_unrecognised_op_table_leaves_the_load_status_alone(monkeypatch):
    # A future xformers that renames ALL_FW_OPS must not be reported as broken.
    _install_xformers(monkeypatch, [])
    monkeypatch.setattr(probe, "_device_compute_capability", lambda: (12, 0))

    assert probe.probe_xformers()["runs"] is True


def test_a_failed_library_load_still_wins(monkeypatch):
    _install_xformers(monkeypatch, [_op(minimum = (7, 0))])
    sys.modules["xformers._cpp_lib"]._cpp_library_load_exception = OSError("undefined symbol")
    monkeypatch.setattr(probe, "_device_compute_capability", lambda: (12, 0))

    entry = probe.probe_xformers()
    assert entry["runs"] is False and "undefined symbol" in entry["error"]


def test_the_capability_is_parsed_from_the_override(monkeypatch):
    # Reload past the autouse stub: this one is about the real reader.
    import importlib
    fresh = importlib.reload(probe)
    try:
        monkeypatch.setenv("UNSLOTH_PROBE_DEVICE_CC", "12.0")
        assert fresh._device_compute_capability() == (12, 0)
        monkeypatch.setenv("UNSLOTH_PROBE_DEVICE_CC", "not a capability")
        assert fresh._device_compute_capability() is None
    finally:
        importlib.reload(probe)


def test_bitsandbytes_with_dead_native_handles_is_not_working(monkeypatch):
    # The 0.46+ shape: every symbol resolves to a plain closure that raises when called,
    # so attribute reads see a healthy wheel and 4-bit dies inside a kernel.
    def throw_on_call(*args, **kwargs):
        raise RuntimeError("native library not loaded")

    lib = types.SimpleNamespace(
        cdequantize_blockwise_fp32 = throw_on_call,
        cdequantize_blockwise_fp16_nf4 = throw_on_call,
        cdequantize_blockwise_bf16_nf4 = throw_on_call,
        cgemm_4bit_inference_naive_fp16 = throw_on_call,
        cgemm_4bit_inference_naive_bf16 = throw_on_call,
    )
    bnb = types.ModuleType("bitsandbytes")
    functional = types.ModuleType("bitsandbytes.functional")
    functional.lib = lib
    bnb.functional = functional
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)
    monkeypatch.setitem(sys.modules, "bitsandbytes.functional", functional)

    entry = probe.probe_bitsandbytes()
    assert entry["imports"] is True
    assert entry["runs"] is False
    assert "native library did not load" in entry["error"]


def test_bitsandbytes_with_real_handles_is_working(monkeypatch):
    real = types.SimpleNamespace(restype = None)
    lib = types.SimpleNamespace(
        cdequantize_blockwise_fp32 = real,
        cdequantize_blockwise_fp16_nf4 = real,
        cdequantize_blockwise_bf16_nf4 = real,
        cgemm_4bit_inference_naive_fp16 = real,
        cgemm_4bit_inference_naive_bf16 = real,
    )
    bnb = types.ModuleType("bitsandbytes")
    functional = types.ModuleType("bitsandbytes.functional")
    functional.lib = lib
    bnb.functional = functional
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)
    monkeypatch.setitem(sys.modules, "bitsandbytes.functional", functional)

    entry = probe.probe_bitsandbytes()
    assert entry["runs"] is True and entry["error"] is None


def test_the_bitsandbytes_check_is_the_one_the_loader_gates_on():
    # Loaded by path, not reimplemented: the report must not be able to say "Working"
    # about a wheel unsloth's own loader has already written off.
    module = probe._load_bnb_availability()
    assert module is not None
    assert hasattr(module, "check_native_kernels")


def test_torchao_without_native_operators_is_not_working(monkeypatch):
    torchao = types.ModuleType("torchao")
    monkeypatch.setitem(sys.modules, "torchao", torchao)
    torch = types.ModuleType("torch")
    torch.ops = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", torch)

    entry = probe.probe_torchao()
    assert entry["imports"] is True
    assert entry["runs"] is False
    assert "registered no native operators" in entry["error"]


def test_torchao_with_native_operators_is_working(monkeypatch):
    torchao = types.ModuleType("torchao")
    monkeypatch.setitem(sys.modules, "torchao", torchao)

    class _Namespace:
        def __dir__(self):
            return ["quant_llm_linear"]

    torch = types.ModuleType("torch")
    torch.ops = types.SimpleNamespace(torchao = _Namespace())
    monkeypatch.setitem(sys.modules, "torch", torch)

    entry = probe.probe_torchao()
    assert entry["runs"] is True and entry["error"] is None


def test_the_dispatch_table_uses_the_kernel_aware_probes():
    # The parent only ever calls through PROBES; a table entry left on the plain import is
    # the same false all-clear with the checks sitting unused next to it.
    assert probe.PROBES["bitsandbytes"] is probe.probe_bitsandbytes
    assert probe.PROBES["torchao"] is probe.probe_torchao
    assert probe.PROBES["xformers"] is probe.probe_xformers


def test_the_kernel_verdict_also_applies_on_the_register_extensions_path(monkeypatch):
    # The older xformers layout has no _cpp_library_load_exception and is probed by
    # re-registering the extensions. Same question, same answer required.
    _install_xformers(monkeypatch, [_op(maximum = (9, 0))])
    cpp_lib = sys.modules["xformers._cpp_lib"]
    del cpp_lib._cpp_library_load_exception
    cpp_lib._register_extensions = lambda: None
    monkeypatch.setattr(probe, "_device_compute_capability", lambda: (12, 0))

    entry = probe.probe_xformers()
    assert entry["runs"] is False and "no memory-efficient attention kernel" in entry["error"]
