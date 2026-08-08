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
    monkeypatch.setattr(probe, "_device_compute_capabilities", lambda: ())


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
    monkeypatch.setattr(probe, "_device_compute_capabilities", lambda: ((12, 0),))

    entry = probe.probe_xformers()
    assert entry["imports"] is True
    assert entry["runs"] is False
    assert "no memory-efficient attention kernel" in entry["error"]
    assert "12.0" in entry["error"]


def test_a_kernel_that_covers_this_gpu_is_unknown_not_working(monkeypatch):
    """The op table admitting this GPU is not evidence the build ships a kernel image for it.

    CUDA_MINIMUM/MAXIMUM_COMPUTE_CAPABILITY are class constants describing what the op
    supports in principle. A source build with TORCH_CUDA_ARCH_LIST set for other
    architectures, or a wheel that dropped one, registers the very same op and fails the
    first launch with "no kernel image is available". Establishing coverage needs a launch,
    which this child does not do, so the honest answer is the one probe_flash_attn gives."""
    _install_xformers(monkeypatch, [_op(maximum = (9, 0)), _op(minimum = (10, 0))])
    monkeypatch.setattr(probe, "_device_compute_capabilities", lambda: ((12, 0),))

    entry = probe.probe_xformers()
    assert entry["runs"] is None
    assert "no kernel was launched" in entry["error"]


def test_an_unknown_capability_leaves_the_load_status_alone(monkeypatch):
    # No nvidia-smi, no answer. "Cannot be checked" must not become "broken" -- nor "Working".
    _install_xformers(monkeypatch, [_op(maximum = (9, 0))])

    entry = probe.probe_xformers()
    assert entry["runs"] is None
    assert "compute capability could not be read" in entry["error"]


def test_an_unrecognised_op_table_leaves_the_load_status_alone(monkeypatch):
    # A future xformers that renames ALL_FW_OPS must not be reported as broken.
    _install_xformers(monkeypatch, [])
    monkeypatch.setattr(probe, "_device_compute_capabilities", lambda: ((12, 0),))

    entry = probe.probe_xformers()
    assert entry["runs"] is None
    assert "could not be enumerated" in entry["error"]


def test_a_cpp_lib_that_raises_on_import_is_broken_not_unknown(monkeypatch):
    """The parent adds a package to `degraded` on imports=False or runs=False only, so a
    native load error raised by _cpp_lib itself showed as "Not checked" with no banner --
    on precisely the corrupt install this report exists to name."""
    import builtins

    _install_xformers(monkeypatch, [_op(minimum = (7, 0))])
    real_import = builtins.__import__

    def _raise_for_cpp_lib(name, *args, **kwargs):
        if name == "xformers._cpp_lib" or (name == "xformers" and "_cpp_lib" in (args[2] or ())):
            raise OSError("libc10.so: undefined symbol")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "xformers._cpp_lib")
    monkeypatch.setattr(builtins, "__import__", _raise_for_cpp_lib)
    entry = probe.probe_xformers()
    assert entry["imports"] is True
    assert entry["runs"] is False
    assert "undefined symbol" in entry["error"]


def test_a_layout_without_a_cpp_lib_at_all_is_still_unknown(monkeypatch):
    # A future rename is not a dead install, so ModuleNotFoundError stays unknown.
    import builtins

    _install_xformers(monkeypatch, [_op(minimum = (7, 0))])
    real_import = builtins.__import__

    def _missing(name, *args, **kwargs):
        if name == "xformers._cpp_lib" or (name == "xformers" and "_cpp_lib" in (args[2] or ())):
            raise ModuleNotFoundError("No module named 'xformers._cpp_lib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "xformers._cpp_lib")
    monkeypatch.setattr(builtins, "__import__", _missing)
    assert probe.probe_xformers()["runs"] is None


def test_an_unresolvable_mask_does_not_fall_back_to_the_whole_box(monkeypatch):
    """The parent answers None when it cannot resolve a numeric mask, and used to serialize
    that as an empty override -- which this child reads as no override at all and answers
    from nvidia-smi over every physical GPU, the exact verdict the parent declined to give."""
    import shutil as _shutil
    import subprocess as _subprocess

    monkeypatch.setenv("UNSLOTH_PROBE_DEVICE_CC", probe._CC_UNKNOWN)
    monkeypatch.setattr(
        _shutil, "which", lambda name: pytest.fail("the child must not look for nvidia-smi")
    )
    monkeypatch.setattr(
        _subprocess, "run", lambda *a, **k: pytest.fail("the child must not ask nvidia-smi")
    )
    assert probe._device_compute_capabilities() == ()


def test_a_failed_library_load_still_wins(monkeypatch):
    _install_xformers(monkeypatch, [_op(minimum = (7, 0))])
    sys.modules["xformers._cpp_lib"]._cpp_library_load_exception = OSError("undefined symbol")
    monkeypatch.setattr(probe, "_device_compute_capabilities", lambda: ((12, 0),))

    entry = probe.probe_xformers()
    assert entry["runs"] is False and "undefined symbol" in entry["error"]


def test_every_visible_gpu_has_to_have_a_kernel(monkeypatch):
    """CUDA_VISIBLE_DEVICES=0,1 across a mixed pair. The rank that lands on the sm_120 card
    falls back to SDPA whatever the sm_90 card can do, so a verdict taken from the first
    visible GPU is the same false all-clear in a smaller box."""
    _install_xformers(monkeypatch, [_op(maximum = (9, 0))])
    monkeypatch.setattr(probe, "_device_compute_capabilities", lambda: ((9, 0), (12, 0)))

    entry = probe.probe_xformers()
    assert entry["runs"] is False
    assert "12.0" in entry["error"], "the report must name the GPU that is not covered"


def test_a_pair_the_build_covers_is_unknown_not_working(monkeypatch):
    # Covering both cards clears the proven-broken verdict; it does not earn "Working".
    _install_xformers(monkeypatch, [_op(minimum = (7, 0))])
    monkeypatch.setattr(probe, "_device_compute_capabilities", lambda: ((9, 0), (12, 0)))
    assert probe.probe_xformers()["runs"] is None


@pytest.mark.parametrize("capabilities", [((12, 0),), ((9, 0),), ((8, 6),), ()])
def test_flash_attn_kernel_coverage_is_unknown_without_a_launch(monkeypatch, capabilities):
    """Importing the extension is not the question, on any card.

    A build with no cubin or PTX image for this architecture imports fine and fails its
    first launch with "no kernel image is available" -- true of source builds and of older
    wheels, not only of the sm_100+ cards our installer refuses to fetch a wheel for.
    flash-attn exposes no list of the architectures it was compiled for, and this child
    never launches a kernel, so nothing here can establish support."""
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))
    monkeypatch.setitem(sys.modules, "flash_attn.flash_attn_interface", types.ModuleType("iface"))
    monkeypatch.setattr(probe, "_device_compute_capabilities", lambda: capabilities)

    entry = probe.probe_flash_attn()
    assert entry["imports"] is True
    assert entry["runs"] is None, "unknown, not a verdict either way"
    assert "without launching one" in entry["error"]


def test_a_flash_attn_that_cannot_import_is_still_broken(monkeypatch):
    # Unknown is for "cannot be established", never for "it raised". A None entry in
    # sys.modules is how the import system spells a submodule that will not load.
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))
    monkeypatch.setitem(sys.modules, "flash_attn.flash_attn_interface", None)

    entry = probe.probe_flash_attn()
    assert entry["imports"] is True
    assert entry["runs"] is False and entry["error"]


def test_the_capability_is_parsed_from_the_override(monkeypatch):
    # Reload past the autouse stub: this one is about the real reader.
    import importlib
    fresh = importlib.reload(probe)
    try:
        monkeypatch.setenv("UNSLOTH_PROBE_DEVICE_CC", "12.0")
        assert fresh._device_compute_capabilities() == ((12, 0),)
        # The parent sends every visible one, comma separated; the child cannot re-resolve
        # the mask because the parent cleared it.
        monkeypatch.setenv("UNSLOTH_PROBE_DEVICE_CC", "9.0,12.0")
        assert fresh._device_compute_capabilities() == ((9, 0), (12, 0))
        monkeypatch.setenv("UNSLOTH_PROBE_DEVICE_CC", "not a capability")
        assert fresh._device_compute_capabilities() == ()
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


def test_torchao_without_native_operators_is_unknown_not_broken(monkeypatch):
    """The dispatcher's table, NOT dir(torch.ops.torchao): touching that attribute creates an
    empty _OpNamespace whose dir() already lists __name__, __spec__ and friends, so the
    no-native-operators case -- the one this exists to catch -- read as healthy.

    And the verdict is UNKNOWN, not broken: the managed stack pins torchao 0.17 on torch
    2.10+cu130 knowing its extension is cleanly skipped, and torchao keeps working through
    its Python fallbacks. Calling that degraded would put a destructive banner on the
    standard configuration."""
    monkeypatch.setitem(sys.modules, "torchao", types.ModuleType("torchao"))
    monkeypatch.setattr(probe, "_registered_ops", lambda namespace: 0)

    entry = probe.probe_torchao()
    assert entry["imports"] is True
    assert entry["runs"] is None
    assert "registered no native operators" in entry["error"]


def test_torchao_with_native_operators_is_working(monkeypatch):
    monkeypatch.setitem(sys.modules, "torchao", types.ModuleType("torchao"))
    monkeypatch.setattr(probe, "_registered_ops", lambda namespace: 12)

    entry = probe.probe_torchao()
    assert entry["runs"] is True and entry["error"] is None


def test_an_empty_ops_namespace_is_not_mistaken_for_a_loaded_extension():
    """Against the real torch in this environment: torch.ops.<ns> materialises on attribute
    access, so the count has to come from the dispatcher."""
    pytest.importorskip("torch")
    import torch

    assert probe._registered_ops("aten") > 0
    # A namespace nothing has registered under. Reading it through torch.ops would create it.
    assert probe._registered_ops("unsloth_no_such_namespace") == 0
    assert len(dir(getattr(torch.ops, "unsloth_no_such_namespace"))) > 0


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
    monkeypatch.setattr(probe, "_device_compute_capabilities", lambda: ((12, 0),))

    entry = probe.probe_xformers()
    assert entry["runs"] is False and "no memory-efficient attention kernel" in entry["error"]
