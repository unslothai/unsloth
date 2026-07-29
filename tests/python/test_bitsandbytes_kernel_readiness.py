# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`ALLOW_BITSANDBYTES` must follow the kernels, not the mere presence of the module.

From bitsandbytes 0.46 a wheel whose native library never loaded still imports and
resolves every ctypes handle: `BNBNativeLibrary.__getattr__` returns a `throw_on_call`
closure, and a dead library is replaced wholesale by `ErrorHandlerMockBNBNativeLibrary`,
which does the same for every name. Nothing raises while `kernels/utils.py` binds them,
so a probe made of attribute reads alone sees a healthy wheel, the loader selects a 4bit
checkpoint, and the failure lands inside a kernel mid-run instead of degrading to 16bit.
"""

from __future__ import annotations

import ast
import importlib.util
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_probe():
    """Import unsloth/bnb_availability.py by path, not as ``unsloth.bnb_availability``,
    which would run the package __init__ and pull in torch. Works only because the
    module is a leaf - the property that lets device_type.py, imported very early, use
    it without a cycle."""
    path = REPO_ROOT / "unsloth" / "bnb_availability.py"
    spec = importlib.util.spec_from_file_location("_unsloth_bnb_availability", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_bnb(lib):
    functional = types.ModuleType("bitsandbytes.functional")
    functional.get_ptr = lambda tensor: None
    functional.lib = lib
    bnb = types.ModuleType("bitsandbytes")
    bnb.__version__ = "0.50.0"
    bnb.functional = functional
    return bnb


class _DeferredFailureLib:
    """What bitsandbytes >= 0.46 hands back when the native library is dead."""

    def __getattr__(self, name):
        def throw_on_call(*args, **kwargs):
            raise RuntimeError(f"Method '{name}' not available in CPU-only version")

        return throw_on_call


class _RealHandleLib:
    """ctypes caches the function object on first lookup; its handles carry restype."""

    def __getattr__(self, name):
        def handle(*args, **kwargs):
            return None

        handle.restype = None
        setattr(self, name, handle)
        return handle


def test_probe_covers_every_module_scope_ctypes_bind():
    """kernels/utils.py binds these off ``bnb.functional.lib`` at import time, so a
    probe that misses one lets that shape through."""
    tree = ast.parse((REPO_ROOT / "unsloth" / "kernels" / "utils.py").read_text(encoding = "utf-8"))
    bound = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "lib"
    }
    probe = _load_probe()
    xpu = set(probe.bitsandbytes_symbols("xpu"))
    cuda = set(probe.bitsandbytes_symbols("cuda"))
    assert bound == xpu | cuda, f"probe and module-scope binds differ: {bound ^ (xpu | cuda)}"
    # xpu binds the gemv pair, every other device the naive gemm pair; neither probes
    # the other's names, which its wheel will not export.
    assert xpu - cuda and cuda - xpu, "the device split collapsed"


def test_a_deferred_failure_handle_is_not_ready():
    probe = _load_probe()
    bnb = _fake_bnb(_DeferredFailureLib())
    for device in ("cuda", "xpu"):
        assert probe.native_kernels_ready(bnb, device) is False, device


def test_a_real_ctypes_handle_is_ready():
    probe = _load_probe()
    bnb = _fake_bnb(_RealHandleLib())
    for device in ("cuda", "xpu"):
        assert probe.native_kernels_ready(bnb, device) is True, device


def test_a_lib_that_never_loaded_is_not_ready():
    """bitsandbytes 0.45.5, the floor in pyproject.toml, sets ``functional.lib = None``."""
    probe = _load_probe()
    assert probe.native_kernels_ready(_fake_bnb(None), "cuda") is False


def test_a_partially_exporting_library_stays_ready():
    """One missing symbol does not mean a dead library.

    ``ALLOW_BITSANDBYTES`` gates 8bit as well as 4bit - loader.py clears both - so
    writing the wheel off here would silently downgrade a working LLM.int8 request.
    The missing symbol raises where kernels/utils.py binds it, which is a crash no
    flag can rescue, not something to trade 8bit for.
    """

    class _MissingOne(_RealHandleLib):
        def __getattr__(self, name):
            if name == "cgemm_4bit_inference_naive_bf16":
                raise AttributeError(name)
            return super().__getattr__(name)

    probe = _load_probe()
    assert probe.native_kernels_ready(_fake_bnb(_MissingOne()), "cuda") is True


def test_absent_bitsandbytes_is_not_ready():
    probe = _load_probe()
    assert probe.native_kernels_ready(None, "cuda") is False


def test_device_type_gates_the_flags_on_the_kernels():
    """The flags must follow ``native_kernels_ready``, not the bare import."""
    head = (REPO_ROOT / "unsloth" / "device_type.py").read_text(encoding = "utf-8")
    head = head.split('if DEVICE_TYPE == "hip":')[0]
    assert "import bitsandbytes as _bnb_probe" in head
    assert 'find_spec("bitsandbytes")' not in head, "find_spec cannot see a broken wheel"
    assert "native_kernels_ready(_bnb_probe, DEVICE_TYPE)" in head
    assert (
        head.count("ALLOW_BITSANDBYTES = False") >= 2
    ), "both the failed-import path and the dead-kernels path must clear the flag"


def test_the_kernel_check_reads_the_submodule_not_the_parent_attribute():
    """A bitsandbytes whose __init__ died part way leaves the parent without
    ``functional`` while the submodule stays in sys.modules, so the check has to go
    through ``import bitsandbytes.functional``, which reads sys.modules directly."""
    probe = _load_probe()
    bnb = types.ModuleType("bitsandbytes")  # zombie: parent has no `functional`
    bnb.__version__ = "0.50.0"
    import sys

    real = sys.modules.get("bitsandbytes.functional")
    if real is None:
        return  # bitsandbytes not importable here; the fallback has nothing to read
    # Must not raise AttributeError on the missing parent attribute: it falls back
    # to the cached submodule and returns a verdict either way.
    assert probe.native_kernels_ready(bnb, "cuda") in (True, False)
