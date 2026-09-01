# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`ALLOW_BITSANDBYTES` must follow the kernels, not the mere presence of the module.

From bitsandbytes 0.46 a wheel whose native library never loaded still imports and
resolves every ctypes handle to a `throw_on_call` closure, so a probe made of attribute
reads alone sees a healthy wheel, the loader selects a 4bit checkpoint, and the failure
lands inside a kernel mid-run instead of degrading to 16bit.
"""

from __future__ import annotations

import ast
import importlib.util
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_probe():
    """Import by path, not as ``unsloth.bnb_availability``, which would run the package
    __init__ and pull in torch. Works only because the module is a leaf - the property
    that lets device_type.py, imported very early, use it without a cycle."""
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
    """A probe that misses one of the import-time binds lets a dead wheel through."""
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
    # xpu probes the gemv pair, every other device the naive gemm pair, never both.
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


def test_a_partially_exporting_library_is_not_ready():
    """One resolvable symbol is not enough: the same verdict gates the module-scope
    binds, so a partial library would pass here and raise `AttributeError` at the bind."""

    class _MissingOne(_RealHandleLib):
        def __getattr__(self, name):
            if name == "cgemm_4bit_inference_naive_bf16":
                raise AttributeError(name)
            return super().__getattr__(name)

    probe = _load_probe()
    assert probe.native_kernels_ready(_fake_bnb(_MissingOne()), "cuda") is False


def test_one_dead_handle_among_live_ones_is_not_ready():
    """The realistic partial shape: the library loaded but one symbol is a closure."""

    class _OneDeferred(_RealHandleLib):
        def __getattr__(self, name):
            if name == "cdequantize_blockwise_bf16_nf4":
                return lambda *a, **k: None
            return super().__getattr__(name)

    probe = _load_probe()
    assert probe.native_kernels_ready(_fake_bnb(_OneDeferred()), "cuda") is False


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


def test_the_ctypes_binds_are_gated_on_the_same_verdict():
    """Clearing the flag is not enough on its own: ``bnb is None`` alone let an
    importable-but-dead wheel reach the binds, and 0.45.5 sets ``functional.lib = None``
    on a native-load failure, so they killed ``import unsloth`` outright instead of
    degrading to 16bit."""
    source = (REPO_ROOT / "unsloth" / "kernels" / "utils.py").read_text(encoding = "utf-8")
    assert "from ..bnb_availability import native_kernels_ready" in source
    assert (
        "if bnb is None or not native_kernels_ready(bnb, DEVICE_TYPE):" in source
    ), "the ctypes bind block must take the _bnb_required branch on a dead library too"
    guarded = source.split("if bnb is None or not native_kernels_ready(bnb, DEVICE_TYPE):")[1]
    # Anchor on the symbol, not the module alias: #7580 renamed the binding from `bnb.functional.lib` to
    # `bnb_functional.lib`, which is exactly the kind of rename this assertion should survive.
    assert "lib.cdequantize_blockwise_fp32" in guarded, "the binds must sit under that guard"


def test_the_kernel_check_reads_the_submodule_not_the_parent_attribute():
    """A part-initialised bitsandbytes leaves the parent without ``functional`` while
    the submodule stays in sys.modules, which ``import bitsandbytes.functional`` reads
    directly."""
    probe = _load_probe()
    bnb = types.ModuleType("bitsandbytes")
    bnb.__version__ = "0.50.0"
    import sys

    real = sys.modules.get("bitsandbytes.functional")
    if real is None:
        return  # bitsandbytes not importable here;
    assert probe.native_kernels_ready(bnb, "cuda") in (True, False)
