# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Can bitsandbytes actually run a 4bit kernel on this host?

A successful `import bitsandbytes` does not answer that. From 0.46 onwards a wheel
whose native library never loaded still imports, and `BNBNativeLibrary.__getattr__`
hands back a plain `throw_on_call` closure for every symbol - a dead library is
replaced wholesale by `ErrorHandlerMockBNBNativeLibrary`, which does the same. So
the ctypes handles `kernels/utils.py` binds at import time all resolve, nothing
raises, `ALLOW_BITSANDBYTES` stays true, the loader selects a 4bit checkpoint, and
the run dies inside a kernel with "Method 'cdequantize_blockwise_fp32' is not
available in CPU-only version" instead of falling back to 16bit up front.

A real handle is a ctypes function pointer and carries `restype`; a deferred failure
is a Python function and does not. That is the whole test.

Deliberately narrow. This decides the capability flags only - it never decides
whether the module is importable, because these shapes import perfectly well and
treating them as absent would disable a wheel whose Python side works. A CPU-only
install is exactly that shape.

A leaf module on purpose: it imports nothing from unsloth (device_type.py is
imported very early and would be a cycle) and takes the device type as an argument.
"""

__all__ = [
    "bitsandbytes_symbols",
    "check_native_kernels",
    "native_kernels_ready",
]

# The ctypes handles kernels/utils.py binds at module scope. Keep in step with the
# `bnb.functional.lib.*` reads there - a test asserts the two match.
_C_SYMBOLS = (
    "cdequantize_blockwise_fp32",
    "cdequantize_blockwise_fp16_nf4",
    "cdequantize_blockwise_bf16_nf4",
)
# 4bit inference is a gemv on xpu and a naive gemm everywhere else, so probing the
# xpu names on cuda would write off a perfectly good wheel.
_C_SYMBOLS_XPU = (
    "cgemv_4bit_inference_fp16",
    "cgemv_4bit_inference_bf16",
)
_C_SYMBOLS_GEMM = (
    "cgemm_4bit_inference_naive_fp16",
    "cgemm_4bit_inference_naive_bf16",
)


def bitsandbytes_symbols(device_type):
    """Names kernels/utils.py reads off `bitsandbytes.functional.lib`."""
    tail = _C_SYMBOLS_XPU if device_type == "xpu" else _C_SYMBOLS_GEMM
    return _C_SYMBOLS + tail


def check_native_kernels(bnb, device_type):
    """Raise unless every 4bit handle on `bnb` is a real native kernel.

    Safe to repeat: ctypes caches the function object on the first lookup and
    bitsandbytes memoizes its wrapper, so these are the handles bound later.
    """
    if bnb is None:
        raise ImportError("Unsloth: `bitsandbytes` is not installed.")
    functional = getattr(bnb, "functional", None)
    if functional is None:
        # A bitsandbytes whose __init__ died part way through leaves the parent without
        # the attribute while the submodule stays in sys.modules, and `import x.y as z`
        # reads sys.modules directly where plain attribute access does not.
        import bitsandbytes.functional as functional

    lib = functional.lib  # None on a 0.45.5 native-load failure
    for symbol in bitsandbytes_symbols(device_type):
        if not hasattr(getattr(lib, symbol), "restype"):
            raise AttributeError(
                f"Unsloth: `bitsandbytes.functional.lib.{symbol}` is not a native "
                "handle - the bitsandbytes native library did not load."
            )


def native_kernels_ready(bnb, device_type):
    """Can 4bit actually run here? Gates the capability flags, never the import."""
    try:
        check_native_kernels(bnb, device_type)
    except Exception:
        return False
    return True
