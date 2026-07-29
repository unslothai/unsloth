# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Can bitsandbytes actually run a 4bit kernel here? A successful import does not say.

From 0.46 a wheel whose native library never loaded still imports and hands back a
`throw_on_call` closure for every symbol, so attribute reads alone see a healthy wheel,
`ALLOW_BITSANDBYTES` stays true and 4bit dies inside a kernel instead of falling back to
16bit up front. A real handle is a ctypes function pointer and carries `restype`; a
deferred failure is a plain Python function and does not. That is the whole test.

Decides the capability flags only, never importability - a CPU-only install is exactly
this shape and its Python side works. A leaf module: imports nothing from unsloth
(device_type.py imports it very early, so anything else is a cycle) and takes the
device type as an argument.
"""

__all__ = [
    "bitsandbytes_symbols",
    "check_native_kernels",
    "native_kernels_ready",
]

# The ctypes handles kernels/utils.py binds at module scope; a test asserts they match.
_C_SYMBOLS = (
    "cdequantize_blockwise_fp32",
    "cdequantize_blockwise_fp16_nf4",
    "cdequantize_blockwise_bf16_nf4",
)
# 4bit inference is a gemv on xpu and a naive gemm elsewhere; probing the wrong pair
# would write off a perfectly good wheel.
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
    """Raise when `bnb`'s native library is dead: no handle it offers is a real kernel.

    Not "every handle resolves": a partially exporting library is alive, and
    `ALLOW_BITSANDBYTES` gates 8bit as well as 4bit (loader.py clears both), so writing
    it off would silently downgrade a working LLM.int8 request. Safe to repeat - ctypes
    caches each handle on first lookup, so these are the ones bound later.
    """
    if bnb is None:
        raise ImportError("Unsloth: `bitsandbytes` is not installed.")
    functional = getattr(bnb, "functional", None)
    if functional is None:
        # A part-initialised bitsandbytes leaves the parent without the attribute while
        # the submodule stays in sys.modules, which `import x.y as z` reads directly.
        import bitsandbytes.functional as functional

    lib = functional.lib
    if lib is None:
        # 0.45.5, the floor in pyproject.toml, on a native-load failure.
        raise AttributeError("Unsloth: `bitsandbytes.functional.lib` is None.")
    for symbol in bitsandbytes_symbols(device_type):
        try:
            handle = getattr(lib, symbol)
        except Exception:
            continue  # not exported: a partial library, not a dead one
        if hasattr(handle, "restype"):
            return  # a ctypes function pointer, so the library really loaded
    raise AttributeError(
        "Unsloth: no `bitsandbytes.functional.lib` 4bit handle is a native function "
        "pointer - the bitsandbytes native library did not load."
    )


def native_kernels_ready(bnb, device_type):
    """Is the bitsandbytes native library alive? Gates the flags, never the import."""
    try:
        check_native_kernels(bnb, device_type)
    except Exception:
        return False
    return True
