# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Can bitsandbytes actually run a 4bit kernel here? A successful import does not say.

From 0.46 a wheel whose native library never loaded still imports and hands back a
`throw_on_call` closure for every symbol, so attribute reads alone see a healthy wheel,
`ALLOW_BITSANDBYTES` stays true and 4bit dies inside a kernel instead of falling back to
16bit up front. A real handle is a ctypes function pointer and carries `restype`; a
deferred failure is a plain Python function and does not. That is the whole test, applied
to every probed handle: the same verdict gates the module-scope binds in kernels/utils.py,
where one bad symbol is the crash this exists to prevent.

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
# 4bit inference is a gemv on xpu and a naive gemm elsewhere; probing the wrong pair would write off
# a perfectly good wheel.
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
    """Raise unless every handle kernels/utils.py is about to bind is a real kernel.

    All of them: one that resolves here but not at the bind gives back the AttributeError
    this prevents. Partial export costs 8bit too (`ALLOW_BITSANDBYTES` gates both), but a
    wheel missing a symbol is a shape no flag makes safe. Safe to repeat - ctypes caches
    each handle on first lookup, so these are the ones bound later.
    """
    if bnb is None:
        raise ImportError("Unsloth: `bitsandbytes` is not installed.")
    functional = getattr(bnb, "functional", None)
    if functional is None:
        # A part-initialised bitsandbytes leaves the parent without the attribute while the submodule stays
        # in sys.modules, which `import x.y as z` reads directly.
        import bitsandbytes.functional as functional

    lib = functional.lib
    if lib is None:
        # 0.45.5, the floor in pyproject.toml, on a native-load failure.
        raise AttributeError("Unsloth: `bitsandbytes.functional.lib` is None.")
    for symbol in bitsandbytes_symbols(device_type):
        handle = getattr(lib, symbol)  # AttributeError here is itself a failed check
        if not hasattr(handle, "restype"):
            raise AttributeError(
                f"Unsloth: `bitsandbytes.functional.lib.{symbol}` is not a native "
                "function pointer - the bitsandbytes native library did not load."
            )


def native_kernels_ready(bnb, device_type):
    """Is the bitsandbytes native library alive? Gates the flags, never the import."""
    try:
        check_native_kernels(bnb, device_type)
    except Exception:
        return False
    return True
