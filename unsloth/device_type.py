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

__all__ = [
    "is_hip",
    "is_mps",
    "get_device_type",
    "DEVICE_TYPE",
    "DEVICE_TYPE_TORCH",
    "DEVICE_COUNT",
    "ALLOW_PREQUANTIZED_MODELS",
    "ALLOW_BITSANDBYTES",
]

import torch
import functools
import inspect
from unsloth_zoo.utils import Version


@functools.cache
def is_hip():
    return bool(getattr(getattr(torch, "version", None), "hip", None))


@functools.cache
def is_mps():
    """Check if Apple Metal Performance Shaders (MPS) backend is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


@functools.cache
def get_device_type():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if is_hip():
            return "hip"
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    elif is_mps():
        return "mps"
    # Check torch.accelerator
    if hasattr(torch, "accelerator"):
        if not torch.accelerator.is_available():
            raise NotImplementedError(
                "Unsloth cannot find any torch accelerator? You need a GPU."
            )
        accelerator = str(torch.accelerator.current_accelerator())
        if accelerator in ("cuda", "xpu", "hip"):
            raise RuntimeError(
                f"Unsloth: Weirdly `torch.cuda.is_available()`, `torch.xpu.is_available()` and `is_hip` all failed.\n"
                f"But `torch.accelerator.current_accelerator()` works with it being = `{accelerator}`\n"
                f"Please reinstall torch - it's most likely broken :("
            )
    raise NotImplementedError(
        "Unsloth currently only works on NVIDIA, AMD, Intel GPUs, and Apple Silicon (MPS).\n"
        "If you're on a Mac with Apple Silicon, ensure you have:\n"
        "  1. PyTorch 2.1+ installed: pip install torch>=2.1.0\n"
        '  2. MPS backend available: python -c "import torch; print(torch.backends.mps.is_available())"'
    )


DEVICE_TYPE: str = get_device_type()
# HIP fails for autocast and other torch functions. Use CUDA instead
DEVICE_TYPE_TORCH = DEVICE_TYPE
if DEVICE_TYPE_TORCH == "hip":
    DEVICE_TYPE_TORCH = "cuda"


@functools.cache
def get_device_count():
    if DEVICE_TYPE in ("cuda", "hip"):
        return torch.cuda.device_count()
    elif DEVICE_TYPE == "xpu":
        return torch.xpu.device_count()
    elif DEVICE_TYPE == "mps":
        # MPS follows a single-GPU paradigm on Apple Silicon
        return 1
    else:
        return 1


DEVICE_COUNT: int = get_device_count()

# 4-bit quantization requires a block size of 64
# this is not supported on AMD Instinct GPUs currently
# | Device Type     | Warp Size | Block Size |
# |-----------------|-----------|------------|
# | CUDA            |    32     |     64     |
# | Radeon (Navi)   |    32     |     64     |
# | Instinct (MI)   |    64     |    128     |
#
# Since bitsandbytes 0.49.0, pre-quantized models with 64 blockwise now works
# on Radeon GPUs, but not Instinct MI300x for eg [WIP]
# See https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1748

ALLOW_PREQUANTIZED_MODELS: bool = True
# HSA_STATUS_ERROR_EXCEPTION checks - sometimes AMD fails for BnB
ALLOW_BITSANDBYTES: bool = True
if DEVICE_TYPE == "hip":
    try:
        import bitsandbytes
    except:
        print(
            "Unsloth: `bitsandbytes` is not installed - 4bit QLoRA unallowed, but 16bit and full finetuning works."
        )
        ALLOW_PREQUANTIZED_MODELS = False
        ALLOW_BITSANDBYTES = False
    if ALLOW_BITSANDBYTES:
        ALLOW_BITSANDBYTES = Version(bitsandbytes.__version__) > Version("0.48.2.dev0")
        if Version(bitsandbytes.__version__) > Version("0.49.0"):
            try:
                # Pre-quantized bitsandbytes models use blocksize 64, so we need to check the GPU
                from bitsandbytes.cextension import ROCM_WARP_SIZE_64

                ALLOW_PREQUANTIZED_MODELS = not ROCM_WARP_SIZE_64
            except Exception as e:
                print(
                    "Unsloth: Checking `from bitsandbytes.cextension import ROCM_WARP_SIZE_64` had error = \n"
                    f"{str(e)}\n"
                    "4bit QLoRA disabled for now, but 16bit and full finetuning works."
                )
                ALLOW_PREQUANTIZED_MODELS = False
                ALLOW_BITSANDBYTES = False
        elif ALLOW_BITSANDBYTES:
            from bitsandbytes.nn.modules import Params4bit

            if "blocksize = 64 if not HIP_ENVIRONMENT else 128" in inspect.getsource(
                Params4bit
            ):
                ALLOW_PREQUANTIZED_MODELS = False

elif DEVICE_TYPE == "mps":
    # bitsandbytes does not support MPS/Apple Silicon yet
    # See https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1292
    print(
        "Unsloth: bitsandbytes does not support Apple Silicon (MPS) yet. "
        "4-bit/8-bit quantization is disabled, but 16-bit LoRA and full finetuning work."
    )
    ALLOW_PREQUANTIZED_MODELS = False
    ALLOW_BITSANDBYTES = False
