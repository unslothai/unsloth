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
    "get_device_type",
    "DEVICE_TYPE",
    "DEVICE_TYPE_TORCH",
    "DEVICE_COUNT",
    "ALLOW_PREQUANTIZED_MODELS",
    "ALLOW_BITSANDBYTES",
]

import torch
import functools
from unsloth_zoo.utils import Version
import inspect

@functools.cache
def is_hip():
    return bool(getattr(getattr(torch, "version", None), "hip", None))
pass

@functools.cache
def get_device_type():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if is_hip():
            return "hip"
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    # Check torch.accelerator
    if hasattr(torch, "accelerator"):
        if not torch.accelerator.is_available():
            raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
        accelerator = str(torch.accelerator.current_accelerator())
        if accelerator in ("cuda", "xpu", "hip"):
            raise RuntimeError(
                f"Unsloth: Weirdly `torch.cuda.is_available()`, `torch.xpu.is_available()` and `is_hip` all failed.\n"\
                f"But `torch.accelerator.current_accelerator()` works with it being = `{accelerator}`\n"\
                f"Please reinstall torch - it's most likely broken :("
            )
    raise NotImplementedError("Unsloth currently only works on NVIDIA, AMD and Intel GPUs.")
pass
DEVICE_TYPE : str = get_device_type()
# HIP fails for autocast and other torch functions. Use CUDA instead
DEVICE_TYPE_TORCH = DEVICE_TYPE
if DEVICE_TYPE_TORCH == "hip": DEVICE_TYPE_TORCH = "cuda"

@functools.cache
def get_device_count():
    if DEVICE_TYPE in ("cuda", "hip"):
        return torch.cuda.device_count()
    elif DEVICE_TYPE == "xpu":
        return torch.xpu.device_count()
    else:
        return 1
pass

DEVICE_COUNT : int = get_device_count()

# Check blocksize for 4bit -> 64 for CUDA, 128 for AMD
# If AMD, we cannot load pre-quantized models for now :(
ALLOW_PREQUANTIZED_MODELS : bool = True
# HSA_STATUS_ERROR_EXCEPTION checks - sometimes AMD fails for BnB
ALLOW_BITSANDBYTES : bool = True
if DEVICE_TYPE == "hip":
    try:
        from bitsandbytes.nn.modules import Params4bit
        if "blocksize = 64 if not HIP_ENVIRONMENT else 128" in inspect.getsource(Params4bit):
            ALLOW_PREQUANTIZED_MODELS = False
        import bitsandbytes
        ALLOW_BITSANDBYTES = Version(bitsandbytes.__version__) > Version("0.48.2.dev0")
    except:
        pass
pass
