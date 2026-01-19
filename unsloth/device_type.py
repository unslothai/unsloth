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
    "DeviceContext",
    "device_context",
    "clean_gpu_cache",
    "get_current_device",
]

import torch
import functools
import inspect
from unsloth_zoo.utils import Version


@functools.cache
def is_hip():
    return bool(getattr(getattr(torch, "version", None), "hip", None))


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
        "Unsloth currently only works on NVIDIA, AMD and Intel GPUs."
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


class DeviceContext:
    """Encapsulates device-specific operations for XPU/HIP/CUDA."""

    DEVICE_MODULE_MAP = {
        "cuda": torch.cuda,
        "hip": torch.cuda,
        **({"xpu": torch.xpu} if hasattr(torch, "xpu") else {}),
    }
    DEVICE_NAME_MAP = {"xpu": "Intel XPU", "cuda": "NVIDIA GPU", "hip": "AMD GPU"}

    def __init__(self, device_type: str = DEVICE_TYPE) -> None:
        if device_type not in self.DEVICE_MODULE_MAP:
            raise ValueError(f"Unsloth: Unsupported device type: {device_type}")
        self.device_type = device_type
        # Cache the torch module for this device
        self.torch_module = self.DEVICE_MODULE_MAP[device_type]

    def get_stats(self) -> tuple[str, str, float]:
        """Return (name, stats_snippet, max_memory_gb)."""
        gpu_stats = self.torch_module.get_device_properties(0)
        max_mem = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        # Device name
        name = gpu_stats.name + ". " if gpu_stats.name else self._get_default_name()

        # Toolkit snippet
        snippet = self._get_toolkit_snippet(gpu_stats)

        return name, snippet, max_mem

    def _get_default_name(self) -> str:
        """Get default device name when props.name is empty."""
        return self.DEVICE_NAME_MAP[self.device_type] + " Device. "

    def _get_toolkit_snippet(self, props) -> str:
        """Get toolkit version snippet."""
        if self.device_type == "cuda":
            return f"CUDA: {props.major}.{props.minor}. CUDA Toolkit: {torch.version.cuda}."
        elif self.device_type == "hip":
            return f"ROCm Toolkit: {torch.version.hip}."
        else:  # xpu
            return f"Intel Toolkit: {torch.version.xpu}."


# Singleton instance
device_context = DeviceContext()


# Module-level functions for backward compatibility
def clean_gpu_cache() -> None:
    """Clear GPU cache for current device type."""
    device_context.torch_module.empty_cache()


def get_current_device() -> int:
    """Get current device index."""
    return device_context.torch_module.current_device()
