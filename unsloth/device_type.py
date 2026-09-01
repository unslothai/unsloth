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
    "get_device_stats",
    "clean_gpu_cache",
    "get_current_device",
    "resolve_hip_gpu_stats_name",
    "arch_lacks_bf16",
    "hip_visible_archs",
    "is_mlx_available",
]

import functools
import inspect
import os
import re
from unsloth_zoo.utils import Version
from .bnb_availability import native_kernels_ready


def is_mlx_available():
    try:
        from unsloth_zoo.mlx import is_mlx_available as _is_mlx_available
    except ImportError:
        return False
    return _is_mlx_available()


_IS_MLX = is_mlx_available()

if not _IS_MLX:
    import torch


@functools.cache
def is_hip():
    if _IS_MLX:
        return False
    return bool(getattr(getattr(torch, "version", None), "hip", None))


@functools.cache
def get_device_type():
    # MLX first: torch is never imported on the MLX runtime, so claiming "cuda" here would NameError in
    # get_device_count. Matches unsloth/__init__.py and unsloth_zoo.device_type.
    if _IS_MLX:
        return "mlx"
    # Test-only CPU fallback: report "cuda" so every DEVICE_TYPE == "cuda" branch behaves identically.
    # Read once per process (function is cached).
    if os.environ.get("UNSLOTH_ALLOW_CPU", "0") == "1":
        return "cuda"
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if is_hip():
            return "hip"
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch, "accelerator"):
        if not torch.accelerator.is_available():
            raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
        accelerator = str(torch.accelerator.current_accelerator())
        if accelerator in ("cuda", "xpu", "hip"):
            raise RuntimeError(
                f"Unsloth: Weirdly `torch.cuda.is_available()`, `torch.xpu.is_available()` and `is_hip` all failed.\n"
                f"But `torch.accelerator.current_accelerator()` works with it being = `{accelerator}`\n"
                f"Please reinstall torch - it's most likely broken :("
            )
    raise NotImplementedError("Unsloth currently only works on NVIDIA, AMD and Intel GPUs.")


DEVICE_TYPE: str = get_device_type()
# HIP fails for autocast and other torch functions. Use CUDA instead
DEVICE_TYPE_TORCH = DEVICE_TYPE
if DEVICE_TYPE_TORCH == "hip":
    DEVICE_TYPE_TORCH = "cuda"
elif DEVICE_TYPE_TORCH == "mlx":
    DEVICE_TYPE_TORCH = "mps"


@functools.cache
def get_device_count():
    if DEVICE_TYPE in ("cuda", "hip"):
        return torch.cuda.device_count()
    elif DEVICE_TYPE == "xpu":
        return torch.xpu.device_count()
    else:
        return 1


DEVICE_COUNT: int = get_device_count()

# 4-bit quantization requires a block size of 64: Instinct (MI) has a warp size of 64 against 32
# elsewhere. Since bitsandbytes 0.49.0 pre-quantized 64-blockwise models work on Radeon (Navi)
# but not Instinct (bitsandbytes-foundation/bitsandbytes#1748); since 0.49.2 blocksize=64 4-bit
# is supported on CDNA (MI Instinct / gfx9xx) too (#1856).

ALLOW_PREQUANTIZED_MODELS: bool = True
# HSA_STATUS_ERROR_EXCEPTION checks - sometimes AMD fails for BnB
ALLOW_BITSANDBYTES: bool = True
# Unusable bitsandbytes on any backend, not just hip: clear the flags the loader reads before it
# picks a 4bit checkpoint. A guarded import, not find_spec, since importable is not usable: from
# 0.46 a dead native library still resolves every ctypes handle to a closure that raises only when
# called, so 4bit would die mid-run rather than fall back here.
try:
    import bitsandbytes as _bnb_probe
except Exception:
    ALLOW_PREQUANTIZED_MODELS = False
    ALLOW_BITSANDBYTES = False
else:
    if not native_kernels_ready(_bnb_probe, DEVICE_TYPE):
        ALLOW_PREQUANTIZED_MODELS = False
        ALLOW_BITSANDBYTES = False
    del _bnb_probe
# gfx906 (MI50 / Radeon VII / Vega 20): Dynamo/Inductor codegen is broken on this legacy GCN arch
# (ROCm dropped it after 6.3), so compiled graphs crash or miscompile while eager trains fine.
if DEVICE_TYPE == "hip":
    try:
        _gcn_arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0].strip().lower()
    except Exception:
        _gcn_arch = ""
    if _gcn_arch == "gfx906":
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
        print(
            "Unsloth: gfx906 (MI50 / Radeon VII) detected - torch.compile disabled "
            "(community-maintained legacy GCN path)."
        )
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
        if Version(bitsandbytes.__version__) >= Version("0.49.2"):
            pass
        elif Version(bitsandbytes.__version__) >= Version("0.49.0"):
            try:
                # Pre-quantized bitsandbytes models use blocksize 64.
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
            if "blocksize = 64 if not HIP_ENVIRONMENT else 128" in inspect.getsource(Params4bit):
                ALLOW_PREQUANTIZED_MODELS = False


def arch_lacks_bf16(gcn_arch):
    """gfx10 (RDNA 1/2) claims bf16 it lacks, and Triton's dot then kills the process in LLVM
    with no Python exception (issue 7922). gfx11 has bf16, so the prefix must stay 5 chars."""
    return str(gcn_arch or "").split(":", 1)[0].strip().lower().startswith("gfx10")


def hip_visible_archs():
    """Guarded per device: one unreadable device must not discard the archs beside it, or a
    gfx10 keeps bf16 and dies in Triton (#7922). Only an unreadable count returns []."""
    try:
        count = torch.cuda.device_count()
    except Exception:
        return []
    archs = []
    for i in range(count):
        try:
            archs.append(str(getattr(torch.cuda.get_device_properties(i), "gcnArchName", "")))
        except Exception:
            continue
    return archs


def resolve_hip_gpu_stats_name(gpu_stats):
    name = str(getattr(gpu_stats, "name", "") or "").strip()
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    normalized_name = name.lower().strip(". ")
    if normalized_name and normalized_name not in ("amd radeon graphics",):
        return name + ". "

    try:
        torch_name = str(torch.cuda.get_device_name(0) or "").strip()
        torch_name = re.sub(r"\s*\([^)]*\)\s*$", "", torch_name).strip()
    except Exception:
        torch_name = ""
    normalized_torch_name = torch_name.lower().strip(". ")
    if normalized_torch_name and normalized_torch_name not in ("amd radeon graphics",):
        return torch_name + ". "

    arch_name = ""
    for key in ("gcnArchName", "gcn_arch_name", "arch_name", "gfx_arch_name"):
        value = getattr(gpu_stats, key, None)
        if value is not None and str(value).strip():
            arch_name = str(value).strip()
            break

    if arch_name:
        match = re.search(r"(gfx[0-9a-z]+)", arch_name, flags = re.I)
        if match:
            return f"AMD {match.group(1).lower()} GPU. "
    return "AMD GPU. "


_DEVICE_MODULE = None if _IS_MLX else getattr(torch, DEVICE_TYPE_TORCH, None)
if not _IS_MLX and _DEVICE_MODULE is None:
    raise RuntimeError(f"Unsloth: PyTorch does not provide the {DEVICE_TYPE_TORCH} backend.")


def get_device_stats() -> tuple[str, str, float]:
    """Return (name, stats_snippet, max_memory_gb)."""
    if _DEVICE_MODULE is None:
        raise RuntimeError("Unsloth: GPU statistics are unavailable on the MLX runtime.")
    gpu_stats = _DEVICE_MODULE.get_device_properties(0)
    max_memory = round(gpu_stats.total_memory / 1024**3, 3)

    if DEVICE_TYPE == "hip":
        name = resolve_hip_gpu_stats_name(gpu_stats)
        snippet = f"ROCm Toolkit: {torch.version.hip}."
    elif DEVICE_TYPE == "xpu":
        name = gpu_stats.name + ". " if gpu_stats.name else "Intel XPU Device. "
        snippet = f"Intel Toolkit: {torch.version.xpu}."
    else:
        name = gpu_stats.name + ". " if gpu_stats.name else "NVIDIA GPU Device. "
        snippet = f"CUDA: {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit: {torch.version.cuda}."
    return name, snippet, max_memory


def clean_gpu_cache() -> None:
    """Clear GPU cache for current device type."""
    if _DEVICE_MODULE is not None:
        _DEVICE_MODULE.empty_cache()


def get_current_device() -> int:
    """Get current device index."""
    return 0 if _DEVICE_MODULE is None else _DEVICE_MODULE.current_device()
