# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Parsing llama-server stdout: log levels, GPU offload and device lines."""

import re

from typing import (
    Iterable,
    Optional,
)

from .memory_policy import _extra_args_set_any_flag

_PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR = (
    "This Mac's Metal device is virtualised, where offloaded layers can return "
    "corrupt output, and the installed unsloth_zoo diffusion shim has no --ngl, "
    "so the zero-layer CPU pin cannot be applied and the runner would use Metal "
    "anyway. Run 'unsloth studio update' to get a shim with --ngl, or set "
    "UNSLOTH_ALLOW_PARAVIRTUAL_METAL=1 to keep the GPU on a VM you have verified."
)

_VULKAN_DIFFUSION_GPU_IDS_ERROR = (
    "GPU selection (gpu_ids) is not supported for a DiffusionGemma "
    "GGUF on a Vulkan llama.cpp build: the diffusion runner selects "
    "its device by CUDA physical index, which has no defined mapping "
    "to ggml Vulkan device ordinals. Omit gpu_ids to use the default "
    "device."
)


# llama-server can serve HTTP 200 while running a model entirely on CPU when a
# GPU backend fails to init (#5807 / #5106 / #5830). Classify the startup log so
# Unsloth can warn. Priority: explicit "offloaded N/M layers to GPU" counts
# (authoritative), then GPU "model buffer size" lines (host-pinned _Host
# excluded), then the "device_info:" device table (disconfirm only).
_GPU_OFFLOAD_MARKERS = (
    "CUDA",
    "ROCm",
    "ROCM",
    "HIP",
    "Metal",
    "Vulkan",
    "OpenCL",
    "SYCL",
    "MUSA",
    "CANN",
)
_OFFLOADED_LAYERS_RE = re.compile(
    r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers?\s+to\s+gpu", re.IGNORECASE
)
# llama.cpp #23021 moved libllama INFO (offload counts, device table, buffer sizes) to trace (4).
_LLAMA_TRACE_VERBOSITY = 4
_TRACE_VERBOSITY_HELP_RE = re.compile(r"\b4:\s*trace\b", re.IGNORECASE)
_LOG_VERBOSITY_FLAGS = frozenset(
    {"-lv", "--verbosity", "--log-verbosity", "-v", "--verbose", "--log-verbose", "--log-disable"}
)
# Past this, the active llama-server log keeps only warnings and errors.
_LLAMA_LOG_FULL_BYTES = 64 * 1024 * 1024
# llama.cpp's prefix (stamp optional, colours allowed); a mid-line "x W " is record text.
_ANSI = r"(?:\x1b\[[0-9;]*m)*"
_LOG_LEVEL_TOKEN_RE = re.compile(rf"^{_ANSI}(?:\d+\.\d+\.\d+\.\d+{_ANSI} )?{_ANSI}([A-Z]) ")
# The startup head is never trimmed: its load lines are read for the server's whole life.
_STDOUT_TRIM_AT = 20000
_STDOUT_TAIL_KEEP = 5000
_STDOUT_HEAD_MAX = 5000


def _llama_line_is_warning_or_error(line: str, line_lower: str) -> bool:
    level = _LOG_LEVEL_TOKEN_RE.match(line)
    if level is not None:
        return level.group(1) in "WE"
    return "error" in line_lower or "fail" in line_lower or "warn" in line_lower


def _trace_verbosity_args(caps: dict, extra_args: Optional[Iterable[str]], environ) -> list[str]:
    """``--verbosity 4`` when the build hides load lines at its default and the user chose none."""
    if not caps.get("supports_trace_verbosity"):
        return []
    if _extra_args_set_any_flag(extra_args, _LOG_VERBOSITY_FLAGS):
        return []
    if "LLAMA_ARG_LOG_VERBOSITY" in environ:
        return []
    return ["--verbosity", str(_LLAMA_TRACE_VERBOSITY)]


_GPU_MODEL_BUFFER_RE = re.compile(
    r"\b(?:CUDA|ROCm|ROCM|HIP|SYCL)(\d+)\s+model buffer size\s*=\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s+MiB\b",
    re.IGNORECASE,
)
_DEVICE_ROW_RE = re.compile(
    # MTL: ggml-metal names its devices MTL0, MTL1 (GGML_METAL_NAME), never "Metal".
    r"-\s*(CUDA|ROCm|ROCM|HIP|Metal|MTL|Vulkan|SYCL|OpenCL|MUSA|CANN|CPU)\w*\s*:",
    re.IGNORECASE,
)
_GPU_DEVICE_PREFIXES = (
    "cuda",
    "rocm",
    "hip",
    "metal",
    "mtl",
    "vulkan",
    "sycl",
    "opencl",
    "musa",
    "cann",
)


def parse_gpu_offload_counts(lines: "list[str]") -> "Optional[tuple[int, int]]":
    """``(offloaded, total)`` for the largest-M line, or None; ties keep the first
    line since llama.cpp logs the main model before its drafter."""
    max_total = -1
    offloaded_at_max = 0
    for line in lines:
        match = _OFFLOADED_LAYERS_RE.search(line)
        if not match:
            continue
        offloaded, total = int(match.group(1)), int(match.group(2))
        if total > max_total:
            max_total, offloaded_at_max = total, offloaded
    if max_total <= 0:
        return None
    return offloaded_at_max, max_total


def llama_saw_gpu_device(lines: "list[str]") -> Optional[bool]:
    """Whether llama.cpp's device table (rows after the header) lists a GPU, or None
    with no table. All-CPU means the GPU backend failed to load."""
    after_header = False
    saw_device_row = False
    saw_gpu_device = False
    for line in lines:
        if "device_info:" in line:
            after_header = True
            continue
        if not after_header:
            continue
        match = _DEVICE_ROW_RE.search(line)
        if not match:
            continue
        saw_device_row = True
        if match.group(1).lower().startswith(_GPU_DEVICE_PREFIXES):
            saw_gpu_device = True
    if not saw_device_row:
        return None
    return saw_gpu_device


def classify_gpu_offload_lines(lines: "list[str]") -> Optional[bool]:
    """True if the model landed on a GPU, False if it stayed on CPU despite GPU
    intent, None when the log has no usable signal."""
    # Counted offload is authoritative, keyed on the model with the most layers.
    # A separate MTP/draft model logs its own (much smaller) "offloaded N/M"
    # line, so decide on the largest-M line: a drafter that fits on GPU must not
    # mask a main model running on CPU. N>0 on that model is True, 0 is False.
    max_total = -1
    offloaded_at_max = 0
    for line in lines:
        match = _OFFLOADED_LAYERS_RE.search(line)
        if not match:
            continue
        offloaded, total = int(match.group(1)), int(match.group(2))
        if total > max_total or (total == max_total and offloaded > offloaded_at_max):
            max_total, offloaded_at_max = total, offloaded
    if max_total >= 0:
        return offloaded_at_max > 0

    # GPU marker on a *model* buffer; _Host buffers are CPU-pinned, not offload.
    # Buffer lines are authoritative: present but none on a GPU means CPU-only,
    # so do not let the device table below override that.
    saw_model_buffer = False
    for line in lines:
        if "model buffer size" not in line:
            continue
        saw_model_buffer = True
        if "_Host" not in line and any(m in line for m in _GPU_OFFLOAD_MARKERS):
            return True
    if saw_model_buffer:
        return False

    # device_info: lists *available* devices (printed whenever a GPU backend is
    # visible), not where the model loaded, so it can only disconfirm: an
    # all-CPU table means no usable GPU. A visible GPU device is not proof the
    # model used it, so it does not return True.
    if llama_saw_gpu_device(lines) is False:
        return False
    return None
