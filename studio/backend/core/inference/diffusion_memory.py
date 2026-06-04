# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime memory planning helpers for Studio diffusion inference.

This module is intentionally backend-agnostic. It estimates the cost of a
requested diffusion load from model storage hints, the requested workload, and
the current hardware snapshot, then recommends a high-level runtime profile.
The actual loader still owns implementation details such as Diffusers CPU
offload, GGUF CPU residency, and safetensors quantization.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


DIFFUSION_MEMORY_MODE_AUTO = "auto"
DIFFUSION_MEMORY_MODE_FAST = "fast"
DIFFUSION_MEMORY_MODE_BALANCED = "balanced"
DIFFUSION_MEMORY_MODE_LOW_VRAM = "low_vram"
DIFFUSION_MEMORY_MODE_MANUAL = "manual"
DIFFUSION_MEMORY_MODES = {
    DIFFUSION_MEMORY_MODE_AUTO,
    DIFFUSION_MEMORY_MODE_FAST,
    DIFFUSION_MEMORY_MODE_BALANCED,
    DIFFUSION_MEMORY_MODE_LOW_VRAM,
    DIFFUSION_MEMORY_MODE_MANUAL,
}

OFFLOAD_POLICY_AGGRESSIVE = "aggressive"
OFFLOAD_POLICY_BALANCED = "balanced"
OFFLOAD_POLICY_NONE = "none"
OFFLOAD_POLICY_LESS_AGGRESSIVE = "less_aggressive"

SAFETENSORS_QUANT_NONE = "none"
SAFETENSORS_QUANT_BNB_4BIT_NF4 = "bitsandbytes_4bit_nf4"

DEFAULT_IMAGE_WIDTH = 1024
DEFAULT_IMAGE_HEIGHT = 1024
DEFAULT_VIDEO_FRAMES = 121


@dataclass(frozen = True)
class MemoryDeviceSnapshot:
    backend: str
    device_id: Optional[str] = None
    vendor: Optional[str] = None
    name: Optional[str] = None
    memory_kind: str = "unknown"
    total_mib: Optional[int] = None
    free_mib: Optional[int] = None
    supports_bf16: Optional[bool] = None
    supports_fp8: Optional[bool] = None

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "device_id": self.device_id,
            "vendor": self.vendor,
            "name": self.name,
            "memory_kind": self.memory_kind,
            "total_mib": self.total_mib,
            "free_mib": self.free_mib,
            "supports_bf16": self.supports_bf16,
            "supports_fp8": self.supports_fp8,
        }


@dataclass(frozen = True)
class HardwareMemorySnapshot:
    backend: str
    devices: tuple[MemoryDeviceSnapshot, ...] = ()
    system_total_mib: Optional[int] = None
    system_available_mib: Optional[int] = None

    @property
    def primary_device(self) -> Optional[MemoryDeviceSnapshot]:
        return self.devices[0] if self.devices else None

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "devices": [device.as_public_dict() for device in self.devices],
            "system_total_mib": self.system_total_mib,
            "system_available_mib": self.system_available_mib,
        }


@dataclass(frozen = True)
class ModelComponentEstimate:
    name: str
    format: str
    quantization: Optional[str] = None
    storage_mib: Optional[int] = None
    packed_device_mib: Optional[int] = None
    dense_device_mib: Optional[int] = None

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "format": self.format,
            "quantization": self.quantization,
            "storage_mib": self.storage_mib,
            "packed_device_mib": self.packed_device_mib,
            "dense_device_mib": self.dense_device_mib,
        }


@dataclass(frozen = True)
class ModelMemoryEstimate:
    components: tuple[ModelComponentEstimate, ...] = ()
    source: str = "request_metadata"
    confidence: str = "low"

    @property
    def has_gguf_components(self) -> bool:
        return any(component.format == "gguf" for component in self.components)

    @property
    def has_safetensors_components(self) -> bool:
        return any(component.format == "safetensors" for component in self.components)

    def total_storage_mib(self) -> Optional[int]:
        return _sum_optional(component.storage_mib for component in self.components)

    def total_packed_device_mib(self) -> Optional[int]:
        return _sum_optional(component.packed_device_mib for component in self.components)

    def total_dense_device_mib(self) -> Optional[int]:
        return _sum_optional(component.dense_device_mib for component in self.components)

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "confidence": self.confidence,
            "has_gguf_components": self.has_gguf_components,
            "has_safetensors_components": self.has_safetensors_components,
            "total_storage_mib": self.total_storage_mib(),
            "total_packed_device_mib": self.total_packed_device_mib(),
            "total_dense_device_mib": self.total_dense_device_mib(),
            "components": [component.as_public_dict() for component in self.components],
        }


@dataclass(frozen = True)
class DiffusionWorkloadEstimate:
    media_kind: str = "image"
    family: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    num_frames: Optional[int] = None
    batch_size: int = 1
    guidance_scale: Optional[float] = None
    requires_image_input: bool = False

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "media_kind": self.media_kind,
            "family": self.family,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "batch_size": self.batch_size,
            "guidance_scale": self.guidance_scale,
            "requires_image_input": self.requires_image_input,
        }


@dataclass(frozen = True)
class DiffusionMemoryPlan:
    requested_mode: str
    selected_mode: str
    selected_offload_policy: Optional[str]
    selected_safetensors_quantization: Optional[str]
    selected_safetensors_quantization_components: Optional[tuple[str, ...]]
    estimates: dict[str, Optional[int]]
    hardware: HardwareMemorySnapshot
    model: ModelMemoryEstimate
    workload: DiffusionWorkloadEstimate
    confidence: str
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "2026-06-04",
            "requested_mode": self.requested_mode,
            "selected_mode": self.selected_mode,
            "selected_offload_policy": self.selected_offload_policy,
            "selected_safetensors_quantization": (
                self.selected_safetensors_quantization
            ),
            "selected_safetensors_quantization_components": (
                list(self.selected_safetensors_quantization_components)
                if self.selected_safetensors_quantization_components is not None
                else None
            ),
            "estimates": dict(self.estimates),
            "hardware": self.hardware.as_public_dict(),
            "model": self.model.as_public_dict(),
            "workload": self.workload.as_public_dict(),
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


def normalize_memory_mode(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized:
        return None
    if normalized not in DIFFUSION_MEMORY_MODES:
        valid = ", ".join(sorted(DIFFUSION_MEMORY_MODES))
        raise ValueError(f"Unsupported diffusion memory_mode '{value}'. Use one of: {valid}.")
    return normalized


def memory_planner_options() -> dict[str, Any]:
    return {
        "modes": [
            {
                "name": DIFFUSION_MEMORY_MODE_AUTO,
                "description": "Choose the fastest mode expected to fit the current device.",
            },
            {
                "name": DIFFUSION_MEMORY_MODE_FAST,
                "description": "Prefer resident weights and no CPU offload; fall back only if needed.",
            },
            {
                "name": DIFFUSION_MEMORY_MODE_BALANCED,
                "description": "Prefer the measured Studio middle ground for VRAM and throughput.",
            },
            {
                "name": DIFFUSION_MEMORY_MODE_LOW_VRAM,
                "description": "Prefer the lowest practical VRAM path for the selected format.",
            },
            {
                "name": DIFFUSION_MEMORY_MODE_MANUAL,
                "description": "Respect explicit runtime/offload/quantization choices.",
            },
        ],
        "input_estimates": [
            "component storage size when cached or local",
            "component format and quantization label",
            "requested width, height, frames, batch, and family",
            "runtime free memory on CUDA, ROCm, XPU, MPS, MLX, or CPU",
        ],
        "supported_formats": [
            "safetensors_bf16_fp16",
            "safetensors_bitsandbytes_4bit_nf4",
            "safetensors_fp8_estimate",
            "gguf_quantized",
        ],
        "supported_memory_kinds": [
            "discrete_vram",
            "unified_memory",
            "system_memory",
            "unknown",
        ],
        "applies_when": (
            "runtime.memory_mode controls planner behavior. When omitted, "
            "Studio defaults to auto unless an explicit offload_policy makes "
            "the request manual."
        ),
    }


def _cuda_memory_kind(torch: Any, index: int) -> str:
    try:
        props = torch.cuda.get_device_properties(index)
    except Exception:
        return "discrete_vram"
    for attr in ("integrated", "is_integrated"):
        try:
            if bool(getattr(props, attr, False)):
                return "unified_memory"
        except Exception:
            pass
    return "discrete_vram"


def snapshot_hardware_memory() -> HardwareMemorySnapshot:
    system_total, system_available = _system_memory_mib()
    try:
        import torch

        if torch.cuda.is_available():
            devices: list[MemoryDeviceSnapshot] = []
            backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            vendor = "amd" if backend == "rocm" else "nvidia"
            for index in range(torch.cuda.device_count()):
                try:
                    free, total = torch.cuda.mem_get_info(index)
                    free_mib = int(free // (1024 * 1024))
                    total_mib = int(total // (1024 * 1024))
                except Exception:
                    free_mib = None
                    total_mib = None
                try:
                    name = torch.cuda.get_device_name(index)
                except Exception:
                    name = None
                devices.append(
                    MemoryDeviceSnapshot(
                        backend = backend,
                        device_id = f"{backend}:{index}",
                        vendor = vendor,
                        name = name,
                        memory_kind = _cuda_memory_kind(torch, index),
                        total_mib = total_mib,
                        free_mib = free_mib,
                        supports_bf16 = _cuda_supports_bf16(torch, index),
                        supports_fp8 = _cuda_supports_fp8(torch, index),
                    )
                )
            return HardwareMemorySnapshot(
                backend = backend,
                devices = tuple(devices),
                system_total_mib = system_total,
                system_available_mib = system_available,
            )

        xpu = getattr(torch, "xpu", None)
        if xpu is not None and callable(getattr(xpu, "is_available", None)) and xpu.is_available():
            devices = []
            count = int(xpu.device_count()) if callable(getattr(xpu, "device_count", None)) else 1
            for index in range(count):
                free_mib = None
                total_mib = None
                mem_get_info = getattr(xpu, "mem_get_info", None)
                if callable(mem_get_info):
                    try:
                        free, total = mem_get_info(index)
                        free_mib = int(free // (1024 * 1024))
                        total_mib = int(total // (1024 * 1024))
                    except Exception:
                        pass
                devices.append(
                    MemoryDeviceSnapshot(
                        backend = "xpu",
                        device_id = f"xpu:{index}",
                        vendor = "intel",
                        memory_kind = "discrete_vram",
                        total_mib = total_mib,
                        free_mib = free_mib,
                    )
                )
            return HardwareMemorySnapshot(
                backend = "xpu",
                devices = tuple(devices),
                system_total_mib = system_total,
                system_available_mib = system_available,
            )

        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
            return HardwareMemorySnapshot(
                backend = "mps",
                devices = (
                    MemoryDeviceSnapshot(
                        backend = "mps",
                        device_id = "mps:0",
                        vendor = "apple",
                        memory_kind = "unified_memory",
                        total_mib = system_total,
                        free_mib = system_available,
                    ),
                ),
                system_total_mib = system_total,
                system_available_mib = system_available,
            )
    except Exception:
        pass

    if importlib.util.find_spec("mlx") is not None:
        return HardwareMemorySnapshot(
            backend = "mlx",
            devices = (
                MemoryDeviceSnapshot(
                    backend = "mlx",
                    device_id = "mlx:0",
                    vendor = "apple",
                    memory_kind = "unified_memory",
                    total_mib = system_total,
                    free_mib = system_available,
                ),
            ),
            system_total_mib = system_total,
            system_available_mib = system_available,
        )

    return HardwareMemorySnapshot(
        backend = "cpu",
        devices = (
            MemoryDeviceSnapshot(
                backend = "cpu",
                device_id = "cpu",
                memory_kind = "system_memory",
                total_mib = system_total,
                free_mib = system_available,
            ),
        ),
        system_total_mib = system_total,
        system_available_mib = system_available,
    )


def component_estimate_from_file_hint(
    *,
    name: str,
    fmt: str,
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    quantization: Optional[str] = None,
) -> ModelComponentEstimate:
    normalized_fmt = fmt.strip().lower()
    storage_mib = _resolve_storage_mib(repo_id = repo_id, filename = filename)
    inferred_quant = quantization or _infer_quantization_label(filename)
    if normalized_fmt == "gguf":
        packed_mib = storage_mib
        dense_mib = _estimate_gguf_dense_mib(storage_mib, inferred_quant)
    else:
        packed_mib = storage_mib
        dense_mib = storage_mib
    return ModelComponentEstimate(
        name = name,
        format = normalized_fmt,
        quantization = inferred_quant,
        storage_mib = storage_mib,
        packed_device_mib = packed_mib,
        dense_device_mib = dense_mib,
    )


def select_diffusion_memory_plan(
    *,
    requested_mode: Optional[str],
    explicit_offload_policy: Optional[str],
    explicit_safetensors_quantization: Optional[str],
    explicit_safetensors_quantization_components: Optional[list[str]],
    model: ModelMemoryEstimate,
    workload: DiffusionWorkloadEstimate,
    hardware: Optional[HardwareMemorySnapshot] = None,
    balanced_gguf_cache_mib: int = 4096,
) -> DiffusionMemoryPlan:
    mode = normalize_memory_mode(requested_mode) or DIFFUSION_MEMORY_MODE_AUTO
    hardware = hardware or snapshot_hardware_memory()
    reasons: list[str] = []
    warnings: list[str] = []
    device = hardware.primary_device
    unified_or_system_memory = bool(
        device is not None
        and device.memory_kind in {"unified_memory", "system_memory"}
    )
    safe_device_mib = _safe_device_budget_mib(device, hardware)
    runtime_mib = estimate_runtime_memory_mib(workload)
    base_overhead_mib = 2048 if workload.media_kind == "image" else 4096
    packed_mib = model.total_packed_device_mib()
    dense_mib = model.total_dense_device_mib()
    storage_mib = model.total_storage_mib()

    estimates: dict[str, Optional[int]] = {
        "safe_device_budget_mib": safe_device_mib,
        "runtime_headroom_mib": runtime_mib,
        "base_overhead_mib": base_overhead_mib,
        "model_storage_mib": storage_mib,
        "model_packed_device_mib": packed_mib,
        "model_dense_device_mib": dense_mib,
        "gguf_balanced_cache_mib": balanced_gguf_cache_mib,
        "resident_required_mib": None,
        "balanced_required_mib": None,
        "low_vram_required_mib": None,
    }

    if explicit_offload_policy is not None or mode == DIFFUSION_MEMORY_MODE_MANUAL:
        reasons.append("explicit offload or manual memory mode requested")
        if explicit_offload_policy is not None and unified_or_system_memory:
            warnings.append(
                "explicit offload_policy was requested on unified/system memory; "
                "device placement may ignore CPU-offload-specific savings"
            )
        return DiffusionMemoryPlan(
            requested_mode = mode,
            selected_mode = DIFFUSION_MEMORY_MODE_MANUAL,
            selected_offload_policy = explicit_offload_policy,
            selected_safetensors_quantization = explicit_safetensors_quantization,
            selected_safetensors_quantization_components = _tuple_or_none(
                explicit_safetensors_quantization_components
            ),
            estimates = estimates,
            hardware = hardware,
            model = model,
            workload = workload,
            confidence = _combined_confidence(model, safe_device_mib),
            reasons = tuple(reasons),
            warnings = tuple(warnings),
        )

    if safe_device_mib is None:
        warnings.append("device free memory is unknown; using conservative defaults")
        selected = (
            OFFLOAD_POLICY_BALANCED
            if model.has_gguf_components
            else explicit_offload_policy
        )
        if mode == DIFFUSION_MEMORY_MODE_LOW_VRAM:
            selected = OFFLOAD_POLICY_AGGRESSIVE
        if unified_or_system_memory:
            selected = OFFLOAD_POLICY_NONE
            warnings.append(
                "unified/system memory detected; CPU offload policies are not "
                "selected automatically"
            )
        return DiffusionMemoryPlan(
            requested_mode = mode,
            selected_mode = (
                DIFFUSION_MEMORY_MODE_LOW_VRAM
                if selected == OFFLOAD_POLICY_AGGRESSIVE
                else DIFFUSION_MEMORY_MODE_BALANCED
            ),
            selected_offload_policy = selected,
            selected_safetensors_quantization = explicit_safetensors_quantization,
            selected_safetensors_quantization_components = _tuple_or_none(
                explicit_safetensors_quantization_components
            ),
            estimates = estimates,
            hardware = hardware,
            model = model,
            workload = workload,
            confidence = "low",
            reasons = tuple(reasons),
            warnings = tuple(warnings),
        )

    if model.has_gguf_components:
        resident_required = _sum_known(packed_mib, runtime_mib, base_overhead_mib)
        balanced_required = _sum_known(
            min(balanced_gguf_cache_mib, packed_mib or balanced_gguf_cache_mib),
            runtime_mib,
            base_overhead_mib,
        )
        aggressive_required = _sum_known(runtime_mib, base_overhead_mib)
        estimates["resident_required_mib"] = resident_required
        estimates["balanced_required_mib"] = balanced_required
        estimates["low_vram_required_mib"] = aggressive_required

        if unified_or_system_memory:
            reasons.append(
                "unified/system memory detected; selecting resident GGUF execution "
                "instead of CPU-offload policies"
            )
            selected_mode = DIFFUSION_MEMORY_MODE_FAST
            selected_policy = OFFLOAD_POLICY_NONE
            if mode in {
                DIFFUSION_MEMORY_MODE_BALANCED,
                DIFFUSION_MEMORY_MODE_LOW_VRAM,
            }:
                warnings.append(
                    f"{mode} was requested, but CPU offload is not a useful "
                    "memory boundary on unified/system memory"
                )
        elif mode == DIFFUSION_MEMORY_MODE_LOW_VRAM:
            reasons.append("low_vram requested for GGUF")
            selected_mode = DIFFUSION_MEMORY_MODE_LOW_VRAM
            selected_policy = OFFLOAD_POLICY_AGGRESSIVE
        elif mode == DIFFUSION_MEMORY_MODE_BALANCED:
            reasons.append("balanced requested for GGUF")
            selected_mode = DIFFUSION_MEMORY_MODE_BALANCED
            selected_policy = OFFLOAD_POLICY_BALANCED
        elif mode == DIFFUSION_MEMORY_MODE_FAST:
            if resident_required <= safe_device_mib:
                reasons.append("fast requested and packed GGUF weights fit on device")
                selected_mode = DIFFUSION_MEMORY_MODE_FAST
                selected_policy = OFFLOAD_POLICY_NONE
            elif balanced_required <= safe_device_mib:
                reasons.append("fast requested but resident GGUF estimate does not fit")
                selected_mode = DIFFUSION_MEMORY_MODE_BALANCED
                selected_policy = OFFLOAD_POLICY_BALANCED
            else:
                reasons.append("fast requested but only aggressive estimate fits")
                selected_mode = DIFFUSION_MEMORY_MODE_LOW_VRAM
                selected_policy = OFFLOAD_POLICY_AGGRESSIVE
        elif resident_required <= int(safe_device_mib * 0.80):
            reasons.append("packed GGUF resident estimate has large device headroom")
            selected_mode = DIFFUSION_MEMORY_MODE_FAST
            selected_policy = OFFLOAD_POLICY_NONE
        elif balanced_required <= safe_device_mib:
            reasons.append("balanced GGUF cache estimate fits current device")
            selected_mode = DIFFUSION_MEMORY_MODE_BALANCED
            selected_policy = OFFLOAD_POLICY_BALANCED
        else:
            reasons.append("balanced GGUF estimate exceeds current device budget")
            selected_mode = DIFFUSION_MEMORY_MODE_LOW_VRAM
            selected_policy = OFFLOAD_POLICY_AGGRESSIVE

        return DiffusionMemoryPlan(
            requested_mode = mode,
            selected_mode = selected_mode,
            selected_offload_policy = selected_policy,
            selected_safetensors_quantization = explicit_safetensors_quantization,
            selected_safetensors_quantization_components = _tuple_or_none(
                explicit_safetensors_quantization_components
            ),
            estimates = estimates,
            hardware = hardware,
            model = model,
            workload = workload,
            confidence = _combined_confidence(model, safe_device_mib),
            reasons = tuple(reasons),
            warnings = tuple(warnings),
        )

    dense_required = _sum_known(dense_mib, runtime_mib, base_overhead_mib)
    bnb_required = _sum_known(
        int((dense_mib or 0) * 0.45) if dense_mib is not None else None,
        runtime_mib,
        base_overhead_mib,
    )
    offload_required = _sum_known(runtime_mib, base_overhead_mib)
    estimates["resident_required_mib"] = dense_required
    estimates["balanced_required_mib"] = bnb_required
    estimates["low_vram_required_mib"] = offload_required

    selected_quant = explicit_safetensors_quantization
    selected_components = _tuple_or_none(explicit_safetensors_quantization_components)
    selected_policy = explicit_offload_policy

    if unified_or_system_memory:
        reasons.append(
            "unified/system memory detected; avoiding automatic CPU offload policy"
        )
        selected_mode = (
            DIFFUSION_MEMORY_MODE_BALANCED
            if mode == DIFFUSION_MEMORY_MODE_BALANCED
            else DIFFUSION_MEMORY_MODE_FAST
        )
        selected_policy = OFFLOAD_POLICY_NONE
        if mode == DIFFUSION_MEMORY_MODE_LOW_VRAM:
            selected_mode = DIFFUSION_MEMORY_MODE_LOW_VRAM
            warnings.append(
                "low_vram requested on unified/system memory; selecting quantization "
                "when supported instead of CPU offload"
            )
            if _can_select_bnb_nf4(hardware) and not selected_quant:
                selected_quant = SAFETENSORS_QUANT_BNB_4BIT_NF4
                selected_components = ("transformer", "unet")
                warnings.append("selected BnB NF4 quantization to reduce memory")
    elif mode == DIFFUSION_MEMORY_MODE_LOW_VRAM:
        reasons.append("low_vram requested for safetensors")
        selected_mode = DIFFUSION_MEMORY_MODE_LOW_VRAM
        selected_policy = OFFLOAD_POLICY_AGGRESSIVE
        if _can_select_bnb_nf4(hardware) and not selected_quant:
            selected_quant = SAFETENSORS_QUANT_BNB_4BIT_NF4
            selected_components = ("transformer", "unet")
            warnings.append("selected BnB NF4 quantization to reduce VRAM")
    elif mode == DIFFUSION_MEMORY_MODE_BALANCED:
        reasons.append("balanced requested for safetensors")
        selected_mode = DIFFUSION_MEMORY_MODE_BALANCED
        selected_policy = (
            OFFLOAD_POLICY_NONE
            if dense_required <= safe_device_mib
            else OFFLOAD_POLICY_AGGRESSIVE
        )
    elif dense_required <= safe_device_mib:
        reasons.append("full safetensors resident estimate fits current device")
        selected_mode = DIFFUSION_MEMORY_MODE_FAST
        selected_policy = OFFLOAD_POLICY_NONE
    elif _can_select_bnb_nf4(hardware) and bnb_required <= safe_device_mib:
        reasons.append("full safetensors estimate is tight; BnB NF4 estimate fits")
        selected_mode = DIFFUSION_MEMORY_MODE_LOW_VRAM
        selected_policy = OFFLOAD_POLICY_NONE
        if not selected_quant:
            selected_quant = SAFETENSORS_QUANT_BNB_4BIT_NF4
            selected_components = ("transformer", "unet")
            warnings.append("selected BnB NF4 quantization to reduce VRAM")
    else:
        reasons.append("safetensors resident estimate exceeds current device budget")
        selected_mode = DIFFUSION_MEMORY_MODE_LOW_VRAM
        selected_policy = OFFLOAD_POLICY_AGGRESSIVE

    return DiffusionMemoryPlan(
        requested_mode = mode,
        selected_mode = selected_mode,
        selected_offload_policy = selected_policy,
        selected_safetensors_quantization = selected_quant,
        selected_safetensors_quantization_components = selected_components,
        estimates = estimates,
        hardware = hardware,
        model = model,
        workload = workload,
        confidence = _combined_confidence(model, safe_device_mib),
        reasons = tuple(reasons),
        warnings = tuple(warnings),
    )


def estimate_runtime_memory_mib(workload: DiffusionWorkloadEstimate) -> int:
    width = max(64, int(workload.width or DEFAULT_IMAGE_WIDTH))
    height = max(64, int(workload.height or DEFAULT_IMAGE_HEIGHT))
    batch = max(1, int(workload.batch_size or 1))
    frames = max(
        1,
        int(
            workload.num_frames
            if workload.num_frames is not None
            else (DEFAULT_VIDEO_FRAMES if workload.media_kind == "video" else 1)
        ),
    )
    pixel_scale = (width * height * batch * frames) / float(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT)
    base = 8192 if workload.media_kind == "image" else 28672
    family = (workload.family or "").lower()
    multiplier = 1.0
    if "edit" in family or "layered" in family or workload.requires_image_input:
        multiplier *= 1.35
    if "ltx" in family or workload.media_kind == "video":
        multiplier *= 1.25
    if (
        "turbo" in family
        or "distilled" in family
        or "schnell" in family
        or (workload.guidance_scale is not None and workload.guidance_scale <= 1.0)
    ):
        multiplier *= 0.85
    return max(2048, int(base * max(0.25, pixel_scale) * multiplier))


def _system_memory_mib() -> tuple[Optional[int], Optional[int]]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.total // (1024 * 1024)), int(vm.available // (1024 * 1024))
    except Exception:
        pass
    try:
        total_pages = os.sysconf("SC_PHYS_PAGES")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (
            int(total_pages * page_size // (1024 * 1024)),
            int(available_pages * page_size // (1024 * 1024)),
        )
    except Exception:
        return None, None


def _cuda_supports_bf16(torch: Any, index: int) -> Optional[bool]:
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return None


def _cuda_supports_fp8(torch: Any, index: int) -> Optional[bool]:
    try:
        major, minor = torch.cuda.get_device_capability(index)
        return (major, minor) >= (8, 9)
    except Exception:
        return None


def _sum_optional(values: Any) -> Optional[int]:
    total = 0
    saw = False
    for value in values:
        if value is None:
            return None
        total += int(value)
        saw = True
    return total if saw else None


def _sum_known(*values: Optional[int]) -> int:
    return sum(int(value or 0) for value in values)


def _tuple_or_none(values: Optional[list[str] | tuple[str, ...]]) -> Optional[tuple[str, ...]]:
    if values is None:
        return None
    return tuple(str(value) for value in values)


def _resolve_storage_mib(*, repo_id: Optional[str], filename: Optional[str]) -> Optional[int]:
    if filename:
        try:
            path = Path(filename).expanduser()
            if path.is_file():
                return max(1, int(path.stat().st_size // (1024 * 1024)))
        except (OSError, ValueError):
            pass
    if repo_id and filename:
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(repo_id, filename)
            if isinstance(cached, str):
                path = Path(cached)
                if path.is_file():
                    return max(1, int(path.stat().st_size // (1024 * 1024)))
        except Exception:
            pass
    return None


def _infer_quantization_label(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    stem = Path(filename).name
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    parts = [part for part in stem.replace("-", "_").split("_") if part]
    if not parts:
        return None
    upper = [part.upper() for part in parts]
    for index, part in enumerate(upper):
        if part in {"BF16", "F16", "FP16", "FP8", "Q8", "Q6", "Q5", "Q4", "Q3", "Q2", "Q1"}:
            suffix = upper[index + 1 :]
            if suffix and suffix[0] in {"K", "M", "S", "L", "XS", "XXS"}:
                return "_".join([part] + suffix[:2])
            return part
        if part.startswith("IQ") or part.startswith("UD"):
            return "_".join(upper[index : index + 3])
    return None


def _estimate_gguf_dense_mib(
    storage_mib: Optional[int],
    quantization: Optional[str],
) -> Optional[int]:
    if storage_mib is None:
        return None
    q = (quantization or "").upper()
    if any(token in q for token in ("BF16", "F16", "FP16")):
        return storage_mib
    if "FP8" in q or "Q8" in q:
        return int(storage_mib * 2.0)
    if "Q6" in q:
        return int(storage_mib * 2.8)
    if "Q5" in q:
        return int(storage_mib * 3.3)
    if "Q4" in q or "IQ4" in q or "UD" in q:
        return int(storage_mib * 4.0)
    if "Q3" in q or "IQ3" in q:
        return int(storage_mib * 5.3)
    if "Q2" in q or "Q1" in q or "IQ2" in q or "IQ1" in q:
        return int(storage_mib * 8.0)
    return int(storage_mib * 4.0)


def _safe_device_budget_mib(
    device: Optional[MemoryDeviceSnapshot],
    hardware: HardwareMemorySnapshot,
) -> Optional[int]:
    if device is None:
        available = hardware.system_available_mib
        total = hardware.system_total_mib
        memory_kind = "system_memory"
    else:
        available = device.free_mib
        total = device.total_mib
        memory_kind = device.memory_kind
    if available is None:
        return None
    if memory_kind == "unified_memory":
        reserve = max(2048, int((total or available) * 0.20))
    elif memory_kind == "system_memory":
        reserve = max(1024, int((total or available) * 0.10))
    else:
        reserve = max(2048, int((total or available) * 0.10))
    return max(0, int(available) - reserve)


def _combined_confidence(model: ModelMemoryEstimate, safe_device_mib: Optional[int]) -> str:
    if safe_device_mib is None:
        return "low"
    if model.confidence == "high":
        return "high"
    if model.total_storage_mib() is not None:
        return "medium"
    return "low"


def _can_select_bnb_nf4(hardware: HardwareMemorySnapshot) -> bool:
    return hardware.backend == "cuda"
