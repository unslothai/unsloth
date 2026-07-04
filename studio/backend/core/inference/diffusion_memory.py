# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Measured-budget memory policy for the local diffusion backend.

Given the resolved device target (from ``diffusion_device``), a snapshot of free
device memory, and a coarse estimate of the model's device footprint, this picks
a CPU-offload policy and whether to slice/tile the VAE, then applies them to a
built diffusers pipeline. It keeps a model that would not fit resident running by
streaming weights through the GPU one module at a time, which is lossless: offload
and VAE slicing change *placement / decode chunking*, not numerics.

The choice is deliberately coarse (it sizes the model, not every activation), so
``auto`` is best-effort and the explicit ``fast`` / ``balanced`` / ``low_vram``
modes give the operator a hard override. torch / psutil are imported lazily so the
module stays importable in a no-torch runtime (mirrors ``diffusion_device.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


# ── memory modes (operator intent) ───────────────────────────────────────────
MEMORY_MODE_AUTO = "auto"
MEMORY_MODE_FAST = "fast"
MEMORY_MODE_BALANCED = "balanced"
MEMORY_MODE_LOW_VRAM = "low_vram"
MEMORY_MODES = (
    MEMORY_MODE_AUTO,
    MEMORY_MODE_FAST,
    MEMORY_MODE_BALANCED,
    MEMORY_MODE_LOW_VRAM,
)

# ── offload policies (what the loader actually does) ──────────────────────────
# none   -> all weights resident on the device (fastest; fits only if there is
#           room). model -> diffusers enable_model_cpu_offload(): one top-level
#           module on the GPU at a time (modest VRAM cut, small speed cost).
# group  -> apply_group_offloading() on the transformer: stream it a few blocks at
#           a time with a prefetch stream (lowest practical VRAM for the dominant
#           module, less penalty than submodule sequential). sequential ->
#           enable_sequential_cpu_offload(): submodule-level (broken for GGUF on
#           diffusers 0.38, kept only as an explicit escape hatch).
OFFLOAD_NONE = "none"
OFFLOAD_MODEL = "model"
OFFLOAD_GROUP = "group"
OFFLOAD_SEQUENTIAL = "sequential"

# Blocks of the transformer kept resident per group under group offloading: fewer
# = lower peak VRAM, more host<->device traffic. One is the lowest-VRAM setting.
DEFAULT_GROUP_BLOCKS = 1

DEFAULT_IMAGE_WIDTH = 1024
DEFAULT_IMAGE_HEIGHT = 1024
# A flat allowance for the pipeline's fixed costs (scheduler, embeddings, the
# CUDA context, fragmentation) on top of the model weights and per-step runtime.
DEFAULT_BASE_OVERHEAD_MIB = 2048


def normalize_memory_mode(value: Optional[str]) -> Optional[str]:
    """Lower/strip a requested mode, accepting dashes; None passes through.

    Raises ValueError for an unsupported mode so a bad request can be rejected
    cheaply (the route surfaces it as a 4xx before any GPU work)."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized:
        return None
    if normalized not in MEMORY_MODES:
        valid = ", ".join(MEMORY_MODES)
        raise ValueError(f"Unsupported diffusion memory_mode '{value}'. Use one of: {valid}.")
    return normalized


@dataclass(frozen = True)
class DeviceMemory:
    """A point-in-time view of the active device's memory, in MiB.

    ``memory_kind`` distinguishes discrete VRAM (CPU offload helps) from unified /
    system memory (CPU offload moves bytes within the same pool, so it does not)."""

    backend: str
    device: str
    memory_kind: str  # "discrete_vram" | "unified_memory" | "system_memory" | "unknown"
    free_mib: Optional[int] = None
    total_mib: Optional[int] = None

    @property
    def is_unified(self) -> bool:
        return self.memory_kind in ("unified_memory", "system_memory")

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "memory_kind": self.memory_kind,
            "free_mib": self.free_mib,
            "total_mib": self.total_mib,
        }


@dataclass(frozen = True)
class MemoryPlan:
    """The chosen runtime profile for one load."""

    requested_mode: str
    offload_policy: str
    vae_tiling: bool
    vae_slicing: bool
    device_memory: DeviceMemory
    estimates: dict[str, Optional[int]]
    reasons: tuple[str, ...] = ()

    @property
    def engages_offload(self) -> bool:
        return self.offload_policy in (OFFLOAD_MODEL, OFFLOAD_SEQUENTIAL)

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "offload_policy": self.offload_policy,
            "vae_tiling": self.vae_tiling,
            "vae_slicing": self.vae_slicing,
            "device_memory": self.device_memory.as_public_dict(),
            "estimates": dict(self.estimates),
            "reasons": list(self.reasons),
        }


# ── hardware snapshot ─────────────────────────────────────────────────────────


def snapshot_device_memory(target: Any) -> DeviceMemory:
    """Free / total memory for the device named by ``target`` (a
    ``DiffusionDeviceTarget``). Never raises: any probe failure yields None
    counts, which the planner treats as "budget unknown" (stay resident)."""
    device = getattr(target, "device", "cpu")
    backend = getattr(target, "backend", device)

    if device == "cuda":
        free, total, kind = _cuda_memory(backend)
        return DeviceMemory(backend, device, kind, free, total)
    if device == "xpu":
        free, total = _xpu_memory()
        return DeviceMemory(backend, device, "discrete_vram", free, total)
    if device == "mps":
        # Apple Silicon shares one pool between CPU and GPU; system memory is the
        # right budget and CPU offload is pointless there.
        total, free = _system_memory_mib()
        return DeviceMemory(backend, device, "unified_memory", free, total)

    total, free = _system_memory_mib()
    return DeviceMemory(backend, device, "system_memory", free, total)


def _cuda_memory(backend: str) -> tuple[Optional[int], Optional[int], str]:
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
        kind = "discrete_vram"
        try:
            # Query the CURRENT device, not device 0: mem_get_info() above already
            # reports the active device, so hardcoding 0 would inspect the wrong GPU
            # (and misclassify discrete vs unified) when the active device isn't 0.
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            if bool(getattr(props, "integrated", False) or getattr(props, "is_integrated", False)):
                kind = "unified_memory"  # e.g. Jetson / integrated SoC
        except Exception:
            pass
        return int(free // (1024 * 1024)), int(total // (1024 * 1024)), kind
    except Exception:
        return None, None, "discrete_vram"


def _xpu_memory() -> tuple[Optional[int], Optional[int]]:
    try:
        import torch
        mem_get_info = getattr(getattr(torch, "xpu", None), "mem_get_info", None)
        if callable(mem_get_info):
            free, total = mem_get_info()
            return int(free // (1024 * 1024)), int(total // (1024 * 1024))
    except Exception:
        pass
    return None, None


def _system_memory_mib() -> tuple[Optional[int], Optional[int]]:
    """(total, available) host RAM in MiB, via psutil then POSIX sysconf."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.total // (1024 * 1024)), int(vm.available // (1024 * 1024))
    except Exception:
        pass
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page
        avail = os.sysconf("SC_AVPHYS_PAGES") * page
        return int(total // (1024 * 1024)), int(avail // (1024 * 1024))
    except Exception:
        return None, None


# ── size estimates ────────────────────────────────────────────────────────────


def file_size_mib(path: Any) -> Optional[int]:
    """On-disk size of ``path`` in MiB, or None if it can't be stat'd."""
    try:
        from pathlib import Path
        return max(1, int(Path(path).expanduser().stat().st_size // (1024 * 1024)))
    except Exception:
        return None


def estimate_gguf_resident_mib(storage_mib: Optional[int]) -> Optional[int]:
    """Approximate the RESIDENT device size of a GGUF transformer loaded through
    diffusers' ``GGUFQuantizationConfig``.

    The weights stay PACKED on the device as quantised bytes (``GGUFParameter`` /
    uint8); ``GGUFLinear.forward`` dequantises each weight to the bf16 compute dtype
    transiently for its matmul and frees it immediately, so the persistent footprint
    is ~= the on-disk tensor size, NOT the unpacked bf16 size. Measured on
    Z-Image-Turbo: Q2_K 3.64 GiB -> 3.68 GiB, Q8_0 7.22 GiB -> 7.25 GiB resident.
    The transient per-op dequant is covered by the separate runtime headroom.

    (The prior per-quant expansion assumed a full unpack that never happens on this
    path; it over-estimated e.g. Q2 ~7.6x, forcing needless offload.)"""
    if storage_mib is None:
        return None
    return int(storage_mib * 1.05)  # small margin for allocator + bf16 norms/biases


def estimate_safetensors_dense_mib(storage_mib: Optional[int]) -> Optional[int]:
    """Resident size of a safetensors checkpoint, in MiB.

    Unlike a GGUF (which is dequantised to bf16/fp16 on load, so a 4-bit file
    expands ~4x), a safetensors checkpoint loads near its on-disk size: a dense
    bf16 file is already bf16, and a bnb-4bit / fp8 file stays compressed in VRAM.
    So the on-disk size is the estimate, returned unchanged (None passes through).
    """
    return storage_mib


def estimate_image_runtime_mib(
    *,
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
    family: Optional[str] = None,
) -> int:
    """Per-call activation / latent headroom for an image generation, scaled by
    pixel area and batch. Distilled / turbo models (few steps, no CFG) need less."""
    w = max(64, int(width or DEFAULT_IMAGE_WIDTH))
    h = max(64, int(height or DEFAULT_IMAGE_HEIGHT))
    batch = max(1, int(batch_size or 1))
    pixel_scale = (w * h * batch) / float(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT)
    fam = (family or "").lower()
    multiplier = 1.0
    if "edit" in fam:
        multiplier *= 1.35
    if "turbo" in fam or "distilled" in fam or "schnell" in fam:
        multiplier *= 0.85
    return max(1024, int(8192 * max(0.25, pixel_scale) * multiplier))


def estimate_video_runtime_mib(
    *,
    width: Optional[int],
    height: Optional[int],
    num_frames: Optional[int],
) -> int:
    """Per-call activation / latent / decode headroom for a video generation.

    The image estimator is pixel-area only and badly undershoots video: latents
    carry a frames dimension and the VAE DECODE is the peak -- the decoded clip
    materialises as num_frames full-resolution fp32 frames plus the decoder's
    intermediates, typically dwarfing the denoise activations. Scale by the
    decoded-clip footprint (frames x H x W x 3 x 4 bytes) with a 3x factor for
    decoder intermediates + the PIL/tensor copy held during export, on top of a
    fixed denoise-side base.
    """
    w = max(64, int(width or 768))
    h = max(64, int(height or 512))
    frames = max(1, int(num_frames or 121))
    decoded_mib = (frames * w * h * 3 * 4) / float(1024 * 1024)
    return max(3072, int(4096 + 3.0 * decoded_mib))


def _safe_device_budget_mib(memory: DeviceMemory) -> Optional[int]:
    """Free memory minus a headroom reserve, so a plan that "fits" leaves room for
    fragmentation and other tenants. None when free memory is unknown."""
    if memory.free_mib is None:
        return None
    base = memory.total_mib or memory.free_mib
    if memory.memory_kind == "unified_memory":
        reserve = max(2048, int(base * 0.20))  # the OS + CPU share this pool
    elif memory.memory_kind == "system_memory":
        reserve = max(1024, int(base * 0.10))
    else:
        reserve = max(2048, int(base * 0.10))
    return max(0, int(memory.free_mib) - reserve)


def _sum_required(*values: Optional[int]) -> Optional[int]:
    total = 0
    for value in values:
        if value is None:
            return None
        total += int(value)
    return total


# ── the planner ───────────────────────────────────────────────────────────────


def plan_diffusion_memory(
    *,
    target: Any,
    device_memory: DeviceMemory,
    model_dense_mib: Optional[int],
    runtime_headroom_mib: int,
    companion_dense_mib: Optional[int] = None,
    base_overhead_mib: int = DEFAULT_BASE_OVERHEAD_MIB,
    requested_mode: Optional[str] = None,
    explicit_offload: bool = False,
) -> MemoryPlan:
    """Pick an offload policy + VAE memory savers for the current load.

    ``model_dense_mib`` is the estimated resident device size of all weights
    (transformer + companion text-encoder / VAE); ``companion_dense_mib`` is just
    the companions, which stay resident under streamed (group) offload while the
    transformer is streamed block by block. ``explicit_offload`` is the back-compat
    ``cpu_offload=True`` request: it forces whole-module offload.

    Policy meanings, ordered by measured speed/VRAM tradeoff:
      none   - everything resident: fastest, highest VRAM.
      group  - stream the transformer, companions resident: near-resident speed,
               moderate VRAM cut (the balanced tradeoff).
      model  - offload every component incl. the text encoder: lowest VRAM, slow.
    """
    mode = normalize_memory_mode(requested_mode) or MEMORY_MODE_AUTO
    can_offload = bool(getattr(target, "supports_model_cpu_offload", False))
    budget = _safe_device_budget_mib(device_memory)
    required = _sum_required(model_dense_mib, runtime_headroom_mib, base_overhead_mib)
    # The resident floor under group offload: companions stay, the transformer streams.
    group_floor = _sum_required(companion_dense_mib, runtime_headroom_mib, base_overhead_mib)
    reasons: list[str] = []
    estimates: dict[str, Optional[int]] = {
        "safe_device_budget_mib": budget,
        "model_dense_mib": model_dense_mib,
        "companion_dense_mib": companion_dense_mib,
        "runtime_headroom_mib": runtime_headroom_mib,
        "base_overhead_mib": base_overhead_mib,
        "resident_required_mib": required,
        "group_floor_mib": group_floor,
    }

    def _group_fits() -> bool:
        # Group offload only helps if the resident remainder (companions) fits; when
        # the text encoder itself is too big, only whole-module offload will do.
        return group_floor is not None and budget is not None and group_floor <= budget

    if not can_offload or device_memory.is_unified:
        # MPS / CPU can't stream to a separate device, and on unified / system
        # memory CPU offload just shuffles bytes within the same pool.
        policy = OFFLOAD_NONE
        if device_memory.is_unified:
            reasons.append("unified/system memory: CPU offload frees no device memory")
        else:
            reasons.append(f"{device_memory.backend}: CPU offload unavailable; staying resident")
    elif mode == MEMORY_MODE_FAST:
        policy = OFFLOAD_NONE
        if budget is not None and required is not None and required > budget:
            # Doesn't fit resident: the fastest offload is the streamed transformer.
            policy = OFFLOAD_GROUP if _group_fits() else OFFLOAD_MODEL
            reasons.append("fast requested but weights do not fit resident; offloading")
        else:
            reasons.append("fast requested; weights resident on device")
    elif mode == MEMORY_MODE_BALANCED:
        policy = OFFLOAD_GROUP
        reasons.append("balanced requested; streamed block-level transformer offload")
    elif mode == MEMORY_MODE_LOW_VRAM:
        policy = OFFLOAD_MODEL
        reasons.append("low_vram requested; whole-module offload of every component")
    elif budget is None or required is None:
        policy = OFFLOAD_NONE
        reasons.append("device budget or model size unknown; staying resident")
    elif required <= int(budget * 0.85):
        policy = OFFLOAD_NONE
        reasons.append("weights fit resident with headroom")
    elif _group_fits():
        policy = OFFLOAD_GROUP
        reasons.append("tight fit; stream the transformer, companions resident")
    else:
        policy = OFFLOAD_MODEL
        reasons.append("companions exceed budget; whole-module offload of every component")

    # The legacy cpu_offload flag only applies when NO memory_mode was supplied:
    # the API documents memory_mode as overriding cpu_offload when set, so an
    # explicit `fast` request must stay resident even if the caller also left the
    # old flag enabled.
    if (
        explicit_offload
        and normalize_memory_mode(requested_mode) is None
        and policy == OFFLOAD_NONE
        and can_offload
        and not device_memory.is_unified
    ):
        policy = OFFLOAD_MODEL
        reasons.append("explicit cpu_offload overrides resident placement")

    # VAE savers cap the decode-time spike that dominates peak VRAM at high res.
    # Slicing (decode a batch one image at a time) is EXACT, so enable it on any
    # offload tier / non-discrete backend. Tiling (spatial chunks) is only bit-
    # identical for a single tile (<=1MP), so restrict it to the lowest tiers where
    # the VAE itself is offloaded (model / sequential) or there is no spare device
    # pool (MPS / CPU). Under group offload the transformer streams but the VAE stays
    # resident and fits, so it keeps exact full-image decode -> balanced is both
    # faster and bit-identical. On a roomy discrete GPU both stay off.
    any_offload = policy != OFFLOAD_NONE or device_memory.backend in ("mps", "cpu")
    tile = policy in (OFFLOAD_MODEL, OFFLOAD_SEQUENTIAL) or device_memory.backend in ("mps", "cpu")
    return MemoryPlan(
        requested_mode = mode,
        offload_policy = policy,
        vae_tiling = tile,
        vae_slicing = any_offload,
        device_memory = device_memory,
        estimates = estimates,
        reasons = tuple(reasons),
    )


# ── apply to a built pipeline ─────────────────────────────────────────────────


def apply_memory_plan(
    pipe: Any,
    plan: MemoryPlan,
    *,
    device: str,
    logger: Any = None,
) -> tuple[str, bool]:
    """Apply ``plan`` to a freshly built diffusers pipeline: enable the VAE memory
    savers then place / offload the weights. Exactly one placement call runs, so the
    pipeline ends up either fully resident or wired for offload, never both.

    Returns ``(offload_policy, vae_tiling)`` ACTUALLY engaged, which can differ from
    the plan: VAE tiling is a no-op on a pipeline that exposes no tiling control, and
    block-level / sequential offload fall back to the robust whole-module offload if
    the transformer doesn't support them (e.g. submodule sequential is broken for
    GGUF on diffusers 0.38). Status then reflects what really happened."""
    tiling_engaged = False
    if plan.vae_tiling:
        tiling_engaged = _enable_vae_saver(pipe, "enable_vae_tiling", "enable_tiling", logger)
    if plan.vae_slicing:
        _enable_vae_saver(pipe, "enable_vae_slicing", "enable_slicing", logger)

    def _fallback_to_model_offload() -> None:
        # Group offload keeps the VAE resident, so the GROUP plan set vae_tiling=False.
        # When group offload is unavailable and we drop to whole-module offload, the card
        # is in the low-VRAM situation where the decode-time spike can OOM, so turn VAE
        # tiling on now (if not already engaged) to cap it.
        nonlocal tiling_engaged
        pipe.enable_model_cpu_offload(device = device)
        if not tiling_engaged:
            tiling_engaged = _enable_vae_saver(pipe, "enable_vae_tiling", "enable_tiling", logger)

    policy = plan.offload_policy
    if policy == OFFLOAD_MODEL:
        pipe.enable_model_cpu_offload(device = device)
    elif policy == OFFLOAD_GROUP:
        if not _apply_group_offload(pipe, device, logger):
            _fallback_to_model_offload()
            policy = OFFLOAD_MODEL
    elif policy == OFFLOAD_SEQUENTIAL:
        try:
            pipe.enable_sequential_cpu_offload(device = device)
        except Exception as exc:  # noqa: BLE001 — keep the model loadable
            if logger is not None:
                logger.warning(
                    "diffusion.memory: sequential offload failed (%s); "
                    "falling back to whole-module offload",
                    exc,
                )
            _fallback_to_model_offload()
            policy = OFFLOAD_MODEL
    else:
        pipe.to(device)
    return policy, tiling_engaged


def _enable_vae_saver(pipe: Any, pipe_method: str, vae_method: str, logger: Any) -> bool:
    """Turn on a VAE memory saver, trying the pipeline-level shortcut first and the
    VAE submodule directly otherwise (some pipelines, e.g. Z-Image, only expose it
    on ``pipe.vae``). Returns whether it actually engaged."""
    for owner, method in ((pipe, pipe_method), (getattr(pipe, "vae", None), vae_method)):
        fn = getattr(owner, method, None)
        if not callable(fn):
            continue
        try:
            fn()
            return True
        except Exception as exc:  # noqa: BLE001 — a VAE saver is an optimisation, never fatal
            if logger is not None:
                logger.warning("diffusion.memory: %s() failed: %s", method, exc)
    return False


def _apply_group_offload(pipe: Any, device: str, logger: Any) -> bool:
    """Stream the transformer through the device a few blocks at a time via
    diffusers group offloading, keeping the smaller components resident. Returns
    False (so the caller falls back to whole-module offload) on any failure."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return False
    try:
        import inspect

        import torch
        from diffusers.hooks import apply_group_offloading

        onload = torch.device(device)
        use_stream = onload.type == "cuda"  # overlap H2D copies with compute on CUDA
        gkwargs: dict[str, Any] = {
            "onload_device": onload,
            "offload_device": torch.device("cpu"),
            "offload_type": "block_level",
            "num_blocks_per_group": DEFAULT_GROUP_BLOCKS,
            "use_stream": use_stream,
        }
        # On the CUDA stream path, overlap each block's host->device copy with
        # compute: non_blocking issues the copy asynchronously and record_stream
        # defers the free until the copy's stream is done. Lossless (only transfer
        # scheduling changes). Safe for the group tier specifically, where the
        # companions stay resident; gated on the installed signature so an older
        # diffusers that lacks these kwargs still works (no hard fallback).
        if use_stream:
            _params = inspect.signature(apply_group_offloading).parameters
            if "non_blocking" in _params:
                gkwargs["non_blocking"] = True
            if "record_stream" in _params:
                gkwargs["record_stream"] = True
        # Place the remaining (smaller) components resident BEFORE attaching the
        # transformer's group-offload hooks. If a companion .to() OOMs we return False
        # with NO hooks installed, so the caller's whole-module offload fallback works:
        # diffusers REJECTS enable_model_cpu_offload on a pipeline that already carries
        # group-offload hooks, which would otherwise turn the intended fallback into a
        # load-time crash. The streamed transformer manages its own placement via the
        # offloading hooks applied next.
        for name, comp in getattr(pipe, "components", {}).items():
            if name == "transformer":
                continue
            if isinstance(comp, torch.nn.Module):
                comp.to(onload)
        apply_group_offloading(transformer, **gkwargs)
        return True
    except Exception as exc:  # noqa: BLE001 — fall back to whole-module offload
        if logger is not None:
            logger.warning(
                "diffusion.memory: group offload failed (%s); falling back to "
                "whole-module offload",
                exc,
            )
        return False
