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

# A flat allowance for the pipeline's fixed costs (scheduler, embeddings, the
# CUDA context, fragmentation) on top of the model weights and per-step runtime.
DEFAULT_BASE_OVERHEAD_MIB = 2048


def normalize_memory_mode(value: Optional[str]) -> Optional[str]:
    """Lower/strip a requested mode, accepting dashes; None passes through.

    Raises ValueError for an unsupported mode. The route already rejects bad values
    at the Pydantic Literal boundary (422, before any GPU work); this is a defense-in-
    depth guard for direct / script callers (bench, quality harness) that bypass it,
    and runs on the load thread."""
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


@dataclass(frozen = True)
class MemoryPlan:
    """The chosen runtime profile for one load."""

    requested_mode: str
    offload_policy: str
    vae_tiling: bool
    vae_slicing: bool
    device_memory: DeviceMemory
    reasons: tuple[str, ...] = ()


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


def infer_gguf_quant_label(filename: Optional[str]) -> Optional[str]:
    """Pull a quant tag (Q4_K_M, Q8_0, BF16, ...) out of a GGUF filename."""
    if not filename:
        return None
    from pathlib import Path

    stem = Path(filename).name
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    parts = [p.upper() for p in stem.replace("-", "_").split("_") if p]
    for index, part in enumerate(parts):
        if part in ("BF16", "F16", "FP16", "FP8", "Q8", "Q6", "Q5", "Q4", "Q3", "Q2"):
            suffix = parts[index + 1 :]
            # Quant names carry either a K-family suffix (Q4_K_M) or a legacy
            # numeric one (Q8_0, Q5_1); keep up to two suffix tokens.
            if suffix and suffix[0] in ("K", "M", "S", "L", "XS", "XXS", "0", "1"):
                return "_".join([part] + suffix[:2])
            return part
        if part.startswith("IQ") or part.startswith("UD"):
            return "_".join(parts[index : index + 3])
    return None


def estimate_gguf_dense_mib(storage_mib: Optional[int], quant: Optional[str]) -> Optional[int]:
    """Approximate the dequantised (device) size of a GGUF from its on-disk size
    and quant label. The compute dtype is bf16/fp16, so a 4-bit file roughly
    quadruples once unpacked; higher-bit quants expand less."""
    if storage_mib is None:
        return None
    q = (quant or "").upper()
    if any(t in q for t in ("BF16", "F16", "FP16")):
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
    return int(storage_mib * 4.0)  # unknown: assume 4-bit-ish


def estimate_image_runtime_mib(*, family: Optional[str] = None) -> int:
    """Per-call activation / latent headroom for an image generation (at the
    default ~1MP resolution the planner budgets for). Distilled / turbo models
    (few steps, no CFG) need less; editing pipelines need more."""
    fam = (family or "").lower()
    multiplier = 1.0
    if "edit" in fam:
        multiplier *= 1.35
    if "turbo" in fam or "distilled" in fam or "schnell" in fam:
        multiplier *= 0.85
    return max(1024, int(8192 * multiplier))


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

    # Legacy cpu_offload=True means whole-module offload. Honor it over the AUTO plan
    # whenever auto chose something lighter (resident OR streamed group offload), so the
    # flag keeps its historical meaning on a tight card instead of silently degrading to
    # group offload. An explicit memory_mode (fast/balanced/low_vram) still wins.
    if (
        explicit_offload
        and mode == MEMORY_MODE_AUTO
        and policy != OFFLOAD_MODEL
        and can_offload
        and not device_memory.is_unified
    ):
        policy = OFFLOAD_MODEL
        reasons.append("explicit cpu_offload requests whole-module offload")

    # VAE tiling/slicing decode the image in chunks, capping the decode-time spike
    # that often dominates peak VRAM at high resolution. Turn it on whenever weights
    # are being offloaded (the device is already tight) or the backend has no spare
    # device pool (MPS/CPU). On a roomy discrete GPU it stays off so output is
    # bit-identical to a plain resident run.
    tile = policy != OFFLOAD_NONE or device_memory.backend in ("mps", "cpu")
    return MemoryPlan(
        requested_mode = mode,
        offload_policy = policy,
        vae_tiling = tile,
        vae_slicing = tile,
        device_memory = device_memory,
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

    policy = plan.offload_policy
    if policy == OFFLOAD_MODEL:
        pipe.enable_model_cpu_offload()
    elif policy == OFFLOAD_GROUP:
        if not _apply_group_offload(pipe, device, logger):
            pipe.enable_model_cpu_offload()
            policy = OFFLOAD_MODEL
    elif policy == OFFLOAD_SEQUENTIAL:
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception as exc:  # noqa: BLE001 — keep the model loadable
            if logger is not None:
                logger.warning(
                    "diffusion.memory: sequential offload failed (%s); "
                    "falling back to whole-module offload",
                    exc,
                )
            pipe.enable_model_cpu_offload()
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
        import torch
        from diffusers.hooks import apply_group_offloading

        onload = torch.device(device)
        use_stream = onload.type == "cuda"  # overlap H2D copies with compute on CUDA
        apply_group_offloading(
            transformer,
            onload_device = onload,
            offload_device = torch.device("cpu"),
            offload_type = "block_level",
            num_blocks_per_group = DEFAULT_GROUP_BLOCKS,
            use_stream = use_stream,
        )
        # Place the remaining (smaller) components resident; the streamed
        # transformer manages its own placement via the offloading hooks.
        for name, comp in getattr(pipe, "components", {}).items():
            if name == "transformer":
                continue
            if isinstance(comp, torch.nn.Module):
                comp.to(onload)
        return True
    except Exception as exc:  # noqa: BLE001 — fall back to whole-module offload
        if logger is not None:
            logger.warning(
                "diffusion.memory: group offload failed (%s); falling back to "
                "whole-module offload",
                exc,
            )
        return False
