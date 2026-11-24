# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Host RAM a video load needs on CPU-only torch, priced before anything downloads.

With no GPU PyTorch can use, a video pipeline resolves to the float32 CPU target: every dense component is upcast
from its stored bf16 to float32 and held in anonymous host memory, which nothing can page out. A GGUF denoiser is read
as views of its file mapping (page cache the kernel can drop and re-read), so it adds no anonymous floor. A host short
of the floor is OOM-killed mid-load or mid-render, after a multi-GB download; this refuses it up front instead.

The estimate is a floor, not a forecast, so a load that fits is never refused: unmeasured routes take the cheapest
reading of each term (a hosted fp8 / int8 encoder at its stored size even if CPU later widens it). Calibration, Colab
CPU-HM (8 vCPU, 50 GiB), LTX-2.5 distilled UD-Q4_K_XL GGUF at 768x512x33, 4 steps, process tree RssAnon:
28.7 GB peak uncapped (42.1 GB RSS with the mapped GGUF), 29.2 GB peak with only 32.2 GB available (rendered).
This module prices that load at 28.4 GB. MiniMax-H3 GGUF runs on the native engine (sd.cpp) on such a host and keeps
its weights mapped: 5.1 GB anonymous peak at 864x480x124 (23.9 GB RSS).

``UNSLOTH_DIFFUSION_HOST_RAM_CHECK=0`` (the image and LTX-2.5 fp16 host checks' switch) turns this off."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

# LTX-2.5's Gemma 4 encoder worker holding the hosted int8 ConvRot file on CPU: 8,087 MiB peak worker RSS (measured,
# CPU-HM). Lower than the 13.1 GB file because the worker reads it through a mapping.
LTX25_CPU_TE_INT8_HOST_GB = 8.5
# MiniMax-H3 GGUF on the native engine on CPU: whole render tree RssAnon peak, 864x480x124 (measured, CPU-HM).
H3_CPU_NATIVE_HOST_GB = 5.1
# Activations, GGUF dequant scratch and the tiled VAE decode for a short clip at the smallest size: the LTX-2.5 peak
# above minus its resident weights (28.7 - 8.5 - 18.4 = 1.8 GB), rounded down so the estimate stays under it.
CPU_VIDEO_WORKING_SET_GB = 1.5

_MIB = 1 << 20
_GB_PER_MIB = _MIB / 1e9


@dataclass(frozen = True)
class CpuVideoHostNeed:
    """Decimal GB of anonymous host memory, per term. ``denoiser_mapped``: a GGUF read from its file mapping."""

    denoiser_gb: float
    text_encoder_gb: float
    other_gb: float
    working_gb: float
    denoiser_mapped: bool = False
    text_encoder_dense: bool = False
    text_encoder_quant_gb: Optional[float] = None
    text_encoder_quant_label: Optional[str] = None
    native: bool = False

    @property
    def total_gb(self) -> float:
        return self.denoiser_gb + self.text_encoder_gb + self.other_gb + self.working_gb


def _sizes(fam: Any) -> Optional[tuple[float, float, float]]:
    sizes = tuple(float(x) for x in (getattr(fam, "bf16_components_gb", None) or ()))
    return sizes if len(sizes) == 3 else None  # type: ignore[return-value]


def _text_encoder_gb(
    fam: Any, text_encoder_quant: Optional[str], bf16_gb: float
) -> tuple[float, bool, Optional[float], Optional[str]]:
    """(host GB, priced dense, quantized alternative GB, its label). Raises ValueError for an unknown scheme."""
    from .diffusion_precision import normalize_te_quant, te_quant_is_auto

    if bf16_gb <= 0:
        return 0.0, False, None, None
    if getattr(fam, "name", None) == "ltx-2.5":
        from .video_ltx25 import ltx25_te_quant_scheme

        # Auto takes the hosted int8 file (the floor); none / off and an Auto that cannot confirm it read bf16,
        # widened to float32 on the CPU host.
        if ltx25_te_quant_scheme(text_encoder_quant):
            return LTX25_CPU_TE_INT8_HOST_GB, False, None, None
        return bf16_gb * 2.0, True, LTX25_CPU_TE_INT8_HOST_GB, "int8"
    auto = getattr(fam, "te_quant_auto", None)
    hosted = bool(getattr(fam, "te_prequant_repos", None))
    if te_quant_is_auto(text_encoder_quant):
        scheme = normalize_te_quant(auto) if auto else None
    else:
        scheme = normalize_te_quant(text_encoder_quant)
    if scheme is not None:
        # A pre-cast fp8 / int8 encoder stores one byte per weight; CPU may widen it later, but never below this.
        return bf16_gb / 2.0, False, None, None
    if auto and hosted:
        return bf16_gb * 2.0, True, bf16_gb / 2.0, str(normalize_te_quant(auto))
    return bf16_gb * 2.0, True, None, None


def _denoiser_gb(
    fam: Any, model_kind: str, transformer_quant: Optional[str], bf16_gb: float
) -> tuple[float, bool]:
    """(host GB, read from a file mapping). Raises ValueError for an unknown scheme."""
    if model_kind == "gguf":
        return 0.0, True
    from .diffusion_transformer_quant import normalize_transformer_quant

    pinned = normalize_transformer_quant(transformer_quant)
    if pinned is not None and pinned != "auto":
        stored = dict(getattr(fam, "prequant_resident_gb_by_scheme", None) or ())
        return float(stored.get(pinned, bf16_gb / 2.0)), False
    return bf16_gb * 2.0, False


def cpu_video_host_need(
    fam: Any,
    *,
    model_kind: str,
    text_encoder_quant: Optional[str] = None,
    transformer_quant: Optional[str] = None,
) -> Optional[CpuVideoHostNeed]:
    """The anonymous host-RAM floor of this load on the float32 CPU target, or None when it cannot be sized."""
    if getattr(fam, "name", None) == "minimax-h3":
        # GGUF runs on the native engine with its weights mapped; a dense H3 pick is refused on CPU elsewhere.
        if model_kind != "gguf":
            return None
        return CpuVideoHostNeed(
            0.0, 0.0, 0.0, H3_CPU_NATIVE_HOST_GB, denoiser_mapped = True, native = True
        )
    sizes = _sizes(fam)
    if sizes is None:
        return None
    try:
        denoiser, mapped = _denoiser_gb(fam, model_kind, transformer_quant, sizes[0])
        te, te_dense, te_alt, te_alt_label = _text_encoder_gb(fam, text_encoder_quant, sizes[1])
    except ValueError:
        return None  # an unsupported scheme is refused by the precision checks with its own message
    # The table's third term is bf16-resident except where the family already keeps its VAE in float32.
    other = sizes[2] * (1.0 if getattr(fam, "vae_force_fp32", False) else 2.0)
    return CpuVideoHostNeed(
        denoiser,
        te,
        other,
        CPU_VIDEO_WORKING_SET_GB,
        denoiser_mapped = mapped,
        text_encoder_dense = te_dense,
        text_encoder_quant_gb = te_alt,
        text_encoder_quant_label = te_alt_label,
    )


def _status_held_bytes(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/status", encoding = "utf-8") as handle:
            found = {}
            for line in handle:
                key, _, value = line.partition(":")
                if key in ("RssAnon", "RssShmem"):
                    found[key] = int(value.split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return (found["RssAnon"] + found["RssShmem"]) * 1024 if len(found) == 2 else None


def _process_tree_held_mib() -> int:
    """Anonymous + shmem MiB held by this process and its children (encoder workers, engines a load evicts). Credited
    back to the reading so a model this load replaces is not counted against it: the bias is toward admitting."""
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return 0
    try:
        me = psutil.Process()
        procs = [me]
        try:
            procs += me.children(recursive = True)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        return 0
    total = 0
    for proc in procs:
        held = _status_held_bytes(proc.pid)
        if held is None:
            try:
                held = int(proc.memory_info().rss)
            except Exception:  # noqa: BLE001
                held = 0
        total += held
    return total // _MIB


def cpu_video_host_capacity_mib() -> Optional[int]:
    """Host MiB a CPU video load can take: available RAM (capped by an enforcing cgroup's headroom and the
    ``UNSLOTH_DIFFUSION_HOST_RAM_LIMIT_MIB`` test hook) plus what this process tree already holds, capped at the
    capacity (system total or cgroup limit). None when available RAM cannot be read."""
    from .diffusion_small_host import host_ram_mib

    total, available = host_ram_mib()
    if available is None:
        return None
    capacity = int(available) + _process_tree_held_mib()
    if total:
        capacity = min(capacity, int(total))
    return capacity


def _advice(fam: Any, need: CpuVideoHostNeed, capacity_gb: float) -> str:
    gguf_repo = getattr(fam, "gguf_repo", None)
    has_gguf = bool(gguf_repo) or bool(getattr(fam, "is_moe", False))
    options = []
    floor = need.total_gb
    if not need.denoiser_mapped and has_gguf:
        floor -= need.denoiser_gb
        where = f" from '{gguf_repo}'" if gguf_repo else ""
        options.append(f"load a GGUF of it{where}, whose denoiser is read from disk as it runs")
    if need.text_encoder_dense and need.text_encoder_quant_gb is not None:
        floor -= need.text_encoder_gb - need.text_encoder_quant_gb
        options.append(
            f"leave the text encoder precision on Auto for the {need.text_encoder_quant_label} text encoder "
            f"(about {need.text_encoder_quant_gb:.1f} GB instead of {need.text_encoder_gb:.1f})"
        )
    tail = (
        "Free memory by closing other applications or unloading other models, or use a machine with more RAM or a "
        "supported GPU."
    )
    if options:
        joined = options[0] if len(options) == 1 else f"{options[0]}, and {options[1]}"
        if floor <= capacity_gb:
            return f"To fit, {joined}: about {floor:.1f} GB."
        return (
            f"Even if you {joined}, it needs about {floor:.1f} GB, more than is available. {tail}"
        )
    if need.denoiser_mapped and not need.native:
        return (
            f"Its text encoder, VAE and other components plus the working set need about {need.total_gb:.1f} GB "
            f"in float32 whatever the denoiser, so a lower GGUF quant does not fit either. {tail}"
        )
    return f"No smaller build of it fits this device. {tail}"


def cpu_video_host_shortfall(
    fam: Any,
    *,
    model_kind: str,
    text_encoder_quant: Optional[str] = None,
    transformer_quant: Optional[str] = None,
) -> Optional[str]:
    """The refusal when this load's CPU host floor exceeds the host RAM it can take, else None. Call only on the
    float32 CPU target. Never raises; anything it cannot read refuses nothing."""
    try:
        from .diffusion_small_host import HOST_RAM_CHECK_ENV, host_ram_check_disabled

        if host_ram_check_disabled():
            return None
        need = cpu_video_host_need(
            fam,
            model_kind = model_kind,
            text_encoder_quant = text_encoder_quant,
            transformer_quant = transformer_quant,
        )
        if need is None:
            return None
        capacity_mib = cpu_video_host_capacity_mib()
        if capacity_mib is None:
            return None
        capacity_gb = capacity_mib * _GB_PER_MIB
        if need.total_gb <= capacity_gb:
            return None
        label = getattr(fam, "label", None) or getattr(fam, "name", None) or "This model"
        if need.native:
            how = "runs on the CPU through the native engine and"
            parts = f"its working set at the default size, measured: {need.working_gb:.1f}"
        else:
            how = "runs on the CPU in float32 and"
            denoiser = (
                "denoiser read from its GGUF file"
                if need.denoiser_mapped
                else f"denoiser {need.denoiser_gb:.1f}"
            )
            parts = (
                f"{denoiser}, text encoder {need.text_encoder_gb:.1f}, VAE and other components "
                f"{need.other_gb:.1f}, working set {need.working_gb:.1f}"
            )
        advice = _advice(fam, need, capacity_gb)
        scope = (
            "The estimate was measured at 864x480 with 124 frames; larger sizes and longer clips need more."
            if need.native
            else "The estimate is for a short clip at the smallest size; longer clips and larger sizes need more."
        )
    except Exception:  # noqa: BLE001 -- a probe must never refuse a load it cannot size
        return None
    return (
        f"Not enough system RAM to load {label} on this device: it has no GPU PyTorch can use, so the model {how} "
        f"needs about {need.total_gb:.1f} GB of system RAM ({parts}), but about {capacity_gb:.1f} GB is available. "
        f"{advice} {scope} Set {HOST_RAM_CHECK_ENV}=0 to try anyway."
    )


def refuse_cpu_video_host_shortfall(
    fam: Any,
    *,
    model_kind: str,
    text_encoder_quant: Optional[str] = None,
    transformer_quant: Optional[str] = None,
    logger: Any = None,
    resolve_target: Optional[Callable[[], Any]] = None,
) -> None:
    """RuntimeError (the routes' 409) for a video load the CPU host cannot hold; a no-op unless the diffusion target is
    the CPU, so GPU / MPS / XPU loads never reach the sizing. Runs before eviction and before any download.
    ``resolve_target``: the caller's own target resolver (the one its other precision checks use)."""
    from . import diffusion_device

    # Probe-free first, so an accelerator host never runs the dtype probes; then the target the loader itself uses.
    if not diffusion_device.diffusion_target_is_cpu():
        return
    target = (resolve_target or diffusion_device.resolve_diffusion_device_target)()
    if getattr(target, "backend", None) != "cpu":
        return
    shortfall = cpu_video_host_shortfall(
        fam,
        model_kind = model_kind,
        text_encoder_quant = text_encoder_quant,
        transformer_quant = transformer_quant,
    )
    if shortfall is None:
        return
    if logger is not None:
        logger.error("video.cpu_host_ram: refusing load: %s", shortfall)
    raise RuntimeError(shortfall)


__all__ = [
    "CPU_VIDEO_WORKING_SET_GB",
    "CpuVideoHostNeed",
    "H3_CPU_NATIVE_HOST_GB",
    "LTX25_CPU_TE_INT8_HOST_GB",
    "cpu_video_host_capacity_mib",
    "cpu_video_host_need",
    "cpu_video_host_shortfall",
    "refuse_cpu_video_host_shortfall",
]
