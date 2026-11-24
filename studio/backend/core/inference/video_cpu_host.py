# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Host RAM a video load needs on CPU-only torch, priced before anything downloads.

With no GPU PyTorch can use, a video pipeline resolves to the float32 CPU target: every dense component is upcast
from its stored bf16 to float32 and held in anonymous host memory, which nothing can page out. A GGUF denoiser read
through Studio's mmap hook is views of its file mapping (page cache the kernel can drop and re-read), so it adds no
anonymous floor; with the hook off (Windows, ``UNSLOTH_DIFFUSION_GGUF_MMAP=0``) diffusers copies the whole file. A host
short of the floor is OOM-killed mid-load or mid-render, after a multi-GB download; this refuses it up front instead.

The text encoder is priced as the loader settles it (``video.settle_video_te_quant``): on CPU an automatic fp8 / int8
pick is dropped and the released encoder loads in float32. Everything else is a floor, so a load that fits is not
refused. Calibration, Colab CPU-HM (8 vCPU, 50 GiB), LTX-2.5 distilled UD-Q4_K_XL GGUF at 768x512x33, 4 steps,
process tree RssAnon: 28.7 / 29.0 GB peak uncapped (42.1 GB RSS with the mapped GGUF), 29.2 GB with only 32.2 GB
available (rendered). This module prices that load at 28.4 GB. MiniMax-H3 GGUF runs on the native engine (sd.cpp) on
such a host and keeps its weights mapped: 5.1 GB anonymous peak at 864x480x124 (23.9 GB RSS).

``UNSLOTH_DIFFUSION_HOST_RAM_CHECK=0`` (the image and LTX-2.5 fp16 host checks' switch) turns this off."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# LTX-2.5's Gemma 4 encoder worker holding the hosted int8 ConvRot file on CPU: 8,087 MiB peak worker RSS (measured,
# CPU-HM). Lower than the 13.1 GB file because the worker reads it through a mapping.
LTX25_CPU_TE_INT8_HOST_GB = 8.5
# MiniMax-H3 GGUF on the native engine on CPU: whole render tree RssAnon peak, 864x480x124 (measured, CPU-HM).
H3_CPU_NATIVE_HOST_GB = 5.1
# Activations, GGUF dequant scratch and the tiled VAE decode for a short clip at the smallest size: the LTX-2.5 peak
# above minus its resident weights (28.7 - 8.5 - 18.4 = 1.8 GB), rounded down so the estimate stays under it.
CPU_VIDEO_WORKING_SET_GB = 1.5
# Families measured on CPU at their default size, where the working set is larger. Wan2.1-T2V-1.3B pipeline at
# 832x480x33 (2 steps + decode), Colab CPU-HM: 9.45 GB RssAnon / Pss_Anon peak = DiT 5.68 (float32 file) + VAE 0.51
# + 3.26 working set; priced at 3.0 so the estimate (9.1 GB) stays just under the measurement.
CPU_VIDEO_WORKING_SET_GB_BY_FAMILY = {"wan2.1-t2v-1.3b": 3.0}
# Official Diffusers repos whose text_encoder/ is stored in float32 (umT5, 22.72 GB): the float32 CPU target reads it
# as views of the file mapping, no conversion, so it adds no anonymous memory (Wan2.1 1.3B above: 22.8 GB RssFile).
STORED_FP32_TEXT_ENCODER_FAMILIES = frozenset({"wan2.1-t2v-1.3b", "wan2.1-t2v-14b"})


def cpu_video_working_set_gb(fam: Any) -> float:
    return CPU_VIDEO_WORKING_SET_GB_BY_FAMILY.get(
        getattr(fam, "name", None), CPU_VIDEO_WORKING_SET_GB
    )


# A copied GGUF whose size cannot be read: the smallest curated quant's share of bf16 (LTX-2.5 Q2_K 8.46 / 42.0).
GGUF_MIN_BF16_FRACTION = 0.2

_MIB = 1 << 20
_GB_PER_MIB = _MIB / 1e9


@dataclass(frozen = True)
class CpuVideoHostNeed:
    """Decimal GB of anonymous host memory, per term. ``denoiser_mapped``: a GGUF read from its file mapping;
    ``denoiser_gguf_copy``: a GGUF diffusers copies into memory (the hook is off)."""

    denoiser_gb: float
    text_encoder_gb: float
    other_gb: float
    working_gb: float
    denoiser_mapped: bool = False
    denoiser_gguf_copy: bool = False
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
    fam: Any, te_scheme: Optional[str], te_hosted: bool, te_hosted_available: bool, bf16_gb: float
) -> tuple[float, bool, Optional[float], Optional[str]]:
    """(host GB, priced dense, quantized alternative GB, its label) for the scheme the loader settled on."""
    if bf16_gb <= 0:
        return 0.0, False, None, None
    if getattr(fam, "name", None) == "ltx-2.5":
        if te_scheme and te_hosted:
            return LTX25_CPU_TE_INT8_HOST_GB, False, None, None
        # bf16 widened to float32 on the CPU host; an int8 the worker quantizes at load reads that dense copy first.
        alt = LTX25_CPU_TE_INT8_HOST_GB if te_hosted_available else None
        return bf16_gb * 2.0, True, alt, ("int8" if alt is not None else None)
    if te_scheme:
        # A pre-cast fp8 / int8 encoder stores one byte per weight; CPU may widen it later, but never below this.
        return bf16_gb / 2.0, False, None, None
    if getattr(fam, "name", None) in STORED_FP32_TEXT_ENCODER_FAMILIES:
        return 0.0, False, None, None
    return bf16_gb * 2.0, True, None, None


def _denoiser_gb(
    fam: Any,
    model_kind: str,
    transformer_quant: Optional[str],
    bf16_gb: float,
    gguf_copy_gb: Optional[float],
    single_file_packed_gb: Optional[float] = None,
) -> tuple[float, bool]:
    """(host GB, read from a file mapping). Raises ValueError for an unknown scheme."""
    if model_kind == "gguf":
        if gguf_copy_gb is None:
            return 0.0, True
        return float(gguf_copy_gb), False
    from .diffusion_transformer_quant import normalize_transformer_quant

    pinned = normalize_transformer_quant(transformer_quant)
    # The loader keeps the checkpoint's quantization (comfy_keep) unless the request pins the dense DiT ("none" / "off").
    keeps_packed = transformer_quant is None or pinned is not None
    if model_kind == "single_file" and single_file_packed_gb is not None and keeps_packed:
        # An fp8 checkpoint the loader keeps packed when widening would not fit (fp8_storage_wanted): its stored size.
        return float(single_file_packed_gb), False
    if pinned is not None and pinned != "auto":
        stored = dict(getattr(fam, "prequant_resident_gb_by_scheme", None) or ())
        return float(stored.get(pinned, bf16_gb / 2.0)), False
    return bf16_gb * 2.0, False


def cpu_video_host_need(
    fam: Any,
    *,
    model_kind: str,
    te_scheme: Optional[str] = None,
    te_hosted: bool = False,
    te_hosted_available: bool = False,
    transformer_quant: Optional[str] = None,
    gguf_copy_gb: Optional[float] = None,
    single_file_packed_gb: Optional[float] = None,
) -> Optional[CpuVideoHostNeed]:
    """The anonymous host-RAM floor of this load on the float32 CPU target, or None when it cannot be sized.
    ``te_scheme``: the encoder scheme the loader runs (None = the released encoder); ``te_hosted``: LTX-2.5's hosted
    int8 file serves it; ``gguf_copy_gb``: the GGUF denoiser's in-memory copy, None when it is read mapped."""
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
        denoiser, mapped = _denoiser_gb(
            fam, model_kind, transformer_quant, sizes[0], gguf_copy_gb, single_file_packed_gb
        )
    except ValueError:
        return None  # an unsupported scheme is refused by the precision checks with its own message
    te, te_dense, te_alt, te_alt_label = _text_encoder_gb(
        fam, te_scheme, te_hosted, te_hosted_available, sizes[1]
    )
    # The table's third term is bf16-resident except where the family already keeps its VAE in float32.
    other = sizes[2] * (1.0 if getattr(fam, "vae_force_fp32", False) else 2.0)
    return CpuVideoHostNeed(
        denoiser,
        te,
        other,
        cpu_video_working_set_gb(fam),
        denoiser_mapped = mapped,
        denoiser_gguf_copy = model_kind == "gguf" and not mapped,
        text_encoder_dense = te_dense,
        text_encoder_quant_gb = te_alt,
        text_encoder_quant_label = te_alt_label,
    )


def _checkpoint_size(repo: Optional[str], filename: Optional[str]) -> tuple[int, Optional[str]]:
    """(bytes, local path) of a checkpoint: a local file, the Hub cache, then the Hub (size only). (0, None) unread."""
    if not (repo and filename):
        return 0, None
    for candidate in (Path(str(repo)) / str(filename), Path(str(filename))):
        try:
            if candidate.is_file():
                return candidate.stat().st_size, str(candidate)
        except OSError:
            pass
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(str(repo), str(filename))
        if isinstance(hit, str) and os.path.isfile(hit):
            return os.path.getsize(hit), hit
    except Exception:  # noqa: BLE001
        pass
    from .diffusion import _hub_file_size

    return int(_hub_file_size(str(repo), str(filename), None) or 0), None


_FP8_NAME = re.compile(r"(?<![a-z0-9])(fp8|float8|e4m3(fn)?|e5m2)(?![a-z0-9])", re.IGNORECASE)
_FP8_DTYPES = frozenset({"F8_E4M3", "F8_E5M2"})


def single_file_packed_gb(
    fam: Any, repo: Optional[str], filename: Optional[str]
) -> Optional[float]:
    """GB an fp8 single-file denoiser holds when the loader keeps it packed (one byte per fp8 weight), else None (a
    dense checkpoint, or fp8 storage unavailable / disabled, which widens it). fp8 is read from the safetensors header
    when the file is local, else from its name. The stored size is the file's (both experts' for a dual-expert family);
    unread, half the bf16 table. Never raises."""
    try:
        from .diffusion_comfy_quant import _read_header, comfy_fp8_storage

        if not comfy_fp8_storage(None):
            return None
        size, local = _checkpoint_size(repo, filename)
        fp8 = None
        if local is not None:
            try:
                header, _base = _read_header(local)
                fp8 = any(
                    isinstance(entry, dict) and entry.get("dtype") in _FP8_DTYPES
                    for key, entry in header.items()
                    if key != "__metadata__"
                )
            except Exception:  # noqa: BLE001 -- not a readable safetensors header: judge by name
                fp8 = None
        if fp8 is None:
            fp8 = bool(_FP8_NAME.search(Path(str(filename or "")).name))
        if not fp8:
            return None
        if size and getattr(fam, "is_moe", False):
            # A dual-expert pick loads its partner expert too (``_resolve_moe_partner_path``): add its stored size,
            # or this file's again when the partner cannot be read.
            from .video_moe_pair import moe_partner_filename

            partner = moe_partner_filename(filename)
            partner_size = _checkpoint_size(repo, partner)[0] if partner else 0
            size += partner_size or size
        if size:
            return size / 1e9
        sizes = _sizes(fam)  # the table's denoiser term already covers both experts
        return sizes[0] / 2.0 if sizes else None
    except Exception:  # noqa: BLE001 -- unsized: price it dense, as before
        return None


def gguf_copy_gb(
    fam: Any,
    repo: Optional[str],
    filename: Optional[str],
    *,
    mapped: Optional[bool] = None,
) -> Optional[float]:
    """GB diffusers copies into memory for this GGUF denoiser, or None when Studio's mmap hook reads it mapped. The size
    comes from a local file, the Hub cache, then the Hub; unreadable, the smallest curated quant's share of bf16. A
    dual-expert family loads both expert files. Never raises."""
    try:
        if mapped is None:
            from .diffusion_gguf_mmap import gguf_mmap_enabled
            mapped = gguf_mmap_enabled()
        if mapped:
            return None
        size, _local = _checkpoint_size(repo, filename)
        if size:
            # One expert's file; a dual-expert family loads both.
            return size / 1e9 * (2.0 if getattr(fam, "is_moe", False) else 1.0)
        # Unreadable: the table's denoiser term already covers both experts.
        sizes = _sizes(fam)
        return sizes[0] * GGUF_MIN_BF16_FRACTION if sizes else 0.0
    except Exception:  # noqa: BLE001 -- unsized: price the mapped floor
        return None


def _pss_bytes(pid: int) -> Optional[int]:
    """Proportional anon + shmem bytes (shared pages split between their mappers), else RssAnon + RssShmem."""
    try:
        with open(f"/proc/{pid}/smaps_rollup", encoding = "utf-8") as handle:
            found = {}
            for line in handle:
                key, _, value = line.partition(":")
                if key in ("Pss_Anon", "Pss_Shmem"):
                    found[key] = int(value.split()[0])
        if len(found) == 2:
            return (found["Pss_Anon"] + found["Pss_Shmem"]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open(f"/proc/{pid}/status", encoding = "utf-8") as handle:
            found = {}
            for line in handle:
                key, _, value = line.partition(":")
                if key in ("RssAnon", "RssShmem"):
                    found[key] = int(value.split()[0])
        return (found["RssAnon"] + found["RssShmem"]) * 1024 if len(found) == 2 else None
    except (OSError, ValueError, IndexError):
        return None


def _process_tree_pss_mib() -> int:
    """PSS (anon + shmem) MiB of this process and its children: the most a resident video model can hand back."""
    try:
        import psutil

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
        held = _pss_bytes(proc.pid)
        if held is None:
            try:
                held = int(proc.memory_info().rss)
            except Exception:  # noqa: BLE001
                held = 0
        total += held
    return total // _MIB


def cpu_video_host_capacity_mib(release_gb: float = 0.0) -> Optional[int]:
    """Host MiB a CPU video load can take: available RAM (capped by an enforcing cgroup's headroom and the
    ``UNSLOTH_DIFFUSION_HOST_RAM_LIMIT_MIB`` test hook) plus what the load releases, capped at the capacity (system
    total or cgroup limit). ``release_gb``: the resident video model this load replaces (the CPU branch leaves chat
    and images resident), credited only up to what this process tree measurably holds. None when unreadable."""
    from .diffusion_small_host import host_ram_mib

    total, available = host_ram_mib()
    if available is None:
        return None
    credit = 0
    if release_gb > 0:
        credit = min(int(release_gb * 1e9) // _MIB, _process_tree_pss_mib())
    capacity = int(available) + credit
    if total:
        capacity = min(capacity, int(total))
    return capacity


def _advice(fam: Any, need: CpuVideoHostNeed, capacity_gb: float) -> str:
    gguf_repo = getattr(fam, "gguf_repo", None)
    has_gguf = bool(gguf_repo) or bool(getattr(fam, "is_moe", False))
    tail = (
        "Free memory by closing other applications or unloading other models, or use a machine with more RAM or a "
        "supported GPU."
    )
    options = []
    floor = need.total_gb
    if not need.denoiser_mapped and not need.denoiser_gguf_copy and has_gguf:
        from .diffusion_gguf_mmap import gguf_mmap_enabled
        if gguf_mmap_enabled():
            floor -= need.denoiser_gb
            where = f" from '{gguf_repo}'" if gguf_repo else ""
            options.append(f"load a GGUF of it{where}, whose denoiser is read from disk as it runs")
    if need.text_encoder_dense and need.text_encoder_quant_gb is not None:
        floor -= need.text_encoder_gb - need.text_encoder_quant_gb
        options.append(
            f"use the {need.text_encoder_quant_label} text encoder (text encoder precision Auto or "
            f"{need.text_encoder_quant_label}, speed not Off; about {need.text_encoder_quant_gb:.1f} GB instead of "
            f"{need.text_encoder_gb:.1f})"
        )
    if options:
        joined = options[0] if len(options) == 1 else f"{options[0]}, and {options[1]}"
        if floor <= capacity_gb:
            return f"To fit, {joined}: about {floor:.1f} GB."
        return (
            f"Even if you {joined}, it needs about {floor:.1f} GB, more than is available. {tail}"
        )
    if need.denoiser_gguf_copy:
        rest = need.total_gb - need.denoiser_gb
        room = capacity_gb - rest
        unmap = (
            ""
            if os.name == "nt"
            else " Or unset UNSLOTH_DIFFUSION_GGUF_MMAP=0 so the GGUF is read from disk instead of copied."
        )
        if room >= 1.0:
            return (
                f"The GGUF denoiser is copied into memory here (about {need.denoiser_gb:.1f} GB): a GGUF quant of "
                f"at most about {room:.1f} GB fits.{unmap}"
            )
        return (
            f"Its text encoder, VAE and other components plus the working set need about {rest:.1f} GB in float32 "
            f"whatever the denoiser, so a lower GGUF quant does not fit either.{unmap} {tail}"
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
    te_scheme: Optional[str] = None,
    te_hosted: bool = False,
    te_hosted_available: bool = False,
    transformer_quant: Optional[str] = None,
    gguf_copy: Optional[float] = None,
    single_file_packed: Optional[float] = None,
    release_gb: float = 0.0,
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
            te_scheme = te_scheme,
            te_hosted = te_hosted,
            te_hosted_available = te_hosted_available,
            transformer_quant = transformer_quant,
            gguf_copy_gb = gguf_copy,
            single_file_packed_gb = single_file_packed,
        )
        if need is None:
            return None
        capacity_mib = cpu_video_host_capacity_mib(release_gb)
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
            if need.denoiser_mapped:
                denoiser = "denoiser read from its GGUF file"
            elif need.denoiser_gguf_copy:
                denoiser = f"GGUF denoiser copied into memory {need.denoiser_gb:.1f}"
            else:
                denoiser = f"denoiser {need.denoiser_gb:.1f}"
            parts = (
                f"{denoiser}, "
                + (
                    "text encoder read from its float32 file"
                    if getattr(fam, "name", None) in STORED_FP32_TEXT_ENCODER_FAMILIES
                    and need.text_encoder_gb == 0.0
                    else f"text encoder {need.text_encoder_gb:.1f}"
                )
                + ", VAE and other components "
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


__all__ = [
    "CPU_VIDEO_WORKING_SET_GB",
    "CPU_VIDEO_WORKING_SET_GB_BY_FAMILY",
    "STORED_FP32_TEXT_ENCODER_FAMILIES",
    "cpu_video_working_set_gb",
    "CpuVideoHostNeed",
    "GGUF_MIN_BF16_FRACTION",
    "H3_CPU_NATIVE_HOST_GB",
    "LTX25_CPU_TE_INT8_HOST_GB",
    "cpu_video_host_capacity_mib",
    "cpu_video_host_need",
    "cpu_video_host_shortfall",
    "gguf_copy_gb",
    "single_file_packed_gb",
]
