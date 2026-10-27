# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hardware-aware auto-policy for the diffusion loader: a pure decision layer.

The loader historically planned memory from the GGUF file size BEFORE considering the dense
transformer-quant fast path, which hid the fast path where it matters: where the GGUF-size plan
picks offload, a dense int8/fp8 transformer (~half the bf16 bytes) often fits resident and beats
the offloaded GGUF. This module supplies the pieces to fix that:

  * a per-family bf16 component-size table (transformer / text encoders / VAE) with per-scheme
    scaling, so a candidate's footprint is estimated BEFORE anything is downloaded; and
  * ``resolve_dense_quant_candidate``, which turns a request + device into a concrete estimate the
    loader re-plans memory against.

It also builds the ``resolved`` record for status: per Advanced control, the engaged value plus
its provenance (explicit / auto) and a short reason. Pure: no torch import at module import, so the
decision logic unit-tests on CPU-only hosts.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from .diffusion_fp16_variant import is_fp16_twin, shadowed_twins

_MIB_PER_GB = 1000.0**3 / (1024.0 * 1024.0)  # component sizes below are decimal GB

# Steady torchao-quantised size relative to bf16, measured on live loads.
_QUANT_STEADY_FACTOR: dict[str, float] = {
    "int8": 0.55,
    "fp8": 0.55,
    "mxfp8": 0.58,
    "nvfp4": 0.33,
}

# Measured policy residency over bf16 + ~0.04 slack; 0.33 would keep a non-fitting model resident.
_POLICY_STEADY_FACTOR: dict[str, float] = {
    "zimg_rg76_v1": 0.50,
    "flux_r420_v1": 0.45,
    "qwen21_r020_v1": 0.50,
    "qwen2512_m120_attn8_v1": 0.42,
    "qwen_p02_v1": 0.47,
}


def policy_steady_factor(family: Any, base_repo: Optional[str] = None) -> Optional[float]:
    """The NVFP4 POLICY steady factor for ``(family, base_repo)``, or None. Never raises."""
    from .diffusion_nvfp4_flag import nvfp4_diffusion_enabled

    if not nvfp4_diffusion_enabled():
        return None
    try:
        from .diffusion_nvfp4_policy import resolve_policy
        policy = resolve_policy(getattr(family, "name", family), base_repo)
    except Exception:  # noqa: BLE001 -- an unresolvable policy just means the plain factor
        return None
    return None if policy is None else _POLICY_STEADY_FACTOR.get(policy.policy_id)


# bf16-resident sizes in decimal GB from HF metadata, not download size (Z-Image-Turbo ships fp32).
_FAMILY_BF16_GB: dict[str, tuple[float, float, float]] = {
    "flux.1": (23.8, 9.8, 0.2),
    "flux.1-kontext": (23.8, 9.8, 0.2),
    "flux.2-klein": (7.8, 8.0, 0.2),
    # FLUX.3 Image Klein: 5.55B DiT (11.1 GB bf16) + Qwen3-VL-4B (8.9 GB) + the FLUX.2 VAE (stored fp32, 0.17 GB
    # bf16). From the black-forest-labs/flux-3-image-klein component folders; the base shares the architecture.
    "flux.3-klein": (11.1, 8.9, 0.2),
    "flux.3-klein-base": (11.1, 8.9, 0.2),
    "flux.2-dev": (64.5, 48.0, 0.4),
    "qwen-image": (40.9, 16.6, 0.3),
    "qwen-image-edit": (40.9, 16.6, 0.3),
    # Same DiT and Qwen2.5-VL 7B encoder; the RGBA VAE is 0.25 GB.
    "qwen-image-layered": (40.9, 16.6, 0.3),
    # A different architecture, not a refreshed Qwen-Image: 32 single-stream blocks against 60
    # dual-stream ones, so the DiT is 14.2 GB rather than 40.9, while the Qwen3-VL 8B encoder is
    # LARGER than Qwen-Image's Qwen2.5-VL 7B. Ships bf16, so resident equals on-disk. Read off the
    # Hub sibling metadata for Qwen/Qwen-Image-2.1: transformer 14.23, text_encoder 17.53, vae 1.35.
    "qwen-image-2.1": (14.2, 17.5, 1.4),
    "z-image": (12.3, 8.0, 0.2),
    "krea-2": (26.3, 8.9, 0.5),
    # MOSS-Image: 5.08B DiT + headless Qwen3.5-9B + fp32 FLUX.2 VAE (MOSS-Image-SFT index metadata).
    "moss-image": (10.2, 15.9, 0.3),
    # Ships fp32 (10.4 + 10.5 + 0.3 GB of shards); bf16-resident is half.
    "lumina-2": (5.2, 5.2, 0.2),
    # 17B dual-stream DiT (32.5 GB bf16 on disk) + Qwen2.5-VL 15.5 GB + ByT5 0.8 GB.
    "hunyuanimage-2.1": (32.5, 16.3, 0.8),
    # 17B MoE DiT (34.2 GB bf16) + FOUR text encoders: CLIP-L 0.5 + CLIP-G 2.8 + T5-XXL 9.5, plus Llama-3.1-8B (~16 GB
    # bf16) from the open mirror at load time.
    "hidream-i1": (34.2, 28.8, 0.2),
    # Two ~9.3B DiTs (Ideogram's dual-branch CFG), both resident, plus a Qwen3-VL encoder. The vendor stores raw float8,
    # so each doubles at bf16.
    "ideogram-4": (37.2, 16.3, 0.2),
}

# Hub DOWNLOAD bytes relative to the bf16-resident sizes above, for families whose published checkpoints are not
# stored at bf16. The free-disk gate needs what lands in the HF cache, which differs by 2x in either direction here.
# From HF metadata: z-image and lumina-2 publish fp32 (2.00x), ideogram-4 publishes fp8 (0.50x). Anything absent ships
# bf16 and downloads what it occupies (measured 0.99-1.07x).
_FAMILY_HUB_DOWNLOAD_FACTOR: dict[str, float] = {
    "z-image": 2.0,
    "lumina-2": 2.0,
    "ideogram-4": 0.5,
}

# Z-Image base ships bf16, unlike the fp32 Turbo the family factor targets.
_BASE_REPO_HUB_DOWNLOAD_FACTOR: dict[str, float] = {
    "tongyi-mai/z-image": 1.0,
}


DENOISER_SUBFOLDERS = ("transformer/", "unconditional_transformer/")

# Download bytes per resident byte; unknown bases stay 1:1.
_BASE_RESIDENT_FACTORS: dict = {
    "tongyi-mai/z-image-turbo": (2.0, 1.0),
    "tongyi-mai/z-image": (1.0, 1.0),
    "alpha-vllm/lumina-image-2.0": (2.0, 2.0),
    "ideogram-ai/ideogram-4-fp8": (0.5, 0.5),
    # SDXL defaults are fp32 (12.9 GB vs 6.5 GB loaded), unet/ is a companion: both factors halve. fp16 twins price 1:1.
    "stabilityai/stable-diffusion-xl-base-1.0": (2.0, 2.0),
    "stabilityai/sdxl-turbo": (2.0, 2.0),
}


def resident_bytes_from_declared(
    base_repo: Optional[str],
    declared_files: Any,
    *,
    prequant_bytes: int = 0,
    extra_bf16_bytes: int = 0,
    dtype_scale: float = 1.0,
) -> Optional[int]:
    """Prequant bytes add unchanged; extra_bf16 widens when float32; unknown families priced 1:1."""
    denoiser_factor, companion_factor = _BASE_RESIDENT_FACTORS.get(
        _base_key(base_repo) if base_repo else "", (1.0, 1.0)
    )
    denoiser = 0
    companions = 0
    twins = 0  # already at load width: 1:1
    declared_files = list(declared_files or ())
    shadowed = shadowed_twins(path for path, _size in declared_files)
    for path, size in declared_files:
        size = int(size or 0)
        if size <= 0 or str(path) in shadowed:
            continue
        if is_fp16_twin(str(path)):
            twins += size
        elif str(path).startswith(DENOISER_SUBFOLDERS):
            denoiser += size
        else:
            companions += size
    prequant_bytes = max(0, int(prequant_bytes or 0))
    extra_bf16_bytes = max(0, int(extra_bf16_bytes or 0))
    dtype_scale = max(1.0, float(dtype_scale or 1.0))
    if (
        denoiser <= 0
        and companions <= 0
        and twins <= 0
        and prequant_bytes <= 0
        and extra_bf16_bytes <= 0
    ):
        return None
    dtype_resident = (
        int(denoiser / max(denoiser_factor, 0.01))
        + int(companions / max(companion_factor, 0.01))
        + twins
        + extra_bf16_bytes
    )
    return int(dtype_resident * dtype_scale) + prequant_bytes


def companion_bytes_from_declared(
    base_repo: Optional[str],
    declared_files: Any,
    *,
    dtype_scale: float = 1.0,
) -> Optional[int]:
    """Non-denoiser (text encoder, VAE) share of ``resident_bytes_from_declared``; None when none declared."""
    _denoiser_factor, companion_factor = _BASE_RESIDENT_FACTORS.get(
        _base_key(base_repo) if base_repo else "", (1.0, 1.0)
    )
    # An SDXL-style UNet is a denoiser too.
    declared_files = list(declared_files or ())
    shadowed = shadowed_twins(path for path, _size in declared_files)
    kept = [
        (str(path), max(0, int(size or 0)))
        for path, size in declared_files
        if not str(path).startswith(DENOISER_SUBFOLDERS + ("unet/",)) and str(path) not in shadowed
    ]
    companions = sum(size for path, size in kept if not is_fp16_twin(path))
    twins = sum(size for path, size in kept if is_fp16_twin(path))
    if companions + twins <= 0:
        return None
    return int(
        (int(companions / max(companion_factor, 0.01)) + twins)
        * max(1.0, float(dtype_scale or 1.0))
    )


def _base_key(base_repo: Optional[str]) -> str:
    """Return the canonical, case-insensitive key used by per-base tables."""
    from .diffusion_families import canonical_base
    return canonical_base(base_repo).strip().lower()


def hub_download_factor(fam: Any, base_repo: Optional[str] = None) -> float:
    """Download bytes per bf16-resident byte for this pick, defaulting to 1.0."""
    override = _BASE_REPO_HUB_DOWNLOAD_FACTOR.get(_base_key(base_repo)) if base_repo else None
    if override is not None:
        return override
    return _FAMILY_HUB_DOWNLOAD_FACTOR.get(getattr(fam, "name", None), 1.0)


# klein 9B variants need per-id overrides (2.3x the family default).
_BASE_REPO_BF16_GB: dict[str, tuple[float, float, float]] = {
    "black-forest-labs/flux.2-klein-9b": (18.2, 16.4, 0.2),
    "black-forest-labs/flux.2-klein-base-9b": (18.2, 16.4, 0.2),
}


def base_repo_bf16_components_gb(base_repo: Optional[str]) -> Optional[tuple[float, float, float]]:
    """Per-base override only, with no family fallback, so callers can tell whether this base was named."""
    if not base_repo:
        return None
    return _BASE_REPO_BF16_GB.get(_base_key(base_repo))


def family_bf16_components_gb(
    fam: Any, base_repo: Optional[str] = None
) -> Optional[tuple[float, float, float]]:
    """(transformer, text encoders, VAE) bf16-resident sizes in GB, or None when the family isn't
    in the table (callers fall back to file-size estimates)."""
    override = base_repo_bf16_components_gb(base_repo)
    if override is not None:
        return override
    name = getattr(fam, "name", None)
    return _FAMILY_BF16_GB.get(name) if name else None


@dataclass(frozen = True)
class DenseQuantEstimate:
    """Footprint estimate for one dense transformer-quant candidate.

    ``transient_transformer_mib`` is the build peak (dense bf16 when quantising on the fly, or the
    quantised size when a prequant checkpoint loads via meta). ``steady_transformer_mib`` is what
    stays resident for generation. ``download_transformer_mib`` is what the base-repo transformer
    costs on DISK, which is not the same number whenever the checkpoint publishes something other
    than bf16 (see ``hub_download_factor``)."""

    scheme: str
    steady_transformer_mib: int
    transient_transformer_mib: int
    companions_mib: int
    prequant: bool
    download_transformer_mib: int = 0
    # 0 means no split (no encoder-streaming tier).
    text_encoders_mib: int = 0

    @property
    def transient_total_mib(self) -> int:
        return self.transient_transformer_mib + self.companions_mib

    @property
    def steady_total_mib(self) -> int:
        return self.steady_transformer_mib + self.companions_mib


def estimate_dense_quant(
    fam: Any,
    scheme: str,
    *,
    base_repo: Optional[str] = None,
    prequant_available: bool = False,
) -> Optional[DenseQuantEstimate]:
    """Estimate the candidate's footprint from the family table, or None when the
    family (or scheme factor) is unknown."""
    from .diffusion_nvfp4_flag import nvfp4_blocked

    if nvfp4_blocked(scheme):
        return None
    components = family_bf16_components_gb(fam, base_repo)
    factor = _QUANT_STEADY_FACTOR.get(scheme)
    if scheme == "nvfp4":
        factor = policy_steady_factor(fam, base_repo) or factor
    if components is None or factor is None:
        return None
    transformer_gb, text_encoders_gb, vae_gb = components
    steady = int(transformer_gb * factor * _MIB_PER_GB)
    transient = steady if prequant_available else int(transformer_gb * _MIB_PER_GB)
    companions = int((text_encoders_gb + vae_gb) * _MIB_PER_GB)
    hub_factor = hub_download_factor(fam, base_repo)
    return DenseQuantEstimate(
        scheme = scheme,
        steady_transformer_mib = steady,
        transient_transformer_mib = transient,
        companions_mib = companions,
        prequant = prequant_available,
        download_transformer_mib = int(transformer_gb * hub_factor * _MIB_PER_GB),
        # Same conversion as `companions`, of which this is the text-encoder half, so the planner's `companions -
        # text_encoders` is the VAE and nothing else.
        text_encoders_mib = int(text_encoders_gb * _MIB_PER_GB),
    )


def _hf_cache_free_mib() -> Optional[int]:
    """Free MiB on the filesystem holding the HF model cache (None when unprobeable)."""
    try:
        import shutil

        try:
            from huggingface_hub.constants import HF_HUB_CACHE as cache_dir
        except Exception:  # noqa: BLE001 -- hub missing/old: probe the conventional path
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        probe = str(cache_dir)
        while probe and not os.path.isdir(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        return int(shutil.disk_usage(probe).free // (1024 * 1024))
    except Exception:  # noqa: BLE001 -- disk probing must never sink the candidate
        return None


def _has_usable_prequant(
    fam: Any, scheme: str, prequant_path: Optional[str], base_repo: Optional[str]
) -> bool:
    """Whether a hosted or operator-supplied prequant checkpoint for ``scheme`` is usable; False on failure."""
    try:
        from .diffusion_prequant import usable_prequant_source
        return (
            usable_prequant_source(fam, scheme, path_override = prequant_path, base_repo = base_repo)
            is not None
        )
    except Exception:  # noqa: BLE001 -- prequant probing must never sink the candidate
        return False


def resolve_dense_quant_candidate(
    *,
    fam: Any,
    target: Any,
    requested: Optional[str],
    base_repo: Optional[str] = None,
    prequant_path: Optional[str] = None,
    force_dense: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Optional[DenseQuantEstimate]:
    """None means no basis to re-plan, so the loader keeps today's behaviour for unlisted families."""
    from .diffusion_transformer_quant import (
        dense_transformer_supported,
        normalize_transformer_quant,
        select_transformer_quant_scheme,
    )

    if normalize_transformer_quant(requested) is None:
        return None
    if not dense_transformer_supported(target):
        return None
    scheme = select_transformer_quant_scheme(
        target,
        requested,
        family = getattr(fam, "name", None),
        base_repo = base_repo,
        has_prequant = lambda candidate: _has_usable_prequant(
            fam, candidate, prequant_path, base_repo
        ),
    )
    if scheme is None:
        return None
    prequant_available = False
    prequant_cached = False
    # force_dense: the loader will SKIP the prequant shortcut (e.g. a LoRA bake), so size the candidate for the dense
    # build.
    if not force_dense:
        try:
            from .diffusion_prequant import prequant_checkpoint_cached, usable_prequant_source

            # usable_: a local override counts only if allowlisted AND present.
            src = usable_prequant_source(
                fam, scheme, path_override = prequant_path, base_repo = base_repo
            )
            prequant_available = src is not None
            if src is not None and getattr(src, "kind", None) == "path":
                # Local overrides download nothing, so the space gate does not apply.
                prequant_cached = True
            elif src is not None:
                # Pin the active cache root, else a moved cache reads as uncached; import from utils (no cycle).
                from utils.hf_cache_settings import active_hf_hub_cache
                prequant_cached = prequant_checkpoint_cached(src, cache_dir = active_hf_hub_cache())
        except Exception:  # noqa: BLE001 -- prequant probing must never sink the candidate
            prequant_available = False
            prequant_cached = False
    estimate = estimate_dense_quant(
        fam, scheme, base_repo = base_repo, prequant_available = prequant_available
    )
    if estimate is not None and logger is not None:
        logger.info(
            "diffusion.auto_policy: dense %s candidate steady=%d MiB transient=%d MiB "
            "companions=%d MiB prequant=%s",
            scheme,
            estimate.steady_transformer_mib,
            estimate.transient_transformer_mib,
            estimate.companions_mib,
            prequant_available,
        )
    # A cached prequant downloads nothing, so skip the space gate.
    if estimate is not None and not (estimate.prequant and prequant_cached):
        # Distinct unreplayed weight placements a block may see before giving up (address churn from streaming or disk offload); each costs one eager call
        needed_mib = (
            estimate.steady_transformer_mib
            if estimate.prequant
            else estimate.download_transformer_mib
        )
        free_mib = _hf_cache_free_mib()
        if free_mib is not None and free_mib < needed_mib + 10 * 1024:
            if logger is not None:
                logger.info(
                    "diffusion.auto_policy: skipping dense %s (~%d MiB download, "
                    "only %d MiB free in the model cache)",
                    scheme,
                    needed_mib,
                    free_mib,
                )
            return None
    return estimate


# Explicit precision fails closed; UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK restores fallback.
_PRECISION_FALLBACK_ENV = "UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK"


def precision_fallback_allowed() -> bool:
    """Whether a declined EXPLICIT precision request may fall back silently (opt-in)."""
    value = str(os.environ.get(_PRECISION_FALLBACK_ENV, "")).strip().lower()
    return value in ("1", "true", "yes", "on")


def precision_refusal_message(
    control: str,
    requested: str,
    reason: str,
    *,
    off_label: str,
    auto_available: bool = True,
) -> str:
    """Refusal for a declined precision; text_encoder_quant has no auto mode, so Auto is never suggested."""
    remedy = (
        "Choose Auto to let the backend pick the fastest precision this host can run, "
        f"or {off_label}."
        if auto_available
        else f"{off_label[:1].upper()}{off_label[1:]}."
    )
    return f"{control}='{requested}' could not be used: {reason}. {remedy}"


# applied: honored or no ask; fell_back: explicit ask declined, something else engaged;
# unsupported: the explicit ask cannot run on this host / model.
RESOLVED_APPLIED = "applied"  # the ask was honored (or there was no ask)
RESOLVED_FELL_BACK = "fell_back"  # an explicit ask was declined and something ELSE engaged
RESOLVED_UNSUPPORTED = "unsupported"  # an explicit ask cannot run on this host / model at all

# Requests that mean "do not engage this control", so an "off" engagement HONORS them.
_RESOLVED_OFF_VALUES = frozenset({"", "none", "off", "false", "0"})

# memory_mode and attention_backend use different request/engaged vocabularies.
_RESOLVED_COMPARABLE = frozenset({"transformer_quant", "text_encoder_quant"})


def _resolved_norm(value: Any) -> str:
    """A control value folded to its comparison key; every "off" spelling folds to ""."""
    if value is None:
        return ""
    text = str(value).strip().lower().replace("-", "_")
    return "" if text in _RESOLVED_OFF_VALUES else text


def _resolved_values_match(explicit: Any, engaged: Any) -> bool:
    """Whether ``engaged`` is what ``explicit`` asked for (cpu_offload compares as a boolean)."""
    if isinstance(explicit, bool) or isinstance(engaged, bool):
        return bool(explicit) == bool(engaged)
    return _resolved_norm(explicit) == _resolved_norm(engaged)


def build_resolved_record(controls: dict[str, tuple]) -> dict[str, dict[str, Any]]:
    """Per-control record of requested vs engaged value from (explicit, engaged, reason[, status]) tuples."""
    record: dict[str, dict[str, Any]] = {}
    for name, entry in controls.items():
        explicit, engaged, reason = entry[0], entry[1], entry[2]
        status = entry[3] if len(entry) > 3 else None
        left_to_backend = explicit is None or (
            isinstance(explicit, str) and explicit.strip().lower() in ("", "auto")
        )
        if left_to_backend:
            # The backend chose it, so there is nothing to decline: this is the "Auto: X" badge.
            status = RESOLVED_APPLIED
        elif status is None:
            status = (
                RESOLVED_FELL_BACK
                if name in _RESOLVED_COMPARABLE and not _resolved_values_match(explicit, engaged)
                else RESOLVED_APPLIED
            )
        record[name] = {
            "value": engaged,
            "requested": None if left_to_backend else explicit,
            "source": "auto" if left_to_backend else "explicit",
            "status": status,
            "reason": reason,
        }
    return record


def format_resolved_for_log(record: Optional[dict[str, dict[str, Any]]]) -> str:
    """Resolved record as one ``name=value(source)`` log line."""
    parts = []
    for name, entry in (record or {}).items():
        value = entry.get("value")
        if entry.get("source") == "auto":
            note = "auto"
        else:
            note = f"requested {entry.get('requested')}"
            if entry.get("status") != RESOLVED_APPLIED:
                note += f", {entry.get('status')}"
        parts.append(f"{name}={value}({note})")
    return " ".join(parts)


def format_generation_for_log(
    result: dict[str, Any],
    *,
    engine: str,
    steps: Any = None,
    strength: Any = None,
    upscale: Any = None,
    loras: Any = None,
) -> str:
    """One-line log summary of a generate() result."""
    images = result.get("images") or ()
    size = getattr(images[0], "size", None) if images else None
    if not isinstance(size, tuple):
        size = None  # a latent tensor (Hires fix base pass) has no pixel size
    fields = {
        "engine": engine,
        "workflow": result.get("workflow"),
        "images": len(images),
        "size": f"{size[0]}x{size[1]}" if size else None,
        "seeds": result.get("seeds") or result.get("seed"),
        "steps": steps,
        "strength": strength,
        "upscale": upscale,
        "loras": loras or result.get("active_loras") or None,
        "reference_resolution": result.get("reference_resolution"),
        "localized_edit": result.get("localized_edit"),
    }
    return " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
