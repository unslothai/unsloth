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

_MIB_PER_GB = 1000.0**3 / (1024.0 * 1024.0)  # component sizes below are decimal GB

# Steady size of a torchao-quantised transformer relative to bf16: int8/fp8 store one byte per param plus per-row scales
# (~0.52x with slack for bf16 norms/embeddings); nvfp4 packs two per byte plus block scales. Measured on live loads.
_QUANT_STEADY_FACTOR: dict[str, float] = {
    "int8": 0.55,
    "fp8": 0.55,
    "mxfp8": 0.58,
    "nvfp4": 0.33,
}

# bf16-RESIDENT component sizes in decimal GB: (transformer, text encoders, VAE). What they occupy on device after the dtype
# cast, NOT the download size (Z-Image-Turbo ships fp32: 24.6 GB of shards -> 12.3 GB bf16). From HF sibling metadata.
_FAMILY_BF16_GB: dict[str, tuple[float, float, float]] = {
    "flux.1": (23.8, 9.8, 0.2),
    "flux.1-kontext": (23.8, 9.8, 0.2),
    "flux.2-klein": (7.8, 8.0, 0.2),
    "flux.2-dev": (64.5, 48.0, 0.4),
    "qwen-image": (40.9, 16.6, 0.3),
    "qwen-image-edit": (40.9, 16.6, 0.3),
    "z-image": (12.3, 8.0, 0.2),
    "krea-2": (26.3, 8.9, 0.5),
    # Ships fp32 (10.4 + 10.5 + 0.3 GB of shards); bf16-resident is half.
    "lumina-2": (5.2, 5.2, 0.2),
    # 17B dual-stream DiT (32.5 GB bf16 on disk) + Qwen2.5-VL 15.5 GB + ByT5 0.8 GB.
    "hunyuanimage-2.1": (32.5, 16.3, 0.8),
    # 17B MoE DiT (34.2 GB bf16) + FOUR text encoders: CLIP-L 0.5 + CLIP-G 2.8 + T5-XXL 9.5, plus Llama-3.1-8B (~16 GB bf16) from the open mirror at load time.
    "hidream-i1": (34.2, 28.8, 0.2),
    # Two ~9.3B DiTs (Ideogram's dual-branch CFG), both resident, plus a Qwen3-VL encoder. The vendor stores raw float8, so each doubles at bf16.
    "ideogram-4": (37.2, 16.3, 0.2),
}

# Hub DOWNLOAD bytes relative to the bf16-resident sizes above, for families whose published checkpoints are not stored at
# bf16. The free-disk gate needs what lands in the HF cache, which differs by 2x in either direction here. From HF metadata:
#   z-image     Turbo transformer/ 23,479 MiB fp32 -> 11,730 MiB resident (2.00x)
#   lumina-2    transformer/  9,956 MiB fp32 ->  4,959 MiB resident (2.01x)
#   ideogram-4  transformer/ + unconditional_transformer/ 17,718 MiB fp8 -> 35,477 MiB resident (0.50x)
# Anything absent ships bf16 and downloads what it occupies (measured 0.99-1.07x).
_FAMILY_HUB_DOWNLOAD_FACTOR: dict[str, float] = {
    "z-image": 2.0,
    "lumina-2": 2.0,
    "ideogram-4": 0.5,
}

# Per-base overrides of the factor above, for a checkpoint stored at a different precision from its
# family default. Only Tongyi-MAI/Z-Image needs one: the family key exists because the distilled
# Turbo publishes fp32 (23,479 MiB), while the undistilled base ships bf16 (11,740 MiB) and so
# downloads exactly what it occupies. Without it the free-disk gate demands twice the real size.
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
    # SDXL's default variant is fp32 throughout (headers read 2026-08-25); the loader skips the
    # fp16 twins. At 1:1 its 12.9 GB prices against a 6.5 GB bf16 load and refuses on any 16 GB
    # pool. Its denoiser is ``unet/``, so it lands in the companion bucket: both factors halve.
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
    """Estimate resident bytes from Hub file sizes and component storage precision.

    ``prequant_bytes`` is added unchanged because hosted pre-cast encoders are already stored at
    their load precision. ``extra_bf16_bytes`` covers separately hosted dense components, and is
    widened with the pipeline when the resolved target is float32. Unknown families are treated
    as 1:1 with their download size.
    """
    denoiser_factor, companion_factor = _BASE_RESIDENT_FACTORS.get(
        _base_key(base_repo) if base_repo else "", (1.0, 1.0)
    )
    denoiser = 0
    companions = 0
    for path, size in declared_files or ():
        size = int(size or 0)
        if size <= 0:
            continue
        if str(path).startswith(DENOISER_SUBFOLDERS):
            denoiser += size
        else:
            companions += size
    prequant_bytes = max(0, int(prequant_bytes or 0))
    extra_bf16_bytes = max(0, int(extra_bf16_bytes or 0))
    dtype_scale = max(1.0, float(dtype_scale or 1.0))
    if denoiser <= 0 and companions <= 0 and prequant_bytes <= 0 and extra_bf16_bytes <= 0:
        return None
    dtype_resident = (
        int(denoiser / max(denoiser_factor, 0.01))
        + int(companions / max(companion_factor, 0.01))
        + extra_bf16_bytes
    )
    return int(dtype_resident * dtype_scale) + prequant_bytes


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


# Base-repo overrides for families offering multiple sizes under one entry (the table carries the family default).
# flux.2-klein ships FOUR checkpoints under one entry: 4B / base-4B (the family default, Qwen3-4B encoder) and
# 9B / base-9B (18.2 GB transformer, Qwen3-8B encoder). The 9B pair needs an override on BOTH ids: sizing
# klein-BASE-9B off the family default understates it by 2.3x, and the base variants are the ones the upstream
# guidance points fine-tuning at, so it is the likelier of the two to be loaded.
_BASE_REPO_BF16_GB: dict[str, tuple[float, float, float]] = {
    "black-forest-labs/flux.2-klein-9b": (18.2, 16.4, 0.2),
    "black-forest-labs/flux.2-klein-base-9b": (18.2, 16.4, 0.2),
}


def base_repo_bf16_components_gb(base_repo: Optional[str]) -> Optional[tuple[float, float, float]]:
    """The per-base override for ``base_repo``, or None when it has none.

    No family fallback, unlike ``family_bf16_components_gb``: a caller that already holds a
    better number for the family default needs to know whether this particular base was
    actually named in the table, not receive the family figure back."""
    if not base_repo:
        return None
    return _BASE_REPO_BF16_GB.get(_base_key(base_repo))


def family_bf16_components_gb(
    fam: Any, base_repo: Optional[str] = None
) -> Optional[tuple[float, float, float]]:
    """(transformer, text encoders, VAE) bf16-resident sizes in GB, or None when the family isn't
    in the table (callers fall back to file-size estimates)."""
    # A per-base miss falls through quietly to the coarser family table.
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
    # The TEXT-ENCODER share of ``companions_mib``. The memory planner needs it to price the
    # group tier that streams the encoders instead of keeping them resident, and this table is
    # the only place the split exists on the dense-candidate path. Defaulted so any construction
    # that predates it still works (the planner reads a 0 split as "no split", i.e. no new tier).
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
    components = family_bf16_components_gb(fam, base_repo)
    factor = _QUANT_STEADY_FACTOR.get(scheme)
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
        # Same conversion as `companions`, of which this is the text-encoder half, so the planner's
        # `companions - text_encoders` is the VAE and nothing else.
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
    """The dense-quant candidate the loader should re-plan memory against, or None.

    None means "no basis to re-plan" (request off, device can't run dense, no scheme resolves, or
    no size entry); the loader keeps today's behaviour, so unlisted families see no change."""
    from .diffusion_transformer_quant import (
        dense_transformer_supported,
        normalize_transformer_quant,
        select_transformer_quant_scheme,
    )

    if normalize_transformer_quant(requested) is None:
        return None
    if not dense_transformer_supported(target):
        return None
    scheme = select_transformer_quant_scheme(target, requested, family = getattr(fam, "name", None))
    if scheme is None:
        return None
    prequant_available = False
    prequant_cached = False
    # force_dense: the loader will SKIP the prequant shortcut (e.g. a LoRA bake), so size the candidate for the dense build.
    if not force_dense:
        try:
            from .diffusion_prequant import prequant_checkpoint_cached, usable_prequant_source

            # usable_ (not resolve_): a local path override counts only when the loader will accept it (allowlisted AND present), else it rebuilds dense after eviction.
            src = usable_prequant_source(
                fam, scheme, path_override = prequant_path, base_repo = base_repo
            )
            prequant_available = src is not None
            if src is not None and getattr(src, "kind", None) == "path":
                # A local override is the operator's own file on disk: it downloads nothing, so the
                # space gate has no claim on it. prequant_checkpoint_cached only answers for hosted
                # repos (_cached_in_root returns None for any other kind), so asking it here would
                # report False and re-apply the gate to a file that costs no bytes, which is the
                # opposite of what the retry assumes about local paths.
                prequant_cached = True
            elif src is not None:
                # Pin the ACTIVE root, as the retry and the loader both do. Unpinned,
                # cached_checkpoint_path searches only huggingface_hub's import-time constant, so
                # after a cache-folder change the retry proves the checkpoint cached in the live
                # root and this would still call it uncached and re-apply the gate. Imported from
                # utils rather than diffusion.hub_cache_dir, which would be a circular import.
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
    # A cached prequant checkpoint downloads nothing, so the space gate has no claim on it. The
    # gate used to run anyway, which discarded exactly the candidate the auto retry exists to find:
    # that retry only ever proposes a rung whose checkpoint is already cached, so on a low-disk or
    # moved-cache install every retry fell back to the GGUF despite a resident-fit local artifact.
    if estimate is not None and not (estimate.prequant and prequant_cached):
        # The dense path may DOWNLOAD the artifact into the HF cache, which must never wedge a nearly full disk. Size it by what lands on DISK: a prequant fetches the quantised checkpoint, else the base repo's.
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


# ── strict precision (fail closed on a declined EXPLICIT request) ────────────
# An `auto` precision is a delegation, so falling back down the ladder is the feature working. An
# EXPLICIT scheme is a contract: silently loading the GGUF (or a dense bf16 DiT) instead produced a
# perfectly good image at a precision nobody asked for, which is exactly what made a successful
# render worthless as proof that the requested precision ran. Escape hatch for anyone who relied on
# the old behaviour; unset (the default) fails closed.
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
    """The client-facing refusal for a declined explicit precision: what was asked, why it could
    not run, and the settings that always work. ``off_label`` names this backend's "run the
    checkpoint as-is" option, which differs between the image and video loaders.

    ``auto_available`` is False for controls with no auto mode. ``text_encoder_quant`` is one:
    both request models restrict it to fp8 / fp8_dynamic / int8 / nvfp4, so a user who followed a
    "Choose Auto" instruction there got a 422 from request validation instead of a working load."""
    remedy = (
        "Choose Auto to let the backend pick the fastest precision this host can run, "
        f"or {off_label}."
        if auto_available
        else f"{off_label[:1].upper()}{off_label[1:]}."
    )
    return f"{control}='{requested}' could not be used: {reason}. {remedy}"


# ── resolved-record (status surface) ─────────────────────────────────────────
# How the engaged value relates to what the caller asked for. Additive on the status payload: every
# honored request and every auto decision reports "applied", so a client that ignores the field sees
# exactly today's behaviour.
RESOLVED_APPLIED = "applied"  # the ask was honored (or there was no ask)
RESOLVED_FELL_BACK = "fell_back"  # an explicit ask was declined and something ELSE engaged
RESOLVED_UNSUPPORTED = "unsupported"  # an explicit ask cannot run on this host / model at all

# Requests that mean "do not engage this control", so an "off" engagement HONORS them.
_RESOLVED_OFF_VALUES = frozenset({"", "none", "off", "false", "0"})

# Controls whose REQUEST and ENGAGED value share one vocabulary, so a mismatch is derivable here
# rather than trusted from the call site (a decline site that forgets to classify itself still
# reports the truth). memory_mode is deliberately absent: it requests a MODE ("low_vram") and
# engages an offload POLICY ("sequential"), so comparing the two would report every honored
# request as a fallback. attention_backend is absent for the same reason ("cudnn" engages as
# "_native_cudnn").
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
    """The per-control ``resolved`` record for status: engaged value + provenance.

    ``controls`` maps a control name to ``(explicit, engaged, reason)`` or
    ``(explicit, engaged, reason, status)``, where ``explicit`` is the raw request
    (None / "" / "auto" = left to the backend) and ``status`` is one of the ``RESOLVED_*``
    constants (omit it to let this derive one).

    Each entry carries BOTH sides of the story: ``requested`` (what the caller asked for, verbatim,
    ``null`` when they left it to us) and ``value`` (what actually engaged), plus the ``status``
    saying whether those agree. A successful load whose requested precision was declined therefore
    stays visible instead of being erased into a bare ``source: "explicit"``, which rendered no
    badge at all while the dropdown kept advertising the request."""
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
