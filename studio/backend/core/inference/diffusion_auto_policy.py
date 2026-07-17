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

# Steady size of a torchao-quantised transformer relative to bf16: int8/fp8 store one byte per
# param plus per-row scales (~0.52x, plus slack for bf16 norms/embeddings/proj_out); nvfp4 packs
# two params per byte plus block scales. Measured on live int8/fp8 loads.
_QUANT_STEADY_FACTOR: dict[str, float] = {
    "int8": 0.55,
    "fp8": 0.55,
    "mxfp8": 0.58,
    "nvfp4": 0.33,
}

# bf16-RESIDENT component sizes in decimal GB: (transformer, text encoders, VAE). What the
# components occupy on device after the dtype cast, NOT the download size (Z-Image ships fp32:
# 24.6 GB of shards -> 12.3 GB bf16). From each base repo's HF sibling metadata, cross-checked
# against the training-side dense_bf16_gb table and measured loads.
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
    # 17B MoE DiT (34.2 GB bf16) + FOUR text encoders: CLIP-L 0.5 + CLIP-G 2.8 + T5-XXL 9.5
    # from the repo, plus the Llama-3.1-8B text_encoder_4 (~16 GB bf16) assembled from the
    # open mirror at load time (diffusion_hidream.py).
    "hidream-i1": (34.2, 28.8, 0.2),
    # Two ~9.3B DiTs (conditional + unconditional_transformer for Ideogram's dual-branch CFG),
    # both resident, plus a Qwen3-VL encoder. The vendor stores them as raw float8; these are the
    # bf16-resident sizes after the dtype cast, so each doubles (37.2 = 2 x 18.6, encoder 16.3).
    "ideogram-4": (37.2, 16.3, 0.2),
}

# Base-repo overrides for families offering multiple sizes under one entry (the table carries the
# family default).
_BASE_REPO_BF16_GB: dict[str, tuple[float, float, float]] = {
    "black-forest-labs/FLUX.2-klein-9B": (18.2, 16.4, 0.2),
}


def family_bf16_components_gb(
    fam: Any, base_repo: Optional[str] = None
) -> Optional[tuple[float, float, float]]:
    """(transformer, text encoders, VAE) bf16-resident sizes in GB, or None when the family isn't
    in the table (callers fall back to file-size estimates)."""
    if base_repo:
        override = _BASE_REPO_BF16_GB.get(base_repo)
        if override is not None:
            return override
    name = getattr(fam, "name", None)
    return _FAMILY_BF16_GB.get(name) if name else None


@dataclass(frozen = True)
class DenseQuantEstimate:
    """Footprint estimate for one dense transformer-quant candidate.

    ``transient_transformer_mib`` is the build peak (dense bf16 when quantising on the fly, or the
    quantised size when a prequant checkpoint loads via meta). ``steady_transformer_mib`` is what
    stays resident for generation."""

    scheme: str
    steady_transformer_mib: int
    transient_transformer_mib: int
    companions_mib: int
    prequant: bool

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
    return DenseQuantEstimate(
        scheme = scheme,
        steady_transformer_mib = steady,
        transient_transformer_mib = transient,
        companions_mib = companions,
        prequant = prequant_available,
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
    # force_dense: the loader will SKIP the prequant shortcut (e.g. a LoRA bake attaches
    # adapters on the dense transformer), so the candidate must be sized for the dense build.
    if not force_dense:
        try:
            from .diffusion_prequant import usable_prequant_source

            # usable_ (not resolve_): a local path override counts only when the loader will
            # accept it (allowlisted AND present), else load_prequantized_transformer refuses it
            # and rebuilds dense after the resident pipe is unloaded (the evict-then-OOM this
            # prefetch avoids).
            src = usable_prequant_source(
                fam, scheme, path_override = prequant_path, base_repo = base_repo
            )
            prequant_available = src is not None
        except Exception:  # noqa: BLE001 -- prequant probing must never sink the candidate
            prequant_available = False
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
    if estimate is not None:
        # The dense path may DOWNLOAD the artifact into the HF cache; this must never wedge a
        # nearly-full disk. An already-cached re-download is a no-op, so the gate only
        # false-positives on an already-critically-full disk, where the GGUF fallback is right anyway.
        needed_mib = (
            estimate.steady_transformer_mib
            if estimate.prequant
            else estimate.transient_transformer_mib
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


# ── resolved-record (status surface) ─────────────────────────────────────────
def build_resolved_record(
    controls: dict[str, tuple[Optional[Any], Any, str]],
) -> dict[str, dict[str, Any]]:
    """The per-control ``resolved`` record for status: engaged value + provenance.

    ``controls`` maps a control name to ``(explicit, engaged, reason)``, where ``explicit`` is the
    raw request (None / "" / "auto" = left to the backend). Rendered as an "Auto: X" badge."""
    record: dict[str, dict[str, Any]] = {}
    for name, (explicit, engaged, reason) in controls.items():
        left_to_backend = explicit is None or (
            isinstance(explicit, str) and explicit.strip().lower() in ("", "auto")
        )
        record[name] = {
            "value": engaged,
            "source": "auto" if left_to_backend else "explicit",
            "reason": reason,
        }
    return record
