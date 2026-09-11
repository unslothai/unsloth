# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pure, torch-free per-family footprint summary for the inference info endpoint.

Turns the auto-policy's per-family bf16 component table plus its per-scheme steady factors into a
static "what would this cost resident" summary the frontend shows BEFORE loading, so a user can
size the int8/fp8/nvfp4/mxfp8 tradeoff. Estimates are the STEADY resident footprint (transformer *
factor + companions), matching ``estimate_dense_quant``'s ``steady_total``. Pure: torch-free, no
GPU probing, unit-tests on CPU-only hosts.
"""

from __future__ import annotations

from typing import Any

from .diffusion_auto_policy import _FAMILY_BF16_GB, _QUANT_STEADY_FACTOR, policy_steady_factor
from .diffusion_families import _FAMILIES


def _round1(value: float) -> float:
    return round(value, 1)


def _family_base_repos() -> dict[str, str]:
    """Family name -> its DEFAULT base repo, the only base a pre-load per-family estimate can
    price. A variant load prices itself through ``estimate_dense_quant``."""
    return {fam.name.strip().lower(): fam.base_repo for fam in _FAMILIES}


def family_inference_infos() -> list[dict[str, Any]]:
    """Per-family bf16 component sizes + estimated resident footprint per quant scheme.

    One dict per family (registry order): the bf16-resident component sizes and estimated resident
    GB under bf16 and each scheme (bf16 = transformer + companions; each scheme scales the
    transformer by its steady factor and adds the same companions).
    """
    infos: list[dict[str, Any]] = []
    bases = _family_base_repos()
    for name, (transformer_gb, text_encoders_gb, vae_gb) in _FAMILY_BF16_GB.items():
        companions_gb = text_encoders_gb + vae_gb
        estimated = {"bf16": _round1(transformer_gb + companions_gb)}
        # A policy artifact is mostly fp8 by weight, so the whole-model 0.33 would advertise a
        # footprint gigabytes below what the load takes.
        policy_factor = policy_steady_factor(name, bases.get(name.strip().lower()))
        for scheme, factor in _QUANT_STEADY_FACTOR.items():
            if scheme == "nvfp4" and policy_factor is not None:
                factor = policy_factor
            estimated[scheme] = _round1(transformer_gb * factor + companions_gb)
        infos.append(
            {
                "family": name,
                "transformer_bf16_gb": _round1(transformer_gb),
                "text_encoders_bf16_gb": _round1(text_encoders_gb),
                "vae_bf16_gb": _round1(vae_gb),
                "estimated_resident_gb": estimated,
            }
        )
    return infos
