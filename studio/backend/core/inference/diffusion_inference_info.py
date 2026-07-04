# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pure, torch-free per-family footprint summary for the inference info endpoint.

The Advanced Dtype selector offers int8/fp8/nvfp4/mxfp8 dense transformer-quant
schemes. This module turns the auto-policy's per-family bf16 component table plus
its per-scheme steady factors into a static "what would this cost resident" summary
the frontend can show BEFORE anything is loaded, so a user can size the tradeoff.

Estimates here are the STEADY resident footprint (transformer * factor + companions),
not the transient build peak -- the same quantity ``estimate_dense_quant`` reports as
``steady_total`` -- so the numbers match what stays on the card during generation.

Pure by design: only imports the auto-policy tables (themselves torch-free), does no
GPU probing, so it unit-tests on CPU-only hosts.
"""

from __future__ import annotations

from typing import Any

from .diffusion_auto_policy import _FAMILY_BF16_GB, _QUANT_STEADY_FACTOR


def _round1(value: float) -> float:
    return round(value, 1)


def family_inference_infos() -> list[dict[str, Any]]:
    """Per-family bf16 component sizes + estimated resident footprint per quant scheme.

    One dict per family in the auto-policy bf16 table (registry order), each carrying the
    bf16-resident component sizes and the estimated resident GB under bf16 and each dense
    quant scheme. bf16's estimate is the un-quantised transformer + companions; each
    scheme scales the transformer by its steady factor and adds the same companions.
    """
    infos: list[dict[str, Any]] = []
    for name, (transformer_gb, text_encoders_gb, vae_gb) in _FAMILY_BF16_GB.items():
        companions_gb = text_encoders_gb + vae_gb
        estimated = {"bf16": _round1(transformer_gb + companions_gb)}
        for scheme, factor in _QUANT_STEADY_FACTOR.items():
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
