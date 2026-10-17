# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The image and video models Unsloth Studio can offer, built from the family registries.

Pure and torch-free. Rows carry no path, so account filtering keys on ``id``; the route filters
per request and this payload is never filtered in place.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel

from .diffusion_families import _FAMILIES as IMAGE_FAMILIES
from .diffusion_families import family_selectable, mirror_repo
from .diffusion_inference_info import family_inference_infos
from .video_families import pipeline_available_video_families


class MediaCatalogModel(BaseModel):
    """One image or video model: a family's base repo, or a video family's GGUF repo."""

    id: str
    kind: Literal["image", "video"]
    family: str
    display_name: str
    # Decimal bytes of the bf16 weights resident at once; None when the registry has no size.
    size_bytes: Optional[int] = None
    # A reduced-memory artifact (a GGUF repo) rather than the full-precision base.
    low_vram: bool = False


class MediaCatalogResponse(BaseModel):
    models: List[MediaCatalogModel] = []


def _gb_to_bytes(gb: Optional[float]) -> Optional[int]:
    return None if gb is None else round(gb * 1e9)


def _row(
    kind: str,
    family: str,
    repo: str,
    size_gb: Optional[float],
    low_vram: bool = False,
):
    # The unsloth mirror when there is one, as the model picker shows it; gated vendor repos stay out.
    repo_id = mirror_repo(repo) or repo
    return MediaCatalogModel(
        id = repo_id,
        kind = kind,
        family = family,
        display_name = repo_id.rsplit("/", 1)[-1],
        size_bytes = _gb_to_bytes(size_gb),
        low_vram = low_vram,
    )


def media_catalog_payload() -> MediaCatalogResponse:
    """Every selectable image family, then every available video family and its GGUF repo, in registry order."""
    bf16_gb = {
        info["family"]: info["estimated_resident_gb"].get("bf16")
        for info in family_inference_infos()
    }
    rows = [
        _row("image", fam.name, fam.base_repo, bf16_gb.get(fam.name))
        for fam in IMAGE_FAMILIES
        if family_selectable(fam)
    ]
    for fam in pipeline_available_video_families():
        sizes = fam.bf16_components_gb
        rows.append(_row("video", fam.name, fam.base_repo, sum(sizes) if sizes else None))
        if fam.gguf_repo:
            rows.append(_row("video", fam.name, fam.gguf_repo, None, low_vram = True))
    return MediaCatalogResponse(models = rows)
