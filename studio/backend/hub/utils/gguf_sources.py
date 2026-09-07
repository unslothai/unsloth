# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolve logical GGUF variants across remembered download folders."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hub.utils.gguf import GgufVariantInfo

CHAT_GGUF_TASKS = (None, "text-generation", "image-text-to-text")


def gguf_cache_snapshots(repo_id: str):
    """Active cache first, then remembered caches; newest snapshots first in each."""
    from hub.utils.hf_cache_state import hf_cache_roots
    from utils.models.model_config import _iter_hf_cache_snapshots
    from utils.hf_cache_settings import get_hf_cache_paths

    active = get_hf_cache_paths().hub_cache
    roots = [active, *hf_cache_roots()]
    seen = set()
    for root in roots:
        key = str(Path(root).resolve())
        if key in seen:
            continue
        seen.add(key)
        yield from _iter_hf_cache_snapshots(repo_id, cache_dir = root)


@dataclass(frozen = True)
class CachedGgufSource:
    variant: "GgufVariantInfo"
    snapshot: Path
    has_vision: bool

    @property
    def cache_path(self) -> str:
        return str(self.snapshot.parent.parent)


def cached_gguf_sources(repo_id: str) -> dict[str, CachedGgufSource]:
    """One complete source per quant. Never assemble split weights across snapshots."""
    from hub.services.models.catalog_classification import _gguf_path_task
    from hub.utils.gguf import list_local_gguf_variants
    from hub.utils.inventory_scan import complete_snapshot_variants

    sources = {}
    for snapshot in gguf_cache_snapshots(repo_id):
        # Media GGUFs have a separate download/load pipeline and retain their existing scope.
        if _gguf_path_task(snapshot, (repo_id,)) not in CHAT_GGUF_TASKS:
            continue
        variants, has_vision = list_local_gguf_variants(str(snapshot))
        complete = complete_snapshot_variants(str(snapshot)) or set()
        for variant in variants:
            if variant.quant and variant.quant in complete:
                sources.setdefault(
                    variant.quant.lower(), CachedGgufSource(variant, snapshot, has_vision)
                )
    return sources


def cached_gguf_action_path(
    repo_id: str,
    variant: str | None,
    cache_path: str | None = None,
) -> str | None:
    """An explicit folder wins; logical quant actions use the same source as the listing."""
    if cache_path or not variant:
        return cache_path
    source = cached_gguf_sources(repo_id).get(variant.lower())
    return source.cache_path if source else None
