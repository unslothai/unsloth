# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolve logical GGUF variants across remembered download folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hub.utils.gguf import GgufVariantInfo

CHAT_GGUF_TASKS = (None, "text-generation", "image-text-to-text")


def gguf_cache_snapshots(repo_id: str):
    """Active cache first, then remembered caches; newest snapshots first in each."""
    from hub.services.models.catalog_classification import _gguf_path_task
    from hub.utils.gguf import iter_hf_cache_snapshots
    from hub.utils.hf_cache_state import hf_cache_roots
    from utils.hf_cache_settings import get_hf_cache_paths

    active = get_hf_cache_paths().hub_cache
    roots = [active, *hf_cache_roots()]
    seen = set()
    for index, root in enumerate(roots):
        key = str(Path(root).resolve())
        if key in seen:
            continue
        seen.add(key)
        for snapshot in iter_hf_cache_snapshots(repo_id, root = root):
            if index and _gguf_path_task(snapshot, (repo_id,)) not in CHAT_GGUF_TASKS:
                continue
            yield snapshot


def cached_gguf_source_partial(repo_id: str, quant: str, snapshot: Path) -> bool:
    """Whether this snapshot's quant is incomplete by its own manifest, cancellation marker or blobs.

    The per-snapshot twin of the readiness check a listing runs for the cache it names. A
    caller merging remembered folders has to ask it about every source it reports: with
    ``prefer_local_cache``/``offline`` there is no Hub answer to fall back on, so an
    interrupted companion download would otherwise be advertised as a complete copy that
    the picker offers to load instead of to resume.

    Local-only by construction -- it reads the manifest and marker next to *snapshot* and
    never consults the Hub -- and judged against that same snapshot's own repo cache
    directory, since a cancellation marker belongs to the attempt that wrote it.
    """
    from hub.utils import inventory_scan

    repo_cache_dir = snapshot.parent.parent
    return inventory_scan.is_variant_partial(
        repo_id,
        quant,
        snapshot,
        repo_cache_dir = repo_cache_dir,
        repo_signal_applies = inventory_scan.repo_signal_applies_to_snapshot(
            repo_cache_dir, snapshot
        ),
    )


def _prefer_duplicate(
    repo_id: str,
    quant: str,
    previous: Path,
    candidate: Path,
    scoped_ready = None,
) -> bool:
    """Whether *candidate* should replace *previous* as this quant's source.

    Manifest verification alone misses a copy that its own snapshot state marks partial -- a
    cancel marker or an unfinished companion -- which the merge then reports as unusable even
    though a complete duplicate of the same quant exists. Rank on the full per-snapshot verdict
    so the healthy copy wins, and keep the completed-vs-incomplete manifest rule it refines.

    *scoped_ready* adds the one verdict no local rule can see: whether a copy satisfies the
    companion set the CURRENT revision asks for, which only that copy's Hub answer knows.
    """
    if scoped_ready is not None:
        previous_ready = scoped_ready(previous, quant)
        candidate_ready = scoped_ready(candidate, quant)
        if previous_ready is not None and candidate_ready is not None:
            if previous_ready != candidate_ready:
                return candidate_ready
    if not cached_gguf_manifest_complete(
        repo_id, quant, previous
    ) and cached_gguf_manifest_complete(repo_id, quant, candidate):
        return True
    return cached_gguf_source_partial(repo_id, quant, previous) and not cached_gguf_source_partial(
        repo_id, quant, candidate
    )


def cached_gguf_manifest_complete(repo_id: str, quant: str, snapshot: Path) -> bool:
    """Prefer completed downloads without applying a newer revision's manifest to an older copy."""
    from hub.services.models.catalog_classification import _gguf_path_task
    from hub.utils.download_manifest import read_manifest, verify_against_disk

    if _gguf_path_task(snapshot, (repo_id,)) not in CHAT_GGUF_TASKS:
        return True
    manifest = read_manifest("model", repo_id, quant, hub_cache = snapshot.parent.parent.parent)
    if manifest is None or manifest.commit_hash not in (None, snapshot.name):
        return True
    return verify_against_disk(manifest, snapshot).ok


@dataclass(frozen = True)
class CachedGgufSource:
    variant: "GgufVariantInfo"
    snapshot: Path
    has_vision: bool
    # Listed by the snapshot but incomplete by its own manifest, marker or shards. The
    # fallback pass keeps such a source so a resume can name its folder; the merge then
    # reports it partial instead of as a loadable copy.
    incomplete: bool = False

    @property
    def cache_path(self) -> str:
        return str(self.snapshot.parent.parent)


def cached_gguf_sources(repo_id: str, *, scoped_ready = None) -> dict[str, CachedGgufSource]:
    """One complete source per quant. Never assemble split weights across snapshots."""
    from hub.services.models.catalog_classification import _gguf_path_task
    from hub.utils.gguf import list_local_gguf_variants
    from hub.utils.inventory_scan import complete_snapshot_variants

    sources = {}
    partials: dict[str, CachedGgufSource] = {}
    for snapshot in gguf_cache_snapshots(repo_id):
        # Media GGUFs have a separate download/load pipeline and retain their existing scope.
        if _gguf_path_task(snapshot, (repo_id,)) not in CHAT_GGUF_TASKS:
            continue
        variants, has_vision = list_local_gguf_variants(str(snapshot))
        complete = complete_snapshot_variants(str(snapshot)) or set()
        for variant in variants:
            if not variant.quant:
                continue
            key = variant.quant.lower()
            if variant.quant in complete:
                previous = sources.get(key)
                if previous is None or _prefer_duplicate(
                    repo_id,
                    variant.quant,
                    previous.snapshot,
                    snapshot,
                    scoped_ready,
                ):
                    sources[key] = CachedGgufSource(variant, snapshot, has_vision)
                partials.pop(key, None)
                continue
            # An interrupted split quant is listed but not complete. Keep it as a fallback
            # instead of dropping it, so the merge can still show the row, mark it partial
            # and point a resume at the folder that holds it.
            partials.setdefault(
                key, CachedGgufSource(variant, snapshot, has_vision, incomplete = True)
            )
    for key, source in partials.items():
        sources.setdefault(key, source)
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
