# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Delete preflight and orphaned-companion cleanup for image-model assets.

Three answers live here, all derived from the cache scan at call time (see
``hub.utils.companion_assets`` for why nothing is counted):

  :func:`delete_impact_response`  what a pending delete reclaims, and what it leaves behind
  :func:`companion_dependents`    who still needs a companion base, used as a delete guard
  :func:`orphan_companions_response`  companion bases no installed model needs any more

Sizes are real on-disk blob bytes from the HF cache scan, deduped per blob, not Hub metadata:
the number in a delete dialog has to be the number the disk gives back.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from loggers import get_logger

from hub.utils import companion_assets
from hub.utils.gguf import extract_quant_label
from hub.services.models import cache_inventory
from hub.services.models.common import _is_gguf_filename, _is_main_gguf_filename
from hub.utils.paths import is_valid_gguf_variant as _is_valid_gguf_variant
from hub.utils.paths import is_valid_repo_id as _is_valid_repo_id

logger = get_logger(__name__)


def _repo_blob_bytes(repo_info, *, only = None) -> int:
    """On-disk bytes of *repo_info*, deduped by blob so a file shared across revisions counts once.

    ``only`` is an optional predicate on the snapshot-relative file name.
    """
    unique: dict[str, int] = {}
    for revision in getattr(repo_info, "revisions", ()) or ():
        rev_id = getattr(revision, "commit_hash", None) or str(id(revision))
        snapshot = getattr(revision, "snapshot_path", None)
        for f in getattr(revision, "files", ()) or ():
            name = str(getattr(f, "file_name", "") or "")
            path = getattr(f, "file_path", None)
            if path and snapshot:
                try:
                    name = Path(path).relative_to(Path(snapshot)).as_posix()
                except ValueError:
                    pass
            if only is not None and not only(name):
                continue
            blob_path = getattr(f, "blob_path", None)
            size = int(getattr(f, "size_on_disk", 0) or 0)
            unique[str(blob_path) if blob_path else f"{rev_id}:{name}"] = size
    return sum(unique.values())


_DENOISER_DIRS = ("transformer/", "unet/")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".ckpt", ".pt", ".pth", ".gguf")


def _holds_denoiser(name: str) -> bool:
    """A denoiser weight: a GGUF anywhere, or a shard under the pipeline's denoiser folder."""
    lowered = name.lower()
    if _is_gguf_filename(name):
        return True
    return lowered.startswith(_DENOISER_DIRS) and lowered.endswith(_WEIGHT_SUFFIXES)


def _repos_by_id(cache_scans) -> dict[str, list]:
    out: dict[str, list] = {}
    for scan in cache_scans or ():
        for repo in getattr(scan, "repos", ()) or ():
            try:
                if str(getattr(repo, "repo_type", "")) != "model":
                    continue
                key = str(getattr(repo, "repo_id", "") or "").strip().lower()
            except Exception:  # noqa: BLE001 -- one unreadable row never hides the rest
                continue
            if key:
                out.setdefault(key, []).append(repo)
    return out


def _variant_bytes(repo_info, variant: str) -> int:
    target = variant.strip().lower()

    def _matches(name: str) -> bool:
        return _is_main_gguf_filename(name) and extract_quant_label(name).lower() == target

    return _repo_blob_bytes(repo_info, only = _matches)


def _remaining_main_gguf_variants(repo_info, *, excluding: Optional[str] = None) -> set[str]:
    skip = (excluding or "").strip().lower()
    found: set[str] = set()
    for revision in getattr(repo_info, "revisions", ()) or ():
        snapshot = getattr(revision, "snapshot_path", None)
        for f in getattr(revision, "files", ()) or ():
            name = str(getattr(f, "file_name", "") or "")
            path = getattr(f, "file_path", None)
            if path and snapshot:
                try:
                    name = Path(path).relative_to(Path(snapshot)).as_posix()
                except ValueError:
                    pass
            if not _is_main_gguf_filename(name):
                continue
            label = extract_quant_label(name).lower()
            if label and label != skip:
                found.add(label)
    return found


def companion_dependents(
    base_repo_id: str,
    cache_scans = None,
    *,
    ignore_repo_ids = (),
) -> list[str]:
    """Installed checkpoints that would still need *base_repo_id* after ignoring *ignore_repo_ids*.

    Sorted for a stable message. Empty means the base is safe to remove.
    """
    scans = cache_scans if cache_scans is not None else cache_inventory.all_hf_cache_scans()
    required = companion_assets.required_companion_bases(scans, ignore_repo_ids = ignore_repo_ids)
    return sorted(required.get((base_repo_id or "").strip().lower(), set()))


def _delete_impact_blocking(repo_id: str, variant: Optional[str]) -> dict:
    scans = cache_inventory.all_hf_cache_scans()
    by_id = _repos_by_id(scans)
    key = repo_id.strip().lower()
    repos = by_id.get(key, [])

    reclaimed = 0
    for repo_info in repos:
        reclaimed += _variant_bytes(repo_info, variant) if variant else _repo_blob_bytes(repo_info)

    # Would this delete leave the repo with no runnable checkpoint? Only then can its companions
    # become reclaimable; while a sibling quant survives they stay in use.
    removes_last_checkpoint = True
    if variant:
        for repo_info in repos:
            if _remaining_main_gguf_variants(repo_info, excluding = variant):
                removes_last_checkpoint = False
                break

    ignore = [repo_id] if removes_last_checkpoint else []
    required_after = companion_assets.required_companion_bases(scans, ignore_repo_ids = ignore)

    # Companion bases THIS pick uses, from the same derivation the loader's resolver feeds.
    own_bases = companion_assets.required_companion_bases(
        [_SingleRepoScan(repos)] if repos else [],
    )
    retained: list[dict] = []
    freeable: list[dict] = []
    for base_key in sorted(own_bases):
        base_repos = by_id.get(base_key, [])
        if not base_repos:
            continue
        base_bytes = sum(_repo_blob_bytes(r) for r in base_repos)
        display = str(getattr(base_repos[0], "repo_id", base_key))
        holders = sorted(required_after.get(base_key, set()))
        entry = {"repo_id": display, "size_bytes": base_bytes, "needed_by": holders}
        (retained if holders else freeable).append(entry)

    return {
        "repo_id": repo_id,
        "variant": variant,
        "reclaimed_bytes": reclaimed,
        "retained_companions": retained,
        "freeable_companions": freeable,
        "blocked_by": (
            companion_dependents(repo_id, scans, ignore_repo_ids = [repo_id])
            if companion_assets.is_companion_base(repo_id) and variant is None
            else []
        ),
    }


class _SingleRepoScan:
    """Adapter presenting a fixed repo list with the attribute the derivation reads."""

    def __init__(self, repos):
        self.repos = repos


async def delete_impact_response(repo_id: str, variant: Optional[str] = None) -> dict:
    """What a delete of *repo_id* (/*variant*) would reclaim, retain, and be blocked by."""
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(status_code = 400, detail = "Invalid repo_id format")
    variant = (variant or "").strip() or None
    if variant is not None and not _is_valid_gguf_variant(variant):
        raise HTTPException(status_code = 400, detail = f"Invalid gguf_variant: {variant!r}")
    return await asyncio.to_thread(_delete_impact_blocking, repo_id, variant)


def _orphan_companions_blocking() -> dict:
    scans = cache_inventory.all_hf_cache_scans()
    by_id = _repos_by_id(scans)
    required = companion_assets.required_companion_bases(scans)
    known = companion_assets.known_companion_base_ids()

    orphans: list[dict] = []
    for base_key in sorted(known & set(by_id)):
        if required.get(base_key):
            continue
        repos = by_id[base_key]
        # A repo that holds a runnable denoiser is a model the user installed, not a leftover.
        # Several curated bases are perfectly good pipelines in their own right, so this is the
        # difference between the two ways the same repo id reaches the cache: a companion fetch
        # takes everything BUT the denoiser folder (``_base_file_downloaded`` skips
        # ``transformer/``), while a pipeline pick takes it. Its presence is therefore the
        # derived answer to "did the user ask for this repo, or did a GGUF drag it in".
        if any(_repo_blob_bytes(r, only = _holds_denoiser) for r in repos):
            continue
        # One row per cache root. A delete is scoped to a single cache, so pooling copies from
        # several would promise bytes one removal cannot deliver.
        for repo in repos:
            size = _repo_blob_bytes(repo)
            if size <= 0:
                continue
            # The repo dir itself, not its parent: ``scoped_delete_root`` resolves the owning
            # cache by walking up to the ``models--`` component, so a bare root resolves to
            # nothing and the delete comes back "Invalid cache_path".
            try:
                cache_path = str(Path(getattr(repo, "repo_path")))
            except (TypeError, OSError):
                cache_path = None
            orphans.append(
                {
                    "repo_id": str(getattr(repo, "repo_id", base_key)),
                    "size_bytes": size,
                    "cache_path": cache_path,
                }
            )
    return {
        "companions": orphans,
        "total_bytes": sum(o["size_bytes"] for o in orphans),
    }


async def orphan_companions_response() -> dict:
    """Cached companion bases that no installed model needs. Listing only; nothing is deleted."""
    return await asyncio.to_thread(_orphan_companions_blocking)
