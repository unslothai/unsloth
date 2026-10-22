# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""delete previews and orphan cleanup report deduplicated HF cache blob bytes, not Hub metadata."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from loggers import get_logger

from hub.utils import companion_assets, download_registry
from hub.utils.gguf import gguf_variant_key
from hub.services.models import cache_inventory
from hub.services.models.common import _is_main_gguf_filename
from hub.utils.paths import is_valid_gguf_variant as _is_valid_gguf_variant
from hub.utils.paths import is_valid_repo_id as _is_valid_repo_id

logger = get_logger(__name__)


def _repo_blob_bytes(repo_info, *, only = None) -> int:
    """return deduplicated bytes; filtered blobs count only if every snapshot reference matches."""
    unique: dict[str, int] = {}
    retained: set[str] = set()
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
            from hub.services.models.cache_inventory import _blob_key

            key = _blob_key(f, f"{rev_id}:{name}")
            if only is not None and not only(name, Path(path) if path else None):
                retained.add(key)
                continue
            unique[key] = int(getattr(f, "size_on_disk", 0) or 0)
    return sum(size for key, size in unique.items() if key not in retained)


_repo_holds_denoiser = companion_assets.repo_holds_denoiser


def _account_scans() -> list:
    """managed callers see only repositories covered by their grants."""
    scans = cache_inventory.all_hf_cache_scans()
    access = cache_inventory._account_access()
    if not access.managed_account():
        return scans
    return [
        replace(scan, repos = frozenset(access.filter_model_rows(list(scan.repos or ()))))
        for scan in scans
    ]


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


def _variant_keys(
    repo_info,
    variant: str,
    preferred_path: Optional[str] = None,
) -> set[str]:
    """use deletion keys because path-qualified checkpoints do not match bare quant labels."""
    from hub.services.models.deletion import _variant_keys_to_delete
    return {key.lower() for key in _variant_keys_to_delete(repo_info, variant, preferred_path)}


def _variant_bytes(
    repo_info,
    variant: str,
    preferred_path: Optional[str] = None,
) -> int:
    from hub.services.models.deletion import (
        _audio_cpp_package_scope,
        _main_gguf_variant_keys,
        _main_key,
    )

    wanted = _variant_keys(repo_info, variant, preferred_path)
    main_keys = _main_gguf_variant_keys(repo_info, preferred_path)
    package_files, protected = _audio_cpp_package_scope(repo_info, variant)

    def _matches(name: str, path: Optional[Path]) -> bool:
        return (
            path is not None
            and name not in protected
            and (
                name in package_files
                or (_is_main_gguf_filename(name) and _main_key(path, main_keys) in wanted)
            )
        )

    return _repo_blob_bytes(repo_info, only = _matches)


def _remaining_main_gguf_variants(
    repo_info,
    *,
    excluding: Optional[str] = None,
    preferred_path: Optional[str] = None,
) -> set[str]:
    from hub.services.models.deletion import (
        _audio_cpp_package_scope,
        _main_gguf_variant_keys,
        _repo_file_matches,
    )

    skip = _variant_keys(repo_info, excluding, preferred_path) if excluding else set()
    package_files, protected = (
        _audio_cpp_package_scope(repo_info, excluding) if excluding else (frozenset(), frozenset())
    )
    removed_package_paths = {
        snap
        for snap, _blob, name in _repo_file_matches(repo_info, lambda name: name in package_files)
        if name not in protected
    }
    # deletion keys keep sibling builds with the same quant distinct
    return {
        key.lower()
        for path, key in _main_gguf_variant_keys(repo_info, preferred_path).items()
        if path not in removed_package_paths and key and key.lower() not in skip
    }


def companion_dependents(
    base_repo_id: str,
    cache_scans = None,
    *,
    ignore_repo_ids = (),
) -> list[str]:
    """return sorted checkpoints needing *base_repo_id*; an empty result makes it removable."""
    scans = cache_scans if cache_scans is not None else cache_inventory.all_hf_cache_scans()
    required = companion_assets.required_companion_bases(scans, ignore_repo_ids = ignore_repo_ids)
    return sorted(required.get((base_repo_id or "").strip().lower(), set()))


def _variant_is_a_required_companion_asset(repo_id: str, variant: str) -> bool:
    """share the deletion guard so preview and refusal agree."""
    from hub.services.models.deletion import _variant_is_a_required_companion_asset as _impl
    return _impl(repo_id, variant)


def _delete_impact_blocking(
    repo_id: str,
    variant: Optional[str],
    cache_path: Optional[str] = None,
    preferred_path: Optional[str] = None,
) -> dict:
    scans = _account_scans()
    by_id = _repos_by_id(scans)
    key = repo_id.strip().lower()
    all_copies = by_id.get(key, [])
    from hub.utils.gguf_sources import cached_gguf_action_path

    cache_path = cached_gguf_action_path(repo_id, variant, cache_path, preferred_path)
    repos = all_copies
    surviving = []
    if cache_path:
        from hub.utils.hf_cache_state import resolve_delete_target_root

        root = resolve_delete_target_root(
            "model",
            repo_id,
            cache_path,
            {Path(repo.repo_path).parent.resolve() for repo in all_copies},
        )
        if root is None:
            raise HTTPException(status_code = 400, detail = "Invalid cache_path")
        repos = [repo for repo in all_copies if Path(repo.repo_path).parent.resolve() == root]
        surviving = [repo for repo in all_copies if repo not in repos]

    reclaimed = 0
    for repo_info in repos:
        reclaimed += (
            _variant_bytes(repo_info, variant, preferred_path)
            if variant
            else _repo_blob_bytes(repo_info)
        )

    # companions become reclaimable only when no runnable checkpoint survives
    removes_last_checkpoint = not any(_repo_holds_denoiser(repo) for repo in surviving)
    if variant:
        for repo_info in repos:
            if _remaining_main_gguf_variants(
                repo_info, excluding = variant, preferred_path = preferred_path
            ):
                removes_last_checkpoint = False
                break

    ignore = [repo_id] if removes_last_checkpoint else []
    required_after = companion_assets.required_companion_bases(scans, ignore_repo_ids = ignore)

    own_bases = companion_assets.required_companion_bases(
        [_SingleRepoScan(repos)] if repos else [],
    )
    retained: list[dict] = []
    freeable: list[dict] = []
    offerable = companion_assets.known_companion_base_ids()
    for base_key in sorted(own_bases):
        base_repos = by_id.get(base_key, [])
        if not base_repos:
            continue
        base_bytes = sum(_repo_blob_bytes(r) for r in base_repos)
        display = str(getattr(base_repos[0], "repo_id", base_key))
        holders = sorted(required_after.get(base_key, set()))
        entry = {"repo_id": display, "size_bytes": base_bytes, "needed_by": holders}
        if holders:
            retained.append(entry)
        # The SAME offerability test orphan_companions_response applies, since this row points at that list: a borrowed chat GGUF repo is a curated companion id but holds a denoiser, so advertising it sent the user to Free up space to remove a row that is never there.
        elif base_key in offerable and any(not _repo_holds_denoiser(r) for r in base_repos):
            freeable.append(entry)

        # A base only a recorded link names: the orphan endpoint is table-only by design, so advertising it here pointed the user at a Free up space list it will never appear in.
    return {
        "repo_id": repo_id,
        "variant": variant,
        "reclaimed_bytes": reclaimed,
        "cache_path": cache_path,
        "retained_companions": retained,
        "freeable_companions": freeable,
        # Same predicate the destructive path uses: the native Qwen-Image encoder is a named quant inside a chat GGUF repo, so previewing only whole-repo deletes left Delete enabled and the refusal arriving after the user confirmed.
        "blocked_by": (
            companion_dependents(repo_id, scans, ignore_repo_ids = [repo_id])
            if companion_assets.is_companion_base(repo_id)
            and (variant is None or _variant_is_a_required_companion_asset(repo_id, variant))
            else []
        ),
    }


class _SingleRepoScan:
    """present a fixed repo list through the scan interface."""

    def __init__(self, repos):
        self.repos = repos


def _folder_delete_impact_blocking(
    repo_id: str, variant: Optional[str], umbrella: str, keys: list[str], cache_path: Optional[str]
) -> dict:
    """The impact of deleting an umbrella folder row: the sum over its own variant keys."""
    impacts = [_delete_impact_blocking(umbrella, key, cache_path) for key in keys]

    def merged(field: str) -> list[dict]:
        rows = {row["repo_id"]: row for impact in impacts for row in impact[field]}
        return list(rows.values())

    return {
        "repo_id": repo_id,
        "variant": variant,
        "reclaimed_bytes": sum(impact["reclaimed_bytes"] for impact in impacts)
        + (_shared_folder_extra_bytes(repo_id, keys) if len(keys) > 1 else 0),
        "cache_path": impacts[0]["cache_path"],
        "retained_companions": merged("retained_companions"),
        "freeable_companions": merged("freeable_companions"),
        "blocked_by": sorted({name for impact in impacts for name in impact["blocked_by"]}),
    }


def _shared_folder_extra_bytes(repo_id: str, keys: list[str]) -> int:
    """Bytes of the files several deleted variants of one folder share and no kept variant lists
    (a voice embedding beside two quants). Each per-variant preview counts them as still needed by
    the other, but deleting the variants in turn removes them with the last."""
    from core.inference.audio_cpp_models import resolve, snapshot_dirs

    model = resolve(repo_id, network = False)
    if model is None or not model.repo_id:
        return 0
    from hub.services.models import downloads

    # The delete keeps a variant's extras while a sibling in its directory downloads; so does this.
    busy = {
        key.rsplit("/", 1)[0].lower() + "/"
        for key in keys
        if downloads.registry.has_active_peer_variant(
            model.repo_id, key, prefix = key.rsplit("/", 1)[0] + "/"
        )
    }
    deleting = {key.lower() for key in keys}
    listed: dict[str, int] = {}
    kept: set[str] = set()
    for candidate in model.variants:
        files = {f.path for f in candidate.files}
        if candidate.primary and gguf_variant_key(candidate.primary).lower() in deleting:
            for path in files:
                listed[path] = listed.get(path, 0) + 1
        else:
            kept |= files
    total = 0
    seen: set[Path] = set()
    shared = (
        p
        for p, count in listed.items()
        if count > 1 and p not in kept and not any(p.lower().startswith(d) for d in busy)
    )
    for path in sorted(shared):
        for snapshot in snapshot_dirs(model.repo_id):
            try:
                blob = (snapshot / path).resolve(strict = True)
            except OSError:
                continue
            if blob not in seen:
                seen.add(blob)
                total += blob.stat().st_size
    return total


async def delete_impact_response(
    repo_id: str,
    variant: Optional[str] = None,
    cache_path: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> dict:
    """preview bytes, retained assets, and blockers for deleting *repo_id* or its *variant*."""
    from core.inference.audio_cpp_models import cached_folder_target

    folder = await asyncio.to_thread(cached_folder_target, repo_id, (variant or "").strip() or None)
    if folder is not None:
        umbrella, keys = folder
        if not keys:
            raise HTTPException(status_code = 404, detail = "Model not found in cache")
        return await asyncio.to_thread(
            _folder_delete_impact_blocking, repo_id, variant, umbrella, keys, cache_path
        )
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(status_code = 400, detail = "Invalid repo_id format")
    variant = (variant or "").strip() or None
    if variant is not None and not _is_valid_gguf_variant(variant):
        raise HTTPException(status_code = 400, detail = f"Invalid gguf_variant: {variant!r}")
    preferred_path = None
    if variant is not None:
        try:
            from hub.services.models import gguf_variants
            preferred_path = await gguf_variants.cached_gguf_variant_source_path(
                repo_id, variant, hf_token
            )
        except Exception as exc:  # noqa: BLE001 -- local readiness is the offline fallback
            logger.debug(
                "Could not resolve the scoped GGUF source for %s [%s]: %s",
                repo_id,
                variant,
                download_registry.scrub_secrets(str(exc), hf_token = hf_token),
            )
    return await asyncio.to_thread(
        _delete_impact_blocking, repo_id, variant, cache_path, preferred_path
    )


def _orphan_companions_blocking() -> dict:
    scans = _account_scans()
    by_id = _repos_by_id(scans)
    required = companion_assets.required_companion_bases(scans)
    known = companion_assets.known_companion_base_ids()
    # An artifact mirror with a GGUF in ANY root is an installed model: a folders-only copy of it in another root is
    # what that GGUF loads its companions from, never an orphan. The per-root denoiser check below cannot see that.
    artifact_ggufs = cache_inventory._artifact_gguf_repo_ids(scans)

    orphans: list[dict] = []
    for base_key in sorted(known & set(by_id)):
        if required.get(base_key) or base_key in artifact_ggufs:
            continue
        repos = by_id[base_key]
        # exclude user-installed copies with denoisers; companion fetches omit transformer/ and orphan rows are per root.
        # An artifact mirror is judged by its root checkpoints only (an FP8 mirror's checkpoint is installed): a
        # pipeline load of its id is refused, so with no GGUF left (checked above) transformer/ shards a GGUF speed
        # load left are no installed checkpoint.
        folders = not cache_inventory._hosts_artifacts(base_key)
        repos = [r for r in repos if not _repo_holds_denoiser(r, folders = folders)]
        if not repos:
            continue
        # One row per cache root: a delete is scoped to a single cache, so pooling copies from several would promise bytes one removal cannot deliver.
        for repo in repos:
            size = _repo_blob_bytes(repo)
            if size <= 0:
                continue
            # The repo dir itself, not its parent: scoped_delete_root walks up to the models-- component, so a bare root resolves to nothing and the delete comes back "Invalid cache_path".
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
