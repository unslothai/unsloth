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
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from loggers import get_logger

from hub.utils import companion_assets
from hub.utils.gguf import gguf_variant_key
from hub.utils.hf_cache_state import AmbiguousDeleteTargetError, resolve_delete_target_root
from hub.utils.state_dir import normalize_hub_cache
from hub.services.models import cache_inventory
from hub.services.models.common import _is_main_gguf_filename
from hub.utils.paths import is_valid_gguf_variant as _is_valid_gguf_variant
from hub.utils.paths import is_valid_repo_id as _is_valid_repo_id
from utils.paths.path_utils import is_appledouble_metadata

logger = get_logger(__name__)

_DELETE_IMPACT_LOAD_STATE_TIMEOUT_S = 1.0
_DELETE_IMPACT_LOAD_STATE_CAPACITY = threading.BoundedSemaphore(2)
_DELETE_IMPACT_LOAD_STATE_LOCK = threading.Lock()
_DELETE_IMPACT_LOAD_STATE_PROBES: dict[tuple[str, Optional[str], Optional[str]], Future] = {}


def _delete_impact_load_state_probe_key(
    repo_id: str, variant: Optional[str], cache_root: Optional[str | Path]
) -> tuple[str, Optional[str], Optional[str]]:
    return (
        repo_id.strip().lower(),
        (variant or "").strip().lower() or None,
        normalize_hub_cache(cache_root) if cache_root is not None else None,
    )


def _submit_delete_impact_load_state_probe(
    target,
    repo_id: str,
    variant: Optional[str],
    cache_root: Optional[str | Path] = None,
):
    capacity = _DELETE_IMPACT_LOAD_STATE_CAPACITY
    key = _delete_impact_load_state_probe_key(repo_id, variant, cache_root)
    with _DELETE_IMPACT_LOAD_STATE_LOCK:
        current = _DELETE_IMPACT_LOAD_STATE_PROBES.get(key)
        if current is not None and not current.done():
            return current
        if not capacity.acquire(blocking = False):
            return None
        probe = Future()
        _DELETE_IMPACT_LOAD_STATE_PROBES[key] = probe

    def run_probe() -> None:
        if not probe.set_running_or_notify_cancel():
            return
        try:
            if cache_root is None:
                result = target(repo_id, variant)
            else:
                result = target(repo_id, variant, cache_root = cache_root)
        except BaseException as exc:
            probe.set_exception(exc)
        else:
            probe.set_result(result)

    def release_probe(_future: Future) -> None:
        with _DELETE_IMPACT_LOAD_STATE_LOCK:
            if _DELETE_IMPACT_LOAD_STATE_PROBES.get(key) is probe:
                _DELETE_IMPACT_LOAD_STATE_PROBES.pop(key, None)
        capacity.release()

    try:
        threading.Thread(
            target = run_probe,
            name = "delete-impact-load-state",
            daemon = True,
        ).start()
    except BaseException:
        with _DELETE_IMPACT_LOAD_STATE_LOCK:
            if _DELETE_IMPACT_LOAD_STATE_PROBES.get(key) is probe:
                _DELETE_IMPACT_LOAD_STATE_PROBES.pop(key, None)
        capacity.release()
        raise
    probe.add_done_callback(release_probe)
    return probe


def _abandon_delete_impact_load_state_probe(
    repo_id: str,
    variant: Optional[str],
    probe: Future,
    cache_root: Optional[str | Path] = None,
) -> None:
    key = _delete_impact_load_state_probe_key(repo_id, variant, cache_root)
    with _DELETE_IMPACT_LOAD_STATE_LOCK:
        if _DELETE_IMPACT_LOAD_STATE_PROBES.get(key) is probe:
            _DELETE_IMPACT_LOAD_STATE_PROBES.pop(key, None)


def _retrieve_probe_exception(probe: asyncio.Future) -> None:
    if not probe.cancelled():
        probe.exception()


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


# One definition, so the orphan listing and the delete preview cannot disagree about which cached
# repos are leftovers (see companion_assets.repo_holds_denoiser).
_repo_holds_denoiser = companion_assets.repo_holds_denoiser


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


def _variant_keys(repo_info, variant: str) -> set[str]:
    """The variant keys *variant* names in *repo_info*, from the destructive path's own resolver.

    The inventory and the delete both identify a row by ``gguf_variant_key``, which for a
    path-qualified checkpoint (``distilled/ltx-2.3-22b-distilled-Q6_K``) is not the bare quant
    label. Comparing labels here made the preview miss the file entirely: 0 B reclaimed, and the
    last checkpoint of a repo read as if a sibling survived, so its companions were described as
    retained rather than freed."""
    from hub.services.models.deletion import _variant_keys_to_delete
    return {key.lower() for key in _variant_keys_to_delete(repo_info, variant)}


def _variant_bytes(repo_info, variant: str) -> int:
    wanted = _variant_keys(repo_info, variant)

    def _matches(name: str) -> bool:
        return _is_main_gguf_filename(name) and gguf_variant_key(name).lower() in wanted

    return _repo_blob_bytes(repo_info, only = _matches)


def _remaining_main_gguf_variants(repo_info, *, excluding: Optional[str] = None) -> set[str]:
    skip = _variant_keys(repo_info, excluding) if excluding else set()
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
            # The delete this previews ignores proven metadata, so counting it here reports a
            # checkpoint as surviving that the deletion itself does not see.
            if path and is_appledouble_metadata(Path(path)):
                continue
            key = gguf_variant_key(name).lower()
            if key and key not in skip:
                found.add(key)
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


def _variant_is_a_required_companion_asset(repo_id: str, variant: str) -> bool:
    """The deletion guard's predicate, shared so the preview and the refusal cannot disagree."""
    from hub.services.models.deletion import _variant_is_a_required_companion_asset as _impl
    return _impl(repo_id, variant)


def _delete_impact_blocking(
    repo_id: str, variant: Optional[str], cache_path: Optional[str]
) -> dict:
    scans = cache_inventory.all_hf_cache_scans()
    by_id = _repos_by_id(scans)
    key = repo_id.strip().lower()
    all_repos = by_id.get(key, [])
    owners: dict[Path, list] = {}
    for repo_info in all_repos:
        try:
            owner = Path(repo_info.repo_path).parent.resolve(strict = False)
        except (OSError, RuntimeError, ValueError):
            continue
        owners.setdefault(owner, []).append(repo_info)
    try:
        target_root = resolve_delete_target_root("model", repo_id, cache_path, owners.keys())
    except AmbiguousDeleteTargetError as exc:
        raise HTTPException(status_code = 409, detail = exc.detail) from exc
    if target_root is None:
        raise HTTPException(status_code = 400, detail = "Invalid cache_path")
    repos = owners.get(target_root, [])

    reclaimed = 0
    for repo_info in repos:
        reclaimed += _variant_bytes(repo_info, variant) if variant else _repo_blob_bytes(repo_info)

    # Would this delete leave the repo with no runnable checkpoint? Only then can its companions
    # become reclaimable; while a sibling quant survives they stay in use.
    removes_last_checkpoint = len(repos) == len(all_repos)
    if variant and removes_last_checkpoint:
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
        # The SAME offerability test orphan_companions_response applies, because this row is a
        # pointer at that list. A borrowed chat GGUF repo (the native Qwen-Image text encoder) is a
        # curated companion id but holds a denoiser, so the orphan endpoint skips it; advertising
        # it here sent the user to Free up space to remove a row that is never there. Per copy,
        # like the listing: one pipeline copy in another cache root must not hide a companion-only
        # copy that Free up space really will offer.
        elif base_key in offerable and any(not _repo_holds_denoiser(r) for r in base_repos):
            freeable.append(entry)
        # Else: a base only a recorded link names. It is unheld, but the orphan endpoint is
        # table-only by design (a mis-recorded link must never turn an unrelated repo into a
        # delete candidate), so advertising it here pointed the user at a Free up space list it
        # will never appear in.

    return {
        "repo_id": repo_id,
        "variant": variant,
        "reclaimed_bytes": reclaimed,
        "retained_companions": retained,
        "freeable_companions": freeable,
        # Same predicate the destructive path uses, so the dialog says what the delete will do.
        # A variant delete usually cannot strand anything, but the native Qwen-Image encoder is a
        # named quant inside a chat GGUF repo, and previewing only whole-repo deletes left Delete
        # enabled on it and the refusal to arrive after the user confirmed.
        "blocked_by": (
            companion_dependents(repo_id, scans, ignore_repo_ids = [repo_id])
            if companion_assets.is_companion_base(repo_id)
            and (variant is None or _variant_is_a_required_companion_asset(repo_id, variant))
            else []
        ),
    }


class _SingleRepoScan:
    """Adapter presenting a fixed repo list with the attribute the derivation reads."""

    def __init__(self, repos):
        self.repos = repos


async def delete_impact_response(
    repo_id: str,
    variant: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> dict:
    """What a delete of *repo_id* (/*variant*) would reclaim, retain, and be blocked by."""
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(status_code = 400, detail = "Invalid repo_id format")
    variant = (variant or "").strip() or None
    if variant is not None and not _is_valid_gguf_variant(variant):
        raise HTTPException(status_code = 400, detail = f"Invalid gguf_variant: {variant!r}")
    from hub.services.models import deletion

    cache_root = deletion._explicit_delete_cache_root(repo_id, cache_path)
    reservation_block = deletion.cache_reservation_delete_block(repo_id, variant)
    if reservation_block is not None:
        impact = await asyncio.to_thread(_delete_impact_blocking, repo_id, variant, cache_path)
        delete_block = reservation_block
    else:

        async def bounded_load_state():
            try:
                probe = _submit_delete_impact_load_state_probe(
                    deletion._load_state_delete_block,
                    repo_id,
                    variant,
                    cache_root,
                )
                if probe is None:
                    return 503, deletion._LOAD_STATE_UNVERIFIABLE_DETAIL
                wrapped_probe = asyncio.wrap_future(probe)
                wrapped_probe.add_done_callback(_retrieve_probe_exception)
                done, _ = await asyncio.wait(
                    (wrapped_probe,),
                    timeout = _DELETE_IMPACT_LOAD_STATE_TIMEOUT_S,
                )
                if not done:
                    _abandon_delete_impact_load_state_probe(
                        repo_id,
                        variant,
                        probe,
                        cache_root,
                    )
                    return 503, deletion._LOAD_STATE_UNVERIFIABLE_DETAIL
                return wrapped_probe.result()
            except Exception as exc:
                logger.warning(
                    f"Load-state verification failed for {repo_id}; refusing delete: {exc}"
                )
                return 503, deletion._LOAD_STATE_UNVERIFIABLE_DETAIL

        impact_result, delete_block_result = await asyncio.gather(
            asyncio.to_thread(_delete_impact_blocking, repo_id, variant, cache_path),
            bounded_load_state(),
            return_exceptions = True,
        )
        if isinstance(impact_result, BaseException):
            raise impact_result
        if isinstance(delete_block_result, BaseException):
            raise delete_block_result
        impact = impact_result
        delete_block = delete_block_result
        reservation_block = deletion.cache_reservation_delete_block(repo_id, variant)
    if reservation_block is not None and (delete_block is None or delete_block[0] == 503):
        delete_block = reservation_block
    impact["delete_block"] = (
        {
            "status_code": delete_block[0],
            "detail": delete_block[1],
            "retryable": delete_block[0] == deletion._DELETE_RETRY_LATER,
        }
        if delete_block
        else None
    )
    return impact


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
        # Per COPY, not pooled: the loop below emits one row per cache root because a delete is
        # scoped to one, so a full pipeline copy in one root must not suppress an orphaned
        # companion-only copy in another that nothing can otherwise reclaim.
        repos = [r for r in repos if not _repo_holds_denoiser(r)]
        if not repos:
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
