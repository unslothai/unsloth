# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cached model deletion."""

from __future__ import annotations

import asyncio
import errno
import os
import stat
from pathlib import Path
from typing import Callable, Optional

from fastapi import HTTPException
from loggers import get_logger

from hub.utils import download_manifest
from hub.utils import download_registry
from hub.utils import inventory_scan as hf_cache_scan
from hub.utils.gguf import (
    bare_quant_alias,
    extract_quant_token,
    gguf_variant_scopes_overlap,
    gguf_variant_key,
    is_qualified_gguf_variant_key,
    quant_token_with_bpw,
    remove_appledouble_sidecar,
    is_reclaimable_drafter_path as _is_reclaimable_drafter_path,
)
from hub.utils.hf_cache_state import (
    AmbiguousDeleteTargetError,
    INCOMPLETE_SUFFIX,
    iter_active_repo_cache_dirs,
    iter_repo_cache_dirs,
    purge_partial_repo,
    purge_repo_cache_dirs,
    resolve_delete_target_root,
    scoped_delete_root,
)
from hub.utils.snapshot_reclaim import SnapshotRefsUnverifiable, referenced_snapshot_revisions
from hub.utils.paths import (
    is_redirect_stat,
    is_valid_gguf_variant as _is_valid_gguf_variant,
    is_valid_repo_id as _is_valid_repo_id,
    resolve_cached_repo_id_case,
)
from hub.services import resolve_destructive_repo_ids
from hub.services.models import cache_inventory, downloads, gguf_variants
from hub.services.models.common import (
    _is_gguf_filename,
    _is_imatrix_filename,
    _is_main_gguf_filename,
    _is_mmproj_filename,
)
from utils.model_cache_reservations import wait_for_reserved_worker
from utils.paths.path_utils import is_appledouble_metadata

logger = get_logger(__name__)


def _unresolved_variant_partial_detail(repo_id: str, variant: str) -> str:
    return (
        f"Couldn't fully delete {variant} for {repo_id}: partial download bytes "
        "exist but this variant's blob hashes are unavailable. Delete the entire "
        "cached model to remove its partial downloads offline, or reconnect/provide "
        "access and try deleting this variant again."
    )


class _CacheBlobReferencesUnverifiable(RuntimeError):
    pass


def _snapshot_entry_revision(snap: Path, repo_dir: Optional[Path]) -> Optional[str]:
    if repo_dir is None:
        return None
    try:
        repo = repo_dir.expanduser().resolve(strict = True)
        raw_snapshots = repo_dir.expanduser() / "snapshots"
        snapshots_stat = raw_snapshots.lstat()
        if is_redirect_stat(snapshots_stat) or not stat.S_ISDIR(snapshots_stat.st_mode):
            return None
        snapshots = raw_snapshots.resolve(strict = True)
        if snapshots != repo / "snapshots":
            return None
        relative_parent = snap.parent.resolve(strict = True).relative_to(snapshots)
        if not relative_parent.parts:
            return None
        revision = download_manifest.normalized_commit_hash(relative_parent.parts[0])
        if revision is None:
            return None
        revision_dir = snapshots / revision
        revision_stat = revision_dir.lstat()
        if is_redirect_stat(revision_stat) or not stat.S_ISDIR(revision_stat.st_mode):
            return None
        if revision_dir.resolve(strict = True) != revision_dir:
            return None
        return revision
    except (OSError, RuntimeError, ValueError):
        return None


def _snapshot_blob_reference_counts(repo_dir: Optional[Path]) -> dict[Path, int]:
    """Map each blob's realpath to its live snapshot symlink count, so per-variant deletion never unlinks a blob another revision still references (call after the target variant's own symlinks are removed)."""
    counts: dict[Path, int] = {}
    if repo_dir is None:
        raise _CacheBlobReferencesUnverifiable("cache repository location is unavailable")
    snapshots = repo_dir / "snapshots"
    try:
        try:
            snapshots_stat = snapshots.lstat()
        except FileNotFoundError:
            return counts
        if is_redirect_stat(snapshots_stat) or not stat.S_ISDIR(snapshots_stat.st_mode):
            raise _CacheBlobReferencesUnverifiable(
                "snapshots path is redirected or is not a directory"
            )

        def raise_walk_error(error: OSError) -> None:
            raise error

        def lstat_if_present(path: Path) -> Optional[os.stat_result]:
            for _attempt in range(2):
                try:
                    return path.lstat()
                except FileNotFoundError:
                    pass
            return None

        def readlink_if_present(path: Path) -> Optional[Path]:
            for _attempt in range(2):
                try:
                    return path.readlink()
                except FileNotFoundError:
                    pass
            return None

        for current, directories, files in os.walk(
            snapshots,
            topdown = True,
            onerror = raise_walk_error,
            followlinks = False,
        ):
            current_path = Path(current)
            for name in list(directories):
                directory_stat = lstat_if_present(current_path / name)
                if directory_stat is None:
                    directories.remove(name)
                    continue
                if is_redirect_stat(directory_stat) or not stat.S_ISDIR(directory_stat.st_mode):
                    raise _CacheBlobReferencesUnverifiable(
                        f"snapshot directory is redirected or is not a directory: {name}"
                    )
            for name in files:
                link = current_path / name
                link_stat = lstat_if_present(link)
                if link_stat is None or not stat.S_ISLNK(link_stat.st_mode):
                    continue
                raw_target = readlink_if_present(link)
                if raw_target is None:
                    continue
                candidate = raw_target if raw_target.is_absolute() else link.parent / raw_target
                target = candidate.resolve(strict = False)
                counts[target] = counts.get(target, 0) + 1
    except _CacheBlobReferencesUnverifiable:
        raise
    except (OSError, RuntimeError) as exc:
        raise _CacheBlobReferencesUnverifiable(
            f"snapshot scan failed ({type(exc).__name__}: {exc})"
        ) from exc
    return counts


def _blob_hash_from_path(blob: Path) -> Optional[str]:
    name = blob.name
    if not name or name.endswith(INCOMPLETE_SUFFIX):
        return None
    return name


def _path_exists_or_symlink(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return True


def _unlink_snapshot_entry(snap: Path) -> int:
    """Unlink one snapshot entry, plus any AppleDouble sidecar beside it.

    Returns the entries removed, which never counts the sidecar: it is metadata about a file the
    caller asked to remove, not a second file.
    """
    removed = 0
    if _path_exists_or_symlink(snap):
        snap.unlink()
        removed += 1
    remove_appledouble_sidecar(snap)
    return removed


def _blob_key(blob: Path) -> Path:
    try:
        return blob.resolve()
    except (OSError, RuntimeError):
        return blob


def _unlink_snapshot_matches(
    matches: list[tuple[Path, Optional[Path], str]],
) -> tuple[int, int, dict[Path, list[tuple[Path, Path, str]]], set[Path], list[str]]:
    removed = 0
    removed_bytes = 0
    removed_links: dict[Path, list[tuple[Path, Path, str]]] = {}
    protected_blobs: set[Path] = set()
    failures: list[str] = []
    for snap, blob, name in matches:
        blob_key = _blob_key(blob) if blob is not None else None
        link_target: Optional[Path] = None
        direct_size = 0
        try:
            entry_stat = snap.lstat()
            if stat.S_ISLNK(entry_stat.st_mode):
                if blob is None:
                    failures.append(f"{name}: cache link target is unavailable")
                    continue
                link_target = snap.readlink()
                candidate = link_target if link_target.is_absolute() else snap.parent / link_target
                if candidate.resolve(strict = False) != blob.resolve(strict = False):
                    failures.append(f"{name}: cache link target changed during delete")
                    if blob_key is not None:
                        protected_blobs.add(blob_key)
                    continue
            elif not is_redirect_stat(entry_stat) and stat.S_ISREG(entry_stat.st_mode):
                direct_size = entry_stat.st_size
            else:
                failures.append(f"{name}: cache entry is not a regular file or symlink")
                if blob_key is not None:
                    protected_blobs.add(blob_key)
                continue
        except FileNotFoundError:
            continue
        except (OSError, RuntimeError) as e:
            failures.append(f"{name}: couldn't inspect cache entry: {e}")
            if blob_key is not None:
                protected_blobs.add(blob_key)
            continue
        try:
            removed_now = _unlink_snapshot_entry(snap)
            removed += removed_now
            removed_bytes += direct_size if removed_now else 0
            if removed_now and blob is not None and link_target is not None:
                removed_links.setdefault(_blob_key(blob), []).append((snap, link_target, name))
        except OSError as e:
            failures.append(f"{name}: {e}")
            if blob_key is not None:
                protected_blobs.add(blob_key)
    return removed, removed_bytes, removed_links, protected_blobs, failures


def _restore_snapshot_links(links: list[tuple[Path, Path, str]]) -> tuple[int, list[str]]:
    restored = 0
    failures: list[str] = []
    for snap, target, name in links:
        if _path_exists_or_symlink(snap):
            restored += 1
            continue
        try:
            snap.symlink_to(target)
            restored += 1
        except OSError as e:
            failures.append(f"{name}: couldn't restore cache link after delete failed: {e}")
    return restored, failures


def _delete_unreferenced_match_blobs(
    matches: list[tuple[Path, Optional[Path], str]],
    repo_dir: Optional[Path],
    removed_links: dict[Path, list[tuple[Path, Path, str]]],
    protected_blobs: set[Path],
) -> tuple[int, int, int, list[str]]:
    reference_error: Optional[str] = None
    try:
        ref_counts = _snapshot_blob_reference_counts(repo_dir)
    except _CacheBlobReferencesUnverifiable as exc:
        ref_counts = None
        reference_error = str(exc)
    seen_blobs: set[Path] = set()
    deleted_blobs = 0
    deleted_bytes = 0
    restored_snapshots = 0
    failures: list[str] = []
    if ref_counts is None:
        for links in removed_links.values():
            restored, restore_failures = _restore_snapshot_links(links)
            restored_snapshots += restored
            failures.extend(restore_failures)
        has_remaining_blob = False
        for _snap, blob, _name in matches:
            if blob is None:
                continue
            try:
                blob.lstat()
                has_remaining_blob = True
                break
            except FileNotFoundError:
                continue
            except OSError:
                has_remaining_blob = True
                break
        if removed_links or has_remaining_blob:
            failures.insert(
                0,
                "cache blob references could not be verified: "
                f"{reference_error or 'snapshot scan failed'}. Repair the cache path or "
                "permissions, or delete the entire cached model.",
            )
        return 0, 0, restored_snapshots, failures

    for _snap, blob, name in matches:
        if blob is None:
            continue
        if not cache_inventory._is_real_cache_blob(blob, repo_dir):
            continue
        blob_key = _blob_key(blob)
        if blob_key in seen_blobs:
            continue
        seen_blobs.add(blob_key)
        if blob_key in protected_blobs:
            continue
        if ref_counts.get(blob_key, 0) > 0:
            continue
        try:
            blob_stat = blob.lstat()
            if is_redirect_stat(blob_stat) or not stat.S_ISREG(blob_stat.st_mode):
                raise OSError(errno.EINVAL, "cache blob is not a regular file", str(blob))
            size = blob_stat.st_size
            blob.unlink()
            deleted_bytes += size
            deleted_blobs += 1
        except FileNotFoundError:
            continue
        except OSError as e:
            failures.append(f"{name}: {e}")
            restored, restore_failures = _restore_snapshot_links(removed_links.get(blob_key, []))
            restored_snapshots += restored
            failures.extend(restore_failures)
    return deleted_blobs, deleted_bytes, restored_snapshots, failures


def _validated_keep_snapshot(
    keep_snapshot: Optional[str | Path], repo_dir: Optional[Path]
) -> Optional[Path]:
    if keep_snapshot is None or repo_dir is None:
        return None
    try:
        raw_repo = repo_dir.expanduser()
        repo_stat = raw_repo.lstat()
        if is_redirect_stat(repo_stat) or not stat.S_ISDIR(repo_stat.st_mode):
            return None
        repo = raw_repo.resolve(strict = True)
        raw_snapshots = raw_repo / "snapshots"
        snapshots_stat = raw_snapshots.lstat()
        if is_redirect_stat(snapshots_stat) or not stat.S_ISDIR(snapshots_stat.st_mode):
            return None
        snapshots = raw_snapshots.resolve(strict = True)
        if snapshots != repo / "snapshots":
            return None
        raw_candidate = Path(keep_snapshot).expanduser()
        candidate_stat = raw_candidate.lstat()
        if is_redirect_stat(candidate_stat) or not stat.S_ISDIR(candidate_stat.st_mode):
            return None
        candidate = raw_candidate.resolve(strict = True)
        if candidate != snapshots / raw_candidate.name:
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate


def _is_stale_snapshot_copy(
    snap: Path, blob: Optional[Path], repo_dir: Optional[Path], keep_snapshot: Optional[Path]
) -> bool:
    if blob is None or repo_dir is None or keep_snapshot is None:
        return False
    try:
        snap_stat = snap.lstat()
        if is_redirect_stat(snap_stat) or not stat.S_ISREG(snap_stat.st_mode):
            return False
        snapshots = (repo_dir.expanduser().resolve(strict = False) / "snapshots").resolve(
            strict = False
        )
        snap_path = snap.resolve(strict = False)
        blob_path = blob.resolve(strict = False)
        relative = snap_path.relative_to(snapshots)
        if len(relative.parts) < 2 or snap_path != blob_path:
            return False
        return not snap_path.is_relative_to(keep_snapshot)
    except (OSError, RuntimeError, ValueError):
        return False


def _snapshot_entry_is_in_keep_snapshot(snap: Path, keep_snapshot: Optional[Path]) -> bool:
    if keep_snapshot is None:
        return False
    try:
        candidate = snap.parent.resolve(strict = False) / snap.name
        return candidate.is_relative_to(keep_snapshot)
    except (OSError, RuntimeError, ValueError):
        return False


def _repo_file_matches(target_repo, predicate) -> list[tuple[Path, Optional[Path], str]]:
    """Files whose snapshot-relative path satisfies *predicate*.

    Relative, not the bare ``file_name``: huggingface_hub sets that to
    ``file_path.name`` (and our own recovery scan to ``entry.name``), so a
    companion in ``dspark/`` or ``MTP/`` arrived here indistinguishable from a
    root file. Every predicate below keys on the directory for at least one
    supported layout, and the quant labels they extract are unchanged by the
    prefix.
    """
    matches: list[tuple[Path, Optional[Path], str]] = []
    for rev in getattr(target_repo, "revisions", ()):
        snapshot = getattr(rev, "snapshot_path", None)
        for f in getattr(rev, "files", ()):
            name = str(getattr(f, "file_name", ""))
            file_path = getattr(f, "file_path", None)
            if snapshot and file_path:
                try:
                    name = Path(file_path).relative_to(Path(snapshot)).as_posix()
                except ValueError:
                    pass
            if not predicate(name):
                continue
            if not file_path:
                continue
            # Every predicate here keys on the name, which a sidecar answers exactly as its
            # neighbour does, so it would be counted as a deleted model in its own right.
            # Proven metadata only: anything else carrying this key is a file to delete.
            if is_appledouble_metadata(Path(file_path)):
                continue
            blob_path = getattr(f, "blob_path", None)
            matches.append(
                (
                    Path(file_path),
                    Path(blob_path) if blob_path else None,
                    name,
                )
            )
    return matches


def _has_remaining_main_gguf(target_repo) -> bool:
    return any(
        _path_exists_or_symlink(snap)
        for snap, _blob, _name in _repo_file_matches(
            target_repo,
            _is_main_gguf_filename,
        )
    )


def _remove_empty_variant_dirs(target_repos: list, variant: str) -> tuple[int, list[str]]:
    """Remove now-empty ``snapshots/<rev>/<quant>/`` folders for *variant* (the
    quant label names the folder); only empty dirs go, so siblings are safe.
    Returns (count removed, removal failures other than a concurrent refill)."""
    # A qualified key names its own folder; its quant token belongs to sibling checkpoints too,
    # so it must not reach for a <quant>/ dir it does not own. Qualified means a path
    # (``distilled/...-Q6_K``), an H3 root stem, or a bpw modifier (``IQ4_XS-3.53bpw``, whose
    # token-only ``IQ4_XS/`` folder is a different build's).
    qualified = (
        is_qualified_gguf_variant_key(variant)
        or (quant_token_with_bpw(variant) or "").lower() == variant.lower()
    )
    variant_key = (
        variant.lower() if qualified else (extract_quant_token(variant) or variant).lower()
    )
    removed = 0
    failures: list[str] = []
    for target_repo in target_repos:
        repo_path = getattr(target_repo, "repo_path", None)
        if not repo_path:
            continue
        snapshots = Path(repo_path) / "snapshots"
        if not snapshots.is_dir():
            continue
        try:
            snap_dirs = [s for s in snapshots.iterdir() if s.is_dir() and not s.is_symlink()]
        except OSError:
            continue
        for snap in snap_dirs:
            try:
                subs = list(snap.iterdir())
            except OSError:
                continue
            for sub in subs:
                try:
                    if sub.is_symlink() or not sub.is_dir():
                        continue
                    folder_quant = quant_token_with_bpw(sub.name)
                    matches = (
                        folder_quant is not None and folder_quant.lower() == variant_key
                    ) or sub.name.lower() == variant.lower()
                    if not matches or any(sub.iterdir()):
                        continue
                except OSError:
                    continue
                try:
                    sub.rmdir()
                    removed += 1
                except OSError as e:
                    # A concurrent download refilling the dir (ENOTEMPTY) is not a
                    # failure; a read-only cache or locked dir is, so surface it.
                    if e.errno != errno.ENOTEMPTY:
                        failures.append(f"{sub.name}: {e}")
    return removed, failures


def _remove_empty_snapshot_dirs(
    target_repos: list, *, preserve_refs: bool = False
) -> tuple[int, list[str]]:
    removed = 0
    failures: list[str] = []
    for target_repo in target_repos:
        repo_path = getattr(target_repo, "repo_path", None)
        if not repo_path:
            continue
        referenced_revisions: frozenset[str] = frozenset()
        if preserve_refs:
            try:
                referenced_revisions = referenced_snapshot_revisions(Path(repo_path))
            except SnapshotRefsUnverifiable as exc:
                failures.append(f"refs: {exc}")
                continue
        snapshots = Path(repo_path) / "snapshots"
        if not snapshots.is_dir():
            continue
        try:
            snap_dirs = [s for s in snapshots.iterdir() if s.is_dir() and not s.is_symlink()]
        except OSError:
            continue
        for snap in snap_dirs:
            if snap.name in referenced_revisions:
                continue
            try:
                snap.rmdir()
                removed += 1
            except OSError as e:
                if e.errno != errno.ENOTEMPTY:
                    failures.append(f"{snap.name}: {e}")
    return removed, failures


def _variant_keys_to_delete(target_repo, variant: str) -> set[str]:
    """The variant keys in *target_repo* that *variant* names, lowercased.

    Its own key, always. Plus the unambiguous bare-quant alias the download side already admits
    (``gguf_plan.plan_for_variant``): a repo filing its sole Q4_K_M under a shared container
    (``weights/model-Q4_K_M.gguf``) qualifies that key, because the key is a pure function of the
    path and cannot know the directory disambiguates nothing, so every stored pin and every
    explicit ``repo:Q4_K_M`` names it by quant alone. Admitting the alias for the download and not
    for the delete answered "not found" and left the weights on disk.

    Only when it is unambiguous, exactly as the download side decides it: a repo that really does
    hold several checkpoints at one quant gets no fallback, because there the bare name genuinely
    does not name one of them and deleting the wrong one is unrecoverable.
    """
    wanted = (variant or "").strip().lower()
    if not wanted or "/" in wanted:
        return {wanted}
    keys = {
        gguf_variant_key(name).lower()
        for _snap, _blob, name in _repo_file_matches(target_repo, _is_main_gguf_filename)
    }
    if wanted in keys:
        return {wanted}
    # PATH-qualified keys only, not is_qualified_gguf_variant_key: an H3 root stem's bare quant
    # names both partitions, so it must not delete either.
    aliased = {key for key in keys if "/" in key and bare_quant_alias(key).lower() == wanted}
    return aliased if len(aliased) == 1 else {wanted}


def _delete_gguf_variant_from_repos(
    repo_id: str,
    variant: str,
    target_repos: list,
    hf_token: Optional[str],
    *,
    root: Optional[Path] = None,
) -> dict:
    failures: list[str] = []
    removed_snapshots = 0
    deleted_bytes = 0
    deleted_blobs = 0
    completed_hashes: set[str] = set()
    companion_targets: list[tuple[object, Optional[Path]]] = []
    single_target = len(target_repos) == 1

    for target_repo in target_repos:
        repo_dir = Path(target_repo.repo_path) if getattr(target_repo, "repo_path", None) else None
        wanted_keys = _variant_keys_to_delete(target_repo, variant)
        matched = _repo_file_matches(
            target_repo,
            lambda name, keys = wanted_keys: _is_main_gguf_filename(name)
            and gguf_variant_key(name).lower() in keys,
        )
        variant_partial = (
            not matched
            and single_target
            and repo_dir is not None
            and hf_cache_scan.is_variant_partial(
                repo_id,
                variant,
                repo_cache_dir = repo_dir,
            )
        )

        removed, freed, removed_links, protected_blobs, unlink_failures = _unlink_snapshot_matches(
            matched
        )
        removed_snapshots += removed
        deleted_bytes += freed
        failures.extend(unlink_failures)
        # One refused entry leaves its siblings' links already gone, so the reference scan
        # would read their blobs as unreferenced and unlink the other shards of the variant we
        # just declined to delete. Restore and commit no blob: the delete stays all-or-nothing.
        if unlink_failures:
            for links in removed_links.values():
                restored, restore_failures = _restore_snapshot_links(links)
                removed_snapshots -= restored
                failures.extend(restore_failures)
            continue

        for _snap, blob, _name in matched:
            if blob is None:
                continue
            blob_hash = _blob_hash_from_path(blob)
            if blob_hash:
                completed_hashes.add(blob_hash)
        deleted, freed, restored, blob_failures = _delete_unreferenced_match_blobs(
            matched,
            repo_dir,
            removed_links,
            protected_blobs,
        )
        deleted_blobs += deleted
        deleted_bytes += freed
        removed_snapshots -= restored
        failures.extend(blob_failures)

        if (
            (matched or variant_partial)
            and not unlink_failures
            and not blob_failures
            and not _has_remaining_main_gguf(target_repo)
        ):
            companion_targets.append((target_repo, repo_dir))

    # A main entry that would not unlink, and a locked main blob, both restore their snapshot
    # links before this barrier, leaving every shared companion intact for the retry.
    if not failures:
        for target_repo, repo_dir in companion_targets:
            companion_matches = _repo_file_matches(
                target_repo,
                # Companions: mmproj and the drafters Studio downloads (MTP with
                # every variant, DSpark on opt-in). No main GGUF is left, so they
                # cannot be launched; reclaim them with the last variant. An imatrix
                # joins them: no longer offered as a variant, so a copy an older build
                # fetched as one would be unreachable from the UI.
                lambda name: _is_gguf_filename(name)
                and (
                    _is_mmproj_filename(name)
                    or _is_reclaimable_drafter_path(name)
                    or _is_imatrix_filename(name)
                ),
            )
            (
                removed,
                freed,
                companion_links,
                companion_protected,
                unlink_failures,
            ) = _unlink_snapshot_matches(companion_matches)
            removed_snapshots += removed
            deleted_bytes += freed
            failures.extend(unlink_failures)
            # Same barrier as the main phase: a refused entry means something still has the
            # set open, so committing the rest would unlink a live mmproj or drafter.
            if unlink_failures:
                for links in companion_links.values():
                    restored, restore_failures = _restore_snapshot_links(links)
                    removed_snapshots -= restored
                    failures.extend(restore_failures)
                continue
            for _snap, blob, _name in companion_matches:
                if blob is None:
                    continue
                blob_hash = _blob_hash_from_path(blob)
                if blob_hash:
                    completed_hashes.add(blob_hash)
            deleted, freed, restored, blob_failures = _delete_unreferenced_match_blobs(
                companion_matches,
                repo_dir,
                companion_links,
                companion_protected,
            )
            deleted_blobs += deleted
            deleted_bytes += freed
            removed_snapshots -= restored
            failures.extend(blob_failures)

    if failures:
        reference_failure = next(
            (
                failure
                for failure in failures
                if failure.startswith("cache blob references could not be verified:")
            ),
            None,
        )
        # "fully" says a delete got half done, which is now the uncommon case: the main phase
        # rolls its snapshot links back and commits no blob when an entry refuses to unlink, so
        # the ordinary in-use refusal leaves the variant exactly as it was. Read it off the
        # counters rather than the branch -- a rollback that itself failed, an unrestorable
        # direct copy, and a companion phase that ran after the main one committed are all
        # genuinely partial, and each of those leaves a count behind.
        committed = removed_snapshots > 0 or deleted_blobs > 0
        raise HTTPException(
            status_code = 409,
            detail = (
                f"Couldn't fully delete {variant} for {repo_id}: {reference_failure}"
                if reference_failure is not None
                else (
                    f"Couldn't {'fully ' if committed else ''}delete {variant} for {repo_id}: "
                    f"{len(failures)} file(s) are in use. "
                    f"{'' if committed else 'Nothing was removed. '}"
                    "Unload the model and try again."
                )
            ),
        )

    incomplete_result = gguf_variants.delete_variant_incomplete_blobs_result(
        repo_id,
        variant,
        hf_token,
        extra_hashes = frozenset(completed_hashes),
        companions = True,
        root = root,
    )
    if incomplete_result.unresolved:
        raise HTTPException(
            status_code = 409,
            detail = _unresolved_variant_partial_detail(repo_id, variant),
        )

    state_purged = download_manifest.purge_state("model", repo_id, variant, hub_cache = root)
    # Reclaim the empty quant folder so it stops 404ing on delete.
    removed_dirs, dir_failures = _remove_empty_variant_dirs(target_repos, variant)
    removed_snap_dirs, snap_dir_failures = _remove_empty_snapshot_dirs(target_repos)
    removed_dirs += removed_snap_dirs
    dir_failures.extend(snap_dir_failures)
    if dir_failures:
        raise HTTPException(
            status_code = 409,
            detail = (
                f"Couldn't fully delete {variant} for {repo_id}: "
                f"{len(dir_failures)} folder(s) could not be removed "
                "(read-only cache or in use). Try again."
            ),
        )
    if (
        removed_snapshots == 0
        and deleted_blobs == 0
        and incomplete_result.deleted == 0
        and not state_purged
        and removed_dirs == 0
    ):
        raise HTTPException(
            status_code = 404,
            detail = f"Variant {variant} not found in cache for {repo_id}",
        )

    freed_mb = deleted_bytes / (1024 * 1024)
    logger.info(
        f"Deleted {removed_snapshots} file(s) for {repo_id} variant {variant}: "
        f"{freed_mb:.1f} MB freed"
    )
    return {"status": "deleted", "repo_id": repo_id, "variant": variant}


def reclaim_replaced_gguf_variant(
    repo_id: str,
    variant: str,
    keep_main_hashes: frozenset[str],
    hf_token: Optional[str] = None,
    *,
    hub_cache: Optional[str | Path] = None,
    keep_snapshot: Optional[str | Path] = None,
) -> dict:
    """Prune stale main-GGUF files for a variant after a replacement verified.

    This is intentionally narrower than user-driven delete: it removes only
    same-variant main files outside the verified replacement, then unlinks
    their blobs only if no remaining snapshot references them.
    Shared companions and sibling variants are left intact.
    """
    if not keep_main_hashes:
        logger.info(
            "Skipping stale GGUF reclaim for %s [%s]: current main hashes unresolved",
            repo_id,
            variant,
        )
        return {
            "status": "skipped",
            "repo_id": repo_id,
            "variant": variant,
            "reason": "unresolved_hashes",
        }
    if not _is_valid_repo_id(repo_id) or not _is_valid_gguf_variant(variant):
        return {
            "status": "skipped",
            "repo_id": repo_id,
            "variant": variant,
            "reason": "invalid_target",
        }

    failures: list[str] = []
    removed_snapshots = 0
    deleted_blobs = 0
    deleted_bytes = 0
    variant_key = variant.lower()

    try:
        cache_scans = cache_inventory.all_hf_cache_scans()
    except Exception as e:
        logger.warning(
            "Skipping stale GGUF reclaim for %s [%s]: cache scan failed: %s",
            repo_id,
            variant,
            download_registry.scrub_secrets(str(e), hf_token = hf_token),
        )
        return {
            "status": "skipped",
            "repo_id": repo_id,
            "variant": variant,
            "reason": "scan_failed",
        }

    if hub_cache is None:
        from utils.hf_cache_settings import get_hf_cache_paths
        hub_cache = get_hf_cache_paths().hub_cache
    try:
        target_hub_cache = Path(hub_cache).expanduser().resolve(strict = False)
    except (OSError, RuntimeError, ValueError):
        target_hub_cache = Path(hub_cache).expanduser()

    candidate_repos = [
        repo_info
        for hf_cache in cache_scans
        for repo_info in hf_cache.repos
        if str(getattr(repo_info, "repo_type", "")) == "model"
        and str(getattr(repo_info, "repo_id", "")).lower() == repo_id.lower()
        and getattr(repo_info, "repo_path", None)
        and Path(repo_info.repo_path).parent.resolve(strict = False) == target_hub_cache
    ]
    try:
        matched_repo_ids = resolve_destructive_repo_ids(
            repo_id,
            [str(getattr(repo_info, "repo_id", "")) for repo_info in candidate_repos],
            noun = "models",
        )
    except HTTPException as e:
        detail = getattr(e, "detail", str(e))
        logger.warning(
            "Skipping stale GGUF reclaim for %s [%s]: %s",
            repo_id,
            variant,
            download_registry.scrub_secrets(str(detail), hf_token = hf_token),
        )
        return {
            "status": "skipped",
            "repo_id": repo_id,
            "variant": variant,
            "reason": "ambiguous_repo",
        }
    target_repos = [
        repo_info
        for repo_info in candidate_repos
        if str(getattr(repo_info, "repo_id", "")) in matched_repo_ids
    ]

    cleanup_repos: list = []
    for target_repo in target_repos:
        repo_dir = Path(target_repo.repo_path) if getattr(target_repo, "repo_path", None) else None
        current_snapshot = _validated_keep_snapshot(keep_snapshot, repo_dir)
        try:
            if repo_dir is None:
                raise SnapshotRefsUnverifiable("cache repository location is unavailable")
            referenced_revisions = referenced_snapshot_revisions(repo_dir)
        except SnapshotRefsUnverifiable as exc:
            failures.append(f"refs: {exc}")
            continue
        stale_matches: list[tuple[Path, Optional[Path], str]] = []
        matches = _repo_file_matches(
            target_repo,
            lambda name: _is_main_gguf_filename(name)
            and gguf_variant_key(name).lower() == variant_key,
        )
        entries_verifiable = True
        for snap, blob, name in matches:
            if _snapshot_entry_is_in_keep_snapshot(snap, current_snapshot):
                continue
            revision = _snapshot_entry_revision(snap, repo_dir)
            if revision is None:
                failures.append(f"{name}: snapshot revision could not be verified")
                entries_verifiable = False
                break
            if revision in referenced_revisions:
                continue
            if cache_inventory._is_real_cache_blob(blob, repo_dir):
                blob_hash = _blob_hash_from_path(blob) if blob is not None else None
                if blob_hash is not None and blob_hash not in keep_main_hashes:
                    stale_matches.append((snap, blob, name))
            elif _is_stale_snapshot_copy(snap, blob, repo_dir, current_snapshot):
                stale_matches.append((snap, blob, name))

        if not entries_verifiable:
            continue
        if not stale_matches:
            cleanup_repos.append(target_repo)
            continue
        try:
            referenced_revisions |= referenced_snapshot_revisions(repo_dir)
        except SnapshotRefsUnverifiable as exc:
            failures.append(f"refs: {exc}")
            continue
        filtered_matches: list[tuple[Path, Optional[Path], str]] = []
        for match in stale_matches:
            revision = _snapshot_entry_revision(match[0], repo_dir)
            if revision is None:
                failures.append(f"{match[2]}: snapshot revision could not be reverified")
                entries_verifiable = False
                break
            if revision not in referenced_revisions:
                filtered_matches.append(match)
        if not entries_verifiable:
            continue
        stale_matches = filtered_matches
        cleanup_repos.append(target_repo)
        if not stale_matches:
            continue

        removed, freed, removed_links, protected_blobs, unlink_failures = _unlink_snapshot_matches(
            stale_matches
        )
        removed_snapshots += removed
        deleted_bytes += freed
        failures.extend(unlink_failures)
        deleted, freed, restored, blob_failures = _delete_unreferenced_match_blobs(
            stale_matches,
            repo_dir,
            removed_links,
            protected_blobs,
        )
        deleted_blobs += deleted
        deleted_bytes += freed
        removed_snapshots -= restored
        failures.extend(blob_failures)

    removed_dirs = 0
    dir_failures: list[str] = []
    if cleanup_repos:
        verified_cleanup_repos: list = []
        for target_repo in cleanup_repos:
            try:
                referenced_snapshot_revisions(Path(target_repo.repo_path))
            except SnapshotRefsUnverifiable as exc:
                failures.append(f"refs: {exc}")
                continue
            verified_cleanup_repos.append(target_repo)
        removed_dirs, dir_failures = _remove_empty_variant_dirs(
            verified_cleanup_repos,
            variant,
        )
        removed_snap_dirs, snap_dir_failures = _remove_empty_snapshot_dirs(
            verified_cleanup_repos,
            preserve_refs = True,
        )
        removed_dirs += removed_snap_dirs
        dir_failures.extend(snap_dir_failures)
        failures.extend(dir_failures)

    if failures:
        logger.warning(
            "Stale GGUF reclaim for %s [%s] left %d failure(s): %s",
            repo_id,
            variant,
            len(failures),
            "; ".join(failures[:3]),
        )

    if removed_snapshots or deleted_blobs or removed_dirs:
        cache_inventory.invalidate_hf_cache_scans()
        logger.info(
            "Reclaimed stale GGUF %s [%s]: snapshots=%d blobs=%d dirs=%d freed=%.1f MB",
            repo_id,
            variant,
            removed_snapshots,
            deleted_blobs,
            removed_dirs,
            deleted_bytes / (1024 * 1024),
        )

    return {
        "status": "reclaimed",
        "repo_id": repo_id,
        "variant": variant,
        "removed_snapshots": removed_snapshots,
        "deleted_blobs": deleted_blobs,
        "removed_dirs": removed_dirs,
    }


def _loaded_id_matches_repo(
    loaded_id: str,
    repo_id: str,
    *,
    cache_root: Optional[str | Path] = None,
) -> bool:
    """Match a repo ID, or a path in *cache_root* (any known root when omitted)."""
    rid = repo_id.lower()
    lid = loaded_id.lower()
    if lid == rid or lid.startswith(f"{rid}/"):
        return True

    try:
        loaded_path = Path(loaded_id).expanduser().resolve(strict = False)
    except (OSError, RuntimeError, ValueError):
        if cache_root is not None:
            raise
        return False
    if cache_root is None:
        repo_dirs = iter_repo_cache_dirs("model", repo_id)
    else:
        scan_errors: list[Exception] = []
        repo_dirs = tuple(
            iter_active_repo_cache_dirs(
                "model",
                repo_id,
                root = Path(cache_root),
                scan_errors = scan_errors,
            )
        )
        if scan_errors:
            raise scan_errors[0]
    for repo_dir in repo_dirs:
        try:
            resolved_repo = repo_dir.resolve(strict = False)
            if loaded_path == resolved_repo or loaded_path.is_relative_to(resolved_repo):
                return True
        except (OSError, RuntimeError, ValueError):
            if cache_root is not None:
                raise
            continue
    return False


def _loaded_repo_variant_blocks_delete(
    loaded_id: str,
    repo_id: str,
    delete_variant: Optional[str],
    loaded_variant: Optional[str],
    *,
    cache_root: Optional[str | Path] = None,
) -> bool:
    if not _loaded_id_matches_repo(loaded_id, repo_id, cache_root = cache_root):
        return False
    if not delete_variant:
        return True
    if not loaded_variant:
        return True
    return gguf_variant_scopes_overlap(loaded_variant, delete_variant)


_LOAD_STATE_UNVERIFIABLE_DETAIL = (
    "Couldn't verify whether this model is still loaded for inference. "
    "Unload it if it is active, then try deleting again."
)
_LOAD_STATE_REWRITE_UNVERIFIABLE_DETAIL = (
    "Couldn't verify whether this model is still loaded for inference. "
    "Unload it if it is active, then try the download again."
)
_MODEL_ACTIVE_DELETE_DETAIL = "Unload the model before deleting"
_MODEL_LOADING_DELETE_DETAIL = "Cannot delete a model while it is loading"
_DeleteBlock = tuple[int, str]
# The status code separates the two kinds of refusal, and the delete preview polls on it: 409
# names a holder that releases on its own (a load, a download, another delete), 400 one the user
# has to clear first. Answering 400 for a transient holder leaves Delete greyed out until the
# dialog is reopened; answering 409 for a resident model polls a cache scan that never changes.
_DELETE_RETRY_LATER = 409
_DELETE_USER_MUST_ACT = 400


def _raise_load_state_delete_block(block: Optional[_DeleteBlock]) -> None:
    if block is None:
        return
    status_code, detail = block
    raise HTTPException(status_code = status_code, detail = detail)


def _llama_cpp_blocks_delete(
    repo_id: str,
    variant: Optional[str],
    *,
    cache_root: Optional[str | Path] = None,
) -> Optional[_DeleteBlock]:
    """Why the llama.cpp backend blocks deleting *repo_id* (/variant), if it does."""
    try:
        from routes.inference import get_llama_cpp_backend
        backend = get_llama_cpp_backend()
    except Exception as e:
        logger.debug(f"llama.cpp backend unavailable during delete guard for {repo_id}: {e}")
        return None
    loaded_id = backend.model_identifier
    if cache_root is not None:
        loaded_id = getattr(backend, "gguf_path", None) or loaded_id
    loaded_variant = getattr(backend, "hf_variant", None)
    if backend.is_active and not backend.is_loaded and loaded_id:
        if _loaded_repo_variant_blocks_delete(
            loaded_id,
            repo_id,
            variant,
            loaded_variant,
            cache_root = cache_root,
        ):
            return _DELETE_RETRY_LATER, _MODEL_LOADING_DELETE_DETAIL
    if backend.is_loaded and loaded_id:
        if _loaded_repo_variant_blocks_delete(
            loaded_id,
            repo_id,
            variant,
            loaded_variant,
            cache_root = cache_root,
        ):
            return _DELETE_USER_MUST_ACT, _MODEL_ACTIVE_DELETE_DETAIL
    return None


def _inference_backend_delete_block(
    repo_id: str, *, cache_root: Optional[str | Path] = None
) -> Optional[_DeleteBlock]:
    """Why the subprocess inference backend blocks deleting *repo_id*, if it does."""
    try:
        from core.inference.orchestrator import peek_inference_backend

        # Peek, never construct: building one just to learn nothing is loaded imports torch.
        backend = peek_inference_backend()
    except Exception as e:
        logger.debug(f"Inference backend unavailable during delete guard for {repo_id}: {e}")
        return None
    if backend is None:
        return None
    # active_model_name is published only after the subprocess reports a successful
    # load. Until then, both Transformers and MLX targets live exclusively in this
    # set; deleting one in that window can unlink weights under the loading worker.
    # Snapshot it before reading active_model_name so the loading -> active handoff
    # cannot land between the two reads and briefly make the model look unheld.
    loading_names = tuple(getattr(backend, "loading_models", ()) or ())
    active_name = backend.active_model_name
    if active_name and _loaded_id_matches_repo(active_name, repo_id, cache_root = cache_root):
        return _DELETE_USER_MUST_ACT, _MODEL_ACTIVE_DELETE_DETAIL
    if any(
        loading_name and _loaded_id_matches_repo(loading_name, repo_id, cache_root = cache_root)
        for loading_name in loading_names
    ):
        return _DELETE_RETRY_LATER, _MODEL_LOADING_DELETE_DETAIL
    return None


def _media_variant_exempt(status: dict, held_id: str, variant: Optional[str]) -> bool:
    """Whether *held_id* is the engine's own checkpoint at a quantization other than
    *variant*.

    A managed variant download or delete touches that quant's main GGUF files. A resident sibling
    quant is outside that scope, while companion repositories remain protected independently.
    Both sides are reduced to the bare quant token, which is what ``status()`` publishes.
    """
    if not variant:
        return False
    resident = str(status.get("gguf_variant") or "").strip()
    checkpoint = str(status.get("repo_id") or "").strip()
    if not resident or not checkpoint:
        return False
    if held_id.strip().lower() != checkpoint.lower():
        return False
    wanted = extract_quant_token(variant)
    return bool(wanted) and wanted.lower() != resident.lower()


def _diffusion_blocks_delete(
    repo_id: str,
    variant: Optional[str] = None,
    *,
    cache_root: Optional[str | Path] = None,
) -> Optional[_DeleteBlock]:
    """The block if the Images backend holds *repo_id*, else None.

    Queries the ACTIVE engine: on a native selection the diffusers singleton reports
    unloaded while sd-cli still generates from the cached GGUF. Same
    fail-open-on-acquire contract as :func:`_llama_cpp_blocks_delete`.
    """
    try:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        engine = get_active_diffusion_engine()
    except Exception as e:
        logger.debug(f"Diffusion engine unavailable during delete guard for {repo_id}: {e}")
        return None
    status = engine.status()
    if status.get("loaded") and status.get("repo_id"):
        held = str(status["repo_id"])
        if not _media_variant_exempt(status, held, variant) and _loaded_id_matches_repo(
            held, repo_id, cache_root = cache_root
        ):
            return _DELETE_USER_MUST_ACT, _MODEL_ACTIVE_DELETE_DETAIL
    # sd.cpp re-reads companion VAE / text-encoder files every generation and status().repo_id covers only the main GGUF, so refuse the companions too.
    for lid in getattr(engine, "loaded_repo_ids", tuple)():
        if _media_variant_exempt(status, str(lid), variant):
            continue
        if _loaded_id_matches_repo(str(lid), repo_id, cache_root = cache_root):
            return _DELETE_USER_MUST_ACT, _MODEL_ACTIVE_DELETE_DETAIL
    # A downloading repo still reports loaded=False, but deleting would pull blobs from under the in-flight fetch.
    for lid in getattr(engine, "loading_repo_ids", tuple)():
        if _loaded_id_matches_repo(str(lid), repo_id, cache_root = cache_root):
            return (
                _DELETE_RETRY_LATER,
                "An Images model load is using this repo; wait for it to finish",
            )
    return None


def _video_blocks_delete(
    repo_id: str,
    variant: Optional[str] = None,
    *,
    cache_root: Optional[str | Path] = None,
) -> Optional[_DeleteBlock]:
    """The block if the Video backend holds or is fetching *repo_id*, else None.

    Video repos share the On Device delete action, so a live Wan / LTX / Hunyuan
    pipeline could otherwise lose its snapshot. Mirrors :func:`_diffusion_blocks_delete`.
    """
    try:
        from core.inference.video import get_video_backend
        backend = get_video_backend()
    except Exception as e:
        logger.debug(f"Video backend unavailable during delete guard for {repo_id}: {e}")
        return None
    status = backend.status()
    wanted = (variant or "").strip().replace("\\", "/").lower() or None
    for held_id, held_variant in getattr(backend, "loaded_gguf_dependency_scopes", tuple)():
        held_key = str(held_variant).strip().replace("\\", "/").lower()
        if _loaded_id_matches_repo(str(held_id), repo_id, cache_root = cache_root) and (
            wanted is None or wanted == held_key
        ):
            return _DELETE_USER_MUST_ACT, _MODEL_ACTIVE_DELETE_DETAIL
    if status.get("loaded"):
        # Diffusers GGUF/single-file loads read base_repo; native sd.cpp reports it only as metadata.
        keys = ("repo_id",) if status.get("engine") == "sd_cpp" else ("repo_id", "base_repo")
        for key in keys:
            held = status.get(key)
            if not held or _media_variant_exempt(status, str(held), variant):
                continue
            if _loaded_id_matches_repo(str(held), repo_id, cache_root = cache_root):
                return _DELETE_USER_MUST_ACT, _MODEL_ACTIVE_DELETE_DETAIL
    # Refuse the additional repositories a native runtime re-reads between generations.
    for lid in getattr(backend, "loaded_repo_ids", tuple)():
        if _media_variant_exempt(status, str(lid), variant):
            continue
        if _loaded_id_matches_repo(str(lid), repo_id, cache_root = cache_root):
            return _DELETE_USER_MUST_ACT, _MODEL_ACTIVE_DELETE_DETAIL
    for lid in getattr(backend, "loading_repo_ids", tuple)():
        if _loaded_id_matches_repo(str(lid), repo_id, cache_root = cache_root):
            return (
                _DELETE_RETRY_LATER,
                "A Video model load is using this repo; wait for it to finish",
            )
    return None


def _load_state_delete_block(
    repo_id: str,
    variant: Optional[str],
    *,
    cache_root: Optional[str | Path] = None,
) -> Optional[_DeleteBlock]:
    cache_scope = {} if cache_root is None else {"cache_root": cache_root}
    if llama_cpp_block := _llama_cpp_blocks_delete(repo_id, variant, **cache_scope):
        return llama_cpp_block
    inference_block = _inference_backend_delete_block(repo_id, **cache_scope)
    if inference_block:
        return inference_block
    return _diffusion_blocks_delete(repo_id, variant, **cache_scope) or _video_blocks_delete(
        repo_id, variant, **cache_scope
    )


async def load_state_delete_block(
    repo_id: str,
    variant: Optional[str],
    *,
    cache_root: Optional[str | Path] = None,
) -> Optional[_DeleteBlock]:
    try:
        cache_scope = {} if cache_root is None else {"cache_root": cache_root}
        return await asyncio.to_thread(
            _load_state_delete_block,
            repo_id,
            variant,
            **cache_scope,
        )
    except Exception as exc:
        logger.warning(f"Load-state verification failed for {repo_id}; refusing delete: {exc}")
        return 503, _LOAD_STATE_UNVERIFIABLE_DETAIL


def _explicit_delete_cache_root(repo_id: str, cache_path: Optional[str]) -> Optional[Path]:
    if not cache_path:
        return None
    cache_root = scoped_delete_root("model", repo_id, cache_path)
    if cache_root is None:
        raise HTTPException(status_code = 400, detail = "Invalid cache_path")
    return cache_root


def load_state_rewrite_block_now(
    repo_id: str,
    variant: Optional[str],
    *,
    fail_closed: bool | Callable[[], bool],
    cache_root: str | Path,
) -> Optional[_DeleteBlock]:
    """Why a download cannot rewrite *repo_id*/*variant*, or None.

    A managed variant download replaces the revision it supersedes, and reclaims exactly the
    main GGUF files carrying that quant label -- so the question is per-quant of every backend
    that reads one quantization, not just chat: a resident sibling quant stays downloadable.
    Backends with no quant scope still answer whole-repo -- a safetensors or MLX load, and the
    media backends' companion repos, whose files no variant scope names.
    """
    try:
        return _load_state_delete_block(repo_id, variant, cache_root = cache_root)
    except Exception as exc:
        try:
            refuse = fail_closed() if callable(fail_closed) else fail_closed
        except Exception as cache_exc:
            logger.warning(
                "Cache-state verification failed for %s while load-state verification "
                "was unavailable; refusing rewrite: %s",
                repo_id,
                cache_exc,
            )
            refuse = True
        outcome = "refusing" if refuse else "allowing"
        logger.warning(f"Load-state verification failed for {repo_id}; {outcome} download: {exc}")
        if refuse:
            return 503, _LOAD_STATE_REWRITE_UNVERIFIABLE_DETAIL
        return None


async def load_state_rewrite_block(
    repo_id: str,
    variant: Optional[str],
    *,
    fail_closed: bool | Callable[[], bool],
    cache_root: str | Path,
) -> Optional[_DeleteBlock]:
    return await asyncio.to_thread(
        load_state_rewrite_block_now,
        repo_id,
        variant,
        fail_closed = fail_closed,
        cache_root = cache_root,
    )


def _cache_conflict_delete_detail(variant: Optional[str], reason: Optional[str]) -> str:
    """Why the cache scope is unavailable, in the caller's terms. *reason* comes from the
    reservation registry; naming the wrong holder sends the user to cancel a download that
    is not there, and naming the wrong scope sends them to stop a load of a quantization
    nothing is loading."""
    scope = "this model's cache files" if variant is not None else "this model"
    if reason == "deleting":
        return f"A delete of {scope} is already running. Wait for it to finish."
    if reason == "inference_loading":
        # Repo-scoped on purpose: delete admission answers whole-repo for a held repository,
        # so the holder may well be a sibling quantization of the one being deleted.
        return "A model load is reading this model's files. Wait for it to finish, then delete."
    if reason == "repository_owned":
        return (
            "A dictation model download is writing this model's files. "
            "Wait for it to finish, then delete."
        )
    if reason == "downloading":
        return f"A download is writing {scope}. Cancel it (or wait for it), then delete."
    return f"Another operation is using {scope}. Wait for it to finish, then delete."


def cache_reservation_delete_block(repo_id: str, variant: Optional[str]) -> Optional[_DeleteBlock]:
    reason = downloads.registry.delete_admission_conflict(repo_id)
    if reason is None:
        return None
    return _DELETE_RETRY_LATER, _cache_conflict_delete_detail(variant, reason)


def _is_companion_base_repo(repo_id: str) -> bool:
    """Whether *repo_id* is a curated image-family companion base (pure table lookup, no I/O)."""
    try:
        from hub.utils import companion_assets
        return companion_assets.is_companion_base(repo_id)
    except Exception as e:  # noqa: BLE001 -- an unavailable table just skips the extra guard
        logger.debug(f"Companion base classification unavailable for {repo_id}: {e}")
        return False


def _variant_is_a_required_companion_asset(repo_id: str, variant: str) -> bool:
    """Whether *variant* names a file an installed checkpoint's native load actually opens.

    Not "would this empty the repo": the asset is a FIXED filename, so a sibling quant left
    behind substitutes for nothing. Fails CLOSED, and cheaply -- a True here only runs the
    dependants check, which answers "nobody needs it" for every ordinary repo and lets the
    delete through.
    """
    from hub.services.models import cache_inventory
    from hub.utils import companion_assets
    from hub.utils.gguf import extract_quant_label

    try:
        wanted = companion_assets.required_companion_asset_files(
            cache_inventory.all_hf_cache_scans()
        ).get((repo_id or "").strip().lower(), set())
        target = (variant or "").strip().lower()
        return any(extract_quant_label(name).lower() == target for name in wanted)
    except Exception as exc:  # noqa: BLE001 -- an unreadable cache is not permission to delete
        logger.warning(f"Could not check companion assets for {repo_id}: {exc}")
        return True


def _companion_share_blocks_delete(repo_id: str) -> Optional[str]:
    """The 400 detail when installed models still need *repo_id*'s shared assets, else None."""
    from hub.services.models import companion_cleanup

    holders = companion_cleanup.companion_dependents(repo_id, ignore_repo_ids = [repo_id])
    if not holders:
        return None
    shown = ", ".join(holders[:3])
    extra = len(holders) - 3
    if extra > 0:
        shown = f"{shown} and {extra} more"
    return (
        f"{repo_id} holds the text encoder, VAE and tokenizer that {shown} still "
        "needs. Delete those models first, then remove these shared assets."
    )


async def delete_cached_model_response(
    repo_id: str,
    variant: Optional[str] = None,
    hf_token: Optional[str] = None,
    cache_path: Optional[str] = None,
    only_if_orphan: bool = False,
):
    """Delete a cached model repo (or a specific GGUF variant) from the HF cache.

    When *variant* is provided, only the GGUF files matching that quant label
    are removed (e.g. ``UD-Q4_K_XL``).  Otherwise the entire repo is deleted.
    Refuses when the requested cache scope is used by a model loading or loaded for inference.

    *only_if_orphan* is Free up space's precondition: 409 rather than delete when the repo has
    become an installed checkpoint since the list the caller is acting on was built.
    """
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(status_code = 400, detail = "Invalid repo_id format")
    variant = (variant or "").strip() or None
    if variant is not None and not _is_valid_gguf_variant(variant):
        raise HTTPException(
            status_code = 400,
            detail = f"Invalid gguf_variant: {variant!r}",
        )

    cache_root = _explicit_delete_cache_root(repo_id, cache_path)
    blocks_detail = await load_state_delete_block(
        repo_id,
        variant,
        cache_root = cache_root,
    )
    _raise_load_state_delete_block(blocks_detail)

    repo_key = await asyncio.to_thread(resolve_cached_repo_id_case, repo_id, repo_type = "model")
    conflict = downloads.registry.begin_delete(repo_key)
    if conflict is not None:
        # Logged, not just returned: a reservation outlives its UI marker (a load worker owns it
        # until it returns), so a wedged worker shows up here as a repo that never becomes
        # deletable and nowhere else.
        logger.info("Delete of %s [%s] refused: cache scope held (%s)", repo_key, variant, conflict)
        raise HTTPException(
            status_code = _DELETE_RETRY_LATER,
            detail = _cache_conflict_delete_detail(variant, conflict),
        )
    try:
        blocks_detail = await load_state_delete_block(
            repo_id,
            variant,
            cache_root = cache_root,
        )
        _raise_load_state_delete_block(blocks_detail)
        # Shielded: a client disconnect must not run the finally below -- ending the delete
        # reservation -- while the worker is still unlinking blobs.
        return await wait_for_reserved_worker(
            asyncio.to_thread(
                _delete_cached_model_blocking,
                repo_id,
                variant,
                hf_token,
                cache_path,
                only_if_orphan = only_if_orphan,
            )
        )
    finally:
        downloads.registry.end_delete(repo_key)
        cache_inventory.invalidate_hf_cache_scans()


def _delete_cached_model_blocking(
    repo_id: str,
    variant: Optional[str],
    hf_token: Optional[str],
    cache_path: Optional[str] = None,
    *,
    only_if_orphan: bool = False,
) -> dict:
    # Free up space asks for this: the row it is removing was an orphan when the list was built,
    # and the list can be minutes old. A download of that same repo finishing in the background
    # turns it into an installed checkpoint, and neither guard below catches that -- begin_delete
    # only refuses a download still in flight, and the companion guard deliberately ignores the
    # target as its own dependent. Re-derived here, after begin_delete has closed the repo to new
    # downloads, so the answer cannot go stale between the check and the unlink.
    if only_if_orphan:
        from hub.services.models import companion_cleanup
        from hub.utils import companion_assets

        try:
            copies = companion_cleanup._repos_by_id(cache_inventory.all_hf_cache_scans()).get(
                repo_id.strip().lower(), []
            )
            # Only the copy being removed. The orphan listing emits one row per cache root
            # precisely because a delete is scoped to one, so a full-pipeline copy in another
            # remembered cache must not veto removing the companion-only copy that was listed.
            if cache_path:
                wanted = Path(cache_path)
                copies = [
                    r
                    for r in copies
                    if getattr(r, "repo_path", None) and Path(getattr(r, "repo_path")) == wanted
                ]
                if not copies:
                    # No fallback to the other copies. An empty match means the target root is
                    # not in this scan (its scan failed, or the copy is gone), and the delete
                    # below can still purge that directory by path -- so concluding "orphan" from
                    # copies we did not look at is exactly the fail-open this precondition exists
                    # to prevent. Raising here lands in the fail-closed 503 below.
                    raise RuntimeError(f"cache root not present in the scan: {cache_path}")
            still_orphan = not any(companion_assets.repo_holds_denoiser(repo) for repo in copies)
        except Exception as e:
            logger.warning(f"Orphan re-check failed for {repo_id}; refusing delete: {e}")
            raise HTTPException(
                status_code = 503,
                detail = ("Couldn't confirm these assets are still unused. Try again in a moment."),
            )
        if not still_orphan:
            raise HTTPException(
                status_code = 409,
                detail = (
                    f"{repo_id} now holds an installed model, so it is no longer an unused "
                    "asset. Reopen Free up space to see the current list."
                ),
            )

    # A companion base repo carries the text encoders, VAE and tokenizer for every quant of its
    # family, so removing it while one is installed leaves that quant unloadable with nothing on
    # screen to say why. Derived from what is installed right now, never from a stored count, and
    # only for a WHOLE-repo delete: a variant delete cannot touch another repo. Deleting the
    # dependants first makes the base an orphan, which Free up space then offers.
    #
    # Here rather than in the async caller so it shares this function's cache walk and its stubs:
    # the check IS part of the destructive stage, and a caller that replaces that stage should not
    # end up with half of it still running.
    # A variant delete normally cannot touch another repo, so the guard is a whole-repo check.
    # The exception is a companion whose asset IS a named GGUF variant: native Qwen-Image opens
    # exactly Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf inside a chat GGUF repo, so removing that one
    # quant strands the image checkpoint however many siblings remain, and none of them is a
    # substitute for a fixed filename. A FLAG, never a rewrite of `variant`: that name is the
    # destructive scope, and widening it here would delete every revision and manifest the user
    # did not ask for, and purge a sibling quant out from under an in-flight download.
    guard_this_delete = variant is None or _variant_is_a_required_companion_asset(repo_id, variant)
    if guard_this_delete and _is_companion_base_repo(repo_id):
        # Fails CLOSED, and only here: the lookup above already established this repo IS a
        # companion base, so an unreadable cache means the dependants cannot be enumerated, not
        # that there are none. Every other repo skips the check entirely and is unaffected.
        try:
            shared_detail = _companion_share_blocks_delete(repo_id)
        except Exception as e:
            logger.warning(f"Companion dependency check failed for {repo_id}; refusing delete: {e}")
            raise HTTPException(
                status_code = 503,
                detail = (
                    "Couldn't check whether other installed models still need these shared "
                    "assets. Try again in a moment."
                ),
            )
        if shared_detail:
            raise HTTPException(status_code = 400, detail = shared_detail)

    try:
        cache_scans = cache_inventory.all_hf_cache_scans()

        # A repo can live in several remembered caches. Group its copies by the
        # cache root that owns each, then target exactly one cache so a delete
        # never removes copies in other, previously selected caches.
        owners: dict = {}
        for hf_cache in cache_scans:
            for repo_info in hf_cache.repos:
                if str(repo_info.repo_type) != "model":
                    continue
                if repo_info.repo_id.lower() != repo_id.lower():
                    continue
                try:
                    owner = Path(repo_info.repo_path).parent.resolve(strict = False)
                except (OSError, RuntimeError, ValueError):
                    continue
                owners.setdefault(owner, []).append((hf_cache, repo_info))

        try:
            target_root = resolve_delete_target_root("model", repo_id, cache_path, owners.keys())
        except AmbiguousDeleteTargetError as exc:
            raise HTTPException(status_code = 409, detail = exc.detail) from exc
        if target_root is None:
            raise HTTPException(status_code = 400, detail = "Invalid cache_path")
        candidate_entries = owners.get(target_root, [])

        matched_repo_ids = resolve_destructive_repo_ids(
            repo_id,
            [str(repo_info.repo_id) for _hf_cache, repo_info in candidate_entries],
            noun = "models",
        )
        target_entries = [
            (hf_cache, repo_info)
            for hf_cache, repo_info in candidate_entries
            if str(repo_info.repo_id) in matched_repo_ids
        ]

        if not target_entries:
            if variant is None:
                cache_purged = purge_repo_cache_dirs(
                    "model", repo_id, root = target_root
                ) or purge_partial_repo("model", repo_id, root = target_root)
                state_purged = (
                    download_manifest.purge_all_state_for_repo(
                        "model", repo_id, hub_cache = target_root
                    )
                    > 0
                )
                if cache_purged or state_purged:
                    return {"status": "deleted", "repo_id": repo_id}
            if variant:
                incomplete_result = gguf_variants.delete_variant_incomplete_blobs_result(
                    repo_id,
                    variant,
                    hf_token,
                    companions = True,
                    root = target_root,
                )
                if incomplete_result.unresolved:
                    raise HTTPException(
                        status_code = 409,
                        detail = _unresolved_variant_partial_detail(repo_id, variant),
                    )
                state_purged = download_manifest.purge_state(
                    "model",
                    repo_id,
                    variant,
                    hub_cache = target_root,
                )
                if incomplete_result.deleted > 0 or state_purged:
                    return {
                        "status": "deleted",
                        "repo_id": repo_id,
                        "variant": variant,
                    }
            raise HTTPException(status_code = 404, detail = "Model not found in cache")

        if variant:
            return _delete_gguf_variant_from_repos(
                repo_id,
                variant,
                [repo for _cache, repo in target_entries],
                hf_token,
                root = target_root,
            )

        deleted_revisions = False
        for hf_cache, repo_info in target_entries:
            revision_hashes = [
                rev.commit_hash for rev in repo_info.revisions if getattr(rev, "commit_hash", None)
            ]
            if not revision_hashes:
                continue
            delete_strategy = hf_cache.delete_revisions(*revision_hashes)
            logger.info(
                f"Deleting cached model {repo_id} from "
                f"{getattr(hf_cache, 'cache_dir', '<unknown>')}: "
                f"{delete_strategy.expected_freed_size_str} will be freed"
            )
            delete_strategy.execute()
            deleted_revisions = True

        cache_purged = purge_repo_cache_dirs("model", repo_id, root = target_root)
        partial_purged = purge_partial_repo("model", repo_id, root = target_root)
        state_purged = (
            download_manifest.purge_all_state_for_repo("model", repo_id, hub_cache = target_root) > 0
        )

        if not (deleted_revisions or cache_purged or partial_purged or state_purged):
            raise HTTPException(status_code = 404, detail = "No revisions found for model")

        return {"status": "deleted", "repo_id": repo_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error deleting cached model %s: %s",
            repo_id,
            download_registry.scrub_secrets(str(e), hf_token = hf_token),
        )
        raise HTTPException(
            status_code = 500,
            detail = "Failed to delete cached model: "
            + download_registry.scrub_secrets(str(e), hf_token = hf_token),
        )
