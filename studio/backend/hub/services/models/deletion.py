# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cached model deletion."""

from __future__ import annotations

import asyncio
import errno
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from loggers import get_logger

from hub.utils import download_manifest
from hub.utils import download_registry
from hub.utils import inventory_scan as hf_cache_scan
from hub.utils.gguf import (
    bare_quant_alias,
    extract_quant_token,
    gguf_variant_key,
    is_qualified_gguf_variant_key,
    quant_token_with_bpw,
    remove_appledouble_sidecar,
    is_reclaimable_drafter_path as _is_reclaimable_drafter_path,
)
from hub.utils.hf_cache_state import (
    INCOMPLETE_SUFFIX,
    iter_repo_cache_dirs,
    purge_partial_repo,
    purge_repo_cache_dirs,
    resolve_delete_target_root,
)
from hub.utils.paths import (
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
from utils.paths.path_utils import is_appledouble_metadata

logger = get_logger(__name__)


def _snapshot_blob_reference_counts(repo_dir: Optional[Path]) -> dict[Path, int]:
    """Map each blob's realpath to its live snapshot symlink count, so per-variant deletion never unlinks a blob another revision still references (call after the target variant's own symlinks are removed)."""
    counts: dict[Path, int] = {}
    if repo_dir is None:
        return counts
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return counts
    try:
        entries = list(snapshots.rglob("*"))
    except OSError:
        return counts
    for link in entries:
        try:
            if not link.is_symlink():
                continue
            target = link.resolve()
        except OSError:
            continue
        counts[target] = counts.get(target, 0) + 1
    return counts


def _blob_hash_from_path(blob: Path) -> Optional[str]:
    name = blob.name
    if not name or name.endswith(INCOMPLETE_SUFFIX):
        return None
    return name


def _path_exists_or_symlink(path: Path) -> bool:
    try:
        return path.is_symlink() or path.exists()
    except OSError:
        return False


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
            # Every predicate here keys on the name, which a sidecar answers exactly as its neighbour does.
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
    # A qualified key names its own folder and must not reach for a <quant>/ dir it does not own;
    # qualified means a path (distilled/...-Q6_K), an H3 root stem, or a bpw modifier
    # (IQ4_XS-3.53bpw, whose token-only IQ4_XS/ folder is a different build's).
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
                    # A concurrent download refilling the dir (ENOTEMPTY) is not a failure; a read-only cache or locked
                    # dir is, so surface it.
                    if e.errno != errno.ENOTEMPTY:
                        failures.append(f"{sub.name}: {e}")
    return removed, failures


def _remove_empty_snapshot_dirs(target_repos: list) -> tuple[int, list[str]]:
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
    # PATH-qualified keys only, not is_qualified_gguf_variant_key: an H3 root stem's bare quant names
    # both partitions, so it must not delete either.
    aliased = {key for key in keys if "/" in key and bare_quant_alias(key).lower() == wanted}
    return aliased if len(aliased) == 1 else {wanted}


def _delete_gguf_variant_from_repos(
    repo_id: str,
    variant: str,
    target_repos: list,
    hf_token: Optional[str],
    *,
    sibling_active: bool = False,
    root: Optional[Path] = None,
) -> dict:
    failures: list[str] = []
    removed_snapshots = 0
    deleted_bytes = 0
    deleted_blobs = 0
    completed_hashes: set[str] = set()

    for target_repo in target_repos:
        repo_dir = Path(target_repo.repo_path) if getattr(target_repo, "repo_path", None) else None
        wanted_keys = _variant_keys_to_delete(target_repo, variant)
        matched = _repo_file_matches(
            target_repo,
            lambda name, keys = wanted_keys: _is_main_gguf_filename(name)
            and gguf_variant_key(name).lower() in keys,
        )

        for snap, _blob, name in matched:
            try:
                removed_snapshots += _unlink_snapshot_entry(snap)
            except OSError as e:
                failures.append(f"{name}: {e}")

        companion_matches: list[tuple[Path, Optional[Path], str]] = []
        if matched and not sibling_active and not _has_remaining_main_gguf(target_repo):
            companion_matches = _repo_file_matches(
                target_repo,
                # No main GGUF is left, so the companions cannot be launched; reclaim them with the last variant.
                lambda name: _is_gguf_filename(name)
                and (
                    _is_mmproj_filename(name)
                    or _is_reclaimable_drafter_path(name)
                    or _is_imatrix_filename(name)
                ),
            )
            for snap, _blob, name in companion_matches:
                try:
                    removed_snapshots += _unlink_snapshot_entry(snap)
                except OSError as e:
                    failures.append(f"{name}: {e}")

        ref_counts = _snapshot_blob_reference_counts(repo_dir)
        seen_blobs: set[Path] = set()
        for _snap, blob, name in [*matched, *companion_matches]:
            if blob is None:
                continue
            blob_hash = _blob_hash_from_path(blob)
            if blob_hash:
                completed_hashes.add(blob_hash)
            try:
                blob_key = blob.resolve()
            except OSError:
                blob_key = blob
            if blob_key in seen_blobs:
                continue
            seen_blobs.add(blob_key)
            if ref_counts.get(blob_key, 0) > 0:
                continue
            try:
                if blob.exists():
                    deleted_bytes += blob.stat().st_size
                    blob.unlink()
                    deleted_blobs += 1
            except OSError as e:
                failures.append(f"{name}: {e}")

    if failures:
        raise HTTPException(
            status_code = 409,
            detail = (
                f"Couldn't fully delete {variant} for {repo_id}: "
                f"{len(failures)} file(s) are in use. "
                "Unload the model and try again."
            ),
        )

    incomplete_result = gguf_variants.delete_variant_incomplete_blobs_result(
        repo_id,
        variant,
        hf_token,
        extra_hashes = frozenset(completed_hashes),
        companions = not sibling_active,
        root = root,
    )
    if incomplete_result.unresolved:
        raise HTTPException(
            status_code = 409,
            detail = (
                f"Couldn't fully delete {variant} for {repo_id}: partial "
                "download bytes exist but this variant's blob hashes are unavailable. "
                "Reconnect or provide access to the repo, then try again."
            ),
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
) -> dict:
    """Prune stale main-GGUF files for a variant after a replacement verified.

    This is intentionally narrower than user-driven delete: it removes only
    same-variant main files whose local blob hash is not in *keep_main_hashes*,
    then unlinks their blobs only if no remaining snapshot references them.
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

    for target_repo in target_repos:
        repo_dir = Path(target_repo.repo_path) if getattr(target_repo, "repo_path", None) else None
        stale_matches: list[tuple[Path, Optional[Path], str]] = []
        matches = _repo_file_matches(
            target_repo,
            lambda name: _is_main_gguf_filename(name)
            and gguf_variant_key(name).lower() == variant_key,
        )
        for snap, blob, name in matches:
            # Prune only a file we can identify as a real, stale cache blob: a no-symlink snapshot file has no
            # identifiable blob hash, so keep it.
            blob_hash = (
                _blob_hash_from_path(blob)
                if cache_inventory._is_real_cache_blob(blob, repo_dir)
                else None
            )
            if blob_hash is None or blob_hash in keep_main_hashes:
                continue
            stale_matches.append((snap, blob, name))

        if not stale_matches:
            continue

        for snap, _blob, name in stale_matches:
            try:
                removed_snapshots += _unlink_snapshot_entry(snap)
            except OSError as e:
                failures.append(f"{name}: {e}")

        ref_counts = _snapshot_blob_reference_counts(repo_dir)
        seen_blobs: set[Path] = set()
        for _snap, blob, name in stale_matches:
            if blob is None:
                continue
            try:
                blob_key = blob.resolve()
            except OSError:
                blob_key = blob
            if blob_key in seen_blobs:
                continue
            seen_blobs.add(blob_key)
            if ref_counts.get(blob_key, 0) > 0:
                continue
            try:
                if blob.exists():
                    deleted_bytes += blob.stat().st_size
                    blob.unlink()
                    deleted_blobs += 1
            except OSError as e:
                failures.append(f"{name}: {e}")

    removed_dirs = 0
    dir_failures: list[str] = []
    if target_repos:
        removed_dirs, dir_failures = _remove_empty_variant_dirs(target_repos, variant)
        removed_snap_dirs, snap_dir_failures = _remove_empty_snapshot_dirs(target_repos)
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


def _loaded_id_matches_repo(loaded_id: str, repo_id: str) -> bool:
    """Match a loaded repo ID or an on-disk path inside any copy of the repo."""
    rid = repo_id.lower()
    lid = loaded_id.lower()
    if lid == rid or lid.startswith(f"{rid}/"):
        return True

    try:
        loaded_path = Path(loaded_id).expanduser().resolve(strict = False)
    except (OSError, RuntimeError, ValueError):
        return False
    for repo_dir in iter_repo_cache_dirs("model", repo_id):
        try:
            resolved_repo = repo_dir.resolve(strict = False)
            if loaded_path == resolved_repo or loaded_path.is_relative_to(resolved_repo):
                return True
        except (OSError, RuntimeError, ValueError):
            continue
    return False


def _loaded_repo_variant_blocks_delete(
    loaded_id: str, repo_id: str, delete_variant: Optional[str], loaded_variant: Optional[str]
) -> bool:
    if not _loaded_id_matches_repo(loaded_id, repo_id):
        return False
    if not delete_variant:
        return True
    if not loaded_variant:
        return True
    return loaded_variant.lower() == delete_variant.lower()


_LOAD_STATE_UNVERIFIABLE_DETAIL = (
    "Couldn't verify whether this model is still loaded for inference. "
    "Unload it if it is active, then try deleting again."
)


def _llama_cpp_blocks_delete(repo_id: str, variant: Optional[str]) -> bool:
    """Whether the llama.cpp backend holds *repo_id* (/variant). Acquiring fails open (import error means nothing loaded); reading load state is unguarded so a raise propagates and the caller fails closed rather than delete a live model."""
    try:
        from routes.inference import get_llama_cpp_backend
        backend = get_llama_cpp_backend()
    except Exception as e:
        logger.debug(f"llama.cpp backend unavailable during delete guard for {repo_id}: {e}")
        return False
    loaded_id = backend.model_identifier
    loaded_variant = getattr(backend, "hf_variant", None)
    if backend.is_active and not backend.is_loaded and loaded_id:
        return _loaded_repo_variant_blocks_delete(
            loaded_id,
            repo_id,
            variant,
            loaded_variant,
        )
    if backend.is_loaded and loaded_id:
        return _loaded_repo_variant_blocks_delete(
            loaded_id,
            repo_id,
            variant,
            loaded_variant,
        )
    return False


def _inference_backend_blocks_delete(repo_id: str) -> bool:
    """Whether the subprocess inference backend holds *repo_id*; same fail-open-on-acquire / surface-on-query contract as :func:`_llama_cpp_blocks_delete`."""
    try:
        from core.inference.orchestrator import peek_inference_backend

        # Peek, never construct: building one just to learn nothing is loaded imports torch.
        backend = peek_inference_backend()
    except Exception as e:
        logger.debug(f"Inference backend unavailable during delete guard for {repo_id}: {e}")
        return False
    if backend is None:
        return False
    active_name = backend.active_model_name
    return bool(active_name) and _loaded_id_matches_repo(active_name, repo_id)


def _diffusion_blocks_delete(repo_id: str) -> Optional[str]:
    """The 400 detail if the Images backend holds *repo_id*, else None.

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
        if _loaded_id_matches_repo(str(status["repo_id"]), repo_id):
            return "Unload the model before deleting"
    # sd.cpp re-reads companion VAE / text-encoder files every generation and status().repo_id covers
    # only the main GGUF, so refuse the companions too.
    for lid in getattr(engine, "loaded_repo_ids", tuple)():
        if _loaded_id_matches_repo(str(lid), repo_id):
            return "Unload the model before deleting"
    # A downloading repo still reports loaded=False, but deleting would pull blobs from under the in-flight fetch.
    for lid in getattr(engine, "loading_repo_ids", tuple)():
        if _loaded_id_matches_repo(str(lid), repo_id):
            return "An Images model load is using this repo; wait for it to finish"
    return None


def _video_blocks_delete(repo_id: str) -> Optional[str]:
    """The 400 detail if the Video backend holds or is fetching *repo_id*, else None.

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
    if status.get("loaded"):
        # repo_id names the checkpoint; for a GGUF / single-file load the companion base supplies the VAE
        # and text encoders, so refuse it too.
        for key in ("repo_id", "base_repo"):
            held = status.get(key)
            if held and _loaded_id_matches_repo(str(held), repo_id):
                return "Unload the model before deleting"
    # The native H3 runtime re-reads its Qwen encoder and both VAEs from companion repos that are
    # neither of the two ids above, so refuse those as well.
    for lid in getattr(backend, "loaded_repo_ids", tuple)():
        if _loaded_id_matches_repo(str(lid), repo_id):
            return "Unload the model before deleting"
    for lid in getattr(backend, "loading_repo_ids", tuple)():
        if _loaded_id_matches_repo(str(lid), repo_id):
            return "A Video model load is using this repo; wait for it to finish"
    return None


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
    Refuses if the model is currently loaded for inference.

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

    # Fail closed with 503 rather than unlink weights under a running process.
    def _load_state_blocks_delete() -> Optional[str]:
        if _llama_cpp_blocks_delete(repo_id, variant) or (
            _inference_backend_blocks_delete(repo_id)
        ):
            return "Unload the model before deleting"
        # The guards above are chat-only; Images / Video hold their own pipelines.
        return _diffusion_blocks_delete(repo_id) or _video_blocks_delete(repo_id)

    try:
        blocks_detail = await asyncio.to_thread(_load_state_blocks_delete)
    except Exception as e:
        logger.warning(f"Load-state verification failed for {repo_id}; refusing delete: {e}")
        raise HTTPException(
            status_code = 503,
            detail = _LOAD_STATE_UNVERIFIABLE_DETAIL,
        )
    if blocks_detail:
        raise HTTPException(
            status_code = 400,
            detail = blocks_detail,
        )

    repo_key = await asyncio.to_thread(resolve_cached_repo_id_case, repo_id, repo_type = "model")
    if not downloads.registry.begin_delete(repo_key, variant):
        detail = (
            f"Cancel the {variant} download before deleting it."
            if variant is not None
            else "Cancel the active downloads before deleting."
        )
        raise HTTPException(status_code = 400, detail = detail)
    try:
        # Re-derived now the scope is reserved, as only_if_orphan re-derives its own answer below: the
        # first read ran before the reservation existed, so a load starting in between published its claim
        # too late, and begin_delete misses it too.
        try:
            blocks_detail = await asyncio.to_thread(_load_state_blocks_delete)
        except Exception as e:
            logger.warning(f"Load-state verification failed for {repo_id}; refusing delete: {e}")
            raise HTTPException(
                status_code = 503,
                detail = _LOAD_STATE_UNVERIFIABLE_DETAIL,
            )
        if blocks_detail:
            raise HTTPException(
                status_code = 400,
                detail = blocks_detail,
            )
        return await asyncio.to_thread(
            _delete_cached_model_blocking,
            repo_id,
            variant,
            hf_token,
            cache_path,
            only_if_orphan = only_if_orphan,
        )
    finally:
        downloads.registry.end_delete(repo_key, variant)
        cache_inventory.invalidate_hf_cache_scans()


def _delete_cached_model_blocking(
    repo_id: str,
    variant: Optional[str],
    hf_token: Optional[str],
    cache_path: Optional[str] = None,
    *,
    only_if_orphan: bool = False,
) -> dict:
    # Free up space's list can be minutes old, and a background download finishing turns that orphan
    # into an installed checkpoint neither guard below catches.
    if only_if_orphan:
        from hub.services.models import companion_cleanup
        from hub.utils import companion_assets

        try:
            copies = companion_cleanup._repos_by_id(cache_inventory.all_hf_cache_scans()).get(
                repo_id.strip().lower(), []
            )
            # Only the copy being removed: the orphan listing emits one row per cache root, so a full-pipeline
            # copy in another remembered cache must not veto removing the companion-only copy listed.
            if cache_path:
                wanted = Path(cache_path)
                copies = [
                    r
                    for r in copies
                    if getattr(r, "repo_path", None) and Path(getattr(r, "repo_path")) == wanted
                ]
                if not copies:
                    # An empty match means the target root is not in this scan, and concluding "orphan" from copies we
                    # did not look at is the fail-open this precondition prevents; raising lands in the 503.
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
    # family, so removing it while one is installed leaves that quant unloadable. Derived from what is
    # installed right now, and only for a WHOLE-repo delete.
    # Here rather than in the async caller so it shares this function's cache walk and stubs: the check
    # IS part of the destructive stage.
    # The exception is a companion whose asset IS a named GGUF variant: native Qwen-Image opens one
    # fixed filename inside a chat GGUF repo, so removing that quant strands the image checkpoint. A
    # FLAG, never a rewrite of `variant`: widening the scope would delete revisions the user did not
    # ask for.
    guard_this_delete = variant is None or _variant_is_a_required_companion_asset(repo_id, variant)
    if guard_this_delete and _is_companion_base_repo(repo_id):
        # Fails CLOSED, and only here: the lookup above already established this repo IS a companion base,
        # so an unreadable cache means the dependants cannot be enumerated, not that there are none.
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
        # If a sibling quant is downloading concurrently, restrict this delete to the variant's own files
        # and leave the shared mmproj companion for it.
        sibling_active = bool(
            variant and downloads.registry.has_active_peer_variant(repo_id, variant)
        )

        cache_scans = cache_inventory.all_hf_cache_scans()

        # A repo can live in several remembered caches, so target exactly one or a delete removes copies in
        # other, previously selected caches.
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

        target_root = resolve_delete_target_root("model", repo_id, cache_path, owners.keys())
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
                    companions = not sibling_active,
                    root = target_root,
                )
                if incomplete_result.unresolved:
                    raise HTTPException(
                        status_code = 409,
                        detail = (
                            f"Couldn't fully delete {variant} for {repo_id}: partial "
                            "download bytes exist but this variant's blob hashes are unavailable. "
                            "Reconnect or provide access to the repo, then try again."
                        ),
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
                sibling_active = sibling_active,
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
