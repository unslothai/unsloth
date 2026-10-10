# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cached model deletion."""

from __future__ import annotations

from hub.services.models import account_access

import asyncio
import errno
import inspect
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub.utils._shared_blobs import shared_blob_target, sweep_shared_blob

    # Private hf_hub API; if the signature changed, fall back to plain unlink.
    inspect.signature(shared_blob_target).bind(Path(), Path())
    inspect.signature(sweep_shared_blob).bind(Path(), cache_dir = Path())
except (ImportError, TypeError, ValueError):
    shared_blob_target = None
    sweep_shared_blob = None

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


def _unlink_variant_blob(blob: Path, cache_dir: Optional[Path]) -> int:
    shared_target = None
    if shared_blob_target is not None:
        # hf_hub matches paths lexically; try the blob's own root form first (symlinks, 8.3 names).
        for root in dict.fromkeys(
            r for r in (blob.parent.parent.parent, cache_dir) if r is not None
        ):
            shared_target = shared_blob_target(blob, root)
            if shared_target is not None:
                cache_dir = root
                break
    if shared_target is None:
        freed = blob.stat().st_size
        blob.unlink()
        return freed
    blob.unlink()
    return sweep_shared_blob(shared_target, cache_dir = cache_dir)


def _path_exists_or_symlink(path: Path) -> bool:
    try:
        return path.is_symlink() or path.exists()
    except OSError:
        return False


def _unlink_snapshot_entry(snap: Path) -> int:
    """Unlink one snapshot entry, plus any AppleDouble sidecar beside it. Returns the entries removed, which never counts the sidecar: it is metadata about a file the caller asked to remove, not a second file."""
    removed = 0
    if _path_exists_or_symlink(snap):
        snap.unlink()
        removed += 1
    remove_appledouble_sidecar(snap)
    return removed


def _repo_file_matches(target_repo, predicate) -> list[tuple[Path, Optional[Path], str]]:
    """Files whose snapshot-relative path satisfies *predicate*. Relative, not the bare ``file_name``: huggingface_hub sets that to ``file_path.name`` (and our own recovery scan to ``entry.name``), so a companion in ``dspark/`` or ``MTP/`` arrived here indistinguishable from a root file. Every predicate below keys on the directory for at least one supported layout, and the quant labels they extract are unchanged by the prefix."""
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
    """Remove now-empty ``snapshots/<rev>/<quant>/`` folders for *variant* (the quant label names the folder); only empty dirs go, so siblings are safe. Returns (count removed, removal failures other than a concurrent refill)."""
    # Qualified keys (path, H3 stem, bpw modifier) must not reach for a <quant>/ dir they don't own.
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
                    # ENOTEMPTY means a concurrent download refilled it; other errors surface.
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
    """The variant keys in *target_repo* that *variant* names, lowercased. Its own key, always, plus the unambiguous bare-quant alias the download side already admits (``gguf_plan.plan_for_variant``): a repo filing its sole Q4_K_M under a shared container (``weights/model-Q4_K_M.gguf``) qualifies that key, because the key is a pure function of the path and cannot know the directory disambiguates nothing, so every stored pin and every explicit ``repo:Q4_K_M`` names it by quant alone, and admitting the alias for the download and not for the delete answered "not found" and left the weights on disk. Only when it is unambiguous, exactly as the download side decides it: a repo that really does hold several checkpoints at one quant gets no fallback, because there the bare name genuinely does not name one of them and deleting the wrong one is unrecoverable."""
    wanted = (variant or "").strip().lower()
    if not wanted or "/" in wanted:
        return {wanted}
    keys = {
        gguf_variant_key(name).lower()
        for _snap, _blob, name in _repo_file_matches(target_repo, _is_main_gguf_filename)
    }
    if wanted in keys:
        return {wanted}
    # PATH-qualified keys only: an H3 root stem's bare quant names both partitions.
    aliased = {key for key in keys if "/" in key and bare_quant_alias(key).lower() == wanted}
    return aliased if len(aliased) == 1 else {wanted}


def _audio_cpp_package_scope(target_repo, variant: str) -> tuple[frozenset[str], frozenset[str]]:
    """``(own files, protected files)`` of an audio.cpp package mix being deleted.

    A package mix (MiniMax Music 3, YuE2) is several GGUFs plus configs, some of them shared with
    other mixes, so the quant key alone neither finds all of its files nor knows which a sibling
    still needs. Own files are the mix's whole file list; protected files are every file of the
    other mixes still fully on disk. Both empty for any other repo.
    """
    try:
        from core.inference.audio_cpp_models import package_variant_files
    except Exception:  # noqa: BLE001 - no audio.cpp support, no package layout
        return frozenset(), frozenset()
    names = [name for _snap, _blob, name in _repo_file_matches(target_repo, lambda name: True)]
    packages = package_variant_files(names)
    if not packages:
        return frozenset(), frozenset()
    wanted = (variant or "").strip().lower()
    own = next((files for key, files in packages.items() if key.lower() == wanted), ())
    protected = {path for key, files in packages.items() if key.lower() != wanted for path in files}
    return frozenset(own), frozenset(protected)


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
        package_files, protected = _audio_cpp_package_scope(target_repo, variant)
        matched = [
            match
            for match in _repo_file_matches(
                target_repo,
                lambda name, keys = wanted_keys: name in package_files
                or (_is_main_gguf_filename(name) and gguf_variant_key(name).lower() in keys),
            )
            # Keep components another downloaded mix still loads (MiniMax Q4_0/Q8_0 share a decoder).
            if match[2] not in protected
        ]

        for snap, _blob, name in matched:
            try:
                removed_snapshots += _unlink_snapshot_entry(snap)
            except OSError as e:
                failures.append(f"{name}: {e}")

        companion_matches: list[tuple[Path, Optional[Path], str]] = []
        if matched and not sibling_active and not _has_remaining_main_gguf(target_repo):
            companion_matches = _repo_file_matches(
                target_repo,
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
        cache_dir = root
        if cache_dir is None and repo_dir is not None:
            cache_dir = repo_dir.parent
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
                    deleted_bytes += _unlink_variant_blob(blob, cache_dir)
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
    """Prune stale main-GGUF files for a variant after a replacement verified. Intentionally narrower than user-driven delete: it removes only same-variant main files whose local blob hash is not in *keep_main_hashes*, then unlinks their blobs only if no remaining snapshot references them. Shared companions and sibling variants are left intact."""
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
            # Prune only identifiable stale blobs; a no-symlink snapshot file has no hash.
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
                    deleted_bytes += _unlink_variant_blob(blob, target_hub_cache)
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
        from core.inference import model_slots
        from routes.inference import get_llama_cpp_backend

        backends = [get_llama_cpp_backend(), *(slot.llama for slot in model_slots.resident())]
        filling = model_slots.filling_model()
    except Exception as e:
        logger.debug(f"llama.cpp backend unavailable during delete guard for {repo_id}: {e}")
        return False
    if filling and _loaded_id_matches_repo(filling, repo_id):
        return True
    for backend in backends:
        loaded_id = backend.model_identifier
        if (backend.is_active or backend.is_loaded) and loaded_id:
            if _loaded_repo_variant_blocks_delete(
                loaded_id,
                repo_id,
                variant,
                getattr(backend, "hf_variant", None),
            ):
                return True
    return False


def _inference_backend_blocks_delete(repo_id: str) -> bool:
    """Whether the subprocess inference backend holds *repo_id*; same fail-open-on-acquire / surface-on-query contract as :func:`_llama_cpp_blocks_delete`."""
    try:
        from core.inference.orchestrator import peek_inference_backend
        from core.inference import model_slots

        primary = peek_inference_backend()
        kept = [slot.orchestrator for slot in model_slots.resident()]
    except Exception as e:
        logger.debug(f"Inference backend unavailable during delete guard for {repo_id}: {e}")
        return False
    for backend in (primary, *kept):
        if backend is None:
            continue
        active_name = backend.active_model_name
        if active_name and _loaded_id_matches_repo(active_name, repo_id):
            return True
    for backend in kept:
        if any(_loaded_id_matches_repo(m, repo_id) for m in getattr(backend, "loading_models", ())):
            return True
    return False


def _diffusion_blocks_delete(repo_id: str) -> Optional[str]:
    """The 400 detail if the Images backend holds *repo_id*, else None. Queries the ACTIVE engine: on a native selection the diffusers singleton reports unloaded while sd-cli still generates from the cached GGUF. Same fail-open-on-acquire contract as :func:`_llama_cpp_blocks_delete`."""
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
    # sd.cpp re-reads companion files every generation; refuse those too.
    for lid in getattr(engine, "loaded_repo_ids", tuple)():
        if _loaded_id_matches_repo(str(lid), repo_id):
            return "Unload the model before deleting"
    # Deleting during a download would pull blobs from under the fetch.
    for lid in getattr(engine, "loading_repo_ids", tuple)():
        if _loaded_id_matches_repo(str(lid), repo_id):
            return "An Images model load is using this repo; wait for it to finish"
    for lid in getattr(engine, "draining_repo_ids", tuple)():
        if _loaded_id_matches_repo(str(lid), repo_id):
            return "An Images model load is still releasing this repo; wait for it to finish"
    return None


def any_model_load_blocks_cache_clear() -> Optional[str]:
    """The refusal detail if ANY inference backend is holding a cached model, else None.

    The guards above ask whether one repo is in use. Emptying the whole Hugging Face cache is
    every repo at once, so there is no repo to match on and anything loaded or loading is enough.
    sd.cpp in particular re-reads its companion VAE and text-encoder files for every generation,
    so a clear can break a model that was loaded long before it.

    Fail-open on ACQUIRE, like the guards above: a backend that cannot be reached is not holding
    anything this process can see. A backend that IS reachable and raises while being asked is a
    different matter, and the caller fails closed on it rather than unlink weights blindly.
    """
    try:
        from core.inference import model_slots
        from routes.inference import get_llama_cpp_backend

        backend = get_llama_cpp_backend()
        kept = model_slots.resident()
        kept_loading = model_slots.any_loading()
    except Exception as exc:  # noqa: BLE001 - unavailable is not "in use"
        logger.debug(f"llama.cpp backend unavailable during the cache-clear guard: {exc}")
    else:
        if (backend.is_loaded or backend.is_active) and backend.model_identifier:
            return "Unload the model before clearing the model cache"
        if any(
            model_slots.in_use(slot) or slot.llama.is_loaded or slot.llama.is_active
            for slot in kept
        ):
            return "Unload the model before clearing the model cache"
        if kept_loading:
            return "A model load is using the cache; wait for it to finish"

    # An HF-backed chat load has no llama-server until its GGUF downloads, and those files
    # bypass the download registry, so is_active and the reservation miss it.
    try:
        from core.inference.llama_cpp import chat_load_active
        loading_chat = chat_load_active()
    except Exception as exc:  # noqa: BLE001 - unavailable is not "in use", as above
        logger.debug(f"Chat load state unavailable during the cache-clear guard: {exc}")
    else:
        if loading_chat:
            return "A model load is using the cache; wait for it to finish"

    try:
        from core.inference.orchestrator import peek_inference_backend

        # Peek, never construct: constructing imports torch.
        engine = peek_inference_backend()
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Inference backend unavailable during the cache-clear guard: {exc}")
    else:
        if engine is not None and engine.active_model_name:
            return "Unload the model before clearing the model cache"

    for label, load in (
        ("Images", "core.inference.diffusion_engine_router:get_active_diffusion_engine"),
        ("Video", "core.inference.video:get_video_backend"),
    ):
        module_name, _, attr = load.partition(":")
        try:
            module = __import__(module_name, fromlist = [attr])
            held = getattr(module, attr)()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"{label} backend unavailable during the cache-clear guard: {exc}")
            continue
        if held is None:
            continue
        if held.status().get("loaded"):
            return "Unload the model before clearing the model cache"
        if any(getattr(held, "loaded_repo_ids", tuple)()):
            return "Unload the model before clearing the model cache"
        if any(getattr(held, "loading_repo_ids", tuple)()):
            return f"An {label} model load is using the cache; wait for it to finish"
        # Cancelled loads keep reading in _prefetch_files while draining; clear-all must refuse too.
        if any(getattr(held, "draining_repo_ids", tuple)()):
            return f"An {label} model load is still unwinding; wait for it to finish"

    # Dictation sidecars read the same hub cache, so a resident STT worker blocks the clear.
    try:
        from core.inference import stt_registry
        dictation = stt_registry.resident()
    except Exception as exc:  # noqa: BLE001 - unavailable is not "in use", as above
        logger.debug(f"Dictation unavailable during the cache-clear guard: {exc}")
    else:
        if dictation.get("model"):
            return "Unload the dictation model before clearing the model cache"
        if dictation.get("loading"):
            return "A dictation model load is using the cache; wait for it to finish"
    return None


def _video_blocks_delete(repo_id: str) -> Optional[str]:
    """The 400 detail if the Video backend holds or is fetching *repo_id*, else None. Video repos share the On Device delete action, so a live Wan / LTX / Hunyuan pipeline could otherwise lose its snapshot. Mirrors :func:`_diffusion_blocks_delete`."""
    try:
        from core.inference.video import get_video_backend
        backend = get_video_backend()
    except Exception as e:
        logger.debug(f"Video backend unavailable during delete guard for {repo_id}: {e}")
        return None
    status = backend.status()
    if status.get("loaded"):
        for key in ("repo_id", "base_repo"):
            held = status.get(key)
            if held and _loaded_id_matches_repo(str(held), repo_id):
                return "Unload the model before deleting"
    # Native H3 re-reads encoder and VAEs from separate companion repos.
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
    """Whether *variant* names a file an installed checkpoint's native load actually opens. Not "would this empty the repo": the asset is a FIXED filename, so a sibling quant left behind substitutes for nothing. Fails CLOSED, and cheaply: a True here only runs the dependants check, which answers "nobody needs it" for every ordinary repo and lets the delete through."""
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
    account_access.require_installation_owner()
    if not _is_valid_repo_id(repo_id):
        raise HTTPException(status_code = 400, detail = "Invalid repo_id format")
    variant = (variant or "").strip() or None
    if variant is not None and not _is_valid_gguf_variant(variant):
        raise HTTPException(
            status_code = 400,
            detail = f"Invalid gguf_variant: {variant!r}",
        )

    def _load_state_blocks_delete() -> Optional[str]:
        if _llama_cpp_blocks_delete(repo_id, variant) or (
            _inference_backend_blocks_delete(repo_id)
        ):
            return "Unload the model before deleting"
        if _audio_cpp_blocks_delete(repo_id):
            return "Unload the audio model before deleting"
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
        # Re-derived after reserving scope: a load starting before the reservation is missed otherwise.
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
        result = await asyncio.to_thread(
            _delete_cached_model_blocking,
            repo_id,
            variant,
            hf_token,
            cache_path,
            only_if_orphan = only_if_orphan,
        )
        # audio.cpp link farm hardlinks deleted blobs; prune so disk space is freed.
        from core.inference.audio_cpp_files import prune_link_farm
        from hub.utils.hf_cache_state import hf_cache_roots

        # Every remembered root: an omitted cache_path deletes from the sole owner, maybe not active.
        def _prune_all() -> None:
            for root in hf_cache_roots():
                prune_link_farm(root)

        await asyncio.to_thread(_prune_all)
        from core.inference.audio_cpp_models import forget

        forget()
        return result
    finally:
        downloads.registry.end_delete(repo_key, variant)
        cache_inventory.invalidate_hf_cache_scans()


def _audio_cpp_blocks_delete(repo_id: str) -> bool:
    """Whether an audio.cpp model from this repo is resident or loading. An umbrella id names a
    subfolder of the repo, so the id comparisons of the other guards never match it."""
    from core.inference.audio_cpp_models import repo_of
    from core.inference.orchestrator import peek_inference_backend
    from core.inference.stt_audiocpp_sidecar import get_audio_cpp_stt_sidecar

    wanted = (repo_id or "").strip().lower()

    def holds(name) -> bool:
        repo = repo_of(name) if isinstance(name, str) else None
        return bool(repo and repo.lower() == wanted)

    backend = peek_inference_backend()
    active = getattr(backend, "active_model_name", None) if backend is not None else None
    loading = tuple(getattr(backend, "loading_models", ()) or ()) if backend is not None else ()
    sidecar = get_audio_cpp_stt_sidecar()
    return (
        holds(active)
        or any(holds(name) for name in loading)
        or holds(sidecar.loaded_model)
        or holds(sidecar.loading_model)
    )


def _delete_cached_model_blocking(
    repo_id: str,
    variant: Optional[str],
    hf_token: Optional[str],
    cache_path: Optional[str] = None,
    *,
    only_if_orphan: bool = False,
) -> dict:
    from hub.utils.gguf_sources import cached_gguf_action_path

    cache_path = cached_gguf_action_path(repo_id, variant, cache_path)
    # The orphan list can be stale; a finished background download makes it a real checkpoint.
    if only_if_orphan:
        from hub.services.models import companion_cleanup
        from hub.utils import companion_assets

        try:
            copies = companion_cleanup._repos_by_id(cache_inventory.all_hf_cache_scans()).get(
                repo_id.strip().lower(), []
            )
            # Only the copy being removed: a full copy in another cache must not veto it.
            if cache_path:
                wanted = Path(cache_path)
                copies = [
                    r
                    for r in copies
                    if getattr(r, "repo_path", None) and Path(getattr(r, "repo_path")) == wanted
                ]
                if not copies:
                    # Unscanned target root: refuse rather than fail open (lands as 503).
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

    # Whole-repo deletes of a companion base, or a GGUF variant that IS a required asset, are guarded.
    # A flag, not a rewrite of `variant`: widening scope would delete things not asked for.
    guard_this_delete = variant is None or _variant_is_a_required_companion_asset(repo_id, variant)
    if guard_this_delete and _is_companion_base_repo(repo_id):
        # Fails CLOSED: unreadable cache means dependants cannot be enumerated.
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
        # Sibling downloading: leave the shared mmproj for it.
        sibling_active = bool(
            variant and downloads.registry.has_active_peer_variant(repo_id, variant)
        )

        cache_scans = cache_inventory.all_hf_cache_scans()

        # A repo can live in several caches; target exactly one.
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
