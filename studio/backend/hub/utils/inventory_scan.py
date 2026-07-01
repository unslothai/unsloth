# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HF cache inventory scanner.

Read-only walks of the HuggingFace hub cache plus legacy/default
cache locations. Builds the foundation that Hub inventory endpoints
and the DownloadRegistry both consume.

The worker spawn / transport-marker preparation / DownloadRegistry
layers built on top of these primitives live in download_registry.py.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

from hub.utils.gguf import (
    extract_quant_label,
    is_gguf_filename,
    is_mmproj_filename,
    is_mtp_drafter_path,
)
from hub.utils.state_dir import RepoType

from hub.utils.hf_cache_state import (
    INCOMPLETE_SUFFIX,
    has_incomplete_blobs,
    hf_cache_root,
    iter_repo_cache_dirs,
    latest_snapshot_dir,
    repo_cache_dir_has_incomplete_blobs,
)

# Inventory is invalidated explicitly on every app-driven cache mutation, so
# this TTL only bounds staleness from out-of-band edits while skipping re-walks
# on rapid UI navigation.
_HF_CACHE_SCANS_TTL_SECONDS = 15.0
_GGUF_SPLIT_RE = re.compile(r"-(\d{3,})-of-(\d{3,})(?=\.gguf$)", re.IGNORECASE)
_hf_cache_scans_lock = threading.Lock()


@dataclass
class _HfCacheScanFlight:
    event: threading.Event
    epoch: int
    result: Optional[list] = None
    error: Optional[BaseException] = None


_hf_cache_scans_flight: Optional[_HfCacheScanFlight] = None
_hf_cache_scans_result: Optional[list] = None
_hf_cache_scans_cached_at: float = 0.0
# Bumped on every invalidation. A scan tags itself with the epoch it began
# under; an invalidation mid-scan changes the epoch so the in-flight result is
# neither cached nor served to callers that arrived after the mutation.
_hf_cache_scans_epoch: int = 0


def invalidate_hf_cache_scans() -> None:
    global _hf_cache_scans_result, _hf_cache_scans_cached_at, _hf_cache_scans_epoch
    with _hf_cache_scans_lock:
        _hf_cache_scans_result = None
        _hf_cache_scans_cached_at = 0.0
        _hf_cache_scans_epoch += 1


def all_hf_cache_scans() -> list:
    global _hf_cache_scans_flight, _hf_cache_scans_result, _hf_cache_scans_cached_at

    now = time.monotonic()
    with _hf_cache_scans_lock:
        if (
            _hf_cache_scans_result is not None
            and (now - _hf_cache_scans_cached_at) < _HF_CACHE_SCANS_TTL_SECONDS
        ):
            return list(_hf_cache_scans_result)
        start_epoch = _hf_cache_scans_epoch
        flight = _hf_cache_scans_flight
        # Only coalesce onto an in-flight scan from the current epoch; one that
        # began before an intervening invalidation is superseded so
        # post-mutation callers never receive pre-mutation data.
        if flight is None or flight.epoch != start_epoch:
            flight = _HfCacheScanFlight(event = threading.Event(), epoch = start_epoch)
            _hf_cache_scans_flight = flight
            owner = True
        else:
            owner = False

    if not owner:
        flight.event.wait()
        if flight.error is not None:
            raise flight.error
        return list(flight.result or [])

    try:
        scans = _compute_all_hf_cache_scans()
        with _hf_cache_scans_lock:
            flight.result = scans
            if _hf_cache_scans_epoch == flight.epoch:
                _hf_cache_scans_result = scans
                _hf_cache_scans_cached_at = time.monotonic()
        return scans
    except Exception as exc:
        with _hf_cache_scans_lock:
            if _hf_cache_scans_epoch == flight.epoch:
                _hf_cache_scans_result = None
                _hf_cache_scans_cached_at = 0.0
            flight.error = exc
        raise
    finally:
        with _hf_cache_scans_lock:
            if _hf_cache_scans_flight is flight:
                _hf_cache_scans_flight = None
            flight.event.set()


def _compute_all_hf_cache_scans() -> list:
    from huggingface_hub import scan_cache_dir
    from hub.utils.paths import legacy_hf_cache_dir, hf_default_cache_dir

    scans: list = []
    seen: set[str] = set()
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        active = Path(HF_HUB_CACHE).resolve()
        seen.add(str(active))
        if active.is_dir():
            scans.append(scan_cache_dir())
    except Exception as exc:
        logger.warning("Could not scan active HF cache: %s", exc)

    for extra_fn in (legacy_hf_cache_dir, hf_default_cache_dir):
        try:
            extra = extra_fn()
            # is_dir()/resolve() can raise on an inaccessible path; skip it.
            if not extra.is_dir():
                continue
            resolved = str(extra.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            scans.append(scan_cache_dir(cache_dir = str(extra)))
        except Exception as exc:
            logger.warning("Could not scan HF cache %s: %s", extra_fn.__name__, exc)
    return scans


def token_fingerprint(hf_token: Optional[str]) -> str:
    """16-char SHA256 prefix used as a cache-key qualifier for gated repos.

    Lets per-token size/snapshot caches refuse to serve a previously
    fetched value back to a different token (a private/gated repo's
    metadata is only valid for the credential that fetched it).
    """
    if not hf_token:
        return ""
    return hashlib.sha256(hf_token.encode()).hexdigest()[:16]


def resolve_hf_cache_realpath(repo_dir: Path) -> Optional[str]:
    """Pick the most useful on-disk path for a HF cache repo dir.

    Prefers the most-recent snapshot dir (what ``from_pretrained``
    actually points at). Falls back to the cache repo root. Returns the
    resolved realpath so symlinks under ``snapshots/`` are followed back
    to ``blobs/``.
    """
    try:
        latest = latest_snapshot_dir(repo_dir)
        if latest is not None:
            return str(latest.resolve())
        return str(repo_dir.resolve())
    except Exception:
        return None


def resolve_snapshot_dir_for_scan(
    repo_type: str,
    repo_id: str,
    repo_cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Latest snapshot dir for a cache row, or the first populated HF cache root.

    Scanner-side counterpart to snapshot_download()'s return value (which the
    scanner cannot access). With a *repo_cache_dir*, returns its newest
    snapshot. Otherwise scans roots in priority order (active, legacy, default)
    and returns the newest snapshot in the first root that holds one; active is
    where snapshot_download writes, so it is authoritative. Within a root,
    picks by mtime (what from_pretrained resolves to) rather than refs/main,
    since the user may have downloaded a non-main commit.
    """
    if repo_cache_dir is not None:
        latest = latest_snapshot_dir(repo_cache_dir)
        if latest is None:
            return None
        try:
            return latest.resolve()
        except OSError:
            return None
    for repo_dir in iter_repo_cache_dirs(repo_type, repo_id):
        latest = latest_snapshot_dir(repo_dir)
        if latest is None:
            continue
        try:
            return latest.resolve()
        except OSError:
            continue
    return None


def _compose_partial(*signals: Callable[[], bool]) -> bool:
    return any(signal() for signal in signals)


def _state_applies_to_repo_cache_dir(repo_cache_dir: Optional[Path]) -> bool:
    if repo_cache_dir is None:
        return True
    root = hf_cache_root()
    if root is None:
        return False
    try:
        return repo_cache_dir.resolve().parent == root.resolve()
    except OSError:
        return False


def _legacy_partial(
    repo_type: str,
    repo_id: str,
    repo_cache_dir: Optional[Path] = None,
) -> bool:
    if repo_cache_dir is not None:
        return repo_cache_dir_has_incomplete_blobs(repo_cache_dir)
    return has_incomplete_blobs(repo_type, repo_id)


def _repo_cache_dir_incomplete_hashes(repo_cache_dir: Path) -> set[str]:
    blobs_dir = repo_cache_dir / "blobs"
    if not blobs_dir.is_dir():
        return set()
    hashes: set[str] = set()
    try:
        entries = list(blobs_dir.iterdir())
    except OSError:
        return hashes
    for blob in entries:
        try:
            if blob.is_file() and blob.name.endswith(INCOMPLETE_SUFFIX):
                hashes.add(blob.name[: -len(INCOMPLETE_SUFFIX)])
        except OSError:
            continue
    return hashes


def _repo_cache_dir_has_non_gguf_broken_snapshot_symlinks(repo_cache_dir: Path) -> bool:
    latest = latest_snapshot_dir(repo_cache_dir)
    if latest is None:
        return False
    try:
        entries = list(latest.rglob("*"))
    except OSError:
        return False
    for entry in entries:
        try:
            if not entry.is_symlink() or entry.exists():
                continue
            rel = entry.relative_to(latest).as_posix()
            if is_gguf_filename(rel):
                continue
            return True
        except OSError:
            continue
    return False


def _gguf_variant_manifest_blob_hashes(repo_id: str) -> frozenset[str]:
    from hub.utils import download_manifest

    hashes: set[str] = set()
    for variant, _path in download_manifest.iter_variant_manifests("model", repo_id):
        manifest = download_manifest.read_manifest("model", repo_id, variant)
        if manifest is None:
            continue
        for expected in manifest.expected_files:
            if expected.sha256 and is_gguf_filename(expected.path):
                hashes.add(expected.sha256)
    return frozenset(hashes)


def _repo_cache_dir_has_snapshot_legacy_partial(
    repo_cache_dir: Path, *, ignored_blob_hashes: frozenset[str]
) -> bool:
    incomplete_hashes = _repo_cache_dir_incomplete_hashes(repo_cache_dir)
    if any(blob_hash not in ignored_blob_hashes for blob_hash in incomplete_hashes):
        return True
    return _repo_cache_dir_has_non_gguf_broken_snapshot_symlinks(repo_cache_dir)


def _snapshot_legacy_partial(
    repo_type: str,
    repo_id: str,
    repo_cache_dir: Optional[Path] = None,
) -> bool:
    if repo_type != "model":
        return _legacy_partial(repo_type, repo_id, repo_cache_dir)
    ignored_hashes = _gguf_variant_manifest_blob_hashes(repo_id)
    if repo_cache_dir is not None:
        return _repo_cache_dir_has_snapshot_legacy_partial(
            repo_cache_dir,
            ignored_blob_hashes = ignored_hashes,
        )
    return any(
        _repo_cache_dir_has_snapshot_legacy_partial(
            entry,
            ignored_blob_hashes = ignored_hashes,
        )
        for entry in iter_repo_cache_dirs(repo_type, repo_id)
    )


def _completed_gguf_variants(snapshot_dir: Optional[Path]) -> set[str]:
    if snapshot_dir is None:
        return set()
    complete: set[str] = set()
    split_groups: dict[str, dict[int, set[int]]] = {}
    try:
        paths = list(snapshot_dir.rglob("*"))
    except OSError:
        return set()
    for path in paths:
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                continue
        except OSError:
            continue
        rel = path.relative_to(snapshot_dir).as_posix()
        if not is_gguf_filename(rel) or is_mmproj_filename(rel) or is_mtp_drafter_path(rel):
            continue
        quant = extract_quant_label(rel)
        split = _GGUF_SPLIT_RE.search(path.name)
        if split is None:
            complete.add(quant)
            continue
        index = int(split.group(1))
        total = int(split.group(2))
        if index <= 0 or total <= 0 or index > total:
            continue
        split_groups.setdefault(quant, {}).setdefault(total, set()).add(index)
    for quant, groups in split_groups.items():
        for total, indices in groups.items():
            if indices == set(range(1, total + 1)):
                complete.add(quant)
                break
    return complete


def _manifest_partial(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    snapshot_dir: Optional[Path] = None,
    repo_cache_dir: Optional[Path] = None,
) -> bool:
    from hub.utils import download_manifest

    if not _state_applies_to_repo_cache_dir(repo_cache_dir):
        return False
    manifest = download_manifest.read_manifest(repo_type, repo_id, variant)
    if manifest is None:
        return False
    resolved = (
        snapshot_dir
        if snapshot_dir is not None
        else resolve_snapshot_dir_for_scan(repo_type, repo_id, repo_cache_dir)
    )
    if resolved is None:
        return True
    if repo_type == "model" and variant is not None:
        if download_manifest.verify_against_disk(manifest, resolved).ok:
            return False
        for candidate in _manifest_snapshot_dirs(repo_type, repo_id, repo_cache_dir):
            if candidate == resolved:
                continue
            if download_manifest.verify_against_disk(manifest, candidate).ok:
                return False
        return True
    return not download_manifest.verify_against_disk(manifest, resolved).ok


def _manifest_snapshot_dirs(
    repo_type: RepoType,
    repo_id: str,
    repo_cache_dir: Optional[Path] = None,
) -> list[Path]:
    repo_dirs = (
        [repo_cache_dir]
        if repo_cache_dir is not None
        else list(iter_repo_cache_dirs(repo_type, repo_id))
    )
    snapshots: list[Path] = []
    seen: set[str] = set()
    for repo_dir in repo_dirs:
        if repo_dir is None:
            continue
        snapshots_dir = repo_dir / "snapshots"
        try:
            if not snapshots_dir.is_dir():
                continue
            entries = list(snapshots_dir.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                if not entry.is_dir():
                    continue
                resolved = entry.resolve()
            except OSError:
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            snapshots.append(resolved)
    return snapshots


def is_snapshot_partial(
    repo_type: RepoType,
    repo_id: str,
    repo_cache_dir: Optional[Path] = None,
) -> bool:
    """Repo-row partial flag for snapshot-style downloads (full-snapshot
    models — safetensors/adapter/checkpoint — and all datasets).

    Composes three signals, cheapest first:
      1. Cancel marker (single stat).
      2. Snapshot-attributed legacy .incomplete blob / broken-symlink check.
      3. Manifest walk (stat per expected file under the latest snapshot).

    A manifest without a resolvable snapshot is partial: the worker got
    far enough to record expectations but did not leave a usable snapshot."""
    from hub.utils import download_manifest

    state_applies = _state_applies_to_repo_cache_dir(repo_cache_dir)
    return _compose_partial(
        lambda: state_applies and download_manifest.has_cancel_marker(repo_type, repo_id, None),
        lambda: _snapshot_legacy_partial(repo_type, repo_id, repo_cache_dir),
        lambda: _manifest_partial(
            repo_type,
            repo_id,
            None,
            None,
            repo_cache_dir,
        ),
    )


def is_variant_partial(
    repo_id: str,
    variant: str,
    snapshot_dir: Optional[Path] = None,
    *,
    incomplete_blob_hashes: Optional[set[str]] = None,
    variant_blob_hashes: Optional[frozenset[str]] = None,
    repo_cache_dir: Optional[Path] = None,
) -> bool:
    """Per-variant partial detection. Owns its manifest, owns its marker.
    Used by the GGUF variants endpoint to flag a specific quant as broken
    without contaminating other quants in the same repo.

    snapshot_dir is an optional hint to avoid re-walking the cache when a
    caller is checking many variants of the same repo (see
    is_gguf_repo_partial for that usage)."""
    from hub.utils import download_manifest

    state_applies = _state_applies_to_repo_cache_dir(repo_cache_dir)
    return _compose_partial(
        lambda: state_applies and download_manifest.has_cancel_marker("model", repo_id, variant),
        lambda: bool(
            incomplete_blob_hashes
            and variant_blob_hashes
            and incomplete_blob_hashes.intersection(variant_blob_hashes)
        ),
        lambda: _manifest_partial(
            "model",
            repo_id,
            variant,
            snapshot_dir,
            repo_cache_dir,
        ),
    )


def is_gguf_repo_partial(repo_id: str, repo_cache_dir: Optional[Path] = None) -> bool:
    """Repo-row partial flag for a GGUF repo. The inventory shows ONE row per
    GGUF repo (requires_variant=True); per-variant detail lives in
    GET /api/models/gguf-variants and uses is_variant_partial.

    *** DO NOT simplify this to "any variant partial -> repo partial" ***

    Tripwire scenario: user downloads Q8_0 fully, then starts Q4_K_M and
    cancels. Both variants share ONE inventory row. If row.partial flips True,
    _capabilities_for_format flips can_chat=False, so the user can no longer
    chat with the perfectly-good Q8_0 because of an unrelated cancelled Q4_K_M.

    Correct semantics: partial=True only when at least one variant is broken
    AND no other variant is clean. "Simplifying" to the obvious "any broken"
    form re-introduces this Q8+Q4 mixed-state regression.

    Composes signals:
      1. Cheap legacy fast-path (.incomplete blobs / broken symlinks).
      2. Per-variant manifest + marker enumeration, gated on "all broken".
    """
    from hub.utils import download_manifest

    has_legacy_partial = _legacy_partial("model", repo_id, repo_cache_dir)
    state_applies = _state_applies_to_repo_cache_dir(repo_cache_dir)
    snapshot_dir = resolve_snapshot_dir_for_scan(
        "model",
        repo_id,
        repo_cache_dir,
    )
    variants: set[str] = set(_completed_gguf_variants(snapshot_dir))
    if state_applies:
        for variant, _path in download_manifest.iter_variant_manifests(
            "model",
            repo_id,
        ):
            variants.add(variant)
        for variant, _path in download_manifest.iter_variant_markers(
            "model",
            repo_id,
        ):
            variants.add(variant)
    if not variants:
        return has_legacy_partial
    has_clean = False
    has_broken = has_legacy_partial
    for variant in variants:
        if is_variant_partial(
            repo_id,
            variant,
            snapshot_dir,
            repo_cache_dir = repo_cache_dir,
        ):
            has_broken = True
        else:
            has_clean = True
    return has_broken and not has_clean


def partial_transport_for(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    repo_cache_dir: Optional[Path] = None,
) -> Optional[str]:
    """Transport to surface on a partial row's resume affordance.

    Prefers the cancel marker's transport, then the manifest's. The fallback
    matters for rows partial without a marker (an errored/interrupted download
    leaves the manifest but no marker) so the UI can still show HTTP-resume vs
    XET-redownload instead of the neutral retry label. ``None`` when neither is
    available."""
    from hub.utils import download_manifest

    if not _state_applies_to_repo_cache_dir(repo_cache_dir):
        return None
    marker_transport = download_manifest.read_cancel_marker_transport(
        repo_type,
        repo_id,
        variant,
    )
    if marker_transport is not None:
        return marker_transport
    manifest = download_manifest.read_manifest(repo_type, repo_id, variant)
    return manifest.transport if manifest is not None else None
