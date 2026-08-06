# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hub download manifest + cancel-marker primitives.

Manifests record what a download was supposed to fetch (path + declared
size per expected file). Consumed by:
  - the worker post-download, to verify on-disk sizes match what HF
    declared, so a resume that no-ops doesn't get classified as success;
  - the inventory scanner, to mark a row partial when expected files
    are absent or undersized, so a half-finished GGUF/dataset doesn't
    masquerade as a complete on-device row.

Cancel markers record that a user-initiated cancel landed for a
(repo_type, repo_id, variant) triple. *Existence* is the signal the
scanner reads; the body carries debuggability metadata. Markers are
cleared at the start of a new download attempt (supersedes prior cancel)
and on successful completion (defensive, in case the start clear failed).

I/O contracts:
  - Writes are atomic via ``tmp + os.replace``: a SIGKILL mid-write
    cannot leave a half-written file readable to the next reader.
  - Manifest reads fail *open*: missing/corrupt/schema-mismatched
    manifests return ``None`` and the scanner falls through to the
    legacy on-disk-only check (matches HF-cache imports and pre-fix
    downloads that never wrote a manifest).
  - Cancel-marker reads fail *closed*: file existence is the signal
    regardless of body parseability, so a corrupt marker still
    suppresses the "on device" classification.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterator, Optional, Sequence
from urllib.parse import unquote

from loggers import get_logger

from hub.utils.state_dir import (
    RepoType,
    cancelled_dir,
    manifest_path,
    manifests_dir,
    marker_path,
    variant_filename_prefix,
)

logger = get_logger(__name__)


_MANIFEST_VERSION = 2
_LEGACY_MANIFEST_VERSION = 1
_SUPPORTED_MANIFEST_VERSIONS = frozenset({_LEGACY_MANIFEST_VERSION, _MANIFEST_VERSION})
_MARKER_VERSION = 2
_LEGACY_MARKER_VERSION = 1
_DATASET_COMPLETION_VARIANT_PREFIX = "_studio-dataset-complete-"
_MANIFEST_MIGRATION_MAX_FILES = 10_000
_MANIFEST_MIGRATION_MAX_BYTES = 32 * 1024 * 1024
_MANIFEST_MIGRATION_PREFIX_BYTES = 256
_V2_MANIFEST_PREFIX = re.compile(rb'^\s*\{\s*"version"\s*:\s*2\s*,')

# Verbatim phrase the worker emits on a degraded completion; shared so emit and match stay coupled.
MANIFEST_DEGRADED_MARKER = "completed without a manifest so partial detection is degraded"


@dataclass(frozen = True)
class ExpectedFile:
    path: str
    size: int
    sha256: Optional[str] = None


@dataclass(frozen = True)
class Manifest:
    repo_type: RepoType
    repo_id: str
    variant: Optional[str]
    started_at: str
    expected_files: tuple[ExpectedFile, ...]
    transport: Optional[str] = None
    hub_cache: Optional[str] = None
    version: int = _LEGACY_MANIFEST_VERSION
    commit_hash: Optional[str] = None
    metadata_derived: bool = False


@dataclass(frozen = True)
class VerifyResult:
    ok: bool
    missing: tuple[str, ...]
    size_mismatched: tuple[str, ...]


def _canonical_hub_cache(hub_cache: Optional[str | Path] = None) -> Optional[str]:
    if hub_cache is None:
        try:
            from utils.hf_cache_settings import get_hf_cache_paths
            hub_cache = get_hf_cache_paths().hub_cache
        except Exception:
            return None
    try:
        resolved = str(Path(hub_cache).expanduser().resolve(strict = False))
    except (OSError, RuntimeError, ValueError):
        resolved = str(hub_cache)
    return os.path.normcase(resolved)


def _read_state_payload(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("Could not read Hub state %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _legacy_state_applies(
    path: Path,
    requested_hub_cache: Optional[str],
    *,
    fail_closed: bool = False,
) -> bool:
    """Whether an old unscoped state file belongs to the requested cache.

    Transitional files that recorded their cache keep that ownership. Older
    files with no ownership can only be attributed to the currently selected
    cache, which matches the single-cache behavior under which they were
    written without leaking them into remembered inactive caches.
    """
    data = _read_state_payload(path)
    if data is not None:
        recorded = data.get("hub_cache")
        if isinstance(recorded, str) and recorded:
            return _canonical_hub_cache(recorded) == requested_hub_cache
    elif not fail_closed:
        return False
    return requested_hub_cache == _canonical_hub_cache()


def _state_read_path(
    path_factory,
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str],
    hub_cache: Optional[str | Path],
    *,
    fail_closed: bool = False,
) -> Optional[Path]:
    requested = _canonical_hub_cache(hub_cache)
    scoped = path_factory(repo_type, repo_id, variant, hub_cache = requested)
    try:
        if scoped is not None and scoped.is_file():
            return scoped
    except OSError:
        pass
    legacy = path_factory(repo_type, repo_id, variant)
    if legacy is None or legacy == scoped:
        return None
    try:
        if not legacy.is_file():
            return None
    except OSError:
        return None
    return legacy if _legacy_state_applies(legacy, requested, fail_closed = fail_closed) else None


def _atomic_write_json(path: Path, payload: dict) -> bool:
    # Per-write uuid suffix so a concurrent caller or a stale tmp cannot collide.
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        with tmp.open("w", encoding = "utf-8") as handle:
            handle.write(json.dumps(payload, indent = 2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("Atomic write failed for %s: %s", path, exc)
        try:
            tmp.unlink(missing_ok = True)
        except OSError:
            pass
        return False
    if os.name != "nt":
        try:
            flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                flags |= os.O_DIRECTORY
            parent_fd = os.open(path.parent, flags)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except OSError as exc:
            logger.debug("Parent dir fsync failed for %s: %s", path, exc)
    return True


def write_manifest(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str],
    expected_files: Sequence[ExpectedFile],
    transport: Optional[str] = None,
    *,
    hub_cache: Optional[str | Path] = None,
    commit_hash: Optional[str] = None,
    metadata_derived: bool = False,
    _schema_version: int = _LEGACY_MANIFEST_VERSION,
) -> bool:
    """Write/overwrite the manifest for this triple. Best-effort.

    ``False`` on write failure must not be treated as fatal: the
    worst-case fallback is the pre-fix scanner behavior (one missed
    partial detection), which is no regression.
    """
    recorded_hub_cache = _canonical_hub_cache(hub_cache)
    path = manifest_path(
        repo_type,
        repo_id,
        variant,
        hub_cache = recorded_hub_cache,
    )
    if path is None:
        return False
    normalized_commit = normalized_commit_hash(commit_hash)
    metadata_attestation = bool(metadata_derived and normalized_commit)
    # Ordinary download manifests retain the additive v1 schema so a Studio
    # downgrade can still detect missing/undersized files.  The v2-only
    # revision attestation lives under a separate completion-manifest key and
    # opts in below, so an older build never mistakes that record for the
    # ordinary download state it knows how to consume.
    if _schema_version not in _SUPPORTED_MANIFEST_VERSIONS:
        return False
    payload = {
        "version": _schema_version,
        "repo_type": repo_type,
        "repo_id": repo_id,
        "variant": variant,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "expected_files": [
            {
                "path": f.path,
                "size": int(f.size),
                **({"sha256": f.sha256} if f.sha256 else {}),
            }
            for f in expected_files
        ],
        "transport": transport,
        "hub_cache": recorded_hub_cache,
        "commit_hash": normalized_commit if metadata_attestation else None,
        "metadata_derived": metadata_attestation,
    }
    return _atomic_write_json(path, payload)


def normalized_commit_hash(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 256
        or normalized in {".", ".."}
        or Path(normalized).name != normalized
        or PureWindowsPath(normalized).name != normalized
    ):
        return None
    return normalized


def read_manifest(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    *,
    hub_cache: Optional[str | Path] = None,
) -> Optional[Manifest]:
    """Return the manifest if present and parseable; ``None`` otherwise.

    Treats missing-file, parse-error, and any schema mismatch all as
    ``None`` (fail-open). Scanner callers fall through to on-disk-only
    behavior on ``None`` so this never regresses legacy/imported repos
    that have no manifest.

    """
    path = _state_read_path(
        manifest_path,
        repo_type,
        repo_id,
        variant,
        hub_cache,
    )
    if path is None or not path.is_file():
        return None
    data = _read_state_payload(path)
    if data is None:
        return None
    version = data.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in _SUPPORTED_MANIFEST_VERSIONS
    ):
        logger.debug(
            "Manifest %s has unknown version %r; ignoring.",
            path,
            data.get("version"),
        )
        return None
    has_metadata_attestation = data.get("metadata_derived") is True
    if version == _MANIFEST_VERSION or has_metadata_attestation:
        recorded_repo_type = data.get("repo_type")
        recorded_repo_id = data.get("repo_id")
        recorded_variant = data.get("variant")
        if (
            recorded_repo_type != repo_type
            or not isinstance(recorded_repo_id, str)
            or recorded_repo_id.casefold() != repo_id.casefold()
            or not (recorded_variant is None or isinstance(recorded_variant, str))
            or (
                recorded_variant.strip().casefold()
                if isinstance(recorded_variant, str) and recorded_variant.strip()
                else None
            )
            != (variant.strip().casefold() if variant and variant.strip() else None)
        ):
            return None
    raw_files = data.get("expected_files")
    if not isinstance(raw_files, list):
        return None
    expected: list[ExpectedFile] = []
    for item in raw_files:
        if not isinstance(item, dict):
            return None
        file_path = item.get("path")
        size = item.get("size")
        if (
            not isinstance(file_path, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            return None
        sha256 = item.get("sha256")
        expected.append(
            ExpectedFile(
                path = file_path,
                size = size,
                sha256 = sha256 if isinstance(sha256, str) and sha256 else None,
            )
        )
    raw_variant = data.get("variant")
    transport = data.get("transport")
    commit_hash = normalized_commit_hash(data.get("commit_hash"))
    metadata_derived = bool(
        has_metadata_attestation and commit_hash is not None
    )
    return Manifest(
        repo_type = repo_type,
        repo_id = str(data.get("repo_id", repo_id)),
        variant = raw_variant if raw_variant else None,
        started_at = str(data.get("started_at", "")),
        expected_files = tuple(expected),
        transport = transport if transport in ("http", "xet") else None,
        hub_cache = data.get("hub_cache") if isinstance(data.get("hub_cache"), str) else None,
        version = version,
        commit_hash = commit_hash if metadata_derived else None,
        metadata_derived = metadata_derived,
    )


def _dataset_completion_variant(commit_hash: str) -> str:
    digest = hashlib.sha256(commit_hash.encode("utf-8")).hexdigest()[:32]
    return f"{_DATASET_COMPLETION_VARIANT_PREFIX}{digest}"


def write_dataset_completion(
    repo_id: str,
    commit_hash: str,
    expected_files: Sequence[ExpectedFile],
    transport: Optional[str] = None,
    *,
    hub_cache: Optional[str | Path] = None,
) -> bool:
    normalized_commit = normalized_commit_hash(commit_hash)
    recorded_hub_cache = _canonical_hub_cache(hub_cache)
    if (
        normalized_commit is None
        or recorded_hub_cache is None
        or not expected_files
        or any(
            not expected_path_is_safe(expected.path)
            or not isinstance(expected.size, int)
            or isinstance(expected.size, bool)
            or expected.size < 0
            for expected in expected_files
        )
    ):
        return False
    return write_manifest(
        "dataset",
        repo_id,
        _dataset_completion_variant(normalized_commit),
        expected_files,
        transport,
        hub_cache = recorded_hub_cache,
        commit_hash = normalized_commit,
        metadata_derived = True,
        _schema_version = _MANIFEST_VERSION,
    )


def read_dataset_completion(
    repo_id: str,
    commit_hash: str,
    *,
    hub_cache: Optional[str | Path] = None,
) -> Optional[Manifest]:
    normalized_commit = normalized_commit_hash(commit_hash)
    requested_hub_cache = _canonical_hub_cache(hub_cache)
    if normalized_commit is None or requested_hub_cache is None:
        return None
    variant = _dataset_completion_variant(normalized_commit)
    manifest = read_manifest(
        "dataset",
        repo_id,
        variant,
        hub_cache = requested_hub_cache,
    )
    if (
        manifest is None
        or manifest.version != _MANIFEST_VERSION
        or manifest.repo_id.casefold() != repo_id.casefold()
        or manifest.variant != variant
        or manifest.commit_hash != normalized_commit
        or not manifest.metadata_derived
        or not isinstance(manifest.hub_cache, str)
        or not manifest.hub_cache.strip()
        or _canonical_hub_cache(manifest.hub_cache) != requested_hub_cache
        or not manifest.expected_files
        or any(
            not expected_path_is_safe(expected.path)
            or not isinstance(expected.size, int)
            or isinstance(expected.size, bool)
            or expected.size < 0
            for expected in manifest.expected_files
        )
    ):
        return None
    return manifest


def expected_path_is_safe(path_value: str) -> bool:
    if (
        not isinstance(path_value, str)
        or not path_value
        or len(path_value) > 4096
        or "\x00" in path_value
        or "\\" in path_value
    ):
        return False
    decoded = unquote(path_value)
    if not decoded or decoded in {".", ".."} or "\x00" in decoded or "\\" in decoded:
        return False
    posix = PurePosixPath(decoded)
    windows = PureWindowsPath(decoded)
    return (
        not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and all(":" not in part for part in posix.parts)
        and ".." not in posix.parts
        and ".." not in windows.parts
    )


def _is_reserved_dataset_completion_payload(data: dict) -> bool:
    variant = data.get("variant")
    return (
        data.get("repo_type") == "dataset"
        and isinstance(variant, str)
        and variant.casefold().startswith(_DATASET_COMPLETION_VARIANT_PREFIX)
    )


def _read_migration_payload(path: Path) -> Optional[dict]:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(_MANIFEST_MIGRATION_PREFIX_BYTES)
            if _V2_MANIFEST_PREFIX.match(prefix) is None:
                return None
            if len(prefix) > _MANIFEST_MIGRATION_MAX_BYTES:
                return None
            raw = prefix + handle.read(_MANIFEST_MIGRATION_MAX_BYTES - len(prefix) + 1)
        if len(raw) > _MANIFEST_MIGRATION_MAX_BYTES:
            return None
        data = json.loads(raw)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _migration_manifest_paths(parent: Path) -> Iterator[Path]:
    yield from parent.glob("*.json")
    for scoped in parent.glob("cache-*"):
        if scoped.is_symlink() or not scoped.is_dir():
            continue
        yield from scoped.glob("*.json")


def migrate_ordinary_v2_manifests_for_downgrade() -> int:
    """Make prior ordinary v2 records readable after a Studio downgrade.

    Run once after orphan workers are reaped. Dataset-completion records stay
    v2, and any invalid or changed record is left untouched.
    """
    parent = manifests_dir()
    if parent is None:
        return 0

    migrated = 0
    try:
        candidates = _migration_manifest_paths(parent)
        for index, path in enumerate(candidates):
            if index >= _MANIFEST_MIGRATION_MAX_FILES:
                logger.debug("Stopped Hub manifest migration at the file limit")
                break
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                data = _read_migration_payload(path)
                if data is None or data.get("version") != _MANIFEST_VERSION:
                    continue
                if _is_reserved_dataset_completion_payload(data):
                    continue

                repo_type = data.get("repo_type")
                repo_id = data.get("repo_id")
                variant = data.get("variant")
                recorded_hub_cache = data.get("hub_cache")
                if (
                    repo_type not in ("model", "dataset")
                    or not isinstance(repo_id, str)
                    or not repo_id
                    or not (variant is None or isinstance(variant, str))
                    or not (
                        recorded_hub_cache is None
                        or (
                            isinstance(recorded_hub_cache, str)
                            and bool(recorded_hub_cache.strip())
                        )
                    )
                ):
                    continue

                selected = _state_read_path(
                    manifest_path,
                    repo_type,
                    repo_id,
                    variant,
                    recorded_hub_cache,
                )
                if selected != path:
                    continue
                manifest = read_manifest(
                    repo_type,
                    repo_id,
                    variant,
                    hub_cache = recorded_hub_cache,
                )
                if (
                    manifest is None
                    or manifest.version != _MANIFEST_VERSION
                    or any(
                        not expected_path_is_safe(expected.path)
                        for expected in manifest.expected_files
                    )
                ):
                    continue

                current = _read_migration_payload(path)
                if current != data:
                    continue
                migrated_payload = dict(current)
                migrated_payload["version"] = _LEGACY_MANIFEST_VERSION
                if _atomic_write_json(path, migrated_payload):
                    migrated += 1
            except Exception as exc:
                logger.debug("Could not migrate Hub manifest %s: %s", path, exc)
    except Exception as exc:
        logger.debug("Could not enumerate Hub manifests for migration: %s", exc)
    return migrated


def verify_against_disk(manifest: Manifest, snapshot_dir: Path) -> VerifyResult:
    """Check every expected file is present in *snapshot_dir* at its declared size.

    Presence + size only, not content integrity: it converts a
    no-op-on-cached ``snapshot_download`` into a clear error when shards are
    missing or truncated, and marks a scanner row partial when expected bytes
    aren't on disk. Byte-level integrity is already covered upstream by
    ``huggingface_hub`` (size check on HTTP, content-addressed chunk hashes on
    XET), so re-hashing finalized multi-GB weights here would only duplicate
    that at a large cost. ``Path.stat()`` follows symlinks, so HF's symlink and
    Windows copy cache layouts both verify correctly.
    """
    missing: list[str] = []
    mismatched: list[str] = []
    for expected in manifest.expected_files:
        if not expected_path_is_safe(expected.path):
            missing.append(expected.path)
            continue
        target = snapshot_dir / expected.path
        try:
            actual_size = target.stat().st_size
        except OSError:
            missing.append(expected.path)
            continue
        # expected.size == 0 means HF metadata had no declared size: verify existence only.
        if expected.size > 0 and actual_size != expected.size:
            mismatched.append(expected.path)
    return VerifyResult(
        ok = not missing and not mismatched,
        missing = tuple(missing),
        size_mismatched = tuple(mismatched),
    )


def expected_files_from_snapshot_dir(snapshot_dir: Path) -> list[ExpectedFile]:
    """Derive expected-file entries from a completed snapshot directory.

    Last-resort manifest source for when HF metadata was unreachable for the
    whole download. ``snapshot_download`` has already exited cleanly, so every
    regular file is a finished, correctly-sized blob; recording them keeps the
    scanner's completion check in agreement with the worker's exit-0 success
    instead of leaving a finished repo perpetually partial. ``stat()`` follows
    HF's symlink layout and Windows copies, so the recorded sizes match what
    ``verify_against_disk`` later reads.
    """
    out: list[ExpectedFile] = []
    try:
        entries = sorted(snapshot_dir.rglob("*"))
    except OSError:
        return out
    for path in entries:
        try:
            if not path.is_file():
                continue
            relative = path.relative_to(snapshot_dir).as_posix()
            out.append(
                ExpectedFile(
                    path = relative,
                    size = path.stat().st_size,
                    sha256 = None,
                )
            )
        except OSError:
            continue
    return out


def write_cancel_marker(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    transport: Optional[str] = None,
    *,
    hub_cache: Optional[str | Path] = None,
) -> bool:
    """Record that this triple was cancelled. Idempotent across repeated cancels.

    ``transport`` ("http"/"xet") is surfaced via partial_transport on
    inventory rows so the UI labels HTTP retries as continuable and XET
    retries as full redownloads. None is accepted for forward-compat.
    """
    recorded_hub_cache = _canonical_hub_cache(hub_cache)
    path = marker_path(
        repo_type,
        repo_id,
        variant,
        hub_cache = recorded_hub_cache,
    )
    if path is None:
        return False
    payload = {
        "version": _MARKER_VERSION,
        "repo_type": repo_type,
        "repo_id": repo_id,
        "variant": variant,
        "transport": transport,
        "cancelled_at": datetime.now(timezone.utc).isoformat(),
        "hub_cache": recorded_hub_cache,
    }
    return _atomic_write_json(path, payload)


def read_cancel_marker_transport(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    *,
    hub_cache: Optional[str | Path] = None,
) -> Optional[str]:
    """Return the transport recorded in the cancel marker, or ``None`` if no
    marker exists or it is unreadable.

    Cases:

    * No marker on disk → ``None``.
    * Legacy v1 marker → ``"http"``: v1 markers were only written by the
      HTTP path, so the transport is unambiguous despite the absent field.
    * v2 marker with a valid ``"http"`` / ``"xet"`` transport → that value.
    * Corrupt, non-dict, or v2-with-missing-transport marker → ``None``.
      Defaulting these to ``"http"`` misled the UI into showing a
      byte-resume "Continue" label for what may have been an XET cancel;
      ``None`` keeps the neutral "Retry" label.
    * Unknown future versions → ``None`` (unknown layout, unknown transport).
    """
    path = _state_read_path(
        marker_path,
        repo_type,
        repo_id,
        variant,
        hub_cache,
    )
    if path is None or not path.is_file():
        return None
    data = _read_state_payload(path)
    if data is None:
        return None
    version = data.get("version")
    if version == _LEGACY_MARKER_VERSION:
        return "http"
    if version != _MARKER_VERSION:
        return None
    transport = data.get("transport")
    if isinstance(transport, str) and transport in ("http", "xet"):
        return transport
    return None


def _all_matching_state_paths(
    parent: Optional[Path], repo_type: RepoType, repo_id: str, variant: Optional[str]
) -> tuple[Path, ...]:
    if parent is None:
        return ()
    legacy_path = (
        manifest_path(repo_type, repo_id, variant)
        if parent.name == "manifests"
        else marker_path(repo_type, repo_id, variant)
    )
    if legacy_path is None:
        return ()
    try:
        return tuple(path for path in parent.rglob(legacy_path.name) if path.is_file())
    except OSError:
        return ()


def clear_cancel_marker(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    *,
    hub_cache: Optional[str | Path] = None,
) -> None:
    """Remove the cancel marker for this triple if present.

    Idempotent: a missing marker is not an error. Called at
    download-start (a fresh attempt supersedes prior cancel state) and
    again at successful completion (cleans up if the start clear failed).
    """
    requested = _canonical_hub_cache(hub_cache)
    path = marker_path(
        repo_type,
        repo_id,
        variant,
        hub_cache = requested,
    )
    legacy = marker_path(repo_type, repo_id, variant)
    paths = [path]
    if (
        legacy is not None
        and legacy != path
        and _legacy_state_applies(legacy, requested, fail_closed = True)
    ):
        paths.append(legacy)
    for target in paths:
        if target is None:
            continue
        try:
            target.unlink(missing_ok = True)
        except OSError as exc:
            logger.debug("Could not clear cancel marker %s: %s", target, exc)


def has_cancel_marker(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    *,
    hub_cache: Optional[str | Path] = None,
) -> bool:
    """Return whether a cancel marker applies to the selected cache."""
    path = _state_read_path(
        marker_path,
        repo_type,
        repo_id,
        variant,
        hub_cache,
        fail_closed = True,
    )
    try:
        return path is not None and path.is_file()
    except OSError:
        return False


def delete_manifest(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    *,
    hub_cache: Optional[str | Path] = None,
) -> bool:
    requested = _canonical_hub_cache(hub_cache)
    path = manifest_path(
        repo_type,
        repo_id,
        variant,
        hub_cache = requested,
    )
    legacy = manifest_path(repo_type, repo_id, variant)
    paths = [path]
    if legacy is not None and legacy != path and _legacy_state_applies(legacy, requested):
        paths.append(legacy)
    removed = False
    for target in paths:
        if target is None:
            continue
        try:
            if target.is_file():
                target.unlink()
                removed = True
        except OSError as exc:
            logger.debug("Could not delete manifest %s: %s", target, exc)
    return removed


def purge_state(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    *,
    hub_cache: Optional[str | Path] = None,
) -> bool:
    """Remove manifest + cancel marker for this triple. Returns ``True``
    when anything was present on disk before the call. Idempotent.

    With ``hub_cache`` set, only that cache's scoped state (plus any legacy
    unscoped file that belongs to it) is removed, so purging one cache's copy
    never clears another cache's resumable/cancel state."""
    if hub_cache is None:
        paths = (
            *_all_matching_state_paths(manifests_dir(), repo_type, repo_id, variant),
            *_all_matching_state_paths(cancelled_dir(), repo_type, repo_id, variant),
        )
    else:
        requested = _canonical_hub_cache(hub_cache)
        candidates = [
            manifest_path(repo_type, repo_id, variant, hub_cache = hub_cache),
            marker_path(repo_type, repo_id, variant, hub_cache = hub_cache),
        ]
        # Legacy unscoped state is shared: an unowned file belongs to the active cache, so only purge
        # it when it belongs to the cache being deleted, else deleting an inactive cache erases the
        # active cache's resume/cancel state.
        for path_factory in (manifest_path, marker_path):
            legacy = path_factory(repo_type, repo_id, variant)
            if legacy is not None and _legacy_state_applies(legacy, requested):
                candidates.append(legacy)
        paths = tuple(p for p in candidates if p is not None)
    removed = False
    for path in paths:
        try:
            if path.is_file():
                path.unlink()
                removed = True
        except OSError as exc:
            logger.debug("Could not purge Hub state %s: %s", path, exc)
    return removed


def purge_all_state_for_repo(
    repo_type: RepoType,
    repo_id: str,
    *,
    hub_cache: Optional[str | Path] = None,
) -> int:
    """Remove the snapshot-level manifest + marker AND every variant-keyed
    manifest + marker for this repo. Used by the route delete handlers so
    scanner state never outlives the cache it described. Returns the count
    of (repo, variant) triples that had any state on disk.

    With ``hub_cache`` set, only that cache's scoped state (plus any legacy
    unscoped file) is enumerated and removed, so deleting one cache's copy does
    not clear another cache's resumable/cancel state."""
    removed = 0
    if purge_state(repo_type, repo_id, None, hub_cache = hub_cache):
        removed += 1
    variants: set[str] = set()
    prefix = variant_filename_prefix(repo_type, repo_id)
    if hub_cache is None:
        search = [(p, True) for p in (manifests_dir(), cancelled_dir()) if p is not None]
    else:
        # This cache's scoped dir plus the legacy unscoped base; glob (not rglob) so other caches are untouched.
        search = []
        for scoped, base in (
            (manifest_path(repo_type, repo_id, None, hub_cache = hub_cache), manifests_dir()),
            (marker_path(repo_type, repo_id, None, hub_cache = hub_cache), cancelled_dir()),
        ):
            if scoped is not None:
                search.append((scoped.parent, False))
            if base is not None:
                search.append((base, False))
    for parent, recursive in search:
        try:
            entries = tuple(
                parent.rglob(f"{prefix}*.json") if recursive else parent.glob(f"{prefix}*.json")
            )
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            fallback = entry.stem[len(prefix) :]
            variants.add(fallback)
    for variant in variants:
        if purge_state(repo_type, repo_id, variant, hub_cache = hub_cache):
            removed += 1
    return removed


def _variant_from_state_file(path: Path, fallback: str) -> str:
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return fallback
    if not isinstance(data, dict):
        return fallback
    variant = data.get("variant")
    return variant if isinstance(variant, str) and variant else fallback


def _iter_variant_state_files(
    parent: Optional[Path],
    repo_type: RepoType,
    repo_id: str,
    hub_cache: Optional[str | Path],
    *,
    cancel_markers: bool,
) -> Iterator[tuple[str, Path]]:
    if parent is None:
        return
    path_factory = marker_path if cancel_markers else manifest_path
    requested = _canonical_hub_cache(hub_cache)
    scoped_probe = path_factory(
        repo_type,
        repo_id,
        None,
        hub_cache = requested,
    )
    if scoped_probe is None:
        return
    prefix = variant_filename_prefix(repo_type, repo_id)
    seen: set[str] = set()
    for directory, legacy in ((scoped_probe.parent, False), (parent, True)):
        if legacy and directory == scoped_probe.parent:
            continue
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(".json"):
                continue
            stem = entry.name[: -len(".json")]
            if not stem.lower().startswith(prefix) or entry.name in seen:
                continue
            if legacy and not _legacy_state_applies(
                entry,
                requested,
                fail_closed = cancel_markers,
            ):
                continue
            fallback = stem[len(prefix) :]
            if fallback:
                seen.add(entry.name)
                yield _variant_from_state_file(entry, fallback), entry


def iter_variant_manifests(
    repo_type: RepoType,
    repo_id: str,
    *,
    hub_cache: Optional[str | Path] = None,
) -> Iterator[tuple[str, Path]]:
    """Yield (variant, manifest_path) for every variant-keyed manifest
    written for this repo. Used by is_gguf_repo_partial to enumerate all
    variants present on disk so the all-variants-broken gate can run."""
    yield from _iter_variant_state_files(
        manifests_dir(),
        repo_type,
        repo_id,
        hub_cache,
        cancel_markers = False,
    )


def iter_variant_markers(
    repo_type: RepoType,
    repo_id: str,
    *,
    hub_cache: Optional[str | Path] = None,
) -> Iterator[tuple[str, Path]]:
    """Yield (variant, marker_path) for every variant-keyed cancel marker.
    Companion to iter_variant_manifests: catches variants cancelled
    before download-start ever wrote a manifest (very early failures)."""
    yield from _iter_variant_state_files(
        cancelled_dir(),
        repo_type,
        repo_id,
        hub_cache,
        cancel_markers = True,
    )
