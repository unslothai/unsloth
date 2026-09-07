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
from typing import Iterable, Iterator, Optional, Sequence
from urllib.parse import unquote

from loggers import get_logger

from hub.utils.state_dir import (
    RepoType,
    cache_scope_names,
    cancelled_dir,
    manifest_path,
    manifests_dir,
    marker_path,
    normalize_hub_cache,
    state_filename_is_ambiguous,
    variant_filename_prefix,
    variant_key_fragments,
)
from utils.paths.path_utils import drop_appledouble_metadata

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


class VariantState:
    def __init__(
        self,
        manifests: Optional[dict[str, tuple[str, Optional[Manifest]]]] = None,
        markers: Optional[dict[str, str]] = None,
    ) -> None:
        self._manifests = manifests or {}
        self._markers = markers or {}

    def manifests(self) -> Iterator[tuple[str, Optional[Manifest]]]:
        yield from self._manifests.values()

    def marker_variants(self) -> Iterator[str]:
        yield from self._markers.values()

    def manifest_for(self, variant: str) -> Optional[Manifest]:
        entry = self._manifests.get(variant.lower())
        return entry[1] if entry is not None else None

    def has_marker(self, variant: str) -> bool:
        if variant.lower() in self._markers:
            return True
        # An unreadable marker loses its payload, so its only identity is the filename, which for a
        # variant stored hashed is the digest rather than the variant: a plain lookup missed it and left a
        # cancelled variant advertised as complete.
        try:
            fragments = variant_key_fragments(variant)
        except (UnicodeError, ValueError):
            return False
        return any(fragment in self._markers for fragment in fragments)

    def summary(self) -> tuple[bool, int]:
        return bool(self._manifests or self._markers), sum(
            sum(max(0, int(file.size or 0)) for file in manifest.expected_files)
            for _variant, manifest in self._manifests.values()
            if manifest is not None
        )


class VariantStateIndex:
    def __init__(self, states: dict[tuple[str, RepoType, str], VariantState]) -> None:
        self._states = states

    def for_repo(self, repo_type: RepoType, repo_id: str, *, hub_cache: str | Path) -> VariantState:
        cache = _canonical_hub_cache(hub_cache)
        if cache is None:
            return VariantState()
        return self._states.get((cache, repo_type, repo_id.lower()), VariantState())


@dataclass(frozen = True)
class VerifyResult:
    ok: bool
    missing: tuple[str, ...]
    size_mismatched: tuple[str, ...]


def _hub_cache_spellings(
    hub_cache: Optional[str | Path] = None,
) -> tuple[Optional[str], Optional[str | Path]]:
    """``(canonical, as supplied)`` for one cache path, defaulting to the active cache.

    The canonical half is what ownership comparisons and the scope digest are
    built from. The raw half exists only so a reader can also probe the digest
    of the spelling it was handed: the two differ exactly when ``resolve``
    changes the path, and state written while ``resolve`` was unavailable sits
    under the raw one.
    """
    if hub_cache is None:
        try:
            from utils.hf_cache_settings import get_hf_cache_paths
            hub_cache = get_hf_cache_paths().hub_cache
        except Exception:
            return None, None
    # Shared with state_dir.cache_scope_name so the ownership string recorded in a payload and the
    # cache-<digest> directory it is filed under can never come from two different normalizations.
    return normalize_hub_cache(hub_cache), hub_cache


def _canonical_hub_cache(hub_cache: Optional[str | Path] = None) -> Optional[str]:
    return _hub_cache_spellings(hub_cache)[0]


def _scope_spellings(hub_cache: str | Path) -> tuple[str, ...]:
    """Every scope dir this cache's state can sit in, for a caller whose own
    spelling may already be resolved.

    ``cache_scope_names`` recovers the pre-``resolve`` digest only from an
    unresolved path: hand it one that resolves to itself and it returns the
    canonical digest alone. The read path is fed the raw configured setting and
    so gets both, but the delete and the inventory index are not -- every
    production caller of ``purge_all_state_for_repo`` passes
    ``resolve_delete_target_root``, whose every branch resolves, and
    ``build_variant_state_index``'s inventory callers pass a directory derived
    from ``huggingface_hub.scan_cache_dir``, which resolves too. Left alone,
    reads probe two scopes while deletes clear one: a purged variant stays on
    disk under the legacy digest for the next read to resurrect, and the two
    inventory endpoints disagree about state the progress endpoint can see.

    So also probe the configured cache's own spelling -- but only when it names
    the same directory as the path we were handed. Without that guard, deleting
    a repo out of a NON-active cache would sweep the active cache's state for
    the same repo, which is a far worse failure than the one being fixed.
    """
    scopes = list(cache_scope_names(hub_cache))
    canonical = normalize_hub_cache(hub_cache)
    try:
        from utils.hf_cache_settings import get_hf_cache_paths
        configured = get_hf_cache_paths().hub_cache
    except Exception:
        return tuple(scopes)
    if configured is None or normalize_hub_cache(configured) != canonical:
        return tuple(scopes)
    for scope in cache_scope_names(configured):
        if scope not in scopes:
            scopes.append(scope)
    return tuple(scopes)


def _read_state_payload(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    # The decoder raises RecursionError for adversarially deep JSON, and one corrupt state file must not
    # abort the shared one-pass index.
    except (OSError, ValueError, RecursionError) as exc:
        logger.debug("Could not read Hub state %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _manifest_from_payload(
    data: Optional[dict], repo_type: RepoType, repo_id: str
) -> Optional[Manifest]:
    if data is None:
        return None
    version = data.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in _SUPPORTED_MANIFEST_VERSIONS
    ):
        return None
    payload_repo_id = _payload_text(data.get("repo_id"))
    variant_value = data.get("variant")
    raw_variant = _payload_text(variant_value)
    if variant_value is not None and raw_variant is None:
        return None
    has_metadata_attestation = data.get("metadata_derived") is True
    if version == _MANIFEST_VERSION or has_metadata_attestation:
        if (
            data.get("repo_type") != repo_type
            or payload_repo_id is None
            or payload_repo_id.casefold() != repo_id.casefold()
        ):
            return None
    raw_files = data.get("expected_files")
    if not isinstance(raw_files, list):
        return None
    expected: list[ExpectedFile] = []
    for item in raw_files:
        if not isinstance(item, dict):
            return None
        file_path = _payload_file_path(item.get("path"))
        size = item.get("size")
        if (
            file_path is None
            or not expected_path_is_safe(file_path)
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
    transport = data.get("transport")
    commit_hash = normalized_commit_hash(data.get("commit_hash"))
    metadata_derived = bool(has_metadata_attestation and commit_hash is not None)
    return Manifest(
        repo_type = repo_type,
        repo_id = payload_repo_id or repo_id,
        variant = raw_variant if raw_variant else None,
        started_at = str(data.get("started_at", "")),
        expected_files = tuple(expected),
        transport = transport if transport in ("http", "xet") else None,
        hub_cache = _payload_cache_path(data.get("hub_cache")),
        version = version,
        commit_hash = commit_hash if metadata_derived else None,
        metadata_derived = metadata_derived,
    )


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
        raw_recorded = data.get("hub_cache")
        recorded = _payload_cache_path(raw_recorded)
        if recorded:
            return _canonical_hub_cache(recorded) == requested_hub_cache
        if "hub_cache" in data and raw_recorded not in (None, ""):
            return fail_closed and requested_hub_cache == _canonical_hub_cache()
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
    def applies(path: Path) -> bool:
        payload = _read_state_payload(path)
        plausible = _state_payload_identity_matches_entry(path, payload, repo_type)
        # Cancellation stays fail-closed when a marker cannot identify its owner; parseable state from
        # another repository is never borrowed.
        if fail_closed and not plausible:
            return True
        if plausible and variant is not None:
            recorded_variant = _payload_text(payload.get("variant"))
            if recorded_variant is None:
                return fail_closed
            if recorded_variant.strip().lower() != variant.strip().lower():
                return False
        return _state_entry_belongs_to_repo(path, payload, repo_type, repo_id, variant)

    requested, raw = _hub_cache_spellings(hub_cache)
    scoped = _state_paths(
        path_factory,
        repo_type,
        repo_id,
        variant,
        requested,
        raw_hub_cache = raw,
    )
    for path in scoped:
        try:
            if path.is_file() and applies(path):
                return path
        except OSError:
            continue
    for path in _state_paths(path_factory, repo_type, repo_id, variant, None):
        if path in scoped:
            continue
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        if _legacy_state_applies(path, requested, fail_closed = fail_closed) and applies(path):
            return path
    return None


def _state_paths(
    path_factory,
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str],
    hub_cache: Optional[str | Path],
    *,
    raw_hub_cache: Optional[str | Path] = None,
) -> tuple[Path, ...]:
    """Canonical-first read/delete paths across the filename migration.

    Writers use only the default state-dir path. Readers and cleanup also probe
    the prior repository and double-hyphen variant encodings, plus the
    pre-``resolve`` cache-scope digest, deduplicating when a repository, variant
    or cache scope never needed migration.

    ``raw_hub_cache`` is the caller's own spelling of ``hub_cache`` before
    canonicalization, and is what the extra scope digest has to come from: the
    canonical string resolves to itself, so deriving the fallback from it would
    reproduce the canonical digest and probe nothing. The scope fan-out
    collapses to one entry whenever the two spellings agree, which is every path
    that resolves to itself.
    """
    # _scope_spellings, not cache_scope_names: the single-variant delete route hands over a root
    # resolve_delete_target_root has ALREADY resolved, so the raw spelling reproduces the canonical
    # digest and the pre-resolve scope is never probed, leaving state for the next read to resurrect.
    # It adds the configured cache's own spelling only when it names the same directory.
    scopes = (
        (None,)
        if hub_cache is None
        else _scope_spellings(raw_hub_cache if raw_hub_cache is not None else hub_cache)
    )
    paths = [
        path_factory(
            repo_type,
            repo_id,
            variant,
            hub_cache = hub_cache,
            create = False,
            legacy_variant_key = legacy_variant,
            legacy_repo_key = legacy_repo,
            legacy_hash_key = legacy_hash,
            cache_scope = scope,
        )
        for scope in scopes
        for legacy_repo in (False, True)
        for legacy_variant in ((False, True) if variant is not None else (False,))
        for legacy_hash in (False, True)
    ]
    return tuple(path for path in dict.fromkeys(paths) if path is not None)


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


def _unlink_state_paths(paths: Iterable[Optional[Path]]) -> bool:
    removed = False
    for path in dict.fromkeys(paths):
        if path is None:
            continue
        try:
            path.unlink()
            removed = True
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("Could not remove Hub state %s: %s", path, exc)
    return removed


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
        create = True,
    )
    if path is None:
        return False
    normalized_commit = normalized_commit_hash(commit_hash)
    metadata_attestation = bool(metadata_derived and normalized_commit)
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
    if path is None:
        return None
    data = _read_state_payload(path)
    if data is None:
        return None
    payload_identity = _state_payload_identity(data)
    if payload_identity is None or not _state_payload_identity_matches_entry(path, data, repo_type):
        return None
    recorded_type, recorded_id = payload_identity
    if recorded_id != repo_id.lower() or recorded_type not in (None, repo_type):
        return None
    manifest = _manifest_from_payload(data, repo_type, repo_id)
    if manifest is None:
        return None
    recorded_variant = (
        manifest.variant.strip().casefold()
        if manifest.variant and manifest.variant.strip()
        else None
    )
    requested_variant = variant.strip().casefold() if variant and variant.strip() else None
    return manifest if recorded_variant == requested_variant else None


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
                            isinstance(recorded_hub_cache, str) and bool(recorded_hub_cache.strip())
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
        # Counted missing rather than skipped, so an unverifiable entry can never read as verified. A
        # Windows-separator path cannot be folded onto its posix spelling in expected_path_is_safe: that
        # guard also fronts resolved_dataset_snapshot_file, which splits on PurePosixPath, where "a\b" is
        # one component and accepting it would hand a traversal through. No writer here produces one (HF
        # rfilenames are posix and expected_files_from_snapshot_dir calls as_posix).
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
        # Baking a companion into the completion contract makes the download read as partial forever once
        # macOS cleans it up.
        entries = drop_appledouble_metadata(sorted(snapshot_dir.rglob("*")))
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
    inventory rows so the UI only offers a byte-resume for an HTTP partial.
    None is accepted for forward-compat.
    """
    recorded_hub_cache = _canonical_hub_cache(hub_cache)
    path = marker_path(
        repo_type,
        repo_id,
        variant,
        hub_cache = recorded_hub_cache,
        create = True,
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
    if path is None:
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
    path_factory = manifest_path if parent.name == "manifests" else marker_path
    probes = _state_paths(path_factory, repo_type, repo_id, variant, None)
    if not probes:
        return ()
    try:
        return tuple(
            path for probe in probes for path in parent.rglob(probe.name) if path.is_file()
        )
    except OSError:
        return ()


def _owned_state_paths(
    path_factory,
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str],
    requested: Optional[str],
    *,
    fail_closed: bool,
    raw_hub_cache: Optional[str | Path] = None,
) -> list[Path]:
    scoped = list(
        _state_paths(
            path_factory,
            repo_type,
            repo_id,
            variant,
            requested,
            raw_hub_cache = raw_hub_cache,
        )
    )
    paths = list(scoped)
    for legacy in _state_paths(path_factory, repo_type, repo_id, variant, None):
        if legacy not in scoped and _legacy_state_applies(
            legacy, requested, fail_closed = fail_closed
        ):
            paths.append(legacy)
    return paths


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
    requested, raw = _hub_cache_spellings(hub_cache)
    paths = _owned_state_paths(
        marker_path,
        repo_type,
        repo_id,
        variant,
        requested,
        fail_closed = True,
        raw_hub_cache = raw,
    )
    paths = [
        candidate
        for candidate in paths
        if candidate is not None
        and _state_entry_belongs_to_repo(
            candidate,
            _read_state_payload(candidate),
            repo_type,
            repo_id,
            variant,
        )
    ]
    _unlink_state_paths(paths)


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
    return path is not None


def delete_manifest(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    *,
    hub_cache: Optional[str | Path] = None,
) -> bool:
    requested, raw = _hub_cache_spellings(hub_cache)
    paths = _owned_state_paths(
        manifest_path,
        repo_type,
        repo_id,
        variant,
        requested,
        fail_closed = False,
        raw_hub_cache = raw,
    )
    paths = [
        candidate
        for candidate in paths
        if candidate is not None
        and _state_entry_belongs_to_repo(
            candidate,
            _read_state_payload(candidate),
            repo_type,
            repo_id,
            variant,
        )
    ]
    return _unlink_state_paths(paths)


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
        requested, raw = _hub_cache_spellings(hub_cache)
        candidates: list[Path] = []
        # Legacy unscoped state is shared: an unowned file belongs to the active cache, so purge it only
        # when it belongs to the cache being deleted, else deleting an inactive cache erases the active
        # cache's state.
        for path_factory, fail_closed in (
            (manifest_path, False),
            (marker_path, True),
        ):
            candidates.extend(
                _owned_state_paths(
                    path_factory,
                    repo_type,
                    repo_id,
                    variant,
                    requested,
                    fail_closed = fail_closed,
                    raw_hub_cache = raw,
                )
            )
        paths = tuple(dict.fromkeys(candidates))
    paths = tuple(
        path
        for path in paths
        if _state_entry_belongs_to_repo(
            path,
            _read_state_payload(path),
            repo_type,
            repo_id,
            variant,
        )
    )
    return _unlink_state_paths(paths)


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
    not clear another cache's resumable/cancel state.

    Variant entries are unlinked by their enumerated paths, never by
    reconstructing the paired manifest/marker name. The legacy filename scheme
    is not injective when repository IDs or variants contain ``variant``
    delimiters, so a reconstructed counterpart can belong to another repo.
    Valid payload identity decides ownership. Corrupt delimiter-ambiguous state
    is retained as a safe orphan instead of risking an active download marker."""
    removed = 0
    if purge_state(repo_type, repo_id, None, hub_cache = hub_cache):
        removed += 1
    variant_paths: dict[str, set[Path]] = {}
    prefixes = tuple(
        dict.fromkeys(
            variant_filename_prefix(
                repo_type,
                repo_id,
                legacy_repo_key = legacy_repo,
                legacy_hash_key = legacy_hash,
            )
            for legacy_repo in (False, True)
            for legacy_hash in (False, True)
        )
    )
    requested = _canonical_hub_cache(hub_cache)
    if hub_cache is None:
        search = [
            (parent, True, False, cancel_markers)
            for parent, cancel_markers in (
                (manifests_dir(), False),
                (cancelled_dir(), True),
            )
            if parent is not None
        ]
    else:
        # This cache's scoped dirs plus the legacy unscoped base; glob (not rglob) so other caches are untouched.
        # Both scope spellings, else a repo delete leaves the pre-resolve copy behind for a later read to
        # resurrect as state for a cache that is gone.
        search = []
        # _scope_spellings, not cache_scope_names: every production caller resolves its target root first,
        # so the raw-spelling probe would be inert here while the read path still has it.
        scopes = _scope_spellings(hub_cache)
        for path_factory, base, cancel_markers in (
            (manifest_path, manifests_dir(create = False), False),
            (marker_path, cancelled_dir(create = False), True),
        ):
            for scope in scopes:
                scoped = path_factory(
                    repo_type,
                    repo_id,
                    None,
                    hub_cache = hub_cache,
                    create = False,
                    cache_scope = scope,
                )
                if scoped is not None:
                    search.append((scoped.parent, False, False, cancel_markers))
            if base is not None:
                search.append((base, False, True, cancel_markers))
    for parent, recursive, legacy, cancel_markers in search:
        try:
            entries = tuple(
                dict.fromkeys(
                    entry
                    for prefix in prefixes
                    for entry in (
                        parent.rglob(f"{prefix}*.json")
                        if recursive
                        else parent.glob(f"{prefix}*.json")
                    )
                )
            )
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            if legacy and not _legacy_state_applies(
                entry,
                requested,
                fail_closed = cancel_markers,
            ):
                continue
            prefix = next(
                candidate for candidate in prefixes if entry.stem.lower().startswith(candidate)
            )
            fallback = entry.stem[len(prefix) :]
            payload = _read_state_payload(entry)
            if not _state_entry_belongs_to_repo(entry, payload, repo_type, repo_id):
                continue
            variant = _variant_from_state_payload(payload, fallback)
            variant_paths.setdefault(variant.lower(), set()).add(entry)
    for paths in variant_paths.values():
        if _unlink_state_paths(paths):
            removed += 1
    return removed


def _variant_from_state_payload(data: Optional[dict], fallback: str) -> str:
    if data is None:
        return fallback
    variant = _payload_text(data.get("variant"))
    return variant if variant else fallback


def _payload_text(value: object) -> Optional[str]:
    """Return only text safe for the filename hashing and path helpers."""
    if not isinstance(value, str):
        return None
    try:
        value.encode("utf-8")
    except UnicodeError:
        return None
    return value


def _payload_file_path(value: object) -> Optional[str]:
    """Accept only path text that Python filesystem APIs can consume.

    Rejecting the whole malformed manifest preserves its fail-open contract.
    """
    text = _payload_text(value)
    if text is None or "\0" in text or text in ("", "."):
        return None
    posix, windows = PurePosixPath(text), PureWindowsPath(text)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or windows.root
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        return None
    return text


def _payload_cache_path(value: object) -> Optional[str]:
    """Return cache ownership only when it is safe for path normalization."""
    text = _payload_text(value)
    if text is None or "\0" in text:
        return None
    if not (PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute()):
        return None
    return text


def _state_payload_identity(payload: Optional[dict]) -> Optional[tuple[Optional[RepoType], str]]:
    if payload is None:
        return None
    recorded_type = payload.get("repo_type")
    recorded_id = _payload_text(payload.get("repo_id"))
    if recorded_id is None:
        return None
    repo_type = recorded_type if recorded_type in ("model", "dataset") else None
    return repo_type, recorded_id.lower()


def _state_payload_identity_matches_entry(
    entry: Path, payload: Optional[dict], fallback_repo_type: RepoType
) -> bool:
    """Whether payload ownership could have generated this exact state name.

    False means corrupt ownership, not automatically foreign ownership; callers
    retain fail-closed or ambiguity-safe behavior according to their operation.
    """
    identity = _state_payload_identity(payload)
    if identity is None:
        return False
    recorded_type, recorded_id = identity
    recorded_variant = _payload_text(payload.get("variant")) if payload is not None else None
    expected = _state_paths(
        marker_path,
        recorded_type or fallback_repo_type,
        recorded_id,
        recorded_variant,
        None,
    )
    return any(path.name == entry.name for path in expected)


def _state_entry_belongs_to_repo(
    entry: Path,
    payload: Optional[dict],
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
) -> bool:
    """Attribute an exact state path without guessing across legacy delimiter
    collisions; unreadable ambiguous names are retained rather than deleted."""
    payload_identity = _state_payload_identity(payload)
    # A parseable payload is authoritative even when its filename has multiple splits.
    if payload_identity is not None and _state_payload_identity_matches_entry(
        entry, payload, repo_type
    ):
        recorded_type, recorded_id = payload_identity
        if recorded_id != repo_id.lower() or recorded_type not in (None, repo_type):
            return False
        if variant is None:
            return True
        recorded_variant = _payload_text(payload.get("variant"))
        return recorded_variant is not None and (
            recorded_variant.strip().lower() == variant.strip().lower()
        )
    return not state_filename_is_ambiguous(entry.name)


def _state_file_records(
    entry: Path, repo_keys: dict[str, set[tuple[RepoType, str]]], *, fail_closed: bool
) -> tuple[tuple[RepoType, str, str, Optional[dict]], ...]:
    name = entry.name
    if not name.endswith(".json"):
        return ()
    stem = name[: -len(".json")]
    try:
        if not entry.is_file():
            return ()
    except OSError:
        return ()
    payload = _read_state_payload(entry)
    lower_stem, separator = stem.lower(), "--variant--"
    matches: list[tuple[tuple[RepoType, str], str]] = []
    offset = lower_stem.find(separator)
    while offset >= 0:
        repos = repo_keys.get(stem[:offset].lower(), ())
        fallback = stem[offset + len(separator) :]
        if fallback:
            matches.extend((repo, fallback) for repo in sorted(repos))
        offset = lower_stem.find(separator, offset + 1)
    if not matches:
        return ()
    payload_identity = _state_payload_identity(payload)
    payload_identified = False
    if payload_identity is not None:
        if _state_payload_identity_matches_entry(entry, payload, matches[0][0][0]):
            recorded_type, recorded_id = payload_identity
            identified = [
                match
                for match in matches
                if match[0][1] == recorded_id
                and (recorded_type is None or match[0][0] == recorded_type)
            ]
            matches = identified
            payload_identified = bool(identified)
        elif not fail_closed:
            matches = []
    elif not fail_closed:
        matches = []
    variant_payload = payload if not fail_closed or payload_identified else None
    return tuple(
        (repo[0], repo[1], _variant_from_state_payload(variant_payload, fallback), payload)
        for repo, fallback in matches
    )


def build_variant_state_index(
    repositories: Iterable[tuple[RepoType, str, str | Path]], *, active_hub_cache: str | Path
) -> VariantStateIndex:
    targets: set[tuple[str, RepoType, str]] = set()
    repo_keys: dict[str, set[tuple[RepoType, str]]] = {}
    caches_by_scope: dict[str, set[str]] = {}
    for repo_type, repo_id, hub_cache in repositories:
        canonical_cache = _canonical_hub_cache(hub_cache)
        if canonical_cache is None:
            continue
        normalized_repo = repo_id.lower()
        targets.add((canonical_cache, repo_type, normalized_repo))
        for legacy_repo in (False, True):
            for legacy_hash in (False, True):
                prefix = variant_filename_prefix(
                    repo_type,
                    repo_id,
                    legacy_repo_key = legacy_repo,
                    legacy_hash_key = legacy_hash,
                )
                repo_keys.setdefault(prefix[: -len("--variant--")], set()).add(
                    (repo_type, normalized_repo)
                )
        # Both spellings of the scope dir, derived from the caller's own rather than the canonical one, so
        # state filed under the pre-resolve digest is indexed instead of read as "no manifest": the
        # inventory callers arrive with a directory huggingface_hub.scan_cache_dir already resolved, so
        # the raw probe alone finds nothing.
        for scope in _scope_spellings(hub_cache):
            caches_by_scope.setdefault(scope, set()).add(canonical_cache)

    active_cache = _canonical_hub_cache(active_hub_cache)
    mutable: dict[
        tuple[str, RepoType, str],
        tuple[dict[str, tuple[str, Optional[Manifest], tuple[bool, bool]]], dict[str, str]],
    ] = {}

    def add_entry(
        cache: Optional[str],
        record: tuple[RepoType, str, str, Optional[dict]],
        *,
        cancel_marker: bool,
        priority: tuple[bool, bool],
    ) -> None:
        if cache is None:
            return
        repo_type, repo_id, variant, payload = record
        target = (cache, repo_type, repo_id)
        if target not in targets:
            return
        manifests, markers = mutable.setdefault(target, ({}, {}))
        key = variant.lower()
        if cancel_marker:
            markers.setdefault(key, variant)
        elif key not in manifests or priority > manifests[key][2]:
            manifests[key] = (
                variant,
                _manifest_from_payload(payload, repo_type, repo_id),
                priority,
            )

    def add_path(
        cache: Optional[str], entry: Path, record, *, cancel_marker: bool, scoped: bool
    ) -> None:
        repo_type, repo_id, variant, _payload = record
        try:
            canonical = marker_path(repo_type, repo_id, variant, create = False)
        except (UnicodeError, ValueError, OSError):
            # A state filename can carry a byte the filesystem encoding cannot decode, which iterdir()
            # surfaces as a lone surrogate, and hashing it raises UnicodeEncodeError. This index is built once
            # for the whole scan and outside the per-repo try, so letting it escape turns one corrupt filename
            # into a 500 that hides every cached model; treat the entry as non-canonical and keep indexing.
            canonical = None
        add_entry(
            cache,
            record,
            cancel_marker = cancel_marker,
            priority = (scoped, canonical is not None and canonical.name == entry.name),
        )

    def index_directory(parent: Optional[Path], *, cancel_markers: bool) -> None:
        if parent is None:
            return
        try:
            entries = list(parent.iterdir())
        except OSError:
            return
        scoped = {entry.name: entry for entry in entries if entry.name in caches_by_scope}
        for scope, caches in caches_by_scope.items():
            directory = scoped.get(scope)
            if directory is None:
                continue
            try:
                scoped_entries = list(directory.iterdir())
            except OSError:
                continue
            for entry in scoped_entries:
                for record in _state_file_records(entry, repo_keys, fail_closed = cancel_markers):
                    for cache in caches:
                        add_path(
                            cache,
                            entry,
                            record,
                            cancel_marker = cancel_markers,
                            scoped = True,
                        )
        for entry in entries:
            for record in _state_file_records(entry, repo_keys, fail_closed = cancel_markers):
                payload = record[3]
                raw_cache = payload.get("hub_cache") if payload is not None else None
                recorded_cache = _payload_cache_path(raw_cache)
                if recorded_cache:
                    cache = _canonical_hub_cache(recorded_cache)
                elif payload is not None and "hub_cache" in payload and raw_cache not in (None, ""):
                    cache = active_cache if cancel_markers else None
                elif payload is not None or cancel_markers:
                    cache = active_cache
                else:
                    cache = None
                add_path(
                    cache,
                    entry,
                    record,
                    cancel_marker = cancel_markers,
                    scoped = False,
                )

    index_directory(manifests_dir(create = False), cancel_markers = False)
    index_directory(cancelled_dir(create = False), cancel_markers = True)
    return VariantStateIndex(
        {
            key: VariantState(
                {
                    variant: (name, manifest)
                    for variant, (name, manifest, _priority) in manifests.items()
                },
                markers,
            )
            for key, (manifests, markers) in mutable.items()
        }
    )


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
    requested, raw = _hub_cache_spellings(hub_cache)
    # Every scope this cache's state can sit in, so a variant filed under the pre-resolve digest is
    # enumerated instead of reading as "no variant state".
    # _scope_spellings, since cache_scope_names recovers the legacy digest only from an UNRESOLVED path,
    # while a variant request carrying local_path arrives here already resolved.
    scopes = _scope_spellings(raw) if requested is not None and raw is not None else (None,)
    scoped_dirs: list[Path] = []
    for scope in scopes:
        scoped_probe = path_factory(
            repo_type,
            repo_id,
            None,
            hub_cache = requested,
            create = False,
            cache_scope = scope,
        )
        if scoped_probe is None:
            return
        if scoped_probe.parent not in scoped_dirs:
            scoped_dirs.append(scoped_probe.parent)
    prefixes = tuple(
        dict.fromkeys(
            variant_filename_prefix(
                repo_type,
                repo_id,
                legacy_repo_key = legacy_repo,
                legacy_hash_key = legacy_hash,
            )
            for legacy_repo in (False, True)
            for legacy_hash in (False, True)
        )
    )
    selected: dict[str, tuple[tuple[bool, bool], str, Path]] = {}
    for directory, legacy in (*((path, False) for path in scoped_dirs), (parent, True)):
        if legacy and directory in scoped_dirs:
            continue
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.name.endswith(".json"):
                continue
            stem = entry.name[: -len(".json")]
            prefix = next(
                (candidate for candidate in prefixes if stem.lower().startswith(candidate)),
                None,
            )
            if prefix is None:
                continue
            try:
                if not entry.is_file():
                    continue
            except OSError:
                continue
            if legacy and not _legacy_state_applies(
                entry,
                requested,
                fail_closed = cancel_markers,
            ):
                continue
            fallback = stem[len(prefix) :]
            if fallback:
                payload = _read_state_payload(entry)
                payload_identity = _state_payload_identity(payload)
                variant_payload = payload
                if payload_identity is not None:
                    if _state_payload_identity_matches_entry(entry, payload, repo_type):
                        recorded_type, recorded_id = payload_identity
                        if recorded_id != repo_id.lower() or recorded_type not in (None, repo_type):
                            continue
                    elif cancel_markers:
                        variant_payload = None
                    else:
                        continue
                elif cancel_markers:
                    variant_payload = None
                variant = _variant_from_state_payload(variant_payload, fallback)
                try:
                    # Unscoped: only the basename is compared below, and a scoped path would re-derive the digest, and
                    # pay its resolve, once per candidate file.
                    canonical = path_factory(
                        repo_type,
                        repo_id,
                        variant,
                        hub_cache = None,
                        create = False,
                    )
                except (UnicodeError, ValueError, OSError):
                    # Same undecodable-filename case as add_path: letting the hash escape would abort the whole
                    # iteration and lose the repo's valid state alongside it.
                    canonical = None
                priority = (not legacy, canonical is not None and canonical.name == entry.name)
                key = variant.lower()
                if key not in selected or priority > selected[key][0]:
                    selected[key] = (priority, variant, entry)
    for _priority, variant, entry in selected.values():
        yield variant, entry


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
        manifests_dir(create = False),
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
        cancelled_dir(create = False),
        repo_type,
        repo_id,
        hub_cache,
        cancel_markers = True,
    )
