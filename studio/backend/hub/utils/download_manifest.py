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

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Sequence

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


_MANIFEST_VERSION = 1
_MARKER_VERSION = 2
_LEGACY_MARKER_VERSION = 1

# Verbatim phrase the worker emits on a degraded completion and the download
# lifecycle escalates to a warning log. Shared so the emit and match stay coupled.
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
        return str(Path(hub_cache).expanduser().resolve(strict = False))
    except (OSError, RuntimeError, ValueError):
        return str(hub_cache)


def _read_state_payload(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("Could not read Hub state %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _atomic_write_json(path: Path, payload: dict) -> bool:
    # Per-write uuid suffix so a concurrent caller or a stale tmp from a
    # previous crash cannot collide with the in-flight write.
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
    payload = {
        "version": _MANIFEST_VERSION,
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
    }
    return _atomic_write_json(path, payload)


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

    Forward-compat: accepts only ``version == 1``; an unknown version is
    treated as no manifest. A future v2 schema MUST either keep v1's
    ``expected_files`` shape on the same filename (bump
    ``_MANIFEST_VERSION`` and widen this check) or live under a different
    filename, so an incompatible payload can never mis-classify rows.
    """
    path = manifest_path(
        repo_type,
        repo_id,
        variant,
        hub_cache = _canonical_hub_cache(hub_cache),
    )
    if path is None or not path.is_file():
        return None
    data = _read_state_payload(path)
    if data is None:
        return None
    if data.get("version") != _MANIFEST_VERSION:
        logger.debug(
            "Manifest %s has unknown version %r; ignoring.",
            path,
            data.get("version"),
        )
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
        if not isinstance(file_path, str) or not isinstance(size, int):
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
    return Manifest(
        repo_type = repo_type,
        repo_id = str(data.get("repo_id", repo_id)),
        variant = raw_variant if raw_variant else None,
        started_at = str(data.get("started_at", "")),
        expected_files = tuple(expected),
        transport = transport if transport in ("http", "xet") else None,
        hub_cache = data.get("hub_cache") if isinstance(data.get("hub_cache"), str) else None,
    )


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
        target = snapshot_dir / expected.path
        try:
            actual_size = target.stat().st_size
        except OSError:
            missing.append(expected.path)
            continue
        # expected.size == 0 means HF metadata had no declared size: verify
        # existence only rather than flagging every such file as mismatched.
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
    path = marker_path(
        repo_type,
        repo_id,
        variant,
        hub_cache = _canonical_hub_cache(hub_cache),
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
    parent: Optional[Path],
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str],
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
    path = marker_path(
        repo_type,
        repo_id,
        variant,
        hub_cache = _canonical_hub_cache(hub_cache),
    )
    if path is None:
        return
    try:
        path.unlink(missing_ok = True)
    except OSError as exc:
        logger.debug("Could not clear cancel marker %s: %s", path, exc)


def has_cancel_marker(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    *,
    hub_cache: Optional[str | Path] = None,
) -> bool:
    """Return whether a cancel marker applies to the selected cache."""
    path = marker_path(
        repo_type,
        repo_id,
        variant,
        hub_cache = _canonical_hub_cache(hub_cache),
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
    path = manifest_path(
        repo_type,
        repo_id,
        variant,
        hub_cache = _canonical_hub_cache(hub_cache),
    )
    if path is None:
        return False
    try:
        if not path.is_file():
            return False
        path.unlink()
        return True
    except OSError as exc:
        logger.debug("Could not delete manifest %s: %s", path, exc)
        return False


def purge_state(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
) -> bool:
    """Remove manifest + cancel marker for this triple. Returns ``True``
    when anything was present on disk before the call. Idempotent."""
    paths = (
        *_all_matching_state_paths(manifests_dir(), repo_type, repo_id, variant),
        *_all_matching_state_paths(cancelled_dir(), repo_type, repo_id, variant),
    )
    removed = False
    for path in paths:
        try:
            if path.is_file():
                path.unlink()
                removed = True
        except OSError as exc:
            logger.debug("Could not purge Hub state %s: %s", path, exc)
    return removed


def purge_all_state_for_repo(repo_type: RepoType, repo_id: str) -> int:
    """Remove the snapshot-level manifest + marker AND every variant-keyed
    manifest + marker for this repo. Used by the route delete handlers so
    scanner state never outlives the cache it described. Returns the count
    of (repo, variant) triples that had any state on disk."""
    removed = 0
    if purge_state(repo_type, repo_id, None):
        removed += 1
    variants: set[str] = set()
    for parent in (manifests_dir(), cancelled_dir()):
        if parent is None:
            continue
        prefix = variant_filename_prefix(repo_type, repo_id)
        try:
            entries = tuple(parent.rglob(f"{prefix}*.json"))
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            fallback = entry.stem[len(prefix) :]
            variants.add(_variant_from_state_file(entry, fallback))
    for variant in variants:
        if purge_state(repo_type, repo_id, variant):
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
    scoped_probe = path_factory(
        repo_type,
        repo_id,
        None,
        hub_cache = _canonical_hub_cache(hub_cache),
    )
    if scoped_probe is None:
        return
    prefix = variant_filename_prefix(repo_type, repo_id)
    try:
        entries = list(scoped_probe.parent.iterdir())
    except OSError:
        return
    for entry in entries:
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue
        stem = entry.name[: -len(".json")]
        if not stem.lower().startswith(prefix):
            continue
        fallback = stem[len(prefix) :]
        if fallback:
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
