# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Filesystem layout for Hub download state.

State directory sits beside HF's cache (under Unsloth's own cache root)
so it survives ``huggingface-cli delete-cache`` and any other HF-side
cache lifecycle. Two subdirectories:

    <studio cache>/hub-state/
        manifests/cache-<digest>/<key>.json   expected-files manifest
        cancelled/cache-<digest>/<key>.json   cancel marker

The cache digest isolates state for the same repo across selectable Hub caches.
The ``<key>`` mirrors HF's cache dir naming while the resulting manifest,
cancel-marker, and atomic-write temp filenames fit common filesystem basename
limits. Very long repo IDs use a stable hash in the state key:

    models--<owner>--<name>                       full snapshot
    models--<owner>--<name>--variant--<variant>   GGUF variant
    datasets--<owner>--<name>                     dataset snapshot

All path accessors return ``Optional[Path]`` and yield ``None`` when
the directory can't be created (read-only FS, permission error).
Callers must treat ``None`` as "no state available" and fall through
to existing on-disk-only behavior; this module never raises on a
configuration failure.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Literal, Optional, get_args

from loggers import get_logger

from hub.utils.paths import cache_root

logger = get_logger(__name__)


RepoType = Literal["model", "dataset"]

_VALID_REPO_TYPES: tuple[RepoType, ...] = get_args(RepoType)


_HUB_STATE_DIRNAME = "hub-state"
_MANIFESTS_SUBDIR = "manifests"
_CANCELLED_SUBDIR = "cancelled"
_WORKERS_SUBDIR = "workers"
_SAFE_VARIANT_FRAGMENT = re.compile(r"^[a-z0-9._-]{1,64}$")
_MAX_STATE_BASENAME_BYTES = 255
_STATE_EXTENSION = ".json"
# _atomic_write_json writes ".<target>.tmp-<8hex>" beside the final file.
_ATOMIC_WRITE_TMP_OVERHEAD = len(".") + len(".tmp-") + 8
_MAX_VARIANT_FRAGMENT_LENGTH = 64
_CACHE_SCOPE_DIGEST_LENGTH = 32


def state_root() -> Optional[Path]:
    """Return the Hub state root, creating it if needed. ``None`` on failure."""
    root = cache_root() / _HUB_STATE_DIRNAME
    try:
        root.mkdir(parents = True, exist_ok = True)
    except OSError as exc:
        logger.debug("Could not create hub state root %s: %s", root, exc)
        return None
    return root


def _subdir(name: str) -> Optional[Path]:
    root = state_root()
    if root is None:
        return None
    path = root / name
    try:
        path.mkdir(parents = True, exist_ok = True)
    except OSError as exc:
        logger.debug("Could not create hub state subdir %s: %s", path, exc)
        return None
    return path


def repo_cache_basename(repo_type: RepoType, repo_id: str) -> str:
    # Reject a bad repo_type at runtime: a wrong value would silently produce a
    # wrong filename and a misclassified scanner row (the Literal only guards
    # statically; dynamic/JSON-sourced values slip past it).
    if repo_type not in _VALID_REPO_TYPES:
        raise ValueError(f"repo_type must be one of {_VALID_REPO_TYPES}, got {repo_type!r}")
    return f"{repo_type}s--{repo_id.replace('/', '--')}".lower()


def _filename_bytes(name: str) -> int:
    return len(name.encode("utf-8"))


def _state_filename_fits(entry_key: str) -> bool:
    filename = f"{entry_key}{_STATE_EXTENSION}"
    return _filename_bytes(filename) + _ATOMIC_WRITE_TMP_OVERHEAD <= _MAX_STATE_BASENAME_BYTES


def _state_repo_key(repo_type: RepoType, repo_id: str) -> str:
    base = repo_cache_basename(repo_type, repo_id)
    variant_prefix = f"{base}--variant--"
    longest_variant_key = f"{variant_prefix}{'x' * _MAX_VARIANT_FRAGMENT_LENGTH}"
    if _state_filename_fits(longest_variant_key):
        return base
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]
    return f"{repo_type}s--sha256-{digest}"


def variant_filename_prefix(repo_type: RepoType, repo_id: str) -> str:
    """Lowercased prefix every variant-keyed state file for this repo shares.

    The single source the download_manifest enumerators match against, so the
    scheme in :func:`_entry_key` cannot drift from them silently."""
    return f"{_state_repo_key(repo_type, repo_id)}--variant--"


def _entry_key(repo_type: RepoType, repo_id: str, variant: Optional[str]) -> str:
    base = _state_repo_key(repo_type, repo_id)
    if not variant:
        return base
    normalized_variant = variant.strip().lower()
    if _SAFE_VARIANT_FRAGMENT.fullmatch(normalized_variant):
        variant_fragment = normalized_variant
    else:
        digest = hashlib.sha256(normalized_variant.encode("utf-8")).hexdigest()[:32]
        variant_fragment = f"sha256-{digest}"
    return f"{variant_filename_prefix(repo_type, repo_id)}{variant_fragment}"


def _cache_scope(parent: Path, hub_cache: Optional[str | Path]) -> Optional[Path]:
    if hub_cache is None:
        return parent
    normalized = os.path.normcase(str(Path(hub_cache).expanduser()))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:_CACHE_SCOPE_DIGEST_LENGTH]
    scoped = parent / f"cache-{digest}"
    try:
        scoped.mkdir(parents = True, exist_ok = True)
    except OSError as exc:
        logger.debug("Could not create cache-scoped Hub state dir %s: %s", scoped, exc)
        return None
    return scoped


def manifest_path(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    *,
    hub_cache: Optional[str | Path] = None,
) -> Optional[Path]:
    """Path to the manifest file for this triple. May or may not exist."""
    parent = _subdir(_MANIFESTS_SUBDIR)
    if parent is None:
        return None
    parent = _cache_scope(parent, hub_cache)
    if parent is None:
        return None
    return parent / f"{_entry_key(repo_type, repo_id, variant)}.json"


def marker_path(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    *,
    hub_cache: Optional[str | Path] = None,
) -> Optional[Path]:
    """Path to the cancel-marker file for this triple. May or may not exist."""
    parent = _subdir(_CANCELLED_SUBDIR)
    if parent is None:
        return None
    parent = _cache_scope(parent, hub_cache)
    if parent is None:
        return None
    return parent / f"{_entry_key(repo_type, repo_id, variant)}.json"


def manifests_dir() -> Optional[Path]:
    """Manifests subdirectory, created on demand. ``None`` on failure.

    Exposed for iter_variant_manifests, which enumerates the directory to find
    every variant-keyed manifest for a repo (the path helpers above answer
    "where would key X go" but not "what keys exist")."""
    return _subdir(_MANIFESTS_SUBDIR)


def cancelled_dir() -> Optional[Path]:
    """Cancel-marker subdirectory, created on demand. ``None`` on failure.

    See manifests_dir for why this iteration entry point is needed."""
    return _subdir(_CANCELLED_SUBDIR)


def workers_dir() -> Optional[Path]:
    """Worker PID-breadcrumb subdirectory, created on demand. ``None`` on failure.

    Each live download worker drops one breadcrumb here so a backend that
    restarts after a hard crash can reap workers it can no longer reach through
    its in-memory registry."""
    return _subdir(_WORKERS_SUBDIR)
