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

Path accessors are read-only by default. Writers pass ``create=True`` explicitly.
All accessors return ``Optional[Path]`` and yield ``None`` when requested
directory creation fails; callers fall through to existing on-disk-only behavior.
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
_HASH_PREFIXES = ("sha256-", "@sha256-")
_LEGACY_HASH_FRAGMENT = re.compile(r"sha256-[0-9a-f]{32}")
# Either tag _variant_fragment can stamp, so a reader can spot a digest it was
# handed in place of a variant name.
_HASHED_FRAGMENT = re.compile(r"@?sha256-[0-9a-f]{32}")


def state_root(*, create: bool = False) -> Optional[Path]:
    """Return the Hub state root, optionally creating it. ``None`` on failure."""
    root = cache_root() / _HUB_STATE_DIRNAME
    if not create:
        return root
    try:
        root.mkdir(parents = True, exist_ok = True)
    except OSError as exc:
        logger.debug("Could not create hub state root %s: %s", root, exc)
        return None
    return root


def _subdir(name: str, *, create: bool = False) -> Optional[Path]:
    root = state_root(create = create)
    if root is None:
        return None
    path = root / name
    if not create:
        return path
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


def variant_is_hashed_fragment(variant: str) -> bool:
    """Whether *variant* is a digest fragment rather than a variant name.

    A state file whose payload is unreadable falls back to its own filename, and
    for a variant :func:`_variant_fragment` had to hash that is this digest. It
    names nothing: it cannot be spelled back, and re-keying it would hash the
    digest again. Both tags count, the older one carrying no "@" at all.
    """
    return bool(_HASHED_FRAGMENT.fullmatch(variant.strip().lower()))


def state_filename_is_ambiguous(name: str) -> bool:
    """Whether an unreadable state filename has multiple possible owners."""
    stem, separator = name.removesuffix(_STATE_EXTENSION).lower(), "--variant--"
    boundary = stem.find(separator)
    if boundary >= 0 and stem.find(separator, boundary + 1) >= 0:
        return True
    repo_key = stem[:boundary] if boundary >= 0 else stem
    repo_fragment = repo_key.partition("--")[2]
    variant_fragment = stem.rsplit(separator, 1)[-1] if boundary >= 0 else ""
    return bool(
        _LEGACY_HASH_FRAGMENT.fullmatch(repo_fragment)
        or _LEGACY_HASH_FRAGMENT.fullmatch(variant_fragment)
    )


def _state_repo_key(
    repo_type: RepoType,
    repo_id: str,
    *,
    legacy_repo_key: bool = False,
    legacy_hash_key: bool = False,
) -> str:
    """Return an injective write key, or the pre-migration key for reads.

    Repositories ending in ``variant`` otherwise alias a shorter repository's
    legacy double-hyphen variant. Both collision sides use hashes for new state.
    """
    base = repo_cache_basename(repo_type, repo_id)
    repo_fragment = base.partition("--")[2]
    variant_prefix = f"{base}--variant--"
    longest_variant_key = f"{variant_prefix}{'x' * _MAX_VARIANT_FRAGMENT_LENGTH}"
    if _state_filename_fits(longest_variant_key) and (
        legacy_repo_key
        or (not base.endswith("--variant") and not repo_fragment.startswith(_HASH_PREFIXES))
    ):
        return base
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]
    tag = "sha256-" if legacy_hash_key else "@sha256-"
    return f"{repo_type}s--{tag}{digest}"


def variant_filename_prefix(
    repo_type: RepoType,
    repo_id: str,
    *,
    legacy_repo_key: bool = False,
    legacy_hash_key: bool = False,
) -> str:
    """Lowercased prefix every variant-keyed state file for this repo shares.

    The single source the download_manifest enumerators match against, so the
    scheme in :func:`_entry_key` cannot drift from them silently."""
    repo_key = _state_repo_key(
        repo_type,
        repo_id,
        legacy_repo_key = legacy_repo_key,
        legacy_hash_key = legacy_hash_key,
    )
    return f"{repo_key}--variant--"


def _variant_fragment(
    variant: str,
    *,
    legacy_variant_key: bool = False,
    legacy_hash_key: bool = False,
) -> str:
    normalized_variant = variant.strip().lower()
    # Double-hyphen variants can alias a repository component plus the
    # ``--variant--`` delimiter, so give them an injective hashed fragment.
    if _SAFE_VARIANT_FRAGMENT.fullmatch(normalized_variant) and (
        legacy_variant_key
        or ("--" not in normalized_variant and not normalized_variant.startswith(_HASH_PREFIXES))
    ):
        return normalized_variant
    digest = hashlib.sha256(normalized_variant.encode("utf-8")).hexdigest()[:32]
    tag = "sha256-" if legacy_hash_key else "@sha256-"
    return f"{tag}{digest}"


def variant_key_fragments(variant: str) -> tuple[str, ...]:
    """Every ``--variant--`` fragment this variant can be stored under.

    A variant that cannot be spelled literally in a filename is stored hashed, so
    a reader that recovers identity from the filename alone recovers the hash and
    not the variant. Callers holding the variant string need these to match it.
    """
    fragments: list[str] = []
    for legacy_variant_key in (False, True):
        for legacy_hash_key in (False, True):
            try:
                fragment = _variant_fragment(
                    variant,
                    legacy_variant_key = legacy_variant_key,
                    legacy_hash_key = legacy_hash_key,
                )
            except (UnicodeError, ValueError):
                continue
            if fragment not in fragments:
                fragments.append(fragment)
    return tuple(fragments)


def _entry_key(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str],
    *,
    legacy_variant_key: bool = False,
    legacy_repo_key: bool = False,
    legacy_hash_key: bool = False,
) -> str:
    base = _state_repo_key(
        repo_type,
        repo_id,
        legacy_repo_key = legacy_repo_key,
        legacy_hash_key = legacy_hash_key,
    )
    if not variant:
        return base
    variant_fragment = _variant_fragment(
        variant,
        legacy_variant_key = legacy_variant_key,
        legacy_hash_key = legacy_hash_key,
    )
    prefix = variant_filename_prefix(
        repo_type,
        repo_id,
        legacy_repo_key = legacy_repo_key,
        legacy_hash_key = legacy_hash_key,
    )
    return f"{prefix}{variant_fragment}"


def cache_scope_name(hub_cache: str | Path) -> str:
    normalized = os.path.normcase(str(Path(hub_cache).expanduser()))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:_CACHE_SCOPE_DIGEST_LENGTH]
    return f"cache-{digest}"


def _cache_scope(
    parent: Path,
    hub_cache: Optional[str | Path],
    *,
    create: bool = False,
) -> Optional[Path]:
    if hub_cache is None:
        return parent
    scoped = parent / cache_scope_name(hub_cache)
    if not create:
        return scoped
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
    create: bool = False,
    legacy_variant_key: bool = False,
    legacy_repo_key: bool = False,
    legacy_hash_key: bool = False,
) -> Optional[Path]:
    """Path to the manifest file for this triple. May or may not exist."""
    parent = _subdir(_MANIFESTS_SUBDIR, create = create)
    if parent is None:
        return None
    parent = _cache_scope(parent, hub_cache, create = create)
    if parent is None:
        return None
    entry_key = _entry_key(
        repo_type,
        repo_id,
        variant,
        legacy_variant_key = legacy_variant_key,
        legacy_repo_key = legacy_repo_key,
        legacy_hash_key = legacy_hash_key,
    )
    return parent / f"{entry_key}.json"


def marker_path(
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str] = None,
    *,
    hub_cache: Optional[str | Path] = None,
    create: bool = False,
    legacy_variant_key: bool = False,
    legacy_repo_key: bool = False,
    legacy_hash_key: bool = False,
) -> Optional[Path]:
    """Path to the cancel-marker file for this triple. May or may not exist."""
    parent = _subdir(_CANCELLED_SUBDIR, create = create)
    if parent is None:
        return None
    parent = _cache_scope(parent, hub_cache, create = create)
    if parent is None:
        return None
    entry_key = _entry_key(
        repo_type,
        repo_id,
        variant,
        legacy_variant_key = legacy_variant_key,
        legacy_repo_key = legacy_repo_key,
        legacy_hash_key = legacy_hash_key,
    )
    return parent / f"{entry_key}.json"


def manifests_dir(*, create: bool = False) -> Optional[Path]:
    """Manifests subdirectory, created on demand. ``None`` on failure.

    Exposed for iter_variant_manifests, which enumerates the directory to find
    every variant-keyed manifest for a repo (the path helpers above answer
    "where would key X go" but not "what keys exist")."""
    return _subdir(_MANIFESTS_SUBDIR, create = create)


def cancelled_dir(*, create: bool = False) -> Optional[Path]:
    """Cancel-marker subdirectory, created on demand. ``None`` on failure.

    See manifests_dir for why this iteration entry point is needed."""
    return _subdir(_CANCELLED_SUBDIR, create = create)


def workers_dir() -> Optional[Path]:
    """Worker PID-breadcrumb subdirectory, created on demand. ``None`` on failure.

    Each live download worker drops one breadcrumb here so a backend that
    restarts after a hard crash can reap workers it can no longer reach through
    its in-memory registry."""
    return _subdir(_WORKERS_SUBDIR, create = True)
