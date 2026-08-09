# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Clean up the Unsloth compiled cache directory.

unsloth_compiled_cache (created by unsloth_zoo/compiler.py during
FastModel.from_pretrained) holds model-type-specific compiled files. Clear it
selectively between model loads, preserving model-agnostic components (Trainers)
that spawned subprocesses need.
"""

import os
import shutil
import structlog
from loggers import get_logger
from pathlib import Path
from typing import List, Optional

logger = get_logger(__name__)

# Possible locations where unsloth_compiled_cache may appear
_BACKEND_DIR = Path(__file__).resolve().parent.parent  # studio/backend
_PROJECT_ROOT = _BACKEND_DIR.parent.parent  # repo root

_CACHE_DIRS = [
    _BACKEND_DIR / "unsloth_compiled_cache",
    _PROJECT_ROOT / "unsloth_compiled_cache",
    _PROJECT_ROOT / "studio" / "tmp" / "unsloth_compiled_cache",
]


def _configured_cache_dirs() -> List[Path]:
    """Cache dirs outside the source tree: the configured one, and the CWD.

    The candidates above are all source-tree relative, so a cache created in
    the launcher's CWD (the user profile on Windows) was invisible to cleanup.
    The CWD is still checked for installs that predate the pinned location.
    """
    import os

    dirs: List[Path] = []
    configured = (os.environ.get("UNSLOTH_COMPILE_LOCATION") or "").strip()
    if configured:
        dirs.append(Path(configured).expanduser())
    try:
        dirs.append(Path.cwd() / "unsloth_compiled_cache")
    except OSError:
        pass
    return dirs


def get_existing_cache_dirs() -> List[Path]:
    """Return known compiled-cache directories that currently exist on disk."""
    seen: set = set()
    found: List[Path] = []
    for candidate in [*_CACHE_DIRS, *_configured_cache_dirs()]:
        try:
            key = candidate.resolve()
        except OSError:
            key = candidate
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            found.append(candidate)
    return found


# Written when Studio creates the directory, so "we made this" is a fact rather
# than an inference from the contents.
CACHE_MARKER = ".unsloth_compiled_cache"

# Names only the compiler produces, so a cache Studio did not create is still
# recognised once it has been written into.
import re as _re

_GENERATED_NAME_RE = _re.compile(r"\A(unsloth_compiled_module_.+|Unsloth.+Trainer)\.py\Z")
# What may be deleted from a directory we do not own. Narrower on purpose:
# Unsloth*Trainer.py is a convention a user's own subclass can match, and there
# the marker is the only thing that would say we wrote it.
_OWNED_DELETE_RE = _re.compile(r"\Aunsloth_compiled_module_.+\.py\Z")


def _is_dedicated_cache(path: Path) -> bool:
    """True only for a directory Studio created for the cache and nothing else."""
    try:
        return (path / CACHE_MARKER).exists()
    except OSError:
        return False


def _trusted_cache_paths() -> set:
    """Where a cache is ours by where it is: the source-tree candidates, and
    whatever UNSLOTH_COMPILE_LOCATION names, since that is the caller's answer
    to where the cache lives. Never the launch directory."""
    trusted = _builtin_cache_paths()
    configured = (os.environ.get("UNSLOTH_COMPILE_LOCATION") or "").strip()
    if configured:
        trusted.add(str(Path(configured).expanduser()))
    return trusted


def _entries(path: Path) -> list:
    try:
        return list(path.iterdir())
    except OSError:
        return []


def _holds_generated_modules(path: Path) -> bool:
    """True when the compiler has written into this directory.

    A shape test is not enough to own the directory: a directory of plain .py
    files is someone's package, and this decides what gets deleted.
    """
    try:
        return any(_GENERATED_NAME_RE.match(item.name) for item in path.iterdir())
    except OSError:
        return False


def _builtin_cache_paths() -> set:
    """Paths that are ours by construction, so they need no marker.

    The CWD candidate is deliberately not one: Studio is launched from wherever
    the shell happens to be, and a directory there is only ours if it says so.
    """
    return {str(p) for p in _CACHE_DIRS}


def _cleanable_cache_dirs() -> "List[tuple]":
    """``(directory, dedicated)`` for every cache dir something may be removed from.

    UNSLOTH_COMPILE_LOCATION is a user-set variable, so it can name a directory
    that holds other things (`$HOME/.cache`). Built-in paths, and any directory
    carrying the marker, are ours whole. Anywhere else only the generated files
    are ours, so only those may go.
    """
    builtin = _builtin_cache_paths()
    cleanable: "List[tuple]" = []
    for cache_dir in get_existing_cache_dirs():
        # A built-in path is ours by construction only while it IS the directory.
        # Through a link, the marker on the target is the only proof, since the
        # clearing below resolves it and would take whatever it points at.
        owned_by_path = str(cache_dir) in builtin and not cache_dir.is_symlink()
        if owned_by_path or _is_dedicated_cache(cache_dir):
            cleanable.append((cache_dir, True))
        elif _holds_generated_modules(cache_dir):
            cleanable.append((cache_dir, False))
        else:
            logger.warning(
                "Not clearing %s: Studio did not create it and it holds no generated "
                "modules. Point UNSLOTH_COMPILE_LOCATION at a directory used only for "
                "the compiled cache.",
                cache_dir,
            )
    return cleanable


def register_compiled_cache_on_path() -> None:
    """Add all existing compiled-cache directories to sys.path and PYTHONPATH.

    Ensures spawned workers (on 'spawn'-start platforms, i.e. Windows and macOS)
    can import dynamically compiled modules such as UnslothSFTTrainer.
    """
    import os
    import sys

    pypath = os.environ.get("PYTHONPATH", "")
    pypath_entries = [p for p in pypath.split(os.pathsep) if p]

    # Iterate in reverse so earlier _CACHE_DIRS entries (higher priority) are
    # inserted last and thus end up first in sys.path / PYTHONPATH.
    # Same ownership test as cleanup: a directory in the launch dir that merely
    # has the name would otherwise shadow real dependencies for every worker.
    # A directory in the launch dir needs a file only the compiler writes:
    # Unsloth*Trainer.py is a name a user's own subclass can carry, and
    # prepending that directory lets anything else in it shadow real modules for
    # every worker. Where we were pointed, or where we put it, is ours anyway.
    trusted = _trusted_cache_paths()
    registrable = [
        d
        for d, dedicated in _cleanable_cache_dirs()
        if dedicated
        or str(d) in trusted
        or any(_OWNED_DELETE_RE.match(item.name) for item in _entries(d))
    ]
    for cache_dir in reversed(registrable):
        resolved = str(cache_dir.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        if resolved not in pypath_entries:
            pypath_entries.insert(0, resolved)

    os.environ["PYTHONPATH"] = os.pathsep.join(pypath_entries)


def clear_unsloth_compiled_cache(preserve_patterns: Optional[List[str]] = None) -> None:
    """
    Remove compiled files from the cache directory (idempotent).

    Args:
        preserve_patterns: glob patterns for files to keep
                           (e.g., ["Unsloth*Trainer.py"]). If None or empty,
                           the entire cache directory is deleted (legacy behavior).
    """
    for cache_dir, dedicated in _cleanable_cache_dirs():
        if not dedicated:
            # A shared directory we only ever wrote generated modules into, so
            # they are the only thing here that may be removed.
            logger.info(f"Cleaning generated modules from shared directory: {cache_dir}")
            for item in cache_dir.iterdir():
                if not item.is_file() or not _OWNED_DELETE_RE.match(item.name):
                    continue
                if preserve_patterns and any(item.match(p) for p in preserve_patterns):
                    continue
                try:
                    item.unlink()
                except OSError as e:
                    logger.debug(f"Could not delete {item}: {e}")
        elif preserve_patterns:
            logger.info(
                f"Cleaning unsloth compiled cache (preserving {preserve_patterns}): " f"{cache_dir}"
            )

            for item in cache_dir.iterdir():
                if item.is_file():
                    preserve = any(item.match(pattern) for pattern in preserve_patterns)
                    if not preserve:
                        try:
                            item.unlink()
                        except OSError as e:
                            logger.debug(f"Could not delete {item}: {e}")

                elif item.is_dir():
                    # Always clear __pycache__ and other subdirectories
                    shutil.rmtree(item, ignore_errors = True)
        else:
            # Legacy: remove the entire directory. Resolved first: rmtree refuses
            # a symlink, and ignore_errors would leave the whole cache in place.
            logger.info(f"Removing unsloth compiled cache: {cache_dir}")
            shutil.rmtree(Path(os.path.realpath(cache_dir)), ignore_errors = True)
        # The marker goes with whatever was cleared, and nothing rewrites it
        # (setup_cache_env only writes it when it first sets the variable), so
        # the next cleanup would demote our own cache to "shared". Built-in
        # paths are recognised without one and stay deleted.
        if dedicated and str(cache_dir) not in _builtin_cache_paths():
            try:
                restored = Path(os.path.realpath(cache_dir))
                restored.mkdir(parents = True, exist_ok = True)
                (restored / CACHE_MARKER).touch(exist_ok = True)
            except OSError as e:
                logger.debug(f"Could not restore the cache marker in {cache_dir}: {e}")
