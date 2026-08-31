# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Clean up the Unsloth compiled cache directory.

unsloth_compiled_cache (created by unsloth_zoo/compiler.py during
FastModel.from_pretrained) holds model-type-specific compiled files. Clear it
selectively between model loads, preserving model-agnostic components (Trainers)
that spawned subprocesses need.
"""

import contextlib
import errno
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


# Written when Unsloth creates the directory, so "we made this" is a fact rather
# than an inference from the contents.
CACHE_MARKER = ".unsloth_compiled_cache"

# Names only the compiler produces, so a cache Unsloth did not create is still
# recognised once it has been written into.
import re as _re

_GENERATED_NAME_RE = _re.compile(r"\A(unsloth_compiled_module_.+|Unsloth.+Trainer)\.py\Z")
# What may be deleted from a directory we do not own. Narrower on purpose:
# Unsloth*Trainer.py is a convention a user's own subclass can match, and there
# the marker is the only thing that would say we wrote it.
_OWNED_DELETE_RE = _re.compile(r"\Aunsloth_compiled_module_.+\.py\Z")


def _is_dedicated_cache(path: Path) -> bool:
    """True only for a directory Unsloth created for the cache and nothing else.

    A real file, not a link: exists() follows one, so a marker symlinked at any
    existing path would license the rmtree below over somebody's own directory.
    """
    marker = path / CACHE_MARKER
    try:
        return marker.is_file() and not marker.is_symlink()
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
        return any(
            item.is_file() and _GENERATED_NAME_RE.match(item.name) for item in path.iterdir()
        )
    except OSError:
        return False


def _builtin_cache_paths() -> set:
    """Paths that are ours by construction, so they need no marker.

    The CWD candidate is deliberately not one: Unsloth is launched from wherever
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
                "Not clearing %s: Unsloth did not create it and it holds no generated "
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
    # Unsloth*Trainer.py is a name a user's own subclass can carry, and that
    # directory goes on sys.path. Where we put it is ours anyway.
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


def cache_coordination_dir() -> Path:
    """Where backends of this install find each other.

    The studio home, the same scope the startup markers use. Two backends of one
    install share an install-tree compiled cache and that is the case this
    coordinates; two SEPARATE installs pointed at one UNSLOTH_COMPILE_LOCATION
    are not coordinated, and clearing is best effort there, as it was before.
    """
    from utils.paths.storage_roots import studio_root
    return studio_root()


# Held: we may probe and clear. Busy: someone else is in that critical section.
# Unavailable: no lock could be taken at all (unwritable studio home, a
# filesystem without flock), which must not mean "never clear the cache again",
# so the caller falls back to the unserialized probe it did before this lock.
LOCK_HELD = "held"
LOCK_BUSY = "busy"
LOCK_UNAVAILABLE = "unavailable"

# Long enough to outlast a real clear (an rmtree of a few dozen files), short
# enough that a wedged holder cannot stall lifespan startup behind it.
_LOCK_TIMEOUT = 10.0


# flock/msvcrt report contention through these; anything else (ENOSYS,
# EOPNOTSUPP on a network mount) is the lock being unsupported, and retrying it
# for ten seconds only to answer "busy" would pin the cache forever, since busy
# is read as proof of a sibling.
_CONTENTION_ERRNOS = frozenset(
    code
    for code in (
        getattr(errno, "EACCES", None),
        getattr(errno, "EAGAIN", None),
        getattr(errno, "EWOULDBLOCK", None),
        getattr(errno, "EDEADLOCK", None),
        getattr(errno, "EDEADLK", None),
    )
    if code is not None
)


def _try_lock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fd: int) -> None:
    with contextlib.suppress(Exception):
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
    with contextlib.suppress(OSError):
        os.close(fd)


@contextlib.contextmanager
def compiled_cache_lock(timeout: float = _LOCK_TIMEOUT):
    """Serialize a sibling probe plus cache clear against a sibling's publication.

    Without it the probe is a check-then-act race with a real window: A probes and
    finds nobody, B publishes its startup marker and begins compiling, A then
    clears and deletes the modules B just wrote. Holding this across both halves
    (the probe plus clear here, the marker write in run.py) closes it.

    Never raises at the caller and never waits indefinitely: startup runs through
    here, so a lock that cannot be taken has to degrade rather than block.
    """
    import time

    try:
        lock_dir = cache_coordination_dir()
        lock_dir.mkdir(parents = True, exist_ok = True)
        fd = os.open(str(lock_dir / "compiled-cache.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    except Exception as exc:  # noqa: BLE001
        # Resolving or opening it is part of taking it, so it degrades the same
        # way rather than aborting a startup that only wanted to know about
        # siblings.
        logger.debug(f"Could not open the compiled-cache lock ({exc})")
        yield LOCK_UNAVAILABLE
        return

    fds: "List[int]" = []
    state = LOCK_HELD
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                _try_lock(fd)
                fds.append(fd)
                break
            except OSError as exc:
                if exc.errno not in _CONTENTION_ERRNOS:
                    # Not contention: the filesystem cannot lock at all.
                    logger.debug(f"Compiled-cache locking unavailable ({exc})")
                    with contextlib.suppress(OSError):
                        os.close(fd)
                    state = LOCK_UNAVAILABLE
                    break
                if time.monotonic() >= deadline:
                    with contextlib.suppress(OSError):
                        os.close(fd)
                    state = LOCK_BUSY
                    break
                time.sleep(0.05)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Compiled-cache locking unavailable ({exc})")
                with contextlib.suppress(OSError):
                    os.close(fd)
                state = LOCK_UNAVAILABLE
                break
        yield state
    finally:
        for fd in fds:
            _unlock(fd)


def clear_compiled_cache_unless_shared(sibling_probe = None) -> None:
    """Clear the compiled cache, unless another backend of this install is live.

    The cache sits in the install tree, not the studio home, so two of our own
    backends share it and the wipe would delete modules the other one is still
    importing -- including the Unsloth*Trainer.py that the in-process clears
    preserve for spawn workers. run_server supplies the probe; without it (tests,
    an embedded app) the old unconditional clear stands.

    The probe and the clear run under `compiled_cache_lock` so a sibling cannot
    publish itself in between and lose the modules it has already compiled.

    Two launches that overlap from cold both keep a cache neither has cleaned,
    so stale modules can survive until the next start that finds itself alone.
    That is the deliberate direction: the failure this replaces was the two of
    them deleting each other's modules mid-run.
    """
    if not callable(sibling_probe):
        clear_unsloth_compiled_cache()
        return
    with compiled_cache_lock() as lock_state:
        if lock_state == LOCK_BUSY:
            # Somebody is inside the critical section, so there is a sibling by
            # definition; that is already the answer, no probe needed.
            logger.info(
                "Keeping the compiled cache: another backend of this install holds the cache lock"
            )
            return
        sibling = sibling_probe()
        if sibling is None:
            clear_unsloth_compiled_cache()
            return
    logger.info(
        f"Keeping the compiled cache: another backend of this install is live (PID {sibling})"
    )


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
        # The marker goes with whatever was cleared and nothing rewrites it
        # (setup_cache_env writes it only when it first sets the variable), so
        # the next cleanup would demote our own cache to "shared".
        # A built-in path needs no marker, so no restoring either, unless it
        # is a link: the clear removed the target and left it dangling.
        if dedicated and (str(cache_dir) not in _builtin_cache_paths() or cache_dir.is_symlink()):
            try:
                restored = Path(os.path.realpath(cache_dir))
                restored.mkdir(parents = True, exist_ok = True)
                (restored / CACHE_MARKER).touch(exist_ok = True)
            except OSError as e:
                logger.debug(f"Could not restore the cache marker in {cache_dir}: {e}")
