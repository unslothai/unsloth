# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Give huggingface_hub >= 1.18 back its resumable HTTP partials.

1.18 replaced the shared ``<etag>.incomplete``, opened append and continued with a Range request,
with a process-unique ``<etag>.<nonce>.incomplete`` opened ``"wb"`` and unlinked on the way out
(huggingface/huggingface_hub#4228). So a cancelled or killed transfer refetches from zero, and
:mod:`hub.workers.hf_download`, whose SIGKILL then restart loop reads ``.incomplete`` for its resume
offset, has nothing to read.

Only the caller went. ``http_get`` still takes ``resume_size``, still sends the Range header, and
still does ``seek(0)`` + ``truncate()`` when a server answers 200 to a Range request, the case that
would otherwise duplicate bytes. This restores the 1.17 caller and nothing else.

Upstream removed it because the shared name corrupts the cache where ``flock(2)`` does not exclude
every caller (Lustre, GPFS, some NFS): two processes append to one file. So exclusion has to be
shown, and where it cannot be, the stock writer stays and partials keep reporting as unresumable.

Two things have to hold, because a probe run here can only speak for this host. The cache must be
on a local filesystem: NFS mounted ``-o local_lock=flock`` keeps flock locks client-local, so two
hosts each take "the" lock and neither sees ``EWOULDBLOCK``, and no test on one of them can notice.
And ``flock`` must actually exclude a second holder here, which :func:`_lock_is_honoured_at`
measures by taking the lock twice. Only ``EWOULDBLOCK``/``EAGAIN`` counts as exclusion: a
filesystem with no locking answers ``ENOLCK`` or ``EOPNOTSUPP``, and reading that as "refused"
would enable the shared writer on precisely the mounts that cannot support it.

The other corruption route, appending to a sparse XET or parallel-Range partial, belongs to the
transport markers in :mod:`hub.utils.download_registry`. They are bypassed on >= 1.18 only because
no resumer exists, so restoring one brings them back into force.
"""

from __future__ import annotations

import errno
import os
import stat
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

# The last line whose partials the stock writer already appends to.
LAST_STOCK_RESUMABLE_VERSION = (1, 17)
# The newest major whose internals this has been read against. A 2.x is not assumed to look alike.
MAX_SUPPORTED_MAJOR = 1

# What flock reports when another holder has the lock, and nothing else. EACCES belongs to fcntl,
# not flock; ENOLCK and EOPNOTSUPP mean this filesystem cannot lock at all.
_CONTENDED = frozenset({errno.EWOULDBLOCK, errno.EAGAIN})

# Filesystems backed by storage attached to this host, so a lock taken here is the only lock.
# An allowlist rather than a list of network types: a FUSE mount reports whatever name its daemon
# chose, and unless it negotiates FUSE_FLOCK_LOCKS the kernel answers flock locally (libfuse
# fuse_lowlevel.h), so fuse.rclone or fuse.s3fs over shared object storage passes the probe while
# excluding nobody. Naming the ones we know instead means an unrecognised or blank type keeps the
# stock writer rather than silently re-enabling a shared one.
_LOCAL_FSTYPES = frozenset(
    {
        # Linux
        "bcachefs",
        "btrfs",
        "ext2",
        "ext3",
        "ext4",
        "f2fs",
        "jfs",
        "nilfs2",
        "reiser4",
        "reiserfs",
        "xfs",
        "zfs",
        # Block-backed FUSE (ntfs-3g and friends), container and memory-backed roots
        "fuseblk",
        "overlay",
        "overlayfs",
        "ramfs",
        "tmpfs",
        # Removable and cross-platform volumes
        "exfat",
        "fat",
        "fat32",
        "msdos",
        "vfat",
        # macOS
        "apfs",
        "hfs",
        "hfsplus",
        "ufs",
        # Windows, where psutil reports the volume format
        "ntfs",
        "ntfs3",
        "refs",
    }
)


def _hub_version() -> tuple[int, ...]:
    """``(major, minor)`` for the installed huggingface_hub, or ``()`` when it cannot be read."""
    try:
        from huggingface_hub import __version__ as raw
    except Exception as exc:  # noqa: BLE001 - an unimportable hub is not ours to diagnose
        logger.debug("resumable partials: huggingface_hub unreadable (%s)", exc)
        return ()
    parts: list[int] = []
    for chunk in str(raw).split(".")[:2]:
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            return ()
        parts.append(int(digits))
    return tuple(parts)


def _probe_dir(hub_cache: Optional[Path | str] = None) -> Optional[Path]:
    """The cache whose filesystem decides, defaulting to the one the worker will use.

    ``constants.HF_HUB_CACHE`` is resolved at import and moving the cache in Settings does not
    rewrite the live process (see ``hub/services/download_lifecycle.py``), so probing it would
    judge a different filesystem than the partial lands on. The constant is the fallback for
    callers outside Studio.

    *hub_cache* names a specific root instead. Studio remembers several, and a partial sitting in
    one of them is governed by that root's filesystem, not by whichever is currently selected.
    """
    root = None
    if hub_cache is not None:
        root = Path(hub_cache)
    else:
        try:
            from utils.hf_cache_settings import active_hf_hub_cache
            root = Path(active_hf_hub_cache())
        except Exception as exc:  # noqa: BLE001 - outside Studio, use the library's own view
            logger.debug("resumable partials: no Studio cache setting (%s)", exc)
    if root is None:
        try:
            from huggingface_hub import constants
            root = Path(constants.HF_HUB_CACHE)
        except Exception as exc:  # noqa: BLE001 - an unreadable cache is not a lock guarantee
            logger.debug("resumable partials: no hub cache to probe (%s)", exc)
            return None
    if hub_cache is not None:
        # Asked about a named root, so only report on one that is there. Creating it would
        # resurrect a cache the user detached, and an absent root holds no partials to judge.
        return root if root.is_dir() else None
    try:
        root.mkdir(parents = True, exist_ok = True)
        return root
    except Exception as exc:  # noqa: BLE001 - an unwritable cache is not a lock guarantee
        logger.debug("resumable partials: hub cache not writable (%s)", exc)
        return None


def _mounts() -> list[tuple[str, str]]:
    """``(mountpoint, fstype)`` for every mount, via psutil so macOS and Windows answer too."""
    import psutil
    return [(part.mountpoint, part.fstype or "") for part in psutil.disk_partitions(all = True)]


@lru_cache(maxsize = 8)
def _filesystem_is_local(directory: str) -> bool:
    """Whether *directory* sits on a filesystem whose locking this host can speak for.

    A probe here cannot see another client, and NFS mounted ``-o local_lock=flock`` keeps flock
    locks client-local, so two hosts would each take the lock and neither would be refused. A mount
    we cannot identify counts as not local: this decides whether to re-enable a shared writer.
    """
    path = Path(directory).resolve()
    if str(path).startswith("\\\\") or str(path).startswith("//"):
        return False  # UNC share
    try:
        table = _mounts()
    except Exception as exc:  # noqa: BLE001 - an unidentifiable mount is not a local one
        logger.debug("resumable partials: could not read the mount table (%s)", exc)
        return False
    best, fstype = "", None
    for mount, kind in table:
        if str(path) == mount or str(path).startswith(mount.rstrip(os.sep) + os.sep):
            if len(mount) >= len(best):
                best, fstype = mount, kind.lower()
    if fstype is None:
        logger.debug("resumable partials: no mount found for %s", path)
        return False
    if fstype not in _LOCAL_FSTYPES:
        logger.info(
            "Download partials stay unresumable: %s is on %s, which is not a filesystem this host "
            "can prove it locks alone.",
            path,
            fstype or "an unnamed type",
        )
        return False
    return True


# Keyed on the directory, so moving the cache re-probes instead of reusing the old verdict.
@lru_cache(maxsize = 8)
def _lock_is_honoured_at(directory: str) -> bool:
    """Whether ``flock`` under *directory* actually excludes a second holder.

    Take the lock twice and require the second to be refused, since separate ``open()`` calls make
    separate open file descriptions and flock judges them independently. Only contention counts as
    a refusal; anything else, a failed probe included, leaves the stock writer in place.
    """
    import fcntl

    # A random, exclusively created file: the cache can be shared, and a predictable name there
    # lets another user pre-place a symlink that an unguarded open would follow and truncate.
    try:
        handle, name = tempfile.mkstemp(dir = directory, prefix = ".unsloth-flock-probe.")
    except Exception as exc:  # noqa: BLE001 - nowhere to probe is not a working lock
        logger.debug("resumable partials: could not create the probe (%s)", exc)
        return False
    second = None
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        second = os.open(name, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
        try:
            fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in _CONTENDED:
                return True
            # ENOLCK / EOPNOTSUPP / EINTR: not a refusal, so nothing has been shown.
            logger.info(
                "Download partials stay unresumable: locking %s answered %s rather than "
                "contention.",
                directory,
                errno.errorcode.get(exc.errno, exc.errno),
            )
            return False
        logger.info(
            "Download partials stay unresumable: %s grants the same lock twice, so a shared "
            "partial could be written by two processes at once.",
            directory,
        )
        return False
    except Exception as exc:  # noqa: BLE001 - same, an unprovable lock is not a working one
        logger.debug("resumable partials: lock probe failed (%s)", exc)
        return False
    finally:
        for fd in (second, handle):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        try:
            os.unlink(name)
        except OSError:
            pass


def _exclusion_is_provable(hub_cache: Optional[Path | str] = None) -> bool:
    """Whether a shared partial under *hub_cache* (default: the cache in force) has one writer."""
    directory = _probe_dir(hub_cache)
    if directory is None:
        return False
    try:
        import fcntl  # noqa: F401
    except ImportError:
        # No fcntl on Windows, where huggingface_hub locks via msvcrt: mandatory rather than
        # advisory. A network share still cannot be spoken for, which the locality check catches.
        return os.name == "nt" and _filesystem_is_local(str(directory))
    return _filesystem_is_local(str(directory)) and _lock_is_honoured_at(str(directory))


def _hub_is_patchable() -> bool:
    """Whether the installed hub exposes the pieces the restored caller needs."""
    try:
        from huggingface_hub import file_download
    except Exception:  # noqa: BLE001
        return False
    needed = ("_download_to_tmp_and_move", "http_get", "_chmod_and_move", "_check_disk_space")
    return all(hasattr(file_download, name) for name in needed)


def can_restore_partials(hub_cache: Optional[Path | str] = None) -> bool:
    """Whether the shared-name writer is safe for partials under *hub_cache*.

    Read by the server to decide what to tell the UI and by the worker before it patches, so both
    answer the same. Default is the cache in force, which is the one a download will write; pass a
    root to ask about partials already sitting in a different remembered cache, whose filesystem
    may lock differently from the selected one.
    """
    version = _hub_version()
    if not version or version <= LAST_STOCK_RESUMABLE_VERSION or version[0] > MAX_SUPPORTED_MAJOR:
        return False
    return _hub_is_patchable() and _exclusion_is_provable(hub_cache)


def _open_stable_partial(path: Path) -> Optional[Any]:
    """Open the stable partial for append, or ``None`` if it cannot be trusted.

    The 1.18 nonce made this name unguessable; restoring the 1.17 name makes it predictable again,
    so on a cache another account can write, the entry can be pre-created as a symlink or a hard
    link to any file this process may write and an unguarded ``"ab"`` would append the download to
    it (``_chmod_and_move`` would then chmod the target too). ``O_NOFOLLOW`` refuses a symlink and
    closes the race the ``lstat`` alone would leave; the ``lstat`` catches a hard link and a
    non-file, which ``O_NOFOLLOW`` does not see, and covers Windows, which has no ``O_NOFOLLOW``.

    A planted entry is removed and a clean partial started. Where even that is refused, the caller
    falls back to the stock writer, which invents its own name and cannot be steered.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    for last_attempt in (False, True):
        planted = None
        try:
            existing = os.lstat(path)
        except FileNotFoundError:
            existing = None
        except OSError as exc:
            logger.debug("resumable partials: cannot stat %s (%s)", path, exc)
            return None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            planted = "not a regular file"
        elif existing is not None and existing.st_nlink > 1:
            planted = "hard linked from elsewhere"
        if planted is None:
            try:
                handle = os.fdopen(os.open(path, flags, 0o600), "ab")
            except OSError as exc:
                # ELOOP, or EMLINK on some BSDs: it became a symlink after the lstat.
                if exc.errno not in (errno.ELOOP, errno.EMLINK):
                    raise
                planted = "a symlink"
            else:
                return handle
        if last_attempt:
            return None
        logger.warning("Discarding the download partial at %s: it is %s.", path, planted)
        try:
            os.unlink(path)
        except OSError as exc:
            logger.warning("Could not remove it (%s); leaving the resume to the stock writer.", exc)
            return None
    return None


def restore_resumable_partials() -> bool:
    """Patch huggingface_hub in THIS process. Idempotent, and a no-op where it is unsafe."""
    if not can_restore_partials():
        return False

    from huggingface_hub import file_download

    stock = file_download._download_to_tmp_and_move
    if getattr(stock, "_unsloth_resumable", False):
        return True

    def _download_to_tmp_and_move(
        incomplete_path: Path,
        destination_path: Path,
        url_to_download: str,
        headers: dict,
        expected_size: Optional[int],
        filename: str,
        force_download: bool = False,
        xet_file_data: Any = None,
        **kwargs: Any,
    ) -> None:
        if destination_path.exists() and not force_download:
            return

        # A XET-backed repo still comes down over HTTP when hf_xet is absent or disabled, so what
        # matters is whether XET will run, not whether its metadata exists. XET writes its own way.
        uses_xet = xet_file_data is not None and file_download.is_xet_available()
        if force_download or uses_xet:
            return stock(
                incomplete_path = incomplete_path,
                destination_path = destination_path,
                url_to_download = url_to_download,
                headers = headers,
                expected_size = expected_size,
                filename = filename,
                force_download = force_download,
                xet_file_data = xet_file_data,
                **kwargs,
            )

        # The 1.17 caller: a stable name, opened for append, told how far it already got.
        opened = _open_stable_partial(incomplete_path)
        if opened is None:
            return stock(
                incomplete_path = incomplete_path,
                destination_path = destination_path,
                url_to_download = url_to_download,
                headers = headers,
                expected_size = expected_size,
                filename = filename,
                force_download = force_download,
                xet_file_data = xet_file_data,
                **kwargs,
            )
        with opened as handle:
            resume_size = handle.tell()
            if expected_size is not None:
                file_download._check_disk_space(expected_size, incomplete_path.parent)
                file_download._check_disk_space(expected_size, destination_path.parent)
            if resume_size:
                logger.info(
                    "Resuming '%s' from %s of %s bytes",
                    filename,
                    resume_size,
                    expected_size,
                )
            file_download.http_get(
                url_to_download,
                handle,
                resume_size = resume_size,
                headers = headers,
                expected_size = expected_size,
                tqdm_class = kwargs.get("tqdm_class"),
            )
        # Only on success: a failure has to leave the partial where the next attempt looks for it.
        file_download._chmod_and_move(incomplete_path, destination_path)

    _download_to_tmp_and_move._unsloth_resumable = True
    _download_to_tmp_and_move._unsloth_stock = stock
    file_download._download_to_tmp_and_move = _download_to_tmp_and_move
    logger.info("Restored resumable HTTP partials for huggingface_hub %s", _hub_version())
    return True


def invalidate_probe_cache() -> None:
    """Forget every probed filesystem. Called when the cache location changes."""
    # getattr: a test that replaced either probe outright has no cache to clear.
    for probe in (_lock_is_honoured_at, _filesystem_is_local):
        clear = getattr(probe, "cache_clear", None)
        if clear is not None:
            clear()


def reset_probe_cache_for_tests() -> None:
    invalidate_probe_cache()
