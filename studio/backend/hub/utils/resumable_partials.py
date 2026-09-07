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

Neither can be shown on Windows, which has no ``fcntl`` and no way to establish who owns a file
without pywin32, so the shared name stays off there and Windows keeps the stock writer.

A predictable name is also something another account can get to first, so the partial itself is
checked before a byte is appended: ``O_NOFOLLOW`` at open, and then owner, link count and file type
on the descriptor rather than the path, since only the descriptor is the thing about to be written.
See :func:`_objection_to`. Publishing re-checks that the name still holds what was written, because
``_chmod_and_move`` resolves it again.

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

# What flock reports when another holder has the lock, and nothing else: EACCES belongs to fcntl,
# and ENOLCK / EOPNOTSUPP mean this filesystem cannot lock at all.
_CONTENDED = frozenset({errno.EWOULDBLOCK, errno.EAGAIN})

# An allowlist rather than a list of network types: a FUSE mount reports whatever name its daemon
# chose and, unless it negotiates FUSE_FLOCK_LOCKS, the kernel answers flock locally, so
# fuse.rclone or fuse.s3fs over shared object storage would pass a network-name test. An
# unrecognised or blank type keeps the stock writer.
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
    callers outside Unsloth.

    *hub_cache* names a specific root instead. Unsloth remembers several, and a partial sitting in
    one of them is governed by that root's filesystem, not by whichever is currently selected.
    """
    root = None
    if hub_cache is not None:
        root = Path(hub_cache)
    else:
        try:
            from utils.hf_cache_settings import active_hf_hub_cache
            root = Path(active_hf_hub_cache())
        except Exception as exc:  # noqa: BLE001 - outside Unsloth, use the library's own view
            logger.debug("resumable partials: no Unsloth cache setting (%s)", exc)
    if root is None:
        try:
            from huggingface_hub import constants
            root = Path(constants.HF_HUB_CACHE)
        except Exception as exc:  # noqa: BLE001 - an unreadable cache is not a lock guarantee
            logger.debug("resumable partials: no hub cache to probe (%s)", exc)
            return None
    if hub_cache is not None:
        # Asked about a named root, so only report on one that is there: creating it would resurrect a cache
        # the user detached, and an absent root holds no partials to judge.
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


class _ProbeUnavailable(Exception):
    """The probe could not be run. Not a measurement, so it must not be remembered as one."""


def _device_at(directory: str) -> int:
    """The device the path is mounted from, which changes when a different filesystem replaces it.

    Part of the probe cache key: the path alone is not identity. An external cache can be unmounted
    and something else mounted at the same name, and a verdict about the old filesystem says
    nothing about the new one.
    """
    try:
        return os.stat(directory).st_dev
    except OSError as exc:
        raise _ProbeUnavailable(f"cannot stat {directory}: {exc}") from exc


def _filesystem_is_local(directory: str) -> bool:
    """Whether *directory* sits on a filesystem whose locking this host can speak for."""
    return _filesystem_is_local_on(directory, _device_at(directory))


@lru_cache(maxsize = 8)
def _filesystem_is_local_on(directory: str, device: int) -> bool:
    """The cached half, keyed on the mounted device as well as the path.

    A probe here cannot see another client, and NFS mounted ``-o local_lock=flock`` keeps flock
    locks client-local, so two hosts would each take the lock and neither would be refused. A mount
    we cannot identify counts as not local: this decides whether to re-enable a shared writer.
    """
    path = Path(directory).resolve()
    if str(path).startswith("\\\\") or str(path).startswith("//"):
        return False
    try:
        table = _mounts()
    except Exception as exc:  # noqa: BLE001 - unreadable now does not mean unreadable next time
        raise _ProbeUnavailable(f"could not read the mount table: {exc}") from exc
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


def _lock_is_honoured_at(directory: str) -> bool:
    """Whether ``flock`` under *directory* actually excludes a second holder."""
    return _lock_is_honoured_on(directory, _device_at(directory))


# Keyed on the directory and the device, so moving the cache or swapping the mount under it re-
# probes instead of reusing a verdict about a filesystem that is gone.
@lru_cache(maxsize = 8)
def _lock_is_honoured_on(directory: str, device: int) -> bool:
    """Take the lock twice and require the second to be refused.

    Separate ``open()`` calls make separate open file descriptions and flock judges them
    independently. Only contention counts as a refusal; a filesystem that grants both, or answers
    anything else, leaves the stock writer in place. A probe that could not be run at all raises
    instead, since a full disk or a briefly unwritable cache is not a measurement to remember.
    """
    import fcntl

    # A random, exclusively created file: the cache can be shared, and a predictable name lets another
    # user pre-place a symlink an unguarded open would follow and truncate.
    try:
        handle, name = tempfile.mkstemp(dir = directory, prefix = ".unsloth-flock-probe.")
    except Exception as exc:  # noqa: BLE001 - nowhere to probe now is not nowhere to probe later
        raise _ProbeUnavailable(f"could not create the probe in {directory}: {exc}") from exc
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
        # No fcntl on Windows, where huggingface_hub locks via msvcrt, and Windows has no way to establish
        # who owns a partial (os.stat reports st_uid 0 for every file and reading an ACL needs pywin32),
        # so another account on a shared NTFS cache could leave a partial with a chosen prefix and have
        # the remaining range appended to it, which the size-only check would pass.
        return False
    # _ProbeUnavailable is deliberately not caught: a probe that could not run is not an answer, and
    # letting it out keeps any caller from caching one.
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


def _objection_to(descriptor: int) -> Optional[str]:
    """Why the partial now open on *descriptor* must not be appended to, or ``None``.

    Judged on the descriptor rather than the path, so a swap between looking and opening cannot
    slip a different file past: this is the thing that will actually be written.

    Ownership is the load-bearing one. Nothing about a plain file betrays who wrote it, so a
    partial another account left is bytes of their choosing, and appending the server's remaining
    range to a chosen prefix publishes a blob that is the right length and the wrong file.
    huggingface_hub checks only the size afterwards, never the hash (huggingface_hub#3643), so
    nothing downstream would notice.
    """
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        return "not a regular file"
    if info.st_nlink > 1:
        return "hard linked from elsewhere"
    euid = getattr(os, "geteuid", None)
    if euid is None:
        # No owner to compare against, so nothing here can be vouched for.
        return "on a platform where ownership cannot be established"
    if info.st_uid != euid():
        return "owned by another user"
    return None


def _still_the_written_file(path: Path, written: os.stat_result) -> bool:
    """Whether *path* still names the file described by *written*.

    ``(st_dev, st_ino)`` identifies a file; the pathname does not. Checked before publishing
    because the move re-resolves the name, and a shared cache directory lets another account
    unlink or rename the partial after the last byte is written and leave something else there.

    This narrows the window rather than closing it: nothing between this stat and the move is
    atomic, and closing it properly needs a by-descriptor rename that Python does not expose
    portably. Turning "an unrelated file is published as the model" into "the download is retried"
    is the improvement available here.
    """
    try:
        current = os.lstat(path)
    except OSError as exc:
        logger.warning("resumable partials: the partial at %s vanished (%s)", path, exc)
        return False
    return (current.st_dev, current.st_ino) == (written.st_dev, written.st_ino)


def _open_stable_partial(path: Path) -> Optional[Any]:
    """Open the stable partial for append, or ``None`` if it cannot be trusted.

    The 1.18 nonce made this name unguessable; restoring the 1.17 name makes it predictable again,
    so on a cache another account can write, the entry can be pre-created and an unguarded ``"ab"``
    would build the blob on top of whatever is there. ``O_NOFOLLOW`` refuses a symlink outright;
    everything else is settled on the open descriptor by :func:`_objection_to`. The one look at the
    path is for Windows, which has no ``O_NOFOLLOW`` and so cannot refuse a link at open time.

    A partial that fails any of it is removed and a clean one started. One that cannot be opened
    at all is left untouched instead: ``EACCES`` from another account's ``0600`` file is not a
    position from which to judge or delete it. Either way the caller falls back to the stock
    writer, which invents its own name and cannot be steered.
    """
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | nofollow | getattr(os, "O_BINARY", 0)
    for last_attempt in (False, True):
        objection = None
        if not nofollow:
            try:
                if stat.S_ISLNK(os.lstat(path).st_mode):
                    objection = "a symlink"
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.debug("resumable partials: cannot stat %s (%s)", path, exc)
                return None
        if objection is None:
            try:
                descriptor = os.open(path, flags, 0o600)
            except OSError as exc:
                # ELOOP, or EMLINK on some BSDs: O_NOFOLLOW refused a symlink.
                if exc.errno in (errno.ELOOP, errno.EMLINK):
                    objection = "a symlink"
                else:
                    # Anything else is a partial this account cannot use and must not judge (another user's 0600 file
                    # answers EACCES, a directory EISDIR); raising would fail every attempt at the blob for good.
                    logger.warning(
                        "Cannot open the download partial at %s (%s); leaving it alone and "
                        "letting the stock writer fetch the file.",
                        path,
                        exc,
                    )
                    return None
            else:
                objection = _objection_to(descriptor)
                if objection is None:
                    return os.fdopen(descriptor, "ab")
                os.close(descriptor)
        if last_attempt:
            return None
        logger.warning("Discarding the download partial at %s: it is %s.", path, objection)
        try:
            os.unlink(path)
        except OSError as exc:
            logger.warning("Could not remove it (%s); leaving the resume to the stock writer.", exc)
            return None
    return None


def restore_resumable_partials() -> bool:
    """Patch huggingface_hub in THIS process. Idempotent, and a no-op where it is unsafe."""
    try:
        permitted = can_restore_partials()
    except _ProbeUnavailable as exc:
        # The worker calls this at import, so an escaping exception would take the whole download process
        # down instead of leaving it the stock writer.
        logger.debug("resumable partials: %s", exc)
        return False
    if not permitted:
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

        # A XET-backed repo still comes down over HTTP when hf_xet is absent or disabled, so what matters is
        # whether XET will run, not whether its metadata exists.
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
        written = os.fstat(opened.fileno())
        with opened as handle:
            resume_size = handle.tell()
            if expected_size is not None and resume_size > expected_size:
                # Longer than the file is supposed to be, so there is nothing to resume from: a Range starting past
                # the end answers 416 on every retry.
                logger.warning(
                    "Restarting '%s': the partial holds %s bytes but the file is %s.",
                    filename,
                    resume_size,
                    expected_size,
                )
                handle.seek(0)
                handle.truncate()
                resume_size = 0
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
        # _chmod_and_move resolves the name again, so publish only if the name still holds the file that was
        # actually written; otherwise another account could swap something in after the last write.
        if not _still_the_written_file(incomplete_path, written):
            logger.warning(
                "Not publishing '%s': the partial at %s was replaced while it was being written.",
                filename,
                incomplete_path,
            )
            return
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
    for probe in (_lock_is_honoured_on, _filesystem_is_local_on):
        clear = getattr(probe, "cache_clear", None)
        if clear is not None:
            clear()


def reset_probe_cache_for_tests() -> None:
    invalidate_probe_cache()
