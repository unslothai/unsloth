# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Give huggingface_hub >= 1.18 back its resumable HTTP partials.

Up to 1.17 the HTTP resumer opened ``<etag>.incomplete`` in append mode and sent
``Range: bytes={resume_size}-``. 1.18 replaced it with a process-unique ``<etag>.<nonce>.incomplete``
opened ``"wb"`` and unlinked on the way out (huggingface/huggingface_hub#4228), so a cancelled or
killed transfer is refetched from zero and whatever survives a hard kill is litter no later attempt
can find. This is what :mod:`hub.workers.hf_download` is built around: its SIGKILL then restart loop
reads ``.incomplete`` to compute a resume offset, and there is nothing left to read.

Only the caller was removed. ``http_get`` still takes ``resume_size``, still sends the Range header,
and still does ``seek(0)`` + ``truncate()`` when a server answers 200 to a Range request, which is
the case that would otherwise duplicate bytes. So this restores the 1.17 caller and nothing else.

Why upstream removed it, and what that means here
-------------------------------------------------
The shared name corrupts the cache on filesystems where ``flock(2)`` silently succeeds for every
caller (Lustre, GPFS, some NFS mounts): two processes append to one file. That hazard is real, so
it is measured rather than assumed. :func:`can_restore_partials` takes the lock twice and requires
the second attempt to be refused. Where that cannot be shown, the stock writer stays, and Studio
keeps reporting partials as unresumable.

The other corruption route, an HTTP resumer appending to a sparse XET or parallel-Range partial, is
already handled a layer up by the transport markers in :mod:`hub.utils.download_registry`. Those
markers are bypassed on >= 1.18 precisely because no resumer exists; restoring one brings them back
into force, which is where that check belongs.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

# The last line whose partials the stock writer already appends to.
LAST_STOCK_RESUMABLE_VERSION = (1, 17)
# The newest major whose internals this has been read against. A 2.x is not assumed to look alike.
MAX_SUPPORTED_MAJOR = 1

# Per-process, so two Studios probing at once cannot take each other's lock and read a working
# filesystem as a broken one. Both opens below are this process's own, and flock still refuses the
# second, since separate open() calls make separate open file descriptions.
_PROBE_NAME = f".unsloth-flock-probe.{os.getpid()}"


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


def _probe_dir() -> Optional[Path]:
    """The cache the download worker will actually use, not the one this process booted with.

    ``constants.HF_HUB_CACHE`` is resolved at import, and moving the cache in Settings does not
    rewrite the live process (see ``hub/services/download_lifecycle.py``), while a worker is
    spawned with the new one. Probing the stale path would judge a different filesystem than the
    one the partial lands on, so the live setting wins and the import-time constant is the
    fallback for callers outside Studio.
    """
    root = None
    try:
        from utils.hf_cache_settings import active_hf_hub_cache

        root = Path(active_hf_hub_cache())
    except Exception as exc:  # noqa: BLE001 - outside Studio, fall back to the library's own view
        logger.debug("resumable partials: no Studio cache setting (%s)", exc)
    if root is None:
        try:
            from huggingface_hub import constants

            root = Path(constants.HF_HUB_CACHE)
        except Exception as exc:  # noqa: BLE001 - an unreadable cache is not a lock guarantee
            logger.debug("resumable partials: no hub cache to probe (%s)", exc)
            return None
    try:
        root.mkdir(parents = True, exist_ok = True)
        return root
    except Exception as exc:  # noqa: BLE001 - an unwritable cache is not a lock guarantee
        logger.debug("resumable partials: hub cache not writable (%s)", exc)
        return None


# Keyed on the directory, so moving the cache re-probes rather than reusing the old filesystem's
# verdict. Small and bounded: an install moves its cache a handful of times, not continuously.
@lru_cache(maxsize = 8)
def _lock_is_honoured_at(directory: str) -> bool:
    """Whether ``flock`` under *directory* actually excludes a second holder.

    This is the exact hazard 1.18 removed the shared partial for, so it is tested: take the lock
    twice and require the second to be refused. Anything else, including a probe that cannot run,
    answers False and leaves the stock writer in place.
    """
    import fcntl

    probe = Path(directory) / _PROBE_NAME
    try:
        # Binary: this file is a lock, never text, so it has no encoding to get wrong.
        with open(probe, "wb") as first:
            fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with open(probe, "wb") as second:
                try:
                    fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    return True
                logger.info(
                    "Download partials stay unresumable: %s grants the same lock twice, so a "
                    "shared partial could be written by two processes at once.",
                    directory,
                )
                return False
    except Exception as exc:  # noqa: BLE001 - same, an unprovable lock is not a working one
        logger.debug("resumable partials: lock probe failed (%s)", exc)
        return False
    finally:
        try:
            probe.unlink(missing_ok = True)
        except OSError:
            pass


def _lock_is_honoured() -> bool:
    """The probe for the cache in force right now."""
    try:
        import fcntl  # noqa: F401
    except ImportError:
        # Windows has no fcntl. huggingface_hub locks via msvcrt there, which is mandatory rather
        # than advisory, so the silent-sharing failure this probe looks for cannot happen.
        return os.name == "nt"
    directory = _probe_dir()
    return False if directory is None else _lock_is_honoured_at(str(directory))


def _hub_is_patchable() -> bool:
    """Whether the installed hub exposes the pieces the restored caller needs."""
    try:
        from huggingface_hub import file_download
    except Exception:  # noqa: BLE001
        return False
    needed = ("_download_to_tmp_and_move", "http_get", "_chmod_and_move", "_check_disk_space")
    return all(hasattr(file_download, name) for name in needed)


def can_restore_partials() -> bool:
    """Whether :func:`restore_resumable_partials` would take effect on this machine.

    Read from the server process to decide what to tell the UI, and from the worker before it
    patches, so both answer the same thing.
    """
    version = _hub_version()
    if not version or version <= LAST_STOCK_RESUMABLE_VERSION or version[0] > MAX_SUPPORTED_MAJOR:
        return False
    return _hub_is_patchable() and _lock_is_honoured()


def restore_resumable_partials() -> bool:
    """Patch huggingface_hub in THIS process. Returns whether it took effect.

    Idempotent, and a no-op wherever :func:`can_restore_partials` says no.
    """
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

        # A repo can be XET-backed and still come down over HTTP when hf_xet is absent or disabled,
        # so the question is whether XET will be used, not whether its metadata exists. XET has its
        # own writer and cannot resume from a partial either way.
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
        with incomplete_path.open("ab") as handle:
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
    # getattr: a test that replaced the probe outright has no cache to clear.
    clear = getattr(_lock_is_honoured_at, "cache_clear", None)
    if clear is not None:
        clear()


def reset_probe_cache_for_tests() -> None:
    invalidate_probe_cache()
