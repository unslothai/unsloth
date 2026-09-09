# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Process-shared project locks retained until native process trees are dead."""

import contextlib
import hashlib
import os
import stat
import threading
import time
from typing import Optional


def _project_execution_fence_path(fence_id: str) -> str:
    if (
        not isinstance(fence_id, str)
        or not fence_id
        or len(fence_id.encode("utf-8", errors = "strict")) > 1024
    ):
        raise ValueError("Project execution fence identity is invalid.")
    from utils.paths.storage_roots import studio_root  # noqa: PLC0415

    directory = os.path.join(str(studio_root()), "project-execution-fences")
    os.makedirs(directory, mode = 0o700, exist_ok = True)
    directory_metadata = os.lstat(directory)
    if not stat.S_ISDIR(directory_metadata.st_mode) or stat.S_ISLNK(directory_metadata.st_mode):
        raise RuntimeError("Project execution fence directory is unsafe.")
    # Keep independent projects on distinct process-shared lock inodes.
    digest = hashlib.sha256(fence_id.encode("utf-8")).hexdigest()
    return os.path.join(directory, f"{digest}.lock")


def _acquire_project_execution_fence(
    fence_id: str, cancel_event: Optional[threading.Event], deadline: float
) -> int:
    """Acquire a process-wide finalizer fence until the prior tree is dead."""
    if os.name != "posix":
        raise RuntimeError("Project execution fencing is unavailable on this platform.")
    try:
        import fcntl  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - non-POSIX fails above
        raise RuntimeError("Project execution fencing is unavailable on this platform.") from exc
    path = _project_execution_fence_path(fence_id)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError("Project execution fence file is unsafe.")
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Project execution fence wait was cancelled.")
            if time.monotonic() >= deadline:
                raise TimeoutError("Project execution fence wait exceeded its deadline.")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return descriptor
            except BlockingIOError:
                if cancel_event is not None:
                    cancel_event.wait(min(0.05, max(0.0, deadline - time.monotonic())))
                else:
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    except BaseException:
        os.close(descriptor)
        raise


def _release_project_execution_fence(descriptor: int) -> None:
    try:
        import fcntl  # noqa: PLC0415
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)
