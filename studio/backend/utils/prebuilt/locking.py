# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-process install lock (concurrent setup runs share one UNSLOTH_HOME)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .errors import BusyInstallConflict

try:
    from filelock import FileLock, Timeout as FileLockTimeout
except ImportError:
    FileLock = None
    FileLockTimeout = None

INSTALL_LOCK_TIMEOUT_SECONDS = 300


def _windows_hidden_kwargs() -> dict[str, object]:
    if sys.platform != "win32":
        return {}
    kwargs: dict[str, object] = {}
    flag = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if flag:
        kwargs["creationflags"] = flag
    return kwargs


def install_lock_path(install_dir: Path) -> Path:
    return install_dir.parent / f".{install_dir.name}.install.lock"


def _pid_is_alive(pid: int) -> bool:
    """Best-effort liveness check that never signals the process on Windows."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output = True,
                text = True,
                timeout = 5,
                **_windows_hidden_kwargs(),
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            return True
        return f'"{pid}"' in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except ValueError:
        return False
    return True


@contextmanager
def install_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents = True, exist_ok = True)
    if FileLock is None:
        fd: int | None = None
        deadline = time.monotonic() + INSTALL_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(fd, f"{os.getpid()}\n".encode())
                os.fsync(fd)
                break
            except FileExistsError:
                try:
                    raw = lock_path.read_text().strip()
                except FileNotFoundError:
                    continue
                stale = False
                if raw:
                    try:
                        stale = not _pid_is_alive(int(raw))
                    except ValueError:
                        stale = True
                if stale:
                    # Rename before unlinking so only one racer removes the stale
                    # lock; a process recreating it loses the rename and waits.
                    try:
                        stale_path = lock_path.with_name(f"{lock_path.name}.stale.{os.getpid()}")
                        os.replace(str(lock_path), str(stale_path))
                        stale_path.unlink(missing_ok = True)
                    except (OSError, ValueError):
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise BusyInstallConflict(
                        f"timed out after {INSTALL_LOCK_TIMEOUT_SECONDS}s waiting for install lock: {lock_path}"
                    )
                time.sleep(0.5)
        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)
            lock_path.unlink(missing_ok = True)
        return

    try:
        with FileLock(str(lock_path), timeout = INSTALL_LOCK_TIMEOUT_SECONDS):
            yield
    except FileLockTimeout as exc:
        raise BusyInstallConflict(
            f"timed out after {INSTALL_LOCK_TIMEOUT_SECONDS}s waiting for install lock: {lock_path}"
        ) from exc
