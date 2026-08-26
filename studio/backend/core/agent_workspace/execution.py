# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fail-closed OS boundary for commands that run in project workspaces.

The project root is opened before spawn and retained as a directory descriptor.
That descriptor is the cwd on macOS and the bind source on Linux, so replacing
the persisted path between validation and exec cannot redirect a command into a
different directory. Filesystem policy is enforced by sandbox-exec on macOS and
bubblewrap on Linux. Other platforms are intentionally unavailable until they
have an equivalent boundary.
"""

from __future__ import annotations

import functools
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from utils.paths import ensure_dir, tmp_root


class ProjectExecutionUnavailable(RuntimeError):
    """Raised when this host cannot safely run a project command."""


_MACOS_SANDBOX = "/usr/bin/sandbox-exec"
_EXECUTION_CONDITION = threading.Condition()
_ACTIVE_EXECUTION_ROOTS: set[tuple[int, int]] = set()
_MACOS_PROFILE = """
(version 1)
(deny default)
(allow process*)
(allow sysctl-read)
(allow syscall-unix
       (syscall-number SYS___mac_syscall)
       (syscall-number SYS_getfsstat SYS_getfsstat64)
       (syscall-number SYS_map_with_linking_np)
       (syscall-number SYS_open SYS_openat)
       (syscall-number SYS_fstatat SYS_fstatat64)
       (syscall-number SYS_dup))
(allow system-fcntl
       (fcntl-command F_ADDFILESIGS_RETURN F_CHECK_LV F_GETPATH))
(with-filter (mac-policy-name "Sandbox")
  (allow system-mac-syscall (mac-syscall-number 2)))
(allow file-read* file-test-existence (literal "/"))
(allow file-read* (subpath "/System"))
(allow file-read* (subpath "/private/preboot/Cryptexes"))
(allow file-read* (subpath "/usr"))
(allow file-read* (subpath "/bin"))
(allow file-read* (subpath "/sbin"))
(allow file-read* (subpath "/Library/Apple"))
(allow file-read* (subpath "/Library/Developer"))
(allow file-read* (subpath "/Applications/Xcode.app"))
(allow file-read* (subpath "/opt/homebrew/bin"))
(allow file-read* (subpath "/opt/homebrew/sbin"))
(allow file-read* (subpath "/opt/homebrew/lib"))
(allow file-read* (subpath "/opt/homebrew/Cellar"))
(allow file-read* (subpath "/opt/homebrew/opt"))
(allow file-read* (subpath "/opt/homebrew/share"))
(allow file-read* (literal "/private/etc/localtime"))
(allow file-read* (literal "/private/etc/passwd"))
(allow file-read* (literal "/private/etc/protocols"))
(allow file-read* (literal "/private/etc/services"))
(allow file-read* (subpath "/private/var/db/timezone"))
(allow file-read* (subpath "/private/var/select"))
(allow file-read* (literal "/dev/random"))
(allow file-read* (literal "/dev/urandom"))
(allow file-read* (literal "/dev/zero"))
(allow file-read* (subpath "/dev/fd"))
(allow file-read* (subpath (param "PROJECT_ROOT")))
(allow file-read* (subpath (param "SCRATCH_ROOT")))
(allow file-read* (subpath (param "RUNTIME_ROOT")))
(allow file-read* (subpath (param "BASE_RUNTIME_ROOT")))
(allow file-write* (subpath (param "PROJECT_ROOT")))
(allow file-write* (subpath (param "SCRATCH_ROOT")))
(allow file-write* (literal "/dev/null"))
(deny network*)
""".strip()


@dataclass(frozen = True)
class ExecutionBoundaryStatus:
    available: bool
    backend: Optional[str]
    reason: Optional[str]


def _platform_name(platform: Optional[str] = None) -> str:
    value = platform or sys.platform
    if value.startswith("linux"):
        return "linux"
    if value == "darwin":
        return "darwin"
    if value in {"win32", "cygwin", "msys"}:
        return "windows"
    return value


def _bubblewrap_path() -> Optional[str]:
    override = (os.environ.get("UNSLOTH_STUDIO_BWRAP") or "").strip()
    if override:
        candidate = Path(override)
        if candidate.is_absolute() and candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        return None
    candidate = shutil.which("bwrap")
    if not candidate:
        return None
    try:
        resolved = Path(candidate).resolve(strict = True)
    except (OSError, RuntimeError):
        return None
    return str(resolved) if resolved.is_file() and os.access(resolved, os.X_OK) else None


@functools.lru_cache(maxsize = 4)
def _probe_backend(platform: str, executable: str) -> bool:
    """Exercise the same kernel facility used by real runs before advertising it."""
    try:
        if platform == "darwin":
            with tempfile.TemporaryDirectory(prefix = "unsloth-boundary-probe-") as probe_root:
                probe = subprocess.run(
                    [
                        executable,
                        "-D",
                        f"PROJECT_ROOT={probe_root}",
                        "-D",
                        f"SCRATCH_ROOT={probe_root}",
                        "-D",
                        f"RUNTIME_ROOT={Path(sys.prefix).resolve()}",
                        "-D",
                        f"BASE_RUNTIME_ROOT={Path(sys.base_prefix).resolve()}",
                        "-p",
                        _MACOS_PROFILE,
                        "/usr/bin/true",
                    ],
                    stdin = subprocess.DEVNULL,
                    stdout = subprocess.DEVNULL,
                    stderr = subprocess.DEVNULL,
                    timeout = 3,
                    check = False,
                )
        elif platform == "linux":
            probe = subprocess.run(
                [
                    executable,
                    "--die-with-parent",
                    "--unshare-all",
                    "--ro-bind",
                    "/",
                    "/",
                    "--proc",
                    "/proc",
                    "--dev",
                    "/dev",
                    "--",
                    "/bin/true",
                ],
                stdin = subprocess.DEVNULL,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                timeout = 3,
                check = False,
            )
        else:
            return False
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def execution_boundary_status(
    platform: Optional[str] = None, *, probe: bool = True
) -> ExecutionBoundaryStatus:
    """Describe whether arbitrary project commands can be confined on this host."""
    name = _platform_name(platform)
    if name == "darwin":
        executable = _MACOS_SANDBOX
        if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
            return ExecutionBoundaryStatus(False, None, "macOS sandbox-exec is unavailable.")
        if probe and not _probe_backend(name, executable):
            return ExecutionBoundaryStatus(
                False, None, "macOS refused the project execution sandbox."
            )
        return ExecutionBoundaryStatus(True, "sandbox-exec", None)
    if name == "linux":
        executable = _bubblewrap_path()
        if executable is None:
            return ExecutionBoundaryStatus(
                False, None, "bubblewrap is not installed on this Linux host."
            )
        if probe and not _probe_backend(name, executable):
            return ExecutionBoundaryStatus(
                False,
                None,
                "bubblewrap cannot create the required namespaces on this Linux host.",
            )
        return ExecutionBoundaryStatus(True, "bubblewrap", None)
    if name == "windows":
        return ExecutionBoundaryStatus(
            False,
            None,
            "Project command execution is disabled until a Windows filesystem sandbox is available.",
        )
    return ExecutionBoundaryStatus(
        False,
        None,
        f"Project command execution is unsupported on platform {name!r}.",
    )


def _open_directory(path: Path) -> tuple[int, tuple[int, int]]:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProjectExecutionUnavailable(
            "The project folder changed before the command could start."
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ProjectExecutionUnavailable("The project workspace is not a directory.")
        return descriptor, (int(metadata.st_dev), int(metadata.st_ino))
    except Exception:
        os.close(descriptor)
        raise


def _assert_regular_file_links_are_internal(root_fd: int, root_identity: tuple[int, int]) -> None:
    """Reject regular-file inodes that also have a name outside the workspace.

    Path sandboxes cannot distinguish two hardlink names for the same inode. A
    command writing an in-root name could otherwise mutate an out-of-root file.
    Walk from the retained root descriptor without following symlinks and only
    accept multiply-linked files when every link is present below this root.
    """
    try:
        root_metadata = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or (int(root_metadata.st_dev), int(root_metadata.st_ino)) != root_identity
        ):
            raise ProjectExecutionUnavailable(
                "The project folder changed before the command could start."
            )
        root_device = int(root_metadata.st_dev)
        observed_links: dict[tuple[int, int], list[int]] = {}

        def fail_walk(error: OSError) -> None:
            raise error

        for _directory, _subdirectories, names, directory_fd in os.fwalk(
            ".",
            topdown = True,
            onerror = fail_walk,
            follow_symlinks = False,
            dir_fd = root_fd,
        ):
            directory_metadata = os.fstat(directory_fd)
            if int(directory_metadata.st_dev) != root_device:
                raise ProjectExecutionUnavailable(
                    "Mounted directories inside project workspaces cannot run commands."
                )
            for name in names:
                metadata = os.stat(
                    name,
                    dir_fd = directory_fd,
                    follow_symlinks = False,
                )
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink == 1:
                    continue
                if metadata.st_nlink < 1:
                    raise ProjectExecutionUnavailable(
                        "The project folder changed before the command could start."
                    )
                identity = (int(metadata.st_dev), int(metadata.st_ino))
                record = observed_links.setdefault(identity, [0, int(metadata.st_nlink)])
                if record[1] != int(metadata.st_nlink):
                    raise ProjectExecutionUnavailable(
                        "The project folder changed before the command could start."
                    )
                record[0] += 1
        if any(observed != total for observed, total in observed_links.values()):
            raise ProjectExecutionUnavailable(
                "A project file is hard-linked outside the workspace. "
                "Remove the external hardlink before running commands."
            )
        final_metadata = os.fstat(root_fd)
        if (int(final_metadata.st_dev), int(final_metadata.st_ino)) != root_identity:
            raise ProjectExecutionUnavailable(
                "The project folder changed before the command could start."
            )
    except ProjectExecutionUnavailable:
        raise
    except OSError as exc:
        raise ProjectExecutionUnavailable(
            "The project folder changed while command safety was checked."
        ) from exc


def acquire_workspace_execution_slot(identity: tuple[int, int], cancel_event = None) -> bool:
    """Acquire the process-local mutation slot for one opened workspace root."""
    with _EXECUTION_CONDITION:
        while identity in _ACTIVE_EXECUTION_ROOTS:
            if cancel_event is not None and cancel_event.is_set():
                return False
            _EXECUTION_CONDITION.wait(timeout = 0.05)
        if cancel_event is not None and cancel_event.is_set():
            return False
        _ACTIVE_EXECUTION_ROOTS.add(identity)
        return True


def release_workspace_execution_slot(identity: tuple[int, int]) -> None:
    """Release a mutation slot acquired for an opened workspace root."""
    with _EXECUTION_CONDITION:
        _ACTIVE_EXECUTION_ROOTS.discard(identity)
        _EXECUTION_CONDITION.notify_all()


def _validate_policy_path(path: Path) -> Path:
    rendered = str(path)
    if any(character in rendered for character in ("\x00", "\n", "\r")):
        raise ProjectExecutionUnavailable(
            "Project command execution does not support control characters in paths."
        )
    return path


def _install_python_wrapper(scratch: Path) -> Path:
    """Expose a non-symlink ``python`` command for virtualenv-based backends."""
    wrapper_dir = scratch / "bin"
    wrapper_dir.mkdir(mode = 0o700)
    wrapper = wrapper_dir / "python"
    target = str(Path(sys.executable).resolve(strict = True))
    if any(character in target for character in ("\x00", "\n", "\r", "'")):
        raise ProjectExecutionUnavailable("The Python runtime path cannot be represented safely.")
    descriptor = os.open(
        wrapper,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o700,
    )
    try:
        os.write(descriptor, f"#!/bin/sh\nexec '{target}' \"$@\"\n".encode("utf-8"))
    finally:
        os.close(descriptor)
    return wrapper_dir


def _compose_preexec(
    existing: Optional[Callable[[], None]], root_fd: int, identity: tuple[int, int]
) -> Callable[[], None]:
    def _prepare() -> None:
        if existing is not None:
            existing()
        metadata = os.fstat(root_fd)
        if (int(metadata.st_dev), int(metadata.st_ino)) != identity:
            raise OSError("project root identity changed")
        os.fchdir(root_fd)

    return _prepare


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _masked_parent_directories(path: Path, masked_root: Path) -> list[str]:
    """Directories bubblewrap must recreate after masking a home or temp tree."""
    if path == masked_root or not _path_is_within(path, masked_root):
        return []
    relative = path.relative_to(masked_root)
    current = masked_root
    directories: list[str] = []
    for part in relative.parts[:-1]:
        current = current / part
        directories.append(str(current))
    return directories


def _linux_masked_roots(home: Path, temp: Path) -> list[Path]:
    """Return the minimal host trees hidden from a bubblewrap child.

    A read-only bind of ``/`` still exposes pathname Unix sockets. In
    particular, container engines, desktop buses, credential agents, and
    service managers commonly publish sockets below ``/run`` (with
    ``/var/run`` pointing at the same tree). A private network namespace does
    not isolate those pathname sockets, so the runtime tree itself must be
    replaced.

    Callers recreate only parent directories for explicitly exposed
    descriptor-bound roots. Nothing else from a masked tree is copied into the
    sandbox.
    """
    candidates = (
        home,
        temp,
        Path("/tmp"),
        Path("/var/tmp"),
        Path("/dev/shm"),
        Path("/mnt"),
        Path("/media"),
        Path("/run"),
        Path("/var/run"),
        Path("/Volumes"),
    )
    canonical: list[Path] = []
    for candidate in candidates:
        if candidate == Path("/") or not candidate.exists():
            continue
        try:
            resolved = candidate.resolve(strict = True)
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved == Path("/") or resolved in canonical:
            continue
        canonical.append(resolved)

    # If one selected tree contains another, the outer tmpfs already hides it.
    minimal: list[Path] = []
    for candidate in sorted(canonical, key = lambda item: len(item.parts)):
        if any(candidate == parent or _path_is_within(candidate, parent) for parent in minimal):
            continue
        minimal.append(candidate)
    return minimal


class ProjectExecutionBoundary:
    """One opened and identity-bound project command boundary."""

    def __init__(
        self,
        root: Path,
        expected_identity: Optional[tuple[int, int]] = None,
    ):
        status = execution_boundary_status()
        if not status.available or status.backend is None:
            raise ProjectExecutionUnavailable(
                status.reason or "Project command execution is unavailable."
            )
        try:
            if root.is_symlink():
                raise ProjectExecutionUnavailable(
                    "Symbolic-link project roots cannot run commands."
                )
            resolved = root.resolve(strict = True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ProjectExecutionUnavailable(
                "The project folder changed before the command could start."
            ) from exc
        self.root = _validate_policy_path(resolved)
        self.runtime_root = _validate_policy_path(Path(sys.prefix).resolve(strict = True))
        self.base_runtime_root = _validate_policy_path(Path(sys.base_prefix).resolve(strict = True))
        self.backend = status.backend
        self._root_fd, self._root_identity = _open_directory(resolved)
        if expected_identity is not None and self._root_identity != (
            int(expected_identity[0]),
            int(expected_identity[1]),
        ):
            os.close(self._root_fd)
            raise ProjectExecutionUnavailable(
                "The project folder identity changed before the command could start."
            )
        self._runtime_directories: list[tuple[Path, int]] = []
        if self.backend == "bubblewrap":
            try:
                for runtime in (self.runtime_root, self.base_runtime_root):
                    if _path_is_within(runtime, self.root):
                        continue
                    if any(existing == runtime for existing, _fd in self._runtime_directories):
                        continue
                    descriptor, _identity = _open_directory(runtime)
                    self._runtime_directories.append((runtime, descriptor))
            except Exception:
                for _runtime, descriptor in self._runtime_directories:
                    os.close(descriptor)
                os.close(self._root_fd)
                raise
        self.scratch = _validate_policy_path(
            Path(
                tempfile.mkdtemp(prefix = "run-", dir = str(ensure_dir(tmp_root() / "agent-exec")))
            ).resolve(strict = True)
        )
        try:
            self._wrapper_dir = _install_python_wrapper(self.scratch)
            self._scratch_fd, self._scratch_identity = _open_directory(self.scratch)
        except Exception:
            for _runtime, descriptor in self._runtime_directories:
                os.close(descriptor)
            os.close(self._root_fd)
            shutil.rmtree(self.scratch, ignore_errors = True)
            raise
        self._closed = False
        self._execution_slot: Optional[tuple[int, int]] = None

    @classmethod
    def open(
        cls,
        root: Path | str,
        expected_identity: Optional[tuple[int, int]] = None,
    ) -> "ProjectExecutionBoundary":
        return cls(Path(root), expected_identity)

    def __enter__(self) -> "ProjectExecutionBoundary":
        return self

    def __exit__(self, _kind, _value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self.release_execution_slot()
        self._closed = True
        for descriptor in (
            self._scratch_fd,
            self._root_fd,
            *(fd for _runtime, fd in self._runtime_directories),
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass
        shutil.rmtree(self.scratch, ignore_errors = True)

    def acquire_execution_slot(self, cancel_event = None) -> bool:
        """Serialize commands that can mutate the same workspace root."""
        key = self._root_identity
        if self._execution_slot is not None:
            return True
        self.recheck()
        if not acquire_workspace_execution_slot(key, cancel_event):
            return False
        try:
            self.recheck()
        except Exception:
            release_workspace_execution_slot(key)
            raise
        self._execution_slot = key
        return True

    def release_execution_slot(self) -> None:
        key = self._execution_slot
        if key is None:
            return
        self._execution_slot = None
        release_workspace_execution_slot(key)

    def _assert_path_identity(self, path: Path, descriptor: int, expected: tuple[int, int]) -> None:
        try:
            current = path.stat(follow_symlinks = False)
            opened = os.fstat(descriptor)
        except OSError as exc:
            raise ProjectExecutionUnavailable(
                "The project folder changed before the command could start."
            ) from exc
        if (
            not stat.S_ISDIR(current.st_mode)
            or (int(current.st_dev), int(current.st_ino)) != expected
            or (int(opened.st_dev), int(opened.st_ino)) != expected
        ):
            raise ProjectExecutionUnavailable(
                "The project folder changed before the command could start."
            )

    def recheck(self) -> None:
        if self._closed:
            raise ProjectExecutionUnavailable("The project execution boundary is closed.")
        self._assert_path_identity(self.root, self._root_fd, self._root_identity)
        self._assert_path_identity(self.scratch, self._scratch_fd, self._scratch_identity)

    def apply_environment(self, env: dict[str, str]) -> dict[str, str]:
        isolated = dict(env)
        scratch = str(self.scratch)
        for name in (
            "HOME",
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
            "TMP",
            "TEMP",
            "TMPDIR",
        ):
            isolated[name] = scratch
        isolated["PATH"] = os.pathsep.join(
            part for part in (str(self._wrapper_dir), isolated.get("PATH", "")) if part
        )
        return isolated

    def wrap_argv(self, argv: Sequence[str]) -> list[str]:
        self.recheck()
        command = [str(part) for part in argv]
        if self.backend == "sandbox-exec":
            return [
                _MACOS_SANDBOX,
                "-D",
                f"PROJECT_ROOT={self.root}",
                "-D",
                f"SCRATCH_ROOT={self.scratch}",
                "-D",
                f"RUNTIME_ROOT={self.runtime_root}",
                "-D",
                f"BASE_RUNTIME_ROOT={self.base_runtime_root}",
                "-p",
                _MACOS_PROFILE,
                *command,
            ]
        if self.backend == "bubblewrap":
            executable = _bubblewrap_path()
            if executable is None:
                raise ProjectExecutionUnavailable(
                    "bubblewrap disappeared before the command could start."
                )
            home = Path.home().resolve()
            temp = Path(tempfile.gettempdir()).resolve()
            options = [
                executable,
                "--die-with-parent",
                "--unshare-all",
                "--ro-bind",
                "/",
                "/",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
            ]
            masked = _linux_masked_roots(home, temp)
            for hidden in masked:
                options.extend(["--tmpfs", str(hidden)])
            exposed = [
                (self.root, self._root_fd, "--bind"),
                (self.scratch, self._scratch_fd, "--bind"),
                *(
                    (runtime, descriptor, "--ro-bind")
                    for runtime, descriptor in self._runtime_directories
                ),
            ]
            created_directories: set[str] = set()
            for destination, _descriptor, _mode in exposed:
                for hidden in masked:
                    for directory in _masked_parent_directories(destination, hidden):
                        if directory not in created_directories:
                            options.extend(["--dir", directory])
                            created_directories.add(directory)
            for destination, descriptor, mode in exposed:
                options.extend([mode, f"/proc/self/fd/{descriptor}", str(destination)])
            options.extend(
                [
                    "--chdir",
                    str(self.root),
                    "--",
                    *command,
                ]
            )
            return options
        raise ProjectExecutionUnavailable("Project command execution is unavailable.")

    def popen_kwargs(self, preexec_fn: Optional[Callable[[], None]] = None) -> dict:
        """Arguments that keep the opened directory descriptors valid through exec."""
        self.recheck()
        _assert_regular_file_links_are_internal(self._root_fd, self._root_identity)
        self.recheck()
        descriptors = (
            self._root_fd,
            self._scratch_fd,
            *(fd for _runtime, fd in self._runtime_directories),
        )
        if self.backend == "sandbox-exec":
            return {
                "cwd": None,
                "pass_fds": descriptors,
                "preexec_fn": _compose_preexec(preexec_fn, self._root_fd, self._root_identity),
            }
        if self.backend == "bubblewrap":
            options = {"cwd": "/", "pass_fds": descriptors}
            if preexec_fn is not None:
                options["preexec_fn"] = preexec_fn
            return options
        raise ProjectExecutionUnavailable("Project command execution is unavailable.")


__all__ = [
    "acquire_workspace_execution_slot",
    "ExecutionBoundaryStatus",
    "ProjectExecutionBoundary",
    "ProjectExecutionUnavailable",
    "execution_boundary_status",
    "release_workspace_execution_slot",
]
