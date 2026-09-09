# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fail-closed OS boundary for commands in persisted project workspaces.

Linux uses bubblewrap with an opened project root as its writable bind source.
macOS uses a sandbox-exec profile and enters the opened root descriptor before
exec. Windows remains unavailable until it has an equivalent filesystem
boundary. Static command classification is intentionally outside this module.
"""

from __future__ import annotations

import contextlib
import functools
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from utils.paths import ensure_dir, tmp_root

try:
    from .mutation import acquire_workspace_mutation_slot, release_workspace_mutation_slot
    EDIT_BOUNDARY_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name != __package__ + ".mutation":
        raise
    EDIT_BOUNDARY_AVAILABLE = False

    def acquire_workspace_mutation_slot(*args, **kwargs):
        raise ProjectExecutionUnavailable(
            "Confined file edit support must be installed before project commands can run."
        )

    def release_workspace_mutation_slot(*args, **kwargs):
        raise ProjectExecutionUnavailable("Confined file edit support is unavailable.")


_MAX_WORKSPACE_CHECK_ENTRIES = 100_000
_MAX_WORKSPACE_CHECK_SECONDS = 10.0


class ProjectExecutionUnavailable(RuntimeError):
    """The host cannot enforce the required project command boundary."""


_MACOS_SANDBOX = "/usr/bin/sandbox-exec"
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
(allow file-read* (subpath "/Library/Frameworks/Python.framework"))
(allow file-read-metadata (literal "/Library") (literal "/Library/Frameworks"))
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
    # Project commands must not let a caller-controlled PATH or environment
    # variable select the sandbox executable. Resolve only through the host's
    # default system search path, then bind the returned path to a regular file.
    raw = shutil.which("bwrap", path = os.defpath)
    if not raw:
        return None
    try:
        resolved = Path(raw).resolve(strict = True)
        metadata = resolved.stat(follow_symlinks = False)
    except (OSError, RuntimeError, ValueError):
        return None
    if not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
        return None
    return str(resolved)


@functools.lru_cache(maxsize = 4)
def _probe_backend(platform: str, executable: str) -> bool:
    """Exercise the same kernel facility used by real commands."""
    try:
        if platform == "darwin":
            with tempfile.TemporaryDirectory(prefix = "unsloth-boundary-probe-") as root:
                probe = subprocess.run(
                    [
                        executable,
                        "-D",
                        f"PROJECT_ROOT={root}",
                        "-D",
                        f"SCRATCH_ROOT={root}",
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
    if not EDIT_BOUNDARY_AVAILABLE:
        return ExecutionBoundaryStatus(
            False,
            None,
            "Confined file edit support must be installed before project commands can run.",
        )
    name = _platform_name(platform)
    if name == "darwin":
        if not os.path.isfile(_MACOS_SANDBOX) or not os.access(_MACOS_SANDBOX, os.X_OK):
            return ExecutionBoundaryStatus(False, None, "macOS sandbox-exec is unavailable.")
        if probe and not _probe_backend(name, _MACOS_SANDBOX):
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
        False, None, f"Project command execution is unsupported on platform {name!r}."
    )


def _open_directory(path: Path) -> tuple[int, tuple[int, int]]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ProjectExecutionUnavailable("The project workspace is not a directory.")
        return descriptor, (int(metadata.st_dev), int(metadata.st_ino))
    except ProjectExecutionUnavailable:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ProjectExecutionUnavailable(
            "The project folder changed before the command could start."
        ) from exc


def _assert_regular_file_links_are_internal(root_fd: int, root_identity: tuple[int, int]) -> None:
    """Reject project files with a hardlink outside the writable boundary."""
    try:
        root_metadata = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or (
                int(root_metadata.st_dev),
                int(root_metadata.st_ino),
            )
            != root_identity
        ):
            raise ProjectExecutionUnavailable(
                "The project folder changed before the command could start."
            )
        root_device = int(root_metadata.st_dev)
        links: dict[tuple[int, int], list[int]] = {}
        visited = 0
        deadline = time.monotonic() + _MAX_WORKSPACE_CHECK_SECONDS

        def fail_walk(error: OSError) -> None:
            raise error

        for _directory, _subdirectories, names, directory_fd in os.fwalk(
            ".",
            topdown = True,
            onerror = fail_walk,
            follow_symlinks = False,
            dir_fd = root_fd,
        ):
            visited += 1 + len(_subdirectories) + len(names)
            if visited > _MAX_WORKSPACE_CHECK_ENTRIES or time.monotonic() > deadline:
                raise ProjectExecutionUnavailable(
                    "The workspace is too large to check safely before running a command."
                )
            directory_metadata = os.fstat(directory_fd)
            if int(directory_metadata.st_dev) != root_device:
                raise ProjectExecutionUnavailable(
                    "Mounted directories inside project workspaces cannot run commands."
                )
            for name in names:
                if time.monotonic() > deadline:
                    raise ProjectExecutionUnavailable("The workspace safety check timed out.")
                metadata = os.stat(name, dir_fd = directory_fd, follow_symlinks = False)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink == 1:
                    continue
                identity = (int(metadata.st_dev), int(metadata.st_ino))
                record = links.setdefault(identity, [0, int(metadata.st_nlink)])
                if record[1] != int(metadata.st_nlink):
                    raise ProjectExecutionUnavailable(
                        "The project folder changed before the command could start."
                    )
                record[0] += 1
        if any(observed != total for observed, total in links.values()):
            raise ProjectExecutionUnavailable(
                "A project file is hard-linked outside the workspace. Remove the external "
                "hardlink before running commands."
            )
    except ProjectExecutionUnavailable:
        raise
    except OSError as exc:
        raise ProjectExecutionUnavailable(
            "The project folder changed while command safety was checked."
        ) from exc


def _validate_policy_path(path: Path) -> Path:
    if any(character in str(path) for character in ("\x00", "\n", "\r")):
        raise ProjectExecutionUnavailable(
            "Project command execution does not support control characters in paths."
        )
    return path


def _install_python_wrapper(scratch: Path) -> Path:
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
        os.write(descriptor, f"#!/bin/sh\nexec '{target}' \"$@\"\n".encode())
    finally:
        os.close(descriptor)
    return wrapper_dir


def _compose_preexec(
    existing: Optional[Callable[[], None]], root_fd: int, identity: tuple[int, int]
) -> Callable[[], None]:
    def prepare() -> None:
        if existing is not None:
            existing()
        metadata = os.fstat(root_fd)
        if (int(metadata.st_dev), int(metadata.st_ino)) != identity:
            raise OSError("project root identity changed")
        os.fchdir(root_fd)

    return prepare


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


_LINUX_SYSTEM_READ_PATHS = (
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/etc/alternatives",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/localtime",
    "/etc/ssl/certs",
)


def _linux_system_mounts() -> list[tuple[Path, Path, bool]]:
    """Return the fixed, non-user host paths required by normal runtimes."""
    mounts: list[tuple[Path, Path, bool]] = []
    covered: list[Path] = []
    for raw in _LINUX_SYSTEM_READ_PATHS:
        destination = Path(raw)
        if any(destination == parent or _path_is_within(destination, parent) for parent in covered):
            continue
        try:
            source = destination.resolve(strict = True)
        except (OSError, RuntimeError, ValueError):
            continue
        if source.is_dir():
            directory = True
        elif source.is_file():
            directory = False
        else:
            continue
        mounts.append((source, destination, directory))
        if directory:
            covered.append(destination)
    return mounts


def _linux_mountpoint(root: Path, destination: Path, *, directory: bool) -> None:
    """Pre-create one mountpoint inside the read-only empty Linux root."""
    if not destination.is_absolute() or destination == Path("/"):
        raise ProjectExecutionUnavailable("Linux sandbox mount destinations must be absolute.")
    target = root.joinpath(*destination.parts[1:])
    if directory:
        target.mkdir(mode = 0o700, parents = True, exist_ok = True)
        return
    target.parent.mkdir(mode = 0o700, parents = True, exist_ok = True)
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    os.close(descriptor)


class ProjectExecutionBoundary:
    """One opened, serialized, identity-bound project command boundary."""

    def __init__(
        self,
        root: Path | str,
        expected_identity: Optional[tuple[int, int]] = None,
    ) -> None:
        status = execution_boundary_status()
        if not status.available or status.backend is None:
            raise ProjectExecutionUnavailable(status.reason or "Project execution is unavailable.")
        if os.name != "posix":
            raise ProjectExecutionUnavailable(
                "Project command execution has no filesystem boundary on this platform."
            )
        self._closed = False
        self._slot = False
        self._git_mask_fd = None
        self._git_marker_identity = None
        self.backend = status.backend
        workspace = root if hasattr(root, "root") else None
        requested_root = Path(workspace.root if workspace is not None else root)
        if expected_identity is None and workspace is not None:
            expected_identity = (workspace.device_id, workspace.file_id)
        self.root = _validate_policy_path(requested_root.resolve(strict = True))
        self.runtime_root = _validate_policy_path(Path(sys.prefix).resolve(strict = True))
        self.base_runtime_root = _validate_policy_path(Path(sys.base_prefix).resolve(strict = True))
        self._root_fd, self.root_identity = _open_directory(self.root)
        if expected_identity is not None:
            try:
                expected = (int(expected_identity[0]), int(expected_identity[1]))
            except (IndexError, TypeError, ValueError) as exc:
                os.close(self._root_fd)
                raise ProjectExecutionUnavailable("Project root identity is invalid.") from exc
        else:
            expected = self.root_identity
        if self.root_identity != expected:
            os.close(self._root_fd)
            raise ProjectExecutionUnavailable(
                "The project folder identity changed before the command could start."
            )
        self._runtime_directories: list[tuple[Path, int]] = []
        self._linux_system_mounts: list[tuple[Path, Path, bool]] = []
        self._sandbox_root_fd: Optional[int] = None
        try:
            if self.backend == "bubblewrap":
                for runtime in (self.runtime_root, self.base_runtime_root):
                    if _path_is_within(runtime, self.root):
                        continue
                    if any(existing == runtime for existing, _fd in self._runtime_directories):
                        continue
                    descriptor, _identity = _open_directory(runtime)
                    self._runtime_directories.append((runtime, descriptor))
            self._container = _validate_policy_path(
                Path(
                    tempfile.mkdtemp(prefix = "run-", dir = str(ensure_dir(tmp_root() / "agent-exec")))
                ).resolve(strict = True)
            )
            self.scratch = self._container / "scratch"
            self.scratch.mkdir(mode = 0o700)
            self._wrapper_dir = _install_python_wrapper(self.scratch)
            self._scratch_fd, self._scratch_identity = _open_directory(self.scratch)
            if self.backend == "bubblewrap":
                sandbox_root = self._container / "root"
                sandbox_root.mkdir(mode = 0o700)
                self._linux_system_mounts = _linux_system_mounts()
                for _source, destination, directory in self._linux_system_mounts:
                    _linux_mountpoint(sandbox_root, destination, directory = directory)
                for destination in (
                    Path("/proc"),
                    Path("/dev"),
                    self.root,
                    self.scratch,
                    *(runtime for runtime, _fd in self._runtime_directories),
                ):
                    _linux_mountpoint(sandbox_root, destination, directory = True)
                self._sandbox_root = sandbox_root
                self._sandbox_root_fd, self._sandbox_root_identity = _open_directory(sandbox_root)
        except Exception:
            for _runtime, descriptor in self._runtime_directories:
                os.close(descriptor)
            if self._sandbox_root_fd is not None:
                os.close(self._sandbox_root_fd)
            os.close(self._root_fd)
            if hasattr(self, "_container"):
                shutil.rmtree(self._container, ignore_errors = True)
            raise

    def protect_git_metadata(self) -> None:
        """Mask a task checkout's root .git file with a read-only empty mount."""
        if self.backend != "bubblewrap" or not self._slot:
            raise ProjectExecutionUnavailable(
                "Task Git protection requires an owned Linux boundary."
            )
        self.recheck()
        metadata = os.stat(".git", dir_fd = self._root_fd, follow_symlinks = False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ProjectExecutionUnavailable("The task Git marker is unsafe.")
        if self._git_mask_fd is None:
            self._git_mask_fd = os.open(
                "task-git-mask",
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o400,
                dir_fd = self._scratch_fd,
            )
            self._git_marker_identity = (metadata.st_dev, metadata.st_ino)

    @classmethod
    def open(
        cls,
        root: Path | str,
        expected_identity: Optional[tuple[int, int]] = None,
    ) -> "ProjectExecutionBoundary":
        return cls(root, expected_identity)

    def __enter__(self) -> "ProjectExecutionBoundary":
        return self

    def __exit__(self, _kind, _value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self.release_execution_slot()
        self._closed = True
        descriptors = [
            self._scratch_fd,
            self._root_fd,
            *((self._git_mask_fd,) if self._git_mask_fd is not None else ()),
            *(fd for _runtime, fd in self._runtime_directories),
        ]
        if self._sandbox_root_fd is not None:
            descriptors.append(self._sandbox_root_fd)
        for descriptor in descriptors:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        shutil.rmtree(self._container, ignore_errors = True)

    def acquire_execution_slot(self, cancel_event = None) -> bool:
        if self._slot:
            return True
        self.recheck()
        if not acquire_workspace_mutation_slot(self.root_identity, cancel_event):
            return False
        try:
            self.recheck()
        except Exception:
            release_workspace_mutation_slot(self.root_identity)
            raise
        self._slot = True
        return True

    def release_execution_slot(self) -> None:
        if not self._slot:
            return
        self._slot = False
        release_workspace_mutation_slot(self.root_identity)

    @staticmethod
    def _assert_path_identity(path: Path, descriptor: int, expected: tuple[int, int]) -> None:
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
        self._assert_path_identity(self.root, self._root_fd, self.root_identity)
        if self._git_marker_identity is not None:
            marker = os.stat(".git", dir_fd = self._root_fd, follow_symlinks = False)
            if (
                not stat.S_ISREG(marker.st_mode)
                or (marker.st_dev, marker.st_ino) != self._git_marker_identity
            ):
                raise ProjectExecutionUnavailable(
                    "The task Git marker changed before command execution."
                )
        self._assert_path_identity(self.scratch, self._scratch_fd, self._scratch_identity)
        if self._sandbox_root_fd is not None:
            self._assert_path_identity(
                self._sandbox_root,
                self._sandbox_root_fd,
                self._sandbox_root_identity,
            )

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
            options = [
                executable,
                "--die-with-parent",
                "--unshare-all",
                "--ro-bind",
                f"/proc/self/fd/{self._sandbox_root_fd}",
                "/",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
            ]
            for source, destination, _directory in self._linux_system_mounts:
                options.extend(["--ro-bind", str(source), str(destination)])
            exposed = [
                (self.root, self._root_fd, "--bind"),
                (self.scratch, self._scratch_fd, "--bind"),
                *((runtime, fd, "--ro-bind") for runtime, fd in self._runtime_directories),
            ]
            for destination, descriptor, mode in exposed:
                options.extend([mode, f"/proc/self/fd/{descriptor}", str(destination)])
            if self._git_mask_fd is not None:
                options.extend(
                    ["--ro-bind", f"/proc/self/fd/{self._git_mask_fd}", str(self.root / ".git")]
                )
            options.extend(["--chdir", str(self.root), "--", *command])
            return options
        raise ProjectExecutionUnavailable("Project command execution is unavailable.")

    def popen_kwargs(self, preexec_fn: Optional[Callable[[], None]] = None) -> dict:
        self.recheck()
        _assert_regular_file_links_are_internal(self._root_fd, self.root_identity)
        self.recheck()
        descriptors = [
            self._root_fd,
            self._scratch_fd,
            *((self._git_mask_fd,) if self._git_mask_fd is not None else ()),
            *(fd for _runtime, fd in self._runtime_directories),
        ]
        if self._sandbox_root_fd is not None:
            descriptors.append(self._sandbox_root_fd)
        if self.backend == "sandbox-exec":
            return {
                "cwd": None,
                "pass_fds": descriptors,
                "preexec_fn": _compose_preexec(preexec_fn, self._root_fd, self.root_identity),
            }
        if self.backend == "bubblewrap":
            options = {"cwd": "/", "pass_fds": descriptors}
            if preexec_fn is not None:
                options["preexec_fn"] = preexec_fn
            return options
        raise ProjectExecutionUnavailable("Project command execution is unavailable.")


__all__ = [
    "ExecutionBoundaryStatus",
    "ProjectExecutionBoundary",
    "ProjectExecutionUnavailable",
    "execution_boundary_status",
]
