# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fail-closed OS isolation for Studio's local Python and Terminal tools."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import platform
import shutil
import socket
import stat
import struct
import subprocess
import sys
import sysconfig
import tempfile
import threading
from dataclasses import dataclass, field, replace
from typing import Any, BinaryIO, Callable, Literal, Protocol

from loggers import get_logger

logger = get_logger(__name__)

_SCAN_ENTRY_LIMIT = 100_000
_REJECTED_RUNTIME_ENTRY_TYPES = "Unix sockets, FIFOs, block devices, or character devices"
_PROBE_TIMEOUT_SECONDS = 8
_PROBE_TOKEN = "UNSLOTH_OS_SANDBOX_PROBE_OK"
_PROBE_UDP_TOKEN = b"UNSLOTH_OS_SANDBOX_UDP_PROBE"
_AF_VSOCK = 40
_LINUX_SECCOMP_ABIS = {
    "aarch64": (0xC00000B7, 0, 198, 198, 199, 199),
    "arm64": (0xC00000B7, 0, 198, 198, 199, 199),
    "amd64": (0xC000003E, 0x40000000, 41, 0x40000029, 53, 0x40000035),
    "x86_64": (0xC000003E, 0x40000000, 41, 0x40000029, 53, 0x40000035),
}
_LINUX_SYSTEM_ROOTS = (
    "/usr/bin",
    "/usr/sbin",
    "/usr/lib",
    "/usr/lib64",
    "/usr/share",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
)
_LINUX_ETC_FILES = (
    "/etc/alternatives",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/localtime",
    "/etc/nsswitch.conf",
)
_WSL_HIDDEN_PATHS = ("/usr/lib/wsl",)


class SandboxUnavailableError(RuntimeError):
    """The required native sandbox cannot safely launch this tool call."""


@dataclass(frozen = True)
class SandboxCapability:
    backend: str
    qualified: bool
    reason: str
    available: bool | None = None
    transient: bool = False
    environment: str = "unknown"
    protection_state: str = "unavailable"
    profile_id: str = "none"
    limitations: tuple[str, ...] = ()
    probe_generation: str = ""
    environment_fingerprint: str = ""
    remediation: str = "Use Limited mode only for a trusted task, or install a qualified backend."
    retryable: bool = False


ToolExecutionMode = Literal["os_isolation_required", "limited", "full"]


@dataclass(frozen = True)
class ToolExecutionRecord:
    requested_mode: ToolExecutionMode
    effective_mode: ToolExecutionMode
    environment: str
    backend: str
    profile_id: str
    probe_generation: str
    os_isolation: bool
    retained_safeguards: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_mode": self.requested_mode,
            "effective_mode": self.effective_mode,
            "environment": self.environment,
            "backend": self.backend,
            "profile_id": self.profile_id,
            "probe_generation": self.probe_generation,
            "os_isolation": self.os_isolation,
            "retained_safeguards": list(self.retained_safeguards),
            "limitations": list(self.limitations),
        }


@dataclass(frozen = True)
class ToolLaunchPlan:
    """Complete policy inputs for one final Python or Terminal process launch."""

    argv: tuple[str, ...]
    workdir: str
    env: dict[str, str]
    preexec_fn: Callable[[], None] | None = None
    launcher_preexec_fn: Callable[[], None] | None = None
    requested_mode: ToolExecutionMode = "os_isolation_required"
    current_subject: str | None = None
    tool_ui_session_id: str | None = None
    limited_grant: str | None = None
    timeout_seconds: int | None = None
    close_fds: bool = True
    terminate_descendants: bool = True


# Compatibility for focused tests and callers written against the first narrow
# sandbox checkpoint. New integrations should use ToolLaunchPlan.
SandboxLaunchSpec = ToolLaunchPlan


@dataclass
class PreparedSandboxLaunch:
    """A native sandbox argv plus resources owned until the process exits."""

    argv: tuple[str, ...]
    workdir: str
    env: dict[str, str]
    preexec_fn: Callable[[], None] | None
    backend: str
    execution_record: ToolExecutionRecord | None = None
    pass_fds: tuple[int, ...] = ()
    owned_files: list[BinaryIO] = field(default_factory = list)
    cleanup_paths: list[str] = field(default_factory = list)
    timeout_seconds: int | None = None
    close_fds: bool = True
    terminate_descendants: bool = True
    spawn_callback: Callable[["PreparedSandboxLaunch", dict[str, Any]], object] | None = None
    cleanup_callbacks: list[Callable[[], None]] = field(default_factory = list)
    cleanup_diagnostics: list[str] = field(default_factory = list)

    def cleanup(self) -> None:
        while self.cleanup_callbacks:
            callback = self.cleanup_callbacks.pop()
            try:
                callback()
            except Exception as exc:  # noqa: BLE001 - cleanup continues in LIFO order
                diagnostic = f"{type(exc).__name__}: {exc}"
                self.cleanup_diagnostics.append(diagnostic)
                logger.warning("Sandbox cleanup failed: %s", diagnostic, exc_info = True)
        while self.owned_files:
            try:
                self.owned_files.pop().close()
            except Exception as exc:  # noqa: BLE001 - cleanup must continue
                diagnostic = f"{type(exc).__name__}: {exc}"
                self.cleanup_diagnostics.append(diagnostic)
                logger.warning("Could not close sandbox-owned file: %s", diagnostic, exc_info = True)
        while self.cleanup_paths:
            path = self.cleanup_paths.pop()
            try:
                shutil.rmtree(path)
            except OSError:
                self.cleanup_diagnostics.append(f"could not remove private sandbox path: {path}")
                logger.warning("Could not remove private sandbox path %s", path, exc_info = True)


def spawn_prepared_launch(prepared: PreparedSandboxLaunch, **popen_kwargs: Any) -> object:
    """Spawn exactly one prepared launch, using its backend-owned launcher when set."""
    if prepared.spawn_callback is not None:
        return prepared.spawn_callback(prepared, popen_kwargs)
    return subprocess.Popen(prepared.argv, **popen_kwargs)


class SandboxBackend(Protocol):
    identity: str

    def probe(self) -> SandboxCapability: ...

    def prepare(self, spec: ToolLaunchPlan) -> PreparedSandboxLaunch: ...


def _linux_seccomp_filter() -> BinaryIO:
    """Compile a minimal filter for host-channel socket families Bubblewrap cannot hide."""
    abi = _LINUX_SECCOMP_ABIS.get(platform.machine().lower())
    if abi is None:
        raise SandboxUnavailableError(
            f"Linux architecture {platform.machine() or 'unknown'} is not qualified for seccomp"
        )
    audit_arch, x32_syscall_bit, socket_nr, socket_alt_nr, socketpair_nr, socketpair_alt_nr = abi
    # struct seccomp_data: nr@0, arch@4, args[0]@16. Any ABI change kills the
    # process; AF_VSOCK and io_uring are denied while AF_UNIX remains available.
    load_word = 0x20
    jump_equal = 0x15
    jump_bits_set = 0x45
    return_value = 0x06
    kill_process = 0x80000000
    return_errno = 0x00050000 | errno.EPERM
    allow = 0x7FFF0000
    io_uring_setup_nr = 425
    instructions = (
        (load_word, 0, 0, 4),
        (jump_equal, 1, 0, audit_arch),
        (return_value, 0, 0, kill_process),
        (load_word, 0, 0, 0),
        (jump_bits_set, 7, 0, x32_syscall_bit),
        (jump_equal, 6, 0, io_uring_setup_nr),
        (jump_equal, 3, 0, socket_nr),
        (jump_equal, 2, 0, socket_alt_nr),
        (jump_equal, 1, 0, socketpair_nr),
        (jump_equal, 0, 3, socketpair_alt_nr),
        (load_word, 0, 0, 16),
        (jump_equal, 0, 1, _AF_VSOCK),
        (return_value, 0, 0, return_errno),
        (return_value, 0, 0, allow),
    )
    stream = tempfile.TemporaryFile(prefix = "unsloth-sandbox-seccomp-")
    try:
        stream.write(b"".join(struct.pack("=HBBI", *instruction) for instruction in instructions))
        stream.flush()
        stream.seek(0)
    except Exception:
        stream.close()
        raise
    return stream


def _contained(
    path: str,
    root: str,
    *,
    strict: bool = False,
) -> bool:
    try:
        common = os.path.commonpath((os.path.realpath(path), os.path.realpath(root)))
    except (OSError, ValueError):
        return False
    same = common == os.path.realpath(root)
    return same and (not strict or os.path.realpath(path) != os.path.realpath(root))


def _lexically_contained(path: str, root: str) -> bool:
    """Whether the sandbox destination spelling is already under a mounted root."""
    try:
        return os.path.commonpath(
            (os.path.abspath(path), os.path.abspath(root))
        ) == os.path.abspath(root)
    except ValueError:
        return False


def _validate_workdir(workdir: str) -> str:
    """Reject host IPC, device nodes, hardlinks, and nested mounts before binding."""
    wd = os.path.realpath(workdir)
    if not os.path.isabs(wd) or not os.path.isdir(wd) or os.path.dirname(wd) == wd:
        raise SandboxUnavailableError("the session workdir is not a safe canonical directory")

    try:
        root_device = os.lstat(wd).st_dev
    except OSError as exc:
        raise SandboxUnavailableError("the session workdir cannot be inspected") from exc
    entries = 0
    link_counts: dict[tuple[int, int], int] = {}
    link_totals: dict[tuple[int, int], int] = {}
    link_paths: dict[tuple[int, int], str] = {}

    def walk_error(exc: OSError) -> None:
        raise SandboxUnavailableError(
            f"the session workdir cannot be fully inspected: {exc.filename or wd}"
        ) from exc

    for base, dirs, names in os.walk(wd, followlinks = False, onerror = walk_error):
        for name in [*dirs, *names]:
            entries += 1
            if entries > _SCAN_ENTRY_LIMIT:
                raise SandboxUnavailableError(
                    f"the session workdir exceeds the {_SCAN_ENTRY_LIMIT:,}-entry safety scan limit"
                )
            path = os.path.join(base, name)
            try:
                info = os.lstat(path)
            except OSError as exc:
                raise SandboxUnavailableError(
                    f"the session workdir changed during its safety scan: {path}"
                ) from exc
            mode = info.st_mode
            if info.st_dev != root_device:
                raise SandboxUnavailableError(
                    f"the session workdir crosses a filesystem boundary: {path}"
                )
            if stat.S_ISLNK(mode):
                if name in dirs:
                    dirs.remove(name)
                continue
            if stat.S_ISDIR(mode):
                continue
            if stat.S_ISREG(mode):
                if info.st_nlink > 1:
                    key = (info.st_dev, info.st_ino)
                    link_counts[key] = link_counts.get(key, 0) + 1
                    link_totals[key] = max(link_totals.get(key, 0), info.st_nlink)
                    link_paths.setdefault(key, path)
                continue
            raise SandboxUnavailableError(
                f"the session workdir contains {_REJECTED_RUNTIME_ENTRY_TYPES}: {path}"
            )

    external_links = [
        link_paths[key] for key, count in link_counts.items() if count < link_totals[key]
    ]
    if external_links:
        raise SandboxUnavailableError(
            f"the session workdir contains a file hard-linked outside it: {external_links[0]}"
        )

    if sys.platform == "linux":
        for mount in _linux_mount_points():
            if _contained(mount, wd, strict = True):
                raise SandboxUnavailableError(
                    f"the session workdir contains a nested host mount: {mount}"
                )
    return wd


@dataclass(frozen = True)
class _LinuxMount:
    mount_id: str
    parent_id: str
    major_minor: str
    root: str
    mount_point: str
    mount_options: str
    fs_type: str
    source: str
    super_options: str


def _unescape_mountinfo(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _linux_mounts() -> tuple[_LinuxMount, ...]:
    mounts: list[_LinuxMount] = []
    try:
        with open("/proc/self/mountinfo", encoding = "utf-8") as stream:
            for line in stream:
                fields = line.split()
                try:
                    separator = fields.index("-")
                except ValueError as exc:
                    raise SandboxUnavailableError("cannot parse Linux mount topology") from exc
                if len(fields) < 6 or len(fields) <= separator + 3:
                    raise SandboxUnavailableError("cannot parse Linux mount topology")
                mounts.append(
                    _LinuxMount(
                        mount_id = fields[0],
                        parent_id = fields[1],
                        major_minor = fields[2],
                        root = _unescape_mountinfo(fields[3]),
                        mount_point = os.path.realpath(_unescape_mountinfo(fields[4])),
                        mount_options = fields[5],
                        fs_type = fields[separator + 1],
                        source = _unescape_mountinfo(fields[separator + 2]),
                        super_options = fields[separator + 3],
                    )
                )
    except OSError as exc:
        raise SandboxUnavailableError("cannot inspect Linux nested mounts") from exc
    return tuple(mounts)


def _linux_mount_points() -> tuple[str, ...]:
    return tuple(mount.mount_point for mount in _linux_mounts())


def _linux_mount_for_path(path: str) -> _LinuxMount | None:
    canonical = os.path.realpath(path)
    candidates = [
        mount
        for mount in _linux_mounts()
        if _contained(canonical, mount.mount_point) or canonical == mount.mount_point
    ]
    if not candidates:
        return None
    return max(candidates, key = lambda mount: len(mount.mount_point))


def _runtime_read_paths() -> tuple[str, ...]:
    """Return selected interpreter/library roots, never arbitrary inherited sys.path."""
    executable = os.path.abspath(sys.executable)
    candidates: list[str] = [
        executable,
        os.path.realpath(executable),
        os.path.dirname(executable),
        os.path.join(sys.prefix, "pyvenv.cfg"),
        os.path.join(sys.prefix, "lib"),
        os.path.join(sys.prefix, "lib64"),
        os.path.join(sys.base_prefix, "lib"),
        os.path.join(sys.base_prefix, "lib64"),
        os.path.join(sys.base_prefix, "Python"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_site"),
    ]
    try:
        paths = sysconfig.get_paths()
        candidates.extend(
            paths[key] for key in ("stdlib", "platstdlib", "purelib", "platlib") if paths.get(key)
        )
    except (KeyError, OSError):
        pass

    selected: list[str] = []
    for candidate in candidates:
        if not candidate or not os.path.isabs(candidate):
            continue
        path = os.path.abspath(candidate)
        if path == os.path.sep or (sys.platform == "linux" and path == "/usr"):
            continue
        if os.path.exists(path) and path not in selected:
            selected.append(path)
    return tuple(selected)


def _validate_runtime_paths(
    paths: tuple[str, ...],
    workdir: str,
    *,
    include_system_roots: bool = False,
    allow_nested_mounts: bool = False,
) -> None:
    """User-managed runtimes may be read-only, but must not carry host IPC into the jail.

    Preparation calls this immediately before constructing the bind list. A
    trusted same-UID host process can still mutate a bind source afterward;
    read-only binds are not immutable snapshots, and that race is outside the
    trusted-local threat boundary.
    """
    scan_roots: list[str] = []
    for root in paths:
        if _contained(root, workdir):
            raise SandboxUnavailableError(
                f"an interpreter/runtime path is inside the writable session workdir: {root}"
            )
        if (
            not include_system_roots
            and sys.platform == "linux"
            and any(_contained(root, p) for p in _LINUX_SYSTEM_ROOTS)
        ):
            continue
        try:
            root_mode = os.stat(root).st_mode
        except OSError as exc:
            raise SandboxUnavailableError(
                f"an interpreter/runtime path cannot be inspected: {root}"
            ) from exc
        if not (stat.S_ISDIR(root_mode) or stat.S_ISREG(root_mode)):
            raise SandboxUnavailableError(
                f"an interpreter/runtime path is not a regular file or directory: {root}"
            )
        if any(_contained(root, existing) for existing in scan_roots):
            continue
        scan_roots = [existing for existing in scan_roots if not _contained(existing, root)]
        scan_roots.append(root)

    if sys.platform == "linux" and not allow_nested_mounts:
        for mount in _linux_mount_points():
            if any(_contained(mount, root, strict = True) for root in scan_roots):
                raise SandboxUnavailableError(
                    f"an interpreter/runtime path contains a nested host mount: {mount}"
                )

    find = "/usr/bin/find"
    if sys.platform == "linux" and scan_roots and _trusted_linux_executable(find):
        # Follow a symlink used as a scan root, but never links encountered below it.
        find_command = [find, "-H", *scan_roots]
        find_command.append("-xdev" if sys.platform == "linux" else "-x")
        if include_system_roots:
            # Paths the sandbox UID cannot traverse cannot expose their contents.
            find_command.extend(
                [
                    "(",
                    "-type",
                    "d",
                    "!",
                    "-executable",
                    ")",
                    "-prune",
                    "-o",
                ]
            )
        find_command.extend(
            [
                "(",
                "-type",
                "s",
                "-o",
                "-type",
                "p",
                "-o",
                "-type",
                "b",
                "-o",
                "-type",
                "c",
                ")",
                "-print",
                "-quit",
            ]
        )
        try:
            result = subprocess.run(
                find_command,
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                text = True,
                timeout = 8,
                close_fds = True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SandboxUnavailableError("cannot scan interpreter/runtime paths safely") from exc
        if result.returncode != 0:
            raise SandboxUnavailableError(
                f"cannot scan interpreter/runtime paths safely: {result.stderr.strip()[-200:]}"
            )
        if result.stdout.strip():
            raise SandboxUnavailableError(
                f"an interpreter/runtime path contains {_REJECTED_RUNTIME_ENTRY_TYPES}: "
                f"{result.stdout.strip().splitlines()[0]}"
            )
        return

    for root in scan_roots:
        entries = 0
        if os.path.isfile(root):
            continue

        def walk_error(exc: OSError) -> None:
            raise SandboxUnavailableError(
                f"runtime path cannot be fully inspected: {exc.filename or root}"
            ) from exc

        for base, dirs, names in os.walk(root, followlinks = False, onerror = walk_error):
            for name in [*dirs, *names]:
                entries += 1
                if entries > _SCAN_ENTRY_LIMIT:
                    raise SandboxUnavailableError(
                        f"runtime path exceeds the {_SCAN_ENTRY_LIMIT:,}-entry safety scan limit: {root}"
                    )
                path = os.path.join(base, name)
                try:
                    mode = os.lstat(path).st_mode
                except OSError as exc:
                    raise SandboxUnavailableError(
                        f"runtime path changed during its safety scan: {path}"
                    ) from exc
                if stat.S_ISLNK(mode):
                    if name in dirs:
                        dirs.remove(name)
                elif not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                    raise SandboxUnavailableError(
                        f"runtime path contains {_REJECTED_RUNTIME_ENTRY_TYPES}: {path}"
                    )


def _trusted_linux_executable(path: str) -> bool:
    current = os.path.realpath(path)
    first = True
    while True:
        try:
            info = os.stat(current, follow_symlinks = False)
        except OSError:
            return False
        mode = stat.S_IMODE(info.st_mode)
        if first:
            if not stat.S_ISREG(info.st_mode) or not os.access(current, os.X_OK):
                return False
            first = False
        elif not stat.S_ISDIR(info.st_mode):
            return False
        if info.st_uid != 0 or mode & (stat.S_IWGRP | stat.S_IWOTH):
            return False
        parent = os.path.dirname(current)
        if parent == current:
            return True
        current = parent


def _read_text(path: str) -> str:
    try:
        with open(path, encoding = "utf-8") as stream:
            return stream.read()
    except OSError:
        return ""


def _linux_environment() -> str:
    release = ""
    release = _read_text("/proc/sys/kernel/osrelease").lower()
    if "microsoft" in release or os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return "wsl2"
    if os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:
        return "colab"
    cgroup = _read_text("/proc/1/cgroup").lower()
    container_markers = ("docker", "kubepods", "containerd", "libpod", "podman", "lxc")
    if (
        os.path.exists("/.dockerenv")
        or os.path.exists("/run/.containerenv")
        or os.environ.get("container")
        or any(marker in cgroup for marker in container_markers)
    ):
        return "container"
    detector = "/usr/bin/systemd-detect-virt"
    if _trusted_linux_executable(detector):
        try:
            detected = subprocess.run(
                [detector, "--container", "--quiet"],
                stdin = subprocess.DEVNULL,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                timeout = 3,
                close_fds = True,
            )
        except (OSError, subprocess.SubprocessError):
            detected = None
        if detected is not None and detected.returncode == 0:
            return "container"
        if detected is not None and detected.returncode == 1:
            return "native_linux"
    # Qualification still depends on the live sandbox probe. The conservative
    # label prevents an unusual container from receiving the stronger native
    # Linux claim merely because the outer environment could not be classified.
    return "linux_unknown"


def _environment_class() -> str:
    if sys.platform == "linux":
        return _linux_environment()
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "windows"
    return f"unsupported-{sys.platform}"


def _excluded_linux_environment() -> str | None:
    """Compatibility hook: environment labels no longer reject qualification alone."""
    return None


def _environment_fingerprint(backend: "SandboxBackend | None") -> str:
    data: dict[str, object] = {
        "platform": sys.platform,
        "architecture": platform.machine().lower(),
        "environment": _environment_class(),
        "python": os.path.realpath(sys.executable),
    }
    try:
        executable = os.stat(os.path.realpath(sys.executable))
        data["python_stat"] = [
            executable.st_dev,
            executable.st_ino,
            executable.st_size,
            executable.st_mtime_ns,
        ]
    except OSError:
        data["python_stat"] = None
    if sys.platform == "linux":
        data["namespaces"] = {
            name: os.readlink(f"/proc/self/ns/{name}")
            for name in ("mnt", "pid", "net", "ipc", "user")
            if os.path.exists(f"/proc/self/ns/{name}")
        }
        try:
            data["mounts"] = [
                [
                    mount.mount_point,
                    mount.root,
                    mount.fs_type,
                    mount.source,
                    mount.mount_options,
                    mount.super_options,
                ]
                for mount in _linux_mounts()
            ]
        except SandboxUnavailableError as exc:
            data["mount_error"] = str(exc)
        data["namespace_policy"] = {
            path: _read_text(path).strip()
            for path in (
                "/proc/sys/kernel/unprivileged_userns_clone",
                "/proc/sys/user/max_user_namespaces",
                "/proc/sys/kernel/apparmor_restrict_unprivileged_userns",
            )
        }
        status = _read_text("/proc/self/status")
        data["outer_status"] = [
            line
            for line in status.splitlines()
            if line.startswith(("CapEff:", "CapBnd:", "NoNewPrivs:", "Seccomp:"))
        ]
        candidate = shutil.which("bwrap")
        if candidate:
            candidate = os.path.realpath(os.path.abspath(candidate))
            data["backend_executable"] = candidate
            try:
                info = os.stat(candidate)
                data["backend_stat"] = [
                    info.st_dev,
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_uid,
                    stat.S_IMODE(info.st_mode),
                ]
            except OSError:
                data["backend_stat"] = None
    elif backend is not None:
        data["backend"] = backend.identity
        fingerprint_data = getattr(backend, "fingerprint_data", None)
        if callable(fingerprint_data):
            try:
                data["backend_fingerprint"] = fingerprint_data()
            except Exception as exc:  # noqa: BLE001 - inability to inspect must change identity
                data["backend_fingerprint_error"] = f"{type(exc).__name__}: {exc}"
    encoded = json.dumps(data, sort_keys = True, separators = (",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identity_files() -> tuple[str, str, str]:
    directory = tempfile.mkdtemp(prefix = "unsloth-sandbox-identity-")
    uid = getattr(os, "getuid", lambda: 65534)()
    gid = getattr(os, "getgid", lambda: 65534)()
    passwd = os.path.join(directory, "passwd")
    group = os.path.join(directory, "group")
    try:
        with open(passwd, "w", encoding = "utf-8") as stream:
            stream.write(f"studio:x:{uid}:{gid}:Studio sandbox:/nonexistent:/bin/sh\n")
        with open(group, "w", encoding = "utf-8") as stream:
            stream.write(f"studio:x:{gid}:\n")
        os.chmod(passwd, 0o600)
        os.chmod(group, 0o600)
    except Exception:
        shutil.rmtree(directory, ignore_errors = True)
        raise
    return directory, passwd, group


_NPROC_WRAPPER = """import os, resource, sys
try:
    limit = int({limit!r})
    _soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    target = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
    resource.setrlimit(resource.RLIMIT_NPROC, (target, target))
except (AttributeError, OSError, ValueError):
    pass
os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
"""


def _nproc_limit() -> int:
    try:
        return max(64, int(os.environ.get("UNSLOTH_STUDIO_SANDBOX_NPROC", "10000")))
    except ValueError:
        return 10000


def _validate_linux_workdir_environment(workdir: str) -> None:
    if _linux_environment() != "wsl2":
        return
    mount = _linux_mount_for_path(workdir)
    if mount is None:
        raise SandboxUnavailableError("cannot identify the WSL session-workdir filesystem")
    marker = f"{mount.fs_type} {mount.source}".lower()
    if mount.fs_type.lower() in {"9p", "drvfs"} or "drvfs" in marker:
        raise SandboxUnavailableError(
            "WSL OS isolation requires the session workdir on the Linux filesystem, not a Windows-mounted filesystem"
        )


def _nested_exposed_mounts(roots: tuple[str, ...]) -> tuple[_LinuxMount, ...]:
    candidates = [
        mount
        for mount in _linux_mounts()
        if any(_contained(mount.mount_point, root, strict = True) for root in roots)
    ]
    selected: list[_LinuxMount] = []
    for mount in sorted(candidates, key = lambda item: len(item.mount_point)):
        if any(_contained(mount.mount_point, prior.mount_point) for prior in selected):
            continue
        selected.append(mount)
    return tuple(selected)


def _sanitize_linux_environment(env: dict[str, str], environment: str) -> dict[str, str]:
    sanitized = dict(env)
    if environment == "wsl2":
        for key in (
            "WSL_INTEROP",
            "WSLENV",
            "WSL_DISTRO_NAME",
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "PULSE_SERVER",
            "XDG_RUNTIME_DIR",
        ):
            sanitized.pop(key, None)
        path = sanitized.get("PATH", "")
        sanitized["PATH"] = os.pathsep.join(
            entry
            for entry in path.split(os.pathsep)
            if entry
            and not entry.lower().startswith(("/mnt/", "/run/desktop/", "/usr/lib/wsl/"))
            and "windowsapps" not in entry.lower()
        )
        sanitized["XDG_RUNTIME_DIR"] = "/tmp/runtime"
    return sanitized


class LinuxBubblewrapBackend:
    identity = "linux-bubblewrap"
    profile_id = "linux-bubblewrap-v2"

    def __init__(self) -> None:
        self._bwrap: str | None = None

    def probe(self) -> SandboxCapability:
        if platform.machine().lower() not in _LINUX_SECCOMP_ABIS:
            return SandboxCapability(
                self.identity,
                False,
                f"Linux architecture {platform.machine() or 'unknown'} has no reviewed seccomp ABI",
            )
        candidate = shutil.which("bwrap")
        if not candidate:
            return SandboxCapability(
                self.identity,
                False,
                "Bubblewrap is not installed; install the distribution's bubblewrap package",
            )
        candidate = os.path.realpath(os.path.abspath(candidate))
        if not _trusted_linux_executable(candidate):
            return SandboxCapability(
                self.identity,
                False,
                f"Bubblewrap is not a root-controlled executable: {candidate}",
            )
        system_roots = tuple(
            path for path in (*_LINUX_SYSTEM_ROOTS, *_LINUX_ETC_FILES) if os.path.exists(path)
        )
        try:
            _validate_runtime_paths(
                system_roots,
                "/__unsloth_sandbox_probe_workdir__",
                include_system_roots = True,
                allow_nested_mounts = True,
            )
        except SandboxUnavailableError as exc:
            return SandboxCapability(
                self.identity,
                False,
                f"a read-only Linux system root is unsafe to expose: {exc}",
            )
        self._bwrap = candidate
        result = _live_probe(self)
        if result.qualified:
            return replace(
                result,
                available = True,
                profile_id = self.profile_id,
            )
        return replace(result, available = False)

    def prepare(self, spec: ToolLaunchPlan) -> PreparedSandboxLaunch:
        if self._bwrap is None:
            raise SandboxUnavailableError("Bubblewrap was not qualified in this process")
        workdir = _validate_workdir(spec.workdir)
        _validate_linux_workdir_environment(workdir)
        runtime_paths = _runtime_read_paths()
        system_roots = tuple(
            path for path in (*_LINUX_SYSTEM_ROOTS, *_LINUX_ETC_FILES) if os.path.exists(path)
        )
        _validate_runtime_paths(
            system_roots,
            workdir,
            include_system_roots = True,
            allow_nested_mounts = True,
        )
        _validate_runtime_paths(runtime_paths, workdir, allow_nested_mounts = True)
        exposed_roots = tuple(
            path for path in (*system_roots, *runtime_paths) if not _contained(path, workdir)
        )
        nested_mounts: list[tuple[_LinuxMount, bool]] = []
        for mount in _nested_exposed_mounts(exposed_roots):
            try:
                mode = os.stat(mount.mount_point).st_mode
            except OSError as exc:
                raise SandboxUnavailableError(
                    f"a nested host mount cannot be safely masked: {mount.mount_point}"
                ) from exc
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise SandboxUnavailableError(
                    f"a nested host mount has an unsupported type: {mount.mount_point}"
                )
            nested_mounts.append((mount, stat.S_ISDIR(mode)))
        seccomp_filter = _linux_seccomp_filter()
        try:
            identity_dir, passwd, group = _identity_files()
        except Exception:
            seccomp_filter.close()
            raise
        environment = _linux_environment()
        env = _sanitize_linux_environment(spec.env, environment)
        env["HOME"] = workdir
        env["TMPDIR"] = "/tmp"

        argv: list[str] = [
            self._bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--unshare-user",
            "--disable-userns",
            "--cap-drop",
            "ALL",
            "--seccomp",
            str(seccomp_filter.fileno()),
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--dir",
            "/dev/shm",
            "--dir",
            "/tmp",
            "--dir",
            "/etc",
        ]
        for root in _LINUX_SYSTEM_ROOTS:
            argv.extend(("--ro-bind-try", root, root))
        if _contained(sys.executable, "/nix/store") and os.path.isdir("/nix/store"):
            argv.extend(("--ro-bind", "/nix/store", "/nix/store"))
        for path in _LINUX_ETC_FILES:
            argv.extend(("--ro-bind-try", path, path))
        argv.extend(("--ro-bind", passwd, "/etc/passwd"))
        argv.extend(("--ro-bind", group, "/etc/group"))
        private_tmp_runtime_paths: list[str] = []
        for path in runtime_paths:
            if any(_lexically_contained(path, root) for root in _LINUX_SYSTEM_ROOTS):
                continue
            if _lexically_contained(path, "/nix/store"):
                continue
            if _lexically_contained(path, "/tmp"):
                private_tmp_runtime_paths.append(path)
                continue
            argv.extend(("--ro-bind", path, path))
        empty_mask = os.path.join(identity_dir, "empty")
        try:
            with open(empty_mask, "wb"):
                pass
        except Exception:
            seccomp_filter.close()
            shutil.rmtree(identity_dir, ignore_errors = True)
            raise
        for mount, is_directory in nested_mounts:
            if is_directory:
                argv.extend(("--tmpfs", mount.mount_point))
            else:
                argv.extend(("--ro-bind", empty_mask, mount.mount_point))
        if environment == "wsl2":
            for path in _WSL_HIDDEN_PATHS:
                argv.extend(("--tmpfs", path))
        argv.extend(("--dir", workdir, "--remount-ro", "/"))
        argv.extend(("--tmpfs", "/dev/shm", "--tmpfs", "/tmp"))
        argv.extend(("--dir", "/tmp/runtime"))
        for path in private_tmp_runtime_paths:
            argv.extend(("--ro-bind", path, path))
        argv.extend(("--bind", workdir, workdir, "--chdir", workdir))
        argv.extend(("--setenv", "HOME", workdir, "--setenv", "TMPDIR", "/tmp"))
        wrapper = _NPROC_WRAPPER.format(limit = _nproc_limit())
        argv.extend(
            (
                "--",
                sys.executable,
                "-I",
                "-S",
                "-c",
                wrapper,
                *spec.argv,
            )
        )
        return PreparedSandboxLaunch(
            argv = tuple(argv),
            workdir = workdir,
            env = env,
            preexec_fn = spec.launcher_preexec_fn,
            backend = self.identity,
            pass_fds = (seccomp_filter.fileno(),),
            owned_files = [seccomp_filter],
            cleanup_paths = [identity_dir],
            timeout_seconds = spec.timeout_seconds,
            close_fds = spec.close_fds,
            terminate_descendants = spec.terminate_descendants,
        )


class MacOSSeatbeltBackend:
    identity = "macos-seatbelt"
    profile_id = "macos-seatbelt-preview-v1"
    limitations = (
        "deprecated_undocumented_sbpl",
        "detached_descendant_cleanup_unverified",
        "pytorch_posix_shm_namespace_shared",
    )

    def __init__(self) -> None:
        self._sandbox_exec: str | None = None

    def probe(self) -> SandboxCapability:
        candidate = "/usr/bin/sandbox-exec"
        try:
            info = os.stat(candidate, follow_symlinks = False)
        except OSError as exc:
            return SandboxCapability(
                self.identity,
                False,
                f"the system Seatbelt launcher is unavailable: {exc}",
                available = False,
                limitations = self.limitations,
            )
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != 0
            or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            return SandboxCapability(
                self.identity,
                False,
                "/usr/bin/sandbox-exec is not a root-owned, non-user-writable regular file",
                available = False,
                limitations = self.limitations,
            )
        self._sandbox_exec = candidate
        result = _live_macos_probe(self)
        if result.available:
            return replace(
                result,
                qualified = False,
                protection_state = "preview",
                profile_id = self.profile_id,
                limitations = self.limitations,
                reason = (
                    "Seatbelt filesystem and network enforcement passed its live probe; "
                    "SBPL is deprecated and undocumented for third-party products, and "
                    "cleanup of detached setsid/double-fork descendants remains unverified"
                ),
            )
        return replace(result, limitations = self.limitations)

    def prepare(self, spec: ToolLaunchPlan) -> PreparedSandboxLaunch:
        if self._sandbox_exec is None:
            raise SandboxUnavailableError("Seatbelt was not proven available in this process")
        workdir = _validate_workdir(spec.workdir)
        runtime_paths = _runtime_read_paths()
        _validate_runtime_paths(runtime_paths, workdir)
        private_tmp = tempfile.mkdtemp(
            prefix = "us-seatbelt-", dir = "/tmp" if sys.platform == "darwin" else None
        )
        try:
            profile = _macos_seatbelt_profile(
                workdir = workdir,
                private_tmp = private_tmp,
                runtime_paths = runtime_paths,
            )
            env = _sanitize_macos_environment(spec.env, workdir, private_tmp)
            return PreparedSandboxLaunch(
                argv = (self._sandbox_exec, "-p", profile, "--", *spec.argv),
                workdir = workdir,
                env = env,
                preexec_fn = spec.preexec_fn,
                backend = self.identity,
                cleanup_paths = [private_tmp],
                timeout_seconds = spec.timeout_seconds,
                close_fds = spec.close_fds,
                terminate_descendants = spec.terminate_descendants,
            )
        except Exception:
            shutil.rmtree(private_tmp, ignore_errors = True)
            raise


_MACOS_READ_ROOTS = (
    "/Library/Apple/System/Library/Frameworks",
    "/Library/Apple/System/Library/PrivateFrameworks",
    "/Library/Apple/usr/lib",
    "/System/Library/Frameworks",
    "/System/Library/PrivateFrameworks",
    "/System/Library/SubFrameworks",
    "/System/iOSSupport/System/Library/Frameworks",
    "/System/iOSSupport/System/Library/PrivateFrameworks",
    "/System/iOSSupport/System/Library/SubFrameworks",
    "/usr/lib",
    "/usr/share",
    "/bin",
    "/usr/bin",
    "/private/var/db/timezone",
    "/private/etc/localtime",
    "/private/etc/master.passwd",
    "/private/etc/passwd",
    "/private/etc/protocols",
    "/private/etc/services",
    "/System/Library/CoreServices/.SystemVersionPlatform.plist",
    "/System/Library/CoreServices/SystemVersion.plist",
)
_MACOS_DENIED_EXECUTABLES = (
    "/usr/bin/open",
    "/usr/bin/osascript",
    "/usr/bin/security",
    "/bin/launchctl",
    "/usr/bin/sandbox-exec",
)
_MACOS_DEVICES = ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom")
_MACOS_SYSCTL_NAMES = (
    "hw.activecpu",
    "hw.busfrequency_compat",
    "hw.byteorder",
    "hw.cacheconfig",
    "hw.cachelinesize_compat",
    "hw.cpufamily",
    "hw.cpufrequency_compat",
    "hw.cputype",
    "hw.l1dcachesize_compat",
    "hw.l1icachesize_compat",
    "hw.l2cachesize_compat",
    "hw.l3cachesize_compat",
    "hw.logicalcpu",
    "hw.logicalcpu_max",
    "hw.machine",
    "hw.memsize",
    "hw.model",
    "hw.ncpu",
    "hw.nperflevels",
    "hw.packages",
    "hw.pagesize",
    "hw.pagesize_compat",
    "hw.physicalcpu",
    "hw.physicalcpu_max",
    "hw.tbfrequency_compat",
    "hw.vectorunit",
    "kern.argmax",
    "kern.hostname",
    "kern.maxfilesperproc",
    "kern.maxproc",
    "kern.osproductversion",
    "kern.osrelease",
    "kern.ostype",
    "kern.osvariant_status",
    "kern.osversion",
    "kern.secure_kernel",
    "kern.sysv.semmns",
    "kern.usrstack64",
    "kern.version",
    "machdep.cpu.brand_string",
    "sysctl.proc_cputype",
    "vm.loadavg",
)
_MACOS_SYSCTL_PREFIXES = (
    "hw.optional.arm.",
    "hw.optional.armv8_",
    "hw.perflevel",
    "kern.proc.pgrp.",
    "kern.proc.pid.",
)


def _sbpl_path(path: str) -> tuple[str, bool]:
    """Return an encoded canonical SBPL path plus whether it is a directory."""
    if not path or "\0" in path or "\n" in path or "\r" in path or not os.path.isabs(path):
        raise SandboxUnavailableError("Seatbelt paths must be absolute and contain no NUL/newline")
    canonical = os.path.realpath(path)
    if not os.path.isabs(canonical) or "\0" in canonical or "\n" in canonical or "\r" in canonical:
        raise SandboxUnavailableError("a Seatbelt path did not canonicalize safely")
    if not os.path.exists(canonical):
        raise SandboxUnavailableError(f"a required Seatbelt path does not exist: {canonical}")
    return json.dumps(canonical), os.path.isdir(canonical)


def _sbpl_path_filters(paths: tuple[str, ...]) -> list[str]:
    filters: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        _, is_directory = _sbpl_path(path)
        for spelling in _sbpl_path_spellings(path):
            encoded = json.dumps(spelling)
            if encoded in seen:
                continue
            seen.add(encoded)
            filters.append(f"(literal {encoded})")
            if is_directory:
                filters.append(f"(subpath {encoded})")
    return filters


def _sbpl_path_spellings(path: str) -> tuple[str, ...]:
    canonical, _ = _sbpl_path(path)
    selected = [os.path.abspath(path), json.loads(canonical)]
    for spelling in tuple(selected):
        if spelling == "/private/var" or spelling.startswith("/private/var/"):
            selected.append(spelling[len("/private") :])
    return tuple(dict.fromkeys(selected))


def _sbpl_ancestor_filters(paths: tuple[str, ...]) -> list[str]:
    filters: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        for spelling in _sbpl_path_spellings(path):
            current = os.path.dirname(spelling.rstrip(os.path.sep)) or os.path.sep
            while current:
                encoded = json.dumps(current)
                if encoded not in seen:
                    seen.add(encoded)
                    filters.append(f"(literal {encoded})")
                parent = os.path.dirname(current)
                if parent == current:
                    break
                current = parent
    return filters


def _macos_seatbelt_profile(
    *, workdir: str, private_tmp: str, runtime_paths: tuple[str, ...]
) -> str:
    readable_paths = (*_MACOS_READ_ROOTS, *_MACOS_DEVICES, *runtime_paths, workdir, private_tmp)
    read_filters = [
        '(literal "/")',
        *_sbpl_path_filters(readable_paths),
    ]
    metadata_filters = _sbpl_ancestor_filters(readable_paths)
    write_filters = _sbpl_path_filters((workdir, private_tmp))
    temp_encoded, _ = _sbpl_path(private_tmp)
    device_filters = _sbpl_path_filters(_MACOS_DEVICES)
    denied_exec = _sbpl_path_filters(
        tuple(path for path in _MACOS_DENIED_EXECUTABLES if os.path.exists(path))
    )
    sysctl_filters = [
        *(f"(sysctl-name {json.dumps(name)})" for name in _MACOS_SYSCTL_NAMES),
        *(f"(sysctl-name-prefix {json.dumps(name)})" for name in _MACOS_SYSCTL_PREFIXES),
    ]
    lines = [
        "(version 1)",
        "(deny default)",
        # Codex scopes signals and process inspection to descendants in the
        # same Seatbelt instance; Studio keeps the same narrow process surface.
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow signal (target same-sandbox))",
        "(allow process-info* (target same-sandbox))",
        "(deny process-exec " + " ".join(denied_exec) + ")",
        "(allow file-read-metadata " + " ".join(metadata_filters) + ")",
        "(allow file-read* file-test-existence " + " ".join(read_filters) + ")",
        "(allow file-map-executable " + " ".join(read_filters) + ")",
        "(allow file-write* " + " ".join(write_filters) + ")",
        "(allow file-read* file-test-existence file-write-data "
        + " ".join(device_filters[:4])
        + ")",
        '(allow file-read* (regex #"^/dev/fd/(0|1|2)$"))',
        '(allow file-write* (regex #"^/dev/fd/(1|2)$"))',
        "(allow file-ioctl " + " ".join(device_filters) + ")",
        "(allow ipc-posix-sem)",
        "(allow ipc-posix-shm-read-data ipc-posix-shm-write-create "
        "ipc-posix-shm-write-unlink "
        '(ipc-posix-name-regex #"^/__KMP_REGISTERED_LIB_[0-9]+$"))',
        "(allow ipc-posix-shm-read-data ipc-posix-shm-write-create "
        "ipc-posix-shm-write-data ipc-posix-shm-write-unlink "
        '(ipc-posix-name-regex #"^/torch_[0-9]+_[0-9]+_[0-9]+$"))',
        "(allow system-socket (socket-domain AF_UNIX))",
        f"(allow network-bind (local unix-socket (subpath {temp_encoded})))",
        f"(allow network-outbound (remote unix-socket (subpath {temp_encoded})))",
        "(allow sysctl-read " + " ".join(sysctl_filters) + ")",
        '(allow iokit-open (iokit-registry-entry-class "RootDomainUserClient"))',
        "(allow mach-lookup",
        '  (global-name "com.apple.system.opendirectoryd.libinfo")',
        '  (global-name "com.apple.PowerManagement.control"))',
    ]
    return "\n".join(lines) + "\n"


def _sanitize_macos_environment(
    env: dict[str, str], workdir: str, private_tmp: str
) -> dict[str, str]:
    sanitized = {
        key: value
        for key, value in env.items()
        if not key.startswith("DYLD_")
        and key
        not in {
            "DISPLAY",
            "SSH_AUTH_SOCK",
            "XPC_SERVICE_NAME",
            "WAYLAND_DISPLAY",
            "PULSE_SERVER",
        }
    }
    sanitized.update(
        {
            "HOME": workdir,
            "TMPDIR": private_tmp,
            "TMP": private_tmp,
            "TEMP": private_tmp,
            "XDG_RUNTIME_DIR": private_tmp,
        }
    )
    return sanitized


def _macos_probe_payload(
    *,
    workdir: str,
    external_file: str,
    host_socket: str,
    ipv4_address: tuple[str, int],
    ipv6_address: tuple[str, int, int, int],
    inherited_fd: int,
) -> str:
    return f"""import os, socket
wd = {workdir!r}
assert open(os.path.join(wd, 'probe-read'), encoding='utf-8').read() == 'readable'
open(os.path.join(wd, 'probe-write'), 'w', encoding='utf-8').write('ok')
for path in ({external_file!r}, os.path.join(wd, 'escape')):
    try:
        open(path, encoding='utf-8').read()
        raise AssertionError('read outside workdir: ' + path)
    except OSError:
        pass
try:
    open(os.path.join(os.path.dirname(wd), 'outside-write'), 'w').close()
    raise AssertionError('wrote outside workdir')
except OSError:
    pass
try:
    os.fstat({inherited_fd})
    raise AssertionError('inherited host descriptor remained open')
except OSError:
    pass
for family, sock_type, address in (
    (socket.AF_INET, socket.SOCK_STREAM, {ipv4_address!r}),
    (socket.AF_INET6, socket.SOCK_STREAM, {ipv6_address!r}),
    (socket.AF_INET, socket.SOCK_DGRAM, {ipv4_address!r}),
):
    try:
        sock = socket.socket(family, sock_type)
    except OSError:
        continue
    sock.settimeout(0.2)
    try:
        if sock_type == socket.SOCK_DGRAM:
            sock.sendto(b'unsloth-seatbelt-probe', address)
        else:
            sock.connect(address)
        raise AssertionError('IP network was reachable')
    except OSError:
        pass
    finally:
        sock.close()
try:
    socket.getaddrinfo('unsloth-probe.invalid', 443)
    raise AssertionError('DNS was reachable')
except socket.gaierror:
    pass
for path in ('/dev/tty', '/dev/disk0', '/var/run', '/private/var/run'):
    try:
        open(path, 'rb').close()
        raise AssertionError('restricted host path was readable: ' + path)
    except OSError:
        pass
try:
    with open(os.path.realpath(os.__file__), 'ab') as stream:
        stream.write(b'x')
    raise AssertionError('modified the interpreter runtime')
except OSError:
    pass
host = socket.socket(socket.AF_UNIX)
try:
    host.connect({host_socket!r})
    raise AssertionError('host Unix socket was reachable')
except OSError:
    pass
finally:
    host.close()
path = os.path.join(os.environ['TMPDIR'], 'private.sock')
server = socket.socket(socket.AF_UNIX)
client = socket.socket(socket.AF_UNIX)
try:
    server.bind(path)
    server.listen(1)
    client.connect(path)
    accepted, _ = server.accept()
    accepted.close()
finally:
    client.close()
    server.close()
print({_PROBE_TOKEN!r})
"""


def _live_macos_probe(backend: MacOSSeatbeltBackend) -> SandboxCapability:
    host_socket: socket.socket | None = None
    ipv4_socket: socket.socket | None = None
    ipv6_socket: socket.socket | None = None
    inherited_fd: int | None = None
    prepared: PreparedSandboxLaunch | None = None
    try:
        with tempfile.TemporaryDirectory(prefix = "unsloth-seatbelt-probe-") as base:
            workdir = os.path.join(base, "work")
            os.mkdir(workdir)
            with open(os.path.join(workdir, "probe-read"), "w", encoding = "utf-8") as stream:
                stream.write("readable")
            external = os.path.join(base, "host-secret")
            with open(external, "w", encoding = "utf-8") as stream:
                stream.write("secret")
            os.symlink(external, os.path.join(workdir, "escape"))
            host_socket_path = os.path.join(base, "host.sock")
            host_socket = socket.socket(socket.AF_UNIX)
            host_socket.bind(host_socket_path)
            host_socket.listen(1)
            ipv4_socket = socket.socket(socket.AF_INET)
            ipv4_socket.bind(("127.0.0.1", 0))
            ipv4_socket.listen(1)
            ipv6_socket = socket.socket(socket.AF_INET6)
            ipv6_socket.bind(("::1", 0))
            ipv6_socket.listen(1)
            read_fd, write_fd = os.pipe()
            os.close(write_fd)
            inherited_fd = os.dup(read_fd)
            os.close(read_fd)
            os.set_inheritable(inherited_fd, True)
            prepared = backend.prepare(
                ToolLaunchPlan(
                    argv = (
                        sys.executable,
                        "-I",
                        "-S",
                        "-c",
                        _macos_probe_payload(
                            workdir = workdir,
                            external_file = external,
                            host_socket = host_socket_path,
                            ipv4_address = ipv4_socket.getsockname(),
                            ipv6_address = ipv6_socket.getsockname(),
                            inherited_fd = inherited_fd,
                        ),
                    ),
                    workdir = workdir,
                    env = {"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"},
                )
            )
            completed = subprocess.run(
                prepared.argv,
                cwd = prepared.workdir,
                env = prepared.env,
                stdin = subprocess.DEVNULL,
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                text = True,
                timeout = _PROBE_TIMEOUT_SECONDS,
                close_fds = True,
                preexec_fn = prepared.preexec_fn,
            )
            if completed.returncode != 0 or _PROBE_TOKEN not in completed.stdout:
                detail = completed.stderr.strip()[-300:] or completed.stdout.strip()[-300:]
                return SandboxCapability(
                    backend.identity,
                    False,
                    f"the Seatbelt live probe failed ({completed.returncode}): {detail}",
                    available = False,
                )
    except subprocess.TimeoutExpired:
        return SandboxCapability(
            backend.identity,
            False,
            "the Seatbelt live probe timed out",
            available = False,
            transient = True,
        )
    except Exception as exc:  # noqa: BLE001 - capability failure blocks execution
        return SandboxCapability(
            backend.identity,
            False,
            f"the Seatbelt live probe could not run: {exc}",
            available = False,
        )
    finally:
        if prepared is not None:
            prepared.cleanup()
        for stream in (host_socket, ipv4_socket, ipv6_socket):
            if stream is not None:
                stream.close()
        if inherited_fd is not None:
            try:
                os.close(inherited_fd)
            except OSError:
                pass
    return SandboxCapability(
        backend.identity,
        False,
        "Seatbelt live enforcement probe passed",
        available = True,
        protection_state = "preview",
        profile_id = backend.profile_id,
        limitations = backend.limitations,
    )


def _probe_payload(
    workdir: str,
    external_file: str,
    host_socket: str,
    host_pid: int,
    abstract_socket: str | None,
    ipv4_address: tuple[str, int],
    ipv6_address: tuple[str, int, int, int],
    udp_address: tuple[str, int],
    host_namespaces: dict[str, str],
    inherited_fds: tuple[int, ...],
) -> str:
    abstract_check = ""
    if abstract_socket is not None:
        abstract_check = f"""
s = socket.socket(socket.AF_UNIX)
try:
    s.connect({abstract_socket!r})
    raise AssertionError('host abstract Unix socket was reachable')
except OSError:
    pass
finally:
    s.close()
"""
    return f"""import ctypes, errno, os, socket, sys
wd = {workdir!r}
with open(os.path.join(wd, 'probe-read'), encoding='utf-8') as f:
    assert f.read() == 'readable'
with open(os.path.join(wd, 'probe-write'), 'w', encoding='utf-8') as f:
    f.write('ok')
assert not os.path.exists({external_file!r})
try:
    open(os.path.join(wd, 'escape'), encoding='utf-8').close()
    raise AssertionError('followed a workdir symlink outside the sandbox')
except OSError:
    pass
try:
    open('/unsloth-host-escape', 'w').close()
    raise AssertionError('wrote outside workdir')
except OSError:
    pass
try:
    with open(sys.executable, 'ab') as f:
        f.write(b'x')
    raise AssertionError('modified the interpreter')
except OSError:
    pass
assert not os.path.exists('/proc/{host_pid}/environ')
for name, outer in {host_namespaces!r}.items():
    assert os.readlink('/proc/self/ns/' + name) != outer, name + ' namespace was inherited'
status = {{}}
with open('/proc/self/status', encoding='utf-8') as f:
    for line in f:
        if ':' in line:
            key, value = line.split(':', 1)
            status[key] = value.strip()
assert status.get('NoNewPrivs') == '1', status.get('NoNewPrivs')
assert status.get('Seccomp') == '2', status.get('Seccomp')
assert int(status.get('CapEff', '1'), 16) == 0
assert int(status.get('CapBnd', '1'), 16) == 0
for fd in {inherited_fds!r}:
    try:
        os.fstat(fd)
    except OSError as exc:
        assert exc.errno == errno.EBADF
    else:
        raise AssertionError('inherited host descriptor remained open: ' + str(fd))
if hasattr(socket, 'AF_VSOCK'):
    try:
        vsock = socket.socket(socket.AF_VSOCK)
    except OSError:
        pass
    else:
        vsock.close()
        raise AssertionError('AF_VSOCK remained available')
libc = ctypes.CDLL(None, use_errno=True)
result = libc.syscall(425, 1, 0)
assert result == -1 and ctypes.get_errno() == errno.EPERM, 'io_uring_setup was not denied'
for family, address in ((socket.AF_INET, {ipv4_address!r}), (socket.AF_INET6, {ipv6_address!r})):
    s = socket.socket(family)
    s.settimeout(0.2)
    try:
        s.connect(address)
        raise AssertionError('IP network was reachable')
    except OSError:
        pass
    finally:
        s.close()
udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    udp.sendto({_PROBE_UDP_TOKEN!r}, {udp_address!r})
except OSError:
    pass
finally:
    udp.close()
assert not os.path.exists('/etc/resolv.conf')
try:
    socket.getaddrinfo('example.com', 443)
    raise AssertionError('DNS was reachable')
except socket.gaierror:
    pass
for forbidden in (
    '/sys', '/run', '/var/run', '/init', '/mnt/c', '/mnt/e', '/mnt/wsl', '/mnt/wslg',
    '/usr/lib/wsl',
    '/dev/kvm', '/dev/dxg', '/dev/dri', '/dev/fuse', '/dev/vsock', '/dev/mem',
    '/var/run/docker.sock', '/run/containerd/containerd.sock',
    '/var/run/podman/podman.sock', '/var/run/secrets/kubernetes.io',
):
    assert not os.path.exists(forbidden), forbidden + ' was exposed'
s = socket.socket(socket.AF_UNIX)
try:
    s.connect({host_socket!r})
    raise AssertionError('host pathname Unix socket was reachable')
except OSError:
    pass
finally:
    s.close()
{abstract_check}
path = os.path.join(os.environ['TMPDIR'], 'private.sock')
server = socket.socket(socket.AF_UNIX)
client = socket.socket(socket.AF_UNIX)
accepted = None
try:
    server.bind(path)
    server.listen(1)
    client.connect(path)
    accepted, _ = server.accept()
    client.sendall(b'client-to-server')
    assert accepted.recv(16) == b'client-to-server'
    accepted.sendall(b'server-to-client')
    assert client.recv(16) == b'server-to-client'
finally:
    if accepted is not None:
        accepted.close()
    client.close()
    server.close()
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
assert not os.path.exists(path)
print('{_PROBE_TOKEN}')
"""


def _live_probe(backend: SandboxBackend) -> SandboxCapability:
    """Enter the proposed sandbox and verify representative restrictions."""
    host_path_socket: socket.socket | None = None
    host_abstract_socket: socket.socket | None = None
    host_ipv4_socket: socket.socket | None = None
    host_ipv6_socket: socket.socket | None = None
    host_udp_socket: socket.socket | None = None
    inherited_fds: list[int] = []
    prepared: PreparedSandboxLaunch | None = None
    try:
        with tempfile.TemporaryDirectory(prefix = "unsloth-sandbox-probe-") as base:
            workdir = os.path.join(base, "work")
            os.mkdir(workdir)
            with open(os.path.join(workdir, "probe-read"), "w", encoding = "utf-8") as stream:
                stream.write("readable")
            external = os.path.join(base, "host-secret")
            with open(external, "w", encoding = "utf-8") as stream:
                stream.write("secret")
            os.symlink(external, os.path.join(workdir, "escape"))
            host_socket_path = os.path.join(base, "host.sock")
            host_path_socket = socket.socket(socket.AF_UNIX)
            host_path_socket.bind(host_socket_path)
            host_path_socket.listen(1)
            host_ipv4_socket = socket.socket(socket.AF_INET)
            host_ipv4_socket.bind(("127.0.0.1", 0))
            host_ipv4_socket.listen(1)
            ipv4_address = host_ipv4_socket.getsockname()
            host_ipv6_socket = socket.socket(socket.AF_INET6)
            host_ipv6_socket.bind(("::1", 0))
            host_ipv6_socket.listen(1)
            ipv6_address = host_ipv6_socket.getsockname()
            host_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            host_udp_socket.bind(("127.0.0.1", 0))
            udp_address = host_udp_socket.getsockname()
            import fcntl

            probe_sources: list[int] = []
            try:
                probe_sources.append(os.open(external, os.O_RDONLY))
                probe_sources.append(os.open(base, os.O_RDONLY))
                read_fd, write_fd = os.pipe()
                probe_sources.extend((read_fd, write_fd, host_path_socket.fileno()))
                for source_fd in probe_sources:
                    high_fd = fcntl.fcntl(source_fd, fcntl.F_DUPFD, 200)
                    os.set_inheritable(high_fd, True)
                    inherited_fds.append(high_fd)
            finally:
                for source_fd in probe_sources:
                    if source_fd != host_path_socket.fileno():
                        os.close(source_fd)
            abstract_name: str | None = None
            if sys.platform == "linux":
                abstract_name = "\0unsloth-sandbox-probe-" + str(os.getpid())
                host_abstract_socket = socket.socket(socket.AF_UNIX)
                host_abstract_socket.bind(abstract_name)
                host_abstract_socket.listen(1)
            env = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": workdir,
                "TMPDIR": "/tmp",
                "LANG": "C.UTF-8",
                "PYTHONIOENCODING": "utf-8",
            }
            host_namespaces = {
                name: os.readlink(f"/proc/self/ns/{name}")
                for name in ("mnt", "pid", "net", "ipc", "user")
            }
            spec = SandboxLaunchSpec(
                argv = (
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    _probe_payload(
                        workdir,
                        external,
                        host_socket_path,
                        os.getpid(),
                        abstract_name,
                        ipv4_address,
                        ipv6_address,
                        udp_address,
                        host_namespaces,
                        tuple(inherited_fds),
                    ),
                ),
                workdir = workdir,
                env = env,
            )
            prepared = backend.prepare(spec)
            run_kwargs = dict(
                cwd = prepared.workdir,
                env = prepared.env,
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                text = True,
                timeout = _PROBE_TIMEOUT_SECONDS,
                close_fds = True,
                stdin = subprocess.DEVNULL,
            )
            if prepared.preexec_fn is not None and os.name == "posix":
                run_kwargs["preexec_fn"] = prepared.preexec_fn
            if prepared.pass_fds:
                run_kwargs["pass_fds"] = prepared.pass_fds
            completed = subprocess.run(prepared.argv, **run_kwargs)
            if completed.returncode != 0 or _PROBE_TOKEN not in completed.stdout:
                detail = completed.stderr.strip()[-300:] or completed.stdout.strip()[-300:]
                return SandboxCapability(
                    backend.identity,
                    False,
                    f"the restrictive live probe failed ({completed.returncode}): {detail}",
                )
            host_udp_socket.settimeout(0.05)
            try:
                udp_payload, _ = host_udp_socket.recvfrom(256)
            except (TimeoutError, socket.timeout):
                pass
            else:
                if udp_payload == _PROBE_UDP_TOKEN:
                    return SandboxCapability(
                        backend.identity,
                        False,
                        "the restrictive live probe reached the host UDP listener",
                    )
    except subprocess.TimeoutExpired:
        return SandboxCapability(
            backend.identity, False, "the restrictive live probe timed out", transient = True
        )
    except OSError as exc:
        transient = exc.errno == errno.EAGAIN
        return SandboxCapability(
            backend.identity,
            False,
            f"the restrictive live probe could not run: {exc}",
            transient = transient,
        )
    except Exception as exc:  # noqa: BLE001 - capability failure must block, not crash Studio
        return SandboxCapability(
            backend.identity, False, f"the restrictive live probe failed: {exc}"
        )
    finally:
        if prepared is not None:
            prepared.cleanup()
        if host_path_socket is not None:
            host_path_socket.close()
        if host_abstract_socket is not None:
            host_abstract_socket.close()
        if host_ipv4_socket is not None:
            host_ipv4_socket.close()
        if host_ipv6_socket is not None:
            host_ipv6_socket.close()
        if host_udp_socket is not None:
            host_udp_socket.close()
        for inherited_fd in inherited_fds:
            try:
                os.close(inherited_fd)
            except OSError:
                pass
    return SandboxCapability(backend.identity, True, "restrictive live probe passed")


_LINUX_BACKEND = LinuxBubblewrapBackend()
_MACOS_BACKEND = MacOSSeatbeltBackend()
_WINDOWS_BACKEND: SandboxBackend | None = None
_capability_cache: dict[str, SandboxCapability] = {}
_probe_lock = threading.Lock()


def _platform_backend() -> SandboxBackend | None:
    global _WINDOWS_BACKEND
    if sys.platform == "linux":
        return _LINUX_BACKEND
    if sys.platform == "darwin":
        return _MACOS_BACKEND
    if sys.platform == "win32":
        if _WINDOWS_BACKEND is None:
            from .windows_lpac import WindowsLpacBackend
            _WINDOWS_BACKEND = WindowsLpacBackend()
        return _WINDOWS_BACKEND
    return None


def _capability_with_identity(
    capability: SandboxCapability, *, environment: str, fingerprint: str
) -> SandboxCapability:
    available = capability.qualified if capability.available is None else capability.available
    protection_state = capability.protection_state if available else "unavailable"
    if available and protection_state == "unavailable":
        protection_state = "protected" if environment == "native_linux" else "preview"
    profile_id = capability.profile_id if available else "none"
    generation_payload = "\0".join(
        (
            fingerprint,
            capability.backend,
            str(available),
            str(capability.qualified),
            capability.reason,
            protection_state,
            profile_id,
            *capability.limitations,
        )
    ).encode()
    generation = hashlib.sha256(generation_payload).hexdigest()
    remediation = (
        "No remediation required."
        if available
        else "Use Limited mode only for a trusted task, or install and enable a qualified OS sandbox backend."
    )
    return replace(
        capability,
        available = available,
        environment = environment,
        protection_state = protection_state,
        profile_id = profile_id,
        probe_generation = generation,
        environment_fingerprint = fingerprint,
        remediation = remediation,
        retryable = capability.transient,
    )


def capability_snapshot(*, force: bool = False) -> SandboxCapability:
    backend = _platform_backend()
    environment = _environment_class()
    fingerprint = _environment_fingerprint(backend)
    if backend is None:
        return _capability_with_identity(
            SandboxCapability(
                f"unsupported-{sys.platform}",
                False,
                f"OS sandboxing is unsupported on {sys.platform}",
            ),
            environment = environment,
            fingerprint = fingerprint,
        )
    cached = _capability_cache.get(fingerprint)
    if cached is not None and not force:
        return cached
    with _probe_lock:
        current_fingerprint = _environment_fingerprint(backend)
        cached = _capability_cache.get(current_fingerprint)
        if cached is not None and not force:
            return cached
        if force:
            _capability_cache.clear()
        result = _capability_with_identity(
            backend.probe(),
            environment = _environment_class(),
            fingerprint = current_fingerprint,
        )
        if not result.transient:
            _capability_cache.clear()
            _capability_cache[current_fingerprint] = result
        return result


def sandbox_capability() -> SandboxCapability:
    """Backward-compatible capability accessor."""
    return capability_snapshot()


_LIMITED_SAFEGUARDS = (
    "process_guard",
    "command_and_code_analysis",
    "sanitized_environment",
    "resource_limits",
    "descriptor_closure",
    "workdir_policy",
    "streaming",
    "timeout",
    "cancellation",
    "reaping",
    "cleanup",
)
_FULL_SAFEGUARDS = ("timeout", "cancellation", "reaping", "cleanup")


def prepare_tool_launch(spec: ToolLaunchPlan) -> PreparedSandboxLaunch:
    """Finalize one launch in its requested mode without fallback or replay."""
    if not spec.argv:
        raise ValueError("tool launch argv must not be empty")
    if spec.requested_mode not in ("os_isolation_required", "limited", "full"):
        raise SandboxUnavailableError("unknown tool execution mode")
    if not spec.close_fds or not spec.terminate_descendants:
        raise SandboxUnavailableError(
            "tool launches must close inherited descriptors and own descendant cleanup"
        )
    canonical = replace(spec, workdir = os.path.realpath(spec.workdir))
    backend = _platform_backend()
    capability = capability_snapshot()

    if canonical.requested_mode == "limited":
        if capability.available:
            raise SandboxUnavailableError(
                "OS isolation is available; Limited mode is not authorized for this capability generation"
            )
        if not canonical.current_subject or not canonical.tool_ui_session_id:
            raise SandboxUnavailableError(
                "Limited mode requires an authenticated Studio UI session"
            )
        try:
            from .tool_isolation import LimitedGrantError, validate_limited_grant
        except ImportError as exc:
            raise SandboxUnavailableError("Limited mode authorization is unavailable") from exc
        try:
            validate_limited_grant(
                canonical.limited_grant,
                current_subject = canonical.current_subject,
                tool_ui_session_id = canonical.tool_ui_session_id,
                probe_generation = capability.probe_generation,
                requested_mode = "limited",
            )
        except LimitedGrantError as exc:
            raise SandboxUnavailableError(f"Limited mode authorization failed: {exc}") from exc
        record = ToolExecutionRecord(
            requested_mode = "limited",
            effective_mode = "limited",
            environment = capability.environment,
            backend = "process-guard",
            profile_id = "limited-software-safeguards-v1",
            probe_generation = capability.probe_generation,
            os_isolation = False,
            retained_safeguards = _LIMITED_SAFEGUARDS,
        )
        return PreparedSandboxLaunch(
            argv = canonical.argv,
            workdir = canonical.workdir,
            env = canonical.env,
            preexec_fn = canonical.preexec_fn,
            backend = record.backend,
            execution_record = record,
            timeout_seconds = canonical.timeout_seconds,
            close_fds = canonical.close_fds,
            terminate_descendants = canonical.terminate_descendants,
        )

    if canonical.requested_mode == "full":
        record = ToolExecutionRecord(
            requested_mode = "full",
            effective_mode = "full",
            environment = capability.environment,
            backend = "none",
            profile_id = "full-access-v1",
            probe_generation = capability.probe_generation,
            os_isolation = False,
            retained_safeguards = _FULL_SAFEGUARDS,
        )
        return PreparedSandboxLaunch(
            argv = canonical.argv,
            workdir = canonical.workdir,
            env = canonical.env,
            preexec_fn = canonical.preexec_fn,
            backend = record.backend,
            execution_record = record,
            timeout_seconds = canonical.timeout_seconds,
            close_fds = canonical.close_fds,
            terminate_descendants = canonical.terminate_descendants,
        )

    if backend is None or not capability.available:
        raise SandboxUnavailableError(
            f"OS_ISOLATION_UNAVAILABLE: {capability.reason}. {capability.remediation}"
        )
    try:
        prepared = backend.prepare(canonical)
    except SandboxUnavailableError:
        raise
    except Exception as exc:
        raise SandboxUnavailableError(
            f"{backend.identity} could not prepare the process: {exc}"
        ) from exc
    prepared.execution_record = ToolExecutionRecord(
        requested_mode = "os_isolation_required",
        effective_mode = "os_isolation_required",
        environment = capability.environment,
        backend = capability.backend,
        profile_id = capability.profile_id,
        probe_generation = capability.probe_generation,
        os_isolation = True,
        retained_safeguards = (*_LIMITED_SAFEGUARDS, "os_isolation"),
        limitations = capability.limitations,
    )
    return prepared
