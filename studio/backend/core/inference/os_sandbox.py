# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fail-closed OS isolation for Studio's local Python and Terminal tools."""

from __future__ import annotations

import errno
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
from typing import BinaryIO, Callable, Protocol

from loggers import get_logger

logger = get_logger(__name__)

_SCAN_ENTRY_LIMIT = 100_000
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


class SandboxUnavailableError(RuntimeError):
    """The required native sandbox cannot safely launch this tool call."""


@dataclass(frozen = True)
class SandboxCapability:
    backend: str
    qualified: bool
    reason: str
    transient: bool = False


@dataclass(frozen = True)
class SandboxLaunchSpec:
    """Everything the backend needs to prepare one existing process launch."""

    argv: tuple[str, ...]
    workdir: str
    env: dict[str, str]
    preexec_fn: Callable[[], None] | None = None
    launcher_preexec_fn: Callable[[], None] | None = None


@dataclass
class PreparedSandboxLaunch:
    """A native sandbox argv plus resources owned until the process exits."""

    argv: tuple[str, ...]
    workdir: str
    env: dict[str, str]
    preexec_fn: Callable[[], None] | None
    backend: str
    pass_fds: tuple[int, ...] = ()
    owned_files: list[BinaryIO] = field(default_factory = list)
    cleanup_paths: list[str] = field(default_factory = list)

    def cleanup(self) -> None:
        while self.owned_files:
            self.owned_files.pop().close()
        while self.cleanup_paths:
            path = self.cleanup_paths.pop()
            try:
                shutil.rmtree(path)
            except OSError:
                logger.warning("Could not remove private sandbox path %s", path, exc_info = True)


class SandboxBackend(Protocol):
    identity: str

    def probe(self) -> SandboxCapability: ...

    def prepare(self, spec: SandboxLaunchSpec) -> PreparedSandboxLaunch: ...


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
                f"the session workdir contains a socket, FIFO, or device node: {path}"
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


def _linux_mount_points() -> tuple[str, ...]:
    points: list[str] = []
    try:
        with open("/proc/self/mountinfo", encoding = "utf-8") as stream:
            for line in stream:
                fields = line.split()
                if len(fields) > 4:
                    points.append(
                        fields[4]
                        .replace("\\040", " ")
                        .replace("\\011", "\t")
                        .replace("\\012", "\n")
                        .replace("\\134", "\\")
                    )
    except OSError as exc:
        raise SandboxUnavailableError("cannot inspect Linux nested mounts") from exc
    return tuple(points)


def _runtime_read_paths() -> tuple[str, ...]:
    """Return selected interpreter/library roots, never arbitrary inherited sys.path."""
    executable = os.path.abspath(sys.executable)
    candidates: list[str] = [
        executable,
        os.path.realpath(executable),
        os.path.join(sys.prefix, "pyvenv.cfg"),
        os.path.join(sys.prefix, "lib"),
        os.path.join(sys.prefix, "lib64"),
        os.path.join(sys.base_prefix, "lib"),
        os.path.join(sys.base_prefix, "lib64"),
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
) -> None:
    """User-managed runtimes may be read-only, but must not carry host IPC into the jail."""
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

    if sys.platform == "linux":
        for mount in _linux_mount_points():
            if any(_contained(mount, root, strict = True) for root in scan_roots):
                raise SandboxUnavailableError(
                    f"an interpreter/runtime path contains a nested host mount: {mount}"
                )

    find = "/usr/bin/find"
    if scan_roots and _trusted_linux_executable(find):
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
                "an interpreter/runtime path contains a socket, FIFO, or device node: "
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
                        f"runtime path contains a socket, FIFO, or device node: {path}"
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


def _excluded_linux_environment() -> str | None:
    release = ""
    try:
        with open("/proc/sys/kernel/osrelease", encoding = "utf-8") as stream:
            release = stream.read().lower()
    except OSError:
        pass
    if "microsoft" in release or os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return "WSL is not a qualified sandbox host"
    cgroup = ""
    try:
        with open("/proc/1/cgroup", encoding = "utf-8") as stream:
            cgroup = stream.read().lower()
    except OSError:
        pass
    container_markers = ("docker", "kubepods", "containerd", "libpod", "podman", "lxc")
    if (
        os.path.exists("/.dockerenv")
        or os.path.exists("/run/.containerenv")
        or os.environ.get("container")
        or any(marker in cgroup for marker in container_markers)
    ):
        return "containers are not qualified sandbox hosts"
    detector = "/usr/bin/systemd-detect-virt"
    if not _trusted_linux_executable(detector):
        return "cannot verify that this Linux host is outside a container"
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
        return "cannot verify that this Linux host is outside a container"
    if detected.returncode == 0:
        return "containers are not qualified sandbox hosts"
    if detected.returncode != 1:
        return "cannot verify that this Linux host is outside a container"
    if os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:
        return "Colab is not a qualified sandbox host"
    return None


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


class LinuxBubblewrapBackend:
    identity = "linux-bubblewrap"

    def __init__(self) -> None:
        self._bwrap: str | None = None

    def probe(self) -> SandboxCapability:
        excluded = _excluded_linux_environment()
        if excluded:
            return SandboxCapability(self.identity, False, excluded)
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
            )
        except SandboxUnavailableError as exc:
            return SandboxCapability(
                self.identity,
                False,
                f"a read-only Linux system root is unsafe to expose: {exc}",
            )
        self._bwrap = candidate
        return _live_probe(self)

    def prepare(self, spec: SandboxLaunchSpec) -> PreparedSandboxLaunch:
        if self._bwrap is None:
            raise SandboxUnavailableError("Bubblewrap was not qualified in this process")
        workdir = _validate_workdir(spec.workdir)
        runtime_paths = _runtime_read_paths()
        _validate_runtime_paths(runtime_paths, workdir)
        seccomp_filter = _linux_seccomp_filter()
        try:
            identity_dir, passwd, group = _identity_files()
        except Exception:
            seccomp_filter.close()
            raise
        env = dict(spec.env)
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
        argv.extend(("--dir", workdir, "--remount-ro", "/"))
        argv.extend(("--tmpfs", "/dev/shm", "--tmpfs", "/tmp"))
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
        )


class MacOSSeatbeltBackend:
    identity = "macos-seatbelt"

    def probe(self) -> SandboxCapability:
        return SandboxCapability(
            self.identity,
            False,
            "macOS Seatbelt is not qualified because detached sandbox descendants cannot yet "
            "be reliably terminated",
        )

    def prepare(self, spec: SandboxLaunchSpec) -> PreparedSandboxLaunch:
        raise SandboxUnavailableError("macOS Seatbelt is not qualified for Studio tool execution")


def _probe_payload(
    workdir: str,
    external_file: str,
    host_socket: str,
    host_pid: int,
    abstract_socket: str | None,
    ipv4_address: tuple[str, int],
    ipv6_address: tuple[str, int, int, int],
    udp_address: tuple[str, int],
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
    return f"""import os, socket
wd = {workdir!r}
with open(os.path.join(wd, 'probe-write'), 'w', encoding='utf-8') as f:
    f.write('ok')
assert not os.path.exists({external_file!r})
try:
    open('/unsloth-host-escape', 'w').close()
    raise AssertionError('wrote outside workdir')
except OSError:
    pass
assert not os.path.exists('/proc/{host_pid}/environ')
if hasattr(socket, 'AF_VSOCK'):
    try:
        vsock = socket.socket(socket.AF_VSOCK)
    except OSError:
        pass
    else:
        vsock.close()
        raise AssertionError('AF_VSOCK remained available')
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
try:
    server.bind(path)
    server.listen(1)
    client.connect(path)
    accepted, _ = server.accept()
    accepted.close()
finally:
    client.close()
    server.close()
print('{_PROBE_TOKEN}')
"""


def _live_probe(backend: SandboxBackend) -> SandboxCapability:
    """Enter the proposed sandbox and verify representative restrictions."""
    host_path_socket: socket.socket | None = None
    host_abstract_socket: socket.socket | None = None
    host_ipv4_socket: socket.socket | None = None
    host_ipv6_socket: socket.socket | None = None
    host_udp_socket: socket.socket | None = None
    prepared: PreparedSandboxLaunch | None = None
    try:
        with tempfile.TemporaryDirectory(prefix = "unsloth-sandbox-probe-") as base:
            workdir = os.path.join(base, "work")
            os.mkdir(workdir)
            external = os.path.join(base, "host-secret")
            with open(external, "w", encoding = "utf-8") as stream:
                stream.write("secret")
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
    return SandboxCapability(backend.identity, True, "restrictive live probe passed")


_LINUX_BACKEND = LinuxBubblewrapBackend()
_MACOS_BACKEND = MacOSSeatbeltBackend()
_capability_cache: dict[str, SandboxCapability] = {}
_probe_lock = threading.Lock()


def _platform_backend() -> SandboxBackend | None:
    if sys.platform == "linux":
        return _LINUX_BACKEND
    if sys.platform == "darwin":
        return _MACOS_BACKEND
    return None


def sandbox_capability() -> SandboxCapability:
    backend = _platform_backend()
    if backend is None:
        return SandboxCapability(
            f"unsupported-{sys.platform}",
            False,
            f"OS sandboxing is unsupported on {sys.platform}; use Full/Bypass Permissions "
            "only when host execution is explicitly intended",
        )
    cached = _capability_cache.get(backend.identity)
    if cached is not None:
        return cached
    with _probe_lock:
        cached = _capability_cache.get(backend.identity)
        if cached is not None:
            return cached
        result = backend.probe()
        if not result.transient:
            _capability_cache[backend.identity] = result
        return result


def prepare_tool_launch(spec: SandboxLaunchSpec) -> PreparedSandboxLaunch:
    """Prepare one ordinary tool launch; never fall back to the host process."""
    if not spec.argv:
        raise ValueError("sandbox launch argv must not be empty")
    backend = _platform_backend()
    capability = sandbox_capability()
    if backend is None or not capability.qualified:
        raise SandboxUnavailableError(capability.reason)
    canonical = replace(spec, workdir = os.path.realpath(spec.workdir))
    try:
        return backend.prepare(canonical)
    except SandboxUnavailableError:
        raise
    except Exception as exc:
        raise SandboxUnavailableError(
            f"{backend.identity} could not prepare the process: {exc}"
        ) from exc
