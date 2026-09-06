# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fail-closed OS isolation for Studio's local Python and Terminal tools."""

from __future__ import annotations

import errno
import ctypes
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, BinaryIO, Callable, Literal, Protocol

from loggers import get_logger

from .network_proxy import (
    ALLOWLIST_ENV as NETWORK_ALLOWLIST_ENV,
    NO_PROXY_VALUE as _NO_PROXY_VALUE,
    PROXY_ENV_KEYS as _PROXY_ENV_KEYS,
    AllowlistError,
    AllowlistProxy,
    NetworkAllowlist,
    NetworkAudit,
    proxy_environment,
    tls_trust_environment,
    tls_trust_paths,
)

logger = get_logger(__name__)

_SCAN_ENTRY_LIMIT = 100_000
_REJECTED_RUNTIME_ENTRY_TYPES = "Unix sockets, FIFOs, block devices, or character devices"
_PROBE_TIMEOUT_SECONDS = 8
# Interpreter roots are small and scanned on every launch. The read-only system
# roots hold hundreds of thousands of entries on a developer machine or a CI
# image and a cold page cache can take a minute to traverse them, so they get a
# wider budget and a passed scan is remembered for a while (see
# _system_scan_memo). A timeout is reported as transient so it is never cached
# as a permanent "unavailable".
_RUNTIME_SCAN_TIMEOUT_SECONDS = 8
# Measured cold traversals of the hosted GitHub runner images took 44 s (Ubuntu
# 22.04, 328k entries) and 76 s (Ubuntu 24.04, 271k entries); warm they take
# about one second and the memo below makes later launches free.
_SYSTEM_SCAN_TIMEOUT_SECONDS = 120
_SYSTEM_SCAN_MEMO_SECONDS = 600
_SYMLINK_CHAIN_LIMIT = 40
_CLONE_NEWUSER = 0x10000000
_LIMITATION_NESTED_USERNS_SECCOMP = "nested_userns_blocked_by_seccomp"
_LIMITATION_IPV6_UNAVAILABLE = "ipv6_unavailable_on_host"
_LIMITATION_NETWORK_ALLOWLIST_INVALID = "network_allowlist_invalid"
_GENERIC_REMEDIATION = (
    "Use Limited mode only for a trusted task, or install and enable a qualified OS sandbox backend."
)
_APPARMOR_USERNS_SYSCTL = "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
_DEFAULT_BWRAP_PATH = "/usr/bin/bwrap"


def _apparmor_userns_remediation(bwrap_path: str = _DEFAULT_BWRAP_PATH) -> str:
    """The profile text names the Bubblewrap binary that was actually probed.

    AppArmor attaches profiles by executable path, so a profile for
    /usr/bin/bwrap does nothing for a qualified /usr/local/bin/bwrap or a Nix
    store path.
    """
    return (
        "This host restricts unprivileged user namespaces with AppArmor "
        "(kernel.apparmor_restrict_unprivileged_userns=1, the default on Ubuntu 24.04 and later), "
        "so Bubblewrap cannot create its sandbox. Allow Bubblewrap with a profile: as root, create "
        "/etc/apparmor.d/bwrap containing\n"
        "  abi <abi/4.0>,\n"
        "  include <tunables/global>\n"
        f"  profile bwrap {bwrap_path} flags=(unconfined) {{\n"
        "    userns,\n"
        "    include if exists <local/bwrap>\n"
        "  }\n"
        "then run `apparmor_parser -r /etc/apparmor.d/bwrap` (or `systemctl reload apparmor`) and "
        "choose Check again. Until then, use Limited mode only for a trusted task."
    )


_APPARMOR_USERNS_REMEDIATION = _apparmor_userns_remediation()
_APPARMOR_USERNS_MARKERS = (
    "RTM_NEWADDR",
    "Operation not permitted",
    "setting up uid map",
    "No permissions to create",
)
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
# CA trust roots, bound read-only only when the network allowlist is on: without
# them OpenSSL inside the sandbox has no bundle (Debian keeps it under /etc/ssl,
# Fedora and RHEL under /etc/pki) and every HTTPS fetch through the proxy fails
# certificate verification. pip carries certifi, curl, git and urllib do not.
_LINUX_CA_TRUST_PATHS = (
    "/etc/ssl",
    "/etc/pki",
    "/etc/ca-certificates",
    "/etc/ca-certificates.conf",
    "/etc/crypto-policies",
)
_WSL_HIDDEN_PATHS = ("/usr/lib/wsl",)


class SandboxUnavailableError(RuntimeError):
    """The required native sandbox cannot safely launch this tool call.

    ``transient`` marks conditions that may clear on their own (a scan that ran
    out of time on a cold disk cache); the capability layer reports those as
    retryable instead of caching them as a permanent unavailability.
    """

    def __init__(self, message: str = "", *, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


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
    # What a Limited launch on this host runs under. Everywhere but Windows this
    # is the process guard alone; on Windows the write-restricted token launcher
    # takes over once its own live probe passed. Not part of probe_generation:
    # a grant issued before the launcher qualified stays valid.
    limited_backend: str = "process-guard"
    limited_profile_id: str = "limited-software-safeguards-v1"
    limited_limitations: tuple[str, ...] = ()
    limited_reason: str = ""
    # Network policies the backend can enforce for an OS-isolated launch, and the
    # hosts the "allowlist" policy would admit. "deny" is always present; a
    # backend without a loopback bridge (Windows AppContainer) offers only that.
    network_policies: tuple[str, ...] = ("deny",)
    network_allowlist: tuple[str, ...] = ()


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
    # "deny": no network path out of the sandbox. "allowlist": CONNECT tunnels to
    # network_allowlist hosts through the per-launch loopback proxy.
    # "unrestricted": the launch has the host's network (Limited and Full).
    network_policy: str = "deny"
    network_allowlist: tuple[str, ...] = ()

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
            "network_policy": self.network_policy,
            "network_allowlist": list(self.network_allowlist),
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
    # "deny" (default) or "allowlist". Only honored for os_isolation_required;
    # Full has the host network anyway and Limited cannot enforce a proxy.
    network_policy: str = "deny"


NETWORK_POLICIES = ("deny", "allowlist")


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
    # Set when the launch runs behind the allowlist proxy; tools.py reads the
    # refused hosts from it for the result trailer.
    network_audit: NetworkAudit | None = None

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


# Syscall numbers for the user-namespace fallback filter (identical on the two
# reviewed ABIs: x86_64 uses clone=56, unshare=272; aarch64 clone=220, unshare=97).
_LINUX_USERNS_SYSCALLS = {
    "aarch64": (220, 97, 435),
    "arm64": (220, 97, 435),
    "amd64": (56, 272, 435),
    "x86_64": (56, 272, 435),
}


def _linux_seccomp_program(*, block_userns: bool = False) -> bytes:
    """Build the BPF program used by _linux_seccomp_filter (kept separate for tests)."""
    machine = platform.machine().lower()
    abi = _LINUX_SECCOMP_ABIS.get(machine)
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
    return_enosys = 0x00050000 | errno.ENOSYS
    allow = 0x7FFF0000
    io_uring_setup_nr = 425
    instructions: list[tuple[int, int, int, int]] = [
        (load_word, 0, 0, 4),
        (jump_equal, 1, 0, audit_arch),
        (return_value, 0, 0, kill_process),
        (load_word, 0, 0, 0),
    ]
    if block_userns:
        # Bubblewrap older than 0.8.0 has no --disable-userns, so nested user
        # namespaces are refused here instead: unshare() always fails, clone3()
        # reports ENOSYS so libc falls back to clone(), and clone() with
        # CLONE_NEWUSER in its flags word fails. Everything else falls through
        # to the socket family checks below.
        clone_nr, unshare_nr, clone3_nr = _LINUX_USERNS_SYSCALLS[machine]
        instructions.extend(
            [
                (jump_equal, 0, 1, unshare_nr),
                (return_value, 0, 0, return_errno),
                (jump_equal, 0, 1, clone3_nr),
                (return_value, 0, 0, return_enosys),
                (jump_equal, 0, 4, clone_nr),
                (load_word, 0, 0, 16),
                (jump_bits_set, 0, 1, _CLONE_NEWUSER),
                (return_value, 0, 0, return_errno),
                (load_word, 0, 0, 0),
            ]
        )
    instructions.extend(
        [
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
        ]
    )
    return b"".join(struct.pack("=HBBI", *instruction) for instruction in instructions)


def _linux_seccomp_filter(*, block_userns: bool = False) -> BinaryIO:
    """Compile a minimal filter for host-channel socket families Bubblewrap cannot hide."""
    program = _linux_seccomp_program(block_userns = block_userns)
    stream = tempfile.TemporaryFile(prefix = "unsloth-sandbox-seccomp-")
    try:
        stream.write(program)
        stream.flush()
        stream.seek(0)
    except Exception:
        stream.close()
        raise
    return stream


_bwrap_options_cache: dict[tuple[str, int, int], frozenset[str]] = {}


def _bwrap_supported_options(bwrap: str) -> frozenset[str]:
    """Long options the installed Bubblewrap accepts, read once from its usage text.

    Ubuntu 22.04 ships 0.6.1, which predates ``--disable-userns`` (0.8.0), so the
    argv cannot assume it. The result is cached by path and file identity, the
    same facts the environment fingerprint already tracks.
    """
    try:
        info = os.stat(bwrap)
        key = (bwrap, info.st_ino, info.st_mtime_ns)
    except OSError:
        key = (bwrap, 0, 0)
    cached = _bwrap_options_cache.get(key)
    if cached is not None:
        return cached
    text = ""
    try:
        completed = subprocess.run(
            [bwrap, "--help"],
            stdin = subprocess.DEVNULL,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
            close_fds = True,
        )
        text = f"{completed.stdout}\n{completed.stderr}"
    except (OSError, subprocess.SubprocessError):
        text = ""
    options = frozenset(
        token.strip().rstrip(",")
        for line in text.splitlines()
        for token in line.split()
        if token.startswith("--")
    )
    _bwrap_options_cache[key] = options
    return options


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


def _fingerprint_roots() -> tuple[str, ...]:
    """Paths whose mount topology the sandbox actually exposes or masks."""
    roots: list[str] = [
        "/", *_LINUX_SYSTEM_ROOTS, *_LINUX_ETC_FILES, *_LINUX_CA_TRUST_PATHS, "/etc", "/nix/store", "/tmp"
    ]
    try:
        roots.extend(_runtime_read_paths())
    except Exception:  # noqa: BLE001 - interpreter introspection must not break fingerprinting
        pass
    return tuple(roots)


def _relevant_mounts(
    mounts: tuple[_LinuxMount, ...], roots: tuple[str, ...]
) -> tuple[_LinuxMount, ...]:
    """Mounts that sit under, or contain, one of ``roots``.

    A USB stick under /media, a container volume under /var/lib/docker or a
    gvfs mount under /run/user never enter the sandbox, so they must not
    invalidate the capability cache or outstanding Limited grants. The root
    mount is always relevant.
    """
    relevant: list[_LinuxMount] = []
    for mount in mounts:
        point = mount.mount_point
        if point == "/":
            relevant.append(mount)
            continue
        for root in roots:
            if root == "/":
                continue
            if _lexically_contained(point, root) or _lexically_contained(root, point):
                relevant.append(mount)
                break
    return tuple(relevant)


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


def _symlink_chain(path: str) -> list[str]:
    """Every path an exec of ``path`` resolves through, from the given spelling to the target.

    A virtualenv's ``bin/python`` usually points at another link
    (``.../bin/python -> python3 -> python3.12``, pyenv and hosted toolcaches
    do the same). Inside the sandbox the exec follows the same hops, so each
    one has to exist there; binding only the first and the last spelling leaves
    a dangling link in between and ``execvp`` fails with ENOENT.
    """
    hops: list[str] = []
    current = os.path.abspath(path)
    for _ in range(_SYMLINK_CHAIN_LIMIT):
        if current in hops:
            break
        hops.append(current)
        try:
            target = os.readlink(current)
        except OSError:
            break
        if sys.platform == "win32":
            # os.readlink reports absolute targets in extended-length form; keep
            # the ordinary spelling so hops compare and bind like every other path.
            if target.startswith("\\\\?\\UNC\\"):
                target = "\\\\" + target[len("\\\\?\\UNC\\"):]
            elif target.startswith("\\\\?\\"):
                target = target[len("\\\\?\\"):]
        current = os.path.normpath(os.path.join(os.path.dirname(current), target))
    return hops


def _runtime_read_paths() -> tuple[str, ...]:
    """Return selected interpreter/library roots, never arbitrary inherited sys.path."""
    executable = os.path.abspath(sys.executable)
    chain = _symlink_chain(executable)
    candidates: list[str] = [
        executable,
        *chain,
        os.path.realpath(executable),
        os.path.dirname(executable),
        *(os.path.dirname(hop) for hop in chain),
        os.path.dirname(os.path.realpath(executable)),
        os.path.join(sys.prefix, "bin"),
        os.path.join(sys.base_prefix, "bin"),
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


_system_scan_memo: dict[tuple[tuple[str, ...], str], float] = {}
_system_scan_lock = threading.Lock()


def _system_scan_signature(scan_roots: list[str]) -> str:
    """Identity of the mount topology under the scanned roots (a memo key component)."""
    try:
        mounts = _linux_mounts()
    except SandboxUnavailableError:
        return "mounts-unreadable"
    relevant = [
        (mount.mount_point, mount.major_minor, mount.fs_type, mount.source, mount.mount_options)
        for mount in _relevant_mounts(mounts, tuple(scan_roots))
    ]
    return hashlib.sha256(json.dumps(relevant, sort_keys = True).encode()).hexdigest()


def _forget_system_scan_memo() -> None:
    with _system_scan_lock:
        _system_scan_memo.clear()


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

    The root-owned system roots (``/usr`` and friends) are scanned with a wide
    budget and a passed scan is remembered for ``_SYSTEM_SCAN_MEMO_SECONDS`` as
    long as the mount topology under them is unchanged: only root can create a
    device node or socket there, and root is outside this boundary. Interpreter
    roots are user-writable and are re-scanned on every launch.
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

    memo_key: tuple[tuple[str, ...], str] | None = None
    if include_system_roots and sys.platform == "linux" and scan_roots:
        memo_key = (tuple(scan_roots), _system_scan_signature(scan_roots))
        with _system_scan_lock:
            passed_at = _system_scan_memo.get(memo_key)
        if passed_at is not None and time.monotonic() - passed_at < _SYSTEM_SCAN_MEMO_SECONDS:
            return
    scan_timeout = (
        _SYSTEM_SCAN_TIMEOUT_SECONDS if include_system_roots else _RUNTIME_SCAN_TIMEOUT_SECONDS
    )

    find = "/usr/bin/find"
    if sys.platform == "linux" and scan_roots and _trusted_linux_executable(find):
        # Follow a symlink used as a scan root, but never links encountered below it.
        find_command = [find, "-H", *scan_roots]
        find_command.append("-xdev")
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
                encoding = "utf-8",
                errors = "replace",
                timeout = scan_timeout,
                close_fds = True,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxUnavailableError(
                "cannot scan interpreter/runtime paths safely: the scan of "
                f"{', '.join(scan_roots)} exceeded {scan_timeout} s (a cold disk cache or a very "
                "large installation); this usually clears on retry",
                transient = True,
            ) from exc
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
        if memo_key is not None:
            with _system_scan_lock:
                _system_scan_memo.clear()
                _system_scan_memo[memo_key] = time.monotonic()
        return

    deadline = time.monotonic() + scan_timeout
    for root in scan_roots:
        entries = 0
        if os.path.isfile(root):
            continue

        def walk_error(exc: OSError) -> None:
            raise SandboxUnavailableError(
                f"runtime path cannot be fully inspected: {exc.filename or root}"
            ) from exc

        for base, dirs, names in os.walk(root, followlinks = False, onerror = walk_error):
            if time.monotonic() > deadline:
                raise SandboxUnavailableError(
                    "cannot scan interpreter/runtime paths safely: the scan of "
                    f"{', '.join(scan_roots)} exceeded {scan_timeout} s; retry the request",
                    transient = True,
                )
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


def _linux_environment(*, run_detector: bool = True) -> str:
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
    if not run_detector:
        return "linux_unknown"
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


def _environment_class(*, run_detector: bool = True) -> str:
    if sys.platform == "linux":
        if run_detector:
            return _linux_environment()
        return _linux_environment(run_detector = False)
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "windows"
    return f"unsupported-{sys.platform}"


def _excluded_linux_environment() -> str | None:
    """Compatibility hook: environment labels no longer reject qualification alone."""
    return None


def _environment_fingerprint(backend: "SandboxBackend | None", *, run_detector: bool = True) -> str:
    environment = _environment_class() if run_detector else _environment_class(run_detector = False)
    data: dict[str, object] = {
        "platform": sys.platform,
        "architecture": platform.machine().lower(),
        "environment": environment,
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
            data["mounts"] = sorted(
                [
                    mount.mount_point,
                    mount.root,
                    mount.fs_type,
                    mount.source,
                    mount.mount_options,
                    mount.super_options,
                ]
                for mount in _relevant_mounts(_linux_mounts(), _fingerprint_roots())
            )
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

# Runs inside the sandbox (before the exec above) when the launch carries the
# allowlist policy. The loopback listener has to be created here, inside the
# new network namespace, because a socket belongs to the namespace it was
# created in; the descriptor is handed to the host over an inherited AF_UNIX
# socketpair and the host's proxy accepts on it. The wrapper then waits for the
# host to confirm ("K <token>") before it publishes the proxy URL and execs.
_NETWORK_BRIDGE_ENV = "UNSLOTH_STUDIO_NET_CTRL_FD"
_NETWORK_BRIDGE_TIMEOUT_SECONDS = 30.0
_NETWORK_BRIDGE_BLOCK = """import socket
_ctrl_fd = int(os.environ.pop({ctrl_env!r}))
_ctrl = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, fileno = _ctrl_fd)
_ctrl.settimeout({timeout!r})
_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
_listener.bind(("127.0.0.1", 0))
_listener.listen(64)
socket.send_fds(_ctrl, [b"L"], [_listener.fileno()])
_reply = b""
while not _reply.endswith(b"\\n"):
    _chunk = _ctrl.recv(512)
    if not _chunk:
        raise SystemExit("sandbox network bridge: the host closed the control channel")
    _reply += _chunk
    if len(_reply) > 4096:
        raise SystemExit("sandbox network bridge: oversized reply")
_fields = _reply.strip().split(b" ")
if len(_fields) != 2 or _fields[0] != b"K":
    raise SystemExit("sandbox network bridge: unexpected reply")
_url = "http://sandbox:" + _fields[1].decode("ascii") + "@127.0.0.1:" + str(_listener.getsockname()[1])
for _key in {proxy_keys!r}:
    os.environ[_key] = _url
os.environ["NO_PROXY"] = os.environ["no_proxy"] = {no_proxy!r}
_listener.close()
_ctrl.close()
del _ctrl_fd, _ctrl, _listener, _reply, _chunk, _fields, _url, _key
"""


def _linux_wrapper_source(*, limit: int, network_bridge: bool) -> str:
    wrapper = _NPROC_WRAPPER.format(limit = limit)
    if not network_bridge:
        return wrapper
    block = _NETWORK_BRIDGE_BLOCK.format(
        ctrl_env = _NETWORK_BRIDGE_ENV,
        timeout = _NETWORK_BRIDGE_TIMEOUT_SECONDS,
        proxy_keys = tuple(_PROXY_ENV_KEYS),
        no_proxy = _NO_PROXY_VALUE,
    )
    marker = "os.execvpe("
    assert wrapper.count(marker) == 1
    return wrapper.replace(marker, block + marker)


def _receive_bridge_listener(
    host_end: socket.socket, timeout: float | None = None
) -> socket.socket:
    """Take the loopback listener the sandboxed wrapper created and vet it."""
    host_end.settimeout(_NETWORK_BRIDGE_TIMEOUT_SECONDS if timeout is None else timeout)
    try:
        message, fds, _flags, _addr = socket.recv_fds(host_end, 16, 1)
    except socket.timeout as exc:
        raise SandboxUnavailableError(
            "the sandboxed process did not hand over its network listener in time",
            transient = True,
        ) from exc
    except OSError as exc:
        raise SandboxUnavailableError(
            f"the network bridge control channel failed: {exc}", transient = True
        ) from exc
    listener: socket.socket | None = None
    try:
        if not message and not fds:
            raise SandboxUnavailableError(
                "the sandboxed process exited before handing over its network listener",
                transient = True,
            )
        if message != b"L" or len(fds) != 1:
            raise SandboxUnavailableError("the network bridge sent an unexpected message")
        listener = socket.socket(fileno = fds.pop())
        if listener.family != socket.AF_INET or listener.type != socket.SOCK_STREAM:
            raise SandboxUnavailableError("the network bridge listener is not a TCP socket")
        address, port = listener.getsockname()[:2]
        if address != "127.0.0.1" or not port:
            raise SandboxUnavailableError(
                f"the network bridge listener is bound to {address}:{port}, not loopback"
            )
        if listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) != 1:
            raise SandboxUnavailableError("the network bridge listener is not listening")
        return listener
    except Exception:
        if listener is not None:
            listener.close()
        raise
    finally:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass


def _network_allowlist_for_launch() -> NetworkAllowlist:
    try:
        return NetworkAllowlist.from_env()
    except AllowlistError as exc:
        raise SandboxUnavailableError(
            f"the network allowlist is invalid ({exc}); fix {NETWORK_ALLOWLIST_ENV} or disable network access"
        ) from exc


def _network_allowlist_hosts() -> tuple[str, ...] | None:
    """The configured hosts, or None when the environment override does not parse."""
    try:
        return NetworkAllowlist.from_env().hosts
    except AllowlistError:
        return None


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
    supports_network_allowlist = True
    identity = "linux-bubblewrap"
    profile_id = "linux-bubblewrap-v2"

    def __init__(self) -> None:
        self._bwrap: str | None = None
        # True when the installed bwrap accepts --disable-userns (0.8.0+). When it
        # does not, prepare() denies nested user namespaces with seccomp instead
        # and the capability carries _LIMITATION_NESTED_USERNS_SECCOMP.
        self._disable_userns_supported: bool = True

    def _limitations(self) -> tuple[str, ...]:
        return () if self._disable_userns_supported else (_LIMITATION_NESTED_USERNS_SECCOMP,)

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
            path
            for path in (*_LINUX_SYSTEM_ROOTS, *_LINUX_ETC_FILES, *_LINUX_CA_TRUST_PATHS)
            if os.path.exists(path)
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
                transient = exc.transient,
            )
        self._bwrap = candidate
        self._disable_userns_supported = "--disable-userns" in _bwrap_supported_options(candidate)
        result = _live_probe(self)
        limitations = tuple(dict.fromkeys((*result.limitations, *self._limitations())))
        if result.qualified:
            return replace(
                result,
                available = True,
                profile_id = self.profile_id,
                limitations = limitations,
            )
        return replace(
            _explain_linux_probe_failure(result, candidate), available = False, limitations = limitations
        )

    def prepare(self, spec: ToolLaunchPlan) -> PreparedSandboxLaunch:
        if self._bwrap is None:
            raise SandboxUnavailableError("Bubblewrap was not qualified in this process")
        workdir = _validate_workdir(spec.workdir)
        _validate_linux_workdir_environment(workdir)
        runtime_paths = _runtime_read_paths()
        system_roots = tuple(
            path
            for path in (*_LINUX_SYSTEM_ROOTS, *_LINUX_ETC_FILES, *_LINUX_CA_TRUST_PATHS)
            if os.path.exists(path)
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
        seccomp_filter = _linux_seccomp_filter(block_userns = not self._disable_userns_supported)
        try:
            identity_dir, passwd, group = _identity_files()
        except Exception:
            seccomp_filter.close()
            raise
        environment = _linux_environment()
        env = _sanitize_linux_environment(spec.env, environment)
        if spec.network_policy == "allowlist":
            # Not secret and not the proxy URL (that one is set inside the
            # namespace by the wrapper): the trust bundle for interpreters whose
            # OpenSSL has no default store, so TLS through the proxy verifies.
            env.update(tls_trust_environment(env))
        env["HOME"] = workdir
        env["TMPDIR"] = "/tmp"

        argv: list[str] = [
            self._bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--unshare-user",
            *(("--disable-userns",) if self._disable_userns_supported else ()),
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
        if spec.network_policy == "allowlist":
            for path in _LINUX_CA_TRUST_PATHS:
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
        if spec.network_policy == "allowlist":
            # OpenSSL's default store and certifi, only where an existing bind does
            # not already cover them: on Debian /usr/lib/ssl sits under /usr/lib and
            # certifi under the runtime tree, and mounting again on top of a bound
            # tree (through /usr/lib/ssl/certs, a symlink into /etc/ssl) made
            # bwrap exit before the wrapper ran (staging round 8). Bound after the
            # runtime paths so a later bind cannot shadow them.
            covered = (*_LINUX_SYSTEM_ROOTS, *_LINUX_CA_TRUST_PATHS, "/nix/store", *runtime_paths)
            for path in tls_trust_paths():
                if any(_lexically_contained(path, root) for root in covered):
                    continue
                if _lexically_contained(path, "/tmp"):
                    continue
                argv.extend(("--ro-bind-try", path, path))
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
        network_bridge = spec.network_policy == "allowlist"
        pass_fds: list[int] = [seccomp_filter.fileno()]
        cleanup_callbacks: list[Callable[[], None]] = []
        spawn_callback = None
        network_audit = None
        proxy: AllowlistProxy | None = None
        bridge_ends: tuple[socket.socket, socket.socket] | None = None
        # Nothing owns the proxy or the socketpair until PreparedSandboxLaunch
        # holds their cleanup callbacks, so anything that raises in between has
        # to close them here, the way the Seatbelt backend does.
        try:
            if network_bridge:
                allowlist = _network_allowlist_for_launch()
                proxy = AllowlistProxy(allowlist)
                host_end, sandbox_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
                bridge_ends = (host_end, sandbox_end)
                cleanup_callbacks.extend((proxy.close, host_end.close, sandbox_end.close))
                pass_fds.append(sandbox_end.fileno())
                argv.extend(("--setenv", _NETWORK_BRIDGE_ENV, str(sandbox_end.fileno())))
                spawn_callback = _bridged_spawn(proxy, host_end, sandbox_end)
                network_audit = proxy.audit
            wrapper = _linux_wrapper_source(
                limit = _nproc_limit(), network_bridge = network_bridge
            )
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
                pass_fds = tuple(pass_fds),
                owned_files = [seccomp_filter],
                cleanup_paths = [identity_dir],
                timeout_seconds = spec.timeout_seconds,
                close_fds = spec.close_fds,
                terminate_descendants = spec.terminate_descendants,
                spawn_callback = spawn_callback,
                cleanup_callbacks = cleanup_callbacks,
                network_audit = network_audit,
            )
        except Exception:
            if proxy is not None:
                proxy.close()
            for end in bridge_ends or ():
                try:
                    end.close()
                except OSError:
                    pass
            raise


def _bridged_spawn(
    proxy: AllowlistProxy, host_end: socket.socket, sandbox_end: socket.socket
) -> Callable[[PreparedSandboxLaunch, dict[str, Any]], object]:
    """Spawn, then complete the listener handshake before the tool runs.

    The sandboxed wrapper blocks until the host confirms, so the tool's own code
    never starts unless the proxy is accepting on a listener that lives inside
    the sandbox's network namespace. Any failure kills the process and surfaces
    as a transient SandboxUnavailableError instead of a tool that silently has
    no network.
    """

    def spawn(prepared: PreparedSandboxLaunch, popen_kwargs: dict[str, Any]) -> object:
        try:
            proc = subprocess.Popen(prepared.argv, **popen_kwargs)
        finally:
            # The child holds its own copy; the host's copy must go so a child
            # that dies early is seen as EOF instead of a hang until timeout.
            sandbox_end.close()
        try:
            listener = _receive_bridge_listener(host_end)
            proxy.serve_listener(listener)
            host_end.sendall(b"K " + proxy.credential.token.encode("ascii") + b"\n")
        except BaseException as exc:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout = 5)
            except Exception:  # noqa: BLE001 - best effort reap
                pass
            proxy.close()
            if isinstance(exc, SandboxUnavailableError):
                raise
            raise SandboxUnavailableError(
                f"the sandbox network bridge failed: {exc}", transient = True
            ) from exc
        finally:
            host_end.close()
        return proc

    return spawn


def _explain_linux_probe_failure(
    result: SandboxCapability, bwrap_path: str = _DEFAULT_BWRAP_PATH
) -> SandboxCapability:
    """Attach the known cause and its remediation when AppArmor blocked Bubblewrap.

    Ubuntu 24.04 and later ship ``kernel.apparmor_restrict_unprivileged_userns=1``
    and no profile for ``/usr/bin/bwrap``; the raw symptom is a loopback or uid
    map error. The sysctl read here is already part of the environment
    fingerprint, so the explanation cannot go stale silently.
    """
    if result.qualified:
        return result
    if _read_text(_APPARMOR_USERNS_SYSCTL).strip() != "1":
        return result
    if not any(marker in result.reason for marker in _APPARMOR_USERNS_MARKERS):
        return result
    return replace(
        result,
        reason = (
            "AppArmor restricts unprivileged user namespaces on this host "
            f"(kernel.apparmor_restrict_unprivileged_userns=1): {result.reason}"
        ),
        remediation = _apparmor_userns_remediation(os.path.realpath(bwrap_path)),
    )


class MacOSSeatbeltBackend:
    supports_network_allowlist = True
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
        proxy: AllowlistProxy | None = None
        try:
            proxy_port = None
            if spec.network_policy == "allowlist":
                # Seatbelt has no network namespace, so the proxy listens on the
                # host loopback and the profile admits exactly that port; every
                # other outbound destination stays under (deny default).
                proxy = AllowlistProxy(_network_allowlist_for_launch())
                proxy_port = proxy.listen_loopback()
            profile = _macos_seatbelt_profile(
                workdir = workdir,
                private_tmp = private_tmp,
                runtime_paths = runtime_paths,
                proxy_port = proxy_port,
            )
            env = _sanitize_macos_environment(spec.env, workdir, private_tmp)
            developer_paths = _macos_developer_paths()
            if developer_paths and "DEVELOPER_DIR" not in env:
                # xcselect honours DEVELOPER_DIR before it reads the xcode_select
                # link, so the /usr/bin shims (git, make, python3) resolve the real
                # tool without depending on the link being readable.
                env["DEVELOPER_DIR"] = developer_paths[0]
            if proxy is not None:
                env.update(proxy_environment(proxy.port, proxy.credential))
                env.update(tls_trust_environment(env))
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
                cleanup_callbacks = [proxy.close] if proxy is not None else [],
                network_audit = proxy.audit if proxy is not None else None,
            )
        except Exception:
            if proxy is not None:
                proxy.close()
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
    # /usr/bin/git and friends are xcode-select shims that resolve the developer
    # directory through this link and exec the real tool from it; without these
    # every git call inside the sandbox ends with "See man xcode-select".
    "/private/var/db/xcode_select_link",
    "/Library/Developer/CommandLineTools",
    "/Applications/Xcode.app/Contents/Developer",
)
# Readable only when the network allowlist is on: the trust store OpenSSL reads
# (/private/etc/ssl) and what Security.framework needs to evaluate a server
# certificate (the system keychains, the trust settings database, and trustd
# through mach-lookup); curl and Foundation clients cannot verify TLS without
# them, and Python's OpenSSL needs the cert.pem the system ships.
_MACOS_TLS_TRUST_PATHS = (
    "/private/etc/ssl",
    "/System/Library/Keychains",
    "/Library/Keychains",
    "/System/Library/Security",
    "/private/var/db/mds",
    "/Library/Preferences/com.apple.security.plist",
    "/Library/Preferences/com.apple.security.revocation.plist",
)
_MACOS_TLS_MACH_SERVICES = (
    "com.apple.SecurityServer",
    "com.apple.trustd",
    "com.apple.trustd.agent",
    "com.apple.ocspd",
)
_developer_paths_cache: tuple[str, ...] | None = None
_developer_paths_lock = threading.Lock()


def _macos_developer_paths() -> tuple[str, ...]:
    """The active developer directory, resolved once through xcode-select.

    /usr/bin/git, /usr/bin/python3 and the other shims call xcselect to find
    the developer directory and exec the real tool from it, so a profile that
    cannot see that directory ends every such call with "See man xcode-select".
    The directory is versioned on many hosts (/Applications/Xcode_16.4.app) and
    the static list cannot name it; the real path is read from xcode-select and
    kept for the process lifetime.
    """
    global _developer_paths_cache
    with _developer_paths_lock:
        if _developer_paths_cache is not None:
            return _developer_paths_cache
        found: list[str] = []
        if sys.platform == "darwin" and os.path.exists("/usr/bin/xcode-select"):
            try:
                result = subprocess.run(
                    ["/usr/bin/xcode-select", "-p"],
                    capture_output = True,
                    text = True,
                    timeout = 10,
                    check = False,
                )
                candidate = result.stdout.strip()
                if result.returncode == 0 and candidate and os.path.isdir(candidate):
                    found.append(os.path.realpath(candidate))
                    if candidate not in found:
                        found.append(candidate)
                    # xcselect validates the developer directory against the
                    # enclosing app bundle (Info.plist, version.plist), so a
                    # Contents/Developer inside an .app needs the bundle itself.
                    for spelling in list(found):
                        marker = ".app/Contents/Developer"
                        if spelling.endswith(marker):
                            bundle = spelling[: -len(marker) + len(".app")]
                            if os.path.isdir(bundle) and bundle not in found:
                                found.append(bundle)
            except (OSError, subprocess.SubprocessError):
                pass
        _developer_paths_cache = tuple(found)
        return _developer_paths_cache


# Files git and other tools probe on every run and may legitimately be absent.
# The path filter above keeps only paths that exist, and under (deny default)
# an absent file yields EPERM instead of ENOENT, which git reports as
# "unable to access '/etc/gitconfig'" and aborts on. These literals are allowed
# whether or not they exist.
_MACOS_OPTIONAL_READ_LITERALS = (
    "/etc/gitconfig",
    "/private/etc/gitconfig",
    "/etc/gitattributes",
    "/private/etc/gitattributes",
    # xcrun's license check: the Xcode shims refuse ("You have not agreed to the
    # Xcode license agreements") when they cannot read the system-wide record.
    "/Library/Preferences/com.apple.dt.Xcode.plist",
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
    # /etc, /tmp and /var are symlinks into /private; tools spell paths through
    # the symlink (git reads /etc/gitconfig), and resolving that spelling needs
    # the symlink itself and its ancestors in the metadata rules, otherwise the
    # access fails with EPERM before the canonical rule is ever consulted.
    for spelling in tuple(selected):
        for prefix in ("/private/var", "/private/etc", "/private/tmp"):
            if spelling == prefix or spelling.startswith(prefix + "/"):
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
    *,
    workdir: str,
    private_tmp: str,
    runtime_paths: tuple[str, ...],
    proxy_port: int | None = None,
) -> str:
    readable_paths = (
        *_MACOS_READ_ROOTS,
        *_macos_developer_paths(),
        *((*_MACOS_TLS_TRUST_PATHS, *tls_trust_paths()) if proxy_port else ()),
        *_MACOS_DEVICES,
        *runtime_paths,
        workdir,
        private_tmp,
    )
    read_filters = [
        '(literal "/")',
        *_sbpl_path_filters(readable_paths),
    ]
    metadata_filters = _sbpl_ancestor_filters(readable_paths)
    # The optional literals may not exist, so their ancestors are added by name.
    for literal in _MACOS_OPTIONAL_READ_LITERALS:
        current = os.path.dirname(literal)
        while current and current != os.path.sep:
            encoded = f"(literal {json.dumps(current)})"
            if encoded not in metadata_filters:
                metadata_filters.append(encoded)
            current = os.path.dirname(current)
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
        "(allow file-read* file-test-existence "
        + " ".join(f"(literal {json.dumps(path)})" for path in _MACOS_OPTIONAL_READ_LITERALS)
        + ")",
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
        *(
            # Codex and Zed use the same rule for their loopback proxies; the
            # profile stays (deny default) for every other remote endpoint.
            [f'(allow network-outbound (remote ip "localhost:{int(proxy_port)}"))']
            if proxy_port
            else []
        ),
        "(allow sysctl-read " + " ".join(sysctl_filters) + ")",
        '(allow iokit-open (iokit-registry-entry-class "RootDomainUserClient"))',
        "(allow mach-lookup",
        '  (global-name "com.apple.system.opendirectoryd.libinfo")',
        *(
            [f"  (global-name {json.dumps(name)})" for name in _MACOS_TLS_MACH_SERVICES]
            if proxy_port
            else []
        ),
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
    open(os.path.join(os.path.dirname(wd), 'outside-write'), 'w', encoding='utf-8').close()
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
                encoding = "utf-8",
                errors = "replace",
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
    ipv6_address: tuple[str, int, int, int] | None,
    udp_address: tuple[str, int],
    host_namespaces: dict[str, str],
    inherited_fds: tuple[int, ...],
) -> str:
    ip_endpoints: list[tuple[int, tuple]] = [(int(socket.AF_INET), ipv4_address)]
    if ipv6_address is not None:
        ip_endpoints.append((int(socket.AF_INET6), ipv6_address))
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
    open('/unsloth-host-escape', 'w', encoding='utf-8').close()
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
result = libc.unshare({_CLONE_NEWUSER})
assert result == -1, 'a nested user namespace could be created inside the sandbox'
for family, address in {ip_endpoints!r}:
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
    limitations: list[str] = []
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
            ipv6_address: tuple[str, int, int, int] | None = None
            try:
                host_ipv6_socket = socket.socket(socket.AF_INET6)
                host_ipv6_socket.bind(("::1", 0))
                host_ipv6_socket.listen(1)
                ipv6_address = host_ipv6_socket.getsockname()
            except OSError:
                # IPv6 disabled at the kernel or no ::1: nothing to isolate from,
                # so the IPv6 leg is skipped and disclosed rather than failing the
                # whole qualification.
                if host_ipv6_socket is not None:
                    host_ipv6_socket.close()
                    host_ipv6_socket = None
                limitations.append(_LIMITATION_IPV6_UNAVAILABLE)
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
                if os.path.exists(f"/proc/self/ns/{name}")
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
    return SandboxCapability(
        backend.identity,
        True,
        "restrictive live probe passed",
        limitations = tuple(limitations),
    )


_LINUX_BACKEND = LinuxBubblewrapBackend()
_MACOS_BACKEND = MacOSSeatbeltBackend()
_WINDOWS_BACKEND: SandboxBackend | None = None
_WINDOWS_LIMITED_BACKEND: Any = None
_LIMITED_BACKEND = "process-guard"
_LIMITED_PROFILE_ID = "limited-software-safeguards-v1"
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


def _limited_isolation_backend() -> Any:
    """The launcher that hardens Limited mode beyond the process guard, if this host has one.

    Only Windows has one: a write-restricted token (windows_restricted_token).
    Linux and macOS Limited launches keep the process guard alone, and the
    Windows launcher is used only after its own live probe passed.
    """
    global _WINDOWS_LIMITED_BACKEND
    if sys.platform != "win32":
        return None
    if _WINDOWS_LIMITED_BACKEND is None:
        try:
            from .windows_restricted_token import WindowsRestrictedTokenBackend
        except Exception:  # noqa: BLE001 - Limited mode keeps working without it
            logger.warning("Windows restricted-token launcher unavailable", exc_info = True)
            return None
        _WINDOWS_LIMITED_BACKEND = WindowsRestrictedTokenBackend()
    return _WINDOWS_LIMITED_BACKEND


def _with_limited_capability(capability: SandboxCapability, *, force: bool) -> SandboxCapability:
    """Attach what Limited mode runs under on this host to an OS capability snapshot."""
    limited = _limited_isolation_backend()
    if limited is None:
        return capability
    try:
        probe = limited.probe(force = force)
    except Exception as exc:  # noqa: BLE001 - never blocks the OS capability
        logger.warning("Limited launcher probe failed", exc_info = True)
        return replace(capability, limited_reason = f"{type(exc).__name__}: {exc}")
    if not probe.available:
        return replace(capability, limited_reason = probe.reason)
    return replace(
        capability,
        limited_backend = limited.identity,
        limited_profile_id = probe.profile_id,
        limited_limitations = tuple(probe.limitations),
        limited_reason = probe.reason,
    )


def _capability_with_identity(
    capability: SandboxCapability,
    *,
    environment: str,
    fingerprint: str,
    network_allowlist_supported: bool = False,
) -> SandboxCapability:
    available = capability.qualified if capability.available is None else capability.available
    protection_state = capability.protection_state if available else "unavailable"
    if available and protection_state == "unavailable":
        protection_state = "protected" if environment == "native_linux" else "preview"
    profile_id = capability.profile_id if available else "none"
    # The generation binds Limited grants to the security facts that produced
    # them. Free-text reasons are deliberately not part of it: a probe that fails
    # with a different temp path or stderr tail must not revoke every grant.
    generation_payload = "\0".join(
        (
            "generation-v2",
            fingerprint,
            capability.backend,
            str(available),
            str(capability.qualified),
            protection_state,
            profile_id,
            *capability.limitations,
        )
    ).encode()
    generation = hashlib.sha256(generation_payload).hexdigest()
    default_remediation = SandboxCapability.__dataclass_fields__["remediation"].default
    if available:
        remediation = "No remediation required."
    elif capability.remediation and capability.remediation != default_remediation:
        remediation = capability.remediation
    else:
        remediation = _GENERIC_REMEDIATION
    network_policies: tuple[str, ...] = ("deny",)
    network_allowlist: tuple[str, ...] = ()
    limitations = capability.limitations
    if available and network_allowlist_supported:
        hosts = _network_allowlist_hosts()
        if hosts:
            network_policies = ("deny", "allowlist")
            network_allowlist = hosts
        else:
            # A broken or empty UNSLOTH_STUDIO_TOOL_NETWORK_ALLOWLIST must not be
            # advertised as a working allowlist: the launch would refuse every tool
            # call while the toggle looked healthy. Display-only limitation, added
            # after probe_generation was computed so Limited grants do not rotate.
            if _LIMITATION_NETWORK_ALLOWLIST_INVALID not in limitations:
                limitations = (*limitations, _LIMITATION_NETWORK_ALLOWLIST_INVALID)
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
        limitations = limitations,
        network_policies = network_policies,
        network_allowlist = network_allowlist,
    )


def capability_snapshot(*, force: bool = False) -> SandboxCapability:
    backend = _platform_backend()
    environment = _environment_class()
    fingerprint = _environment_fingerprint(backend)
    if backend is None:
        return _with_limited_capability(
            _capability_with_identity(
                SandboxCapability(
                    f"unsupported-{sys.platform}",
                    False,
                    f"OS sandboxing is unsupported on {sys.platform}",
                ),
                environment = environment,
                fingerprint = fingerprint,
            ),
            force = force,
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
            # The scan memo is keyed by mount topology and expires on its own; a
            # forced capability refresh (every pre-send check) must not turn
            # every launch back into a full scan of /usr.
            _capability_cache.clear()
        result = _capability_with_identity(
            backend.probe(),
            environment = _environment_class(),
            fingerprint = current_fingerprint,
            network_allowlist_supported = bool(
                getattr(backend, "supports_network_allowlist", False)
            ),
        )
        result = _with_limited_capability(result, force = force)
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


# pidfd_open(2) and pidfd_send_signal(2) share these numbers on every Linux
# architecture (they postdate the unified syscall table). Used only when the
# interpreter was built without os.pidfd_open, which is the case for the
# python-build-standalone 3.10 to 3.12 builds that uv installs.
_NR_PIDFD_SEND_SIGNAL = 424
_NR_PIDFD_OPEN = 434
_pidfd_support: "bool | None" = None


def _pidfd_open(pid: int) -> int:
    """A file descriptor pinned to exactly this process (raises OSError)."""
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid, 0)
    libc = ctypes.CDLL(None, use_errno = True)
    fd = libc.syscall(_NR_PIDFD_OPEN, ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        errno_value = ctypes.get_errno()
        raise OSError(errno_value, os.strerror(errno_value))
    return int(fd)


def _pidfd_send_signal(pidfd: int, signum: int) -> None:
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(pidfd, signum)
        return
    libc = ctypes.CDLL(None, use_errno = True)
    result = libc.syscall(
        _NR_PIDFD_SEND_SIGNAL, ctypes.c_int(pidfd), ctypes.c_int(signum), None, ctypes.c_uint(0)
    )
    if result < 0:
        errno_value = ctypes.get_errno()
        raise OSError(errno_value, os.strerror(errno_value))


def descendant_sweep_supported() -> bool:
    """Whether Limited launches can reap detached descendants after the leader exits.

    The sweep (tools._sweep_marked_descendants) matches processes by the per-call
    marker in ``/proc/<pid>/environ`` and signals them through a pidfd taken
    before the match, so a pid recycled between the match and the signal is
    never hit. Without ``/proc`` or pidfds (macOS, Linux before 5.3) there is no
    safe sweep and the Limited record discloses
    ``detached_descendant_cleanup_unverified`` instead.
    """
    global _pidfd_support
    if sys.platform != "linux" or not os.path.isdir("/proc"):
        return False
    if _pidfd_support is None:
        try:
            os.close(_pidfd_open(os.getpid()))
            _pidfd_support = True
        except (OSError, AttributeError, TypeError):
            _pidfd_support = False
    return _pidfd_support


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
    if spec.network_policy not in NETWORK_POLICIES:
        raise SandboxUnavailableError(f"unknown network policy: {spec.network_policy!r}")
    canonical = replace(spec, workdir = os.path.realpath(spec.workdir))
    backend = _platform_backend()

    if canonical.requested_mode == "full":
        # Full access does not depend on an OS sandbox. In particular, do not
        # run a live sandbox probe here: the existing escape hatch must not
        # create helper processes or become unavailable because a backend is
        # missing. Its record still carries a deterministic launch identity.
        environment = _environment_class(run_detector = False)
        fingerprint = _environment_fingerprint(backend, run_detector = False)
        identity = _capability_with_identity(
            SandboxCapability(
                backend.identity if backend is not None else f"unsupported-{sys.platform}",
                False,
                "OS sandbox capability was not probed for Full access",
            ),
            environment = environment,
            fingerprint = fingerprint,
        )
        record = ToolExecutionRecord(
            requested_mode = "full",
            effective_mode = "full",
            environment = environment,
            backend = "none",
            profile_id = "full-access-v1",
            probe_generation = identity.probe_generation,
            os_isolation = False,
            retained_safeguards = _FULL_SAFEGUARDS,
            network_policy = "unrestricted",
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

    capability = capability_snapshot()

    if canonical.requested_mode == "limited":
        if canonical.network_policy != "deny":
            # Limited has no OS boundary, so a proxy would be advisory only; the
            # request is refused rather than recorded as enforced.
            raise SandboxUnavailableError(
                "the network allowlist requires OS isolation; Limited mode cannot enforce it"
            )
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
        limitations: tuple[str, ...] = ()
        limited = _limited_isolation_backend()
        if limited is not None and limited.probe().available:
            # Windows: the write-restricted token launcher. It fails closed per
            # launch (workdir too large to scan, ACL grant refused, token API
            # error); that one call then runs under the process guard alone and
            # its record says so, instead of Limited mode disappearing.
            try:
                prepared = limited.prepare(canonical)
            except (SandboxUnavailableError, OSError) as exc:
                logger.warning("Limited launcher declined this launch: %s", exc)
                limitations = ("restricted_token_unavailable",)
            else:
                prepared.execution_record = ToolExecutionRecord(
                    requested_mode = "limited",
                    effective_mode = "limited",
                    environment = capability.environment,
                    backend = limited.identity,
                    profile_id = limited.profile_id,
                    probe_generation = capability.probe_generation,
                    os_isolation = False,
                    retained_safeguards = (
                        *_LIMITED_SAFEGUARDS,
                        "write_restricted_token",
                        "job_object",
                    ),
                    limitations = tuple(limited.limitations),
                    # The token fences writes only; the host network is reachable.
                    network_policy = "unrestricted",
                )
                return prepared
        if not descendant_sweep_supported():
            # Linux sweeps /proc for the per-call run marker after the leader
            # exits (tools._sweep_marked_descendants); macOS has no /proc, so a
            # setsid grandchild there can outlive the call and the record says so.
            limitations = (*limitations, "detached_descendant_cleanup_unverified")
        record = ToolExecutionRecord(
            requested_mode = "limited",
            effective_mode = "limited",
            environment = capability.environment,
            backend = _LIMITED_BACKEND,
            profile_id = _LIMITED_PROFILE_ID,
            probe_generation = capability.probe_generation,
            os_isolation = False,
            retained_safeguards = _LIMITED_SAFEGUARDS,
            limitations = limitations,
            network_policy = "unrestricted",
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
    if canonical.network_policy == "allowlist" and "allowlist" not in capability.network_policies:
        raise SandboxUnavailableError(
            f"the network allowlist is not available with {capability.backend}; "
            "run with network access off or use Full access"
        )
    allowlist_hosts = (
        _network_allowlist_for_launch().hosts if canonical.network_policy == "allowlist" else ()
    )
    try:
        # A backend with more than one profile (Windows LPAC or its AppContainer
        # fallback) prepares the profile the capability was recorded with, so a
        # concurrent re-probe cannot swap profiles between the check and the launch.
        prepare_for_profile = getattr(backend, "prepare_for_profile", None)
        if callable(prepare_for_profile):
            prepared = prepare_for_profile(canonical, capability.profile_id)
        else:
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
        network_policy = canonical.network_policy,
        network_allowlist = allowlist_hosts,
    )
    return prepared
