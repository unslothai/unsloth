# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
OS-level sandbox wrapper for tool execution.
"""

from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import errno
import importlib.machinery
import os
import shutil
import site
import stat
import subprocess
import sys
import tempfile
import threading

from loggers import get_logger

logger = get_logger(__name__)

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"
_BWRAP_PROBE_CANDIDATES = ("/usr/bin/true", "/bin/true", "/run/current-system/sw/bin/true")
_SANDBOX_SITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_site")
_NIX_STORE = "/nix/store"

_LINUX_ACCELERATOR_DEVICE_PATHS = (
    "/dev/dxg",
    "/dev/kfd",
    "/dev/accel",
    "/dev/nvidia-caps",
    "/dev/nvidiactl",
    "/dev/nvidia-modeset",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
)
_LINUX_DRM_DIR = "/dev/dri"
_LINUX_ROCM_OPT_ROOT = "/opt"
_LINUX_ROCM_ROOTS = ("/opt/rocm",)
_LINUX_ROCM_ROOT_ENV_VARS = ("ROCM_PATH", "HIP_PATH")
_LINUX_ROCM_RUNTIME_LIBRARY_PREFIXES = (
    "libamdhip64.so",
    "libhsa-runtime64.so",
    "librocdxg.so",
)
_LINUX_ONEAPI_ROOTS = ("/opt/intel/oneapi",)
_LINUX_ACCELERATOR_SYSFS_CLASS_PATHS = (
    "/sys/class/drm",
    "/sys/class/kfd",
)
_LINUX_MOUNTINFO = "/proc/self/mountinfo"
_LINUX_FIND_BIN = "/usr/bin/find"
_WORKDIR_SCAN_MAX_ENTRIES = 100_000
_WORKDIR_SCAN_MAX_DEPTH = 128
_PTH_NAMESPACE_SCAN_MAX_ENTRIES = 10_000
_PTH_NAMESPACE_SCAN_MAX_DEPTH = 32

_MACOS_DEVELOPER_PREFIXES = (
    "/Library/Developer/CommandLineTools",
    "/Applications/Xcode.app/Contents/Developer",
)


class SandboxProfilePathError(ValueError):
    """A legal host path cannot be safely represented in a sandbox profile."""


class UnsafeSandboxWorkdirError(RuntimeError):
    """The writable workdir would expose an inode outside its path boundary."""


class SandboxArgv(list[str]):
    """Sandbox command plus the narrow file descriptors it must inherit."""

    def __init__(
        self,
        values: list[str],
        pass_fds: tuple[int, ...] = (),
    ):
        super().__init__(values)
        self.pass_fds = pass_fds

    def close_pass_fds(self) -> None:
        for fd in self.pass_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.pass_fds = ()

    def __del__(self):
        self.close_pass_fds()


def sandbox_argv_pass_fds(argv: list[str]) -> tuple[int, ...]:
    """Descriptors a :class:`SandboxArgv` must pass through ``Popen``."""
    return getattr(argv, "pass_fds", ())


def close_sandbox_argv_fds(argv: list[str]) -> None:
    """Release parent copies of descriptors after the sandbox process starts."""
    close = getattr(argv, "close_pass_fds", None)
    if close is not None:
        close()


class _ScmpArgCmp(ctypes.Structure):
    _fields_ = (
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    )


_sandbox_available_cache: bool | None = None
# Absolute path to ``bwrap``, resolved once at probe time so the runtime
# sandbox argv doesn't depend on the child's PATH (``_build_safe_env``
# strips PATH down to a fixed allow-list that won't cover Nix-style or
# custom-prefix installs).
_linux_bwrap_path: str | None = None
_linux_bwrap_keep_groups = False
# Guards probe + cache so concurrent first-callers see a consistent
# (cache, bwrap_path) snapshot rather than racing on partial writes.
_sandbox_probe_lock = threading.Lock()
_sandbox_identity_lock = threading.Lock()
_sandbox_identity_paths: tuple[str, str] | None = None

# Extra macOS exec/read prefixes Studio actually puts on PATH via
# _build_safe_env (Homebrew on Intel + Apple Silicon). Without these
# in the Seatbelt profile, a tool like `bash_exec("uv --version")` or
# even `bash` itself resolving to /usr/local/bin/bash fails with
# Operation not permitted on common dev macs.
_MACOS_EXTRA_EXEC_PREFIXES = (
    "/usr/local/bin",
    "/usr/local/lib",
    "/usr/local/sbin",
    "/usr/local/opt",
    "/usr/local/Cellar",
    "/opt/homebrew/bin",
    "/opt/homebrew/lib",
    "/opt/homebrew/sbin",
    "/opt/homebrew/opt",
    "/opt/homebrew/Cellar",
)

_MACOS_CA_READ_PATHS = (
    "/private/etc/ssl/cert.pem",
    "/private/etc/ssl/certs",
    "/private/etc/ssl/openssl.cnf",
    "/private/etc/ca-certificates",
)

_LINUX_CA_READ_PATHS = (
    "/etc/ssl/cert.pem",
    "/etc/ssl/certs",
    "/etc/ssl/openssl.cnf",
    "/etc/ca-certificates",
    "/etc/pki/ca-trust",
    "/etc/pki/tls/cert.pem",
    "/etc/pki/tls/certs",
    "/etc/pki/tls/openssl.cnf",
)


class _ProbeResult:
    """Three-way probe result: True / False / transient timeout.

    The caller treats a transient timeout as "do not cache; let the
    next caller re-probe". A definite True or False (binary missing,
    bwrap setuid helper denied, kernel userns refusal) is cacheable
    for the lifetime of the process.
    """

    __slots__ = ("ok", "transient")

    def __init__(
        self,
        ok: bool,
        transient: bool = False,
    ):
        self.ok = ok
        self.transient = transient


def _probe(argv: list[str], label: str) -> _ProbeResult:
    """Run *argv*; return _ProbeResult(ok=..., transient=...).

    ``transient=True`` means the answer might change next time (e.g.
    timed out under IO load); the caller should NOT cache the False.
    """
    try:
        run_kwargs = dict(
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            timeout = 5,
        )
        pass_fds = sandbox_argv_pass_fds(argv)
        if pass_fds:
            run_kwargs["pass_fds"] = pass_fds
        proc = subprocess.run(argv, **run_kwargs)
    except subprocess.TimeoutExpired as e:
        # Slow runner / loaded box / cold filesystem. Don't pin the
        # answer to False forever; let the next caller re-probe.
        logger.warning("%s probe timed out (%s); will retry on next tool call", label, e)
        return _ProbeResult(ok = False, transient = True)
    except OSError as e:
        if e.errno == errno.EAGAIN:
            logger.warning(
                "%s probe could not start temporarily (%s); will retry on next tool call",
                label,
                e,
            )
            return _ProbeResult(ok = False, transient = True)
        logger.warning("%s probe failed (%s); tool execution will run unsandboxed", label, e)
        return _ProbeResult(ok = False)
    if proc.returncode != 0:
        stderr_tail = proc.stderr.decode("utf-8", errors = "replace").strip()[-200:]
        logger.warning(
            "%s present but probe returned %s; tool execution will run unsandboxed. stderr: %s",
            label,
            proc.returncode,
            stderr_tail,
        )
        return _ProbeResult(ok = False)
    return _ProbeResult(ok = True)


def _macos_probe() -> _ProbeResult:
    if not os.path.exists(_SANDBOX_EXEC):
        logger.warning("macOS sandbox unavailable (sandbox-exec missing)")
        return _ProbeResult(ok = False)
    return _probe(
        [_SANDBOX_EXEC, "-p", "(version 1)(allow default)", "/usr/bin/true"],
        "macOS sandbox-exec",
    )


def _linux_accelerator_device_nodes() -> list[str]:
    """Existing accelerator device nodes whose DAC permissions matter."""
    nodes: list[str] = []
    for candidate in _LINUX_ACCELERATOR_DEVICE_PATHS:
        try:
            candidate_stat = os.lstat(candidate)
        except OSError:
            continue
        if stat.S_ISCHR(candidate_stat.st_mode) or stat.S_ISBLK(candidate_stat.st_mode):
            nodes.append(candidate)
            continue
        if not stat.S_ISDIR(candidate_stat.st_mode):
            continue
        try:
            with os.scandir(candidate) as entries:
                for entry in entries:
                    try:
                        entry_stat = os.lstat(entry.path)
                    except OSError:
                        continue
                    if stat.S_ISCHR(entry_stat.st_mode) or stat.S_ISBLK(entry_stat.st_mode):
                        nodes.append(entry.path)
        except OSError:
            continue
    nodes.extend(_linux_drm_render_device_paths())
    try:
        with os.scandir("/dev") as entries:
            for entry in entries:
                if not (
                    entry.name.startswith("nvidia") and entry.name.removeprefix("nvidia").isdigit()
                ):
                    continue
                try:
                    entry_stat = os.lstat(entry.path)
                except OSError:
                    continue
                if stat.S_ISCHR(entry_stat.st_mode) or stat.S_ISBLK(entry_stat.st_mode):
                    nodes.append(entry.path)
    except OSError:
        pass
    return list(dict.fromkeys(nodes))


def _linux_drm_render_device_paths() -> list[str]:
    """Existing DRM render nodes, excluding display/control ``card*`` nodes."""
    paths: list[str] = []
    try:
        with os.scandir(_LINUX_DRM_DIR) as entries:
            for entry in entries:
                suffix = entry.name.removeprefix("renderD")
                if not entry.name.startswith("renderD") or not suffix.isdigit():
                    continue
                try:
                    entry_stat = os.lstat(entry.path)
                except OSError:
                    continue
                if stat.S_ISCHR(entry_stat.st_mode) or stat.S_ISBLK(entry_stat.st_mode):
                    paths.append(entry.path)
    except OSError:
        pass
    return paths


def configured_rocm_environment() -> dict[str, str]:
    """Validated ROCm installation roots safe to preserve for the child."""
    configured_roots: dict[str, str] = {}
    for variable in _LINUX_ROCM_ROOT_ENV_VARS:
        configured = os.environ.get(variable)
        if not configured:
            continue
        if not os.path.isabs(configured):
            logger.warning("Ignoring relative %s for sandbox mounts: %s", variable, configured)
            continue
        configured = os.path.normpath(configured)
        if os.path.dirname(configured) == configured:
            logger.warning("Ignoring filesystem-root %s for sandbox mounts", variable)
            continue
        if not os.path.isdir(configured):
            continue
        configured_roots[variable] = configured
    return configured_roots


def _linux_rocm_runtime_bindings() -> list[tuple[str, str]]:
    """Detected ROCm library directories as ``(real source, logical dest)``."""
    roots = [*_LINUX_ROCM_ROOTS, *configured_rocm_environment().values()]
    try:
        with os.scandir(_LINUX_ROCM_OPT_ROOT) as entries:
            roots.extend(
                entry.path
                for entry in entries
                if entry.name.startswith("rocm-") and entry.is_dir(follow_symlinks = True)
            )
    except OSError:
        pass

    bindings: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for root in roots:
        for leaf in ("lib", "lib64"):
            destination = os.path.normpath(os.path.join(root, leaf))
            try:
                with os.scandir(destination) as entries:
                    detected = any(
                        any(
                            entry.name.startswith(prefix)
                            for prefix in _LINUX_ROCM_RUNTIME_LIBRARY_PREFIXES
                        )
                        for entry in entries
                    )
            except OSError:
                continue
            if not detected:
                continue
            binding = (os.path.realpath(destination), destination)
            if binding not in seen:
                seen.add(binding)
                bindings.append(binding)
    return bindings


def _linux_oneapi_runtime_bindings() -> list[tuple[str, str]]:
    """Detected system oneAPI trees as ``(real source, logical dest)``."""
    bindings: list[tuple[str, str]] = []
    for destination in _LINUX_ONEAPI_ROOTS:
        if not os.path.isdir(destination):
            continue
        bindings.append((os.path.realpath(destination), os.path.normpath(destination)))
    return list(dict.fromkeys(bindings))


def _linux_supplementary_group_devices() -> list[str]:
    """GPU nodes whose read/write access depends on a supplementary group."""
    getgroups = getattr(os, "getgroups", None)
    if getgroups is None:
        return []
    try:
        uid = os.getuid()
        primary_gid = os.getgid()
        supplementary = set(getgroups()) - {primary_gid}
    except (AttributeError, OSError):
        return []
    if not supplementary:
        return []

    dependent: list[str] = []
    for path in _linux_accelerator_device_nodes():
        try:
            node_stat = os.stat(path, follow_symlinks = False)
        except OSError:
            continue
        mode = stat.S_IMODE(node_stat.st_mode)
        owner_rw = node_stat.st_uid == uid and mode & 0o600 == 0o600
        group_rw = node_stat.st_gid in supplementary and mode & 0o060 == 0o060
        other_rw = mode & 0o006 == 0o006
        if group_rw and not owner_rw and not other_rw:
            dependent.append(path)
    return dependent


def _bwrap_supports_keep_groups(bwrap: str) -> bool | None:
    """Whether Bubblewrap accepts the optional group-retention flag.

    ``None`` means the capability check failed transiently and must not be
    cached as an old/incompatible Bubblewrap installation.
    """
    try:
        proc = subprocess.run(
            [bwrap, "--help"],
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            timeout = 5,
        )
    except subprocess.TimeoutExpired:
        return None
    except OSError as exc:
        if exc.errno == errno.EAGAIN:
            return None
        return False
    return b"--keep-groups" in proc.stdout


def _linux_executable_path_is_trusted(path: str) -> bool:
    """Return whether *path* and its ancestors resist replacement by this user."""
    if not hasattr(os, "getuid"):
        return False

    current = path
    first = True
    while True:
        try:
            current_stat = os.stat(current, follow_symlinks = False)
        except OSError:
            return False
        mode = stat.S_IMODE(current_stat.st_mode)
        if first:
            if not stat.S_ISREG(current_stat.st_mode) or not os.access(current, os.X_OK):
                return False
            first = False
        elif not stat.S_ISDIR(current_stat.st_mode):
            return False

        # Only a root-owned installation is accepted. Any group/other-writable
        # component permits replacement through a shared directory.
        if current_stat.st_uid != 0:
            return False
        if mode & (stat.S_IWGRP | stat.S_IWOTH):
            return False

        parent = os.path.dirname(current)
        if parent == current:
            return True
        current = parent


def _linux_bwrap_probe_path() -> str | None:
    """Return a fixed, root-controlled no-op executable for the bwrap probe."""
    for candidate in _BWRAP_PROBE_CANDIDATES:
        resolved = os.path.realpath(candidate)
        if _linux_executable_path_is_trusted(resolved):
            return resolved
    return None


def _linux_socket_seccomp_fd() -> int:
    """Export a seccomp filter that blocks host Unix-socket access.

    A network namespace does not isolate pathname or abstract Unix sockets.
    Denying ``socket(AF_UNIX, ...)`` prevents a live writable workdir bind from
    gaining host IPC access if another process creates a socket after the
    bounded preflight scan. ``io_uring_setup`` is denied as well because its
    socket opcode otherwise creates sockets outside the syscall filter.
    """
    library_name = ctypes.util.find_library("seccomp") or "libseccomp.so.2"
    try:
        lib = ctypes.CDLL(library_name, use_errno = True)
    except OSError as exc:
        raise UnsafeSandboxWorkdirError(
            "libseccomp is required to isolate Unix sockets in the Linux sandbox"
        ) from exc

    lib.seccomp_init.argtypes = (ctypes.c_uint32,)
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_release.argtypes = (ctypes.c_void_p,)
    lib.seccomp_rule_add.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    )
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_syscall_resolve_name.argtypes = (ctypes.c_char_p,)
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_export_bpf.argtypes = (ctypes.c_void_p, ctypes.c_int)
    lib.seccomp_export_bpf.restype = ctypes.c_int

    allow = 0x7FFF0000
    errno_action = 0x00050000 | errno.EPERM
    compare_equal = 4
    context = lib.seccomp_init(allow)
    if not context:
        raise UnsafeSandboxWorkdirError("could not initialize the Linux socket seccomp filter")
    fd = -1
    try:
        socket_syscall = lib.seccomp_syscall_resolve_name(b"socket")
        io_uring_syscall = lib.seccomp_syscall_resolve_name(b"io_uring_setup")
        if socket_syscall < 0:
            raise UnsafeSandboxWorkdirError("libseccomp cannot resolve the socket syscall")

        socket_comparison = _ScmpArgCmp(0, compare_equal, 1, 0)  # AF_UNIX
        socket_result = lib.seccomp_rule_add(
            context,
            ctypes.c_uint32(errno_action),
            ctypes.c_int(socket_syscall),
            ctypes.c_uint(1),
            socket_comparison,
        )
        if socket_result != 0:
            raise UnsafeSandboxWorkdirError("could not add the Unix-socket seccomp rule")
        if io_uring_syscall >= 0:
            io_uring_result = lib.seccomp_rule_add(
                context,
                ctypes.c_uint32(errno_action),
                ctypes.c_int(io_uring_syscall),
                ctypes.c_uint(0),
            )
            if io_uring_result != 0:
                raise UnsafeSandboxWorkdirError("could not add the io_uring seccomp rule")

        if not hasattr(os, "memfd_create"):
            raise UnsafeSandboxWorkdirError("memfd_create is required for the Linux seccomp filter")
        fd = os.memfd_create("unsloth-studio-sandbox-seccomp", flags = getattr(os, "MFD_CLOEXEC", 1))
        if lib.seccomp_export_bpf(context, fd) != 0:
            raise UnsafeSandboxWorkdirError("could not export the Linux socket seccomp filter")
        os.lseek(fd, 0, os.SEEK_SET)
        return fd
    except Exception:
        if fd >= 0:
            os.close(fd)
        raise
    finally:
        lib.seccomp_release(context)


def _linux_probe() -> _ProbeResult:
    """Smoke-test that ``bwrap`` can apply a minimal sandbox here.

    Catches the cases where the kernel refuses to create unprivileged
    user namespaces — surfacing at startup instead of first use
    """
    global _linux_bwrap_keep_groups, _linux_bwrap_path
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        logger.warning("bwrap not found on PATH; tool execution will run unsandboxed")
        return _ProbeResult(ok = False)
    bwrap = os.path.realpath(os.path.abspath(bwrap))
    if not _linux_executable_path_is_trusted(bwrap):
        logger.warning(
            "Ignoring untrusted or user-replaceable bwrap executable %s; "
            "tool execution will run unsandboxed",
            bwrap,
        )
        return _ProbeResult(ok = False)
    keep_groups = _bwrap_supports_keep_groups(bwrap)
    if keep_groups is None:
        logger.warning(
            "Bubblewrap capability probe was temporarily unavailable; will retry on next tool call"
        )
        return _ProbeResult(ok = False, transient = True)
    group_devices = _linux_supplementary_group_devices()
    if group_devices and not keep_groups:
        logger.warning(
            "Bubblewrap cannot retain the supplementary group required by accelerator "
            "devices (%s); tool execution will run unsandboxed",
            ", ".join(group_devices[:3]),
        )
        return _ProbeResult(ok = False)
    probe_path = _linux_bwrap_probe_path()
    if probe_path is None:
        logger.warning(
            "No trusted bwrap probe executable found; tool execution will run unsandboxed"
        )
        return _ProbeResult(ok = False)
    try:
        seccomp_fd = _linux_socket_seccomp_fd()
    except UnsafeSandboxWorkdirError as exc:
        logger.warning("Linux sandbox socket isolation is unavailable: %s", exc)
        return _ProbeResult(ok = False)
    probe_argv = SandboxArgv(
        [
            bwrap,
            *(["--keep-groups"] if keep_groups else []),
            "--ro-bind",
            "/",
            "/",
            "--unshare-all",
            "--die-with-parent",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--seccomp",
            str(seccomp_fd),
            probe_path,
        ],
        (seccomp_fd,),
    )
    try:
        result = _probe(probe_argv, "Linux bwrap")
    finally:
        close_sandbox_argv_fds(probe_argv)
    if result.ok:
        _linux_bwrap_path = bwrap
        _linux_bwrap_keep_groups = keep_groups
    return result


def start_sandbox_probe() -> None:
    """Warm the optional sandbox probe without delaying or failing startup."""

    def _warm_sandbox_probe():
        try:
            sandbox_available()
        except Exception as exc:
            logger.debug("sandbox availability probe failed at startup: %s", exc)

    try:
        threading.Thread(target = _warm_sandbox_probe, daemon = True).start()
    except Exception as exc:
        logger.debug("sandbox availability probe could not start: %s", exc)


def sandbox_available() -> bool:
    """True iff the platform's sandbox can be applied in this process context.

    Existence of the binary alone is not enough: a nested-sandboxed
    parent may have ``sandbox-exec`` / ``bwrap`` present but be unable
    to apply additional policies. Confirm by spawning a no-op sandboxed
    ``/usr/bin/true`` once at first call and caching the result.

    Thread-safe: the run.py background probe and concurrent tool calls
    can hit this entry point at the same time. The lock prevents a
    slow-failing probe from overwriting a fast-succeeding probe (or
    vice versa) and ensures _linux_bwrap_path is set before any caller
    observes _sandbox_available_cache=True.

    A transient probe TIMEOUT (slow runner, cold filesystem, loaded
    host) is NOT cached: the next caller re-probes. Without this, a
    one-off timeout would pin the answer to "unavailable" for the
    entire Studio process lifetime even after the underlying load
    cleared.
    """
    global _sandbox_available_cache
    if _sandbox_available_cache is not None:
        return _sandbox_available_cache

    with _sandbox_probe_lock:
        if _sandbox_available_cache is not None:
            return _sandbox_available_cache

        if sys.platform == "darwin":
            result = _macos_probe()
            label = "macOS Seatbelt"
        elif sys.platform == "linux":
            result = _linux_probe()
            label = "Linux bubblewrap"
        else:
            result = _ProbeResult(ok = False)
            label = "no sandbox primitive for this platform"

        ok = result.ok
        if not result.transient:
            _sandbox_available_cache = ok
        if ok:
            logger.info("%s sandbox available; tool execution sandboxed", label)
        elif sys.platform not in ("darwin", "linux"):
            logger.warning("%s; tool execution will run unsandboxed", label)
        return ok


def _safe_subpath(p: str) -> str:
    """Reject paths that cannot be safely embedded in a Seatbelt literal.

    Seatbelt string literals use ``"..."`` with ``\\`` escapes; a path
    containing ``"``, ``\\``, a newline, or a NUL byte could close the
    string and inject Scheme into the profile. macOS paths in practice
    contain none of these, so rejecting them is safer than escaping.
    """
    if any(c in p for c in ('"', "\\", "\n", "\r", "\x00")):
        raise SandboxProfilePathError(f"path unsafe for Seatbelt profile: {p!r}")
    return p


def _assert_no_external_hardlinks(
    workdir: str, _read_only_paths: tuple[str, ...] | list[str] = ()
) -> None:
    """Boundedly verify that *workdir* contains no boundary-crossing nodes.

    A path-scoped writable bind/profile still exposes every inode reachable by
    an in-tree hard link. Count all in-tree aliases and compare them with the
    filesystem link count; internal-only hard links remain valid, while a
    pre-existing link to a host file fails closed before sandbox launch.
    Runtime paths nested below the workdir stay writable so editable installs
    can compile and update their own sources. They receive the same complete
    hard-link and special-node checks as the rest of the writable tree.
    """
    wd = os.path.realpath(workdir)
    try:
        root_dev = os.lstat(wd).st_dev
    except OSError as exc:
        raise UnsafeSandboxWorkdirError(
            f"cannot inspect sandbox workdir safely: {wd!r}: {exc}"
        ) from exc
    inode_counts: dict[tuple[int, int], int] = {}
    inode_links: dict[tuple[int, int], int] = {}
    representatives: dict[tuple[int, int], str] = {}
    stack = [(wd, 0)]
    inspected = 0
    nested_mount_points = _linux_nested_mount_points(wd)

    while stack:
        directory, depth = stack.pop()
        if depth > _WORKDIR_SCAN_MAX_DEPTH:
            raise UnsafeSandboxWorkdirError(
                "sandbox workdir exceeds the safe directory-depth limit "
                f"({_WORKDIR_SCAN_MAX_DEPTH})"
            )
        try:
            entries = os.scandir(directory)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise UnsafeSandboxWorkdirError(
                f"cannot inspect sandbox workdir safely: {directory!r}: {exc}"
            ) from exc

        with entries:
            for entry in entries:
                inspected += 1
                if inspected > _WORKDIR_SCAN_MAX_ENTRIES:
                    raise UnsafeSandboxWorkdirError(
                        "sandbox workdir exceeds the safe entry limit "
                        f"({_WORKDIR_SCAN_MAX_ENTRIES})"
                    )
                try:
                    # DirEntry.stat reports st_nlink=0 on some Windows/Python
                    # combinations; os.lstat returns the filesystem link count.
                    entry_stat = os.lstat(entry.path)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise UnsafeSandboxWorkdirError(
                        f"cannot inspect sandbox workdir safely: {entry.path!r}: {exc}"
                    ) from exc
                if entry_stat.st_dev != root_dev:
                    raise UnsafeSandboxWorkdirError(
                        f"sandbox workdir crosses a filesystem boundary: {entry.path!r}"
                    )
                if os.path.normpath(entry.path) in nested_mount_points:
                    raise UnsafeSandboxWorkdirError(
                        f"sandbox workdir contains a nested mount point: {entry.path!r}"
                    )
                if stat.S_ISDIR(entry_stat.st_mode):
                    stack.append((entry.path, depth + 1))
                    continue
                if stat.S_ISLNK(entry_stat.st_mode):
                    continue
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise UnsafeSandboxWorkdirError(
                        f"sandbox workdir contains a special filesystem node: {entry.path!r}"
                    )
                if entry_stat.st_nlink <= 1:
                    continue
                key = (entry_stat.st_dev, entry_stat.st_ino)
                inode_counts[key] = inode_counts.get(key, 0) + 1
                inode_links[key] = max(inode_links.get(key, 0), entry_stat.st_nlink)
                representatives.setdefault(key, entry.path)

    external = [
        representatives[key] for key, count in inode_counts.items() if count < inode_links[key]
    ]
    if external:
        sample = ", ".join(repr(path) for path in external[:3])
        suffix = " ..." if len(external) > 3 else ""
        raise UnsafeSandboxWorkdirError(
            "sandbox workdir contains regular files hard-linked outside its boundary: "
            f"{sample}{suffix}"
        )


def _assert_external_read_paths_have_no_special_nodes(
    workdir: str, read_paths: tuple[str, ...] | list[str]
) -> None:
    """Reject host IPC/device nodes in external trees before read-only binds."""
    wd = os.path.realpath(workdir)
    trusted_roots = ("/usr", "/bin", "/sbin", "/lib", "/lib64", _NIX_STORE)
    scan_roots: list[str] = []
    for path in read_paths:
        root = os.path.realpath(path)
        if _path_is_within(root, wd) or any(
            _path_is_within(root, trusted) for trusted in trusted_roots
        ):
            continue
        if any(_path_is_within(root, existing) for existing in scan_roots):
            continue
        scan_roots = [existing for existing in scan_roots if not _path_is_within(existing, root)]
        scan_roots.append(root)

    inspected = 0
    stack: list[tuple[str, int, int]] = []
    directory_roots: list[tuple[str, int]] = []
    for root in scan_roots:
        try:
            root_stat = os.lstat(root)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise UnsafeSandboxWorkdirError(
                f"cannot inspect external Python read path safely: {root!r}: {exc}"
            ) from exc
        if stat.S_ISREG(root_stat.st_mode):
            continue
        if not stat.S_ISDIR(root_stat.st_mode):
            raise UnsafeSandboxWorkdirError(
                f"external Python read path is a special filesystem node: {root!r}"
            )
        directory_roots.append((root, root_stat.st_dev))

    if sys.platform == "linux" and _linux_executable_path_is_trusted(_LINUX_FIND_BIN):
        native_roots: list[str] = []
        for root, _root_dev in directory_roots:
            nested_mounts = _linux_nested_mount_points(root)
            if nested_mounts:
                sample = sorted(nested_mounts)[0]
                raise UnsafeSandboxWorkdirError(
                    f"external Python read path contains a nested mount point: {sample!r}"
                )
            native_roots.append(root)
        if not native_roots:
            return
        try:
            find_result = subprocess.run(
                [
                    _LINUX_FIND_BIN,
                    "-P",
                    *native_roots,
                    "-xdev",
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
                ],
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                timeout = 5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise UnsafeSandboxWorkdirError(
                f"cannot inspect external Python read paths safely: {exc}"
            ) from exc
        if find_result.returncode != 0:
            detail = find_result.stderr.decode(errors = "replace").strip()[:200]
            raise UnsafeSandboxWorkdirError(
                f"cannot inspect external Python read paths safely: {detail}"
            )
        if find_result.stdout:
            special = find_result.stdout.decode(errors = "replace").strip()[:200]
            raise UnsafeSandboxWorkdirError(
                f"external Python read path contains a special filesystem node: {special!r}"
            )
        return

    stack.extend((root, 0, root_dev) for root, root_dev in directory_roots)

    while stack:
        directory, depth, root_dev = stack.pop()
        if depth > _WORKDIR_SCAN_MAX_DEPTH:
            raise UnsafeSandboxWorkdirError(
                "external Python read path exceeds the safe directory-depth limit "
                f"({_WORKDIR_SCAN_MAX_DEPTH})"
            )
        try:
            entries = os.scandir(directory)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise UnsafeSandboxWorkdirError(
                f"cannot inspect external Python read path safely: {directory!r}: {exc}"
            ) from exc
        with entries:
            for entry in entries:
                inspected += 1
                if inspected > _WORKDIR_SCAN_MAX_ENTRIES:
                    raise UnsafeSandboxWorkdirError(
                        "external Python read paths exceed the safe entry limit "
                        f"({_WORKDIR_SCAN_MAX_ENTRIES})"
                    )
                try:
                    if entry.is_symlink():
                        continue
                    # DirEntry normally answers regular-file type from the
                    # directory record without an lstat syscall. Runtime trees
                    # are file-heavy, so keep the secure scan cheap on the hot
                    # sandbox-launch path.
                    if entry.is_file(follow_symlinks = False):
                        continue
                    entry_stat = entry.stat(follow_symlinks = False)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise UnsafeSandboxWorkdirError(
                        f"cannot inspect external Python read path safely: {entry.path!r}: {exc}"
                    ) from exc
                if entry_stat.st_dev != root_dev:
                    raise UnsafeSandboxWorkdirError(
                        f"external Python read path crosses a filesystem boundary: {entry.path!r}"
                    )
                if stat.S_ISDIR(entry_stat.st_mode):
                    stack.append((entry.path, depth + 1, root_dev))
                    continue
                raise UnsafeSandboxWorkdirError(
                    f"external Python read path contains a special filesystem node: {entry.path!r}"
                )


def _path_is_within(
    path: str,
    root: str,
    *,
    strict: bool = False,
) -> bool:
    """Return whether *path* is inside *root*, optionally excluding *root*."""
    try:
        return (not strict or path != root) and os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _validated_user_site_path() -> str | None:
    """Return the parent's bounded user-site directory when it is safe."""
    try:
        user_site = site.getusersitepackages()
    except Exception as e:  # noqa: BLE001 - best-effort; never break tool exec
        logger.debug("site.getusersitepackages() unavailable: %s", e)
        return None
    if not user_site:
        return None
    resolved = os.path.realpath(user_site)
    if not os.path.isdir(resolved) or os.path.dirname(resolved) == resolved:
        return None
    expanded_home = os.path.expanduser("~")
    real_home = os.path.realpath(expanded_home) if expanded_home and expanded_home != "~" else None
    if real_home and _path_is_within(real_home, resolved):
        logger.warning("Ignoring unsafe user-site path containing user home: %s", resolved)
        return None
    return resolved


def opted_in_user_site_path() -> str | None:
    """Return the safe parent user-site path when the operator opts in."""
    if os.environ.get("UNSLOTH_STUDIO_SANDBOX_ALLOW_USER_SITE") != "1":
        return None
    return _validated_user_site_path()


def _plain_pth_source_paths() -> list[str]:
    """Existing source dirs named by non-executable ``.pth`` lines.

    Python permits import statements in ``.pth`` files. Never execute or infer
    from those lines here; finder-based editable installs are handled from the
    already-loaded mapping below.
    """
    site_dirs: list[str] = []
    try:
        site_dirs.extend(site.getsitepackages())
    except Exception as e:  # noqa: BLE001 - best-effort; never break tool exec
        logger.debug("site.getsitepackages() unavailable for .pth scan: %s", e)
    user_site = opted_in_user_site_path()
    if user_site:
        site_dirs.append(user_site)

    paths: list[str] = []
    for site_dir in site_dirs:
        if not os.path.isdir(site_dir):
            continue
        try:
            entries = list(os.scandir(site_dir))
        except OSError as e:
            logger.debug("Cannot scan site-packages %s for .pth files: %s", site_dir, e)
            continue
        for entry in entries:
            if not entry.name.endswith(".pth"):
                continue
            try:
                with open(entry.path, encoding = "utf-8", errors = "surrogateescape") as f:
                    data = f.read(1_048_577)
            except OSError as e:
                logger.debug("Cannot read editable path file %s: %s", entry.path, e)
                continue
            if len(data) > 1_048_576:
                logger.warning("Ignoring oversized editable path file: %s", entry.path)
                continue
            for raw_line in data.splitlines():
                line = raw_line.rstrip()
                if not line or line.startswith("#") or line.startswith(("import ", "import\t")):
                    continue
                candidate = line if os.path.isabs(line) else os.path.join(site_dir, line)
                if "\x00" not in candidate and os.path.isdir(candidate):
                    paths.append(candidate)
    return paths


def plain_pth_pythonpath_roots() -> list[str]:
    """Validated plain-``.pth`` roots safe to place on the child import path.

    The roots themselves are not added to the sandbox read allow-list. Only
    their importable children are mounted/readable, so checkout-level secrets
    remain unavailable even though Python can resolve those child names.
    """
    expanded_home = os.path.expanduser("~")
    real_home = os.path.realpath(expanded_home) if expanded_home and expanded_home != "~" else None
    roots: list[str] = []
    for path in _plain_pth_source_paths():
        resolved = os.path.realpath(path)
        if os.path.dirname(resolved) == resolved:
            logger.warning("Ignoring unsafe filesystem-root editable path: %s", resolved)
            continue
        if real_home and _path_is_within(real_home, resolved):
            logger.warning("Ignoring editable path containing user home: %s", resolved)
            continue
        roots.append(resolved)
    return list(dict.fromkeys(roots))


def _namespace_tree_is_importable(root: str) -> bool:
    """Boundedly find an importable module/package below a PEP 420 root."""
    stack = [(root, 0)]
    inspected = 0
    while stack:
        directory, depth = stack.pop()
        if depth > _PTH_NAMESPACE_SCAN_MAX_DEPTH:
            logger.warning("Ignoring over-deep editable namespace tree: %s", root)
            return False
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for entry in entries:
                inspected += 1
                if inspected > _PTH_NAMESPACE_SCAN_MAX_ENTRIES:
                    logger.warning("Ignoring oversized editable namespace tree: %s", root)
                    return False
                if entry.name.startswith("."):
                    continue
                try:
                    if entry.is_file(follow_symlinks = True):
                        stem, extension = os.path.splitext(entry.name)
                        if extension == ".py" and stem.isidentifier():
                            return True
                        continue
                    if not entry.is_dir(follow_symlinks = True) or not entry.name.isidentifier():
                        continue
                    if os.path.isfile(os.path.join(entry.path, "__init__.py")):
                        return True
                    stack.append((entry.path, depth + 1))
                except OSError:
                    continue
    return False


def _plain_pth_import_paths() -> list[str]:
    """Narrow plain-``.pth`` roots to importable package/module entries.

    Binding the root itself would expose checkout-level files such as ``.env``
    and deployment configuration. Python can still import through the original
    ``sys.path`` entry when only its importable children exist in the sandbox.
    """
    paths: list[str] = []
    for root in _plain_pth_source_paths():
        try:
            entries = os.scandir(root)
        except OSError as exc:
            logger.debug("Cannot narrow editable source root %s: %s", root, exc)
            continue
        with entries:
            for entry in entries:
                name = entry.name
                if name.startswith("."):
                    continue
                try:
                    if entry.is_file(follow_symlinks = True):
                        stem, extension = os.path.splitext(name)
                        native_module = any(
                            name.endswith(suffix) and name[: -len(suffix)].isidentifier()
                            for suffix in importlib.machinery.EXTENSION_SUFFIXES
                        )
                        if (extension == ".py" and stem.isidentifier()) or native_module:
                            paths.append(entry.path)
                        continue
                    if not entry.is_dir(follow_symlinks = True) or not name.isidentifier():
                        continue
                    # Regular packages have __init__.py. Namespace packages
                    # can have several identifier-only levels before the first
                    # module/package, so traverse them with explicit bounds.
                    if os.path.isfile(os.path.join(entry.path, "__init__.py")):
                        paths.append(entry.path)
                        continue
                    if _namespace_tree_is_importable(entry.path):
                        paths.append(entry.path)
                except OSError:
                    continue
    return paths


def _editable_source_paths() -> list[str]:
    """Import paths registered by plain ``.pth`` or PEP 660 editable installs.

    Finder mappings are read from the parent's ``sys.modules`` and are valid
    for the child only while it shares ``sys.executable`` with the parent.
    """
    paths: list[str] = _plain_pth_import_paths()
    user_site = _validated_user_site_path()
    user_site_opted_in = os.environ.get("UNSLOTH_STUDIO_SANDBOX_ALLOW_USER_SITE") == "1"
    for name, mod in list(sys.modules.items()):
        if not (name.startswith("__editable___") and name.endswith("_finder")):
            continue
        origin = getattr(mod, "__file__", None)
        if origin is None:
            origin = getattr(getattr(mod, "__spec__", None), "origin", None)
        if (
            user_site
            and origin
            and _path_is_within(os.path.realpath(origin), user_site)
            and not user_site_opted_in
        ):
            continue
        paths.extend(getattr(mod, "MAPPING", {}).values())
        for ns_paths in getattr(mod, "NAMESPACES", {}).values():
            paths.extend(ns_paths)
    return paths


def _exec_chain_symlinks(executable: str) -> list[str]:
    """Symlinks encountered while resolving *executable* to its real binary.

    Returned paths are the symlinks themselves (not their targets). The
    Linux bwrap argv recreates each link so that the kernel can follow the
    chain during ``execve`` without bind-mounting and exposing an entire
    directory target behind an ancestor symlink.
    """
    out: list[str] = []
    seen_links: set[str] = set()
    current = os.path.abspath(os.path.normpath(executable))
    seen_paths: set[str] = set()
    for _ in range(40):  # cycle guard against pathological symlink loops
        if current in seen_paths:
            break
        seen_paths.add(current)
        parts = current.split(os.sep)
        prefix = "/"
        next_current = None
        for index, p in enumerate(parts[1:]):
            prefix = os.path.normpath(os.path.join(prefix, p))
            try:
                if not os.path.islink(prefix):
                    continue
                if prefix not in seen_links:
                    seen_links.add(prefix)
                    out.append(prefix)
                target = os.readlink(prefix)
            except OSError as e:
                logger.debug("exec-chain link resolution failed for %s: %s", prefix, e)
                continue
            if not os.path.isabs(target):
                target = os.path.join(os.path.dirname(prefix), target)
            suffix = parts[index + 2 :]
            next_current = os.path.abspath(os.path.normpath(os.path.join(target, *suffix)))
            break
        if next_current is None or next_current == current:
            break
        current = next_current
    return out


def _decode_mountinfo_path(path: str) -> str:
    """Decode the octal escapes used for mountinfo path fields."""
    for escaped, literal in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        path = path.replace(escaped, literal)
    return path


def _linux_nested_mount_points(workdir: str) -> set[str]:
    """Return Linux mount points strictly below *workdir*, including bind mounts."""
    if sys.platform != "linux":
        return set()
    wd = os.path.normpath(os.path.realpath(workdir))
    mounts: set[str] = set()
    try:
        with open(_LINUX_MOUNTINFO, encoding = "utf-8") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) < 6:
                    continue
                mount_point = os.path.normpath(_decode_mountinfo_path(fields[4]))
                if _path_is_within(mount_point, wd, strict = True):
                    mounts.add(mount_point)
    except OSError as exc:
        raise UnsafeSandboxWorkdirError(
            f"cannot inspect Linux mount boundaries safely: {_LINUX_MOUNTINFO!r}: {exc}"
        ) from exc
    return mounts


def _linux_accelerator_sysfs_paths() -> list[str]:
    """Return accelerator class trees and their canonical backing trees."""
    paths: list[str] = []
    for class_path in _LINUX_ACCELERATOR_SYSFS_CLASS_PATHS:
        if not os.path.isdir(class_path):
            continue
        sysfs_root = os.path.realpath(os.path.join(class_path, os.pardir, os.pardir))
        paths.append(class_path)
        try:
            with os.scandir(class_path) as entries:
                for entry in entries:
                    for candidate in (entry.path, os.path.join(entry.path, "device")):
                        target = os.path.realpath(candidate)
                        if _path_is_within(target, sysfs_root, strict = True) and os.path.isdir(
                            target
                        ):
                            paths.append(target)
        except OSError as exc:
            logger.debug("accelerator sysfs discovery failed for %s: %s", class_path, exc)
    return list(dict.fromkeys(paths))


def _linux_identity_paths_are_safe(paths: tuple[str, str], workdir: str) -> bool:
    """Validate cached bind sources without following replaceable symlinks."""
    passwd_path, group_path = paths
    identity_dir = os.path.dirname(passwd_path)
    if os.path.dirname(group_path) != identity_dir:
        return False
    if _path_is_within(os.path.realpath(identity_dir), os.path.realpath(workdir)):
        return False
    try:
        directory_stat = os.lstat(identity_dir)
        file_stats = [os.lstat(passwd_path), os.lstat(group_path)]
    except OSError:
        return False
    if not stat.S_ISDIR(directory_stat.st_mode) or stat.S_IMODE(directory_stat.st_mode) != 0o700:
        return False
    return all(
        stat.S_ISREG(file_stat.st_mode) and stat.S_IMODE(file_stat.st_mode) == 0o600
        for file_stat in file_stats
    )


def _linux_sandbox_identity_files(workdir: str) -> tuple[str, str]:
    """Create a private, minimal passwd/group view for the user namespace."""
    global _sandbox_identity_paths
    if _sandbox_identity_paths is not None and _linux_identity_paths_are_safe(
        _sandbox_identity_paths, workdir
    ):
        return _sandbox_identity_paths
    with _sandbox_identity_lock:
        if _sandbox_identity_paths is not None and _linux_identity_paths_are_safe(
            _sandbox_identity_paths, workdir
        ):
            return _sandbox_identity_paths
        _sandbox_identity_paths = None
        temp_root = "/tmp" if sys.platform == "linux" else tempfile.gettempdir()
        if _path_is_within(os.path.realpath(temp_root), os.path.realpath(workdir)):
            raise UnsafeSandboxWorkdirError(
                "cannot create sandbox identity files outside the writable workdir"
            )
        identity_dir = tempfile.mkdtemp(prefix = "unsloth-studio-sandbox-identity-", dir = temp_root)
        os.chmod(identity_dir, 0o700)
        passwd_path = os.path.join(identity_dir, "passwd")
        group_path = os.path.join(identity_dir, "group")
        uid = getattr(os, "getuid", lambda: 65534)()
        gid = getattr(os, "getgid", lambda: 65534)()
        with open(passwd_path, "w", encoding = "utf-8", newline = "\n") as handle:
            handle.write(f"sandbox:x:{uid}:{gid}:Sandbox User:/tmp:/bin/sh\n")
        with open(group_path, "w", encoding = "utf-8", newline = "\n") as handle:
            handle.write(f"sandbox:x:{gid}:\n")
        os.chmod(passwd_path, 0o600)
        os.chmod(group_path, 0o600)
        atexit.register(shutil.rmtree, identity_dir, ignore_errors = True)
        _sandbox_identity_paths = (passwd_path, group_path)
        return _sandbox_identity_paths


def _python_read_paths() -> list[str]:
    """Real paths the Python interpreter needs to read at runtime.

    Returns ``sys.prefix``, ``sys.base_prefix``, the sandbox sitecustomize
    shim, system site-packages, user site-packages, and editable-install source dirs — all
    realpath-normalized, deduplicated, and filtered to existing paths.
    Used by both the macOS Seatbelt profile and the Linux bwrap argv.
    """
    candidates: list[str] = [sys.prefix, sys.base_prefix, _SANDBOX_SITE_DIR]
    # site.getsitepackages / getusersitepackages are absent (older virtualenv
    # site.py) or can raise (embedded / frozen builds) in some environments.
    # This runs in the sandboxed exec path, so degrade gracefully instead of
    # failing the tool call: sys.prefix / sys.base_prefix are bound regardless,
    # so a venv's site-packages under the prefix stays visible even if these do
    # not resolve.
    try:
        candidates.extend(site.getsitepackages())
    except Exception as e:  # noqa: BLE001 - best-effort; never break tool exec
        logger.debug("site.getsitepackages() unavailable: %s", e)
    # user-site is under real $HOME; exposing it defeats the deny-$HOME stance.
    user_site = opted_in_user_site_path()
    if user_site:
        candidates.append(user_site)
    resolved_candidates = [os.path.realpath(p) for p in candidates if p]
    for path in _editable_source_paths():
        if not path:
            continue
        alias = os.path.abspath(os.path.normpath(path))
        resolved_candidates.append(alias)
        resolved_candidates.append(os.path.realpath(alias))
    if any(_path_is_within(p, _NIX_STORE) for p in resolved_candidates):
        # Nix packages keep their ELF loader and shared-library runtime closure
        # in sibling store derivations, not necessarily below sys.prefix.
        resolved_candidates.append(os.path.realpath(_NIX_STORE))

    expanded_home = os.path.expanduser("~")
    real_home = os.path.realpath(expanded_home) if expanded_home and expanded_home != "~" else None
    if real_home:
        shared_local = os.path.realpath(os.path.join(real_home, ".local"))
        narrowed_candidates: list[str] = []
        for rp in resolved_candidates:
            if rp != shared_local:
                narrowed_candidates.append(rp)
                continue
            # ~/.local is a shared application-data root, not a dedicated
            # Python runtime. Keep executable/library trees without exposing
            # unrelated ~/.local/share or ~/.local/state contents.
            narrowed_candidates.extend(os.path.join(rp, name) for name in ("bin", "lib", "lib64"))
        resolved_candidates = narrowed_candidates
    seen: set[str] = set()
    out: list[str] = []
    for rp in resolved_candidates:
        if rp in seen or not os.path.exists(rp):
            continue
        # A root-valued prefix would turn the deny-by-default sandbox into a
        # read-all-files profile or bind mount. Embedded Python builds and
        # root-prefix containers can legitimately report this value.
        canonical = os.path.realpath(rp)
        if os.path.dirname(canonical) == canonical:
            logger.warning("Ignoring unsafe filesystem-root Python read path: %s", canonical)
            continue
        # A prefix equal to $HOME (or an ancestor such as /home) exposes
        # credentials and unrelated user files. Narrow runtime dirs below HOME,
        # including ~/.local and project .venv directories, remain safe to bind.
        if real_home and _path_is_within(real_home, canonical):
            logger.warning("Ignoring Python read path containing user home: %s", canonical)
            continue
        seen.add(rp)
        out.append(rp)
    return out


def _macos_seatbelt_profile(workdir: str) -> str:
    """Build a Seatbelt profile string for ``sandbox-exec -p``."""
    python_read_paths = _python_read_paths()
    _assert_no_external_hardlinks(workdir, python_read_paths)
    wd = _safe_subpath(os.path.realpath(workdir))

    def _path_clause(path: str) -> str:
        kind = "literal" if os.path.isfile(path) else "subpath"
        return f'({kind} "{_safe_subpath(path)}")'

    py_subpaths = [_path_clause(path) for path in python_read_paths]
    py_block = "\n    ".join(py_subpaths)
    ca_block = "\n    ".join(
        f'(literal "{_safe_subpath(p)}")'
        if os.path.splitext(p)[1]
        else f'(subpath "{_safe_subpath(p)}")'
        for p in _MACOS_CA_READ_PATHS
    )
    # Optional Homebrew prefixes (Intel + Apple Silicon). Skipped when
    # the directory doesn't exist so the profile stays minimal on macs
    # without Homebrew. Without this, _build_safe_env's PATH includes
    # /usr/local/bin but Seatbelt blocks exec/read there, and a stock
    # `bash` that resolves to /usr/local/bin/bash fails.
    homebrew_subpaths = [
        f'(subpath "{_safe_subpath(p)}")' for p in _MACOS_EXTRA_EXEC_PREFIXES if os.path.isdir(p)
    ]
    developer_subpaths = [
        f'(subpath "{_safe_subpath(p)}")' for p in _MACOS_DEVELOPER_PREFIXES if os.path.isdir(p)
    ]
    workdir_subpath = f'(subpath "{wd}")'
    # Paths the kernel needs mmap(PROT_EXEC) on so the loader can map
    # binaries and dylibs as code. Narrower than the full read allow
    # because most things we permit reads of are data, not executables.
    # workdir is included so a tool that compiles + dlopens a local
    # .dylib in its session folder works on macOS, matching how the
    # Linux side allows exec from the bind-mounted workdir.
    executable_map_block = "\n    ".join(
        [
            '(subpath "/usr/lib")',
            '(subpath "/usr/bin")',
            '(subpath "/bin")',
            '(subpath "/System/Library/Frameworks")',
            '(subpath "/System/Library/PrivateFrameworks")',
            '(subpath "/System/Cryptexes")',
            '(subpath "/System/Volumes/Preboot/Cryptexes")',
            '(subpath "/Library/Frameworks")',
            *py_subpaths,
            *homebrew_subpaths,
            *developer_subpaths,
            workdir_subpath,
        ]
    )

    # Same symmetry on process-exec: tools may need to run ./run.sh
    # they just generated inside the workdir; Linux already allows that
    # via the workdir bind.
    process_exec_block = "\n    ".join(
        [
            '(subpath "/usr/lib")',
            '(subpath "/usr/bin")',
            '(subpath "/bin")',
            '(subpath "/System/Library/Frameworks")',
            '(subpath "/System/Library/PrivateFrameworks")',
            '(subpath "/System/Cryptexes")',
            '(subpath "/System/Volumes/Preboot/Cryptexes")',
            '(subpath "/Library/Frameworks")',
            *py_subpaths,
            *homebrew_subpaths,
            *developer_subpaths,
            workdir_subpath,
        ]
    )

    homebrew_read_block = "\n    " + "\n    ".join(homebrew_subpaths) if homebrew_subpaths else ""
    developer_read_block = (
        "\n    " + "\n    ".join(developer_subpaths) if developer_subpaths else ""
    )

    return f"""(version 1)
(deny default)

(allow process-fork)
(allow process-exec
    {process_exec_block}
)
(allow signal (target same-sandbox))
(allow process-info-pidinfo (target self))
(allow process-info-pidfdinfo (target self))
(allow sysctl-read)
(allow file-read-metadata)

(allow file-read*
    ; (literal "/") is required: dyld and many runtime resolvers stat
    ; the root directory itself, which is NOT matched by (subpath "/X").
    (literal "/")
    ; --- Execution surface ---
    (subpath "/usr/lib")
    (subpath "/usr/bin")
    (subpath "/bin")
    ; Narrow /System: only the framework + dyld surfaces the loader needs.
    ; Avoids exposing /System/Applications/* (~all installed system apps and
    ; their localized resources) and /System/iOSSupport to the LLM.
    (subpath "/System/Library/Frameworks")
    (subpath "/System/Library/PrivateFrameworks")
    (subpath "/System/Library/dyld")
    (subpath "/System/Cryptexes")
    (subpath "/System/Volumes/Preboot/Cryptexes")
    (subpath "/Library/Frameworks")
    ; --- Runtime data libraries actually consult ---
    (subpath "/usr/share/zoneinfo")        ; tzdata for datetime
    (subpath "/usr/share/icu")             ; ICU data
    (subpath "/private/var/db/dyld")
    (subpath "/private/var/db/timezone")
    ; Narrow /private/etc to runtime essentials; deny passwd/shadow/sudoers etc.
    (literal "/private/etc/hosts")
    (literal "/private/etc/resolv.conf")
    (literal "/private/etc/nsswitch.conf")
    (literal "/private/etc/localtime")
    (literal "/private/etc/protocols")
    (literal "/private/etc/services")
    {ca_block}
    (literal "/dev/null")
    (literal "/dev/zero")
    (literal "/dev/random")
    (literal "/dev/urandom")
    (literal "/dev/dtracehelper")
    (literal "/dev/autofs_nowait")
    (subpath "/dev/fd")
    (literal "/dev/stdin")
    (literal "/dev/stdout")
    (literal "/dev/stderr")
    {py_block}{homebrew_read_block}{developer_read_block}
)

; Required for mmap(PROT_EXEC) on dylibs — without this Python cannot
; load libpython, libsystem_*, or any C-extension .so. Also required
; for /bin/bash and /usr/bin/* under the terminal tool.
(allow file-map-executable
    {executable_map_block}
)

(allow file-read* (subpath "{wd}"))
(allow file-write* (subpath "{wd}"))
(allow file-ioctl (subpath "{wd}"))
(allow file-write-data
    (require-all
        (path "/dev/null")
        (vnode-type CHARACTER-DEVICE)
    )
)
(allow file-write* (subpath "/dev/fd"))

; coreservices.launchservicesd + lsd.mapdb are intentionally NOT allowed:
; together with (allow process-exec /usr/bin) they let a tool run
; `open URL` and have LaunchServices spawn a browser outside the
; sandbox, bypassing (deny network-outbound).
; SecurityServer is also NOT allowed: with `/usr/bin/security` reachable
; it would expose Keychain reads of stored credentials.
(allow mach-lookup
    (global-name "com.apple.trustd.agent")
    (global-name "com.apple.trustd")
    (global-name "com.apple.system.opendirectoryd.libinfo")
    (global-name "com.apple.system.opendirectoryd.membership")
    (global-name "com.apple.system.logger")
    (global-name "com.apple.system.notification_center")
    (global-name "com.apple.system.DirectoryService.libinfo_v1")
)

(deny network-outbound)
(deny network-inbound)
(deny network-bind)
"""


_LINUX_NPROC_WRAPPER_TEMPLATE = (
    "import os, resource, sys\n"
    "try:\n"
    "    nproc = {nproc}\n"
    "    _soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)\n"
    "    target = nproc if hard == resource.RLIM_INFINITY else min(nproc, hard)\n"
    "    resource.setrlimit(resource.RLIMIT_NPROC, (target, target))\n"
    "except (ValueError, OSError, AttributeError):\n"
    "    pass\n"
    "os.execvp(sys.argv[1], sys.argv[1:])\n"
)


_NPROC_DEFAULT = 10000
# Hard floor: Python's own startup needs a handful of threads for the
# GC and signal handlers; multiprocessing needs at least two. A value
# of 0 or 1 would brick the inner interpreter before the LLM-supplied
# code even ran. 64 is well below any realistic legitimate need and
# well above the kernel minimum.
_NPROC_FLOOR = 64


def _resolve_nproc_limit() -> int:
    """Read the sandbox's additional task budget; default 10000.

    ``_build_safe_env`` is a strict whitelist, so the env var is not
    propagated into the sandbox. Bake the value into the wrapper at
    argv-build time so the operator's override still takes effect
    inside the namespace.

    Values below ``_NPROC_FLOOR`` are silently clamped up; a value of
    0 would otherwise prevent the inner Python wrapper itself from
    spawning the LLM-controlled child.
    """
    try:
        value = int(os.environ.get("UNSLOTH_STUDIO_SANDBOX_NPROC", str(_NPROC_DEFAULT)))
    except ValueError:
        return _NPROC_DEFAULT
    if value < _NPROC_FLOOR:
        logger.warning(
            "UNSLOTH_STUDIO_SANDBOX_NPROC=%s below floor %s; clamping",
            value,
            _NPROC_FLOOR,
        )
        return _NPROC_FLOOR
    return value


def _linux_real_uid_task_count() -> int | None:
    """Count tasks currently charged to this process's real Linux UID.

    ``RLIMIT_NPROC`` is accounted against the real host UID even after
    Bubblewrap enters a user namespace. The inner limit therefore needs to
    include the tasks that UID already owns, or a busy host can exhaust the
    limit before the sandbox starts its first worker.
    """
    if sys.platform != "linux":
        return None
    try:
        real_uid = os.getuid()
        process_entries = os.scandir("/proc")
    except (AttributeError, OSError):
        return None

    total = 0
    saw_process = False
    with process_entries:
        for entry in process_entries:
            if not entry.name.isdigit():
                continue
            try:
                with open(os.path.join(entry.path, "status"), encoding = "utf-8") as status_file:
                    uid = None
                    threads = None
                    for line in status_file:
                        if line.startswith("Uid:"):
                            fields = line.split()
                            uid = int(fields[1]) if len(fields) >= 2 else None
                        elif line.startswith("Threads:"):
                            fields = line.split()
                            threads = int(fields[1]) if len(fields) >= 2 else None
                        if uid is not None and threads is not None:
                            break
            except (OSError, ValueError):
                # Processes can exit between scandir and status reads.
                continue
            if uid == real_uid:
                saw_process = True
                total += max(1, threads or 1)
    return total if saw_process else None


def _linux_nproc_rlimit_target() -> int | None:
    """Return the host-UID baseline plus this sandbox's task budget."""
    baseline = _linux_real_uid_task_count()
    if baseline is None:
        logger.warning("Could not count host UID tasks; leaving RLIMIT_NPROC unchanged")
        return None
    return baseline + _resolve_nproc_limit()


def _linux_rlimit_python_path() -> str | None:
    """Return a trusted interpreter that can run the pre-exec limit wrapper."""
    candidates = (
        getattr(sys, "_base_executable", None),
        os.path.join(sys.base_prefix, "bin", "python3"),
        "/usr/bin/python3",
        "/bin/python3",
    )
    for candidate in candidates:
        if not candidate:
            continue
        resolved = os.path.realpath(candidate)
        if _linux_executable_path_is_trusted(resolved):
            return resolved
    return None


# Import-time sanity: catch the case where a maintainer accidentally
# adds a literal `{` to the template (e.g. a dict literal) which would
# turn .format() into a KeyError at every tool call.
assert "12345" in _LINUX_NPROC_WRAPPER_TEMPLATE.format(
    nproc = 12345
), "_LINUX_NPROC_WRAPPER_TEMPLATE does not format cleanly"


def _linux_inner_rlimit_wrapper(inner_argv: list[str]) -> list[str]:
    """Wrap ``inner_argv`` with a tiny Python that sets RLIMIT_NPROC.

    Linux charges this limit to the real host UID, not the user-namespace
    mapping. Bake the observed host-UID task count into the target so the
    configured value is an additional sandbox budget rather than a total that
    a busy host may already exceed. If ``/proc`` cannot be read reliably, do
    not install a misleading or immediately exhausted limit.
    """
    exe = _linux_rlimit_python_path()
    if exe is None:
        logger.warning("No trusted Python executable is available for the RLIMIT_NPROC wrapper")
        return inner_argv
    target = _linux_nproc_rlimit_target()
    if target is None:
        return inner_argv
    script = _LINUX_NPROC_WRAPPER_TEMPLATE.format(nproc = target)
    # Isolated mode keeps the model-writable CWD and PYTHONPATH from shadowing
    # stdlib modules imported by this trusted wrapper. execvp then launches the
    # requested command with its normal environment and import behavior.
    return [exe, "-I", "-S", "-c", script, *inner_argv]


def _linux_bwrap_argv(inner_argv: list[str], workdir: str) -> list[str]:
    """Build a ``bwrap`` argv for the Linux sandbox.

    Deny by omission: the child sees only what we bind-mount. ``net``
    is unshared without loopback, so all network is denied. ``/tmp``
    is a fresh tmpfs so writes don't leak to the host. The inner argv
    is wrapped with a small Python that re-applies RLIMIT_NPROC inside
    the userns (see :func:`_linux_inner_rlimit_wrapper`).
    """
    wd = os.path.realpath(workdir)
    python_read_paths = _python_read_paths()
    oneapi_runtime_bindings = _linux_oneapi_runtime_bindings()
    _assert_no_external_hardlinks(workdir, python_read_paths)
    _assert_external_read_paths_have_no_special_nodes(
        workdir, [*python_read_paths, *(source for source, _dest in oneapi_runtime_bindings)]
    )
    top_ro_dirs = ("/usr", "/bin", "/sbin", "/lib", "/lib64")
    # Narrow /etc to runtime essentials; deny sshd_config, machine-id, etc.
    etc_ro_entries = (
        "/etc/alternatives",
        "/etc/hosts",
        "/etc/resolv.conf",
        "/etc/nsswitch.conf",
        "/etc/localtime",
        "/etc/ld.so.cache",
        "/etc/ld.so.conf",
        "/etc/ld.so.conf.d",
        *_LINUX_CA_READ_PATHS,
    )
    passwd_path, group_path = _linux_sandbox_identity_files(workdir)

    assert _linux_bwrap_path is not None, "bwrap path unset despite successful probe"
    group_devices = _linux_supplementary_group_devices()
    if group_devices and not _linux_bwrap_keep_groups:
        raise UnsafeSandboxWorkdirError(
            "Bubblewrap cannot retain the supplementary group required by accelerator "
            f"devices: {', '.join(group_devices[:3])}"
        )
    args: list[str] = [
        _linux_bwrap_path,
        *(["--keep-groups"] if _linux_bwrap_keep_groups else []),
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/dev/shm",
        "--tmpfs",
        "/tmp",
        "--ro-bind-try",
        "/sys/fs/cgroup",
        "/sys/fs/cgroup",
    ]
    # ``--dev /dev`` creates only a minimal synthetic tree. Restore the host's
    # accelerator device nodes while retaining its DAC/cgroup permissions; the
    # child env separately preserves the operator's visible-device selectors.
    for device_path in _LINUX_ACCELERATOR_DEVICE_PATHS:
        args.extend(["--dev-bind-try", device_path, device_path])
    drm_render_paths = _linux_drm_render_device_paths()
    if drm_render_paths:
        args.extend(["--dir", _LINUX_DRM_DIR])
        for device_path in drm_render_paths:
            args.extend(["--dev-bind-try", device_path, device_path])
    try:
        with os.scandir("/dev") as entries:
            nvidia_devices = sorted(
                entry.path
                for entry in entries
                if entry.name.startswith("nvidia")
                and entry.name.removeprefix("nvidia").isdigit()
                and not entry.is_symlink()
            )
    except OSError:
        nvidia_devices = []
    for device_path in nvidia_devices:
        args.extend(["--dev-bind-try", device_path, device_path])
    for sysfs_path in _linux_accelerator_sysfs_paths():
        args.extend(["--ro-bind-try", sysfs_path, sysfs_path])
    for source, destination in _linux_rocm_runtime_bindings():
        args.extend(["--ro-bind-try", source, destination])
    for source, destination in oneapi_runtime_bindings:
        args.extend(["--ro-bind-try", source, destination])
    # -try variants skip missing paths so the same argv works on
    # usrmerge distros (/lib, /lib64 are symlinks into /usr or absent).
    for d in top_ro_dirs:
        args.extend(["--ro-bind-try", d, d])
    for d in etc_ro_entries:
        args.extend(["--ro-bind-try", d, d])
    args.extend(["--ro-bind", passwd_path, "/etc/passwd"])
    args.extend(["--ro-bind", group_path, "/etc/group"])

    def _is_under_top_ro(path: str) -> bool:
        return any(path == top or path.startswith(top + os.sep) for top in top_ro_dirs)

    # _python_read_paths() already realpaths, filters non-dirs, dedupes,
    # and includes editable-install source dirs (so `pip install -e .`
    # repos like unsloth remain readable inside the sandbox).
    for rp in python_read_paths:
        if _is_under_top_ro(rp) or _path_is_within(rp, wd):
            continue
        args.extend(["--ro-bind-try", rp, rp])

    # Recreate exec-chain symlinks whose parent isn't already covered by
    # an existing bind. A bwrap bind of a symlink source follows it and can
    # expose the whole target tree; --symlink preserves only the link itself.
    # Links under an existing bind are already reachable by path inheritance.
    bind_flags = ("--ro-bind", "--ro-bind-try", "--bind", "--bind-try")
    bound_dests = [
        args[i + 2] for i, arg in enumerate(args) if arg in bind_flags and i + 2 < len(args)
    ]

    bound_links: set[str] = set()
    # Normalize sys.executable so a launcher path containing `..` (e.g.
    # `../.venv/bin/python`) is resolved before walking the symlink
    # chain; an unresolved `..` segment would land outside the bind set
    # and the bwrap child would fail to exec.
    exe = os.path.abspath(os.path.normpath(sys.executable))
    for sym in _exec_chain_symlinks(exe):
        # Once an ancestor link points at the canonical runtime bind, links
        # below it are already present through that bind. Bubblewrap refuses
        # to create a destination beneath a symlink, so do not recreate them.
        if any(_path_is_within(sym, link, strict = True) for link in bound_links):
            continue
        if sym in bound_links or _is_under_top_ro(sym) or _path_is_within(sym, wd):
            continue
        parent = os.path.dirname(sym)
        if any(parent == b or parent.startswith(b + os.sep) for b in bound_dests):
            continue
        try:
            target = os.readlink(sym)
        except OSError as exc:
            raise UnsafeSandboxWorkdirError(
                f"cannot safely recreate executable symlink {sym!r}: {exc}"
            ) from exc
        args.extend(["--dir", parent])
        bound_links.add(sym)
        args.extend(["--symlink", target, sym])

    args.extend(["--bind", wd, wd])

    wrapped_inner_argv = _linux_inner_rlimit_wrapper(inner_argv)
    seccomp_fd = None
    if sys.platform == "linux":
        seccomp_fd = _linux_socket_seccomp_fd()
        args.extend(["--seccomp", str(seccomp_fd)])

    args.append("--")
    args.extend(wrapped_inner_argv)
    return SandboxArgv(args, (seccomp_fd,) if seccomp_fd is not None else ())


def build_sandbox_argv(inner_argv: list[str], workdir: str) -> list[str]:
    """Return an argv that runs *inner_argv* under the platform sandbox.

    Caller MUST gate with :func:`sandbox_available`; reaching the final
    AssertionError indicates the gate was bypassed.
    """
    if not inner_argv:
        raise ValueError("inner_argv must be non-empty")

    if sys.platform == "darwin":
        profile = _macos_seatbelt_profile(workdir)
        return [_SANDBOX_EXEC, "-p", profile, *inner_argv]
    if sys.platform == "linux":
        return _linux_bwrap_argv(inner_argv, workdir)
    raise AssertionError(
        f"build_sandbox_argv called on unsupported platform {sys.platform!r}; "
        "callers must gate with sandbox_available()"
    )
