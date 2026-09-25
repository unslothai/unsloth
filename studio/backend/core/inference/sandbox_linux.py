# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bubblewrap backend. Bind parent directories, never single .so files: dlopen() reaches ones enumeration misses."""

from __future__ import annotations

import os
import shutil
import stat
import site
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from functools import lru_cache

from loggers import get_logger

from . import sandbox_landlock, sandbox_seccomp
from .os_sandbox import (
    CACHE_SCAN_SECONDS,
    PROFILE_VERSION,
    SESSION_PACKAGES_RELPATH,
    PreparedSandboxLaunch,
    SandboxUnavailableError,
    ToolLaunchPlan,
    WorkdirUnsafeError,
    cache_share_hazard,
    directory_witness_matches,
    editable_source_roots,
    scan_workdir_for_host_channels,
    model_library_roots,
    studio_state_roots,
)

logger = get_logger(__name__)

BACKEND_NAME = "bubblewrap"
PROFILE_ID = f"linux-bwrap-{PROFILE_VERSION}"
LIMITATIONS = (
    "unrestricted_network",
    "system_paths_readable",
    "model_cache_writable",
    # --dev builds a fresh /dev, so a sandboxed tool call is CPU-only.
    "gpu_devices_hidden",
    # --unshare-pid: a backgrounded process dies with the foreground command.
    "detached_processes_die_with_the_call",
    # Without the Landlock scope (Linux 6.12+) abstract AF_UNIX sockets reach the session bus and X.
    *(() if sandbox_landlock.abstract_scope_supported() else ("host_abstract_sockets_reachable",)),
    # Read-only mounts do not block socket connect(); system roots are not scanned per call.
    "host_pathname_sockets_reachable",
    "shared_kernel",
)

_SYSTEM_ROOTS = (
    "/usr/bin",
    "/usr/sbin",
    "/usr/lib",
    "/usr/lib64",
    "/usr/libexec",
    "/usr/local/bin",
    "/usr/local/lib",
    "/usr/local/lib64",
    "/usr/local/libexec",
    "/usr/local/sbin",
    "/usr/local/share",
    "/usr/share",
    "/opt",
    "/usr/include",
    "/usr/local/include",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
)


def _trusted_bwrap_path() -> str:
    """Resolve bubblewrap to a system-owned executable that the Studio user cannot replace."""
    candidate = shutil.which("bwrap")
    if candidate is None:
        raise SandboxUnavailableError("bubblewrap (bwrap) is not installed on this host")
    resolved = os.path.realpath(candidate)
    try:
        executable = os.stat(resolved, follow_symlinks = False)
        if (
            not stat.S_ISREG(executable.st_mode)
            or executable.st_uid != 0
            or executable.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            or not os.access(resolved, os.X_OK)
        ):
            raise OSError("the executable is not root-owned, executable, and non-writable")
        parent = os.path.dirname(resolved)
        while True:
            directory = os.stat(parent, follow_symlinks = False)
            if (
                not stat.S_ISDIR(directory.st_mode)
                or directory.st_uid != 0
                or directory.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            ):
                raise OSError(f"its directory is replaceable: {parent}")
            ancestor = os.path.dirname(parent)
            if ancestor == parent:
                break
            parent = ancestor
    except OSError as exc:
        raise SandboxUnavailableError(
            "bubblewrap must come from a trusted system installation; install it with the "
            f"distribution package manager ({exc})"
        ) from exc
    return resolved


def bwrap_identity() -> str:
    """Stable identity included in capability-cache keys."""
    path = _trusted_bwrap_path()
    info = os.stat(path, follow_symlinks = False)
    return repr((path, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode))


# /etc is fresh in the jail; without these nothing dynamically linked starts.
_ETC_FILES = (
    "/etc/alternatives",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/localtime",
    "/etc/nsswitch.conf",
)
# Bound only when it passes _trusted_system_file.
_ETC_FILES_IF_TRUSTED = ("/etc/gitconfig",)
# PUBLIC halves one by one, never /etc/ssl or /etc/pki whole: both hold private keys.
_NETWORK_FILES = (
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/host.conf",
    "/etc/gai.conf",
    "/etc/services",
    "/etc/protocols",
    "/etc/ssl/certs",
    "/etc/ssl/cert.pem",
    "/etc/ssl/openssl.cnf",
    "/etc/pki/tls/certs",
    "/etc/pki/tls/cert.pem",
    "/etc/pki/tls/openssl.cnf",
    "/etc/pki/ca-trust",
    "/etc/ca-certificates",
    "/etc/ca-certificates.conf",
    "/etc/crypto-policies",
    "/var/lib/ca-certificates",
)
# Bound at the jail's own HOME with HF_HOME pinned to match.
_MODEL_CACHE_RELPATH = os.path.join(".cache", "huggingface")
# Excludes $HF_HOME/token and executable modules; the writable hub is a deliberate tradeoff (#5603).
_MODEL_CACHE_SUBDIRS = ("hub", "datasets", "xet", "assets")
# NixOS keeps glibc here, so a store interpreter cannot link without it.
_NIX_STORE = "/nix/store"


def _trusted_system_file(path: str) -> bool:
    """Require a root-owned, non-user-writable regular file with no symlink hops."""
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if not stat.S_ISREG(info.st_mode):
        return False
    return info.st_uid == 0 and not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


def _within(path: str, root: str) -> bool:
    path, root = os.path.abspath(path), os.path.abspath(root)
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


# A WRITABLE cache must never BE or CONTAIN these, or tool code could modify the user's home.
_CACHE_FORBIDDEN_CHILDREN = (
    ".ssh",
    ".aws",
    ".config/gcloud",
    ".kube",
    ".docker",
    ".gnupg",
    ".netrc",
    ".git-credentials",
)


def _too_broad_for_a_cache(path: str) -> bool:
    """Whether ``path`` is a home or system directory rather than a cache."""
    real = os.path.realpath(path)
    if real == os.sep or os.path.dirname(real) == real:
        return True
    home = os.path.realpath(os.path.expanduser("~"))
    if home and home != os.sep and (_within(home, real) or real == home):
        return True
    if real in ("/home", "/Users", "/root", "/etc", "/var", "/usr", "/opt", "/tmp"):
        return True
    # A directory already holding credentials is not a cache, whatever it is called.
    return any(os.path.exists(os.path.join(real, child)) for child in _CACHE_FORBIDDEN_CHILDREN)


def _holds_studio_state(path: str) -> bool:
    """Whether ``path`` IS, or CONTAINS, a Studio state root."""
    real = os.path.realpath(path)
    return any(_within(real, root) or _within(root, real) for root in studio_state_roots())


def _without_studio_state(roots: tuple[str, ...], depth: int = 4) -> tuple[str, ...]:
    """Bind a system root's children instead of the root when Studio's own state lives inside it."""
    state = studio_state_roots()
    if not state:
        return roots
    kept: list[str] = []
    for root in roots:
        real = os.path.realpath(root)
        if any(_within(real, path) for path in state):
            continue  # the root IS Studio state
        if not any(_within(path, real) for path in state):
            kept.append(root)
            continue
        if depth <= 0:
            # Fails CLOSED: restoring an ancestor of the state directory would hand over auth/auth.db.
            logger.warning(
                "Not binding %s read-only: Studio's own state is nested too "
                "deeply inside it to exclude",
                root,
            )
            continue
        try:
            children = sorted(os.path.join(root, name) for name in os.listdir(root))
        except OSError:
            continue  # unreadable: bind nothing rather than everything
        kept.extend(
            _without_studio_state(
                tuple(path for path in children if _bindable_child(path, real)), depth - 1
            )
        )
    return tuple(dict.fromkeys(kept))


def _bindable_child(path: str, root: str) -> bool:
    """A bind-source subdirectory of ``root``; symlinks leaving ``root`` are dropped, since they resolve on the HOST."""
    try:
        if not stat.S_ISDIR(os.lstat(path).st_mode):
            return stat.S_ISDIR(os.stat(path).st_mode) and _within(os.path.realpath(path), root)
    except OSError:
        return False
    return True


@lru_cache(maxsize = 8)
def _bwrap_long_options(identity: tuple[str, int, int]) -> frozenset[str]:
    """Keyed by file identity so a package upgrade under a running Studio is re-read."""
    try:
        completed = subprocess.run(
            [identity[0], "--help"],
            stdin = subprocess.DEVNULL,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
            close_fds = True,
        )
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(
        token.strip().rstrip(",")
        for line in f"{completed.stdout}\n{completed.stderr}".splitlines()
        for token in line.split()
        if token.startswith("--")
    )


def _bwrap_supports(bwrap: str, option: str) -> bool:
    try:
        info = os.stat(bwrap)
        identity = (bwrap, info.st_ino, info.st_mtime_ns)
    except OSError:
        identity = (bwrap, 0, 0)
    return option in _bwrap_long_options(identity)


def _host_mount_points() -> tuple[str, ...]:
    """Every host mount point; unreadable is a refusal. Not resolved: realpath blocks on stale NFS."""
    points: list[str] = []
    try:
        with open("/proc/self/mountinfo", encoding = "utf-8") as stream:
            for line in stream:
                fields = line.split()
                if len(fields) < 5:
                    raise SandboxUnavailableError("cannot parse the host mount table")
                raw = (
                    fields[4]
                    .replace("\\040", " ")
                    .replace("\\011", "\t")
                    .replace("\\012", "\n")
                    .replace("\\134", "\\")
                )
                points.append(raw)
    except OSError as exc:
        raise SandboxUnavailableError("cannot read the host mount table") from exc
    return tuple(points)


def _runtime_paths_under(workdir: str) -> tuple[str, ...]:
    """Protect Studio's runtime after the writable workdir bind; external symlink targets get no bind."""
    canonical_root = os.path.realpath(workdir)
    inside: list[str] = []
    # Standalone Python may have no bin/; the probe executes this file on the host.
    candidates = [sys.executable, *editable_source_roots()]
    for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        candidates.extend(
            os.path.join(prefix, name)
            for name in ("bin", "include", "lib", "lib64", "libexec", "pyvenv.cfg", "ssl")
        )
    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        written, resolved = os.path.abspath(candidate), os.path.realpath(candidate)
        # Protect both target and name: otherwise the writable alias can be replaced.
        if _within(written, canonical_root) and not _within(resolved, canonical_root):
            if candidate is sys.executable:
                # Refuse: protecting this alias would expose its external target.
                raise WorkdirUnsafeError(
                    f"the session workdir holds a link to the Python that runs Studio: {written}"
                )
            continue
        for path in (written, resolved):
            if _within(path, canonical_root) and path not in inside:
                inside.append(path)
    return tuple(inside)


def _validate_workdir(workdir: str) -> tuple[str, tuple[str, ...]]:
    """Re-read the mount table: os.path.ismount misses a same-filesystem bind mount."""
    resolved = os.path.realpath(workdir)
    if not os.path.isdir(resolved) or os.path.dirname(resolved) == resolved:
        raise WorkdirUnsafeError("the session workdir is not a safe canonical directory")
    for mount in _host_mount_points():
        if mount != resolved and _within(mount, resolved):
            raise WorkdirUnsafeError(f"the session workdir contains a nested host mount: {mount}")
    return resolved, scan_workdir_for_host_channels(resolved)


def _runtime_read_paths(
    workdir: str,
    system_roots: tuple[str, ...],
    alias: str | None = None,
) -> tuple[str, ...]:
    """Find runtime roots from the interpreter; exclude both workdir spellings BEFORE resolving."""
    # All four: uv's base_prefix and base_exec_prefix are different spellings.
    prefixes = (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)
    candidates: list[str] = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_site"),
    ]
    # Subdirectories, never the prefix: ``python -m venv .`` makes sys.prefix the project root.
    for prefix in prefixes:
        candidates.extend(
            os.path.join(prefix, name)
            for name in ("bin", "include", "lib", "lib64", "libexec", "pyvenv.cfg", "ssl")
        )
    try:
        paths = sysconfig.get_paths()
        candidates.extend(
            paths[key]
            for key in ("stdlib", "platstdlib", "purelib", "platlib", "include", "platinclude")
            if paths.get(key)
        )
    except (KeyError, OSError):
        pass
    # Editable installs live outside site-packages; added as CANDIDATES so the exclusions below apply.
    candidates.extend(editable_source_roots())
    # Bind argv[0]'s file, not its parent, which may be the user's home.
    candidates.append(sys.executable)
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:
        pass
    multiarch = sysconfig.get_config_var("MULTIARCH")
    if multiarch:
        candidates.extend(os.path.join(prefix, "lib", multiarch) for prefix in prefixes)

    selected: list[str] = []
    for candidate in candidates:
        if not candidate or not os.path.isabs(candidate):
            continue
        # Check the candidate as WRITTEN, against both workdir spellings, or a symlinked venv/lib exposes its target.
        written = os.path.abspath(candidate)
        if any(_within(written, root) for root in (workdir, alias) if root):
            continue
        for path in (os.path.abspath(candidate), os.path.realpath(candidate)):
            if path == os.path.sep:
                raise SandboxUnavailableError(
                    f"this interpreter would require binding the filesystem root: {candidate}"
                )
            if not os.path.exists(path):
                continue
            if any(_within(path, root) for root in (*system_roots, *selected)):
                continue
            selected.append(path)
    return tuple(selected)


def _identity_files() -> tuple[str, str, str]:
    """Synthesise one-entry passwd and group so getpwuid() works without the host database."""
    directory = tempfile.mkdtemp(prefix = "unsloth-sandbox-identity-")
    uid, gid = os.getuid(), os.getgid()
    passwd, group = os.path.join(directory, "passwd"), os.path.join(directory, "group")
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


def _tmpdir(plan: ToolLaunchPlan, workdir: str) -> str:
    """The caller's TMPDIR spelling when inside the workdir: the private /tmp dies with the namespace."""
    requested = plan.env.get("TMPDIR") or ""
    if not requested:
        return "/tmp"
    return requested if _within(os.path.realpath(requested), workdir) else "/tmp"


def _pythonpath(plan: ToolLaunchPlan, packages: str) -> str:
    """Appended, never prepended: a package must not shadow tools.py's sandbox_site shim."""
    inherited = plan.env.get("PYTHONPATH") or ""
    return os.pathsep.join(part for part in (inherited, packages) if part)


def _path(plan: ToolLaunchPlan, packages: str) -> str:
    """Caller's PATH plus the writable package ``bin`` LAST, so a planted binary cannot shadow a safe command."""
    inherited = plan.env.get("PATH") or ""
    return os.pathsep.join(part for part in (inherited, os.path.join(packages, "bin")) if part)


# Bounded wait on a thread: NFS/FUSE can block isdir/scandir; an uninspected cache is not shared.
_CACHE_INSPECT_SECONDS = CACHE_SCAN_SECONDS + 2.0


def _inspect_cache_component(
    name: str,
    path: str,
    witness: "list[tuple] | None" = None,
) -> "str | None":
    """The hazard for one component, or a reason it could not be inspected."""
    if not os.path.isdir(path):
        return "is not a directory"
    nested = next((m for m in _host_mount_points() if m != path and _within(m, path)), None)
    if nested is not None:
        return f"contains a nested host mount: {nested}"
    return cache_share_hazard(path, witness)


# Retain timed-out workers until they finish, preventing a thread leak on a wedged path.
_cache_scan_pending: "dict[str, threading.Thread]" = {}
# Request threads must reserve and remove workers atomically.
_cache_scan_lock = threading.Lock()


# Memoized per-component verdict; reuse re-stats every directory the scan visited, not just the root.
_CACHE_VERDICT_TTL_SECONDS = 300.0
_cache_verdicts: "dict[str, tuple[float, tuple, list[tuple], str | None]]" = {}


def _cache_component_signature(path: str) -> tuple:
    try:
        info = os.stat(path)
    except OSError:
        return ()
    return (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size)


def reset_cache_verdicts() -> None:
    """Drop every memoized component verdict; called when a launch fails."""
    with _cache_scan_lock:
        _cache_verdicts.clear()


def _cache_hazard_within_deadline(name: str, path: str) -> "str | None":
    """Memo hit or walk, both on the bounded worker: revalidating stats the cache too, and NFS/FUSE can block a stat."""
    answer: list[str | None] = []

    def check() -> None:
        try:
            answer.append(_cache_hazard_memoized(name, path))
        except Exception as exc:  # noqa: BLE001 - a launch never fails over this
            answer.append(f"could not be inspected: {exc}")

    with _cache_scan_lock:
        pending = _cache_scan_pending.get(path)
        if pending is not None:
            if pending.is_alive():
                return "was still being inspected when a previous launch gave up (a wedged mount?)"
            del _cache_scan_pending[path]
        worker = threading.Thread(target = check, name = f"unsloth-cache-scan-{name}", daemon = True)
        # Start under the lock, or another caller can replace the not-yet-alive worker.
        _cache_scan_pending[path] = worker
        worker.start()
    worker.join(_CACHE_INSPECT_SECONDS)
    if not answer:
        return f"could not be inspected within {_CACHE_INSPECT_SECONDS:.0f}s (a wedged mount?)"
    with _cache_scan_lock:
        # By identity: another caller may already have replaced it.
        if _cache_scan_pending.get(path) is worker:
            del _cache_scan_pending[path]
    return answer[0]


def _cache_hazard_memoized(name: str, path: str) -> "str | None":
    signature = _cache_component_signature(path)
    now = time.monotonic()
    with _cache_scan_lock:
        cached = _cache_verdicts.get(path)
    if cached is not None:
        expires, cached_signature, witness, verdict = cached
        # Outside the lock: a stalled stat here must not hold up every other launch.
        if now < expires and cached_signature == signature and directory_witness_matches(witness):
            return verdict
        with _cache_scan_lock:
            if _cache_verdicts.get(path) is cached:
                del _cache_verdicts[path]
    witness: "list[tuple]" = []
    verdict = _inspect_cache_component(name, path, witness)
    # Re-read the signature: the component may have changed during the walk.
    if _cache_component_signature(path) == signature:
        with _cache_scan_lock:
            _cache_verdicts[path] = (now + _CACHE_VERDICT_TTL_SECONDS, signature, witness, verdict)
    return verdict


def _model_cache_binds(workdir: str) -> dict[str, str]:
    """Inner cache subdirectory -> host directory, per component (HF_HUB_CACHE=/mnt/models is NOT /mnt/models/hub)."""
    binds: dict[str, str] = {}
    try:
        from utils.hf_cache_settings import get_hf_cache_paths

        paths = get_hf_cache_paths()
        home = os.path.abspath(str(paths.cache_home))
        resolved = {"hub": str(paths.hub_cache), "xet": str(paths.xet_cache)}
    except Exception:  # noqa: BLE001 - a launch never fails over a cache lookup
        logger.debug("could not resolve the configured Hugging Face cache", exc_info = True)
        user_home = os.path.expanduser("~")
        if not os.path.isabs(user_home):
            return binds
        home = os.path.join(user_home, _MODEL_CACHE_RELPATH)
        resolved = {}
    for name in _MODEL_CACHE_SUBDIRS:
        # Canonical: the mount table lists real paths.
        path = os.path.realpath(resolved.get(name) or os.path.join(home, name))
        if _within(path, workdir):
            continue
        # A cache at or above the Studio root would share auth/auth.db WRITABLE.
        if _too_broad_for_a_cache(path):
            logger.warning(
                "Not sharing the %s cache into the sandbox: %s is a home or "
                "system directory rather than a cache",
                name,
                path,
            )
            continue
        if _holds_studio_state(path):
            logger.warning(
                "Not sharing the %s cache into the sandbox: it is at or above "
                "Studio's own state directory",
                name,
            )
            continue
        # Writable caches need the workdir's host-channel checks, including nested bind mounts.
        hazard = _cache_hazard_within_deadline(name, path)
        if hazard is not None:
            logger.warning("Not sharing the %s cache into the sandbox: it %s", name, hazard)
            continue
        binds[name] = path
    return binds


def _make_cache_mountpoints(workdir: str, names: tuple[str, ...]) -> None:
    """Create missing mount points with O_NOFOLLOW (bwrap would follow a planted symlink on the HOST); never remove them."""
    levels = _MODEL_CACHE_RELPATH.split(os.sep)
    fds: list[int] = [os.open(workdir, os.O_RDONLY | os.O_DIRECTORY)]
    try:
        for name in (*levels, *names):
            try:
                os.mkdir(name, dir_fd = fds[-1])
            except FileExistsError:
                pass
            except OSError as exc:
                raise WorkdirUnsafeError(
                    f"the session workdir's model cache path cannot be prepared: {exc}"
                ) from exc
            # Verified even when not descended into: a bad leaf fails after Popen, where auto cannot fall back.
            try:
                opened = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd = fds[-1])
            except OSError as exc:
                raise WorkdirUnsafeError(
                    f"the session workdir's model cache path is not a plain directory: {name}"
                ) from exc
            if name in levels:
                fds.append(opened)
            else:
                os.close(opened)
    finally:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass


def prepare(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    bwrap = _trusted_bwrap_path()
    if not plan.argv:
        raise SandboxUnavailableError("a sandboxed launch needs a command to run")
    workdir, workdir_limitations = _validate_workdir(plan.workdir)
    # The caller's spelling (tools.py built the script path from it); the canonical form is the bind SOURCE.
    inner = os.path.abspath(plan.workdir)
    system_roots = _without_studio_state(
        tuple(path for path in _SYSTEM_ROOTS if os.path.isdir(path))
    )
    if os.path.isdir(_NIX_STORE) and _within(os.path.realpath(sys.executable), _NIX_STORE):
        system_roots += (_NIX_STORE,)
    runtime_paths = _runtime_read_paths(workdir, system_roots, inner)
    silent_roots = tuple(
        root
        for root in model_library_roots()
        if not _within(root, workdir) and not any(_within(root, r) for r in system_roots)
    )
    model_cache = _model_cache_binds(workdir)
    # A runtime under /tmp has to be restored after the tmpfs replaces it.
    tmp_runtime_paths = tuple(path for path in runtime_paths if _within(path, "/tmp"))
    workdir_runtime_paths = _runtime_paths_under(workdir)

    disable_userns = _bwrap_supports(bwrap, "--disable-userns")
    try:
        seccomp = sandbox_seccomp.filter_file(block_userns = not disable_userns)
    except RuntimeError as exc:
        raise SandboxUnavailableError(str(exc)) from exc
    try:
        identity_dir, passwd, group = _identity_files()
    except Exception:
        seccomp.close()
        raise
    try:
        argv: list[str] = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            # Every namespace except the network one: tool calls download models.
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup",
            # bwrap 0.6.1 lacks this; the seccomp filter refuses nested user namespaces there.
            *(("--disable-userns",) if disable_userns else ()),
            "--cap-drop",
            "ALL",
            "--seccomp",
            str(seccomp.fileno()),
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
        for root in system_roots:
            argv += ["--ro-bind-try", root, root]
        # -try: a missing model folder must not fail the launch.
        for root in silent_roots:
            argv += ["--ro-bind-try", root, root]
        trusted = tuple(p for p in _ETC_FILES_IF_TRUSTED if _trusted_system_file(p))
        for path in (*_ETC_FILES, *trusted, *_NETWORK_FILES):
            argv += ["--ro-bind-try", path, path]
        argv += ["--ro-bind", passwd, "/etc/passwd", "--ro-bind", group, "/etc/group"]
        for path in runtime_paths:
            if path not in tmp_runtime_paths:
                argv += ["--ro-bind", path, path]
        # Mount points for both spellings must exist before / goes read-only.
        argv += ["--dir", workdir]
        if inner != workdir:
            argv += ["--dir", inner]
        argv += ["--remount-ro", "/"]
        argv += ["--tmpfs", "/dev/shm", "--tmpfs", "/tmp"]
        for path in tmp_runtime_paths:
            argv += ["--ro-bind", path, path]
        argv += ["--bind", workdir, inner]
        if inner != workdir:
            argv += ["--bind", workdir, workdir]
        # Protect both runtime spellings AFTER both writable binds.
        for path in workdir_runtime_paths:
            argv += ["--ro-bind", path, path]
            if inner != workdir:
                argv += ["--ro-bind", path, inner + path[len(workdir) :]]
        argv += ["--chdir", inner]
        if model_cache:
            inner_cache = os.path.join(inner, _MODEL_CACHE_RELPATH)
            _make_cache_mountpoints(inner, tuple(model_cache))
            for name, host_path in model_cache.items():
                # --bind-try: a failed bind lands after Popen, where auto has no fallback.
                argv += ["--bind-try", host_path, os.path.join(inner_cache, name)]
            argv += ["--setenv", "HF_HOME", inner_cache]
        packages = os.path.join(inner, SESSION_PACKAGES_RELPATH)
        argv += [
            "--setenv",
            "HOME",
            inner,
            "--setenv",
            "TMPDIR",
            _tmpdir(plan, workdir),
            "--setenv",
            "PIP_TARGET",
            packages,
            "--setenv",
            "PYTHONPATH",
            _pythonpath(plan, packages),
            "--setenv",
            "PATH",
            _path(plan, packages),
            "--",
        ]
        argv += list(plan.argv)

        return PreparedSandboxLaunch(
            argv = tuple(argv),
            workdir = workdir,
            env = dict(plan.env),
            # --new-session covers the inner side only; tools.py's killpg still needs the outer setsid.
            preexec_fn = sandbox_landlock.with_abstract_scope(plan.preexec_fn),
            backend = BACKEND_NAME,
            pass_fds = (seccomp.fileno(),),
            owned_files = [seccomp],
            cleanup_paths = [identity_dir],
            timeout_seconds = plan.timeout_seconds,
            close_fds = plan.close_fds,
            terminate_descendants = plan.terminate_descendants,
            launch_limitations = workdir_limitations,
        )
    except Exception:
        seccomp.close()
        shutil.rmtree(identity_dir, ignore_errors = True)
        raise


def probe_argv(
    workdir: str,
    payload_argv: tuple[str, ...],
    env: dict[str, str] | None = None,
) -> PreparedSandboxLaunch:
    """Built through the real argv builder, so the probe cannot qualify a sandbox nothing runs."""
    return prepare(
        ToolLaunchPlan(
            argv = tuple(payload_argv),
            workdir = workdir,
            env = dict(env or {}),
            requested_mode = "required",
        )
    )
