# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The bubblewrap backend: a read-only system, one writable directory, no home.

Binds are parent directories, never individual shared objects: enumerating an
interpreter's ``.so`` files still misses the one dlopen() reaches at runtime.
"""

from __future__ import annotations

import os
import shutil
import stat
import site
import subprocess
import sys
import sysconfig
import tempfile
from functools import lru_cache

from loggers import get_logger

from . import sandbox_landlock, sandbox_seccomp
from .os_sandbox import (
    PROFILE_VERSION,
    SESSION_PACKAGES_RELPATH,
    PreparedSandboxLaunch,
    SandboxUnavailableError,
    ToolLaunchPlan,
    WorkdirUnsafeError,
    cache_share_hazard,
    editable_source_roots,
    scan_workdir_for_host_channels,
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
    # Abstract AF_UNIX sockets live in the shared network namespace, so without
    # the Landlock scope (Linux 6.12+) a launch reaches the session bus and X.
    *(() if sandbox_landlock.abstract_scope_supported() else ("host_abstract_sockets_reachable",)),
    # The same hole, through the filesystem rather than the abstract namespace. A
    # read-only bind does not stop connect(): the mount flag governs write(), and
    # a socket is reached with send() (Viro, LKML 2014, on MNT_READONLY; it is why
    # a read-only docker.sock is still a full Docker API). So a pathname socket
    # under a bound system root -- a service under /opt is the realistic one -- is
    # reachable from inside. Named rather than scanned: the read-only roots are
    # /usr and friends, and walking them on every tool call is not affordable.
    # Unlike the writable cache, this is not a regression against a host launch,
    # which could connect to the same socket directly.
    "host_pathname_sockets_reachable",
    "shared_kernel",
)

_SYSTEM_ROOTS = (
    "/usr/bin",
    "/usr/sbin",
    "/usr/lib",
    "/usr/lib64",
    # git keeps its helpers in /usr/libexec/git-core on Fedora and RHEL.
    "/usr/libexec",
    "/usr/local/bin",
    "/usr/local/lib",
    "/usr/local/lib64",
    "/usr/local/libexec",
    "/usr/local/sbin",
    "/usr/local/share",
    "/usr/share",
    "/opt",
    # Headers, for a pip install with no wheel that builds from source.
    "/usr/include",
    "/usr/local/include",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
)
# /etc is fresh in the jail; without these nothing dynamically linked starts.
_ETC_FILES = (
    "/etc/alternatives",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/localtime",
    "/etc/nsswitch.conf",
)
# Bound only when it passes _trusted_system_file. git reads it for a corporate
# proxy, a custom CA path or a URL rewrite, and /etc is fresh in the jail, so
# without it an otherwise valid clone fails against defaults. macOS already binds
# it; this is the Linux half.
_ETC_FILES_IF_TRUSTED = ("/etc/gitconfig",)
# Resolution and TLS trust. The PUBLIC halves one by one, never /etc/ssl or
# /etc/pki whole: both carry private keys beside the certificates, and a miss
# here breaks verification loudly where a miss in a mask list leaks a key quietly.
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
# Bound at the jail's own HOME with HF_HOME pinned to match: a host HF_HOME
# pointing somewhere unbound would fail on a read-only root.
_MODEL_CACHE_RELPATH = os.path.join(".cache", "huggingface")
# Never the cache root: the access token lives at $HF_HOME/token. "modules" is
# excluded because it holds the generated Python for a trust_remote_code model,
# so sharing it writably is a path to code a later unsandboxed load imports.
_MODEL_CACHE_SUBDIRS = ("hub", "datasets", "xet", "assets")
# NixOS keeps glibc here, so a store interpreter cannot link without it.
_NIX_STORE = "/nix/store"


def _trusted_system_file(path: str) -> bool:
    """A real, root-owned, non-user-writable regular file, reached without a symlink.

    The point of the checks is that this path is bound into a jail whose whole
    claim is that the user's home is not readable. An /etc/gitconfig that is a
    symlink into $HOME, or that the invoking user can rewrite, would carry
    whatever they aimed it at straight back in.
    """
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if not stat.S_ISREG(info.st_mode):
        return False  # a symlink or anything else is not followed
    return info.st_uid == 0 and not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


def _within(path: str, root: str) -> bool:
    path, root = os.path.abspath(path), os.path.abspath(root)
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


@lru_cache(maxsize = 8)
def _bwrap_long_options(identity: tuple[str, int, int]) -> frozenset[str]:
    """Keyed by file identity so a package upgrade under a running Studio is
    re-read, not answered from a verdict about the old binary."""
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
    """Every host mount point; unreadable is a refusal, not an empty list. Not
    resolved: realpath on each entry blocks on a stale NFS mount."""
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
    """Interpreter directories that sit INSIDE the session workdir.

    _runtime_read_paths drops these, correctly: binding them by name would follow
    a <workdir>/venv/lib symlinked at ~/.ssh straight back in. But dropping alone
    leaves them under the recursive WRITABLE workdir bind, so when Studio's own
    virtualenv lives beneath the workdir a tool call can write its site-packages
    or its interpreter, and the next server subprocess launched with
    sys.executable runs that code with the server's authority. They are re-bound
    read-only after the writable bind instead.

    Only when both spellings stay inside the workdir. One that RESOLVES outside is
    the symlink case, and it needs no rule: nothing is bound at the far end, so
    inside the jail it dangles.
    """
    inside: list[str] = []
    for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        for name in ("bin", "include", "lib", "lib64", "libexec", "pyvenv.cfg", "ssl"):
            candidate = os.path.join(prefix, name)
            absolute = os.path.abspath(candidate)
            if not _within(absolute, workdir) or not os.path.exists(absolute):
                continue
            if not _within(os.path.realpath(candidate), workdir):
                continue
            if absolute not in inside:
                inside.append(absolute)
    return tuple(inside)


def _validate_workdir(workdir: str) -> str:
    """The mount table is re-read here because the shared scan's ``os.path.ismount``
    compares device numbers and misses a same-filesystem bind mount, which is what
    the recursive workdir bind would carry in writable."""
    resolved = os.path.realpath(workdir)
    if not os.path.isdir(resolved) or os.path.dirname(resolved) == resolved:
        raise WorkdirUnsafeError("the session workdir is not a safe canonical directory")
    for mount in _host_mount_points():
        if mount != resolved and _within(mount, resolved):
            raise WorkdirUnsafeError(f"the session workdir contains a nested host mount: {mount}")
    scan_workdir_for_host_channels(resolved)
    return resolved


def _runtime_read_paths(workdir: str, system_roots: tuple[str, ...]) -> tuple[str, ...]:
    """Asked of the interpreter, not ``sys.path``, which carries whatever the
    caller inherited."""
    # All four: for a uv-managed interpreter base_prefix and base_exec_prefix are
    # different spellings, and lib-dynload hangs off the alias one alone.
    prefixes = (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)
    candidates: list[str] = [
        os.path.dirname(os.path.realpath(sys.executable)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_site"),
    ]
    # Subdirectories, never the prefix itself: ``python -m venv .`` at a project
    # root makes sys.prefix the project root. "ssl" and "libexec" are for a Conda
    # or Homebrew git-remote-https; "include" is Python.h for a source build.
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
    # An editable install keeps its code outside site-packages, so without this a
    # sandboxed `import unsloth` fails where the same environment imported it a
    # moment earlier. Added as CANDIDATES, so the workdir exclusion, the
    # filesystem-root refusal and the dual-spelling handling below all apply.
    candidates.extend(editable_source_roots())
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:
        pass  # a stripped virtualenv without the real site module
    multiarch = sysconfig.get_config_var("MULTIARCH")
    if multiarch:
        candidates.extend(os.path.join(prefix, "lib", multiarch) for prefix in prefixes)

    selected: list[str] = []
    for candidate in candidates:
        if not candidate or not os.path.isabs(candidate):
            continue
        # On the candidate as WRITTEN: checking only the resolved form would skip
        # <workdir>/venv/lib and then bind the ~/.ssh it was symlinked to.
        if _within(os.path.abspath(candidate), workdir):
            continue
        # Both spellings: a venv reached through a symlink needs the link's own
        # path bound as well as the directory it lands on.
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
    """Synthesise passwd and group so getpwuid() works without the host account
    database. One entry, this uid, no home."""
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
    """The caller's TMPDIR when it is inside the workdir: the private /tmp dies
    with the mount namespace, dropping what ``tempfile`` wrote. Containment is
    decided on the canonical form, but the ANSWER is the caller's spelling."""
    requested = plan.env.get("TMPDIR") or ""
    if not requested:
        return "/tmp"
    return requested if _within(os.path.realpath(requested), workdir) else "/tmp"


def _pythonpath(plan: ToolLaunchPlan, packages: str) -> str:
    """Appended, never prepended: an installed package must not shadow tools.py's
    sandbox_site shim."""
    inherited = plan.env.get("PYTHONPATH") or ""
    return os.pathsep.join(part for part in (inherited, packages) if part)


def _path(plan: ToolLaunchPlan, packages: str) -> str:
    """Caller's PATH plus the package target's ``bin``, LAST: that directory is
    writable by the tool call, so a planted binary must not shadow a bare command
    the approval logic treats as safe."""
    inherited = plan.env.get("PATH") or ""
    return os.pathsep.join(part for part in (inherited, os.path.join(packages, "bin")) if part)


def _model_cache_binds(workdir: str) -> dict[str, str]:
    """Inner cache subdirectory -> the host directory to share there.

    Each component is resolved separately because HF_HUB_CACHE=/mnt/models is NOT
    /mnt/models/hub.
    """
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
        path = os.path.abspath(resolved.get(name) or os.path.join(home, name))
        if not os.path.isdir(path) or _within(path, workdir):
            continue
        # This bind is WRITABLE, so it is held to the workdir's rule: no IPC nodes
        # and no hard link to an inode named outside it. A component that fails is
        # dropped, never refused, so the worst case is the re-download every call
        # did before the cache was shared.
        # The mount table, for the same reason _validate_workdir re-reads it: the
        # shared scan's os.path.ismount compares device numbers and misses a
        # same-filesystem bind mount, which this recursive WRITABLE bind would
        # otherwise carry in.
        nested = next((m for m in _host_mount_points() if m != path and _within(m, path)), None)
        hazard = f"contains a nested host mount: {nested}" if nested else cache_share_hazard(path)
        if hazard is not None:
            logger.warning("Not sharing the %s cache into the sandbox: it %s", name, hazard)
            continue
        binds[name] = path
    return binds


def _make_cache_mountpoints(workdir: str, names: tuple[str, ...]) -> None:
    """bwrap would create a missing destination ON THE HOST, following a
    ``.cache`` symlink an earlier call pointed at the user's home, so this walks
    with ``O_NOFOLLOW``. Nothing is removed afterwards, so two overlapping calls
    in one session cannot unlink each other's mount points."""
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
            # Verified even when not descended into: a leaf left as a file or
            # symlink fails --bind-try after Popen, where auto cannot fall back.
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
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise SandboxUnavailableError("bubblewrap (bwrap) is not installed on this host")
    if not plan.argv:
        raise SandboxUnavailableError("a sandboxed launch needs a command to run")
    workdir = _validate_workdir(plan.workdir)
    # The spelling the CALLER used, since tools.py built the scratch script path
    # from it. The canonical form stays the bind SOURCE and what is checked.
    inner = os.path.abspath(plan.workdir)
    system_roots = tuple(path for path in _SYSTEM_ROOTS if os.path.isdir(path))
    if os.path.isdir(_NIX_STORE) and _within(os.path.realpath(sys.executable), _NIX_STORE):
        system_roots += (_NIX_STORE,)
    runtime_paths = _runtime_read_paths(workdir, system_roots)
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
    # Owned by a PreparedSandboxLaunch that does not exist yet.
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
            # bwrap 0.6.1 (Ubuntu 22.04) predates this; the seccomp filter
            # refuses nested user namespaces there instead.
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
        trusted = tuple(p for p in _ETC_FILES_IF_TRUSTED if _trusted_system_file(p))
        for path in (*_ETC_FILES, *trusted, *_NETWORK_FILES):
            argv += ["--ro-bind-try", path, path]
        argv += ["--ro-bind", passwd, "/etc/passwd", "--ro-bind", group, "/etc/group"]
        for path in runtime_paths:
            if path not in tmp_runtime_paths:
                argv += ["--ro-bind", path, path]
        # Mount points must exist before the root goes read-only. Both spellings
        # need one, since bwrap cannot create either once / is read-only.
        argv += ["--dir", workdir]
        if inner != workdir:
            argv += ["--dir", inner]
        argv += ["--remount-ro", "/"]
        argv += ["--tmpfs", "/dev/shm", "--tmpfs", "/tmp"]
        for path in tmp_runtime_paths:
            argv += ["--ro-bind", path, path]
        argv += ["--bind", workdir, inner]
        # After the writable bind, so the server's own runtime is read-only even
        # when it lives under the workdir; see _runtime_paths_under.
        for path in workdir_runtime_paths:
            argv += ["--ro-bind", path, path]
        if inner != workdir:
            argv += ["--bind", workdir, workdir]
        argv += ["--chdir", inner]
        if model_cache:
            inner_cache = os.path.join(inner, _MODEL_CACHE_RELPATH)
            _make_cache_mountpoints(inner, tuple(model_cache))
            for name, host_path in model_cache.items():
                # --bind-try: it existed when this was resolved, and a failed
                # bind lands after Popen where auto has no fallback left.
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
            # --new-session covers the inner side only; tools.py kills with
            # killpg, so the plan's outer setsid still has to run.
            preexec_fn = sandbox_landlock.with_abstract_scope(plan.preexec_fn),
            backend = BACKEND_NAME,
            pass_fds = (seccomp.fileno(),),
            owned_files = [seccomp],
            cleanup_paths = [identity_dir],
            timeout_seconds = plan.timeout_seconds,
            close_fds = plan.close_fds,
            terminate_descendants = plan.terminate_descendants,
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
    """Built through the argv builder a real tool call uses, so the probe cannot
    qualify a sandbox nothing ever runs."""
    return prepare(
        ToolLaunchPlan(
            argv = tuple(payload_argv),
            workdir = workdir,
            env = dict(env or {}),
            requested_mode = "required",
        )
    )
