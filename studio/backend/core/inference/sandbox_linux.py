# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The bubblewrap backend: what a Studio tool call on Linux actually runs inside.

The shape of the jail is a read-only view of the system the interpreter already
depends on, one writable directory (the session workdir), and no home. Not a
minimal container image: a tool call has to be able to run ``git``, import numpy
and shell out to the same binaries the user has, so /usr and friends are bound
whole, read-only, and the boundary that matters is the write side.

Two things are deliberately not confined, and both are named in ``LIMITATIONS``
rather than left for someone to discover. The network is open, because tool
calls pip-install and download models. And the model-data subdirectories of
``~/.cache/huggingface`` are bound read-write, because re-downloading gigabytes
of weights on every tool call is not a sandbox anyone would leave switched on.
The cache ROOT is not bound: the access token lives directly in it.

The bind list is built from parent directories, never from individual shared
objects. Enumerating an interpreter's ``.so`` files produces hundreds of binds
that still miss the one dlopen() reaches for at runtime; binding the directories
the interpreter reports is both smaller and more complete.
"""

from __future__ import annotations

import os
import shutil
import site
import subprocess
import sys
import sysconfig
import tempfile
from functools import lru_cache

from . import sandbox_seccomp
from .os_sandbox import (
    PROFILE_VERSION,
    PreparedSandboxLaunch,
    SandboxUnavailableError,
    ToolLaunchPlan,
    scan_workdir_for_host_channels,
)

BACKEND_NAME = "bubblewrap"
PROFILE_ID = f"linux-bwrap-{PROFILE_VERSION}"
LIMITATIONS = (
    # The sandbox has the host's network. A tool call that reaches a secret can
    # still send it; the filesystem boundary is the whole of what is claimed.
    "unrestricted_network",
    # /usr and friends are readable. Confidentiality of system files is not the
    # goal, and a jail without them cannot run the tools people call.
    "system_paths_readable",
    "model_cache_writable",
    # --dev builds a fresh /dev, so /dev/nvidia*, /dev/kfd and /dev/dri are gone.
    # A sandboxed tool call is CPU-only.
    "gpu_devices_hidden",
    # --unshare-pid puts the launch in its own PID namespace, so a backgrounded
    # process dies with the foreground command instead of outliving the call. It
    # is what "process_isolation" means and it is stricter than the unisolated
    # path, where tools.py sweeps such a descendant afterwards: output the
    # background job would have written after the leader exited is not collected.
    "detached_processes_die_with_the_call",
    "shared_kernel",
)

# Bound whole and read-only. Parent directories, deliberately: see the module docstring.
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
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
)
# /etc is a fresh directory in the jail, so the loader and the C library need
# their own files back by name or nothing dynamically linked starts.
_ETC_FILES = (
    "/etc/alternatives",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/localtime",
    "/etc/nsswitch.conf",
)
# Resolution and TLS trust. pip carries certifi; curl, git and urllib do not, and
# Debian keeps the bundle under /etc/ssl where Fedora and RHEL use /etc/pki.
_NETWORK_FILES = (
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/host.conf",
    "/etc/gai.conf",
    "/etc/services",
    "/etc/protocols",
    "/etc/ssl",
    "/etc/pki",
    "/etc/ca-certificates",
    "/etc/ca-certificates.conf",
    "/etc/crypto-policies",
)
# The one deliberate hole in the home mask. Model weights are gigabytes and a
# private empty cache per tool call would re-download them every time, which is
# how a sandbox gets turned off. Bound at the jail's own HOME so the default
# huggingface_hub location resolves to it, and HF_HOME is pinned to match: a
# host HF_HOME pointing somewhere unbound would fail on a read-only root.
# A dot directory on purpose: tools.py's _snapshot_workdir_files skips those, so
# the cache never presents as an artifact the model created.
_MODEL_CACHE_RELPATH = os.path.join(".cache", "huggingface")
# The DATA subdirectories only, never the cache root. huggingface_hub keeps the
# access token at $HF_HOME/token and $HF_HOME/stored_tokens, which tools.py
# already treats as credentials (_BYPASS_ENV_CRED_LOCATION_NAMES drops HF_HOME
# for exactly this reason). Binding the root and then pointing HF_HOME at it
# would put a live token at the first path a script looks in, inside a sandbox
# whose network is open by design. Anything not named here resolves to the
# session workdir, so a new cache file is written per-session instead of leaking
# a credential the next release happens to add.
_MODEL_CACHE_SUBDIRS = ("hub", "datasets", "modules", "xet", "assets")
# NixOS keeps glibc and every interpreter dependency here, so an interpreter from
# the store cannot dynamically link anything without it.
_NIX_STORE = "/nix/store"


def _within(path: str, root: str) -> bool:
    """Whether ``path`` is ``root`` or sits under it, lexically."""
    path, root = os.path.abspath(path), os.path.abspath(root)
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


@lru_cache(maxsize = 8)
def _bwrap_long_options(identity: tuple[str, int, int]) -> frozenset[str]:
    """Long options the installed bubblewrap accepts, read once from its usage text.

    Keyed by path and file identity so a package upgrade under a running Studio
    is re-read instead of answered from a verdict about the old binary.
    """
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
    """Every host mount point. Unreadable is a refusal, not an empty list.

    Read straight out of mountinfo without resolving anything: the kernel already
    reports mount points canonically, and calling realpath on each one would stat
    every path component of every mount on the host, on every launch. That blocks
    uninterruptibly on a stale NFS or sshfs mount and triggers automounts that
    were not otherwise in anyone's way.
    """
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


def _validate_workdir(workdir: str) -> str:
    """Canonicalise the session workdir, and refuse one that would carry the host in with it.

    The nested-mount leg is the bubblewrap-specific half: the workdir bind is
    recursive, so a mount under it comes along. The device-node and hard-link
    legs are the boundary both backends claim, so they live in ``os_sandbox``.
    """
    resolved = os.path.realpath(workdir)
    if not os.path.isdir(resolved) or os.path.dirname(resolved) == resolved:
        raise SandboxUnavailableError("the session workdir is not a safe canonical directory")
    for mount in _host_mount_points():
        if mount != resolved and _within(mount, resolved):
            raise SandboxUnavailableError(
                f"the session workdir contains a nested host mount: {mount}"
            )
    scan_workdir_for_host_channels(resolved)
    return resolved


def _runtime_read_paths(workdir: str, system_roots: tuple[str, ...]) -> tuple[str, ...]:
    """The interpreter roots this Python needs that the system roots do not already cover.

    Asked of the interpreter rather than taken from ``sys.path``, which carries
    whatever the caller inherited, and kept to directories so the set stays in
    the tens of entries no matter how large site-packages is.
    """
    # All four, because they are not two paths under two names. A uv-managed base
    # interpreter reports base_prefix as ``cpython-3.12.12-linux-x86_64-gnu`` and
    # base_exec_prefix as the ``cpython-3.12-...`` alias symlink beside it, and
    # lib-dynload -- every C extension in the standard library -- hangs off the
    # alias spelling alone.
    prefixes = (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)
    candidates: list[str] = [
        os.path.dirname(os.path.realpath(sys.executable)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_site"),
    ]
    # The subdirectories a Python installation lives in, never the prefix itself.
    # ``python -m venv .`` at a project root makes sys.prefix the project root, so
    # binding the prefix would hand the jail the entire tree -- sources, .git,
    # .env -- to reach one lib directory, which is the read side this backend
    # exists to keep closed.
    for prefix in prefixes:
        candidates.extend(
            os.path.join(prefix, name) for name in ("bin", "lib", "lib64", "pyvenv.cfg")
        )
    try:
        paths = sysconfig.get_paths()
        candidates.extend(
            paths[key] for key in ("stdlib", "platstdlib", "purelib", "platlib") if paths.get(key)
        )
    except (KeyError, OSError):
        pass
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
        # Both spellings: a venv reached through a symlink needs the link's own
        # path to exist inside the jail as well as the directory it lands on.
        for path in (os.path.abspath(candidate), os.path.realpath(candidate)):
            if path == os.path.sep:
                raise SandboxUnavailableError(
                    f"this interpreter would require binding the filesystem root: {candidate}"
                )
            if not os.path.exists(path):
                continue
            if any(_within(path, root) for root in (*system_roots, workdir, *selected)):
                continue
            selected.append(path)
    return tuple(selected)


def _identity_files() -> tuple[str, str, str]:
    """Synthesise passwd and group so getpwuid() works without the host account database.

    Every tool that asks who it is running as (git, pip, ssh, Python's own
    ``os.path.expanduser``) fails or picks up the host's users if /etc/passwd is
    either missing or bound straight through. One entry, this uid, no home.
    """
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
    """Where the jail's TMPDIR points: the caller's, when it is inside the workdir.

    tools.py puts TMPDIR at ``<workdir>/unsloth-tmp`` so a file a tool call
    writes through ``tempfile`` is still there when the call returns and is
    offered to the user as a download. The private /tmp here is a tmpfs that
    dies with the mount namespace, so pinning TMPDIR at it would silently drop
    every one of those files. It stays the fallback for a caller that named no
    temp directory, or one outside the only writable path in here.

    Canonicalised first, because the workdir is bound inside the jail under its
    realpath and only that spelling exists in there: a TMPDIR reached through a
    symlinked home would otherwise be set to a name nothing can open.
    """
    requested = plan.env.get("TMPDIR") or ""
    if not requested:
        return "/tmp"
    resolved = os.path.realpath(requested)
    return resolved if _within(resolved, workdir) else "/tmp"


def _model_cache_path(workdir: str) -> str | None:
    home = os.path.expanduser("~")
    if not os.path.isabs(home):
        return None
    path = os.path.join(home, _MODEL_CACHE_RELPATH)
    if not os.path.isdir(path) or _within(path, workdir):
        return None
    return path


def _make_cache_mountpoints(workdir: str, names: tuple[str, ...]) -> list[str]:
    """Create the cache mount points, and report exactly the ones this call made.

    bwrap creates a missing bind destination itself, but these sit under the
    workdir bind, so it creates them ON THE HOST and every chat is then left
    holding a .cache tree the user never made. Making them here instead means the
    launch knows which directories are its own and can take back precisely those,
    leaving a real ``.cache`` a tool call wrote alone.
    """
    levels = _MODEL_CACHE_RELPATH.split(os.sep)
    wanted = [os.path.join(workdir, *levels[: index + 1]) for index in range(len(levels))]
    wanted += [os.path.join(workdir, _MODEL_CACHE_RELPATH, name) for name in names]
    created: list[str] = []
    for path in wanted:
        try:
            os.mkdir(path)
        except FileExistsError:
            continue
        except OSError:
            break
        created.append(path)
    return created


def _reclaim_cache_mountpoints(created: list[str]) -> None:
    """Remove the mount points this launch made, innermost first, only while empty."""
    for path in reversed(created):
        try:
            os.rmdir(path)
        except OSError:
            return


def prepare(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    """Turn a launch plan into the bwrap argv that will run it."""
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise SandboxUnavailableError("bubblewrap (bwrap) is not installed on this host")
    if not plan.argv:
        raise SandboxUnavailableError("a sandboxed launch needs a command to run")
    workdir = _validate_workdir(plan.workdir)
    system_roots = tuple(path for path in _SYSTEM_ROOTS if os.path.isdir(path))
    if os.path.isdir(_NIX_STORE) and _within(os.path.realpath(sys.executable), _NIX_STORE):
        system_roots += (_NIX_STORE,)
    runtime_paths = _runtime_read_paths(workdir, system_roots)
    model_cache = _model_cache_path(workdir)
    # A runtime under /tmp has to be restored after the private tmpfs replaces it.
    tmp_runtime_paths = tuple(path for path in runtime_paths if _within(path, "/tmp"))

    disable_userns = _bwrap_supports(bwrap, "--disable-userns")
    try:
        seccomp = sandbox_seccomp.filter_file(block_userns = not disable_userns)
    except RuntimeError as exc:
        raise SandboxUnavailableError(str(exc)) from exc
    try:
        identity_dir, passwd, group = _identity_files()
    except Exception:
        # Nothing owns the descriptor until PreparedSandboxLaunch does, and a
        # leak here is worst exactly when it matters: under descriptor
        # exhaustion, where every retry made it worse.
        seccomp.close()
        raise
    # Everything from here on is owned by a PreparedSandboxLaunch that does not
    # exist yet, so this frame has to release it if the assembly raises.
    mountpoints: list[str] = []
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
            # bwrap 0.6.1 (Ubuntu 22.04) predates this; there the seccomp filter
            # refuses nested user namespaces instead.
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
        for path in (*_ETC_FILES, *_NETWORK_FILES):
            argv += ["--ro-bind-try", path, path]
        argv += ["--ro-bind", passwd, "/etc/passwd", "--ro-bind", group, "/etc/group"]
        for path in runtime_paths:
            if path not in tmp_runtime_paths:
                argv += ["--ro-bind", path, path]
        # The workdir mount point has to exist before the root goes read-only; the
        # writable bind onto it, and the private /tmp, come after.
        argv += ["--dir", workdir, "--remount-ro", "/"]
        argv += ["--tmpfs", "/dev/shm", "--tmpfs", "/tmp"]
        for path in tmp_runtime_paths:
            argv += ["--ro-bind", path, path]
        argv += ["--bind", workdir, workdir, "--chdir", workdir]
        if model_cache is not None:
            inner_cache = os.path.join(workdir, _MODEL_CACHE_RELPATH)
            mountpoints = _make_cache_mountpoints(workdir, _MODEL_CACHE_SUBDIRS)
            for name in _MODEL_CACHE_SUBDIRS:
                argv += [
                    "--bind-try",
                    os.path.join(model_cache, name),
                    os.path.join(inner_cache, name),
                ]
            argv += ["--setenv", "HF_HOME", inner_cache]
        argv += ["--setenv", "HOME", workdir, "--setenv", "TMPDIR", _tmpdir(plan, workdir), "--"]
        argv += list(plan.argv)

        return PreparedSandboxLaunch(
            argv = tuple(argv),
            workdir = workdir,
            env = dict(plan.env),
            # bwrap's --new-session covers the inner side only. tools.py kills a tool
            # call with killpg, so the outer setsid this carries still has to run.
            preexec_fn = plan.preexec_fn,
            backend = BACKEND_NAME,
            pass_fds = (seccomp.fileno(),),
            owned_files = [seccomp],
            cleanup_paths = [identity_dir],
            cleanup_callbacks = [lambda: _reclaim_cache_mountpoints(mountpoints)],
            timeout_seconds = plan.timeout_seconds,
            close_fds = plan.close_fds,
            terminate_descendants = plan.terminate_descendants,
        )
    except Exception:
        seccomp.close()
        shutil.rmtree(identity_dir, ignore_errors = True)
        _reclaim_cache_mountpoints(mountpoints)
        raise


def probe_argv(
    workdir: str,
    payload_argv: tuple[str, ...],
    env: dict[str, str] | None = None,
) -> PreparedSandboxLaunch:
    """Build a probe launch through the argv builder a real tool call uses.

    Returns the whole prepared launch rather than only the argv, because the
    probe has to inherit the seccomp descriptor and release the private identity
    directory afterwards. A probe that assembled its own argv would qualify a
    sandbox that nothing ever runs.
    """
    return prepare(
        ToolLaunchPlan(
            argv = tuple(payload_argv),
            workdir = workdir,
            env = dict(env or {}),
            requested_mode = "required",
        )
    )
