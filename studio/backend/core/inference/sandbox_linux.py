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
# "hub" is NOT excluded for the same reason, and that is a decision rather than
# an oversight: a remote-code model's own modeling_*.py lives in
# hub/models--*/snapshots/*, so a tool call can rewrite it and a later load that
# was given trust_remote_code will run it. Read-only would close that, and would
# also make every download inside a tool call re-fetch gigabytes into a
# directory that is thrown away, which is how a sandbox gets switched off. It
# stays writable on the same footing as model_cache_writable in LIMITATIONS:
# trust_remote_code is opt-in and off by default, and today, with no sandbox at
# all, a tool call can rewrite that file by absolute path with nothing in
# its way. Narrower than main, not a new hole. #5603 is what closes it.
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

    Judged on the RESOLVED path, not the spelling. A venv invoked through a
    symlinked path keeps that alias in sys.prefix -- measured on CPython 3.12,
    where <alias>/venv/bin/python reports sys.prefix = <alias>/venv, not the
    resolved form -- while the caller hands in the canonical workdir. A lexical
    test against either spelling alone rejects the other, and with an
    alias-valued sys.prefix that left nothing protected: a tool call overwrote
    the interpreter's sitecustomize through both spellings, under bubblewrap
    0.11 in a container.

    One that RESOLVES outside still gets no rule, which is the <workdir>/venv/lib
    symlinked at ~/.ssh case: nothing is bound at the far end, so inside the jail
    it dangles.
    """
    canonical_root = os.path.realpath(workdir)
    inside: list[str] = []
    # The interpreter FILE leads the list, not just <prefix>/bin: a standalone
    # build sits directly in its own prefix, so when that prefix is the workdir
    # none of the names below exist and this returned nothing at all. That one
    # is not a cosmetic gap -- sandbox_probe runs sys.executable on the HOST for
    # its positive control, so a replaced one is executed outside the jail.
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
        # Both PROTECTED, because keeping only the resolved one left a symlinked
        # entry -- an editable package pointing at a sibling in the same checkout
        # -- under the writable mount: its target was read-only but its NAME was
        # not, so a tool call could unlink it and put its own package there.
        #
        # The RESOLVED form is the containment test, because --ro-bind resolves
        # its source: binding an alias whose target is outside would mount that
        # outside directory at the alias path, the opposite of what this is for.
        # One that resolves out gets no rule and dangles inside the jail, which
        # is the documented answer. The WRITTEN form only ever adds a rule, never
        # licenses one, so an alias-spelled prefix -- sys.prefix keeps the alias
        # when the venv was invoked through one -- still protects its target.
        if _within(written, canonical_root) and not _within(resolved, canonical_root):
            if candidate is sys.executable:
                # Studio's own interpreter reachable through the tool call's
                # writable directory, with its content outside it. There is no
                # rule that both protects the name and keeps the target hidden,
                # and leaving it writable means the next probe runs whatever was
                # put there -- _host_positive_controls execs sys.executable on
                # the HOST, outside any sandbox.
                raise WorkdirUnsafeError(
                    f"the session workdir holds a link to the Python that runs Studio: {written}"
                )
            continue
        for path in (written, resolved):
            if _within(path, canonical_root) and path not in inside:
                inside.append(path)
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


def _runtime_read_paths(
    workdir: str,
    system_roots: tuple[str, ...],
    alias: str | None = None,
) -> tuple[str, ...]:
    """Asked of the interpreter, not ``sys.path``, which carries whatever the
    caller inherited.

    *alias* is the caller's spelling of the workdir when it differs from the
    canonical one. The as-written exclusion below has to see both: a venv reached
    through a symlinked workdir keeps the ALIAS in sys.prefix, which is not
    lexically beneath the canonical root, so <alias>/venv/lib was not excluded
    and the loop then bound whatever it resolved to -- a ~/.ssh behind it
    included, which is the exact case the exclusion exists to stop.
    """
    # All four: for a uv-managed interpreter base_prefix and base_exec_prefix are
    # different spellings, and lib-dynload hangs off the alias one alone.
    prefixes = (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)
    candidates: list[str] = [
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
    # LAST, and the file itself rather than its parent. Binding the parent read-
    # bound a whole home directory, .ssh and all, whenever a standalone build sat
    # directly in one -- /home/alice/python -- into a jail whose one claim is that
    # the home is not readable. Last, because a venv, conda or uv interpreter is
    # already inside the <prefix>/bin selected above and is then skipped by the
    # containment test below: those layouts keep exactly the binds they had, and
    # only the standalone one gains a bind of the single file it needs.
    # As WRITTEN, not resolved: Studio started through a user-level symlink keeps
    # that spelling in sys.executable and the plan's argv[0] IS that spelling, so
    # binding only the target left argv[0] absent inside the jail. The loop below
    # takes both spellings of every candidate, and a bind of the FILE creates its
    # parent as an empty directory rather than granting it.
    candidates.append(sys.executable)
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
        # <workdir>/venv/lib and then bind the ~/.ssh it was symlinked to. Both
        # spellings of the workdir, since sys.prefix may carry either.
        written = os.path.abspath(candidate)
        if any(_within(written, root) for root in (workdir, alias) if root):
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


# A wedged NFS or FUSE mount blocks in the syscall, not between syscalls, so
# cache_share_hazard's own deadline never gets to run and neither does isdir.
# The whole per-component inspection is therefore done on a worker whose WAIT is
# bounded rather than its work: a thread stuck in scandir cannot be killed, but
# it can be left behind. Dropping the component is the documented answer to a
# hazard anyway, so a mount we cannot inspect in time is simply not shared, and
# the launch proceeds re-downloading exactly as it did before the cache existed.
# Not a fork: this runs per launch from a threaded server, which is the shape
# that made the Landlock probe dangerous.
_CACHE_INSPECT_SECONDS = CACHE_SCAN_SECONDS + 2.0


def _inspect_cache_component(name: str, path: str) -> "str | None":
    """The hazard for one component, or a reason it could not be inspected."""
    if not os.path.isdir(path):
        return "is not a directory"
    nested = next((m for m in _host_mount_points() if m != path and _within(m, path)), None)
    if nested is not None:
        return f"contains a nested host mount: {nested}"
    return cache_share_hazard(path)


# path -> the worker a previous launch gave up on. A thread stuck in scandir on
# a wedged mount never returns, so without this every later launch started
# another one against the same path. Keyed on the THREAD rather than a clock:
# a timed expiry still let one through per interval, which on a permanently
# wedged mount is an unbounded leak with extra steps. The entry clears when the
# original worker finally finishes, so a mount that recovers is picked up again.
_cache_scan_pending: "dict[str, threading.Thread]" = {}
# Tool calls arrive on request threads, so every read and write of the map above
# is contended. Two races, both of which defeat the bookkeeping it exists for: a
# check-then-start that is not atomic lets a burst start one worker each against
# the same wedged path, and a check-then-delete lets one caller remove an entry
# the other is still holding, raising KeyError out of a launch -- which `auto`
# answers by dropping OS isolation and `required` by refusing.
_cache_scan_lock = threading.Lock()


def _cache_hazard_within_deadline(name: str, path: str) -> "str | None":
    answer: list[str | None] = []

    def inspect() -> None:
        try:
            answer.append(_inspect_cache_component(name, path))
        except Exception as exc:  # noqa: BLE001 - a launch never fails over this
            answer.append(f"could not be inspected: {exc}")

    with _cache_scan_lock:
        pending = _cache_scan_pending.get(path)
        if pending is not None:
            if pending.is_alive():
                return "was still being inspected when a previous launch gave up (a wedged mount?)"
            del _cache_scan_pending[path]
        worker = threading.Thread(target = inspect, name = f"unsloth-cache-scan-{name}", daemon = True)
        # Reserved and STARTED under the lock, so the entry a concurrent caller
        # finds is always a thread that is already running: reserving without
        # starting would leave is_alive() False and let that caller replace it.
        _cache_scan_pending[path] = worker
        worker.start()
    worker.join(_CACHE_INSPECT_SECONDS)
    if not answer:
        # The thread is left behind on purpose -- one blocked in scandir cannot be
        # killed -- and its entry stays, so no later launch starts a second one
        # against the same path while this one is still stuck.
        return f"could not be inspected within {_CACHE_INSPECT_SECONDS:.0f}s (a wedged mount?)"
    with _cache_scan_lock:
        # By identity: another caller may already have replaced it.
        if _cache_scan_pending.get(path) is worker:
            del _cache_scan_pending[path]
    return answer[0]


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
        if _within(path, workdir):
            continue
        # This bind is WRITABLE, so it is held to the workdir's rule: no IPC nodes
        # and no hard link to an inode named outside it. A component that fails is
        # dropped, never refused, so the worst case is the re-download every call
        # did before the cache was shared. The mount table is re-read for the same
        # reason _validate_workdir re-reads it: the shared scan's os.path.ismount
        # compares device numbers and misses a same-filesystem bind mount, which
        # this recursive WRITABLE bind would otherwise carry in.
        hazard = _cache_hazard_within_deadline(name, path)
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
    runtime_paths = _runtime_read_paths(workdir, system_roots, inner)
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
        if inner != workdir:
            argv += ["--bind", workdir, workdir]
        # After BOTH writable binds, and at BOTH spellings. A read-only mount
        # placed before the second bind is hidden by it, and one placed at a
        # spelling the jail has not bound yet has no mount point to land on:
        # bwrap then dies with "Can't mkdir parents ... Read-only file system"
        # and the launch fails outright. See _runtime_paths_under.
        for path in workdir_runtime_paths:
            argv += ["--ro-bind", path, path]
            if inner != workdir:
                argv += ["--ro-bind", path, inner + path[len(workdir) :]]
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
