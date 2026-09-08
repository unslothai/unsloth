# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The macOS half of the tool sandbox: a Seatbelt profile driven by sandbox-exec.

The profile is ``(deny default)`` and grants back exactly what a Python or
Terminal tool call needs: the interpreter's own runtime paths, the system
frameworks and command-line tools, the session workdir, and a private tmp. The
user's home is not in that set, so ``~/.ssh``, ``~/.aws`` and the login Keychain
are unreadable even though the process still runs as the user.

The IP network is deliberately NOT confined, matching the Linux backend and what
``os_sandbox`` reports (``network_policy = "unrestricted"``): tool calls still
pip-install and download models, so a script that reaches a secret can still
send it. The filesystem boundary is the claim; IP egress is not. AF_UNIX is the
exception, and it belongs to the filesystem claim rather than to the network
one: a connect() to a host socket such as Docker's is a way out of the boundary
that no file rule governs, so outbound is filtered to the ip domain and the two
unix sockets a launch needs are named.

SBPL is deprecated and undocumented for third-party products, so the filesystem
and process rules are carried over from a profile that was iterated against a
real Mac rather than reasoned about. Three of them exist for reasons invisible
until they bite -- dual path spellings, ancestor metadata, and optional literals
-- and each is commented where it is implemented.

The network rules and the DNS/TLS mach services are the part that has no such
history: they replace an allowlist proxy that used to make them unnecessary. IP
egress is written in the broadest form that still expresses "unrestricted"
(``(remote ip "*:*")``, so TCP and UDP over v4 and v6), because a narrower one
that compiles but silently denies UDP would read as a DNS bug forever, and the
DNS socket carries two spellings for the same reason.
``test_profile_compiles_under_sandbox_exec`` is what checks them, and it only
runs on Darwin.
"""

from __future__ import annotations
import json
import os
import posixpath
import shutil
import site
import stat
import subprocess
import sys
import sysconfig
import tempfile
import threading

from .os_sandbox import (
    PROFILE_VERSION,
    SESSION_PACKAGES_RELPATH,
    PreparedSandboxLaunch,
    SandboxUnavailableError,
    ToolLaunchPlan,
    scan_workdir_for_host_channels,
)

BACKEND_NAME = "macos-seatbelt"
# Derived, not hardcoded: an execution record keyed on this must not read the
# same for a profile that was regenerated with different rules.
PROFILE_ID = f"macos-seatbelt-{PROFILE_VERSION}"
LIMITATIONS = (
    # SBPL has no supported grammar for third-party use and can change between
    # releases, so "preview" is the honest protection state for this backend.
    "deprecated_undocumented_sbpl",
    # The sandbox has the host's network, inbound as well as out. A tool call
    # that reaches a secret can still send it.
    "unrestricted_network",
    # No /proc and no pidfds, so a setsid/double-fork descendant that outlives
    # the leader cannot be swept the way the Linux backend sweeps one.
    "detached_descendant_cleanup_unverified",
    # POSIX shared memory has one namespace per host, and torch's segments and
    # the standard library's are allowed by name pattern rather than isolated.
    "pytorch_posix_shm_namespace_shared",
    # Named semaphores share that host namespace too, and unlike the shared
    # memory rules ipc-posix-sem is granted unfiltered: a launch can open or
    # unlink another same-user process's semaphore if it knows the name. Stated
    # rather than narrowed, because the allowlist would have to cover whatever
    # torch and OpenMP name theirs on a platform none of this can be run on, and
    # a wrong guess breaks multiprocessing instead of confining a filesystem.
    "posix_semaphore_namespace_shared",
    # kern.proc.pid. and kern.proc.pgrp. are in the sysctl allowlist, so a launch
    # can read kinfo_proc metadata for host processes even though process-info*
    # is scoped to the same sandbox. They stay because that sysctl is how ps
    # works, and a Terminal call running ps is ordinary; Seatbelt has no PID
    # namespace to hide the host the way the Linux backend does. Named here so
    # the record does not read as though it did.
    "host_process_metadata_readable",
    # (deny default) grants only RootDomainUserClient, and Metal needs the GPU's
    # IOAccelerator and AGX user clients, so torch's MPS backend cannot initialise
    # a device in here. Named rather than granted: those user clients are a large
    # authority to hand a tool call on a platform nothing here can test, and the
    # Linux backend already treats a CPU-only jail as the trade worth taking.
    "gpu_devices_hidden",
)

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

# Where `pip install` writes, relative to the session workdir. Shared spelling
# with the Linux backend so a chat behaves the same on both.
PACKAGE_TARGET_RELPATH = SESSION_PACKAGES_RELPATH

_READ_ROOTS = (
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
    # Name resolution. getaddrinfo consults both before it ever asks
    # mDNSResponder, and /etc/resolv.conf is itself a symlink into /var/run,
    # which is why the spelling walk below matters even for a plain file.
    "/private/etc/hosts",
    "/private/etc/resolv.conf",
    "/System/Library/CoreServices/.SystemVersionPlatform.plist",
    "/System/Library/CoreServices/SystemVersion.plist",
    # /usr/bin/git and friends are xcode-select shims that resolve the developer
    # directory through this link and exec the real tool from it; without these
    # every git call inside the sandbox ends with "See man xcode-select".
    "/private/var/db/xcode_select_link",
    "/Library/Developer/CommandLineTools",
    "/Applications/Xcode.app/Contents/Developer",
)
# TLS trust. The network is on, so these are unconditional: /private/etc/ssl is
# what OpenSSL reads, and the rest is what Security.framework needs to evaluate
# a server certificate. These are the SYSTEM keychains and the system trust
# settings; the login keychain lives in the user's home and stays unreadable.
_TLS_TRUST_PATHS = (
    "/private/etc/ssl",
    "/System/Library/Keychains",
    "/Library/Keychains",
    "/System/Library/Security",
    "/private/var/db/mds",
    "/Library/Preferences/com.apple.security.plist",
    "/Library/Preferences/com.apple.security.revocation.plist",
)
# Homebrew and /usr/local: PATH inside the sandbox carries /usr/local/bin, so a
# tool found through PATH would fail process-exec without these. Read-only, and
# only the trees that exist on the host.
_OPTIONAL_READ_ROOTS = (
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
# HAZARD 3, optional literals. Files git and other tools probe on every run and
# may legitimately be absent. Under (deny default) an absent file yields EPERM
# rather than ENOENT, and git reports that as "unable to access
# '/etc/gitconfig'" and aborts. The existence-filtered path rules cannot carry
# these, so they are allowed whether or not they exist.
_OPTIONAL_READ_LITERALS = (
    "/etc/gitconfig",
    "/etc/gitattributes",
    # A Homebrew git is built with its own prefix as the system config path, so
    # it reads these instead of /etc and aborts the same way. Both prefixes are
    # listed because the binary on PATH decides which one, not the host's
    # architecture: an Intel-built git under Rosetta uses /usr/local.
    "/usr/local/etc/gitconfig",
    "/usr/local/etc/gitattributes",
    "/opt/homebrew/etc/gitconfig",
    "/opt/homebrew/etc/gitattributes",
    # xcrun's license check: the Xcode shims refuse ("You have not agreed to the
    # Xcode license agreements") when they cannot read the system-wide record.
    "/Library/Preferences/com.apple.dt.Xcode.plist",
)
# Exec-denied even though they sit inside readable system directories. These are
# the ambient-authority tools: each hands work to a daemon that is NOT in the
# sandbox and would run it unconfined.
_DENIED_EXECUTABLES = (
    "/usr/bin/open",
    "/usr/bin/osascript",
    "/usr/bin/security",
    "/bin/launchctl",
    "/usr/bin/sandbox-exec",
)
_DEVICES = ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom")
# mDNSResponder's socket. connect() to a unix socket is a network-outbound
# operation in Seatbelt, not a file operation, so a file rule cannot grant it.
_MDNSRESPONDER_SOCKET = "/private/var/run/mDNSResponder"
# Certificate evaluation, name resolution and proxy configuration only.
# com.apple.SecurityServer is deliberately ABSENT: with it a sandboxed tool can
# read the login Keychain through Security.framework even though
# /usr/bin/security is exec-denied, which turns an unreadable ~/Library/Keychains
# into a readable one. Codex's network profile does grant it; we do not, and
# test_sandbox_macos.py fails if it ever reappears.
_MACH_SERVICES = (
    "com.apple.system.opendirectoryd.libinfo",
    "com.apple.PowerManagement.control",
    "com.apple.trustd",
    "com.apple.trustd.agent",
    "com.apple.ocspd",
    "com.apple.dnssd.service",
    "com.apple.networkd",
    "com.apple.bsd.dirhelper",
    "com.apple.SystemConfiguration.configd",
    "com.apple.SystemConfiguration.DNSConfiguration",
)
# What platform/os/torch read at import. An omission here is a silent breakage
# (empty os.cpu_count(), a torch that cannot size its thread pool), not a denial
# anyone sees, so the list is generous.
# fmt: off
_SYSCTL_NAMES = (
    "hw.activecpu", "hw.busfrequency_compat", "hw.byteorder", "hw.cacheconfig",
    "hw.cachelinesize_compat", "hw.cpufamily", "hw.cpufrequency_compat", "hw.cputype",
    "hw.l1dcachesize_compat", "hw.l1icachesize_compat", "hw.l2cachesize_compat",
    "hw.l3cachesize_compat", "hw.logicalcpu", "hw.logicalcpu_max", "hw.machine", "hw.memsize",
    "hw.model", "hw.ncpu", "hw.nperflevels", "hw.packages", "hw.pagesize", "hw.pagesize_compat",
    "hw.physicalcpu", "hw.physicalcpu_max", "hw.tbfrequency_compat", "hw.vectorunit",
    "kern.argmax", "kern.hostname", "kern.maxfilesperproc", "kern.maxproc",
    "kern.osproductversion", "kern.osrelease", "kern.ostype", "kern.osvariant_status",
    "kern.osversion", "kern.secure_kernel", "kern.sysv.semmns", "kern.usrstack64", "kern.version",
    "machdep.cpu.brand_string", "sysctl.proc_cputype", "vm.loadavg",
)
_SYSCTL_PREFIXES = (
    "hw.optional.arm.", "hw.optional.armv8_", "hw.perflevel", "kern.proc.pgrp.", "kern.proc.pid.",
)
# fmt: on

_developer_paths_cache: tuple[str, ...] | None = None
_developer_paths_lock = threading.Lock()


# ── availability ─────────────────────────────────────────────────────


def available() -> tuple[bool, str]:
    """Whether this host has a usable Seatbelt launcher, as ``(ok, reason)``.

    Presence and executability only. That a profile actually loads and confines
    anything is the live probe's verdict, not this one -- an installed
    sandbox-exec on a host that rejects the profile looks identical to a working
    one until something tries it.
    """
    try:
        info = os.stat(SANDBOX_EXEC, follow_symlinks = False)
    except OSError as exc:
        return False, f"the system Seatbelt launcher is unavailable: {exc}"
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        return False, f"{SANDBOX_EXEC} is not a root-owned, non-user-writable regular file"
    if not os.access(SANDBOX_EXEC, os.X_OK):
        return False, f"{SANDBOX_EXEC} is not executable"
    return True, "the system Seatbelt launcher is present and executable"


# ── path plumbing ────────────────────────────────────────────────────


def _validated(path: str) -> str:
    if not path or not posixpath.isabs(path) or any(c in path for c in "\0\n\r"):
        raise SandboxUnavailableError(
            f"Seatbelt paths must be absolute and free of NUL/newline: {path!r}"
        )
    return path


def _within(path: str, root: str) -> bool:
    """Whether ``path`` is ``root`` or sits under it, by whole path components."""
    return path == root or path.startswith(root.rstrip("/") + "/")


def _rule(operations: str, filters: list[str]) -> str:
    """One SBPL rule, refusing to emit one that lost all of its filters.

    A rule with no filter is UNCONDITIONAL in SBPL, so ``(allow file-write* )``
    -- what an existence-filtered path list renders as once every path in it has
    gone -- silently grants the whole filesystem after ``(deny default)``. That
    is the one failure in this file that turns the sandbox inside out. The
    unconditional rules this profile does want are written out literally, so
    reaching here with nothing left always means a list collapsed, and a refused
    launch is the right answer either way: an empty DENY is not inert either, it
    denies the operation outright and nothing would run.
    """
    if not filters:
        raise SandboxUnavailableError(
            f"the Seatbelt profile would apply {operations!r} unconditionally: no paths survived"
        )
    return f"({operations} " + " ".join(filters) + ")"


def _sbpl_spellings(path: str, *, resolve: bool = True) -> tuple[str, ...]:
    """Every spelling of ``path`` a tool might use, deduplicated, order preserved.

    HAZARD 1, dual path spellings. /etc, /tmp and /var are symlinks into
    /private. Seatbelt evaluates the spelling it was GIVEN, so a rule written
    only against the canonical /private form EPERMs a tool that opens
    /etc/gitconfig before the canonical rule is ever consulted -- and a rule
    written only against the short form fails for anything that resolved the
    path first. Both directions are generated here rather than trusting
    realpath, so the pair is complete on a host where the symlink is missing and
    on a non-darwin host running the tests.

    ``resolve = False`` drops the symlink target, and is how a read allowance
    avoids following a host's symlink somewhere it was never meant to reach: an
    /etc/gitconfig symlinked into ~/dotfiles would otherwise put a home path in
    the read set. A denial keeps the target, because covering more spellings is
    the safe direction for a deny.
    """
    selected = [posixpath.abspath(_validated(path))]
    if resolve:
        # The canonical form is validated too, not just the input: it comes from
        # symlink targets on disk, and a component with a newline in it would
        # end up inside a quoted SBPL string.
        canonical = _validated(os.path.realpath(path))
        if canonical not in selected:
            selected.append(canonical)
    for spelling in tuple(selected):
        for short in ("/etc", "/tmp", "/var"):
            if spelling == short or spelling.startswith(short + "/"):
                selected.append("/private" + spelling)
            private = "/private" + short
            if spelling == private or spelling.startswith(private + "/"):
                selected.append(spelling[len("/private") :])
    return tuple(dict.fromkeys(selected))


def _path_filters(paths: tuple[str, ...]) -> list[str]:
    """``(literal ...)`` for every spelling, plus ``(subpath ...)`` for directories."""
    filters: list[str] = []
    # Keyed by kind as well as spelling: a file and a directory can share a
    # resolved spelling, and a single set would let the file's literal suppress
    # the directory's subpath and silently shrink the rule.
    seen: set[tuple[str, str]] = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        kinds = ("literal", "subpath") if os.path.isdir(path) else ("literal",)
        for spelling in _sbpl_spellings(path):
            encoded = json.dumps(spelling)
            for kind in kinds:
                if (kind, encoded) not in seen:
                    seen.add((kind, encoded))
                    filters.append(f"({kind} {encoded})")
    return filters


def _literal_filters(paths: tuple[str, ...], *, resolve: bool = True) -> list[str]:
    """``(literal ...)`` for every spelling, existence not required.

    Used for the optional read literals (hazard 3) and for the exec denials,
    where filtering by existence would silently drop a rule -- and for a DENY,
    dropping the rule is the wrong direction to fail in.
    """
    filters: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for spelling in _sbpl_spellings(path, resolve = resolve):
            encoded = json.dumps(spelling)
            if encoded not in seen:
                seen.add(encoded)
                filters.append(f"(literal {encoded})")
    return filters


def _ancestor_filters(spellings: tuple[str, ...]) -> list[str]:
    """``(literal ...)`` for every ancestor directory of every spelling given.

    Takes spellings rather than paths so the caller decides whether symlinks
    were followed: the ancestors of a read literal must not name a home
    directory just because the host symlinked /etc/gitconfig into one.

    HAZARD 2, ancestor metadata. Resolving /a/b/c stats /a and /a/b on the way
    down, so a readable leaf whose ancestors are not stat-able fails at an
    intermediate component with EPERM. These feed file-read-metadata only: the
    directories themselves stay unlistable and unreadable.

    posixpath, not os.path: these are profile paths, always POSIX. On Windows
    ntpath.dirname("/") returns "/" while os.path.sep is a backslash, and the
    rstrip-then-walk this replaced never reached its stop condition.
    """
    filters: list[str] = []
    seen: set[str] = set()
    for spelling in spellings:
        current = posixpath.dirname(_validated(spelling))
        while current:
            encoded = json.dumps(current)
            if encoded not in seen:
                seen.add(encoded)
                filters.append(f"(literal {encoded})")
            parent = posixpath.dirname(current)
            if parent == current:
                break
            current = parent
    return filters


def _developer_paths() -> tuple[str, ...]:
    """The active developer directory, resolved once through xcode-select.

    /usr/bin/git and the other shims call xcselect to find the developer
    directory and exec the real tool from it, so a profile that cannot see that
    directory ends every such call with "See man xcode-select". The directory is
    versioned on many hosts (/Applications/Xcode_16.4.app) and the static list
    above cannot name it.
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
                    encoding = "utf-8",
                    timeout = 10,
                    check = False,
                )
                candidate = result.stdout.strip()
                if result.returncode == 0 and candidate and os.path.isdir(candidate):
                    for spelling in (os.path.realpath(candidate), candidate):
                        if spelling not in found:
                            found.append(spelling)
                    # xcselect validates the developer directory against the
                    # enclosing app bundle (Info.plist, version.plist), so a
                    # Contents/Developer inside an .app needs the bundle too.
                    marker = ".app/Contents/Developer"
                    for spelling in list(found):
                        if spelling.endswith(marker):
                            bundle = spelling[: -len(marker) + len(".app")]
                            if os.path.isdir(bundle) and bundle not in found:
                                found.append(bundle)
            except (OSError, subprocess.SubprocessError):
                pass
        _developer_paths_cache = tuple(found)
        return _developer_paths_cache


def runtime_read_paths(workdir: str | None = None) -> tuple[str, ...]:
    """The interpreter's own runtime roots -- never arbitrary inherited sys.path.

    Same shape as the Linux backend's bind list: whatever the selected
    interpreter needs to start and import, and nothing that merely happens to be
    on sys.path because the parent process put it there.

    *workdir* is excluded by ORIGIN, not by where a path resolves to. The workdir
    is the one place a tool call can write, so a runtime path that starts there
    points wherever the last call pointed it: dropping only the resolved form
    would skip ``<workdir>/venv/lib`` and then grant file-read* on the ``~/.ssh``
    it was symlinked to.
    """
    candidates: list[str] = [
        os.path.dirname(os.path.realpath(sys.executable)),
        # The code-interpreter path shim, loaded via PYTHONPATH in every
        # sandboxed launch; unreadable means every tool call dies at startup.
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_site"),
    ]
    # The subdirectories a Python installation lives in, never the prefix itself
    # -- same rule as the Linux backend, for the same reason. ``python -m venv .``
    # at a project root makes sys.prefix the project root, so granting file-read*
    # on the prefix would hand the sandbox the whole tree: sources, .git and .env,
    # to reach one lib directory. That is the read side this backend exists to
    # keep closed, and the network is open, so a readable .env is an exportable one.
    #
    # All four prefixes, because they are not two paths under two names. A
    # uv-managed base interpreter reports base_prefix as
    # ``cpython-3.12.12-macos-aarch64-none`` and base_exec_prefix as the
    # ``cpython-3.12-...`` alias symlink beside it, and lib-dynload -- every C
    # extension in the standard library -- hangs off the alias spelling.
    for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        candidates.extend(
            posixpath.join(prefix, name)
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
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:
        # Some virtualenv layouts omit getsitepackages; sysconfig covers them.
        pass

    selected: list[str] = []
    for candidate in candidates:
        if not candidate or not posixpath.isabs(candidate):
            continue
        if workdir is not None and _within(posixpath.abspath(candidate), workdir):
            continue
        # Both spellings: a venv reached through a symlink needs the link's own
        # path in the read set as well as the directory it lands on. The
        # /private pairing happens later, in _sbpl_spellings.
        for path in (posixpath.abspath(candidate), os.path.realpath(candidate)):
            # "/" and "/usr" as read ROOTS would hand back most of the host. A
            # system interpreter reports one of them as its prefix; the pieces
            # it actually needs are named individually in _READ_ROOTS.
            if path in ("/", "/usr") or not os.path.exists(path):
                continue
            if path not in selected and not any(_within(path, root) for root in selected):
                selected.append(path)
    return tuple(selected)


# ── the profile ──────────────────────────────────────────────────────


def build_profile(
    *,
    workdir: str,
    private_tmp: str,
    runtime_paths: tuple[str, ...],
    developer_paths: tuple[str, ...] = (),
) -> str:
    """Render the SBPL text for one launch.

    Reads the filesystem -- existence and directory-ness decide which rules are
    emitted -- and nothing else: the xcode-select lookup is done by the caller
    and passed in, so this writes nothing, spawns nothing, starts no sandbox,
    and a test can assert on the text on any platform.
    """
    readable_paths = (
        *_READ_ROOTS,
        *_TLS_TRUST_PATHS,
        *_OPTIONAL_READ_ROOTS,
        *developer_paths,
        *_DEVICES,
        *runtime_paths,
        workdir,
        private_tmp,
    )
    read_filters = ['(literal "/")', *_path_filters(readable_paths)]
    # Ancestors of the optional literals are included by name: they are not in
    # readable_paths (they may not exist) but their parents still need stat. So
    # does mDNSResponder's socket -- connecting to it resolves /private/var/run
    # on the way, and that lookup is checked before the network rule is. The
    # literals keep resolve = False here for the same reason as below.
    metadata_filters = _ancestor_filters(
        tuple(
            spelling
            for path in readable_paths + (_MDNSRESPONDER_SOCKET,)
            for spelling in _sbpl_spellings(path)
        )
        + tuple(
            spelling
            for path in _OPTIONAL_READ_LITERALS
            for spelling in _sbpl_spellings(path, resolve = False)
        )
    )
    write_filters = _path_filters((workdir, private_tmp))
    device_filters = _path_filters(_DEVICES)
    # The workdir as well as the private tmp: TMPDIR points into the workdir (see
    # prepare), so multiprocessing's listener socket is created there, and an
    # AF_UNIX bind is network-bind in Seatbelt rather than a file operation. It
    # grants nothing new -- the process can already create that socket, because
    # the whole workdir is writable.
    tmp_subpaths = " ".join(
        f"(subpath {json.dumps(spelling)})"
        for path in (private_tmp, workdir)
        for spelling in _sbpl_spellings(path)
    )
    mdns_filters = " ".join(_literal_filters((_MDNSRESPONDER_SOCKET,)))
    # resolve = False: a host that symlinks /etc/gitconfig into the user's home
    # must not turn a read allowance for a config file into one for a home path.
    optional_filters = _literal_filters(_OPTIONAL_READ_LITERALS, resolve = False)
    sysctl_filters = [
        *(f"(sysctl-name {json.dumps(name)})" for name in _SYSCTL_NAMES),
        *(f"(sysctl-name-prefix {json.dumps(name)})" for name in _SYSCTL_PREFIXES),
    ]
    lines = [
        "(version 1)",
        "(deny default)",
        # Signals and process inspection are scoped to descendants in the same
        # Seatbelt instance, so a tool cannot poke at Studio or at the user's
        # other processes.
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow signal (target same-sandbox))",
        "(allow process-info* (target same-sandbox))",
        _rule("deny process-exec", _literal_filters(_DENIED_EXECUTABLES)),
        _rule("allow file-read-metadata", metadata_filters),
        _rule("allow file-read* file-test-existence", read_filters),
        _rule("allow file-read* file-test-existence", optional_filters),
        _rule("allow file-map-executable", read_filters),
        _rule("allow file-write*", write_filters),
        _rule("allow file-read* file-test-existence file-write-data", device_filters),
        # Every descriptor, not only the standard three: bash process substitution
        # (`diff <(sort a) <(sort b)`) hands the child /dev/fd/63, and denying it
        # fails a command that works unisolated and on the Linux backend, where
        # --dev gives the jail a whole /dev/fd. Opening /dev/fd/N is a dup of a
        # descriptor the process already holds, so this grants no new authority;
        # /dev/fd/0 is the launch's stdin, which tools.py pins to DEVNULL.
        '(allow file-read* (regex #"^/dev/fd/[0-9]+$"))',
        '(allow file-write* (regex #"^/dev/fd/[0-9]+$"))',
        _rule("allow file-ioctl", device_filters),
        "(allow ipc-posix-sem)",
        # write-data included, like the two rules below it: libomp does not only
        # create and unlink its registration segment, it writes the registration
        # into it, and OpenMP initialisation fails on the denial. The live probe
        # never loads an OpenMP workload, so nothing here would have caught it.
        "(allow ipc-posix-shm-read-data ipc-posix-shm-write-create "
        "ipc-posix-shm-write-data ipc-posix-shm-write-unlink "
        '(ipc-posix-name-regex #"^/__KMP_REGISTERED_LIB_[0-9]+$"))',
        "(allow ipc-posix-shm-read-data ipc-posix-shm-write-create "
        "ipc-posix-shm-write-data ipc-posix-shm-write-unlink "
        '(ipc-posix-name-regex #"^/torch_[0-9]+_[0-9]+_[0-9]+$"))',
        # multiprocessing.shared_memory.SharedMemory names its segment
        # "/psm_" + token_hex(4) (CPython's _SHM_NAME_PREFIX), so without this
        # the standard library's own shared memory raises on a backend the probe
        # advertised as usable -- its multiprocessing control only exercises
        # descriptor passing. Same host-wide namespace the torch rule already
        # accepts, and named in LIMITATIONS as such.
        "(allow ipc-posix-shm-read-data ipc-posix-shm-write-create "
        "ipc-posix-shm-write-data ipc-posix-shm-write-unlink "
        '(ipc-posix-name-regex #"^/psm_[0-9a-f]+$"))',
        # The IP network is NOT confined here -- see the module docstring, and the
        # "unrestricted_network" limitation the record carries. system-socket is a
        # separate operation because it gates socket() itself, and it is open
        # rather than domain-filtered because a configd-backed resolver creates
        # route and AF_SYSTEM sockets: that admits raw and kernel-control sockets
        # too, which is authority over the network, never over the filesystem
        # this profile is confining. network-inbound is included for parity with
        # the Linux backend, where the sandbox shares the host's netns and
        # inbound cannot be withheld.
        "(allow system-socket)",
        # IP only, for the same reason outbound is: an unfiltered network-bind
        # also covers AF_UNIX, and a bind() creates a socket at a path that no
        # file rule governs, so a launch could put one anywhere the user can
        # write. The workdir and private tmp binds it really needs are the
        # named rule below, which an unconditional allow above it would make
        # meaningless.
        '(allow network-bind (local ip "*:*"))',
        "(allow network-inbound)",
        # Outbound is the one that is filtered, to "ip" rather than to nothing.
        # An unfiltered network-outbound also covers AF_UNIX, and a connect() to a
        # unix socket is governed by NO file rule, so on a Mac running Docker
        # Desktop a tool call could reach /var/run/docker.sock and start an
        # unconfined container with host bind mounts: a way out of the whole
        # filesystem boundary, through the operation this profile leaves open on
        # purpose. "*:*" is host and port wildcards, so TCP and UDP, v4 and v6,
        # are all still unrestricted -- the "unrestricted" that was meant.
        '(allow network-outbound (remote ip "*:*"))',
        # The AF_UNIX destinations a launch actually needs, now that the rule
        # above no longer covers them. multiprocessing's listener lives under
        # TMPDIR and mDNSResponder's socket is how libsystem_info resolves names,
        # and neither is reachable through a file rule.
        f"(allow network-bind (local unix-socket {tmp_subpaths}))",
        f"(allow network-outbound (remote unix-socket {tmp_subpaths}))",
        # Both spellings for mDNSResponder. A bare path filter and a
        # (remote unix-socket ...) filter are both accepted for this operation,
        # and which one a given macOS release matches is exactly the sort of thing
        # no test here can answer; a missed DNS socket would read as a name
        # resolution bug on every Mac, so the redundant rule is worth its line.
        f"(allow network-outbound {mdns_filters})",
        f"(allow network-outbound (remote unix-socket {mdns_filters}))",
        _rule("allow sysctl-read", sysctl_filters),
        '(allow iokit-open (iokit-registry-entry-class "RootDomainUserClient"))',
        "(allow mach-lookup\n"
        + "\n".join(f"  (global-name {json.dumps(name)})" for name in _MACH_SERVICES)
        + ")",
    ]
    return "\n".join(lines) + "\n"


def _tmpdir(env: dict[str, str], workdir: str, private_tmp: str) -> str:
    """Where the sandbox's TMPDIR points: the caller's, when it is inside the workdir.

    tools.py puts TMPDIR at ``<workdir>/unsloth-tmp`` so a file a tool call
    writes through ``tempfile`` is still there when the call returns and is
    offered to the user as a download. ``private_tmp`` is deleted by
    ``PreparedSandboxLaunch.cleanup`` the moment the call ends, so pinning TMPDIR
    at it would silently drop every one of those files. It stays the fallback for
    a caller that named no temp directory, or one outside the writable set.
    """
    requested = env.get("TMPDIR") or ""
    if requested and _within(posixpath.abspath(requested), workdir):
        return requested
    return private_tmp


def _sandbox_environment(env: dict[str, str], workdir: str, private_tmp: str) -> dict[str, str]:
    """HOME and TMPDIR must point inside the sandbox, or the first write fails.

    The real HOME is unreadable under this profile and the per-user /var/folders
    tmp is unwritable, so anything that calls tempfile or reads a dotfile has to
    be pointed somewhere it is allowed to go. DYLD_* is dropped: it is the
    dynamic-linker injection surface and nothing a tool call needs.
    """
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
    tmpdir = _tmpdir(env, workdir, private_tmp)
    # Same reasoning as the Linux backend: the interpreter's site-packages is not
    # in the write set, so `pip install X` would fail on a permission error. It
    # goes to a session-local target on PYTHONPATH instead, appended so a package
    # installed in here cannot shadow the sandbox_site startup shim.
    packages = posixpath.join(workdir, PACKAGE_TARGET_RELPATH)
    sanitized.update(
        {
            "HOME": workdir,
            "TMPDIR": tmpdir,
            "TMP": tmpdir,
            "TEMP": tmpdir,
            "XDG_RUNTIME_DIR": private_tmp,
            "PIP_TARGET": packages,
            # <target>/bin is where pip writes a console entry point, so without
            # it `pip install black && black .` installs and then fails. Last on
            # PATH: the directory is writable by the tool call, and behind every
            # system directory a planted binary cannot shadow a bare command the
            # approval logic treats as safe.
            "PATH": os.pathsep.join(
                part for part in (env.get("PATH") or "", posixpath.join(packages, "bin")) if part
            ),
            "PYTHONPATH": os.pathsep.join(
                part for part in (env.get("PYTHONPATH") or "", packages) if part
            ),
        }
    )
    developer_paths = _developer_paths()
    if developer_paths and "DEVELOPER_DIR" not in sanitized:
        # xcselect honours DEVELOPER_DIR before it reads the xcode_select link,
        # so the /usr/bin shims resolve the real tool without depending on that
        # link being readable.
        sanitized["DEVELOPER_DIR"] = developer_paths[0]
    return sanitized


# ── launch preparation ───────────────────────────────────────────────


def prepare(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    """Wrap ``plan.argv`` in sandbox-exec with a profile built for this launch."""
    ok, reason = available()
    if not ok:
        # The probe should have caught this, but argv[0] is about to be the
        # launcher and a missing or user-writable one must not reach a Popen.
        raise SandboxUnavailableError(reason)
    workdir = _validated(os.path.abspath(plan.workdir))
    if not os.path.isdir(workdir):
        raise SandboxUnavailableError(f"the session workdir does not exist: {workdir}")
    if posixpath.dirname(workdir) == workdir:
        # "/" as the session workdir would make the entire filesystem the
        # writable set, which is the opposite of what this backend claims.
        raise SandboxUnavailableError(f"the session workdir cannot be a filesystem root: {workdir}")
    # The same scan the Linux backend runs, and for the same reason: this profile
    # grants file-write* over the workdir subpath, so a regular file in here that
    # is hard-linked to one outside writes through to the host inode and the
    # advertised write boundary is not the boundary. A device or IPC node in here
    # is a channel no path rule closes either.
    scan_workdir_for_host_channels(workdir)
    # /tmp, not the per-user /var/folders tmp: the profile has to name this
    # directory, and /tmp keeps it out of the confidential per-user container.
    private_tmp = tempfile.mkdtemp(
        prefix = "us-seatbelt-", dir = "/tmp" if sys.platform == "darwin" else None
    )
    try:
        # A runtime root inside the session workdir is dropped rather than listed
        # twice: the workdir rules already make it readable, and this keeps the
        # read set to paths the sandbox reaches for outside it. The workdir goes
        # in so the drop happens by origin as well as by target.
        runtime_paths = tuple(
            path for path in runtime_read_paths(workdir) if not _within(path, workdir)
        )
        profile = build_profile(
            workdir = workdir,
            private_tmp = private_tmp,
            runtime_paths = runtime_paths,
            developer_paths = _developer_paths(),
        )
        return PreparedSandboxLaunch(
            argv = (SANDBOX_EXEC, "-p", profile, "--", *plan.argv),
            workdir = workdir,
            env = _sandbox_environment(plan.env, workdir, private_tmp),
            # Preserved, not replaced: the pre-exec is the os.setsid() that the
            # killpg teardown in tools.py signals, and dropping it strands the
            # process group on timeout or cancellation.
            preexec_fn = plan.preexec_fn,
            backend = BACKEND_NAME,
            cleanup_paths = [private_tmp],
            timeout_seconds = plan.timeout_seconds,
            close_fds = plan.close_fds,
            terminate_descendants = plan.terminate_descendants,
        )
    except Exception:
        shutil.rmtree(private_tmp, ignore_errors = True)
        raise
