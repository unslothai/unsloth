# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The macOS half of the tool sandbox: a Seatbelt profile driven by sandbox-exec.

The profile is ``(deny default)`` and grants back exactly what a Python or
Terminal tool call needs: the interpreter's own runtime paths, the system
frameworks and command-line tools, the session workdir, and a private tmp. The
user's home is not in that set, so ``~/.ssh``, ``~/.aws`` and the login Keychain
are unreadable even though the process still runs as the user.

The network is deliberately NOT confined, matching the Linux backend and what
``os_sandbox`` reports (``network_policy = "unrestricted"``): tool calls still
pip-install and download models, so a script that reaches a secret can still
send it. The filesystem boundary is the claim; the network is not.

SBPL is deprecated and undocumented for third-party products, so the filesystem
and process rules are carried over from a profile that was iterated against a
real Mac rather than reasoned about. Three of them exist for reasons invisible
until they bite -- dual path spellings, ancestor metadata, and optional literals
-- and each is commented where it is implemented.

The network rules and the DNS/TLS mach services are the part that has no such
history: they replace an allowlist proxy that used to make them unnecessary, and
they are written in the broadest form that expresses "unrestricted" precisely
because a narrower one that compiles but silently denies UDP would read as a DNS
bug forever. ``test_profile_compiles_under_sandbox_exec`` is what checks them,
and it only runs on Darwin.
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
    PreparedSandboxLaunch,
    SandboxUnavailableError,
    ToolLaunchPlan,
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
    # POSIX shared memory has one namespace per host, and torch's segments are
    # allowed by name pattern rather than isolated.
    "pytorch_posix_shm_namespace_shared",
)

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

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


def runtime_read_paths() -> tuple[str, ...]:
    """The interpreter's own runtime roots -- never arbitrary inherited sys.path.

    Same shape as the Linux backend's bind list: whatever the selected
    interpreter needs to start and import, and nothing that merely happens to be
    on sys.path because the parent process put it there.
    """
    candidates: list[str] = [
        sys.prefix,
        sys.base_prefix,
        # The exec pair is not the same two paths under another name. A
        # uv-managed base interpreter reports base_prefix as
        # ``cpython-3.12.12-macos-aarch64-none`` and base_exec_prefix as the
        # ``cpython-3.12-...`` alias symlink beside it, and lib-dynload -- every
        # C extension in the standard library -- hangs off the alias spelling.
        sys.exec_prefix,
        sys.base_exec_prefix,
        os.path.dirname(os.path.realpath(sys.executable)),
        # The code-interpreter path shim, loaded via PYTHONPATH in every
        # sandboxed launch; unreadable means every tool call dies at startup.
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_site"),
    ]
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
        # Some virtualenv layouts omit getsitepackages; sysconfig covers them.
        pass

    selected: list[str] = []
    for candidate in candidates:
        if not candidate or not posixpath.isabs(candidate):
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
    tmp_subpaths = " ".join(f"(subpath {json.dumps(s)})" for s in _sbpl_spellings(private_tmp))
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
        '(allow file-read* (regex #"^/dev/fd/(0|1|2)$"))',
        '(allow file-write* (regex #"^/dev/fd/(1|2)$"))',
        _rule("allow file-ioctl", device_filters),
        "(allow ipc-posix-sem)",
        "(allow ipc-posix-shm-read-data ipc-posix-shm-write-create "
        "ipc-posix-shm-write-unlink "
        '(ipc-posix-name-regex #"^/__KMP_REGISTERED_LIB_[0-9]+$"))',
        "(allow ipc-posix-shm-read-data ipc-posix-shm-write-create "
        "ipc-posix-shm-write-data ipc-posix-shm-write-unlink "
        '(ipc-posix-name-regex #"^/torch_[0-9]+_[0-9]+_[0-9]+$"))',
        # The network is NOT confined here -- see the module docstring, and the
        # "unrestricted_network" limitation the record carries. Unfiltered on
        # purpose: an unfiltered operation is the one spelling of "unrestricted"
        # that cannot be got subtly wrong on a host nobody can test from, and a
        # narrowed one that compiles but denies UDP would look like a DNS bug
        # forever. system-socket is a separate operation because it gates
        # socket() itself, and it is open rather than domain-filtered because a
        # configd-backed resolver creates route and AF_SYSTEM sockets: that
        # admits raw and kernel-control sockets too, which is authority over the
        # network, never over the filesystem this profile is confining.
        # network-inbound is included for parity with the Linux backend, where
        # the sandbox shares the host's netns and inbound cannot be withheld.
        "(allow system-socket)",
        "(allow network-bind)",
        "(allow network-inbound)",
        "(allow network-outbound)",
        # AF_UNIX, subsumed by the rule above and kept explicit anyway. A
        # connect()/bind() on a unix socket is network-outbound / network-bind
        # in Seatbelt, not a file operation, so multiprocessing's socket in the
        # private tmp and mDNSResponder's socket are NOT covered by any file
        # rule. Narrowing network-outbound later without these would break
        # multiprocessing and DNS with a denial that names neither.
        f"(allow network-bind (local unix-socket {tmp_subpaths}))",
        f"(allow network-outbound (remote unix-socket {tmp_subpaths}))",
        f"(allow network-outbound {mdns_filters})",
        _rule("allow sysctl-read", sysctl_filters),
        '(allow iokit-open (iokit-registry-entry-class "RootDomainUserClient"))',
        "(allow mach-lookup\n"
        + "\n".join(f"  (global-name {json.dumps(name)})" for name in _MACH_SERVICES)
        + ")",
    ]
    return "\n".join(lines) + "\n"


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
    sanitized.update(
        {
            "HOME": workdir,
            "TMPDIR": private_tmp,
            "TMP": private_tmp,
            "TEMP": private_tmp,
            "XDG_RUNTIME_DIR": private_tmp,
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
    # /tmp, not the per-user /var/folders tmp: the profile has to name this
    # directory, and /tmp keeps it out of the confidential per-user container.
    private_tmp = tempfile.mkdtemp(
        prefix = "us-seatbelt-", dir = "/tmp" if sys.platform == "darwin" else None
    )
    try:
        # A runtime root inside the session workdir is dropped rather than
        # listed twice: the workdir rules already make it readable, and this
        # keeps the read set to paths the sandbox reaches for outside it.
        runtime_paths = tuple(path for path in runtime_read_paths() if not _within(path, workdir))
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
