# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The macOS half of the tool sandbox: a Seatbelt profile driven by sandbox-exec.

``(deny default)``. IP egress is deliberately unconfined, matching Linux;
AF_UNIX is not, because a connect() to a host socket such as Docker's is
governed by no file rule. Three hazards are invisible until they bite (dual path
spellings, ancestor metadata, optional literals), each commented where it is
implemented, and only ``test_profile_compiles_under_sandbox_exec`` checks the
SBPL grammar, on Darwin only.
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
    WorkdirUnsafeError,
    editable_import_roots,
    editable_source_roots,
    scan_workdir_for_host_channels,
)

BACKEND_NAME = "macos-seatbelt"
PROFILE_ID = f"macos-seatbelt-{PROFILE_VERSION}"
LIMITATIONS = (
    "deprecated_undocumented_sbpl",
    "unrestricted_network",
    # No /proc and no pidfds, so a descendant outliving the leader is not swept.
    "detached_descendant_cleanup_unverified",
    "pytorch_posix_shm_namespace_shared",
    # Unfiltered: a wrong guess at torch's and OpenMP's names breaks them.
    "posix_semaphore_namespace_shared",
    # kern.proc.pid./pgrp. leak host kinfo_proc, but that sysctl is how ps works.
    "host_process_metadata_readable",
    # Only RootDomainUserClient: the user clients Metal needs are too large an
    # authority to grant untested, so MPS cannot initialise.
    "gpu_devices_hidden",
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
    "/private/etc/hosts",
    "/private/etc/resolv.conf",
    "/System/Library/CoreServices/.SystemVersionPlatform.plist",
    "/System/Library/CoreServices/SystemVersion.plist",
    # /usr/bin/git and friends are xcode-select shims and fail without these.
    "/private/var/db/xcode_select_link",
    "/Library/Developer/CommandLineTools",
    "/Applications/Xcode.app/Contents/Developer",
)
# SYSTEM keychains only; the login keychain stays unreadable.
_TLS_TRUST_PATHS = (
    # The PUBLIC components one by one, never /etc/ssl whole. A locally managed
    # OpenSSL keeps its private keys in a directory beside the certificates, and
    # this grants recursive file-read* while the network stays open, so a whole
    # -tree rule is an exfiltratable key. The Linux backend names them separately
    # for exactly this reason and macOS did not, which is the asymmetry here.
    "/private/etc/ssl/cert.pem",
    "/private/etc/ssl/certs",
    "/private/etc/ssl/openssl.cnf",
    "/System/Library/Keychains",
    "/Library/Keychains",
    "/System/Library/Security",
    "/private/var/db/mds",
    "/Library/Preferences/com.apple.security.plist",
    "/Library/Preferences/com.apple.security.revocation.plist",
)
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
# Every optional root has to RESOLVE inside one of these. Homebrew on Intel
# chowns /usr/local to the invoking user, so an /usr/local/bin symlinked at the
# home directory is something a user, or an earlier unisolated tool call, can
# arrange; _path_filters resolves before it emits, and the recursive subpath
# would then be over a home subtree in a profile whose claim is the opposite.
# Ownership is the wrong test here, because that same chown would drop the
# Homebrew trees this exists to keep working. Containment is the right one.
_OPTIONAL_ROOT_PREFIXES = ("/usr/local", "/opt/homebrew")
# HAZARD 3, optional literals. Under (deny default) an absent file yields
# EPERM rather than ENOENT and git aborts, and the existence-filtered path
# rules cannot carry these.
_OPTIONAL_READ_LITERALS = (
    "/etc/gitconfig",
    "/etc/gitattributes",
    # A Homebrew git reads its own prefix, and the binary on PATH decides which.
    "/usr/local/etc/gitconfig",
    "/usr/local/etc/gitattributes",
    "/opt/homebrew/etc/gitconfig",
    "/opt/homebrew/etc/gitattributes",
    "/Library/Preferences/com.apple.dt.Xcode.plist",
)
# Exec-denied: each hands work to a daemon outside the sandbox.
_DENIED_EXECUTABLES = (
    "/usr/bin/open",
    "/usr/bin/osascript",
    "/usr/bin/security",
    "/bin/launchctl",
    "/usr/bin/sandbox-exec",
)
_DEVICES = ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom")
# connect() to a unix socket is network-outbound, not a file operation.
_MDNSRESPONDER_SOCKET = "/private/var/run/mDNSResponder"
# com.apple.SecurityServer is deliberately ABSENT: with it the login Keychain is
# readable through Security.framework despite /usr/bin/security being exec-denied.
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
# An omission is a silent breakage (empty os.cpu_count()), so be generous.
_SYSCTL_NAMES = (
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
_SYSCTL_PREFIXES = (
    "hw.optional.arm.",
    "hw.optional.armv8_",
    "hw.perflevel",
    "kern.proc.pgrp.",
    "kern.proc.pid.",
)
# fmt: on

_developer_paths_cache: tuple[str, ...] | None = None
_developer_paths_lock = threading.Lock()


def available() -> tuple[bool, str]:
    """Presence and executability only; whether a profile loads is the probe's."""
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


def _sbpl_string(value: str) -> str:
    r"""An SBPL string literal, non-ASCII left RAW: SBPL is TinyScheme, which knows
    \", \n, \r, \t and \xDD and no \u, so json's default turned /Users/José into a
    rule matching nothing and every macOS home with an accent lost isolation
    silently. The profile is an argv string, so the raw character arrives as the
    same UTF-8 bytes the path has."""
    return json.dumps(value, ensure_ascii = False)


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
    """A filterless rule is UNCONDITIONAL, so ``(allow file-write* )`` grants the
    whole filesystem. Every rule meant to be unconditional is written literally,
    so reaching here empty always means a list collapsed."""
    if not filters:
        raise SandboxUnavailableError(
            f"the Seatbelt profile would apply {operations!r} unconditionally: no paths survived"
        )
    return f"({operations} " + " ".join(filters) + ")"


def _sbpl_spellings(path: str, *, resolve: bool = True) -> tuple[str, ...]:
    """HAZARD 1, dual path spellings. /etc, /tmp and /var are symlinks into
    /private and Seatbelt evaluates the spelling it was GIVEN, so a rule written
    against only one form EPERMs anything using the other. Both are generated
    rather than trusted to realpath, so the pair is complete where the symlink is
    missing. ``resolve = False`` drops the target, so an /etc/gitconfig symlinked
    into ~/dotfiles cannot put a home path in the read set."""
    selected = [posixpath.abspath(_validated(path))]
    if resolve:
        # Validated: a newline in a symlink target would end up inside a quoted string.
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
    """``(literal ...)`` per spelling, plus ``(subpath ...)`` for directories."""
    filters: list[str] = []
    # Keyed by kind: a file and a directory can share a resolved spelling, and
    # one set would let the file's literal suppress the directory's subpath.
    seen: set[tuple[str, str]] = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        kinds = ("literal", "subpath") if os.path.isdir(path) else ("literal",)
        for spelling in _sbpl_spellings(path):
            encoded = _sbpl_string(spelling)
            for kind in kinds:
                if (kind, encoded) not in seen:
                    seen.add((kind, encoded))
                    filters.append(f"({kind} {encoded})")
    return filters


def _literal_filters(paths: tuple[str, ...], *, resolve: bool = True) -> list[str]:
    """Existence not required: for a deny, dropping the rule is the wrong way to fail."""
    filters: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for spelling in _sbpl_spellings(path, resolve = resolve):
            encoded = _sbpl_string(spelling)
            if encoded not in seen:
                seen.add(encoded)
                filters.append(f"(literal {encoded})")
    return filters


def _ancestor_filters(spellings: tuple[str, ...]) -> list[str]:
    """HAZARD 2, ancestor metadata. Resolving /a/b/c stats /a and /a/b, so a
    readable leaf with un-stat-able ancestors fails mid-path with EPERM.
    file-read-metadata only, so the directories stay unlistable. Takes spellings,
    not paths, so the caller decides whether symlinks were followed. posixpath,
    since on Windows ntpath.dirname("/") returns "/" but os.path.sep does not."""
    filters: list[str] = []
    seen: set[str] = set()
    for spelling in spellings:
        current = posixpath.dirname(_validated(spelling))
        while current:
            encoded = _sbpl_string(current)
            if encoded not in seen:
                seen.add(encoded)
                filters.append(f"(literal {encoded})")
            parent = posixpath.dirname(current)
            if parent == current:
                break
            current = parent
    return filters


def _trusted_system_dir(path: str) -> bool:
    """A real directory owned by root that no one else can write.

    The toolchain is granted recursive reads, so "it exists" is not enough: the
    point of the check is that a path the invoking user controls cannot be turned
    into a read rule over their own home.
    """
    try:
        info = os.stat(path)
    except OSError:
        return False
    if not stat.S_ISDIR(info.st_mode):
        return False
    return info.st_uid == 0 and not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


def _developer_paths() -> tuple[str, ...]:
    """Versioned on many hosts (/Applications/Xcode_16.4.app), so the static list
    above cannot name it."""
    global _developer_paths_cache
    with _developer_paths_lock:
        if _developer_paths_cache is not None:
            return _developer_paths_cache
        found: list[str] = []
        if sys.platform == "darwin" and os.path.exists("/usr/bin/xcode-select"):
            try:
                # DEVELOPER_DIR is stripped, and the answer is then checked
                # rather than trusted. xcode-select honours that variable, so a
                # Studio started with it aimed at a toolchain under $HOME would
                # otherwise return a home directory that this grants recursive
                # file-read* over, including its enclosing .app -- in a profile
                # whose whole claim is that $HOME is not readable.
                environment = {k: v for k, v in os.environ.items() if k != "DEVELOPER_DIR"}
                result = subprocess.run(
                    ["/usr/bin/xcode-select", "-p"],
                    capture_output = True,
                    text = True,
                    encoding = "utf-8",
                    timeout = 10,
                    check = False,
                    env = environment,
                )
                candidate = result.stdout.strip()
                if result.returncode == 0 and candidate and _trusted_system_dir(candidate):
                    for spelling in (os.path.realpath(candidate), candidate):
                        if spelling not in found:
                            found.append(spelling)
                    # xcselect validates against the enclosing app bundle.
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
    """*workdir* is excluded by ORIGIN: dropping only the resolved form would skip
    ``<workdir>/venv/lib`` and grant file-read* on the ~/.ssh behind it."""
    candidates: list[str] = [
        os.path.dirname(os.path.realpath(sys.executable)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_site"),
    ]
    # Subdirectories, never the prefix: ``python -m venv .`` at a project root
    # makes sys.prefix the project root, and the network is open, so a readable
    # .env is an exportable one. All four prefixes, because for a uv-managed
    # interpreter base_prefix and base_exec_prefix differ and lib-dynload hangs
    # off the alias spelling.
    for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        candidates.extend(
            posixpath.join(prefix, name)
            for name in ("bin", "include", "lib", "lib64", "libexec", "pyvenv.cfg", "ssl")
        )
        # A python.org framework build loads its dyld image from <prefix>/Python, a
        # FILE at the top of the prefix and under no read root.
        candidates.append(posixpath.join(prefix, "Python"))
    try:
        paths = sysconfig.get_paths()
        candidates.extend(
            paths[key]
            for key in ("stdlib", "platstdlib", "purelib", "platlib", "include", "platinclude")
            if paths.get(key)
        )
    except (KeyError, OSError):
        pass
    # Same as the Linux backend: an editable install's source root is outside
    # site-packages, and a candidate here inherits every guard below.
    candidates.extend(editable_source_roots())
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:
        pass  # some virtualenv layouts omit getsitepackages; sysconfig covers them

    selected: list[str] = []
    for candidate in candidates:
        if not candidate or not posixpath.isabs(candidate):
            continue
        if workdir is not None and _within(posixpath.abspath(candidate), workdir):
            continue
        # Both spellings: a venv reached through a symlink needs the link's own path
        # as well as the directory it lands on.
        for path in (posixpath.abspath(candidate), os.path.realpath(candidate)):
            # "/" or "/usr" as a read ROOT hands back most of the host.
            if path in ("/", "/usr") or not os.path.exists(path):
                continue
            if path not in selected and not any(_within(path, root) for root in selected):
                selected.append(path)
    return tuple(selected)


def _contained_optional_roots() -> tuple[str, ...]:
    """Optional search roots whose target stays inside an approved prefix.

    Dropped rather than un-resolved: the whole point of a search root is that
    Homebrew's /usr/local/bin entries are symlinks into ../Cellar, so refusing to
    follow them would grant a directory of dangling names.
    """
    kept: list[str] = []
    for root in _OPTIONAL_READ_ROOTS:
        resolved = os.path.realpath(root)
        if any(
            _within(root, prefix) and _within(resolved, prefix)
            for prefix in _OPTIONAL_ROOT_PREFIXES
        ):
            kept.append(root)
    return tuple(kept)


def runtime_paths_under(workdir: str) -> tuple[str, ...]:
    """Interpreter directories inside the session workdir. The Linux twin of this.

    runtime_read_paths drops them so a <workdir>/venv/lib symlinked at ~/.ssh is
    not granted by name, but file-write* covers the workdir subpath, so dropping
    alone leaves Studio's own venv writable when it sits beneath the workdir. A
    tool call could then rewrite site-packages or the interpreter and the next
    server subprocess started from sys.executable would run it with the server's
    authority. Denied after the write allowance instead; Seatbelt is
    last-match-wins.

    Only when both spellings stay inside the workdir. One that RESOLVES outside is
    the symlink case, and denying that path would be denying the user's own home.

    Both spellings of the WORKDIR too. build_profile is handed the caller's
    spelling, and its write allowance covers the resolved form as well, so
    measuring containment against the alias alone rejected every runtime path
    when the workdir was a symlink and no denial was emitted at all.
    """
    roots: list[str] = []
    for root in (posixpath.abspath(workdir), os.path.realpath(workdir)):
        if root not in roots:
            roots.append(root)
    canonical_root = os.path.realpath(workdir)
    inside: list[str] = []
    # "Python" is the framework build's top-level dyld image, which
    # runtime_read_paths already names: omitted here it stayed writable under the
    # workdir allowance, which is the one file a later host subprocess maps.
    for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        for name in ("bin", "include", "lib", "lib64", "libexec", "pyvenv.cfg", "ssl", "Python"):
            candidate = posixpath.join(prefix, name)
            if not os.path.exists(candidate):
                continue
            # The RESOLVED path decides. Pairing the two lexical tests per root
            # answered a different question: an alias-prefixed path is not
            # beneath the canonical root and a canonical one is not beneath the
            # alias, so every path was rejected either way round -- and a venv
            # invoked through a symlink keeps that alias in sys.prefix.
            resolved = os.path.realpath(candidate)
            if not _within(resolved, canonical_root):
                continue
            # Denied under every spelling of the workdir, since Seatbelt judges
            # the path as written and the allowance covers them all.
            relative = posixpath.relpath(resolved, canonical_root)
            for other in roots:
                spelling = posixpath.join(other, relative)
                if spelling not in inside:
                    inside.append(spelling)
    return tuple(inside)


def build_profile(
    *,
    workdir: str,
    private_tmp: str,
    runtime_paths: tuple[str, ...],
    developer_paths: tuple[str, ...] = (),
) -> str:
    """Writes and spawns nothing, so a test can assert on the text anywhere."""
    readable_paths = (
        *_READ_ROOTS,
        *_TLS_TRUST_PATHS,
        *_contained_optional_roots(),
        *developer_paths,
        *_DEVICES,
        *runtime_paths,
        workdir,
        private_tmp,
    )
    read_filters = ['(literal "/")', *_path_filters(readable_paths)]
    # The optional literals may not exist but their parents still need stat, and
    # so does mDNSResponder's socket, resolved before the network rule applies.
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
    # The workdir too, since TMPDIR points into it and an AF_UNIX bind is
    # network-bind, not a file operation.
    tmp_subpaths = " ".join(
        f"(subpath {_sbpl_string(spelling)})"
        for path in (private_tmp, workdir)
        for spelling in _sbpl_spellings(path)
    )
    mdns_filters = " ".join(_literal_filters((_MDNSRESPONDER_SOCKET,)))
    # resolve = False so an /etc/gitconfig symlinked into the home does not turn
    # a config read allowance into a home one.
    # The editable import roots ride here rather than in read_filters because a
    # literal grants the directory itself, which is all a listing needs, while
    # _path_filters would add the subpath and hand back the whole checkout.
    optional_filters = _literal_filters(
        _OPTIONAL_READ_LITERALS + editable_import_roots(), resolve = False
    )
    sysctl_filters = [
        *(f"(sysctl-name {_sbpl_string(name)})" for name in _SYSCTL_NAMES),
        *(f"(sysctl-name-prefix {_sbpl_string(name)})" for name in _SYSCTL_PREFIXES),
    ]
    lines = [
        "(version 1)",
        "(deny default)",
        # Scoped to the same Seatbelt instance, so a tool cannot poke at Studio.
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
        # AFTER the allowance, because Seatbelt is last-match-wins: Studio's own
        # runtime stays read-only even when it lives under the writable workdir.
        # See runtime_paths_under.
        *(
            [_rule("deny file-write*", _path_filters(runtime_under))]
            if (runtime_under := runtime_paths_under(workdir))
            else []
        ),
        _rule("allow file-read* file-test-existence file-write-data", device_filters),
        # bash process substitution hands the child /dev/fd/63, and opening it dups
        # a descriptor already held.
        '(allow file-read* (regex #"^/dev/fd/[0-9]+$"))',
        '(allow file-write* (regex #"^/dev/fd/[0-9]+$"))',
        _rule("allow file-ioctl", device_filters),
        "(allow ipc-posix-sem)",
        # write-data, here and below: libomp writes its registration into the segment
        # and OpenMP init fails without it. The probe loads no OpenMP.
        "(allow ipc-posix-shm-read-data ipc-posix-shm-write-create "
        "ipc-posix-shm-write-data ipc-posix-shm-write-unlink "
        '(ipc-posix-name-regex #"^/__KMP_REGISTERED_LIB_[0-9]+$"))',
        "(allow ipc-posix-shm-read-data ipc-posix-shm-write-create "
        "ipc-posix-shm-write-data ipc-posix-shm-write-unlink "
        '(ipc-posix-name-regex #"^/torch_[0-9]+_[0-9]+_[0-9]+$"))',
        # SharedMemory names its segment "/psm_" + token_hex(4), which the probe,
        # passing only descriptors, misses.
        "(allow ipc-posix-shm-read-data ipc-posix-shm-write-create "
        "ipc-posix-shm-write-data ipc-posix-shm-write-unlink "
        '(ipc-posix-name-regex #"^/psm_[0-9a-f]+$"))',
        # Gates socket() itself, unfiltered because a configd-backed resolver creates
        # route and AF_SYSTEM sockets. That admits raw and kernel-control sockets:
        # authority over the network, never over the filesystem.
        "(allow system-socket)",
        # IP only: an unfiltered bind covers AF_UNIX, creating a socket at a path no
        # file rule governs.
        '(allow network-bind (local ip "*:*"))',
        "(allow network-inbound)",
        # Filtered to ip because an unfiltered outbound covers AF_UNIX, and a
        # connect() to /var/run/docker.sock starts an unconfined container with host
        # bind mounts. "*:*" still leaves TCP and UDP, v4 and v6, open.
        '(allow network-outbound (remote ip "*:*"))',
        # The AF_UNIX destinations the ip filter no longer covers.
        f"(allow network-bind (local unix-socket {tmp_subpaths}))",
        f"(allow network-outbound (remote unix-socket {tmp_subpaths}))",
        # Both filter forms are accepted and which one a macOS release matches is
        # untestable here, while a missed DNS socket looks like a resolver bug.
        f"(allow network-outbound {mdns_filters})",
        f"(allow network-outbound (remote unix-socket {mdns_filters}))",
        _rule("allow sysctl-read", sysctl_filters),
        '(allow iokit-open (iokit-registry-entry-class "RootDomainUserClient"))',
        "(allow mach-lookup\n"
        + "\n".join(f"  (global-name {_sbpl_string(name)})" for name in _MACH_SERVICES)
        + ")",
    ]
    return "\n".join(lines) + "\n"


def _tmpdir(env: dict[str, str], workdir: str, private_tmp: str) -> str:
    """The caller's TMPDIR when it is inside the workdir: ``private_tmp`` is
    deleted when the call ends, dropping what ``tempfile`` wrote."""
    requested = env.get("TMPDIR") or ""
    if requested and _within(posixpath.abspath(requested), workdir):
        return requested
    return private_tmp


def _sandbox_environment(env: dict[str, str], workdir: str, private_tmp: str) -> dict[str, str]:
    """HOME and TMPDIR must point inside the sandbox: the real HOME is unreadable
    and the per-user /var/folders tmp unwritable. DYLD_* is the dynamic-linker
    injection surface."""
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
    # Appended, so an installed package cannot shadow the sandbox_site shim.
    packages = posixpath.join(workdir, SESSION_PACKAGES_RELPATH)
    sanitized.update(
        {
            "HOME": workdir,
            "TMPDIR": tmpdir,
            "TMP": tmpdir,
            "TEMP": tmpdir,
            "XDG_RUNTIME_DIR": private_tmp,
            "PIP_TARGET": packages,
            # <target>/bin holds pip's console entry points. LAST, since it is writable
            # by the tool call and a planted binary must not shadow a bare command the
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
        # xcselect honours DEVELOPER_DIR before the xcode_select link.
        sanitized["DEVELOPER_DIR"] = developer_paths[0]
    return sanitized


def prepare(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    ok, reason = available()
    if not ok:
        # argv[0] is about to be the launcher; a user-writable one must not reach a
        # Popen even if the probe should have caught it.
        raise SandboxUnavailableError(reason)
    workdir = _validated(os.path.abspath(plan.workdir))
    if not os.path.isdir(workdir):
        raise WorkdirUnsafeError(f"the session workdir does not exist: {workdir}")
    if posixpath.dirname(workdir) == workdir:
        # "/" would make the entire filesystem the writable set.
        raise WorkdirUnsafeError(f"the session workdir cannot be a filesystem root: {workdir}")
    # file-write* covers the workdir subpath, so a file hard-linked outside
    # writes through to the host inode.
    scan_workdir_for_host_channels(workdir)
    # /tmp, not /var/folders: the profile has to name this directory, and this
    # keeps it out of the confidential per-user container.
    private_tmp = tempfile.mkdtemp(
        prefix = "us-seatbelt-", dir = "/tmp" if sys.platform == "darwin" else None
    )
    try:
        # Passed in so the drop happens by origin as well as by target.
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
            # Preserved: it is the os.setsid() tools.py's killpg teardown signals.
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
