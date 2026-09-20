# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Default-deny Seatbelt profiles for sandbox-exec.

IP egress is unrestricted. AF_UNIX needs separate rules because file rules do
not block socket connections. SBPL grammar is tested on Darwin.
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
    model_library_roots,
    scan_workdir_for_host_channels,
    studio_state_roots,
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

_REJECTED_CONTROLS = frozenset(chr(code) for code in (*range(0x20), 0x7F))


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
    # Allow public TLS components individually; /etc/ssl can also hold private keys.
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
# Homebrew prefixes are user-owned; check target containment rather than ownership.
_OPTIONAL_ROOT_PREFIXES = ("/usr/local", "/opt/homebrew")
# Allow missing optional files so git gets ENOENT, not a fatal EPERM.
# Existence-filtered rules would omit these paths.
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
    if not path or not posixpath.isabs(path) or any(c in path for c in _REJECTED_CONTROLS):
        # TinyScheme cannot interpret JSON's \b, \f or \u00XX control escapes.
        raise SandboxUnavailableError(
            f"Seatbelt paths must be absolute and free of control characters: {path!r}"
        )
    try:
        # Reject surrogateescaped bytes before encoding the profile into argv.
        path.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SandboxUnavailableError(
            f"Seatbelt paths must be encodable as UTF-8: {path!r}"
        ) from exc
    return path


def _within(path: str, root: str) -> bool:
    """Whether ``path`` is ``root`` or sits under it, by whole path components."""
    return path == root or path.startswith(root.rstrip("/") + "/")


def _within_any(path: str, roots: "tuple[str, ...]") -> bool:
    """``_within`` across every spelling both sides can have.

    On macOS /var, /tmp and /etc are symlinks into /private, so the same
    directory has two names and which one a caller holds depends on where it
    came from: a session workdir arrives resolved, a configured Studio home
    does not. A plain prefix compare says they are unrelated, and the
    consequence is not a missing grant but an active denial, because the deny
    rules are emitted in BOTH spellings while the restore list was built from
    one. Observed on macos-15: with the Studio home under /var/folders the
    sandboxed interpreter could not open its own script and every tool call
    returned "Operation not permitted".
    """
    candidates = _sbpl_spellings(path)
    return any(
        _within(candidate, root_spelling)
        for candidate in candidates
        for root in roots
        for root_spelling in _sbpl_spellings(root)
    )


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
    """Include both /private aliases even when the symlink is absent.

    Seatbelt needs both spellings. ``resolve = False`` excludes symlink targets,
    preventing an /etc/gitconfig link from granting access to home files.
    """
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
    """Allow ancestor metadata for path resolution, without directory listing.

    Accept caller-resolved spellings; use posixpath for Windows-hosted tests.
    """
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
    """Require a root-owned directory writable only by root before granting reads."""
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
                # DEVELOPER_DIR could redirect this recursive grant into the user's home.
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
    """Exclude candidates under either workdir spelling before resolving them.

    Otherwise a venv/lib symlink could grant reads of an external target such as
    ~/.ssh. sys.prefix may retain either spelling, independently of the caller.
    """
    candidates: list[str] = [
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
    # Keep argv[0]'s spelling without granting its possibly private parent directory.
    # Last so existing prefix/bin grants cover ordinary venv, conda and uv layouts.
    candidates.append(sys.executable)
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:
        pass  # some virtualenv layouts omit getsitepackages; sysconfig covers them

    selected: list[str] = []
    for candidate in candidates:
        if not candidate or not posixpath.isabs(candidate):
            continue
        written = posixpath.abspath(candidate)
        if workdir is not None and any(
            _within(written, root) for root in (workdir, os.path.realpath(workdir))
        ):
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
    """Protect Studio's runtime under every writable spelling of the workdir.

    Resolved containment handles aliases in sys.prefix. External targets stay ungranted.
    """
    roots: list[str] = []
    for root in (posixpath.abspath(workdir), os.path.realpath(workdir)):
        if root not in roots:
            roots.append(root)
    canonical_root = os.path.realpath(workdir)
    inside: list[str] = []
    # Include standalone Python and the framework's top-level dyld image.
    candidates = [sys.executable, *editable_source_roots()]
    for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        candidates.extend(
            posixpath.join(prefix, name)
            for name in ("bin", "include", "lib", "lib64", "libexec", "pyvenv.cfg", "ssl", "Python")
        )
    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        resolved = os.path.realpath(candidate)
        if not _within(resolved, canonical_root):
            if candidate is sys.executable and _within(
                posixpath.abspath(candidate), canonical_root
            ):
                # A replaceable interpreter alias would let the next host probe execute tool code.
                raise WorkdirUnsafeError(
                    "the session workdir holds a link to the Python that runs Studio: "
                    f"{posixpath.abspath(candidate)}"
                )
            continue
        # Protect the alias name as well as its target, or the tool can replace the alias.
        written = posixpath.abspath(candidate)
        for form in (resolved, written):
            if not _within(form, canonical_root):
                continue
            relative = posixpath.relpath(form, canonical_root)
            for other in roots:
                spelling = posixpath.join(other, relative)
                if spelling not in inside:
                    inside.append(spelling)
    return tuple(inside)


def _studio_state_rules(
    runtime_paths: tuple[str, ...], developer_paths: tuple[str, ...], workdir: str, private_tmp: str
) -> list[str]:
    """Deny Studio's own state, then restore what the launch genuinely needs.

    A blanket deny would be wrong: on a custom-home install the managed venv
    lives under the Studio root, so the interpreter would stop being readable.
    The restore list is the paths the profile already computed as necessary,
    not a wider re-grant.
    """
    state = studio_state_roots()
    if not state:
        return []
    needed = tuple(
        path
        for path in (*runtime_paths, *developer_paths, workdir, private_tmp)
        if path and _within_any(path, state)
    )
    # file-read-DATA, not file-read*. The workdir lives UNDER the Studio root
    # on a default install, so denying read* also denies stat on the
    # directories leading to it; os.makedirs then decides an existing ancestor
    # is missing, tries to create it and fails with EPERM. Observed on
    # macos-15, where test_python_exec_mnt_data_open_is_remapped_into_workdir
    # failed that way and passed at the merge base. A directory entry existing
    # is not the secret. auth.db's contents are, and reading a file or listing
    # a directory is file-read-data, which stays denied.
    rules = [_rule("deny file-read-data file-map-executable", _path_filters(state))]
    if needed:
        rules.append(
            _rule("allow file-read* file-test-existence file-map-executable", _path_filters(needed))
        )
    return [rule for rule in rules if rule]


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
        # The model folders the approval gate already lets a tool read without
        # asking. Without them a read from a registered folder passes the gate
        # silently and then fails in here.
        *model_library_roots(),
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
    tmp_subpaths = tuple(
        f"(subpath {_sbpl_string(spelling)})"
        for path in (private_tmp, workdir)
        for spelling in _sbpl_spellings(path)
    )
    mdns_filters = _literal_filters((_MDNSRESPONDER_SOCKET,))
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
        # AFTER the read allowances, because Seatbelt is last-match-wins. A
        # custom Studio home under one of the optional read roots (a Homebrew
        # prefix, say) is otherwise recursively readable, which hands over
        # auth/auth.db and the HS256 jwt_secret to model-authored code and
        # walks past tools.py's literal-path guard even in `required` mode.
        # The runtime paths inside it are restored immediately below, since on
        # a custom-home install the interpreter itself lives there.
        *_studio_state_rules(runtime_paths, developer_paths, workdir, private_tmp),
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
        # Each endpoint takes one path filter; combining paths breaks multiprocessing on macOS 15.
        *(f"(allow network-bind (local unix-socket {path}))" for path in tmp_subpaths),
        *(f"(allow network-outbound (remote unix-socket {path}))" for path in tmp_subpaths),
        # Both filter forms are accepted and which one a macOS release matches is
        # untestable here, while a missed DNS socket looks like a resolver bug.
        _rule("allow network-outbound", mdns_filters),
        *(f"(allow network-outbound (remote unix-socket {path}))" for path in mdns_filters),
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
    try:
        workdir = _validated(os.path.abspath(plan.workdir))
    except SandboxUnavailableError as exc:
        # Workdir refusals must not select auto's unisolated fallback.
        raise WorkdirUnsafeError(str(exc)) from exc
    if not os.path.isdir(workdir):
        raise WorkdirUnsafeError(f"the session workdir does not exist: {workdir}")
    if posixpath.dirname(workdir) == workdir:
        # "/" would make the entire filesystem the writable set.
        raise WorkdirUnsafeError(f"the session workdir cannot be a filesystem root: {workdir}")
    # file-write* covers the workdir subpath, so a file hard-linked outside
    # writes through to the host inode.
    workdir_limitations = scan_workdir_for_host_channels(workdir)
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
            launch_limitations = workdir_limitations,
        )
    except Exception:
        shutil.rmtree(private_tmp, ignore_errors = True)
        raise
