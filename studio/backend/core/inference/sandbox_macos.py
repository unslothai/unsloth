# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Default-deny Seatbelt profiles for sandbox-exec; file rules do not block AF_UNIX connect, so sockets get their own."""

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
    # Only RootDomainUserClient: Metal's user clients stay ungranted, so MPS cannot initialise.
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
# com.apple.SecurityServer is deliberately ABSENT: it would expose the login Keychain.
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
    """An SBPL string literal with non-ASCII left RAW: TinyScheme has no u-escape, so json's default breaks accented paths."""
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
    """``_within`` across both /private alias spellings; a miss is an active denial, not a missing grant."""
    candidates = _sbpl_spellings(path)
    return any(
        _within(candidate, root_spelling)
        for candidate in candidates
        for root in roots
        for root_spelling in _sbpl_spellings(root)
    )


def _rule(operations: str, filters: list[str]) -> str:
    """A filterless rule is UNCONDITIONAL, so reaching here empty means a list collapsed."""
    if not filters:
        raise SandboxUnavailableError(
            f"the Seatbelt profile would apply {operations!r} unconditionally: no paths survived"
        )
    return f"({operations} " + " ".join(filters) + ")"


def _sbpl_spellings(path: str, *, resolve: bool = True) -> tuple[str, ...]:
    """Both /private aliases even when absent; ``resolve = False`` keeps symlink targets out."""
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
    # Keyed by kind, so a file's literal cannot suppress a directory's subpath.
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
    """Allow ancestor metadata for path resolution, without directory listing."""
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
    """Versioned on many hosts (/Applications/Xcode_16.4.app), so the static list cannot name it."""
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
    """Exclude candidates under either workdir spelling BEFORE resolving, or a venv/lib symlink grants ~/.ssh."""
    candidates: list[str] = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_site"),
    ]
    # Subdirectories, never the prefix (``python -m venv .`` would expose .env); all four prefixes for uv.
    for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        candidates.extend(
            posixpath.join(prefix, name)
            for name in ("bin", "include", "lib", "lib64", "libexec", "pyvenv.cfg", "ssl")
        )
        # A framework build loads its dyld image from the FILE <prefix>/Python.
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
    # Editable source roots are outside site-packages; candidates inherit every guard below.
    candidates.extend(editable_source_roots())
    # Keep argv[0]'s spelling without granting its possibly private parent.
    candidates.append(sys.executable)
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:
        pass

    selected: list[str] = []
    for candidate in candidates:
        if not candidate or not posixpath.isabs(candidate):
            continue
        written = posixpath.abspath(candidate)
        if workdir is not None and any(
            _within(written, root) for root in (workdir, os.path.realpath(workdir))
        ):
            continue
        for path in (posixpath.abspath(candidate), os.path.realpath(candidate)):
            # "/" or "/usr" as a read ROOT hands back most of the host.
            if path in ("/", "/usr") or not os.path.exists(path):
                continue
            if path not in selected and not any(_within(path, root) for root in selected):
                selected.append(path)
    return tuple(selected)


def _contained_optional_roots() -> tuple[str, ...]:
    """Optional search roots whose resolved target stays inside an approved prefix."""
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
    """Protect Studio's runtime under every writable spelling of the workdir."""
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
    """Deny Studio's own state, then restore only paths the profile already needs (the venv may live there)."""
    state = studio_state_roots()
    if not state:
        return []
    needed = tuple(
        path
        for path in (*runtime_paths, *developer_paths, workdir, private_tmp)
        if path and _within_any(path, state)
    )
    # file-read-data, NOT file-read*: denying stat on the workdir's ancestors breaks os.makedirs.
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
        # Model folders the approval gate already reads silently.
        *model_library_roots(),
        workdir,
        private_tmp,
    )
    read_filters = ['(literal "/")', *_path_filters(readable_paths)]
    # Optional literals and mDNSResponder's socket still need parent stat.
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
    # The workdir too: an AF_UNIX bind is network-bind, not a file operation.
    tmp_subpaths = tuple(
        f"(subpath {_sbpl_string(spelling)})"
        for path in (private_tmp, workdir)
        for spelling in _sbpl_spellings(path)
    )
    mdns_filters = _literal_filters((_MDNSRESPONDER_SOCKET,))
    # resolve = False: a symlinked /etc/gitconfig must not grant the home. Editable roots as literals, not subpaths.
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
        # AFTER the read allowances (Seatbelt is last-match-wins), or a Studio home under a read root leaks auth.db.
        *_studio_state_rules(runtime_paths, developer_paths, workdir, private_tmp),
        _rule("allow file-write*", write_filters),
        # AFTER the allowance (last-match-wins): Studio's runtime stays read-only even under the workdir.
        *(
            [_rule("deny file-write*", _path_filters(runtime_under))]
            if (runtime_under := runtime_paths_under(workdir))
            else []
        ),
        _rule("allow file-read* file-test-existence file-write-data", device_filters),
        # bash process substitution opens /dev/fd/63.
        '(allow file-read* (regex #"^/dev/fd/[0-9]+$"))',
        '(allow file-write* (regex #"^/dev/fd/[0-9]+$"))',
        _rule("allow file-ioctl", device_filters),
        "(allow ipc-posix-sem)",
        # write-data: libomp writes its registration into the segment.
        "(allow ipc-posix-shm-read-data ipc-posix-shm-write-create "
        "ipc-posix-shm-write-data ipc-posix-shm-write-unlink "
        '(ipc-posix-name-regex #"^/__KMP_REGISTERED_LIB_[0-9]+$"))',
        "(allow ipc-posix-shm-read-data ipc-posix-shm-write-create "
        "ipc-posix-shm-write-data ipc-posix-shm-write-unlink "
        '(ipc-posix-name-regex #"^/torch_[0-9]+_[0-9]+_[0-9]+$"))',
        "(allow ipc-posix-shm-read-data ipc-posix-shm-write-create "
        "ipc-posix-shm-write-data ipc-posix-shm-write-unlink "
        '(ipc-posix-name-regex #"^/psm_[0-9a-f]+$"))',
        # Unfiltered: configd resolvers create route and AF_SYSTEM sockets; network authority only, never filesystem.
        "(allow system-socket)",
        # IP only: an unfiltered bind covers AF_UNIX at paths no file rule governs.
        '(allow network-bind (local ip "*:*"))',
        "(allow network-inbound)",
        # Filtered to ip: an unfiltered outbound covers AF_UNIX, e.g. /var/run/docker.sock.
        '(allow network-outbound (remote ip "*:*"))',
        # One path filter per endpoint; combining paths breaks multiprocessing on macOS 15.
        *(f"(allow network-bind (local unix-socket {path}))" for path in tmp_subpaths),
        *(f"(allow network-outbound (remote unix-socket {path}))" for path in tmp_subpaths),
        # Both filter forms: which one a macOS release matches is untestable here.
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
    """The caller's TMPDIR when inside the workdir: ``private_tmp`` is deleted when the call ends."""
    requested = env.get("TMPDIR") or ""
    if requested and _within(posixpath.abspath(requested), workdir):
        return requested
    return private_tmp


def _sandbox_environment(env: dict[str, str], workdir: str, private_tmp: str) -> dict[str, str]:
    """HOME and TMPDIR must point inside the sandbox; DYLD_* is the linker-injection surface."""
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
            # <target>/bin LAST: it is tool-writable and must not shadow a command approval treats as safe.
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
        # A user-writable argv[0] must not reach Popen even if the probe missed it.
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
    # file-write* on the workdir writes through external hard links to the host inode.
    workdir_limitations = scan_workdir_for_host_channels(workdir)
    # /tmp, not /var/folders, keeping it out of the confidential per-user container.
    private_tmp = tempfile.mkdtemp(
        prefix = "us-seatbelt-", dir = "/tmp" if sys.platform == "darwin" else None
    )
    try:
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
