# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
OS-level sandbox wrapper for tool execution.
"""

import errno
import os
import shutil
import site
import stat
import subprocess
import sys
import threading

from loggers import get_logger

logger = get_logger(__name__)

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"
_BWRAP_PROBE_BIN = os.path.realpath(os.path.abspath(shutil.which("true") or "/usr/bin/true"))
_SANDBOX_SITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_site")
_NIX_STORE = "/nix/store"

_LINUX_ACCELERATOR_DEVICE_PATHS = (
    "/dev/dxg",
    "/dev/dri",
    "/dev/kfd",
    "/dev/accel",
    "/dev/nvidia-caps",
    "/dev/nvidiactl",
    "/dev/nvidia-modeset",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
)


class SandboxProfilePathError(ValueError):
    """A legal host path cannot be safely represented in a sandbox profile."""


class UnsafeSandboxWorkdirError(RuntimeError):
    """The writable workdir would expose an inode outside its path boundary."""


_sandbox_available_cache: bool | None = None
# Absolute path to ``bwrap``, resolved once at probe time so the runtime
# sandbox argv doesn't depend on the child's PATH (``_build_safe_env``
# strips PATH down to a fixed allow-list that won't cover Nix-style or
# custom-prefix installs).
_linux_bwrap_path: str | None = None
# Guards probe + cache so concurrent first-callers see a consistent
# (cache, bwrap_path) snapshot rather than racing on partial writes.
_sandbox_probe_lock = threading.Lock()

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
        proc = subprocess.run(
            argv,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            timeout = 5,
        )
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


def _linux_probe() -> _ProbeResult:
    """Smoke-test that ``bwrap`` can apply a minimal sandbox here.

    Catches the cases where the kernel refuses to create unprivileged
    user namespaces — surfacing at startup instead of first use
    """
    global _linux_bwrap_path
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        logger.warning("bwrap not found on PATH; tool execution will run unsandboxed")
        return _ProbeResult(ok = False)
    bwrap = os.path.realpath(os.path.abspath(bwrap))
    result = _probe(
        [
            bwrap,
            "--ro-bind",
            "/",
            "/",
            "--unshare-all",
            "--die-with-parent",
            _BWRAP_PROBE_BIN,
        ],
        "Linux bwrap",
    )
    if result.ok:
        _linux_bwrap_path = bwrap
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


def _assert_no_external_hardlinks(workdir: str) -> None:
    """Reject regular files whose inode has links outside *workdir*.

    A path-scoped writable bind/profile still exposes every inode reachable by
    an in-tree hard link. Count all in-tree aliases and compare them with the
    filesystem link count; internal-only hard links remain valid, while a
    pre-existing link to a host file fails closed before sandbox launch.
    """
    wd = os.path.realpath(workdir)
    inode_counts: dict[tuple[int, int], int] = {}
    inode_links: dict[tuple[int, int], int] = {}
    representatives: dict[tuple[int, int], str] = {}
    stack = [wd]

    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise UnsafeSandboxWorkdirError(
                f"cannot inspect sandbox workdir safely: {directory!r}: {exc}"
            ) from exc

        for entry in entries:
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
            if stat.S_ISDIR(entry_stat.st_mode):
                stack.append(entry.path)
                continue
            if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_nlink <= 1:
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


def opted_in_user_site_path() -> str | None:
    """Return the safe parent user-site path when the operator opts in."""
    if os.environ.get("UNSLOTH_STUDIO_SANDBOX_ALLOW_USER_SITE") != "1":
        return None
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


def _editable_source_paths() -> list[str]:
    """Source dirs registered by plain ``.pth`` or PEP 660 editable installs.

    Finder mappings are read from the parent's ``sys.modules`` and are valid
    for the child only while it shares ``sys.executable`` with the parent.
    """
    paths: list[str] = _plain_pth_source_paths()
    for name, mod in list(sys.modules.items()):
        if not (name.startswith("__editable___") and name.endswith("_finder")):
            continue
        paths.extend(getattr(mod, "MAPPING", {}).values())
        for ns_paths in getattr(mod, "NAMESPACES", {}).values():
            paths.extend(ns_paths)
    return paths


def _exec_chain_symlinks(executable: str) -> list[str]:
    """Symlinks encountered while resolving *executable* to its real binary.

    Returned paths are the symlinks themselves (not their targets). The
    Linux bwrap argv binds each one so that, inside the sandbox, the
    kernel can follow the chain during ``execve`` — otherwise it hits
    ``ENOENT`` on an intermediate symlink we never mounted.
    """
    out: list[str] = []
    seen_links: set[str] = set()
    current = executable
    for _ in range(40):  # cycle guard against pathological symlink loops
        parts = current.split(os.sep)
        prefix = "/"
        for p in parts[1:]:
            prefix = os.path.normpath(os.path.join(prefix, p))
            if prefix in seen_links:
                continue
            try:
                if os.path.islink(prefix):
                    seen_links.add(prefix)
                    out.append(prefix)
            except OSError as e:
                logger.debug("exec-chain islink(%s) failed: %s", prefix, e)
        try:
            if not os.path.islink(current):
                break
            target = os.readlink(current)
        except OSError as e:
            logger.debug("exec-chain readlink(%s) failed: %s", current, e)
            break
        if not target.startswith(os.sep):
            target = os.path.normpath(os.path.join(os.path.dirname(current), target))
        if target == current:
            break
        current = target
    return out


def _python_read_paths() -> list[str]:
    """Real dirs the Python interpreter needs to read at runtime.

    Returns ``sys.prefix``, ``sys.base_prefix``, the sandbox sitecustomize
    shim, system site-packages, user site-packages, and editable-install source dirs — all
    realpath-normalized, deduplicated, and filtered to existing dirs.
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
    candidates.extend(_editable_source_paths())

    resolved_candidates = [os.path.realpath(p) for p in candidates if p]
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
        if rp in seen or not os.path.isdir(rp):
            continue
        # A root-valued prefix would turn the deny-by-default sandbox into a
        # read-all-files profile or bind mount. Embedded Python builds and
        # root-prefix containers can legitimately report this value.
        if os.path.dirname(rp) == rp:
            logger.warning("Ignoring unsafe filesystem-root Python read path: %s", rp)
            continue
        # A prefix equal to $HOME (or an ancestor such as /home) exposes
        # credentials and unrelated user files. Narrow runtime dirs below HOME,
        # including ~/.local and project .venv directories, remain safe to bind.
        if real_home and _path_is_within(real_home, rp):
            logger.warning("Ignoring Python read path containing user home: %s", rp)
            continue
        seen.add(rp)
        out.append(rp)
    return out


def _macos_seatbelt_profile(workdir: str) -> str:
    """Build a Seatbelt profile string for ``sandbox-exec -p``."""
    _assert_no_external_hardlinks(workdir)
    python_read_paths = _python_read_paths()
    wd = _safe_subpath(os.path.realpath(workdir))
    py_subpaths = [f'(subpath "{_safe_subpath(p)}")' for p in python_read_paths]
    py_block = "\n    ".join(py_subpaths)
    nested_runtime_paths = [
        f'(subpath "{_safe_subpath(p)}")'
        for p in python_read_paths
        if _path_is_within(p, os.path.realpath(workdir), strict = True)
    ]
    runtime_write_deny_block = (
        "\n(deny file-write* file-ioctl\n    " + "\n    ".join(nested_runtime_paths) + "\n)"
        if nested_runtime_paths
        else ""
    )
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
            workdir_subpath,
        ]
    )

    homebrew_read_block = "\n    " + "\n    ".join(homebrew_subpaths) if homebrew_subpaths else ""

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
(allow ipc-posix-shm)
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
    {py_block}{homebrew_read_block}
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
{runtime_write_deny_block}
(allow file-write-data
    (require-all
        (path "/dev/null")
        (vnode-type CHARACTER-DEVICE)
    )
)

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
    """Read UNSLOTH_STUDIO_SANDBOX_NPROC on the host; default 10000.

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


# Import-time sanity: catch the case where a maintainer accidentally
# adds a literal `{` to the template (e.g. a dict literal) which would
# turn .format() into a KeyError at every tool call.
assert "12345" in _LINUX_NPROC_WRAPPER_TEMPLATE.format(
    nproc = 12345
), "_LINUX_NPROC_WRAPPER_TEMPLATE does not format cleanly"


def _linux_inner_rlimit_wrapper(inner_argv: list[str]) -> list[str]:
    """Wrap ``inner_argv`` with a tiny Python that sets RLIMIT_NPROC.

    Why: ``_sandbox_preexec_for_bwrap`` cannot call ``setrlimit(NPROC)``
    on the parent because that limit is per-real-UID and bwrap's setuid
    helper would EAGAIN on busy multi-tenant hosts where the operator
    already runs many processes. Inside the bwrap user namespace the
    counter is per-mapped-UID (typically ``nobody``), so applying NPROC
    there does not collide with the host UID's process count. The
    wrapper runs in the namespace, clamps NPROC to the configured value
    (or the inherited hard cap, whichever is smaller), then ``execvp``s
    the original argv so the LLM-controlled command runs with the cap.
    """
    exe = os.path.abspath(os.path.normpath(sys.executable))
    script = _LINUX_NPROC_WRAPPER_TEMPLATE.format(nproc = _resolve_nproc_limit())
    # Isolated mode keeps the model-writable CWD and PYTHONPATH from shadowing
    # stdlib modules imported by this trusted wrapper. execvp then launches the
    # requested command with its normal environment and import behavior.
    return [exe, "-I", "-c", script, *inner_argv]


def _linux_bwrap_argv(inner_argv: list[str], workdir: str) -> list[str]:
    """Build a ``bwrap`` argv for the Linux sandbox.

    Deny by omission: the child sees only what we bind-mount. ``net``
    is unshared without loopback, so all network is denied. ``/tmp``
    is a fresh tmpfs so writes don't leak to the host. The inner argv
    is wrapped with a small Python that re-applies RLIMIT_NPROC inside
    the userns (see :func:`_linux_inner_rlimit_wrapper`).
    """
    _assert_no_external_hardlinks(workdir)
    wd = os.path.realpath(workdir)
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

    assert _linux_bwrap_path is not None, "bwrap path unset despite successful probe"
    args: list[str] = [
        _linux_bwrap_path,
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    # ``--dev /dev`` creates only a minimal synthetic tree. Restore the host's
    # accelerator device nodes while retaining its DAC/cgroup permissions; the
    # child env separately preserves the operator's visible-device selectors.
    for device_path in _LINUX_ACCELERATOR_DEVICE_PATHS:
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
    # -try variants skip missing paths so the same argv works on
    # usrmerge distros (/lib, /lib64 are symlinks into /usr or absent).
    for d in top_ro_dirs:
        args.extend(["--ro-bind-try", d, d])
    for d in etc_ro_entries:
        args.extend(["--ro-bind-try", d, d])

    def _is_under_top_ro(path: str) -> bool:
        return any(path == top or path.startswith(top + os.sep) for top in top_ro_dirs)

    # _python_read_paths() already realpaths, filters non-dirs, dedupes,
    # and includes editable-install source dirs (so `pip install -e .`
    # repos like unsloth remain readable inside the sandbox).
    python_read_paths = _python_read_paths()
    for rp in python_read_paths:
        if _is_under_top_ro(rp):
            continue
        args.extend(["--ro-bind-try", rp, rp])

    # Bind exec-chain symlinks whose parent isn't already covered by
    # an existing bind — binding into a read-only mount fails; symlinks
    # under an existing bind are already reachable via path inheritance.
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
        if sym in bound_links or _is_under_top_ro(sym):
            continue
        parent = os.path.dirname(sym)
        if any(parent == b or parent.startswith(b + os.sep) for b in bound_dests):
            continue
        bound_links.add(sym)
        args.extend(["--ro-bind-try", sym, sym])

    args.extend(["--bind", wd, wd])
    # A later writable parent bind hides earlier nested read-only mounts.
    # Reapply interpreter/runtime paths below the workdir after that bind.
    for rp in python_read_paths:
        if _path_is_within(rp, wd, strict = True):
            args.extend(["--ro-bind-try", rp, rp])

    args.append("--")
    args.extend(_linux_inner_rlimit_wrapper(inner_argv))
    return args


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
