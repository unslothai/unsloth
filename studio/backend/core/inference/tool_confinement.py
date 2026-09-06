# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Filesystem confinement for a managed account's tool subprocesses.

The sandbox in ``tools.py`` gives a child a working directory, a scrubbed
environment, resource limits and a command blocklist. None of that stops a
process from opening ``../../<other account>/`` or the install root, which
holds the owner's data and the authentication database. With one account that
is the user's own machine and their own files; with several accounts it is a
boundary that has to hold at the operating-system level.

So, for a managed account only, every sandboxed child is confined before it
runs:

* Linux: a Landlock ruleset (kernel 5.13 and later, no privileges required)
  applied in the forked child. System directories and the interpreter stay
  readable and executable; the account's own workspace and temporary root are
  writable; everything else, including the owner's root, the shared model
  cache and the operating-system home directory, does not exist for the child.
  Landlock is inherited by every descendant and cannot be lifted.
* macOS: ``sandbox-exec`` with an equivalent profile wrapping the command.
* Anywhere else (Windows, a kernel without Landlock): the call is refused
  rather than run unconfined. The owner can set
  ``UNSLOTH_STUDIO_ALLOW_UNCONFINED_TOOLS=1`` to accept that risk knowingly.

The owner never enters this module's confined path: ``account_confinement``
returns ``None`` after one context read, so a single-account install spawns
its tools exactly as before.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

from utils.account_context import is_owner_context

_OVERRIDE_ENV = "UNSLOTH_STUDIO_ALLOW_UNCONFINED_TOOLS"

# Landlock syscall numbers are the same on every architecture.
_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_MAKE_SYM = 1 << 12
_FS_REFER = 1 << 13  # ABI 2
_FS_TRUNCATE = 1 << 14  # ABI 3
_FS_IOCTL_DEV = 1 << 15  # ABI 5
_SCOPE_SIGNAL = 1 << 1  # ABI 6: signals reach only processes inside the same domain
_FS_ABI1_MASK = (1 << 13) - 1

# Read and execute only. /proc and /sys are readable as they are for the
# owner's sandbox today (the parent already hides its environ).
_SYSTEM_READ_ROOTS = (
    "/usr",
    "/lib",
    "/lib32",
    "/lib64",
    "/bin",
    "/sbin",
    "/etc",
    "/opt",
    "/run",
    "/snap",
    "/nix",
    "/var/lib",
    "/sys",
)
# procfs is opened per entry, not as a whole: the child sees its own process
# and the machine-wide read-only facts, not another account's command lines.
# ``/proc/self`` is opened in the child, so it names the child's own entry.
_PROC_READ_PATHS = (
    "/proc/self",
    "/proc/thread-self",
    "/proc/cpuinfo",
    "/proc/meminfo",
    "/proc/stat",
    "/proc/uptime",
    "/proc/loadavg",
    "/proc/version",
    "/proc/filesystems",
    "/proc/devices",
    "/proc/sys",
)
_DEVICE_ROOT = "/dev"


class ToolConfinementUnavailable(RuntimeError):
    """This host cannot confine a managed account's tool process."""


@dataclass(frozen = True)
class Confinement:
    """How a managed account's child is confined on this host.

    ``preexec`` runs in the forked child after the ordinary sandbox pre-exec
    (Linux). ``wrap`` rewrites the argv (macOS). ``mechanism`` names what was
    applied, for logs and tests.
    """

    mechanism: str
    preexec: Optional[Callable[[], None]] = None
    wrapper: tuple[str, ...] = ()

    def wrap(self, argv: list[str]) -> list[str]:
        return [*self.wrapper, *argv] if self.wrapper else argv


def unconfined_tools_allowed() -> bool:
    return (os.environ.get(_OVERRIDE_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def refusal_message() -> str:
    return (
        "Code execution is unavailable for this account: this host cannot confine tool "
        "processes to your workspace (Landlock on Linux 5.13 or later, sandbox-exec on "
        f"macOS). The installation owner can set {_OVERRIDE_ENV}=1 to allow unconfined "
        "tool processes for managed accounts."
    )


def _existing(paths) -> list[str]:
    seen: list[str] = []
    for raw in paths:
        if not raw:
            continue
        try:
            path = os.path.realpath(raw)
        except (OSError, ValueError):
            continue
        if os.path.exists(path) and path not in seen:
            seen.append(path)
    return seen


def _interpreter_roots() -> list[str]:
    return _existing(
        (
            sys.prefix,
            sys.base_prefix,
            sys.exec_prefix,
            getattr(sys, "base_exec_prefix", ""),
            os.path.dirname(sys.executable),
            os.environ.get("VIRTUAL_ENV", ""),
        )
    )


def _writable_roots() -> list[str]:
    """The account's own roots, created if this is its first tool call: a rule
    can only name a path that exists, and the child's sandbox lives inside."""
    from utils.paths.storage_roots import tmp_root, workspace_root

    roots = []
    for root in (workspace_root(), tmp_root()):
        try:
            root.mkdir(parents = True, exist_ok = True)
        except OSError:
            continue
        roots.append(str(root))
    return _existing(roots)


# ---------------------------------------------------------------- Linux ----


class _RulesetAttr(ctypes.Structure):
    # Only the first field is passed; the kernel accepts the ABI 1 size from
    # every later ABI and treats the missing fields as zero.
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _ScopedRulesetAttr(ctypes.Structure):
    # ABI 6 layout, used only on a kernel that offers it.
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


_libc = None
if sys.platform == "linux":
    try:
        import ctypes.util
        _name = ctypes.util.find_library("c")
        _libc = ctypes.CDLL(_name, use_errno = True) if _name else None
    except (OSError, AttributeError):
        _libc = None

_landlock_abi: Optional[int] = None


def landlock_abi() -> int:
    """Highest Landlock ABI the running kernel offers, 0 when unavailable."""
    global _landlock_abi
    if _landlock_abi is not None:
        return _landlock_abi
    abi = 0
    if sys.platform == "linux" and _libc is not None:
        try:
            got = _libc.syscall(
                _SYS_LANDLOCK_CREATE_RULESET, None, 0, _LANDLOCK_CREATE_RULESET_VERSION
            )
            abi = int(got) if got > 0 else 0
        except (OSError, AttributeError, ValueError):
            abi = 0
    _landlock_abi = abi
    return abi


def _handled_mask(abi: int) -> int:
    mask = _FS_ABI1_MASK
    if abi >= 2:
        mask |= _FS_REFER
    if abi >= 3:
        mask |= _FS_TRUNCATE
    if abi >= 5:
        mask |= _FS_IOCTL_DEV
    return mask


def _landlock_rules(abi: int, sandbox_site_dir: str) -> list[tuple[str, int]]:
    handled = _handled_mask(abi)
    read = _FS_READ_FILE | _FS_READ_DIR | _FS_EXECUTE
    device = _FS_READ_FILE | _FS_WRITE_FILE | (_FS_IOCTL_DEV if abi >= 5 else 0)
    rules: list[tuple[str, int]] = []
    for path in _existing(_SYSTEM_READ_ROOTS):
        rules.append((path, read))
    for path in _existing(_PROC_READ_PATHS):
        # A rule on a plain file may not carry directory rights.
        rules.append((path, read if os.path.isdir(path) else read & ~_FS_READ_DIR))
    for path in _interpreter_roots():
        rules.append((path, read))
    for path in _existing((sandbox_site_dir,)):
        rules.append((path, read))
    for path in _existing((_DEVICE_ROOT,)):
        rules.append((path, device))
    # Everything but creating links: a link planted in the account's own tree
    # and pointing outside it is the one thing the server, which follows links
    # for the owner, must never find there.
    writable = handled & ~_FS_MAKE_SYM
    for path in _writable_roots():
        rules.append((path, writable))
    return rules


def _landlock_preexec(handled: int, rules: list[tuple[str, int]], scoped: int = 0) -> None:
    """Runs in the forked child: no imports, no allocation beyond ctypes."""
    libc = _libc
    attr = _ScopedRulesetAttr(handled, 0, scoped) if scoped else _RulesetAttr(handled)
    ruleset_fd = libc.syscall(
        _SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0
    )
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    try:
        for path, access in rules:
            parent_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                beneath = _PathBeneathAttr(access & handled, parent_fd)
                rc = libc.syscall(
                    _SYS_LANDLOCK_ADD_RULE,
                    ruleset_fd,
                    _LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(beneath),
                    0,
                )
                if rc < 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
            finally:
                os.close(parent_fd)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
        if libc.syscall(_SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset_fd)


def _linux_confinement(sandbox_site_dir: str) -> Optional[Confinement]:
    abi = landlock_abi()
    if abi <= 0:
        return None
    handled = _handled_mask(abi)
    rules = _landlock_rules(abi, sandbox_site_dir)
    return Confinement(
        mechanism = f"landlock-abi{abi}",
        # Signals stay inside the child's own domain (ABI 6), so a tool cannot
        # stop another account's tool process; the file rules already keep it
        # from reading that process's entry under /proc.
        preexec = partial(_landlock_preexec, handled, rules, _SCOPE_SIGNAL if abi >= 6 else 0),
    )


# ---------------------------------------------------------------- macOS ----


def _sbpl(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def macos_profile(
    *, read_roots: list[str], hidden_roots: list[str], writable_roots: list[str]
) -> str:
    """A sandbox-exec profile: later rules win, so the account's own roots are
    allowed after the install root and the home directory are denied."""
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow signal (target same-sandbox))",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow ipc-posix-shm)",
        "(allow network*)",
        "(allow file-read-metadata)",
        '(allow file-read* file-write* (subpath "/dev"))',
        '(allow file-read* (subpath "/private/tmp") (subpath "/private/var/db") '
        '(subpath "/private/var/folders") (subpath "/var/folders"))',
    ]
    for path in read_roots:
        lines.append(f"(allow file-read* (subpath {_sbpl(path)}))")
    for path in hidden_roots:
        lines.append(f"(deny file-read* file-write* (subpath {_sbpl(path)}))")
    for path in writable_roots:
        lines.append(f"(allow file-read* file-write* (subpath {_sbpl(path)}))")
    return "\n".join(lines) + "\n"


def _macos_confinement(sandbox_site_dir: str) -> Optional[Confinement]:
    sandbox_exec = shutil.which("sandbox-exec")
    if not sandbox_exec:
        return None
    from utils.paths.storage_roots import studio_root

    read_roots = _existing(
        (
            "/usr",
            "/bin",
            "/sbin",
            "/etc",
            "/private/etc",
            "/System",
            "/Library",
            "/opt",
            "/Applications",
            *_interpreter_roots(),
            sandbox_site_dir,
        )
    )
    hidden_roots = _existing((str(studio_root()), os.path.expanduser("~")))
    profile = macos_profile(
        read_roots = read_roots,
        hidden_roots = hidden_roots,
        writable_roots = _writable_roots(),
    )
    return Confinement(mechanism = "sandbox-exec", wrapper = (sandbox_exec, "-p", profile))


# ------------------------------------------------------------- entry point --


def account_confinement(sandbox_site_dir: str) -> Optional[Confinement]:
    """The confinement for the acting account's next tool child.

    ``None`` for the installation owner, whose sandbox is unchanged. For a
    managed account, the platform mechanism, or ``ToolConfinementUnavailable``
    when the host has none and the owner has not opted out of confinement.
    """
    if is_owner_context():
        return None
    confinement = None
    if sys.platform == "linux":
        confinement = _linux_confinement(sandbox_site_dir)
    elif sys.platform == "darwin":
        confinement = _macos_confinement(sandbox_site_dir)
    if confinement is not None:
        return confinement
    if unconfined_tools_allowed():
        return Confinement(mechanism = "unconfined-by-owner")
    raise ToolConfinementUnavailable(refusal_message())


def reset_probe_for_tests() -> None:
    global _landlock_abi
    _landlock_abi = None
