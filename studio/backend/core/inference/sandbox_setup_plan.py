# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""What it takes to get OS isolation on this computer, and whether Unsloth can do it from Settings.

Detection only: nothing here runs a privileged command.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

LINUX_INSTALL = "linux-install"
WINDOWS_SETUP = "windows-setup"
WINDOWS_RUNTIME = "windows-runtime"
OPERATIONS = (LINUX_INSTALL, WINDOWS_SETUP, WINDOWS_RUNTIME)

_APPARMOR_SYSCTL = "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
_APPARMOR_PROFILE = "/etc/apparmor.d/bwrap-userns-restrict"
_APPARMOR_EXTRA_PROFILE = "/usr/share/apparmor/extra-profiles/bwrap-userns-restrict"
_CACHE_SECONDS = 5.0
# The elevation check is logged (auth.log) and may mail root: ask rarely.
_ELEVATION_CACHE_SECONDS = 600.0
SYSTEM_BIN_DIRS = ("/usr/sbin", "/usr/bin", "/sbin", "/bin")
ELEVATED_ENV = {
    "PATH": ":".join(SYSTEM_BIN_DIRS),
    "LANG": "C",
    "LC_ALL": "C",
    "DEBIAN_FRONTEND": "noninteractive",
}

# Must match os_sandbox's _BWRAP_INSTALL_COMMANDS / _BWRAP_APPARMOR_FIX (and install.sh).
_APT_INSTALL = ("apt-get", "-o", "DPkg::Lock::Timeout=120", "install", "-y")
_INSTALL_STEPS = {
    "apt-get": (("apt-get", "update"), (*_APT_INSTALL, "bubblewrap")),
    "dnf": (("dnf", "install", "-y", "bubblewrap"),),
    "pacman": (("pacman", "-S", "--needed", "--noconfirm", "bubblewrap"),),
    "zypper": (("zypper", "--non-interactive", "install", "bubblewrap"),),
    "apk": (("apk", "add", "bubblewrap"),),
}


def _apparmor_load() -> tuple[str, ...]:
    return ("apparmor_parser", "-r", _APPARMOR_PROFILE)


def _apparmor_steps() -> tuple[tuple[str, ...], ...]:
    return (
        (*_APT_INSTALL, "apparmor-profiles"),
        ("install", "-m", "644", _APPARMOR_EXTRA_PROFILE, "/etc/apparmor.d/"),
        _apparmor_load(),
    )


_lock = threading.Lock()
_cache: dict[tuple, tuple[float, "SetupPlan"]] = {}
_elevation_cache: "tuple[float, tuple[str | None, str | None]] | None" = None


@dataclass(frozen = True)
class SetupPlan:
    platform: str
    action: str | None = None
    elevation: str | None = None
    steps: tuple[tuple[str, ...], ...] = ()
    manual_command: str = ""
    reason: str = ""
    needs_consent: bool = False

    def as_dict(self) -> dict:
        data = asdict(self)
        data["steps"] = [list(step) for step in self.steps]
        return data


def invalidate() -> None:
    with _lock:
        _cache.clear()


def detect(available: bool | None = None, *, force: bool = False) -> SetupPlan:
    if available is None:
        available = _os_isolation_available()
    key = (sys.platform, bool(available))
    now = time.monotonic()
    with _lock:
        cached = _cache.get(key)
        if not force and cached is not None and now < cached[0]:
            return cached[1]
    if available:
        plan = SetupPlan(
            platform = sys.platform, reason = "OS isolation already works on this computer."
        )
    elif sys.platform == "win32":
        plan = _windows_plan()
    elif sys.platform == "darwin":
        plan = SetupPlan(
            platform = sys.platform,
            reason = "The macOS sandbox (Seatbelt) did not pass its live check.",
        )
    elif sys.platform.startswith("linux"):
        plan = _linux_plan()
    else:
        plan = SetupPlan(
            platform = sys.platform, reason = "No OS sandbox is supported on this platform."
        )
    with _lock:
        _cache[key] = (time.monotonic() + _CACHE_SECONDS, plan)
    return plan


def _os_isolation_available() -> bool:
    """Both tools, as the status and capability reads count it: Python alone is not "works"."""
    from . import os_sandbox
    try:
        return all(
            os_sandbox.capability_snapshot(
                execution_kind = tool,
                selected_executable = os_sandbox.tool_isolation_target(tool),
            ).available
            for tool in os_sandbox.ISOLATED_TOOLS
        )
    except Exception:  # noqa: BLE001 - an unreadable verdict is not a working sandbox
        return False


def _running_as_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def manual_command(steps) -> str:
    prefix = "" if _running_as_root() else "sudo "
    return " && ".join(f"{prefix}{shlex.join(step)}" for step in steps)


def trusted_system_binary(name: str) -> str | None:
    """`name` from SYSTEM_BIN_DIRS only (never PATH), root-owned and not writable by others."""
    from .sandbox_linux import _trusted_system_file

    for directory in SYSTEM_BIN_DIRS:
        candidate = os.path.join(directory, name)
        if not os.path.exists(candidate):
            continue
        resolved = os.path.realpath(candidate)
        if not _trusted_system_file(resolved):
            continue
        try:
            parent = os.stat(os.path.dirname(resolved))
        except OSError:
            continue
        if parent.st_uid == 0 and not parent.st_mode & 0o022:
            return resolved
    return None


def elevated_steps(steps) -> list[list[str]]:
    """Each step with its program pinned to a trusted absolute path; LookupError if one has none."""
    pinned = []
    for step in steps:
        program = trusted_system_binary(step[0])
        if program is None:
            raise LookupError(f"{step[0]} was not found in {', '.join(SYSTEM_BIN_DIRS)}")
        pinned.append([program, *step[1:]])
    return pinned


def _apparmor_restricts_userns() -> bool:
    try:
        with open(_APPARMOR_SYSCTL, encoding = "utf-8") as f:
            return f.read().strip() == "1"
    except OSError:
        return False


def _package_manager() -> str | None:
    from . import os_sandbox
    for manager, _command in os_sandbox._BWRAP_INSTALL_COMMANDS:
        if shutil.which(manager):
            return manager
    return None


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version", encoding = "utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _trusted_tool(name: str) -> str | None:
    from .sandbox_linux import _trusted_system_file

    candidate = shutil.which(name)
    if candidate is None:
        return None
    resolved = os.path.realpath(candidate)
    if not _trusted_system_file(resolved):
        return None
    try:
        parent = os.stat(os.path.dirname(resolved))
    except OSError:
        return None
    if parent.st_uid != 0 or parent.st_mode & 0o022:
        return None
    return resolved


def _sudo_without_password(sudo: str) -> bool:
    try:
        probe = subprocess.run(
            [sudo, "-n", "true"],
            stdin = subprocess.DEVNULL,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            timeout = 5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def _graphical_session() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def linux_elevation(*, force: bool = False) -> tuple[str | None, str | None]:
    """(kind, trusted path): already root, passwordless sudo, then pkexec on a desktop, else neither."""
    global _elevation_cache
    now = time.monotonic()
    with _lock:
        cached = _elevation_cache
    if not force and cached is not None and now < cached[0]:
        return cached[1]
    found = _linux_elevation()
    with _lock:
        _elevation_cache = (time.monotonic() + _ELEVATION_CACHE_SECONDS, found)
    return found


def forget_elevation() -> None:
    global _elevation_cache
    with _lock:
        _elevation_cache = None


def _linux_elevation() -> tuple[str | None, str | None]:
    if _running_as_root():
        return "root", None
    sudo = _trusted_tool("sudo")
    if sudo is not None and _sudo_without_password(sudo):
        return "sudo", sudo
    if _is_wsl() or not _graphical_session():
        return None, None
    pkexec = _trusted_tool("pkexec")
    if pkexec is not None:
        return "pkexec", pkexec
    return None, None


def _bwrap_state() -> str:
    """As the launcher sees it: "trusted", "missing", or "shadowed" (PATH finds an untrusted copy first)."""
    from . import sandbox_linux

    try:
        sandbox_linux._trusted_bwrap_path()
        return "trusted"
    except Exception:  # noqa: BLE001 - any refusal means the launcher will not run it
        pass
    if shutil.which("bwrap") is None or trusted_system_binary("bwrap") is None:
        return "missing"
    return "shadowed"


def _linux_plan() -> SetupPlan:
    from . import os_sandbox

    reason = os_sandbox.linux_unavailable_remediation()
    manager = _package_manager()
    state = _bwrap_state()
    if state == "shadowed":
        return SetupPlan(
            platform = sys.platform,
            reason = (
                f"{shutil.which('bwrap')} comes first on PATH but is not a root-owned system copy, "
                f"so Unsloth will not run it. Remove it, or put {os.path.dirname(trusted_system_binary('bwrap'))} "
                "ahead of it on PATH; nothing needs installing."
            ),
        )
    bwrap_missing = state == "missing"
    if bwrap_missing and manager is None:
        return SetupPlan(platform = sys.platform, reason = reason)
    steps: list[tuple[str, ...]] = []
    if bwrap_missing:
        steps.extend(_INSTALL_STEPS[manager])
    blocked = _apparmor_restricts_userns() and (
        bwrap_missing or os_sandbox._linux_userns_blocked_by_apparmor()
    )
    if blocked and os.path.exists(_APPARMOR_PROFILE):
        steps.append(_apparmor_load())
    elif blocked and manager == "apt-get":
        if not bwrap_missing:
            steps.append(("apt-get", "update"))
        steps.extend(_apparmor_steps())
    if not steps:
        return SetupPlan(platform = sys.platform, reason = reason)
    return SetupPlan(
        platform = sys.platform,
        action = LINUX_INSTALL,
        steps = tuple(steps),
        manual_command = manual_command(steps),
        reason = (
            "bubblewrap is not installed."
            if bwrap_missing
            else "Ubuntu's AppArmor setting stops bubblewrap from starting a sandbox."
        ),
    )


# Same floor as mxc_probe: older Windows has no MXC ProcessContainer to run.
MXC_MIN_WINDOWS_BUILD = 26100


def windows_runtime_unsupported() -> str | None:
    """None when this Windows can run MXC; else "arch" (not x64) or "build" (older than 26100)."""
    import platform

    if platform.machine().lower() not in ("amd64", "x86_64"):
        return "arch"
    try:
        build = sys.getwindowsversion().build
    except AttributeError:
        return None
    return "build" if build < MXC_MIN_WINDOWS_BUILD else None


def windows_runtime_supported() -> bool:
    """The MXC runtime Unsloth installs is built for x64 Windows 11 build 26100 or newer."""
    return windows_runtime_unsupported() is None


_UNSUPPORTED_NOTES = {
    "arch": "The MXC runtime is built for x64 Windows, and this PC is not x64.",
    "build": (
        f"The MXC sandbox needs Windows 11 build {MXC_MIN_WINDOWS_BUILD} (24H2) or newer, "
        "and this PC runs an older build."
    ),
}


def windows_runtime_installed() -> bool:
    from . import mxc_runtime
    try:
        mxc_runtime.installation_identity()
    except Exception:  # noqa: BLE001 - no runtime yet
        return False
    return True


def windows_runtime_plan() -> SetupPlan:
    if windows_runtime_installed():
        return SetupPlan(platform = sys.platform, reason = "The MXC runtime is already installed.")
    unsupported = windows_runtime_unsupported()
    if unsupported is not None:
        return SetupPlan(platform = sys.platform, reason = _UNSUPPORTED_NOTES[unsupported])
    step = tuple(windows_runtime_install_command())
    return SetupPlan(
        platform = sys.platform,
        action = WINDOWS_RUNTIME,
        steps = (step,),
        manual_command = powershell_command((step,)),
        reason = "The MXC runtime is not installed.",
    )


def powershell_command(steps) -> str:
    """What to paste into PowerShell (5.1 has no `&&`): one call-operator line per step."""

    def quote(arg: str) -> str:
        # PowerShell also ends a single-quoted string at a curly quote (about_Quoting_Rules).
        return "'" + re.sub("(['‘’‚‛])", r"\1\1", str(arg)) + "'"

    return "\n".join("& " + " ".join(quote(arg) for arg in step) for step in steps)


def windows_runtime_install_command() -> list[str]:
    from . import mxc_runtime
    return [
        sys.executable,
        str(Path(__file__).resolve().parents[3] / "install_mxc_prebuilt.py"),
        "--install-dir",
        str(mxc_runtime._installed_package_root()),
    ]


def _windows_plan() -> SetupPlan:
    from . import mxc_adapter, mxc_policy, mxc_probe, mxc_runtime
    from utils import mxc_isolation_settings as saved

    unsupported = windows_runtime_unsupported()
    if unsupported is not None:
        return SetupPlan(platform = sys.platform, reason = _UNSUPPORTED_NOTES[unsupported])
    installed = windows_runtime_installed()
    missing = None
    if installed:
        try:
            missing = mxc_runtime.probe_host_prep_steps(env = mxc_adapter._control_environment())
        except Exception:  # noqa: BLE001 - unknown: prepare anyway, the helper is idempotent
            missing = None
    opted_in = mxc_policy.dacl_fallback_enabled()
    locked_off = not opted_in and saved.locked_by_environment(mxc_policy.DACL_FALLBACK_ENV)
    steps: list[tuple[str, ...]] = []
    if not installed:
        steps.append(tuple(windows_runtime_install_command()))
    if not installed or missing is None or missing:
        steps.append(tuple(mxc_probe.host_prep_command()))
    manual = powershell_command(steps)
    if locked_off:
        return SetupPlan(
            platform = sys.platform,
            manual_command = manual,
            reason = (
                f"{mxc_policy.DACL_FALLBACK_ENV} is set in the environment Unsloth runs in and keeps "
                "MXC's AppContainer tier off."
            ),
        )
    if not steps and opted_in:
        return SetupPlan(
            platform = sys.platform,
            reason = "MXC is installed, allowed and prepared, but its live check did not pass.",
        )
    parts = []
    if not installed:
        parts.append("the MXC runtime is not installed")
    if not opted_in:
        parts.append("the Windows sandbox is not turned on yet")
    if installed and missing:
        again = list(missing) == ["prepare-null-device"]
        parts.append(
            "this PC needs its administrator step again after the restart"
            if again
            else "this PC still needs its one-time administrator step"
        )
    elif installed and missing is None:
        parts.append("Unsloth could not tell whether this PC had its administrator step")
    return SetupPlan(
        platform = sys.platform,
        action = WINDOWS_SETUP,
        elevation = "uac" if any("--prepare-host" in step for step in steps) else None,
        steps = tuple(steps),
        manual_command = manual,
        reason = _sentence("; ".join(parts)),
        needs_consent = not opted_in,
    )


def _sentence(text: str) -> str:
    return f"{text[:1].upper()}{text[1:]}." if text else ""


def setup_fields_for(
    request,
    user = None,
    *,
    available: bool | None = None,
) -> dict:
    """Setup fields for a capability response: the action only for the owner, here or without a prompt."""
    from utils.client_ip import is_direct_local_request

    plan = detect(available)
    local = False
    try:
        local = is_direct_local_request(request)
    except Exception:  # noqa: BLE001 - an unreadable request is not local
        local = False
    owner = _is_owner(user)
    can_run = can_run_here(plan, owner = owner, local = local)
    blocked = None
    if plan.action and not can_run:
        blocked = "not_owner" if not owner else "not_local" if not local else "no_elevation"
    # Windows commands leak this install's paths (account name, home): owner only.
    command = plan.manual_command if owner or not sys.platform.startswith("win") else ""
    return {
        "setup_action": plan.action if can_run else None,
        "manual_command": command,
        "can_run_setup": can_run,
        "setup_blocked": blocked,
        "needs_consent": plan.needs_consent if can_run else False,
        "reason": plan.reason,
    }


# Nothing to answer on this computer, so a remote owner session (e.g. behind Colab's proxy) may start it.
PROMPTLESS_ELEVATION = ("root", "sudo")


def can_run_here(plan: SetupPlan, *, owner: bool, local: bool) -> bool:
    """Whether this caller gets the setup button. Blocking: may run the Linux elevation check."""
    if not (plan.action and owner):
        return False
    if plan.action == LINUX_INSTALL:
        return linux_install_allowed(local = local)[0]
    return local


def linux_install_allowed(*, local: bool, force: bool = False) -> tuple[bool, str | None]:
    """(allowed, elevation kind): any elevation for a direct local request, otherwise only one with no prompt."""
    kind = linux_elevation(force = force)[0]
    return kind is not None and (local or kind in PROMPTLESS_ELEVATION), kind


def remote_start_allowed(operation: str) -> bool:
    """Whether a request that is not direct-local may start `operation`: only where no prompt appears here."""
    if operation == WINDOWS_RUNTIME:
        return True
    return (
        operation == LINUX_INSTALL
        and sys.platform.startswith("linux")
        and linux_install_allowed(local = False, force = True)[0]
    )


def _is_owner(user) -> bool:
    if user is None:
        from utils.account_context import is_owner_context
        return is_owner_context()
    if isinstance(user, bool):
        return user
    return bool(getattr(user, "is_owner", False))
