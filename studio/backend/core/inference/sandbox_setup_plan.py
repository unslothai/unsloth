# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""What it takes to get OS isolation on this computer, and whether Unsloth can do it from Settings.

Detection only: nothing here runs a privileged command. The same plan drives the setup job and the
command shown to people who cannot (or would rather not) let Unsloth run it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

LINUX_INSTALL = "linux-install"
WINDOWS_SETUP = "windows-setup"
# Settings > Sandbox "Install runtime": the non-elevated MXC runtime install only.
WINDOWS_RUNTIME = "windows-runtime"
OPERATIONS = (LINUX_INSTALL, WINDOWS_SETUP, WINDOWS_RUNTIME)

_APPARMOR_SYSCTL = "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
_APPARMOR_PROFILE = "/etc/apparmor.d/bwrap-userns-restrict"
_APPARMOR_EXTRA_PROFILE = "/usr/share/apparmor/extra-profiles/bwrap-userns-restrict"
_CACHE_SECONDS = 5.0
# The elevation check is logged by the host (auth.log) and may mail root: ask rarely, and only
# for the owner on this computer.
_ELEVATION_CACHE_SECONDS = 600.0
# Only these directories are trusted for anything run as root, whatever PATH says.
SYSTEM_BIN_DIRS = ("/usr/sbin", "/usr/bin", "/sbin", "/bin")
# The whole environment an elevated setup step sees.
ELEVATED_ENV = {
    "PATH": ":".join(SYSTEM_BIN_DIRS),
    "LANG": "C",
    "LC_ALL": "C",
    "DEBIAN_FRONTEND": "noninteractive",
}

# Per package manager, non-interactive. The package names and paths match os_sandbox's
# _BWRAP_INSTALL_COMMANDS / _BWRAP_APPARMOR_FIX (which install.sh mirrors); only the flags differ.
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
    # linux-install | windows-setup | windows-runtime when there is something Unsloth can set up
    # here. On Linux, whether it can run it itself also takes an elevation (linux_elevation()).
    action: str | None = None
    # uac (Windows prepare) or None; Linux elevation is checked per request (linux_elevation()).
    elevation: str | None = None
    steps: tuple[tuple[str, ...], ...] = ()
    manual_command: str = ""
    reason: str = ""
    # Windows: the MXC opt-in is off, so setup also needs the owner's consent to turn it on.
    needs_consent: bool = False

    def as_dict(self) -> dict:
        data = asdict(self)
        data["steps"] = [list(step) for step in self.steps]
        return data


def invalidate() -> None:
    with _lock:
        _cache.clear()


def detect(available: bool | None = None, *, force: bool = False) -> SetupPlan:
    """The setup plan for this host. `available`: whether OS isolation already works (probed if None)."""
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
            reason = "Seatbelt is built into macOS, so there is nothing to install; the live check did not pass.",
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


# ---------------------------------------------------------------- Linux


def manual_command(steps) -> str:
    """What to paste into a terminal: the same steps, each under sudo."""
    return " && ".join(f"sudo {shlex.join(step)}" for step in steps)


def trusted_system_binary(name: str) -> str | None:
    """`name` from the system bin directories only (never PATH), root-owned and not writable by
    others, as is its directory; None otherwise. What an elevated step runs."""
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
    """A root-owned, non-writable system executable (sudo, pkexec), or None."""
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
    """(kind, trusted path): passwordless sudo first, then pkexec on a desktop session, else neither.

    Runs a non-interactive sudo check, so only for the owner on this computer (and, forced, right
    before a setup starts); remembered for ten minutes.
    """
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
    sudo = _trusted_tool("sudo")
    if sudo is not None and _sudo_without_password(sudo):
        return "sudo", sudo
    if _is_wsl() or not _graphical_session():
        return None, None
    pkexec = _trusted_tool("pkexec")
    if pkexec is not None:
        return "pkexec", pkexec
    return None, None


def _linux_plan() -> SetupPlan:
    """Reads files and PATH only; whether Unsloth may elevate is linux_elevation(), per request."""
    from . import os_sandbox

    reason = os_sandbox.linux_unavailable_remediation()
    manager = _package_manager()
    bwrap_missing = shutil.which("bwrap") is None
    if bwrap_missing and manager is None:
        return SetupPlan(platform = sys.platform, reason = reason)
    steps: list[tuple[str, ...]] = []
    if bwrap_missing:
        steps.extend(_INSTALL_STEPS[manager])
    blocked = _apparmor_restricts_userns() and (
        bwrap_missing or os_sandbox._linux_userns_blocked_by_apparmor()
    )
    if blocked and os.path.exists(_APPARMOR_PROFILE):
        # The profile is there but not in force (or was loaded before bwrap was): load it again.
        steps.append(_apparmor_load())
    elif blocked and manager == "apt-get":
        # Only Ubuntu's apt ships the profile.
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
        reason = reason,
    )


# ---------------------------------------------------------------- Windows


_ARM64_NOTE = (
    "The MXC runtime Unsloth installs is built for x64 Windows, and this PC is not x64, so there "
    "is nothing Unsloth can install here yet."
)


def windows_runtime_supported() -> bool:
    """The MXC runtime Unsloth installs is built for x64 Windows only."""
    import platform
    return platform.machine().lower() in ("amd64", "x86_64")


def windows_runtime_installed() -> bool:
    from . import mxc_runtime
    try:
        mxc_runtime.installation_identity()
    except Exception:  # noqa: BLE001 - no runtime yet
        return False
    return True


def windows_runtime_plan() -> SetupPlan:
    """Settings > Sandbox "Install runtime": the runtime alone, no administrator prompt."""
    if windows_runtime_installed():
        return SetupPlan(platform = sys.platform, reason = "The MXC runtime is already installed.")
    if not windows_runtime_supported():
        return SetupPlan(platform = sys.platform, reason = _ARM64_NOTE)
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
        return "'" + str(arg).replace("'", "''") + "'"

    return "\n".join("& " + " ".join(quote(arg) for arg in step) for step in steps)


def windows_runtime_install_command() -> list[str]:
    """The non-elevated MXC runtime install, as setup.ps1 runs it."""
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

    installed = windows_runtime_installed()
    if not installed and not windows_runtime_supported():
        return SetupPlan(platform = sys.platform, reason = _ARM64_NOTE)
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
        parts.append("OS isolation on this Windows version is not allowed yet")
    if installed and missing:
        parts.append(f"host preparation is missing ({', '.join(missing)})")
    elif installed and missing is None:
        parts.append("host preparation could not be confirmed")
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


# ---------------------------------------------------------------- per request


def setup_fields_for(
    request,
    user = None,
    *,
    available: bool | None = None,
) -> dict:
    """What a capability response may say about setup: the command for everyone, the action only for
    the installation owner on a direct local request (the prompt appears on this computer)."""
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
        # Why there is no button, so the page does not tell the owner at this computer to find the owner.
        blocked = "not_owner" if not owner else "not_local" if not local else "no_elevation"
    # Windows commands carry this install's interpreter, script and runtime paths (the account name
    # and home), which other accounts and API keys are not shown anywhere else; the Linux package
    # commands are the same on every host.
    command = plan.manual_command if owner or not sys.platform.startswith("win") else ""
    return {
        "setup_action": plan.action if can_run else None,
        "manual_command": command,
        "can_run_setup": can_run,
        "setup_blocked": blocked,
        "needs_consent": plan.needs_consent if can_run else False,
        "reason": plan.reason,
    }


def can_run_here(plan: SetupPlan, *, owner: bool, local: bool) -> bool:
    """Whether this caller gets the setup button. Blocking: on Linux it may run the elevation check,
    and only for the owner on this computer, so nobody else makes Unsloth run a host command."""
    if not (plan.action and owner and local):
        return False
    if plan.action == LINUX_INSTALL:
        return linux_elevation()[0] is not None
    return True


def _is_owner(user) -> bool:
    if user is None:
        from utils.account_context import is_owner_context
        return is_owner_context()
    if isinstance(user, bool):
        return user
    return bool(getattr(user, "is_owner", False))  # AccountContext.is_owner is a property
