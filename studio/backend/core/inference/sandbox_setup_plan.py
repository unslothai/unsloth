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
OPERATIONS = (LINUX_INSTALL, WINDOWS_SETUP)

_APPARMOR_SYSCTL = "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
_APPARMOR_PROFILE = "/etc/apparmor.d/bwrap-userns-restrict"
_CACHE_SECONDS = 5.0

_lock = threading.Lock()
_cache: dict[tuple, tuple[float, "SetupPlan"]] = {}


@dataclass(frozen = True)
class SetupPlan:
    platform: str
    # Set only when Unsloth can run the steps itself: linux-install | windows-setup.
    action: str | None = None
    # sudo | pkexec (Linux), uac (Windows prepare), or None (no elevation available or needed).
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
    from . import os_sandbox
    try:
        snapshot = os_sandbox.capability_snapshot(
            execution_kind = "python", selected_executable = sys.executable
        )
    except Exception:  # noqa: BLE001 - an unreadable verdict is not a working sandbox
        return False
    return bool(snapshot.available)


# ---------------------------------------------------------------- Linux


def _argv(command: str) -> tuple[str, ...]:
    """`sudo apt-get install -y bubblewrap` -> ("apt-get", "install", "-y", "bubblewrap")."""
    parts = shlex.split(command)
    if parts and parts[0] == "sudo":
        parts = parts[1:]
    return tuple(parts)


def _apparmor_restricts_userns() -> bool:
    try:
        with open(_APPARMOR_SYSCTL, encoding = "utf-8") as f:
            return f.read().strip() == "1"
    except OSError:
        return False


def _package_manager() -> tuple[str, str] | tuple[None, None]:
    from . import os_sandbox
    for manager, command in os_sandbox._BWRAP_INSTALL_COMMANDS:
        if shutil.which(manager):
            return manager, command
    return None, None


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


def linux_elevation() -> tuple[str | None, str | None]:
    """(kind, trusted path): passwordless sudo first, then pkexec on a desktop session, else neither."""
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
    from . import os_sandbox

    reason = os_sandbox.linux_unavailable_remediation()
    manager, install = _package_manager()
    bwrap_missing = shutil.which("bwrap") is None
    if bwrap_missing and install is None:
        return SetupPlan(platform = sys.platform, reason = reason)
    commands: list[str] = []
    if bwrap_missing:
        commands.append(install)
    # Only Ubuntu's apt ships the profile; a host that already has it installed needs something else.
    if (
        manager == "apt-get"
        and _apparmor_restricts_userns()
        and (bwrap_missing or not os.path.exists(_APPARMOR_PROFILE))
        and (bwrap_missing or os_sandbox._linux_userns_blocked_by_apparmor())
    ):
        commands.extend(part.strip() for part in os_sandbox._BWRAP_APPARMOR_FIX.split("&&"))
    if not commands:
        return SetupPlan(platform = sys.platform, reason = reason)
    manual = " && ".join(commands)
    elevation, _path = linux_elevation()
    return SetupPlan(
        platform = sys.platform,
        action = LINUX_INSTALL if elevation else None,
        elevation = elevation,
        steps = tuple(_argv(command) for command in commands),
        manual_command = manual,
        reason = reason,
    )


# ---------------------------------------------------------------- Windows


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

    try:
        mxc_runtime.installation_identity()
        installed = True
    except Exception:  # noqa: BLE001 - no runtime yet
        installed = False
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
    manual = " && ".join(subprocess.list2cmdline(list(step)) for step in steps)
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
    owner = _is_owner(user)
    local = False
    try:
        local = is_direct_local_request(request)
    except Exception:  # noqa: BLE001 - an unreadable request is not local
        local = False
    can_run = bool(plan.action) and owner and local
    return {
        "setup_action": plan.action if can_run else None,
        "manual_command": plan.manual_command,
        "can_run_setup": can_run,
        "needs_consent": plan.needs_consent if can_run else False,
        "reason": plan.reason,
    }


def _is_owner(user) -> bool:
    if user is None:
        from utils.account_context import is_owner_context
        return is_owner_context()
    if isinstance(user, bool):
        return user
    return bool(getattr(user, "is_owner", False))  # AccountContext.is_owner is a property
