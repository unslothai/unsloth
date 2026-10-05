# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""What Unsloth can set up for the OS sandbox from Settings: the Windows MXC runtime.

Detection only: nothing here runs a privileged command.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import sys
from pathlib import Path

WINDOWS_RUNTIME = "windows-runtime"
OPERATIONS = (WINDOWS_RUNTIME,)

# Same floor as mxc_probe: older Windows has no MXC ProcessContainer to run.
MXC_MIN_WINDOWS_BUILD = 26100


@dataclass(frozen = True)
class SetupPlan:
    platform: str
    # One of OPERATIONS, or None when there is nothing Unsloth can run here.
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
    """Nothing is cached here yet; kept so setup jobs reset every plan cache the same way."""


def windows_runtime_unsupported() -> str | None:
    """None when this Windows can run MXC; else "arch" (not x64) or "build" (older than 26100)."""
    import platform

    if platform.machine().lower() not in ("amd64", "x86_64"):
        return "arch"
    try:
        build = sys.getwindowsversion().build
    except AttributeError:  # not Windows
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
        return "'" + str(arg).replace("'", "''") + "'"

    return "\n".join("& " + " ".join(quote(arg) for arg in step) for step in steps)


def windows_runtime_install_command() -> list[str]:
    from . import mxc_runtime
    return [
        sys.executable,
        str(Path(__file__).resolve().parents[3] / "install_mxc_prebuilt.py"),
        "--install-dir",
        str(mxc_runtime._installed_package_root()),
    ]
