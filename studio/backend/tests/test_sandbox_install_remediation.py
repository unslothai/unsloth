# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The Linux remediation must be one copy-paste: the exact command for this host, never a promise."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from core.inference import os_sandbox

_INSTALL_SH = Path(__file__).resolve().parents[3] / "install.sh"


def _which_only(monkeypatch, present):
    monkeypatch.setattr(
        os_sandbox.shutil, "which", lambda name: f"/usr/bin/{name}" if name in present else None
    )


@pytest.mark.parametrize(
    ("managers", "command"),
    [
        ({"apt-get"}, "sudo apt-get install -y bubblewrap"),
        ({"apt-get", "dnf"}, "sudo apt-get install -y bubblewrap"),
        ({"dnf"}, "sudo dnf install -y bubblewrap"),
        ({"pacman"}, "sudo pacman -S --needed bubblewrap"),
        ({"zypper"}, "sudo zypper install -y bubblewrap"),
        ({"apk"}, "sudo apk add bubblewrap"),
    ],
)
def test_missing_bwrap_names_the_command_for_this_package_manager(monkeypatch, managers, command):
    _which_only(monkeypatch, managers)
    monkeypatch.setattr(os_sandbox, "_linux_userns_blocked_by_apparmor", lambda: False)
    remediation = os_sandbox.linux_unavailable_remediation()
    assert f"`{command}`" in remediation
    # Setup never installed bwrap for an ordinary user, so the text must not send them there.
    assert "re-run Studio setup" not in remediation


def test_missing_bwrap_on_ubuntu_with_userns_restriction_is_still_one_command(monkeypatch):
    # Measured on 24.04: installing bubblewrap alone leaves it "setting up uid map: Permission denied".
    _which_only(monkeypatch, {"apt-get"})
    monkeypatch.setattr(os_sandbox, "_linux_userns_blocked_by_apparmor", lambda: True)
    remediation = os_sandbox.linux_unavailable_remediation()
    assert (
        f"`sudo apt-get install -y bubblewrap && {os_sandbox._BWRAP_APPARMOR_FIX}`" in remediation
    )


def test_missing_bwrap_on_an_unknown_package_manager(monkeypatch):
    _which_only(monkeypatch, set())
    remediation = os_sandbox.linux_unavailable_remediation()
    assert "install bubblewrap with your distribution's package manager" in remediation


def test_apparmor_block_names_the_profile_ubuntu_ships(monkeypatch):
    _which_only(monkeypatch, {"bwrap", "apt-get"})
    monkeypatch.setattr(os_sandbox, "_linux_userns_blocked_by_apparmor", lambda: True)
    remediation = os_sandbox.linux_unavailable_remediation()
    # apparmor-profiles installs it under extra-profiles, disabled; it is not in /etc/apparmor.d.
    assert (
        "/usr/share/apparmor/extra-profiles/bwrap-userns-restrict /etc/apparmor.d/" in remediation
    )
    assert "apparmor_parser -r /etc/apparmor.d/bwrap-userns-restrict" in remediation
    assert "re-run Studio setup" not in remediation


def test_installer_and_backend_give_the_same_commands():
    text = _INSTALL_SH.read_text(encoding = "utf-8")
    fix = re.search(r'^_BWRAP_APPARMOR_FIX="([^"]+)"$', text, re.MULTILINE)
    assert fix and fix.group(1) == os_sandbox._BWRAP_APPARMOR_FIX
    body = text[text.index("_bwrap_install_command() {") :]
    body = body[: body.index("\n}\n")]
    shell = re.findall(r'command -v (\S+) >/dev/null 2>&1; then echo "([^"]+)"', body)
    assert shell == list(os_sandbox._BWRAP_INSTALL_COMMANDS)
