# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The OS sandbox setup plan: which fixed steps a host needs, and who may run them from Settings."""

import os
import sys

import pytest

from core.inference import os_sandbox, sandbox_setup_plan as plan_mod
from utils import client_ip


@pytest.fixture(autouse = True)
def _fresh():
    plan_mod.invalidate()
    yield
    plan_mod.invalidate()


@pytest.fixture
def linux(monkeypatch, tmp_path):
    """A Linux host whose PATH holds only the tools a test puts there."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("DISPLAY", raising = False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising = False)
    state = {"restricted": False, "blocked": False, "sudo_ok": True, "wsl": False}
    monkeypatch.setattr(plan_mod, "_apparmor_restricts_userns", lambda: state["restricted"])
    monkeypatch.setattr(os_sandbox, "_linux_userns_blocked_by_apparmor", lambda: state["blocked"])
    monkeypatch.setattr(plan_mod, "_APPARMOR_PROFILE", str(tmp_path / "bwrap-userns-restrict"))
    monkeypatch.setattr(plan_mod, "_is_wsl", lambda: state["wsl"])
    monkeypatch.setattr(plan_mod, "_sudo_without_password", lambda _sudo: state["sudo_ok"])
    # The fakes are not root-owned; trust is covered by its own test below.
    monkeypatch.setattr(plan_mod, "_trusted_tool", lambda name: plan_mod.shutil.which(name))

    def tool(*names):
        for name in names:
            path = bin_dir / name
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)

    state["tool"] = tool
    state["profile"] = tmp_path / "bwrap-userns-restrict"
    return state


def _manual_parts(command):
    return [part.strip() for part in command.split("&&")]


def test_apt_host_with_the_apparmor_restriction_gets_bwrap_and_the_profile(linux):
    linux["tool"]("apt-get", "sudo")
    linux["restricted"] = True
    plan = plan_mod.detect(False)
    assert plan.action == plan_mod.LINUX_INSTALL and plan.elevation == "sudo"
    assert plan.steps == (
        ("apt-get", "install", "-y", "bubblewrap"),
        ("apt-get", "install", "-y", "apparmor-profiles"),
        (
            "install",
            "-m",
            "644",
            "/usr/share/apparmor/extra-profiles/bwrap-userns-restrict",
            "/etc/apparmor.d/",
        ),
        ("apparmor_parser", "-r", "/etc/apparmor.d/bwrap-userns-restrict"),
    )
    # The copyable command is the same text install.sh and the remediation message print.
    expected = [dict(os_sandbox._BWRAP_INSTALL_COMMANDS)["apt-get"]]
    expected += _manual_parts(os_sandbox._BWRAP_APPARMOR_FIX)
    assert _manual_parts(plan.manual_command) == expected


def test_apt_host_without_the_restriction_only_installs_bwrap(linux):
    linux["tool"]("apt-get", "sudo")
    plan = plan_mod.detect(False)
    assert plan.steps == (("apt-get", "install", "-y", "bubblewrap"),)


@pytest.mark.parametrize("manager", ["dnf", "pacman", "zypper", "apk"])
def test_other_package_managers_install_bwrap_and_never_the_ubuntu_profile(linux, manager):
    linux["tool"](manager, "sudo")
    linux["restricted"] = True
    plan = plan_mod.detect(False)
    command = dict(os_sandbox._BWRAP_INSTALL_COMMANDS)[manager]
    assert plan.manual_command == command
    assert plan.steps == (tuple(command.split()[1:]),)


def test_installed_but_blocked_bwrap_only_needs_the_profile(linux):
    linux["tool"]("apt-get", "sudo", "bwrap")
    linux["restricted"] = linux["blocked"] = True
    plan = plan_mod.detect(False)
    assert [step[0] for step in plan.steps] == ["apt-get", "install", "apparmor_parser"]
    assert "bubblewrap" not in plan.manual_command


def test_a_host_that_already_has_the_profile_is_not_offered_it_again(linux):
    linux["tool"]("apt-get", "sudo", "bwrap")
    linux["restricted"] = linux["blocked"] = True
    linux["profile"].write_text("profile")
    plan = plan_mod.detect(False)
    assert plan.action is None and plan.manual_command == "" and plan.steps == ()
    assert plan.reason


def test_an_unknown_package_manager_gets_no_command(linux):
    plan = plan_mod.detect(False)
    assert plan.action is None and plan.manual_command == ""


def test_without_passwordless_sudo_a_desktop_uses_pkexec(linux, monkeypatch):
    linux["tool"]("apt-get", "sudo", "pkexec")
    linux["sudo_ok"] = False
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    plan = plan_mod.detect(False)
    assert plan.action == plan_mod.LINUX_INSTALL and plan.elevation == "pkexec"


def test_without_sudo_or_a_desktop_only_the_command_is_offered(linux):
    linux["tool"]("apt-get", "sudo", "pkexec")
    linux["sudo_ok"] = False
    plan = plan_mod.detect(False)
    assert plan.action is None and plan.elevation is None
    assert plan.manual_command.startswith("sudo apt-get install -y bubblewrap")


def test_wsl_never_uses_pkexec(linux, monkeypatch):
    linux["tool"]("apt-get", "sudo", "pkexec")
    linux["sudo_ok"] = False
    linux["wsl"] = True
    monkeypatch.setenv("DISPLAY", ":0")
    assert plan_mod.detect(False).action is None


def test_a_working_sandbox_needs_nothing(linux):
    linux["tool"]("apt-get", "sudo")
    plan = plan_mod.detect(True)
    assert plan.action is None and plan.steps == () and plan.manual_command == ""


def test_detection_is_cached_until_invalidated(linux):
    linux["tool"]("apt-get", "sudo")
    first = plan_mod.detect(False)
    linux["tool"]("bwrap")
    assert plan_mod.detect(False) is first
    plan_mod.invalidate()
    assert plan_mod.detect(False).steps == ()


def test_untrusted_elevation_tools_are_ignored(monkeypatch, tmp_path):
    fake = tmp_path / "sudo"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    if os.geteuid() != 0:
        assert plan_mod._trusted_tool("sudo") is None


def test_macos_has_nothing_to_install(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    plan = plan_mod.detect(False)
    assert plan.action is None and plan.manual_command == "" and "Seatbelt" in plan.reason


@pytest.fixture
def windows(monkeypatch):
    from core.inference import mxc_policy, mxc_probe, mxc_runtime
    from utils import mxc_isolation_settings

    monkeypatch.setattr(sys, "platform", "win32")
    state = {
        "installed": True,
        "missing": ("prepare-null-device",),
        "opted_in": False,
        "locked": False,
    }

    def identity():
        if not state["installed"]:
            raise FileNotFoundError("no runtime")
        return "identity"

    monkeypatch.setattr(mxc_runtime, "installation_identity", identity)
    monkeypatch.setattr(mxc_runtime, "probe_host_prep_steps", lambda **_kw: state["missing"])
    monkeypatch.setattr(mxc_policy, "dacl_fallback_enabled", lambda: state["opted_in"])
    monkeypatch.setattr(
        mxc_isolation_settings, "locked_by_environment", lambda _name: state["locked"]
    )
    monkeypatch.setattr(
        mxc_probe,
        "host_prep_command",
        lambda: ["python", "install_mxc_prebuilt.py", "--prepare-host"],
    )
    monkeypatch.setattr(
        plan_mod, "windows_runtime_install_command", lambda: ["python", "install_mxc_prebuilt.py"]
    )
    return state


def test_windows_without_the_runtime_installs_then_prepares(windows):
    windows["installed"] = False
    plan = plan_mod.detect(False)
    assert plan.action == plan_mod.WINDOWS_SETUP and plan.elevation == "uac"
    assert plan.steps == (
        ("python", "install_mxc_prebuilt.py"),
        ("python", "install_mxc_prebuilt.py", "--prepare-host"),
    )
    assert plan.needs_consent is True
    assert "MXC runtime is not installed" in plan.reason


def test_windows_after_a_reboot_only_prepares(windows):
    windows["opted_in"] = True
    plan = plan_mod.detect(False)
    assert plan.steps == (("python", "install_mxc_prebuilt.py", "--prepare-host"),)
    assert plan.needs_consent is False
    assert "prepare-null-device" in plan.reason


def test_windows_prepared_but_not_allowed_only_needs_consent(windows):
    windows["missing"] = ()
    plan = plan_mod.detect(False)
    assert plan.action == plan_mod.WINDOWS_SETUP and plan.steps == () and plan.needs_consent
    assert plan.elevation is None


def test_windows_allowed_and_prepared_has_nothing_left(windows):
    windows["missing"] = ()
    windows["opted_in"] = True
    assert plan_mod.detect(False).action is None


def test_windows_opt_in_locked_off_by_the_environment_is_not_offered(windows):
    windows["locked"] = True
    plan = plan_mod.detect(False)
    assert plan.action is None and "UNSLOTH_MXC_ALLOW_DACL_FALLBACK" in plan.reason
    assert "--prepare-host" in plan.manual_command


def test_setup_fields_name_the_action_only_for_the_owner_here(linux, monkeypatch):
    linux["tool"]("apt-get", "sudo")
    monkeypatch.setattr(client_ip, "is_direct_local_request", lambda _request: True)
    owner = plan_mod.setup_fields_for(object(), True, available = False)
    assert owner["setup_action"] == plan_mod.LINUX_INSTALL and owner["can_run_setup"] is True
    other = plan_mod.setup_fields_for(object(), False, available = False)
    assert other["setup_action"] is None and other["can_run_setup"] is False
    assert other["manual_command"] == owner["manual_command"] != ""
    monkeypatch.setattr(client_ip, "is_direct_local_request", lambda _request: False)
    remote = plan_mod.setup_fields_for(object(), True, available = False)
    assert remote["setup_action"] is None and remote["manual_command"]
