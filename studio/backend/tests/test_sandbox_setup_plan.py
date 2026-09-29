# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The OS sandbox setup plan: which fixed steps a host needs, and who may run them from Settings."""

import os
import platform
import sys

import pytest

from core.inference import os_sandbox, sandbox_setup_plan as plan_mod
from utils import client_ip


@pytest.fixture(autouse = True)
def _fresh():
    plan_mod.invalidate()
    plan_mod.forget_elevation()
    yield
    plan_mod.invalidate()
    plan_mod.forget_elevation()


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
    state["sudo_checks"] = 0

    def sudo_check(_sudo):
        state["sudo_checks"] += 1
        return state["sudo_ok"]

    monkeypatch.setattr(plan_mod, "_sudo_without_password", sudo_check)
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


UPDATE = ("apt-get", "update")
APT = ("apt-get", "-o", "DPkg::Lock::Timeout=120", "install", "-y")


def _profile_steps(profile):
    return (
        (*APT, "apparmor-profiles"),
        (
            "install",
            "-m",
            "644",
            "/usr/share/apparmor/extra-profiles/bwrap-userns-restrict",
            "/etc/apparmor.d/",
        ),
        ("apparmor_parser", "-r", str(profile)),
    )


@pytest.mark.skipif(os.name == "nt", reason = "resolves POSIX system binary paths")
def test_apt_host_with_the_apparmor_restriction_gets_bwrap_and_the_profile(linux):
    linux["tool"]("apt-get")
    linux["restricted"] = True
    plan = plan_mod.detect(False)
    assert plan.action == plan_mod.LINUX_INSTALL and plan.elevation is None
    # A stale package list failed the install on a real Ubuntu 24.04 host: update first.
    assert plan.steps == (UPDATE, (*APT, "bubblewrap"), *_profile_steps(linux["profile"]))
    assert plan.manual_command == " && ".join("sudo " + " ".join(step) for step in plan.steps)


def test_the_steps_install_what_install_sh_and_the_remediation_name():
    # Only the non-interactive flags differ from os_sandbox's constants (mirrored by install.sh).
    for manager, command in os_sandbox._BWRAP_INSTALL_COMMANDS:
        assert plan_mod._INSTALL_STEPS[manager][-1][-1] == command.split()[-1] == "bubblewrap"
    fix = os_sandbox._BWRAP_APPARMOR_FIX
    assert "apparmor-profiles" in fix and plan_mod._APPARMOR_EXTRA_PROFILE in fix
    assert " ".join(plan_mod._apparmor_load()) in fix


def test_apt_host_without_the_restriction_only_installs_bwrap(linux):
    linux["tool"]("apt-get")
    plan = plan_mod.detect(False)
    assert plan.steps == (UPDATE, (*APT, "bubblewrap"))


@pytest.mark.parametrize(
    "manager,flag",
    [("dnf", "-y"), ("pacman", "--noconfirm"), ("zypper", "--non-interactive"), ("apk", "add")],
)
def test_other_package_managers_install_bwrap_and_never_the_ubuntu_profile(linux, manager, flag):
    linux["tool"](manager)
    linux["restricted"] = True
    plan = plan_mod.detect(False)
    assert plan.steps == plan_mod._INSTALL_STEPS[manager] and len(plan.steps) == 1
    assert plan.steps[0][0] == manager and flag in plan.steps[0]
    assert "apparmor" not in plan.manual_command


def test_installed_but_blocked_bwrap_only_needs_the_profile(linux):
    linux["tool"]("apt-get", "bwrap")
    linux["restricted"] = linux["blocked"] = True
    plan = plan_mod.detect(False)
    assert plan.steps == (UPDATE, *_profile_steps(linux["profile"]))
    assert "bubblewrap" not in plan.manual_command


def test_a_profile_that_is_there_but_not_in_force_is_only_loaded(linux):
    linux["tool"]("apt-get", "bwrap")
    linux["restricted"] = linux["blocked"] = True
    linux["profile"].write_text("profile")
    plan = plan_mod.detect(False)
    assert plan.action == plan_mod.LINUX_INSTALL
    assert plan.steps == (("apparmor_parser", "-r", str(linux["profile"])),)


def test_an_unblocked_host_with_bwrap_needs_nothing(linux):
    linux["tool"]("apt-get", "bwrap")
    linux["restricted"] = True
    plan = plan_mod.detect(False)
    assert plan.action is None and plan.steps == () and plan.reason


def test_an_unknown_package_manager_gets_no_command(linux):
    plan = plan_mod.detect(False)
    assert plan.action is None and plan.manual_command == ""


def test_detection_never_runs_the_elevation_check(linux):
    linux["tool"]("apt-get", "sudo")
    plan_mod.detect(False)
    assert linux["sudo_checks"] == 0


@pytest.mark.skipif(os.name == "nt", reason = "resolves POSIX system binary paths")
def test_passwordless_sudo_is_the_first_choice(linux):
    linux["tool"]("sudo")
    kind, path = plan_mod.linux_elevation()
    assert kind == "sudo" and path.endswith("/sudo")


def test_without_passwordless_sudo_a_desktop_uses_pkexec(linux, monkeypatch):
    linux["tool"]("sudo", "pkexec")
    linux["sudo_ok"] = False
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert plan_mod.linux_elevation()[0] == "pkexec"


def test_without_sudo_or_a_desktop_only_the_command_is_offered(linux, monkeypatch):
    linux["tool"]("apt-get", "sudo", "pkexec")
    linux["sudo_ok"] = False
    monkeypatch.setattr(client_ip, "is_direct_local_request", lambda _request: True)
    assert plan_mod.linux_elevation() == (None, None)
    fields = plan_mod.setup_fields_for(object(), True, available = False)
    assert fields["setup_action"] is None and fields["can_run_setup"] is False
    assert fields["manual_command"].startswith("sudo apt-get update && sudo apt-get")


def test_wsl_never_uses_pkexec(linux, monkeypatch):
    linux["tool"]("sudo", "pkexec")
    linux["sudo_ok"] = False
    linux["wsl"] = True
    monkeypatch.setenv("DISPLAY", ":0")
    assert plan_mod.linux_elevation() == (None, None)


def test_the_elevation_check_is_remembered_until_forced(linux):
    linux["tool"]("sudo")
    plan_mod.linux_elevation()
    plan_mod.linux_elevation()
    assert linux["sudo_checks"] == 1
    plan_mod.linux_elevation(force = True)
    assert linux["sudo_checks"] == 2


@pytest.mark.parametrize("owner,local", [(False, True), (True, False), (False, False)])
def test_nobody_but_the_owner_here_makes_unsloth_check_sudo(linux, monkeypatch, owner, local):
    linux["tool"]("apt-get", "sudo")
    monkeypatch.setattr(client_ip, "is_direct_local_request", lambda _request: local)
    fields = plan_mod.setup_fields_for(object(), owner, available = False)
    assert linux["sudo_checks"] == 0
    assert fields["setup_action"] is None and fields["manual_command"]


def test_elevated_steps_ignore_path_and_untrusted_folders(monkeypatch, tmp_path):
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / "apt-get").write_text("#!/bin/sh\n")
    (planted / "apt-get").chmod(0o755)
    monkeypatch.setenv("PATH", str(planted))
    # A folder that is not root-owned is never trusted, whatever it holds.
    monkeypatch.setattr(plan_mod, "SYSTEM_BIN_DIRS", (str(planted),))
    assert plan_mod.trusted_system_binary("apt-get") is None
    with pytest.raises(LookupError):
        plan_mod.elevated_steps([("apt-get", "update")])


@pytest.mark.skipif(not os.path.exists("/usr/bin/env"), reason = "needs a POSIX /usr/bin")
def test_elevated_steps_pin_a_root_owned_system_binary():
    if os.stat("/usr/bin/env").st_uid != 0:
        pytest.skip("this host's /usr/bin is not root-owned")
    assert plan_mod.elevated_steps([("env", "true")]) == [
        [os.path.realpath("/usr/bin/env"), "true"]
    ]


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


@pytest.mark.skipif(os.name == "nt", reason = "POSIX ownership")
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
    # An x64 Windows host whatever this test runs on (an Apple Silicon Mac reads as arm64).
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
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


def test_windows_arm64_is_not_offered_a_runtime_it_cannot_run(windows, monkeypatch):
    windows["installed"] = False
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")
    plan = plan_mod.detect(False)
    assert plan.action is None and "x64" in plan.reason
    assert plan_mod.windows_runtime_plan().action is None


def test_windows_runtime_only_installs_the_runtime(windows, monkeypatch):
    windows["installed"] = False
    plan = plan_mod.windows_runtime_plan()
    assert plan.action == plan_mod.WINDOWS_RUNTIME and plan.elevation is None
    assert plan.steps == (("python", "install_mxc_prebuilt.py"),)
    windows["installed"] = True
    assert plan_mod.windows_runtime_plan().action is None


def test_both_tools_must_work_before_nothing_is_offered(monkeypatch):
    seen = []

    def snapshot(*, execution_kind, selected_executable):
        seen.append(execution_kind)
        return os_sandbox.SandboxCapability(
            backend = "x",
            available = execution_kind == "python",
            reason = "",
            protection_state = "preview",
            limitations = (),
        )

    monkeypatch.setattr(os_sandbox, "capability_snapshot", snapshot)
    monkeypatch.setattr(os_sandbox, "tool_isolation_target", lambda tool: tool)
    assert plan_mod._os_isolation_available() is False
    assert seen == ["python", "terminal"]


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


def test_windows_commands_paste_into_windows_powershell():
    # Windows PowerShell 5.1 has no `&&`, and a quoted path needs the call operator.
    steps = (
        (
            r"C:\Users\Jane Doe\python.exe",
            r"C:\Users\Jane Doe\studio\install_mxc_prebuilt.py",
            "--install-dir",
            "x",
        ),
        (r"C:\it's\python.exe", "--prepare-host"),
    )
    assert plan_mod.powershell_command(steps) == (
        "& 'C:\\Users\\Jane Doe\\python.exe' 'C:\\Users\\Jane Doe\\studio\\install_mxc_prebuilt.py' "
        "'--install-dir' 'x'\n& 'C:\\it''s\\python.exe' '--prepare-host'"
    )
