# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth#9926: a desktop launch must end with the ROCm environment a terminal has.

An RX 7600 SIGSEGVs from the desktop app and trains from ``unsloth studio``,
because ``fix_path_env::fix()`` is ``fix_vars(&["PATH"])`` and keeps only PATH
out of the login shell. These pin the shape of the fix, not the crash, which
needs the card.
"""

from __future__ import annotations

import os
import signal
import sys
import time

import pytest

from utils import desktop_shell_env as dse


@pytest.fixture
def linux(monkeypatch):
    """The module returns before anything else off Linux, so pin the platform.

    Without this the import tests pass on a Windows or macOS runner by asserting
    the early return, which is how staging CI caught them asserting nothing.
    """
    monkeypatch.setattr(dse.sys, "platform", "linux")


def desktop(**extra) -> dict:
    """An environment as the desktop app hands it over: marked, and ROCm-empty."""
    return {dse.DESKTOP_MANAGED_ENV: "1", **extra}


def test_no_amd_gpu_imports_nothing_and_spawns_no_shell(linux, monkeypatch):
    monkeypatch.setattr(dse, "host_has_amd_gpu", lambda: False)

    def _explode(*_a, **_k):
        raise AssertionError("the login shell must not be spawned without an AMD GPU")

    monkeypatch.setattr(dse, "read_login_shell_env", _explode)
    environ = desktop()
    assert dse.import_rocm_env_from_login_shell(environ = environ) == {}
    assert environ == desktop()


def test_amd_host_imports_the_missing_rocm_vars(linux, monkeypatch):
    monkeypatch.setattr(dse, "host_has_amd_gpu", lambda: True)
    monkeypatch.setattr(
        dse,
        "read_login_shell_env",
        lambda *_a, **_k: {
            "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
            "ROCM_PATH": "/opt/rocm",
            "USE_CK": "0",
            "PATH": "/usr/bin",
        },
    )
    environ = desktop(PATH = "/gui/bin")
    imported = dse.import_rocm_env_from_login_shell(environ = environ)
    assert imported == {
        "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
        "ROCM_PATH": "/opt/rocm",
        "USE_CK": "0",
    }
    assert environ["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0"
    # PATH is src-tauri's, deliberately built; taking it here would overwrite it.
    assert environ["PATH"] == "/gui/bin"


def test_a_variable_already_set_is_never_overwritten():
    shell = {"HSA_OVERRIDE_GFX_VERSION": "11.0.0", "ROCM_PATH": "/opt/rocm"}
    environ = {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}
    assert dse.select_missing_vars(environ, shell) == {"ROCM_PATH": "/opt/rocm"}


def test_a_variable_exported_empty_counts_as_set():
    # Replacing an exported empty would be this module choosing a device.
    shell = {"HIP_VISIBLE_DEVICES": "0,1"}
    assert dse.select_missing_vars({"HIP_VISIBLE_DEVICES": ""}, shell) == {}


def test_an_empty_mask_in_the_shell_is_imported_as_empty():
    # An exported empty hides every agent; dropping it hands back the cards.
    assert dse.select_missing_vars({}, {"ROCR_VISIBLE_DEVICES": ""}) == {"ROCR_VISIBLE_DEVICES": ""}


def test_only_allowlisted_names_are_imported():
    shell = {
        "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
        "CUDA_VISIBLE_DEVICES": "3",
        "AWS_SECRET_ACCESS_KEY": "hunter2",
        "LD_PRELOAD": "/tmp/evil.so",
        "PYTHONPATH": "/tmp/whatever",
    }
    assert dse.select_missing_vars({}, shell) == {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"}


def test_the_allowlist_carries_no_other_vendor():
    # One of these on the list changes a GUI launch on someone else's stack.
    forbidden = ("CUDA", "NVIDIA", "NCCL", "ONEAPI", "SYCL", "LEVEL_ZERO", "MLX", "METAL")
    offenders = [
        name
        for name in dse.ROCM_SHELL_ENV_ALLOWLIST
        if any(token in name.upper() for token in forbidden)
    ]
    assert offenders == []


def test_importing_twice_is_a_no_op_the_second_time(linux, monkeypatch):
    monkeypatch.setattr(dse, "host_has_amd_gpu", lambda: True)
    monkeypatch.setattr(dse, "read_login_shell_env", lambda *_a, **_k: {"ROCM_PATH": "/opt/rocm"})
    environ = desktop()
    assert dse.import_rocm_env_from_login_shell(environ = environ) == {"ROCM_PATH": "/opt/rocm"}
    assert dse.import_rocm_env_from_login_shell(environ = environ) == {}


def test_the_opt_out_short_circuits_before_anything_is_read(monkeypatch):
    def _explode(*_a, **_k):
        raise AssertionError("opted out, so nothing should be probed")

    monkeypatch.setattr(dse, "host_has_amd_gpu", _explode)
    environ = desktop(**{dse.DISABLE_ENV_VAR: "1"})
    assert dse.import_rocm_env_from_login_shell(environ = environ) == {}


def test_a_shell_that_fails_yields_nothing(linux, monkeypatch):
    monkeypatch.setattr(dse, "host_has_amd_gpu", lambda: True)

    def _raise(*_a, **_k):
        raise OSError("no shell here")

    monkeypatch.setattr(dse.subprocess, "Popen", _raise)
    environ = desktop()
    assert dse.import_rocm_env_from_login_shell(environ = environ) == {}
    assert environ == desktop()


@pytest.mark.skipif(sys.platform == "win32", reason = "no POSIX shell on Windows")
def test_a_real_login_shell_round_trips_a_value(tmp_path, monkeypatch):
    """``env -0`` is used so a value with a newline cannot split the record."""
    rc = tmp_path / "rc.sh"
    rc.write_text(
        'echo "banner from an rc file"\n'
        'export ROCM_PATH="/opt/rocm"\n'
        'export HSA_OVERRIDE_GFX_VERSION="11.0.0"\n'
        'export UNSLOTH_TEST_MULTILINE="one\ntwo"\n',
        encoding = "utf-8",
    )
    shim = tmp_path / "shell.sh"
    # Stands in for a login shell: takes -ilc and sources an rc file first.
    shim.write_text(
        "#!/bin/sh\n" f'. "{rc}"\n' "shift 1\n" 'exec /bin/sh -c "$1"\n',
        encoding = "utf-8",
    )
    shim.chmod(0o755)

    env = dse.read_login_shell_env(shell = str(shim))
    assert env.get("ROCM_PATH") == "/opt/rocm"
    assert env.get("HSA_OVERRIDE_GFX_VERSION") == "11.0.0"
    assert env.get("UNSLOTH_TEST_MULTILINE") == "one\ntwo"


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason = "the KFD topology only exists on Linux"
)
def test_the_amd_probe_answers_from_the_kernel_without_torch():
    # No assertion about the answer: this box may or may not have an AMD GPU.
    assert dse.host_has_amd_gpu() in (True, False)


def test_a_launch_the_desktop_app_does_not_own_reads_no_shell(linux, monkeypatch):
    def _explode(*_a, **_k):
        raise AssertionError("only a desktop launch lost its environment")

    monkeypatch.setattr(dse, "host_has_amd_gpu", _explode)
    monkeypatch.setattr(dse, "read_login_shell_env", _explode)
    # A terminal, a systemd unit, a container: none carry the marker.
    for environ in ({}, {dse.DESKTOP_MANAGED_ENV: "0"}, {dse.DESKTOP_MANAGED_ENV: ""}):
        before = dict(environ)
        assert dse.import_rocm_env_from_login_shell(environ = environ) == {}
        assert environ == before


def test_an_amd_kfd_node_is_an_amd_gpu():
    amd = "cpu_cores_count 0\nsimd_count 32\ngfx_target_version 110200\nvendor_id 4098\n"
    assert dse._node_is_an_amd_gpu(amd) is True


def test_an_nvidia_open_driver_kfd_node_is_not_an_amd_gpu():
    nvidia = "cpu_cores_count 0\nsimd_count 32\ngfx_target_version 110000\nvendor_id 4318\n"
    assert dse._node_is_an_amd_gpu(nvidia) is False


def test_a_cpu_node_is_not_a_gpu():
    cpu = "cpu_cores_count 16\nsimd_count 0\ngfx_target_version 0\nvendor_id 4098\n"
    assert dse._node_is_an_amd_gpu(cpu) is False


def _shim(tmp_path, rc_body: str):
    """A stand-in login shell: sources an rc file, then runs the -c command."""
    rc = tmp_path / "rc.sh"
    rc.write_text(rc_body, encoding = "utf-8")
    shim = tmp_path / "shell.sh"
    shim.write_text(
        "#!/bin/sh\n" f'. "{rc}"\n' "shift 1\n" 'exec /bin/sh -c "$1"\n', encoding = "utf-8"
    )
    shim.chmod(0o755)
    return str(shim)


@pytest.mark.skipif(sys.platform == "win32", reason = "no POSIX shell on Windows")
def test_a_background_job_in_the_rc_neither_stalls_nor_loses_the_environment(tmp_path):
    """An rc that starts an agent must not cost the whole timeout.

    The child inherits stdout, so a pipe capture waits for it, not for the shell.
    """
    shell = _shim(tmp_path, "sleep 45 &\nexport ROCM_PATH=/opt/rocm\n")
    started = time.monotonic()
    env = dse.read_login_shell_env(shell = shell, timeout = 15.0)
    assert time.monotonic() - started < 10.0
    assert env.get("ROCM_PATH") == "/opt/rocm"


def _wait_for_exit(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(0.1)
    return False


@pytest.mark.skipif(sys.platform == "win32", reason = "no POSIX shell on Windows")
def test_a_successful_read_still_takes_the_shells_children_with_it(tmp_path):
    """The shell exits cleanly, so nothing raises: its children still must go."""
    pidfile = tmp_path / "child.pid"
    shell = _shim(tmp_path, f"sleep 300 &\necho $! > {pidfile}\nexport ROCM_PATH=/opt/rocm\n")
    assert dse.read_login_shell_env(shell = shell).get("ROCM_PATH") == "/opt/rocm"
    child = int(pidfile.read_text().strip())
    if not _wait_for_exit(child):
        os.kill(child, signal.SIGKILL)
        raise AssertionError(f"pid {child} outlived the shell that started it")


@pytest.mark.skipif(sys.platform == "win32", reason = "no POSIX shell on Windows")
def test_the_timeout_takes_the_shells_children_with_it(tmp_path):
    """A hung shell must not leave its children running, once per launch."""
    pidfile = tmp_path / "child.pid"
    shell = _shim(tmp_path, f"sleep 300 &\necho $! > {pidfile}\nsleep 300\n")
    assert dse.read_login_shell_env(shell = shell, timeout = 2.0) == {}
    child = int(pidfile.read_text().strip())
    if not _wait_for_exit(child):
        os.kill(child, signal.SIGKILL)
        raise AssertionError(f"pid {child} outlived the shell that started it")


@pytest.mark.skipif(sys.platform == "win32", reason = "no POSIX shell on Windows")
def test_a_value_the_filesystem_allows_but_utf8_does_not_round_trips(tmp_path):
    """A path carrying a non-UTF-8 byte must arrive as those bytes."""
    shell = _shim(tmp_path, "export ROCM_PATH=\"$(printf '/opt/rocm\\377')\"\n")
    env = dse.read_login_shell_env(shell = shell)
    assert os.fsencode(env["ROCM_PATH"]) == b"/opt/rocm\xff"


def _shell_with_the_reporters_override(monkeypatch):
    monkeypatch.setattr(dse, "host_has_amd_gpu", lambda: True)
    monkeypatch.setattr(
        dse,
        "read_login_shell_env",
        lambda *_a, **_k: {"HSA_OVERRIDE_GFX_VERSION": "11.0.0", "ROCM_PATH": "/opt/rocm"},
    )


def test_an_override_the_install_cannot_serve_is_not_imported(linux, monkeypatch):
    _shell_with_the_reporters_override(monkeypatch)
    environ = desktop(**{dse.ROCM_INSTALLED_ARCH_ENV: "gfx1151"})
    assert dse.import_rocm_env_from_login_shell(environ = environ) == {"ROCM_PATH": "/opt/rocm"}
    assert dse.HSA_OVERRIDE_ENV not in environ


def test_an_override_the_install_can_serve_is_imported(linux, monkeypatch):
    # 11.0.0 is gfx1100, and these wheels carry gfx1100 kernels.
    _shell_with_the_reporters_override(monkeypatch)
    environ = desktop(**{dse.ROCM_INSTALLED_ARCH_ENV: "gfx1100"})
    imported = dse.import_rocm_env_from_login_shell(environ = environ)
    assert imported["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0"


def test_an_install_that_is_not_single_arch_arbitrates_nothing(linux, monkeypatch):
    """No marker means generic or multi-arch wheels, which contradict no override."""
    _shell_with_the_reporters_override(monkeypatch)
    imported = dse.import_rocm_env_from_login_shell(environ = desktop())
    assert imported["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("11.0.0", "gfx1100"),
        ("11.5.1", "gfx1151"),
        ("10.3.0", "gfx1030"),
        ("9.4.2", "gfx942"),
        # A stepping is a hex nibble, so 10 is "a" and 16 is not a target at all.
        ("11.0.10", "gfx110a"),
        ("11.0.16", None),
        ("11.0", None),
        ("", None),
        ("gfx1100", None),
        (None, None),
    ],
)
def test_the_override_parser_matches_the_cli(value, expected):
    """One arbiter, two processes: a launch must not depend on which saw it first."""
    import unsloth_cli.commands.studio as studio_cli

    assert dse.override_gfx_arch(value) == expected
    assert studio_cli._hsa_override_gfx_arch(value) == expected


def test_an_unreadable_override_is_not_ours_to_drop():
    """The CLI guard leaves a value it cannot parse alone, so this must too."""
    assert dse.override_contradicts_install("not-a-version", "gfx1151") is False


def test_the_cli_and_this_module_name_the_same_marker():
    """Two files, one contract: a rename on either side must fail here."""
    import unsloth_cli.commands.studio as studio_cli
    assert studio_cli.ROCM_INSTALLED_ARCH_ENV == dse.ROCM_INSTALLED_ARCH_ENV


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_no_other_platform_reads_a_shell(platform, monkeypatch):
    """Windows inherits the user environment already, and macOS has no KFD."""

    def _explode(*_a, **_k):
        raise AssertionError(f"{platform} must return before anything is read")

    monkeypatch.setattr(dse.sys, "platform", platform)
    monkeypatch.setattr(dse, "host_has_amd_gpu", _explode)
    monkeypatch.setattr(dse, "read_login_shell_env", _explode)
    environ = desktop()
    assert dse.import_rocm_env_from_login_shell(environ = environ) == {}
    assert environ == desktop()
