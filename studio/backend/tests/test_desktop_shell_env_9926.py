# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth#9926: a desktop launch must end with the ROCm environment a terminal has.

The report is an RX 7600 (gfx1102) where training SIGSEGVs the backend from the
desktop app and trains cleanly from ``unsloth studio`` in a terminal, with
``HSA_OVERRIDE_GFX_VERSION`` / ``ROCM_PATH`` / ``USE_CK`` set in ``~/.bashrc``.
The desktop app reads the login shell and keeps only PATH out of it
(``fix_path_env::fix()`` is ``fix_vars(&["PATH"])``).

What these tests pin down is the shape of the fix rather than the crash, which
needs the card: the import happens only on an AMD host, only for names that are
absent, and only for names on the list.
"""

from __future__ import annotations

import os
import signal
import sys
import time

import pytest

from utils import desktop_shell_env as dse


def desktop(**extra) -> dict:
    """An environment as the desktop app hands it over: marked, and ROCm-empty."""
    return {dse.DESKTOP_MANAGED_ENV: "1", **extra}


# --------------------------------------------------------------------------
# The vendor gate. This is the whole isolation argument: on a host with no AMD
# GPU nothing is read and nothing is written, so an NVIDIA / Intel / Apple
# launch is byte-identical to the release before this change.
# --------------------------------------------------------------------------


def test_no_amd_gpu_imports_nothing_and_spawns_no_shell(monkeypatch):
    monkeypatch.setattr(dse, "host_has_amd_gpu", lambda: False)

    def _explode(*_a, **_k):
        raise AssertionError("the login shell must not be spawned without an AMD GPU")

    monkeypatch.setattr(dse, "read_login_shell_env", _explode)
    environ = desktop()
    assert dse.import_rocm_env_from_login_shell(environ = environ) == {}
    assert environ == desktop()


def test_amd_host_imports_the_missing_rocm_vars(monkeypatch):
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
    # PATH is the desktop app's job and it already does it; taking it here too
    # would overwrite a PATH that src-tauri built deliberately.
    assert environ["PATH"] == "/gui/bin"


# --------------------------------------------------------------------------
# Parity, not policy.
# --------------------------------------------------------------------------


def test_a_variable_already_set_is_never_overwritten():
    shell = {"HSA_OVERRIDE_GFX_VERSION": "11.0.0", "ROCM_PATH": "/opt/rocm"}
    environ = {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}
    assert dse.select_missing_vars(environ, shell) == {"ROCM_PATH": "/opt/rocm"}


def test_a_variable_exported_empty_counts_as_set():
    # `export HIP_VISIBLE_DEVICES=` is a deliberate statement, and replacing it
    # with the shell's value would be this module choosing a device.
    shell = {"HIP_VISIBLE_DEVICES": "0,1"}
    assert dse.select_missing_vars({"HIP_VISIBLE_DEVICES": ""}, shell) == {}


def test_an_empty_mask_in_the_shell_is_imported_as_empty():
    # `export ROCR_VISIBLE_DEVICES=` hides every agent. Dropping it would leave the
    # desktop launch holding cards the terminal launch does not have.
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
    # An NVIDIA, Intel or Apple variable on this list would make a GUI launch on
    # those stacks behave differently from the release before it.
    forbidden = ("CUDA", "NVIDIA", "NCCL", "ONEAPI", "SYCL", "LEVEL_ZERO", "MLX", "METAL")
    offenders = [
        name
        for name in dse.ROCM_SHELL_ENV_ALLOWLIST
        if any(token in name.upper() for token in forbidden)
    ]
    assert offenders == []


def test_importing_twice_is_a_no_op_the_second_time(monkeypatch):
    monkeypatch.setattr(dse, "host_has_amd_gpu", lambda: True)
    monkeypatch.setattr(dse, "read_login_shell_env", lambda *_a, **_k: {"ROCM_PATH": "/opt/rocm"})
    environ = desktop()
    assert dse.import_rocm_env_from_login_shell(environ = environ) == {"ROCM_PATH": "/opt/rocm"}
    assert dse.import_rocm_env_from_login_shell(environ = environ) == {}


# --------------------------------------------------------------------------
# Failing open. A slow or broken login shell must not fail a launch.
# --------------------------------------------------------------------------


def test_the_opt_out_short_circuits_before_anything_is_read(monkeypatch):
    def _explode(*_a, **_k):
        raise AssertionError("opted out, so nothing should be probed")

    monkeypatch.setattr(dse, "host_has_amd_gpu", _explode)
    environ = desktop(**{dse.DISABLE_ENV_VAR: "1"})
    assert dse.import_rocm_env_from_login_shell(environ = environ) == {}


def test_a_shell_that_fails_yields_nothing(monkeypatch):
    monkeypatch.setattr(dse, "host_has_amd_gpu", lambda: True)

    def _raise(*_a, **_k):
        raise OSError("no shell here")

    monkeypatch.setattr(dse.subprocess, "Popen", _raise)
    environ = desktop()
    assert dse.import_rocm_env_from_login_shell(environ = environ) == {}
    assert environ == desktop()


@pytest.mark.skipif(sys.platform == "win32", reason = "no POSIX shell on Windows")
def test_a_real_login_shell_round_trips_a_value(tmp_path, monkeypatch):
    """The parser, against a real shell rather than a mock.

    ``env -0`` is used rather than ``env`` precisely so a value containing a
    newline cannot split the record and corrupt the variable after it, so that
    is what this checks.
    """
    rc = tmp_path / "rc.sh"
    rc.write_text(
        'echo "banner from an rc file"\n'
        'export ROCM_PATH="/opt/rocm"\n'
        'export HSA_OVERRIDE_GFX_VERSION="11.0.0"\n'
        'export UNSLOTH_TEST_MULTILINE="one\ntwo"\n',
        encoding = "utf-8",
    )
    shim = tmp_path / "shell.sh"
    # A stand-in for the user's login shell: accepts -ilc and sources an rc file
    # first, which is the behaviour the real `-i` flag provides.
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
    # What must hold is that asking is cheap, total, and does not import torch.
    assert dse.host_has_amd_gpu() in (True, False)


# --------------------------------------------------------------------------
# The launch gate. The bug is that a DESKTOP launch loses the environment, so a
# launch that never lost it must read nothing at all.
# --------------------------------------------------------------------------


def test_a_launch_the_desktop_app_does_not_own_reads_no_shell(monkeypatch):
    def _explode(*_a, **_k):
        raise AssertionError("only a desktop launch lost its environment")

    monkeypatch.setattr(dse, "host_has_amd_gpu", _explode)
    monkeypatch.setattr(dse, "read_login_shell_env", _explode)
    # `unsloth studio` in a terminal, a systemd unit, a container: all of these
    # start the backend without the desktop app's marker.
    for environ in ({}, {dse.DESKTOP_MANAGED_ENV: "0"}, {dse.DESKTOP_MANAGED_ENV: ""}):
        before = dict(environ)
        assert dse.import_rocm_env_from_login_shell(environ = environ) == {}
        assert environ == before


# --------------------------------------------------------------------------
# The vendor guard. NVIDIA's open kernel module registers KFD nodes too, and
# theirs carry a nonzero gfx_target_version, so the node must be vendor checked.
# --------------------------------------------------------------------------


def test_an_amd_kfd_node_is_an_amd_gpu():
    amd = "cpu_cores_count 0\nsimd_count 32\ngfx_target_version 110200\nvendor_id 4098\n"
    assert dse._node_is_an_amd_gpu(amd) is True


def test_an_nvidia_open_driver_kfd_node_is_not_an_amd_gpu():
    nvidia = "cpu_cores_count 0\nsimd_count 32\ngfx_target_version 110000\nvendor_id 4318\n"
    assert dse._node_is_an_amd_gpu(nvidia) is False


def test_a_cpu_node_is_not_a_gpu():
    cpu = "cpu_cores_count 16\nsimd_count 0\ngfx_target_version 0\nvendor_id 4098\n"
    assert dse._node_is_an_amd_gpu(cpu) is False


# --------------------------------------------------------------------------
# Reading the shell, against a real process.
# --------------------------------------------------------------------------


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
    """An rc file that starts an agent must not cost the whole timeout.

    A backgrounded child inherits stdout, so capturing the environment through a
    pipe waits for that child rather than for the shell: 15 seconds, and then the
    environment the shell had already written correctly is thrown away.
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
    """The shell exits cleanly here, so nothing raises and nothing times out.

    What it started does not exit with it, and one orphan adopted by init per
    backend start is the cost of reading the environment, not something the user
    asked this probe to leave behind.
    """
    pidfile = tmp_path / "child.pid"
    shell = _shim(tmp_path, f"sleep 300 &\necho $! > {pidfile}\nexport ROCM_PATH=/opt/rocm\n")
    assert dse.read_login_shell_env(shell = shell).get("ROCM_PATH") == "/opt/rocm"
    child = int(pidfile.read_text().strip())
    if not _wait_for_exit(child):
        os.kill(child, signal.SIGKILL)
        raise AssertionError(f"pid {child} outlived the shell that started it")


@pytest.mark.skipif(sys.platform == "win32", reason = "no POSIX shell on Windows")
def test_the_timeout_takes_the_shells_children_with_it(tmp_path):
    """A shell that hangs must not leave its children running on every launch."""
    pidfile = tmp_path / "child.pid"
    shell = _shim(tmp_path, f"sleep 300 &\necho $! > {pidfile}\nsleep 300\n")
    assert dse.read_login_shell_env(shell = shell, timeout = 2.0) == {}
    child = int(pidfile.read_text().strip())
    if not _wait_for_exit(child):
        os.kill(child, signal.SIGKILL)
        raise AssertionError(f"pid {child} outlived the shell that started it")


@pytest.mark.skipif(sys.platform == "win32", reason = "no POSIX shell on Windows")
def test_a_value_the_filesystem_allows_but_utf8_does_not_round_trips(tmp_path):
    """A path carrying a non-UTF-8 byte must arrive as those bytes.

    os.environ itself uses surrogateescape, so decoding with `replace` would hand
    the backend a path with U+FFFD in it that no longer names the directory.
    """
    shell = _shim(tmp_path, "export ROCM_PATH=\"$(printf '/opt/rocm\\377')\"\n")
    env = dse.read_login_shell_env(shell = shell)
    assert os.fsencode(env["ROCM_PATH"]) == b"/opt/rocm\xff"


# --------------------------------------------------------------------------
# The #7331 interaction. The CLI guard runs against the GUI environment, which on
# a desktop launch never carried the override, so it clears nothing; the profile
# that exports it is the one read here. The arbiter travels, not its verdict.
# --------------------------------------------------------------------------


def _shell_with_the_reporters_override(monkeypatch):
    monkeypatch.setattr(dse, "host_has_amd_gpu", lambda: True)
    monkeypatch.setattr(
        dse,
        "read_login_shell_env",
        lambda *_a, **_k: {"HSA_OVERRIDE_GFX_VERSION": "11.0.0", "ROCM_PATH": "/opt/rocm"},
    )


def test_an_override_the_install_cannot_serve_is_not_imported(monkeypatch):
    _shell_with_the_reporters_override(monkeypatch)
    environ = desktop(**{dse.ROCM_INSTALLED_ARCH_ENV: "gfx1151"})
    assert dse.import_rocm_env_from_login_shell(environ = environ) == {"ROCM_PATH": "/opt/rocm"}
    assert dse.HSA_OVERRIDE_ENV not in environ


def test_an_override_the_install_can_serve_is_imported(monkeypatch):
    # 11.0.0 is gfx1100, and these wheels carry gfx1100 kernels.
    _shell_with_the_reporters_override(monkeypatch)
    environ = desktop(**{dse.ROCM_INSTALLED_ARCH_ENV: "gfx1100"})
    imported = dse.import_rocm_env_from_login_shell(environ = environ)
    assert imported["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0"


def test_an_install_that_is_not_single_arch_arbitrates_nothing(monkeypatch):
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
