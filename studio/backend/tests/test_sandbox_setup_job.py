# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Settings sandbox setup job, run against fake sudo, pkexec and helper scripts on this host.

The fakes record their argv and exit with a chosen code; nothing privileged ever runs.
"""

import json
import os
import shlex
import sys
import time

import pytest

from core.inference import (
    mxc_host_prep_job,
    mxc_probe,
    os_sandbox,
    sandbox_probe,
    sandbox_setup_job as job_mod,
    sandbox_setup_plan as plan_mod,
    tools,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "the fakes are POSIX shell scripts")

_STEPS = (
    ("apt-get", "install", "-y", "bubblewrap"),
    ("apparmor_parser", "-r", "/etc/apparmor.d/bwrap-userns-restrict"),
)


@pytest.fixture
def env(monkeypatch, tmp_path):
    calls = {"resets": 0, "hooks": 0}
    record = tmp_path / "calls.jsonl"
    monkeypatch.setattr(job_mod, "_current", None)
    monkeypatch.setattr(job_mod, "_on_finish", [])
    monkeypatch.setattr(mxc_host_prep_job, "_current", None)

    def count():
        calls["resets"] += 1

    monkeypatch.setattr(sandbox_probe, "reset_probe_cache", count)
    monkeypatch.setattr(os_sandbox, "_linux_userns_blocked_by_apparmor", _Clearable(count))
    monkeypatch.setattr(mxc_probe, "invalidate_cache", count)
    monkeypatch.setattr(tools, "reset_terminal_profile_cache", count)
    monkeypatch.setattr(plan_mod, "invalidate", count)

    def fake(
        name,
        code = 0,
        output = "",
        gate = None,
    ):
        """A script that appends its argv to calls.jsonl, prints `output`, and exits `code`."""
        path = tmp_path / name
        wait = f'while [ ! -e "{gate}" ]; do sleep 0.05; done\n' if gate else ""
        path.write_text(
            "#!/bin/sh\n"
            f"{shlex.quote(sys.executable)} -c "
            '\'import json, sys; open(sys.argv[1], "a").write(json.dumps(sys.argv[2:]) + "\\n")\' '
            f'{shlex.quote(str(record))} "$0" "$@"\n'
            + (f"printf '%s\\n' {shlex.quote(output)}\n" if output else "")
            + wait
            + f"exit {code}\n"
        )
        path.chmod(0o755)
        return str(path)

    def recorded():
        if not record.exists():
            return []
        return [json.loads(line) for line in record.read_text().splitlines()]

    calls["fake"], calls["recorded"] = fake, recorded
    return calls


class _Clearable:
    def __init__(self, hook):
        self.cache_clear = hook

    def __call__(self):
        return False


def _linux_plan(monkeypatch, elevation, path):
    plan = plan_mod.SetupPlan(
        platform = "linux",
        action = plan_mod.LINUX_INSTALL,
        elevation = elevation,
        steps = _STEPS,
        manual_command = "sudo apt-get install -y bubblewrap && sudo apparmor_parser -r x",
    )
    monkeypatch.setattr(plan_mod, "detect", lambda *a, **k: plan)
    monkeypatch.setattr(plan_mod, "linux_elevation", lambda: (elevation, path))
    return plan


def _settle(job, seconds = 30):
    deadline = time.monotonic() + seconds
    while job.state == "running" and time.monotonic() < deadline:
        time.sleep(0.02)
    return job


def test_sudo_runs_each_fixed_step_non_interactively(monkeypatch, env):
    sudo = env["fake"]("sudo")
    _linux_plan(monkeypatch, "sudo", sudo)
    job_mod.add_finish_hook(lambda: env.__setitem__("hooks", env["hooks"] + 1))
    job = _settle(job_mod.start(plan_mod.LINUX_INSTALL))
    assert job.state == "succeeded" and job.exit_code == 0
    assert env["recorded"]() == [[sudo, "-n", *_STEPS[0]], [sudo, "-n", *_STEPS[1]]]
    assert env["resets"] >= 5 and env["hooks"] == 1


def test_sudo_wanting_a_password_stops_and_hands_over_the_command(monkeypatch, env):
    sudo = env["fake"]("sudo", code = 1, output = "sudo: a password is required")
    plan = _linux_plan(monkeypatch, "sudo", sudo)
    job = _settle(job_mod.start(plan_mod.LINUX_INSTALL))
    assert job.state == "failed" and "password" in job.note
    assert len(env["recorded"]()) == 1
    assert job.manual_command == plan.manual_command


def test_pkexec_asks_once_for_one_constant_script(monkeypatch, env):
    pkexec = env["fake"]("pkexec")
    _linux_plan(monkeypatch, "pkexec", pkexec)
    job = _settle(job_mod.start(plan_mod.LINUX_INSTALL))
    assert job.state == "succeeded"
    (call,) = env["recorded"]()
    assert call[:3] == [pkexec, "/bin/sh", "-c"]
    assert (
        call[3]
        == "set -e\napt-get install -y bubblewrap\napparmor_parser -r /etc/apparmor.d/bwrap-userns-restrict\n"
    )
    assert call[3] == job_mod.pkexec_script(_STEPS)


def test_a_dismissed_polkit_prompt_is_declined(monkeypatch, env):
    _linux_plan(monkeypatch, "pkexec", env["fake"]("pkexec", code = job_mod.PKEXEC_DISMISSED))
    job = _settle(job_mod.start(plan_mod.LINUX_INSTALL))
    assert job.state == "declined" and job.manual_command


def test_no_polkit_agent_is_a_named_failure(monkeypatch, env):
    _linux_plan(monkeypatch, "pkexec", env["fake"]("pkexec", code = job_mod.PKEXEC_NOT_AUTHORIZED))
    job = _settle(job_mod.start(plan_mod.LINUX_INSTALL))
    assert job.state == "failed" and "authentication agent" in job.note


def test_no_elevation_means_no_job(monkeypatch, env):
    plan = plan_mod.SetupPlan(platform = "linux", manual_command = "sudo apt-get install -y bubblewrap")
    monkeypatch.setattr(plan_mod, "detect", lambda *a, **k: plan)
    with pytest.raises(job_mod.SetupUnavailable):
        job_mod.start(plan_mod.LINUX_INSTALL)
    with pytest.raises(job_mod.SetupUnavailable):
        job_mod.start("rm-rf")
    assert job_mod.current() is None


def test_one_setup_at_a_time_across_both_jobs(monkeypatch, env, tmp_path):
    gate = tmp_path / "open"
    _linux_plan(monkeypatch, "sudo", env["fake"]("sudo", gate = gate))
    first = job_mod.start(plan_mod.LINUX_INSTALL)
    try:
        assert job_mod.start(plan_mod.LINUX_INSTALL) is first
        # "Prepare this PC" does not start a second host change either.
        assert mxc_host_prep_job.start() is first
    finally:
        gate.write_text("")
    _settle(first)
    assert first.state == "succeeded"
    assert len(env["recorded"]()) == 2


def test_windows_setup_installs_then_prepares_in_order(monkeypatch, env):
    install = env["fake"]("install-runtime")
    prepare = env["fake"](
        "prepare-host",
        output = "[mxc-prebuilt] host prepared: prepare-system-drive, prepare-null-device",
    )
    plan = plan_mod.SetupPlan(
        platform = "win32",
        action = plan_mod.WINDOWS_SETUP,
        elevation = "uac",
        steps = ((install,), (prepare, "--prepare-host")),
    )
    monkeypatch.setattr(plan_mod, "detect", lambda *a, **k: plan)
    job = _settle(job_mod.start(plan_mod.WINDOWS_SETUP))
    assert job.state == "succeeded"
    assert env["recorded"]() == [[install], [prepare, "--prepare-host"]]
    assert job.steps == ["prepare-system-drive", "prepare-null-device"]


def test_windows_declined_administrator_prompt(monkeypatch, env):
    install = env["fake"]("install-runtime")
    prepare = env["fake"]("prepare-host", code = 2, output = "the administrator prompt was declined")
    plan = plan_mod.SetupPlan(
        platform = "win32",
        action = plan_mod.WINDOWS_SETUP,
        steps = ((install,), (prepare, "--prepare-host")),
    )
    monkeypatch.setattr(plan_mod, "detect", lambda *a, **k: plan)
    job = _settle(job_mod.start(plan_mod.WINDOWS_SETUP))
    assert job.state == "declined" and job.exit_code == 2


def test_windows_runtime_failure_stops_before_preparing(monkeypatch, env):
    install = env["fake"]("install-runtime", code = 1, output = "download failed")
    prepare = env["fake"]("prepare-host")
    plan = plan_mod.SetupPlan(
        platform = "win32",
        action = plan_mod.WINDOWS_SETUP,
        steps = ((install,), (prepare, "--prepare-host")),
    )
    monkeypatch.setattr(plan_mod, "detect", lambda *a, **k: plan)
    job = _settle(job_mod.start(plan_mod.WINDOWS_SETUP))
    assert job.state == "failed" and env["recorded"]() == [[install]]
    assert "download failed" in job.output_tail


def test_consent_only_windows_setup_finishes_without_a_process(monkeypatch, env):
    plan = plan_mod.SetupPlan(platform = "win32", action = plan_mod.WINDOWS_SETUP, needs_consent = True)
    monkeypatch.setattr(plan_mod, "detect", lambda *a, **k: plan)
    job = _settle(job_mod.start(plan_mod.WINDOWS_SETUP))
    assert job.state == "succeeded" and env["recorded"]() == []


def test_a_spawn_error_is_a_failed_job(monkeypatch, env):
    _linux_plan(monkeypatch, "sudo", os.path.join(os.sep, "nonexistent", "sudo"))
    job = _settle(job_mod.start(plan_mod.LINUX_INSTALL))
    assert job.state == "failed" and job.exit_code is None
