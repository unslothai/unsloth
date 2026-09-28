# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Settings "Prepare this PC" job: one run at a time, its outcome read from the helper's output."""

import threading
import time

import pytest

from core.inference import mxc_host_prep_job, mxc_probe, tools


class _FakeProc:
    def __init__(
        self,
        lines,
        code,
        gate = None,
    ):
        self._lines, self._code, self._gate = lines, code, gate
        self.stdout = self._stream()

    def _stream(self):
        for line in self._lines:
            yield line + "\n"
        if self._gate is not None:
            self._gate.wait(60)

    def wait(self):
        return self._code


@pytest.fixture
def job_env(monkeypatch):
    calls = {"invalidate": 0, "profile": 0, "spawned": []}
    monkeypatch.setattr(mxc_host_prep_job, "_current", None)
    monkeypatch.setattr(mxc_host_prep_job, "_on_finish", [])
    monkeypatch.setattr(
        mxc_probe, "host_prep_command", lambda: ["python", "install_mxc_prebuilt.py"]
    )
    monkeypatch.setattr(
        mxc_probe,
        "invalidate_cache",
        lambda: calls.__setitem__("invalidate", calls["invalidate"] + 1),
    )
    monkeypatch.setattr(
        tools,
        "reset_terminal_profile_cache",
        lambda: calls.__setitem__("profile", calls["profile"] + 1),
    )
    return calls


def _run(
    monkeypatch,
    calls,
    proc,
    settle = True,
):
    def spawn(argv):
        calls["spawned"].append(argv)
        return proc

    monkeypatch.setattr(mxc_host_prep_job, "_spawn", spawn)
    job = mxc_host_prep_job.start()
    deadline = time.monotonic() + 30
    while settle and job.state == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    return job


def test_success_reports_the_steps_and_resets_the_caches(monkeypatch, job_env):
    hook = []
    mxc_host_prep_job.add_finish_hook(lambda: hook.append(1))
    job = _run(
        monkeypatch,
        job_env,
        _FakeProc(["[mxc-prebuilt] host prepared: prepare-system-drive, prepare-null-device"], 0),
    )
    assert job.state == "succeeded" and job.exit_code == 0
    assert job.steps == ["prepare-system-drive", "prepare-null-device"]
    assert job_env["invalidate"] == 1 and job_env["profile"] == 1 and hook == [1]
    assert job_env["spawned"] == [["python", "install_mxc_prebuilt.py"]]
    assert mxc_host_prep_job.current() is job


def test_already_prepared_has_no_steps(monkeypatch, job_env):
    job = _run(monkeypatch, job_env, _FakeProc(["[mxc-prebuilt] host already prepared"], 0))
    assert job.state == "succeeded" and job.steps == []


def test_a_declined_prompt_is_named(monkeypatch, job_env):
    job = _run(
        monkeypatch,
        job_env,
        _FakeProc(["[mxc-prebuilt] the Windows administrator prompt was declined"], 1),
    )
    assert job.state == "declined" and job.exit_code == 1


def test_a_failure_keeps_the_tail_of_the_output(monkeypatch, job_env):
    lines = [f"line {i}" for i in range(30)] + ["[mxc-prebuilt] wxc-host-prep failed"]
    job = _run(monkeypatch, job_env, _FakeProc(lines, 1))
    assert job.state == "failed"
    assert len(job.output_tail) == mxc_host_prep_job.OUTPUT_TAIL_LINES
    assert job.output_tail[-1] == "[mxc-prebuilt] wxc-host-prep failed"
    assert job_env["invalidate"] == 1


def test_a_spawn_error_is_a_failed_job(monkeypatch, job_env):
    def boom(_argv):
        raise OSError("python is gone")

    monkeypatch.setattr(mxc_host_prep_job, "_spawn", boom)
    job = mxc_host_prep_job.start()
    assert job.state == "failed" and "python is gone" in job.output_tail[0]


def test_only_one_run_at_a_time(monkeypatch, job_env):
    gate = threading.Event()
    first = _run(monkeypatch, job_env, _FakeProc(["working"], 0, gate = gate), settle = False)
    assert first.state == "running"
    again = mxc_host_prep_job.start()
    assert again is first and len(job_env["spawned"]) == 1
    gate.set()
    deadline = time.monotonic() + 30
    while first.state == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert first.state == "succeeded"
    second = mxc_host_prep_job.start()
    assert second is not first and len(job_env["spawned"]) == 2
