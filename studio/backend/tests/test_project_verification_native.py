# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""End-to-end verification evidence over the secure runner prerequisite."""

import importlib
import os
import shlex
import sys
import threading
import time

import pytest

from core.agent_workspace import verification, verification_context as common
from core.agent_workspace import verification_process, verification_state
from core.agent_workspace.verification_context import AgentWorkspaceError
from storage import studio_db


def test_missing_prerequisite_never_falls_back_to_a_host_shell(monkeypatch):
    original = importlib.import_module

    def missing(name, *args, **kwargs):
        if name == "core.agent_workspace.supervisor":
            raise ModuleNotFoundError("missing prerequisite", name = name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", missing)
    assert verification_process.supervised_process_status().available is False
    with pytest.raises(AgentWorkspaceError, match = "must be installed"):
        verification_process.run_project_process("project", ["/bin/sh", "-c", "false"])


@pytest.fixture
def native_project(tmp_path, monkeypatch):
    status = verification_process.supervised_process_status()
    if not status.available:
        if os.environ.get("UNSLOTH_VERIFICATION_NATIVE_REQUIRED") == "1":
            pytest.fail(status.reason)
        pytest.skip(status.reason)
    native = importlib.import_module("core.agent_workspace.supervisor")
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    project = studio_db.upsert_chat_project(
        {"id": "native-verification", "name": "Native verification", "createdAt": 1, "updatedAt": 1}
    )
    workspace = common.project_workspace(project["id"])
    yield native, workspace
    verification.shutdown_project_verifications(timeout_seconds = 10)


def _check(
    command,
    *,
    name = "tests",
    required = True,
):
    return {
        "name": name,
        "kind": "custom",
        "command": command,
        "required": required,
        "timeoutSeconds": 15,
        "logLimitBytes": 4096,
    }


def _python(source):
    return shlex.join([sys.executable, "-I", "-c", source])


def _configure(workspace, checks):
    return verification_state.set_verification_config(
        workspace.project_id,
        checks,
        workspace_identity = (workspace.device_id, workspace.file_id),
        workspace_revision = workspace.revision,
        expected_revision = 0,
    )


def _await_terminal(workspace, run_id):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        run = verification_state.get_verification_run(workspace.project_id, run_id)
        if run["status"] != "running":
            return run
        time.sleep(0.05)
    pytest.fail("verification did not finish")


def test_native_verification_persists_real_output_and_optional_failures(native_project):
    _native, workspace = native_project
    profile = _configure(
        workspace,
        [
            _check(_python("print('native verification')")),
            _check("exit 3", name = "optional", required = False),
        ],
    )
    started = verification.start_project_verification(
        workspace.project_id, config_revision = profile["revision"], workspace_revision = 0
    )
    terminal = _await_terminal(workspace, started["id"])
    assert terminal["status"] == "passed", terminal
    assert "native verification" in terminal["results"][0]["output"]
    assert terminal["results"][1]["exitCode"] == 3
    assert terminal["historySequence"] > 0
    assert terminal["sourceFreshness"] == "unverified"


def test_native_cancellation_stops_detached_descendants_before_retirement(native_project):
    _native, workspace = native_project
    source = "import os,time; from pathlib import Path; pid=os.fork(); os.setsid() if pid == 0 else None; Path('child-ready' if pid == 0 else 'parent-ready').write_text(str(os.getpid())); time.sleep(30)"
    profile = _configure(workspace, [_check(_python(source))])
    started = verification.start_project_verification(
        workspace.project_id, config_revision = profile["revision"], workspace_revision = 0
    )
    deadline = time.monotonic() + 10
    while not (workspace.root / "child-ready").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (workspace.root / "child-ready").exists()
    verification.begin_project_deletion(workspace.project_id)
    try:
        verification.cancel_project_verifications_and_wait(workspace.project_id, timeout_seconds = 10)
        terminal = verification_state.get_verification_run(workspace.project_id, started["id"])
        assert terminal["status"] == "cancelled", terminal
        assert not verification_state.project_execution_may_start(workspace.project_id)
        # The retirement API holds the same process-shared fence until mutation finishes.
        with pytest.raises(TimeoutError):
            verification_process._acquire_project_execution_fence(
                "project:" + workspace.project_id, None, time.monotonic() + 0.03
            )
    finally:
        verification.finish_project_deletion(workspace.project_id)
