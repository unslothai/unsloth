# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Focused security contract for project-backed editing and execution.

These tests are deliberately independent of model inference. Portable tests
exercise the Windows fail-closed branch on every host, while the workflow also
runs the same contract on native Linux, macOS, and Windows filesystems.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from core.agent_workspace import common, execution, mutation, supervisor
from core.agent_workspace.common import AgentWorkspaceError, ProjectWorkspace
from core.agent_workspace.execution import (
    ExecutionBoundaryStatus,
    ProjectExecutionUnavailable,
    execution_boundary_status,
)
from core.agent_workspace.mutation import (
    ProjectFileMutation,
    WindowsMutationRejected,
)
from core.inference import tools


def _workspace(root: Path, project_id: str = "secure-tools") -> ProjectWorkspace:
    metadata = root.stat(follow_symlinks = False)
    return ProjectWorkspace(
        project_id = project_id,
        root = root.resolve(strict = True),
        kind = "managed",
        device_id = int(metadata.st_dev),
        file_id = int(metadata.st_ino),
    )


def _bind_project(
    monkeypatch,
    root: Path,
    project_id: str = "secure-tools",
) -> str:
    workspace = _workspace(root, project_id)
    session_id = tools.project_session_id(project_id)
    monkeypatch.setattr(tools, "_get_project_workdir", lambda _session_id: str(root))
    monkeypatch.setattr(tools, "_get_workdir", lambda _session_id = None: str(root))
    monkeypatch.setattr(common, "project_workspace", lambda _project_id: workspace)

    @contextlib.contextmanager
    def access(received_project_id):
        assert received_project_id == project_id
        yield workspace

    monkeypatch.setattr(common, "project_workspace_access", access)
    return session_id


def test_windows_never_advertises_a_project_command_boundary():
    status = execution_boundary_status("win32", probe = False)

    assert status.available is False
    assert status.backend is None
    assert "Windows" in str(status.reason)


def test_bubblewrap_resolution_ignores_caller_path_and_override(tmp_path, monkeypatch):
    attacker = tmp_path / "bwrap"
    attacker.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    attacker.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_BWRAP", str(attacker))

    resolved = execution._bubblewrap_path()

    assert resolved != str(attacker)


@pytest.mark.skipif(sys.platform != "win32", reason = "native Windows boundary status")
def test_native_windows_project_commands_are_fail_closed():
    status = execution_boundary_status(probe = False)

    assert status.available is False
    assert status.backend is None


def test_project_execution_id_routes_real_project_without_opening_a_boundary(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    session_id = _bind_project(monkeypatch, root)
    monkeypatch.setattr(
        execution.ProjectExecutionBoundary,
        "open",
        lambda *_args, **_kwargs: pytest.fail("tools.py opened its own process boundary"),
    )

    assert tools._project_execution_id(session_id, str(root), disable_sandbox = False) == (
        "secure-tools"
    )


def test_project_execution_id_rejects_a_changed_workspace(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    replacement = tmp_path / "replacement"
    root.mkdir()
    replacement.mkdir()
    session_id = tools.project_session_id("secure-tools")
    monkeypatch.setattr(tools, "_get_project_workdir", lambda _session_id: str(root))

    with pytest.raises(RuntimeError, match = "workspace changed"):
        tools._project_execution_id(
            session_id,
            str(replacement),
            disable_sandbox = False,
        )


@pytest.mark.parametrize("tool_name", ["python", "terminal", "edit_file"])
def test_project_lookup_failure_never_falls_back_to_conversation_execution(
    tool_name, tmp_path, monkeypatch
):
    monkeypatch.setattr(tools, "_get_workdir", lambda _session_id: str(tmp_path))
    monkeypatch.setattr(tools, "_get_project_workdir", lambda _session_id: None)
    monkeypatch.setattr(tools, "_thread_exists", lambda _session_id: False)
    monkeypatch.setattr(
        tools.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("project request fell back to unconfined execution"),
    )
    session_id = tools.project_session_id("unavailable")
    if tool_name == "python":
        result = tools._python_exec("print('unsafe')", session_id = session_id)
    elif tool_name == "terminal":
        result = tools._bash_exec("echo unsafe", session_id = session_id)
    else:
        result = tools._edit_file(
            {"path": "new.txt", "edits": [{"old_string": "", "new_string": "unsafe"}]},
            session_id = session_id,
        )
    assert "workspace is unavailable" in result
    assert not (tmp_path / "new.txt").exists()


def test_stored_chat_with_project_prefix_keeps_its_conversation_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_thread_exists", lambda _session_id: True)
    monkeypatch.setattr(
        tools,
        "_get_project_workdir",
        lambda _session_id: pytest.fail("a stored chat was resolved as a project"),
    )
    assert tools._project_execution_id("project-chat", str(tmp_path), disable_sandbox = False) is None
    workspace = tools._edit_file_workspace("project-chat", str(tmp_path))
    assert workspace.kind == "conversation"
    assert workspace.root == tmp_path.resolve()


def test_hardlinked_file_is_rejected_before_content_is_read(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"private outside content")
    os.link(outside, root / "linked.txt")
    with ProjectFileMutation.open(_workspace(root), "linked.txt") as boundary:
        with pytest.raises((AgentWorkspaceError, WindowsMutationRejected), match = "Hard-linked"):
            boundary.read(1024)
    assert outside.read_bytes() == b"private outside content"


def test_full_access_skips_project_supervisor_resolution(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    monkeypatch.setattr(
        tools,
        "_get_project_workdir",
        lambda _session_id: pytest.fail("resolved a project boundary in Full access"),
    )

    assert (
        tools._project_execution_id(
            tools.project_session_id("secure-tools"),
            str(root),
            disable_sandbox = True,
        )
        is None
    )


@pytest.mark.parametrize("tool_name", ["python", "terminal"])
def test_windows_project_commands_fail_before_popen(tool_name, tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    session_id = _bind_project(monkeypatch, root)
    monkeypatch.setattr(
        execution,
        "execution_boundary_status",
        lambda probe = True: ExecutionBoundaryStatus(
            False,
            None,
            "Project command execution is disabled on Windows.",
        ),
    )
    monkeypatch.setattr(
        tools.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen ran without a project boundary"),
    )

    if tool_name == "python":
        result = tools._python_exec("print('unsafe')", session_id = session_id)
    else:
        result = tools._bash_exec("echo unsafe", session_id = session_id)

    assert "Execution error" in result
    assert "disabled on Windows" in result


@pytest.mark.skipif(sys.platform != "darwin", reason = "native macOS process containment")
@pytest.mark.parametrize("tool_name", ["python", "terminal"])
def test_native_macos_project_tools_fail_before_popen(tool_name, tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    session_id = _bind_project(monkeypatch, root)
    monkeypatch.setattr(
        tools.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("spawned under insufficient process containment"),
    )

    if tool_name == "python":
        result = tools._python_exec("print('unsafe')", session_id = session_id)
    else:
        result = tools._bash_exec("echo unsafe", session_id = session_id)

    assert "Execution error" in result
    assert "detached descendants" in result


@pytest.mark.parametrize("tool_name", ["python", "terminal"])
def test_full_access_project_commands_keep_the_explicit_escape_hatch(
    tool_name, tmp_path, monkeypatch
):
    root = tmp_path / "repository"
    root.mkdir()
    session_id = _bind_project(monkeypatch, root)
    monkeypatch.setattr(tools, "_harden_parent_against_proc_env_leak", lambda: True)

    def reached_popen(*_args, **_kwargs):
        raise OSError("popen reached")

    monkeypatch.setattr(tools.subprocess, "Popen", reached_popen)

    if tool_name == "python":
        result = tools._python_exec(
            "print('full')",
            session_id = session_id,
            disable_sandbox = True,
        )
    else:
        result = tools._bash_exec(
            "echo full",
            session_id = session_id,
            disable_sandbox = True,
        )

    assert "Execution error: popen reached" in result


@pytest.mark.parametrize("tool_name", ["python", "terminal"])
def test_project_tools_dispatch_through_supervisor_with_streaming(tool_name, tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    session_id = _bind_project(monkeypatch, root)
    observed = {}
    streamed = []

    def fake_python(project_id, source, **kwargs):
        observed.update(project_id = project_id, payload = source, kwargs = kwargs)
        kwargs["output_callback"]("supervised\n")
        return supervisor.ProjectProcessResult("passed", 0, "supervised\n", 11, False)

    def fake_terminal(project_id, argv, **kwargs):
        observed.update(project_id = project_id, payload = argv, kwargs = kwargs)
        kwargs["output_callback"]("supervised\n")
        return supervisor.ProjectProcessResult("passed", 0, "supervised\n", 11, False)

    monkeypatch.setattr(supervisor, "run_project_python", fake_python)
    monkeypatch.setattr(supervisor, "run_project_process", fake_terminal)
    monkeypatch.setattr(
        tools.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("tools.py spawned outside the supervisor"),
    )

    if tool_name == "python":
        result = tools._python_exec(
            "print('bounded')",
            session_id = session_id,
            output_callback = streamed.append,
        )
        assert observed["payload"] == "print('bounded')"
    else:
        result = tools._bash_exec(
            "echo bounded",
            session_id = session_id,
            output_callback = streamed.append,
        )
        assert observed["payload"][-2:] == ["-c", "echo bounded"]

    assert result == "supervised\n"
    assert streamed == ["supervised\n"]
    assert observed["project_id"] == "secure-tools"
    assert observed["kwargs"]["cancel_event"] is None
    assert observed["kwargs"]["timeout_seconds"] == 300


@pytest.mark.parametrize("tool_name", ["python", "terminal"])
def test_project_supervisor_results_keep_artifact_cards(tool_name, tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    session_id = _bind_project(monkeypatch, root)

    def write_artifact(*_args, **_kwargs):
        (root / "report.csv").write_bytes(b"a,b\n1,2\n")
        return supervisor.ProjectProcessResult("passed", 0, "", 0, False)

    monkeypatch.setattr(supervisor, "run_project_python", write_artifact)
    monkeypatch.setattr(supervisor, "run_project_process", write_artifact)

    if tool_name == "python":
        result = tools._python_exec("print('ignored')", session_id = session_id)
    else:
        result = tools._bash_exec("echo ignored", session_id = session_id)

    assert result.startswith("(no output)")
    assert '__FILES__:[{"name": "report.csv", "size": 8}]' in result


@pytest.mark.parametrize("tool_name", ["python", "terminal"])
def test_project_supervisor_reports_bounded_output_honestly(tool_name, tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    session_id = _bind_project(monkeypatch, root)

    def bounded_result(*_args, **_kwargs):
        return supervisor.ProjectProcessResult("passed", 0, "prefix", 8192, True)

    monkeypatch.setattr(supervisor, "run_project_python", bounded_result)
    monkeypatch.setattr(supervisor, "run_project_process", bounded_result)

    if tool_name == "python":
        result = tools._python_exec("print('ignored')", session_id = session_id)
    else:
        result = tools._bash_exec("echo ignored", session_id = session_id)

    assert result.startswith("[Process produced 8192 bytes. Only the bounded 6-byte prefix")
    assert result.endswith("prefix")


@pytest.mark.parametrize("tool_name", ["python", "terminal"])
@pytest.mark.parametrize("status", ["timed_out", "cancelled"])
def test_project_supervisor_keeps_truncation_notice_on_early_stop(
    tool_name, status, tmp_path, monkeypatch
):
    root = tmp_path / "repository"
    root.mkdir()
    session_id = _bind_project(monkeypatch, root)
    notice = "\n[Process output was truncated. The capture limit was 1024 bytes.]\n"

    def stopped_result(*_args, **_kwargs):
        return supervisor.ProjectProcessResult(
            status,
            None,
            f"prefix{notice}",
            8192,
            True,
            notice,
        )

    monkeypatch.setattr(supervisor, "run_project_python", stopped_result)
    monkeypatch.setattr(supervisor, "run_project_process", stopped_result)

    if tool_name == "python":
        result = tools._python_exec("print('ignored')", session_id = session_id)
    else:
        result = tools._bash_exec("echo ignored", session_id = session_id)

    expected = "Execution timed out" if status == "timed_out" else "Execution cancelled."
    assert result.startswith(expected)
    assert result.count(notice.strip()) == 1
    assert "prefix" not in result


def test_boundary_open_rejects_an_unavailable_backend_without_touching_the_root(
    tmp_path, monkeypatch
):
    root = tmp_path / "repository"
    root.mkdir()
    monkeypatch.setattr(
        execution,
        "execution_boundary_status",
        lambda: ExecutionBoundaryStatus(False, None, "certified boundary unavailable"),
    )
    monkeypatch.setattr(
        execution.os,
        "open",
        lambda *_args, **_kwargs: pytest.fail("opened a root without a certified boundary"),
    )

    with pytest.raises(ProjectExecutionUnavailable, match = "certified boundary unavailable"):
        execution.ProjectExecutionBoundary.open(_workspace(root))


def test_linux_boundary_uses_an_empty_root_instead_of_binding_the_host(monkeypatch):
    boundary = object.__new__(execution.ProjectExecutionBoundary)
    boundary.backend = "bubblewrap"
    boundary.root = Path("/workspace/project")
    boundary.scratch = Path("/runtime/scratch")
    boundary._root_fd = 11
    boundary._scratch_fd = 12
    boundary._sandbox_root_fd = 13
    boundary._runtime_directories = []
    boundary._linux_system_mounts = [(Path("/usr"), Path("/usr"), True)]
    boundary.recheck = lambda: None
    monkeypatch.setattr(execution, "_bubblewrap_path", lambda: "/usr/bin/bwrap")

    argv = boundary.wrap_argv(["/usr/bin/python", "-c", "print('safe')"])

    assert ["--ro-bind", "/proc/self/fd/13", "/"] == argv[3:6]
    assert ["--ro-bind", "/", "/"] not in [argv[index : index + 3] for index in range(len(argv))]
    assert "/etc/passwd" not in argv


@pytest.mark.skipif(os.name != "posix", reason = "native POSIX command boundary")
def test_native_posix_boundary_allows_project_write_and_denies_sibling_write(tmp_path):
    status = execution_boundary_status()
    if not status.available:
        if os.environ.get("UNSLOTH_SECURE_BOUNDARY_REQUIRED") == "1":
            pytest.fail(status.reason or "the required project command boundary is unavailable")
        pytest.skip(status.reason or "project command boundary is unavailable")

    root = tmp_path / "repository"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    source = """
import sys
from pathlib import Path
Path("inside.txt").write_text("inside", encoding="utf-8")
try:
    Path(sys.argv[1]).write_text("escaped", encoding="utf-8")
except OSError:
    print("outside denied")
else:
    print("outside written")
try:
    Path(sys.argv[2]).read_text(encoding="utf-8")
except OSError:
    print("host read denied")
else:
    print("host read allowed")
"""
    with execution.ProjectExecutionBoundary.open(_workspace(root)) as boundary:
        assert boundary.acquire_execution_slot() is True
        argv = boundary.wrap_argv(
            [str(Path(sys.executable).resolve()), "-c", source, str(outside), "/etc/passwd"]
        )
        environment = boundary.apply_environment(
            {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONIOENCODING": "utf-8",
            }
        )
        completed = subprocess.run(
            argv,
            env = environment,
            capture_output = True,
            text = True,
            timeout = 15,
            check = False,
            **boundary.popen_kwargs(),
        )

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[0] == "outside denied"
    if status.backend == "bubblewrap":
        assert lines[1] == "host read denied"
    assert (root / "inside.txt").read_text(encoding = "utf-8") == "inside"
    assert not outside.exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "native Linux supervisor")
def test_native_project_python_entrypoint_confines_writes(tmp_path, monkeypatch):
    status = supervisor.supervised_process_status()
    if not status.available:
        if os.environ.get("UNSLOTH_SECURE_BOUNDARY_REQUIRED") == "1":
            pytest.fail(status.reason or "the required project process supervisor is unavailable")
        pytest.skip(status.reason or "project process supervisor is unavailable")

    root = tmp_path / "repository"
    root.mkdir()
    (root / "helper.py").write_text("VALUE = 42\n", encoding = "utf-8")
    outside = tmp_path / "outside.txt"
    session_id = _bind_project(monkeypatch, root)
    source = f"""
from pathlib import Path
import helper

print(Path.cwd())
print(helper.VALUE)
print(Path(__file__).name)
Path("inside.txt").write_text("inside", encoding="utf-8")
try:
    Path({str(outside)!r}).write_text("escaped", encoding="utf-8")
except OSError:
    print("outside denied")
else:
    print("outside written")
host_path = "".join(chr(value) for value in (47, 101, 116, 99, 47, 112, 97, 115, 115, 119, 100))
try:
    Path(host_path).read_text(encoding="utf-8")
except OSError:
    print("host read denied")
else:
    print("host read allowed")
"""

    result = tools._python_exec(source, session_id = session_id)

    lines = result.splitlines()
    assert lines[0] == str(root)
    assert lines[1] == "42"
    assert lines[2].startswith("studio_exec_") and lines[2].endswith(".py")
    assert lines[3] == "outside denied"
    assert lines[4] == "host read denied"
    assert (root / "inside.txt").read_text(encoding = "utf-8") == "inside"
    assert not outside.exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "native Linux supervisor")
@pytest.mark.parametrize("tool_name", ["python", "terminal"])
def test_native_project_tools_kill_detached_descendants_before_lease_and_slot_release(
    tool_name, tmp_path, monkeypatch
):
    status = supervisor.supervised_process_status()
    if not status.available:
        if os.environ.get("UNSLOTH_SECURE_BOUNDARY_REQUIRED") == "1":
            pytest.fail(status.reason or "the required project process supervisor is unavailable")
        pytest.skip(status.reason or "project process supervisor is unavailable")

    root = tmp_path / "repository"
    root.mkdir()
    session_id = _bind_project(monkeypatch, root)
    workspace = _workspace(root)
    lease_released = root / "lease-released.txt"
    slot_released = root / "slot-released.txt"
    escaped = root / "escaped.txt"

    @contextlib.contextmanager
    def access(project_id):
        assert project_id == workspace.project_id
        try:
            yield workspace
        finally:
            lease_released.write_text("released", encoding = "utf-8")

    monkeypatch.setattr(common, "project_workspace_access", access)
    child_code = (
        "import time; from pathlib import Path; "
        f"lease=Path({str(lease_released)!r}); slot=Path({str(slot_released)!r}); "
        f"escaped=Path({str(escaped)!r}); deadline=time.monotonic()+10; "
        'exec("while time.monotonic() < deadline and not lease.exists() and not slot.exists():\\n'
        '    time.sleep(0.005)"); '
        "escaped.write_text('survived', encoding='utf-8') "
        "if lease.exists() or slot.exists() else None"
    )
    if tool_name == "python":
        source = (
            "import subprocess, time; "
            f"child=subprocess.Popen([{sys.executable!r}, '-c', {child_code!r}], "
            "start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
            "print(child.pid, flush=True); time.sleep(30)"
        )
        assert tools._check_code_safety(source) is None
        result = tools._python_exec(source, session_id = session_id, timeout = 0.2)
    else:
        command = (
            f"setsid {shlex.quote(sys.executable)} -c {shlex.quote(child_code)} "
            ">/dev/null 2>&1 & echo $!; sleep 30"
        )
        result = tools._bash_exec(command, session_id = session_id, timeout = 0.2)

    identity = (int(workspace.device_id), int(workspace.file_id))
    assert mutation.acquire_workspace_mutation_slot(identity) is True
    try:
        slot_released.write_text("released", encoding = "utf-8")
    finally:
        mutation.release_workspace_mutation_slot(identity)
    time.sleep(0.5)

    assert result.startswith("Execution timed out after 0.2 seconds."), result
    assert lease_released.exists()
    assert not escaped.exists()


@pytest.mark.skipif(os.name == "nt", reason = "POSIX descriptor mutation")
def test_posix_project_mutation_round_trip_is_exact_and_atomic(tmp_path):
    root = tmp_path / "repository"
    target = root / "src" / "module.py"
    target.parent.mkdir(parents = True)
    target.write_bytes(b"before\n")
    workspace = _workspace(root)

    with ProjectFileMutation.open(workspace, "src/module.py") as boundary:
        before, mode, identity = boundary.read(1024)
        assert before == b"before\n"
        assert (
            boundary.replace(
                b"after\n",
                expect = before,
                mode = mode,
                identity = identity,
            )
            is None
        )

    assert target.read_bytes() == b"after\n"
    assert not list(target.parent.glob(".unsloth_edit_*"))


@pytest.mark.skipif(os.name == "nt", reason = "POSIX mode semantics")
def test_posix_project_create_applies_the_process_umask(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    target = root / "created.txt"
    previous_umask = os.umask(0o027)
    try:
        with ProjectFileMutation.open(_workspace(root), "created.txt") as boundary:
            assert boundary.create(b"created") is None
    finally:
        os.umask(previous_umask)

    assert target.stat().st_mode & 0o777 == 0o640


@pytest.mark.skipif(os.name == "nt", reason = "POSIX mode semantics")
def test_posix_project_replace_preserves_the_exact_existing_mode(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    target = root / "executable.sh"
    target.write_bytes(b"before\n")
    target.chmod(0o751)

    with ProjectFileMutation.open(_workspace(root), "executable.sh") as boundary:
        before, mode, identity = boundary.read(1024)
        assert mode == 0o751
        assert (
            boundary.replace(
                b"after\n",
                expect = before,
                mode = mode,
                identity = identity,
            )
            is None
        )

    assert target.stat().st_mode & 0o777 == 0o751


@pytest.mark.skipif(os.name == "nt", reason = "POSIX descriptor mutation")
def test_posix_project_mutation_refuses_parent_symlink_escape(tmp_path):
    root = tmp_path / "repository"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_bytes(b"preserve")
    os.symlink(outside, root / "linked")

    with ProjectFileMutation.open(_workspace(root), "linked/secret.txt") as boundary:
        with pytest.raises((AgentWorkspaceError, OSError)):
            boundary.read(1024)

    assert secret.read_bytes() == b"preserve"


@pytest.mark.skipif(os.name == "nt", reason = "POSIX descriptor mutation")
def test_posix_project_mutation_detects_stale_expected_content(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    target = root / "state.txt"
    target.write_bytes(b"first")

    with ProjectFileMutation.open(_workspace(root), "state.txt") as boundary:
        before, mode, identity = boundary.read(1024)
        target.write_bytes(b"second")
        assert (
            boundary.replace(
                b"third",
                expect = before,
                mode = mode,
                identity = identity,
            )
            == "changed"
        )

    assert target.read_bytes() == b"second"
    assert not list(root.glob(".unsloth_edit_*"))


@pytest.mark.skipif(os.name == "nt", reason = "POSIX descriptor mutation")
def test_posix_project_mutation_enforces_payload_bound_without_residue(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()

    with ProjectFileMutation.open(_workspace(root), "bounded.txt", max_bytes = 4) as boundary:
        with pytest.raises(OverflowError, match = "configured limit"):
            boundary.create(b"12345")

    assert not (root / "bounded.txt").exists()
    assert not list(root.glob(".unsloth_edit_*"))


@pytest.mark.skipif(os.name == "nt", reason = "portable slot fixture uses POSIX mutation")
def test_project_edit_and_command_boundaries_share_one_mutation_slot(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    target = root / "state.txt"
    target.write_bytes(b"first")
    workspace = _workspace(root)
    command_boundary = object.__new__(execution.ProjectExecutionBoundary)
    command_boundary.root_identity = (workspace.device_id, workspace.file_id)
    command_boundary._slot = False
    command_boundary._closed = False
    command_boundary.recheck = lambda: None
    cancelled = threading.Event()
    cancelled.set()

    with ProjectFileMutation.open(workspace, "state.txt"):
        assert command_boundary.acquire_execution_slot(cancelled) is False

    assert command_boundary.acquire_execution_slot() is True
    command_boundary.release_execution_slot()


@pytest.mark.parametrize("creating", [False, True])
def test_sandboxed_edit_file_dispatches_through_project_mutation(creating, tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    target = root / "module.py"
    if not creating:
        target.write_bytes(b"value = 1\n")
    session_id = _bind_project(monkeypatch, root)
    real_open = ProjectFileMutation.open
    observed = []

    def tracking_open(
        workspace,
        path,
        *,
        max_bytes,
        cancel_event = None,
    ):
        observed.append((workspace, path, max_bytes))
        return real_open(workspace, path, max_bytes = max_bytes)

    monkeypatch.setattr(ProjectFileMutation, "open", tracking_open)
    edit = (
        {"old_string": "", "new_string": "value = 1\n"}
        if creating
        else {"old_string": "value = 1", "new_string": "value = 2"}
    )

    result = tools.execute_tool(
        "edit_file",
        {"path": "module.py", "edits": [edit]},
        session_id = session_id,
    )

    assert not result.startswith("Error:"), result
    assert len(observed) == 1
    assert observed[0][0].root == root
    assert observed[0][1] == "module.py"
    assert target.read_bytes() == (b"value = 1\n" if creating else b"value = 2\n")


def test_full_access_edit_file_keeps_the_explicit_outside_escape_hatch(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"before\n")
    session_id = _bind_project(monkeypatch, root)
    monkeypatch.setattr(
        ProjectFileMutation,
        "open",
        lambda *_args, **_kwargs: pytest.fail("opened a confined mutation in Full access"),
    )

    result = tools.execute_tool(
        "edit_file",
        {
            "path": str(outside),
            "edits": [{"old_string": "before", "new_string": "after"}],
        },
        session_id = session_id,
        disable_sandbox = True,
    )

    assert not result.startswith("Error:"), result
    assert outside.read_bytes() == b"after\n"


class _UnusedWindowsOps:
    def open_existing(self, *_args, **_kwargs):
        pytest.fail("opened a path that lexical validation should reject")

    def close(self, _handle):
        pass


def _windows_workspace() -> ProjectWorkspace:
    return ProjectWorkspace(
        project_id = "secure-tools",
        root = Path(r"C:\project"),
        kind = "managed",
        device_id = 11,
        file_id = 22,
    )


def test_windows_containment_preserves_a_drive_root():
    assert mutation._windows_within(r"C:\project\inside.txt", "C:\\")
    assert not mutation._windows_within(r"D:\project\inside.txt", "C:\\")


@pytest.mark.parametrize(
    "target",
    [
        r"..\outside.txt",
        r"C:\other\outside.txt",
        r"\\server\share\outside.txt",
        r"\\?\C:\other\outside.txt",
        r"inside.txt:stream",
        r"NUL",
        r"con.txt",
        "trailing.",
        "trailing ",
        "control\x01.txt",
    ],
)
def test_windows_project_mutation_rejects_ambiguous_or_escaping_paths_portably(target):
    with pytest.raises((AgentWorkspaceError, WindowsMutationRejected, OSError, ValueError)):
        with ProjectFileMutation.open(
            _windows_workspace(),
            target,
            _windows_ops = _UnusedWindowsOps(),
        ):
            pass


class _ReparseWindowsOps(_UnusedWindowsOps):
    def __init__(self):
        self.closed = []

    def open_existing(self, path, **_kwargs):
        assert mutation._windows_key(path) == mutation._windows_key(r"C:\project")
        return 7

    def info(self, handle):
        assert handle == 7
        return mutation._WindowsHandleInfo(
            attributes = mutation._FILE_ATTRIBUTE_DIRECTORY | mutation._FILE_ATTRIBUTE_REPARSE_POINT,
            identity_options = ((11, 22),),
            size = 0,
            modified_ns = 0,
            final_path = r"C:\project",
        )

    def close(self, handle):
        self.closed.append(handle)


def test_windows_project_mutation_rejects_a_reparse_root_portably():
    operations = _ReparseWindowsOps()

    with pytest.raises(WindowsMutationRejected, match = "reparse"):
        ProjectFileMutation.open(
            _windows_workspace(),
            "inside.txt",
            _windows_ops = operations,
        )

    assert operations.closed == [7]


class _ReparseParentWindowsOps(_UnusedWindowsOps):
    def __init__(self):
        self.paths = {}
        self.closed = []

    def open_existing(self, path, **_kwargs):
        handle = len(self.paths) + 1
        self.paths[handle] = mutation._normalize_windows_path(path)
        return handle

    def info(self, handle):
        path = self.paths[handle]
        reparse = mutation._windows_key(path) == mutation._windows_key(r"C:\project\linked")
        return mutation._WindowsHandleInfo(
            attributes = mutation._FILE_ATTRIBUTE_DIRECTORY
            | (mutation._FILE_ATTRIBUTE_REPARSE_POINT if reparse else 0),
            identity_options = ((11, 33 if reparse else 22),),
            size = 0,
            modified_ns = 0,
            final_path = path,
        )

    def close(self, handle):
        self.closed.append(handle)


def test_windows_project_mutation_rejects_a_reparse_parent_portably():
    operations = _ReparseParentWindowsOps()

    with ProjectFileMutation.open(
        _windows_workspace(),
        r"linked\inside.txt",
        _windows_ops = operations,
    ) as boundary:
        with pytest.raises(WindowsMutationRejected, match = "reparse"):
            boundary.read(1024)

    assert operations.closed


class _ReplacementMetadataWindowsOps:
    def __init__(
        self,
        *,
        attributes = 0x20,
        streams = ("::$DATA",),
    ):
        self.attributes = attributes
        self.streams = tuple(streams)
        self.target_basic = mutation._WindowsBasicMetadata(1, attributes)
        self.target_dacl = mutation._WindowsDacl(True, True, b"stable-dacl")
        self.temp_basic = mutation._WindowsBasicMetadata(10, 0x20)
        self.temp_dacl = mutation._WindowsDacl(True, False, b"inherited-dacl")
        self.applied = []
        self.closed = []

    def open_existing(self, path, **kwargs):
        assert mutation._windows_key(path) == mutation._windows_key(r"C:\project\inside.txt")
        assert kwargs["read_control"] is True
        return 7

    def info(self, handle):
        if handle == 7:
            return mutation._WindowsHandleInfo(
                attributes = self.attributes,
                identity_options = ((11, 33),),
                size = len(b"before"),
                modified_ns = 99,
                final_path = r"C:\project\inside.txt",
            )
        assert handle == 8
        return mutation._WindowsHandleInfo(
            attributes = self.temp_basic.attributes,
            identity_options = ((11, 44),),
            size = len(b"after"),
            modified_ns = 100,
            final_path = r"C:\project\.unsloth_edit_temp",
        )

    def read(self, handle, _limit):
        assert handle == 7
        return b"before"

    def stream_names(self, handle):
        return self.streams if handle == 7 else ("::$DATA",)

    def basic_metadata(self, handle):
        return self.target_basic if handle == 7 else self.temp_basic

    def dacl(self, handle):
        return self.target_dacl if handle == 7 else self.temp_dacl

    def apply_dacl(self, handle, dacl):
        assert handle == 8
        self.applied.append("dacl")
        self.temp_dacl = dacl

    def apply_basic_metadata(self, handle, basic):
        assert handle == 8
        self.applied.append("basic")
        self.temp_basic = basic

    def close(self, handle):
        self.closed.append(handle)


def _portable_windows_replacement_backend(operations):
    backend = object.__new__(mutation._WindowsVerifiedMutation)
    backend.path = r"C:\project"
    backend.parts = ("inside.txt",)
    backend._ops = operations
    backend._closed = False
    return backend


def test_windows_replacement_copies_verified_dacl_and_basic_metadata_portably():
    operations = _ReplacementMetadataWindowsOps()
    backend = _portable_windows_replacement_backend(operations)

    assert backend._copy_verified_replacement_metadata(8, b"before", (11, 33)) is True
    assert operations.applied == ["dacl", "basic"]
    assert operations.temp_dacl.comparison_key == operations.target_dacl.comparison_key
    assert operations.temp_basic == operations.target_basic
    assert operations.closed == [7]


@pytest.mark.parametrize(
    "attributes",
    [
        mutation._FILE_ATTRIBUTE_COMPRESSED,
        mutation._FILE_ATTRIBUTE_ENCRYPTED,
        mutation._FILE_ATTRIBUTE_SPARSE_FILE,
    ],
)
def test_windows_replacement_rejects_unsupported_storage_attributes_portably(attributes):
    operations = _ReplacementMetadataWindowsOps(attributes = attributes)
    backend = _portable_windows_replacement_backend(operations)

    with pytest.raises(WindowsMutationRejected, match = "Compressed, encrypted, and sparse"):
        backend._copy_verified_replacement_metadata(8, b"before", (11, 33))

    assert operations.applied == []
    assert operations.closed == [7]


def test_windows_replacement_rejects_named_streams_portably():
    operations = _ReplacementMetadataWindowsOps(streams = ("::$DATA", ":secret:$DATA"))
    backend = _portable_windows_replacement_backend(operations)

    with pytest.raises(WindowsMutationRejected, match = "named streams"):
        backend._copy_verified_replacement_metadata(8, b"before", (11, 33))

    assert operations.applied == []
    assert operations.closed == [7]


@pytest.mark.skipif(os.name != "nt", reason = "native Win32 read-only metadata")
def test_native_windows_read_only_replacement_preserves_target_and_cleans_temp(tmp_path):
    target = tmp_path / "read-only.txt"
    target.write_bytes(b"before")
    target.chmod(0o444)
    try:
        with ProjectFileMutation.open(_workspace(tmp_path), target.name) as boundary:
            before, mode, identity = boundary.read(1024)
            with pytest.raises(WindowsMutationRejected, match = "Read-only"):
                boundary.replace(b"after", expect = before, mode = mode, identity = identity)
        assert target.read_bytes() == b"before"
        assert not list(tmp_path.glob(".unsloth_edit_*"))
    finally:
        target.chmod(0o666)


@pytest.mark.skipif(os.name != "nt", reason = "native Win32 mutation")
@pytest.mark.parametrize("filename", ["a", "a.py", "created.txt", "created-\U0001f9ea.txt"])
def test_native_windows_project_mutation_round_trip(tmp_path, filename):
    root = tmp_path / "repository"
    root.mkdir()
    workspace = _workspace(root)

    with ProjectFileMutation.open(workspace, filename) as boundary:
        assert boundary.create(b"first\r\n") is None

    target = root / filename
    operations = mutation._NativeWindowsMutationOps()
    before_handle = operations.open_existing(
        str(target),
        read_control = True,
        share_write = False,
        share_delete = False,
    )
    try:
        before_basic = operations.basic_metadata(before_handle)
        before_dacl = operations.dacl(before_handle)
    finally:
        operations.close(before_handle)

    with ProjectFileMutation.open(workspace, filename) as boundary:
        before, mode, identity = boundary.read(1024)
        assert before == b"first\r\n"
        assert (
            boundary.replace(
                b"second\r\n",
                expect = before,
                mode = mode,
                identity = identity,
            )
            is None
        )

    assert (root / filename).read_bytes() == b"second\r\n"
    after_handle = operations.open_existing(
        str(target),
        read_control = True,
        share_write = False,
        share_delete = False,
    )
    try:
        after_basic = operations.basic_metadata(after_handle)
        after_dacl = operations.dacl(after_handle)
    finally:
        operations.close(after_handle)
    assert after_basic.attributes == before_basic.attributes
    assert after_basic.creation_time == before_basic.creation_time
    assert after_dacl.comparison_key == before_dacl.comparison_key
    assert not list(root.glob(".unsloth_edit_*"))


@pytest.mark.skipif(os.name != "nt", reason = "native NTFS junction")
def test_native_windows_project_mutation_rejects_junction_escape(tmp_path):
    root = tmp_path / "repository"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_bytes(b"preserve")
    linked = root / "linked"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(linked), str(outside)],
        check = False,
        capture_output = True,
        text = True,
    )
    assert created.returncode == 0, created.stderr or created.stdout

    with ProjectFileMutation.open(_workspace(root), "linked/secret.txt") as boundary:
        with pytest.raises(WindowsMutationRejected, match = "reparse"):
            boundary.read(1024)

    assert secret.read_bytes() == b"preserve"


@pytest.mark.parametrize("tool_name", ["edit_file", "direct"])
def test_cancelled_edit_does_not_wait_for_or_release_another_commands_slot(
    tool_name, tmp_path, monkeypatch
):
    workspace = _workspace(tmp_path)
    session_id = _bind_project(monkeypatch, tmp_path)
    identity = (workspace.device_id, workspace.file_id)
    assert mutation.acquire_workspace_mutation_slot(identity)
    cancelled = threading.Event()
    cancelled.set()
    try:
        if tool_name == "edit_file":
            result = tools.execute_tool(
                "edit_file",
                {"path": "new.txt", "edits": [{"old_string": "", "new_string": "unsafe"}]},
                session_id = session_id,
                cancel_event = cancelled,
            )
            assert "cancelled" in result.casefold()
        else:
            with pytest.raises(AgentWorkspaceError, match = "cancelled"):
                ProjectFileMutation.open(workspace, "new.txt", cancel_event = cancelled)
        assert identity in mutation._ACTIVE_MUTATION_ROOTS
        assert not (tmp_path / "new.txt").exists()
    finally:
        mutation.release_workspace_mutation_slot(identity)


@pytest.mark.skipif(os.name != "posix", reason = "descriptor-relative scan")
def test_command_preflight_refuses_an_oversized_workspace(tmp_path, monkeypatch):
    (tmp_path / "file.txt").write_text("value", encoding = "utf-8")
    monkeypatch.setattr(execution, "_MAX_WORKSPACE_CHECK_ENTRIES", 1)
    descriptor, identity = execution._open_directory(tmp_path)
    try:
        with pytest.raises(ProjectExecutionUnavailable, match = "too large"):
            execution._assert_regular_file_links_are_internal(descriptor, identity)
    finally:
        os.close(descriptor)


def test_persisted_managed_workspace_routes_edits_and_holds_deletion_lease(tmp_path, monkeypatch):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    project = studio_db.upsert_chat_project(
        {"id": "persisted-secure", "name": "Secure project", "createdAt": 1, "updatedAt": 1}
    )
    session_id = tools.project_session_id(project["id"])
    with common.project_workspace_access(project["id"]) as workspace:
        assert workspace.kind == "managed"
        assert workspace.root == Path(project["sandboxPath"])
        assert tools.wait_for_sessions_idle([session_id], timeout = 0) is False
        result = tools.execute_tool(
            "edit_file",
            {"path": "src/value.py", "edits": [{"old_string": "", "new_string": "VALUE = 1\n"}]},
            session_id = session_id,
        )
        assert result.startswith("Created"), result
        assert (workspace.root / "src/value.py").read_text(encoding = "utf-8") == "VALUE = 1\n"
    assert tools.wait_for_sessions_idle([session_id], timeout = 0) is True
