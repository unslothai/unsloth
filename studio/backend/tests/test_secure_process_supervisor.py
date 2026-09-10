# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Security contract for project-scoped supervised processes."""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from importlib.util import find_spec

if find_spec("core.agent_workspace.mutation") is None:
    pytest.skip("Requires the optional confined edit boundary", allow_module_level = True)

from core.agent_workspace import common, execution, mutation, supervisor
from core.agent_workspace.common import AgentWorkspaceError, ProjectWorkspace
from core.agent_workspace.execution import ExecutionBoundaryStatus, ProjectExecutionUnavailable


def _workspace(root: Path, project_id: str = "supervised-project") -> ProjectWorkspace:
    metadata = root.stat(follow_symlinks = False)
    return ProjectWorkspace(
        project_id = project_id,
        root = root.resolve(strict = True),
        kind = "managed",
        device_id = int(metadata.st_dev),
        file_id = int(metadata.st_ino),
    )


class _LocalBoundary:
    backend = "bubblewrap"

    def __init__(self, workspace: ProjectWorkspace, scratch: Path) -> None:
        self.root = workspace.root
        self._root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        self.root_identity = (int(workspace.device_id), int(workspace.file_id))
        self.scratch = scratch
        self.closed = False
        self.slot = False

    def acquire_execution_slot(self, cancel_event) -> bool:
        if not mutation.acquire_workspace_mutation_slot(self.root_identity, cancel_event):
            return False
        self.slot = True
        return True

    def apply_environment(self, environment):
        isolated = dict(environment)
        isolated.update(
            {
                "HOME": str(self.scratch),
                "TMP": str(self.scratch),
                "TEMP": str(self.scratch),
                "TMPDIR": str(self.scratch),
            }
        )
        return isolated

    def wrap_argv(self, argv):
        return list(argv)

    def popen_kwargs(self, preexec_fn):
        options = {"cwd": str(self.root)}
        if preexec_fn is not None:
            options["preexec_fn"] = preexec_fn
        return options

    def close(self):
        if not self.closed:
            os.close(self._root_fd)
        if self.slot:
            mutation.release_workspace_mutation_slot(self.root_identity)
            self.slot = False
        self.closed = True


class _LocalLifecycle:
    instances = []

    def __init__(self) -> None:
        self.closed = False
        self.bound = False
        self.instances.append(self)

    def attach_execution_fence(self, descriptor):
        self.execution_fence_fd = descriptor

    def wrap_argv(self, argv):
        return list(argv)

    def add_popen_options(self, options):
        return dict(options)

    def after_spawn(self, _process):
        return None

    def bind(
        self,
        _process,
        cancel_event = None,
    ):
        self.bound = True
        return cancel_event is None or not cancel_event.is_set()

    def release(self, cancel_event):
        assert self.bound is True
        return cancel_event is None or not cancel_event.is_set()

    def terminate_and_prove(self, process):
        return supervisor._terminate_and_reap(process)

    def close(self):
        self.closed = True


@pytest.fixture
def local_supervisor(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("The local process fixture uses POSIX directory descriptors")
    root = tmp_path / "repository"
    scratch = tmp_path / "scratch"
    root.mkdir()
    scratch.mkdir()
    workspace = _workspace(root)
    lease_active = {"value": False}
    boundaries = []

    @contextlib.contextmanager
    def access(project_id: str):
        assert project_id == workspace.project_id
        assert lease_active["value"] is False
        lease_active["value"] = True
        try:
            yield workspace
        finally:
            lease_active["value"] = False

    def open_boundary(received: ProjectWorkspace):
        assert received is workspace
        boundary = _LocalBoundary(received, scratch)
        boundaries.append(boundary)
        return boundary

    monkeypatch.setattr(common, "project_workspace_access", access)
    monkeypatch.setattr(execution.ProjectExecutionBoundary, "open", open_boundary)
    monkeypatch.setattr(
        supervisor,
        "supervised_process_status",
        lambda: ExecutionBoundaryStatus(True, "bubblewrap", None),
    )
    monkeypatch.setattr(supervisor, "initialize_parent_lifetime", lambda: None)
    monkeypatch.setattr(supervisor, "adopt_pid", lambda _pid: None)
    monkeypatch.setattr(supervisor, "forget_pid", lambda _pid: None)
    # Keep the production lifetime thread: Linux PDEATHSIG belongs to the
    # spawning thread, so a short-lived fixture worker would kill the child as
    # soon as Popen returned and invalidate every process-behavior assertion.
    monkeypatch.setattr(supervisor, "_start_quarantine_retry_owner", lambda: None)
    if os.name == "nt":
        # Portable output/ownership unit tests use a local lifecycle double.
        # Native Windows command tests still require refusal before Popen.
        fence = threading.Lock()

        def acquire_test_fence(_fence_id, cancel_event, deadline):
            while not fence.acquire(timeout = 0.01):
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("fence still held")
            return 0

        monkeypatch.setattr(supervisor, "_acquire_project_execution_fence", acquire_test_fence)
        monkeypatch.setattr(
            supervisor, "_release_project_execution_fence", lambda _fd: fence.release()
        )

    _LocalLifecycle.instances.clear()
    monkeypatch.setattr(supervisor, "_BubblewrapLifecycle", _LocalLifecycle)
    return workspace, lease_active, boundaries


def test_supervisor_owns_workspace_lease_slot_and_minimal_environment(
    local_supervisor, monkeypatch
):
    workspace, lease_active, boundaries = local_supervisor
    parent_only = {
        "OPENAI_API_KEY": "operator-secret",
        "AWS_PROFILE": "operator-profile",
        "CLOUDSDK_CONFIG": "/operator/gcloud",
        "GIT_CONFIG_GLOBAL": "/operator/gitconfig",
        "LD_PRELOAD": "/operator/inject.so",
        "DYLD_INSERT_LIBRARIES": "/operator/inject.dylib",
        "PYTHONPATH": "/operator/python",
        "PYTHONHOME": "/operator/home",
        "NODE_OPTIONS": "--require=/operator/node.js",
        "BASH_ENV": "/operator/bashrc",
        "SSH_AUTH_SOCK": "/operator/agent.sock",
    }
    for name, value in parent_only.items():
        monkeypatch.setenv(name, value)

    source = (
        "import json, os; "
        "print(json.dumps({key: os.environ.get(key) for key in "
        f"{list(parent_only)!r}}})); "
        "print(os.environ['UNSLOTH_STUDIO_PROJECT_ID'])"
    )
    real_popen = supervisor.subprocess.Popen

    def checked_popen(*args, **kwargs):
        assert lease_active["value"] is True
        assert boundaries[-1].slot is True
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.STDOUT
        assert kwargs["env"]["HOME"] == str(boundaries[-1].scratch)
        assert "/usr/local/bin" in kwargs["env"]["PATH"].split(os.pathsep)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(supervisor.subprocess, "Popen", checked_popen)
    result = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", source],
    )

    lines = result.output.splitlines()
    assert result.status == "passed"
    assert result.exit_code == 0
    child_environment = json.loads(lines[0])
    assert {key: child_environment[key] for key in parent_only if key != "PYTHONPATH"} == {
        key: None for key in parent_only if key != "PYTHONPATH"
    }
    assert "/operator/python" not in child_environment["PYTHONPATH"]
    assert str(workspace.root) in child_environment["PYTHONPATH"]
    assert str(boundaries[-1].scratch) in child_environment["PYTHONPATH"]
    assert lines[1] == workspace.project_id
    assert lease_active["value"] is False
    assert boundaries[-1].closed is True
    assert boundaries[-1].slot is False

    assert mutation.acquire_workspace_mutation_slot(boundaries[-1].root_identity) is True
    mutation.release_workspace_mutation_slot(boundaries[-1].root_identity)


def test_project_python_preserves_import_cwd_file_and_streaming_contract(local_supervisor):
    workspace, _lease_active, boundaries = local_supervisor
    (workspace.root / "helper.py").write_text("VALUE = 42\n", encoding = "utf-8")
    (workspace.root / "data.csv").write_text("project-data", encoding = "utf-8")
    streamed = []
    source = (
        "from pathlib import Path; import helper, sys; "
        "print(Path.cwd()); print(helper.VALUE); print(Path(__file__).name); "
        "print(Path(__file__).with_name('data.csv').read_text()); "
        "assert Path(__file__).read_text(); "
        "Path(sys.argv[0]).with_name('result.txt').write_text('kept'); "
        "Path('/mnt/data/remapped.txt').write_text('mapped', encoding='utf-8')"
    )

    result = supervisor.run_project_python(
        workspace.project_id,
        source,
        output_callback = streamed.append,
    )

    assert result.status == "passed"
    assert result.exit_code == 0
    assert "42" in result.output
    assert str(workspace.root) in result.output
    assert "studio_exec_" in result.output
    assert "project-data" in result.output
    assert (workspace.root / "result.txt").read_text() == "kept"
    assert not list(workspace.root.glob("studio_exec_*.py"))
    assert "".join(streamed) == result.output
    assert (workspace.root / "remapped.txt").read_text(encoding = "utf-8") == "mapped"
    assert not list(boundaries[-1].scratch.glob("studio_exec_*.py"))


def test_supervisor_bounds_combined_stdout_and_stderr(local_supervisor):
    workspace, _lease_active, _boundaries = local_supervisor
    streamed = []
    source = (
        "import sys; print('o' * 2048); print('e' * 2048, file=sys.stderr); raise SystemExit(7)"
    )

    result = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", source],
        output_limit_bytes = 1024,
        output_callback = streamed.append,
    )

    assert result.status == "failed"
    assert result.exit_code == 7
    assert result.output_bytes >= 4098
    assert len(result.output.removesuffix(result.truncation_notice).encode("utf-8")) == 1024
    assert result.output_truncated is True
    assert result.truncation_notice
    assert "".join(streamed) == result.output
    assert "".join(streamed).count(result.truncation_notice) == 1


@pytest.mark.skipif(os.name != "posix", reason = "Native process fences use POSIX flock")
def test_execution_fence_excludes_other_processes_and_rejects_unsafe_files(tmp_path):
    fence_id = "project:fence-test"
    descriptor = supervisor._acquire_project_execution_fence(fence_id, None, time.monotonic() + 1)
    try:
        contender = subprocess.run(
            [
                sys.executable,
                "-c",
                "import fcntl,os,sys; "
                "fd=os.open(sys.argv[1], os.O_RDWR); "
                "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)",
                supervisor._project_execution_fence_path(fence_id),
            ],
            capture_output = True,
            timeout = 5,
        )
        assert contender.returncode != 0
        assert b"BlockingIOError" in contender.stderr
        with pytest.raises(TimeoutError):
            supervisor._acquire_project_execution_fence(fence_id, None, time.monotonic() + 0.02)
    finally:
        supervisor._release_project_execution_fence(descriptor)
    descriptor = supervisor._acquire_project_execution_fence(fence_id, None, time.monotonic() + 1)
    supervisor._release_project_execution_fence(descriptor)
    lock = Path(supervisor._project_execution_fence_path(fence_id))
    lock.unlink()
    target = tmp_path / "outside-lock"
    target.write_text("do not touch")
    lock.symlink_to(target)
    with pytest.raises(OSError):
        supervisor._acquire_project_execution_fence(fence_id, None, time.monotonic() + 1)
    assert target.read_text() == "do not touch"


def test_review_preflight_refuses_before_popen_and_releases_ownership(
    local_supervisor, monkeypatch
):
    workspace, lease_active, boundaries = local_supervisor

    def refuse(opened, argv):
        assert opened is workspace
        assert argv == (sys.executable, "-c", "print('never')")
        assert lease_active["value"] and boundaries[-1].slot
        raise AgentWorkspaceError("Reviewed settings changed")

    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("spawned a command with stale review"),
    )
    with pytest.raises(AgentWorkspaceError, match = "Reviewed settings changed"):
        supervisor.run_project_process(
            workspace.project_id, [sys.executable, "-c", "print('never')"], before_start = refuse
        )
    assert not lease_active["value"]
    assert boundaries[-1].closed and not boundaries[-1].slot


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "native blocked namespace")
def test_review_preflight_rechecks_before_releasing_native_user_code(tmp_path, monkeypatch):
    status = supervisor.supervised_process_status()
    if not status.available:
        if os.environ.get("UNSLOTH_SECURE_BOUNDARY_REQUIRED") == "1":
            pytest.fail(status.reason)
        pytest.skip(status.reason)
    workspace = _workspace(tmp_path)
    marker = tmp_path / "must-not-run"
    calls = []

    @contextlib.contextmanager
    def access(project_id):
        assert project_id == workspace.project_id
        yield workspace

    def refuse_release(opened, argv):
        assert opened is workspace
        calls.append(argv)
        if len(calls) == 2:
            raise AgentWorkspaceError("Review revoked during namespace setup")

    monkeypatch.setattr(common, "project_workspace_access", access)
    with pytest.raises(AgentWorkspaceError, match = "Review revoked"):
        supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
            before_start = refuse_release,
        )
    assert len(calls) == 2
    assert not marker.exists()
    identity = (workspace.device_id, workspace.file_id)
    assert mutation.acquire_workspace_mutation_slot(identity)
    mutation.release_workspace_mutation_slot(identity)


def test_live_utf8_truncation_cannot_skip_a_codepoint_then_resume(monkeypatch):
    chunks = [
        bytes.fromhex(value)
        for value in (
            "91",
            "547819d54e94",
            "289fdc18d883d5da",
            "ae",
            "d6b1f818",
            "2da6dc810cf9",
            "",
        )
    ]
    pending = iter(chunks)
    streamed = []
    monkeypatch.setattr(supervisor.os, "read", lambda *_args: next(pending))
    output = supervisor._OutputBuffer(12, streamed.append)
    assert output.read_available(123, max_chunks = 64)
    terminal, _total, truncated, notice = output.result()
    assert truncated
    assert "".join(streamed) == terminal
    assert streamed[-1] == notice
    assert terminal.startswith("\ufffdTx\x19\ufffdN\n")


def test_supervisor_bounds_rendered_invalid_utf8(local_supervisor):
    workspace, _lease_active, _boundaries = local_supervisor
    streamed = []
    result = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", "import os; os.write(1, b'\\xff' * 1024)"],
        output_limit_bytes = 1024,
        output_callback = streamed.append,
    )

    assert result.status == "passed"
    assert result.output_bytes == 1024
    assert len(result.output.removesuffix(result.truncation_notice).encode("utf-8")) <= 1024
    assert result.output_truncated is True
    assert result.output.count(result.truncation_notice) == 1
    assert "".join(streamed) == result.output


def test_slot_wait_is_preparation_and_honors_cancellation_before_popen(
    local_supervisor, monkeypatch
):
    workspace, _lease_active, boundaries = local_supervisor
    identity = (int(workspace.device_id), int(workspace.file_id))
    assert mutation.acquire_workspace_mutation_slot(identity) is True
    cancellation = threading.Event()
    timer = threading.Timer(0.08, cancellation.set)
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("spawned before acquiring the mutation slot"),
    )
    timer.start()
    try:
        result = supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", "print('late')"],
            timeout_seconds = 0.01,
            cancel_event = cancellation,
        )
    finally:
        timer.join(timeout = 1)
        mutation.release_workspace_mutation_slot(identity)

    assert result.status == "cancelled"
    assert result.exit_code is None
    assert boundaries[-1].closed is True
    assert "after bubblewrap" in supervisor.run_project_process.__doc__


def test_pre_cancel_returns_before_capability_probe(monkeypatch):
    cancellation = threading.Event()
    cancellation.set()
    monkeypatch.setattr(
        supervisor,
        "supervised_process_status",
        lambda: pytest.fail("probed the host for a pre-cancelled command"),
    )

    result = supervisor.run_project_process(
        "project",
        ["echo", "cancelled"],
        cancel_event = cancellation,
    )

    assert result == supervisor.ProjectProcessResult("cancelled", None, "", 0, False)


@pytest.mark.skipif(not hasattr(os, "pipe2"), reason = "POSIX lifecycle pipes")
def test_bubblewrap_block_fd_is_blocking_until_pidfd_binding():
    lifecycle = supervisor._BubblewrapLifecycle()
    try:
        assert os.get_blocking(lifecycle._status_read) is False
        assert os.get_blocking(lifecycle._block_read) is True
        options = lifecycle.add_popen_options({"pass_fds": ()})
        assert lifecycle._status_write in options["pass_fds"]
        assert lifecycle._block_read in options["pass_fds"]
        assert lifecycle._block_write in options["pass_fds"]
    finally:
        lifecycle.close()


@pytest.mark.skipif(not hasattr(os, "pipe2"), reason = "POSIX lifecycle pipes")
def test_invalid_pidfd_never_counts_as_process_exit():
    descriptor, peer = os.pipe()
    os.close(descriptor)
    try:
        with pytest.raises(
            supervisor.ProjectProcessContainmentError,
            match = "pidfd became invalid",
        ):
            supervisor._BubblewrapLifecycle._pidfd_ready(descriptor, 0)
    finally:
        os.close(peer)


def _install_test_pipe2(monkeypatch) -> None:
    if hasattr(os, "pipe2"):
        return

    def pipe2(flags):
        read_descriptor, write_descriptor = os.pipe()
        if flags & getattr(os, "O_NONBLOCK", 0):
            os.set_blocking(read_descriptor, False)
            os.set_blocking(write_descriptor, False)
        return read_descriptor, write_descriptor

    monkeypatch.setattr(supervisor.os, "pipe2", pipe2, raising = False)


@pytest.mark.skipif(
    os.name != "posix",
    reason = "POSIX process-group setup cleanup",
)
@pytest.mark.parametrize(
    "status_payload",
    [None, b"{malformed status}\n"],
    ids = ["eof", "malformed"],
)
def test_unbound_status_failure_proves_group_before_idempotent_release(status_payload, monkeypatch):
    _install_test_pipe2(monkeypatch)
    lifecycle = supervisor._BubblewrapLifecycle()
    if status_payload is not None:
        os.write(lifecycle._status_write, status_payload)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout = subprocess.PIPE,
        start_new_session = True,
    )
    lifecycle.after_spawn(process)
    calls = {"unbound": 0, "boundary": 0, "lease": 0}
    real_unbound_cleanup = lifecycle.terminate_unbound_and_prove

    def count_unbound_cleanup(received_process):
        calls["unbound"] += 1
        return real_unbound_cleanup(received_process)

    class Boundary:
        def close(self):
            calls["boundary"] += 1
            if calls["boundary"] == 1:
                raise RuntimeError("transient boundary close failure")

    class Lease:
        def __exit__(self, _kind, _value, _traceback):
            calls["lease"] += 1

    monkeypatch.setattr(lifecycle, "terminate_unbound_and_prove", count_unbound_cleanup)
    run = supervisor._QuarantinedRun(
        lease = Lease(),
        boundary = Boundary(),
        lifecycle = lifecycle,
        process = process,
        after_spawn_done = True,
    )
    try:
        with pytest.raises(RuntimeError, match = "transient boundary close failure"):
            supervisor._advance_quarantined_run(run)

        assert process.poll() is not None
        assert lifecycle._released is False
        assert run.tree_proven is True
        assert run.lifecycle_closed is True
        assert run.boundary_closed is False
        assert run.lease_released is False

        supervisor._advance_quarantined_run(run)
        supervisor._advance_quarantined_run(run)

        assert calls == {"unbound": 1, "boundary": 2, "lease": 1}
        assert run.boundary_closed is True
        assert run.lease_released is True
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout = 5)
        lifecycle.close()


@pytest.mark.skipif(
    os.name != "posix",
    reason = "POSIX process-group setup cleanup",
)
def test_unbound_status_failure_keeps_locks_when_group_death_is_unproven(monkeypatch):
    _install_test_pipe2(monkeypatch)
    lifecycle = supervisor._BubblewrapLifecycle()
    process = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout = subprocess.PIPE,
        start_new_session = True,
    )
    lifecycle.after_spawn(process)
    process.wait(timeout = 5)
    released = {"boundary": False, "lease": False}

    class Boundary:
        def close(self):
            released["boundary"] = True

    class Lease:
        def __exit__(self, _kind, _value, _traceback):
            released["lease"] = True

    monkeypatch.setattr(
        lifecycle,
        "_wait_for_process_group_exit",
        lambda *_args, **_kwargs: False,
    )
    run = supervisor._QuarantinedRun(
        lease = Lease(),
        boundary = Boundary(),
        lifecycle = lifecycle,
        process = process,
        after_spawn_done = True,
    )
    try:
        for _attempt in range(2):
            with pytest.raises(
                supervisor.ProjectProcessContainmentError,
                match = "remains quarantined",
            ):
                supervisor._advance_quarantined_run(run)
            assert run.tree_proven is False
            assert released == {"boundary": False, "lease": False}
    finally:
        lifecycle.close()


@pytest.mark.skipif(not hasattr(os, "pipe2"), reason = "POSIX lifecycle pipes")
def test_bubblewrap_bind_latches_cancellation_while_status_is_pending(monkeypatch):
    lifecycle = supervisor._BubblewrapLifecycle()
    cancellation = threading.Event()
    bound = []

    def report_status():
        cancellation.set()
        time.sleep(0.1)
        cancellation.clear()
        time.sleep(0.1)
        os.write(
            lifecycle._status_write,
            b'{"child-pid": 4242, "pid-namespace": 1234}\n',
        )

    monkeypatch.setattr(
        lifecycle,
        "_bind_child",
        lambda process, document: bound.append((process, document)),
    )
    reporter = threading.Thread(target = report_status)
    reporter.start()
    process = object()
    try:
        assert lifecycle.bind(process, cancellation) is False
    finally:
        reporter.join(timeout = 1)
        lifecycle.close()

    assert bound == [(process, {"child-pid": 4242, "pid-namespace": 1234})]


def test_supervisor_times_out_and_releases_only_after_reap(local_supervisor, monkeypatch):
    workspace, _lease_active, boundaries = local_supervisor
    observed = {}
    real_popen = supervisor.subprocess.Popen

    def capture_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        observed["process"] = process
        return process

    monkeypatch.setattr(supervisor.subprocess, "Popen", capture_popen)
    started = time.monotonic()
    result = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout_seconds = 0.1,
    )

    assert time.monotonic() - started < 5
    assert result.status == "timed_out"
    assert result.exit_code is None
    assert observed["process"].poll() is not None
    assert boundaries[-1].closed is True


def test_supervisor_cancellation_is_bounded_and_deterministic(local_supervisor, monkeypatch):
    workspace, _lease_active, boundaries = local_supervisor
    cancellation = threading.Event()
    spawned = threading.Event()
    real_popen = supervisor.subprocess.Popen

    def capture_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.set()
        return process

    monkeypatch.setattr(supervisor.subprocess, "Popen", capture_popen)
    timer = threading.Thread(
        target = lambda: (spawned.wait(timeout = 3), cancellation.set()),
        daemon = True,
    )
    timer.start()
    result = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cancel_event = cancellation,
    )
    timer.join(timeout = 1)

    assert result.status == "cancelled"
    assert result.exit_code is None
    assert boundaries[-1].closed is True


def test_cancellation_during_lifecycle_bind_never_releases_command(local_supervisor, monkeypatch):
    workspace, _lease_active, boundaries = local_supervisor
    cancellation = threading.Event()
    original_bind = _LocalLifecycle.bind
    original_release = _LocalLifecycle.release
    release_called = {"value": False}

    def cancel_during_bind(
        self,
        process,
        cancel_event = None,
    ):
        assert cancel_event is cancellation
        cancellation.set()
        bound = original_bind(self, process, cancel_event)
        cancellation.clear()
        return bound

    def observe_release(self, cancel_event):
        release_called["value"] = True
        return original_release(self, cancel_event)

    monkeypatch.setattr(_LocalLifecycle, "bind", cancel_during_bind)
    monkeypatch.setattr(_LocalLifecycle, "release", observe_release)

    result = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", "raise SystemExit('must not run')"],
        cancel_event = cancellation,
    )

    assert result.status == "cancelled"
    assert release_called["value"] is False
    assert boundaries[-1].closed is True


def test_post_spawn_exception_reaps_before_boundary_release(local_supervisor, monkeypatch):
    workspace, _lease_active, boundaries = local_supervisor
    observed = {}
    real_popen = supervisor.subprocess.Popen

    def capture_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        observed["process"] = process
        return process

    monkeypatch.setattr(supervisor.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(
        supervisor,
        "adopt_pid",
        lambda _pid: (_ for _ in ()).throw(RuntimeError("adoption failed")),
    )

    with pytest.raises(RuntimeError, match = "adoption failed"):
        supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", "import time; time.sleep(30)"],
        )

    assert observed["process"].poll() is not None
    assert boundaries[-1].closed is True
    assert boundaries[-1].slot is False


def test_reaping_failure_quarantines_process_lease_slot_and_descriptors(
    local_supervisor, monkeypatch
):
    workspace, lease_active, boundaries = local_supervisor
    observed = {}
    real_popen = supervisor.subprocess.Popen
    real_terminate = _LocalLifecycle.terminate_and_prove
    failed_once = {"value": False}

    def capture_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        observed["process"] = process
        return process

    def fail_once(self, process):
        if not failed_once["value"]:
            failed_once["value"] = True
            raise RuntimeError("reap proof failed")
        return real_terminate(self, process)

    monkeypatch.setattr(supervisor.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(_LocalLifecycle, "terminate_and_prove", fail_once)

    with pytest.raises(
        supervisor.ProjectProcessContainmentError,
        match = "workspace lease and mutation slot remain locked",
    ):
        supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds = 0.05,
        )

    assert observed["process"].poll() is None
    assert lease_active["value"] is True
    assert boundaries[-1].closed is False
    assert boundaries[-1].slot is True
    with pytest.raises(TimeoutError):
        supervisor._acquire_project_execution_fence(
            "project:" + workspace.project_id, None, time.monotonic() + 0.02
        )
    assert _LocalLifecycle.instances[-1].closed is False
    cancelled = threading.Event()
    cancelled.set()
    assert (
        mutation.acquire_workspace_mutation_slot(boundaries[-1].root_identity, cancelled) is False
    )

    assert supervisor.retry_quarantined_project_processes() == 0
    assert observed["process"].poll() is not None
    assert lease_active["value"] is False
    assert boundaries[-1].closed is True
    assert boundaries[-1].slot is False
    assert _LocalLifecycle.instances[-1].closed is True


def test_failed_quarantine_retry_keeps_lease_and_slot_locked(local_supervisor, monkeypatch):
    workspace, lease_active, boundaries = local_supervisor
    real_terminate = _LocalLifecycle.terminate_and_prove
    cleanup_allowed = {"value": False}

    def fail_until_allowed(self, process):
        if not cleanup_allowed["value"]:
            raise RuntimeError("pidfd cleanup is temporarily unavailable")
        return real_terminate(self, process)

    monkeypatch.setattr(_LocalLifecycle, "terminate_and_prove", fail_until_allowed)

    with pytest.raises(supervisor.ProjectProcessContainmentError):
        supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds = 0.05,
        )

    assert supervisor.retry_quarantined_project_processes() == 1
    assert lease_active["value"] is True
    assert boundaries[-1].slot is True
    assert boundaries[-1].closed is False

    cleanup_allowed["value"] = True
    assert supervisor.retry_quarantined_project_processes() == 0
    assert lease_active["value"] is False
    assert boundaries[-1].slot is False
    assert boundaries[-1].closed is True


@pytest.mark.parametrize("stop_kind", ["cancel", "preparation_timeout"])
def test_stuck_spawn_returns_bounded_but_keeps_lease_and_slot(
    local_supervisor, monkeypatch, stop_kind
):
    workspace, lease_active, boundaries = local_supervisor
    real_popen = supervisor.subprocess.Popen
    entered = threading.Event()
    allow_spawn = threading.Event()
    cancellation = threading.Event()

    def blocked_popen(*args, **kwargs):
        entered.set()
        assert allow_spawn.wait(5)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(supervisor.subprocess, "Popen", blocked_popen)
    canceller = None
    if stop_kind == "cancel":

        def cancel_after_start():
            assert entered.wait(2)
            cancellation.set()

        canceller = threading.Thread(target = cancel_after_start)
        canceller.start()
    else:
        monkeypatch.setattr(supervisor, "_SPAWN_WAIT_SECONDS", 0.05)
    started = time.monotonic()
    try:
        if stop_kind == "cancel":
            result = supervisor.run_project_process(
                workspace.project_id,
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cancel_event = cancellation,
            )
        else:
            with pytest.raises(
                supervisor.ProjectProcessContainmentError,
                match = "spawn is still resolving",
            ):
                supervisor.run_project_process(
                    workspace.project_id,
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                )
    finally:
        allow_spawn.set()
        if canceller is not None:
            canceller.join(timeout = 2)

    assert time.monotonic() - started < 1
    if stop_kind == "cancel":
        assert result.status == "cancelled"
    assert lease_active["value"] is True
    assert boundaries[-1].slot is True
    with supervisor._QUARANTINE_LOCK:
        run = supervisor._QUARANTINED_RUNS[-1]
    assert run.spawn_attempt is not None
    assert run.spawn_attempt._done.wait(2)
    assert supervisor.retry_quarantined_project_processes() == 0
    assert lease_active["value"] is False
    assert boundaries[-1].slot is False


def test_quarantine_retries_only_the_failed_release_phase(local_supervisor, monkeypatch):
    workspace, lease_active, _boundaries = local_supervisor
    lifecycle_calls = {"terminate": 0, "close": 0}
    boundary_calls = {"close": 0}
    lease_calls = {"exit": 0}
    real_terminate = _LocalLifecycle.terminate_and_prove
    real_lifecycle_close = _LocalLifecycle.close
    real_boundary_close = _LocalBoundary.close

    class FailingLease:
        def __enter__(self):
            assert lease_active["value"] is False
            lease_active["value"] = True
            return workspace

        def __exit__(self, _kind, _value, _traceback):
            lease_calls["exit"] += 1
            if lease_calls["exit"] == 1:
                raise RuntimeError("lease close failed")
            lease_active["value"] = False

    def fail_first_termination(self, process):
        lifecycle_calls["terminate"] += 1
        if lifecycle_calls["terminate"] == 1:
            raise RuntimeError("pidfd proof failed")
        return real_terminate(self, process)

    def count_lifecycle_close(self):
        lifecycle_calls["close"] += 1
        return real_lifecycle_close(self)

    def fail_first_boundary_close(self):
        boundary_calls["close"] += 1
        if boundary_calls["close"] == 1:
            raise RuntimeError("boundary close failed")
        return real_boundary_close(self)

    monkeypatch.setattr(common, "project_workspace_access", lambda _project_id: FailingLease())
    monkeypatch.setattr(_LocalLifecycle, "terminate_and_prove", fail_first_termination)
    monkeypatch.setattr(_LocalLifecycle, "close", count_lifecycle_close)
    monkeypatch.setattr(_LocalBoundary, "close", fail_first_boundary_close)

    with pytest.raises(supervisor.ProjectProcessContainmentError):
        supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds = 0.05,
        )

    assert supervisor.retry_quarantined_project_processes() == 1
    with supervisor._QUARANTINE_LOCK:
        run = supervisor._QUARANTINED_RUNS[-1]
    assert run.tree_proven is True
    assert run.lifecycle_closed is True
    assert run.boundary_closed is False
    assert run.lease_released is False

    assert supervisor.retry_quarantined_project_processes() == 1
    assert run.boundary_closed is True
    assert run.lease_released is False

    assert supervisor.retry_quarantined_project_processes() == 0
    assert lifecycle_calls == {"terminate": 2, "close": 1}
    assert boundary_calls == {"close": 2}
    assert lease_calls == {"exit": 2}
    assert lease_active["value"] is False


def test_normal_service_owner_retries_quarantine_until_release():
    released = threading.Event()
    calls = {"boundary": 0}

    class Lifecycle:
        def close(self):
            return None

    class Boundary:
        def close(self):
            calls["boundary"] += 1
            if calls["boundary"] == 1:
                raise RuntimeError("transient boundary failure")

    class Lease:
        def __exit__(self, _kind, _value, _traceback):
            released.set()

    with supervisor._QUARANTINE_LOCK:
        assert not supervisor._QUARANTINED_RUNS
    supervisor._register_quarantined_run(
        supervisor._QuarantinedRun(
            lease = Lease(),
            boundary = Boundary(),
            lifecycle = Lifecycle(),
            tree_proven = True,
        )
    )

    assert released.wait(2)
    assert calls["boundary"] == 2
    assert supervisor.retry_quarantined_project_processes() == 0


@pytest.mark.parametrize(
    "platform_status",
    [
        ExecutionBoundaryStatus(
            False,
            None,
            "Project command execution is disabled until a Windows filesystem sandbox is available.",
        ),
        ExecutionBoundaryStatus(True, "sandbox-exec", None),
    ],
)
def test_unsupported_process_boundaries_fail_before_workspace_or_popen(
    platform_status, monkeypatch
):
    monkeypatch.setattr(
        execution,
        "execution_boundary_status",
        lambda probe = True: platform_status,
    )
    monkeypatch.setattr(
        common,
        "project_workspace_access",
        lambda _project_id: pytest.fail("resolved a workspace without process containment"),
    )
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("spawned without process containment"),
    )

    with pytest.raises(ProjectExecutionUnavailable):
        supervisor.run_project_process("project", ["echo", "unsafe"])


def test_failed_exact_lifecycle_probe_precedes_workspace_and_user_popen(monkeypatch):
    monkeypatch.setattr(supervisor.os, "pidfd_open", lambda *_args: 1, raising = False)
    monkeypatch.setattr(
        supervisor.signal,
        "pidfd_send_signal",
        lambda *_args: None,
        raising = False,
    )
    monkeypatch.setattr(
        execution,
        "execution_boundary_status",
        lambda probe = False: ExecutionBoundaryStatus(True, "bubblewrap", None),
    )
    monkeypatch.setattr(execution, "_bubblewrap_path", lambda: "/usr/bin/bwrap")
    monkeypatch.setattr(supervisor, "_bubblewrap_identity", lambda _path: (1, 2, 3, 4))
    monkeypatch.setattr(
        supervisor,
        "_probe_supervised_backend",
        lambda _path, _identity: (False, "pidfd lifecycle probe failed"),
    )
    monkeypatch.setattr(
        common,
        "project_workspace_access",
        lambda _project_id: pytest.fail("resolved a workspace after a failed lifecycle probe"),
    )
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("spawned the caller's command after probe failure"),
    )

    with pytest.raises(ProjectExecutionUnavailable, match = "pidfd lifecycle probe failed"):
        supervisor.run_project_process("project", ["echo", "unsafe"])


def test_supervisor_rejects_invalid_inputs_before_capability_probe(monkeypatch):
    monkeypatch.setattr(
        supervisor,
        "supervised_process_status",
        lambda: pytest.fail("probed the host before validating caller input"),
    )

    with pytest.raises(AgentWorkspaceError, match = "sequence of strings"):
        supervisor.run_project_process("project", "echo unsafe")
    with pytest.raises(AgentWorkspaceError, match = "invalid item"):
        supervisor.run_project_process("project", ["echo", "bad\x00item"])
    with pytest.raises(AgentWorkspaceError, match = "timeout"):
        supervisor.run_project_process("project", ["echo"], timeout_seconds = float("inf"))
    with pytest.raises(AgentWorkspaceError, match = "output limit"):
        supervisor.run_project_process("project", ["echo"], output_limit_bytes = True)
    with pytest.raises(AgentWorkspaceError, match = "Project id"):
        supervisor.run_project_process(" project\n", ["echo"])


def test_public_api_has_no_root_identity_or_boundary_injection():
    parameters = inspect.signature(supervisor.run_project_process).parameters

    assert set(parameters) == {
        "project_id",
        "argv",
        "timeout_seconds",
        "output_limit_bytes",
        "cancel_event",
        "output_callback",
        "before_start",
    }
    assert "root" not in inspect.signature(supervisor.run_project_python).parameters
    assert "identity" not in inspect.signature(supervisor.run_project_python).parameters
    assert "boundary" not in inspect.signature(supervisor.run_project_python).parameters


@pytest.mark.skipif(sys.platform != "win32", reason = "native Windows process boundary")
def test_native_windows_supervisor_fails_before_popen(monkeypatch):
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("spawned on unsupported Windows containment"),
    )

    with pytest.raises(ProjectExecutionUnavailable, match = "Windows"):
        supervisor.run_project_process("unresolved", ["cmd.exe", "/c", "echo unsafe"])


@pytest.mark.skipif(sys.platform != "darwin", reason = "native macOS process boundary")
def test_native_macos_rejects_sandbox_exec_before_detached_setsid_payload(monkeypatch):
    status = execution.execution_boundary_status()
    if not status.available:
        if os.environ.get("UNSLOTH_SECURE_BOUNDARY_REQUIRED") == "1":
            pytest.fail(status.reason or "the required macOS filesystem boundary is unavailable")
        pytest.skip(status.reason or "macOS sandbox-exec is unavailable")
    assert status.backend == "sandbox-exec"
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("started the detached setsid payload"),
    )
    payload = (
        "import subprocess, sys; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
        "start_new_session=True)"
    )

    with pytest.raises(ProjectExecutionUnavailable, match = "detached descendants"):
        supervisor.run_project_process("unresolved", [sys.executable, "-c", payload])


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "native Linux PID namespace")
def test_native_linux_bubblewrap_kills_detached_setsid_descendant(tmp_path, monkeypatch):
    status = supervisor.supervised_process_status()
    if not status.available:
        if os.environ.get("UNSLOTH_SECURE_BOUNDARY_REQUIRED") == "1":
            pytest.fail(status.reason or "the required Linux process boundary is unavailable")
        pytest.skip(status.reason or "Linux bubblewrap is unavailable")

    root = tmp_path / "repository"
    root.mkdir()
    workspace = _workspace(root)

    @contextlib.contextmanager
    def access(project_id: str):
        assert project_id == workspace.project_id
        yield workspace

    monkeypatch.setattr(common, "project_workspace_access", access)
    slot_released = root / "slot-released.txt"
    marker = root / "late-write.txt"
    child_code = (
        "import time; from pathlib import Path; "
        f"released=Path({str(slot_released)!r}); marker=Path({str(marker)!r}); "
        "deadline=time.monotonic()+10; "
        'exec("while time.monotonic() < deadline and not released.exists():\\n'
        '    time.sleep(0.005)"); '
        "marker.write_text('escaped', encoding='utf-8') if released.exists() else None"
    )
    payload = (
        "import subprocess, sys; "
        f"code={child_code!r}; "
        "child=subprocess.Popen([sys.executable, '-c', code], start_new_session=True, "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "print(child.pid, flush=True); "
        "import time; time.sleep(30)"
    )

    result = supervisor.run_project_process(
        workspace.project_id,
        [sys.executable, "-c", payload],
        timeout_seconds = 0.2,
    )
    identity = (int(workspace.device_id), int(workspace.file_id))
    assert mutation.acquire_workspace_mutation_slot(identity) is True
    try:
        slot_released.write_text("released", encoding = "utf-8")
    finally:
        mutation.release_workspace_mutation_slot(identity)
    time.sleep(0.5)

    assert result.status == "timed_out", result.output
    assert result.exit_code is None
    assert result.output.strip().isdigit()
    assert not marker.exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "native Linux PID namespace")
@pytest.mark.parametrize("release_state", ["before", "after"])
def test_native_linux_owner_sigkill_contains_project_command(tmp_path, release_state):
    status = supervisor.supervised_process_status()
    if not status.available:
        if os.environ.get("UNSLOTH_SECURE_BOUNDARY_REQUIRED") == "1":
            pytest.fail(status.reason or "the required Linux process boundary is unavailable")
        pytest.skip(status.reason or "Linux bubblewrap is unavailable")

    backend = Path(__file__).resolve().parents[1]
    root = tmp_path / "repository"
    record_dir = tmp_path / "child-records"
    ready = tmp_path / "bound.txt"
    marker = root / "must-not-run.txt"
    trigger = root / "owner-dead.txt"
    root.mkdir()
    record_dir.mkdir()
    owner_code = textwrap.dedent(
        """
        import contextlib
        import sys
        import time
        from pathlib import Path

        from core.agent_workspace import common, supervisor
        from core.agent_workspace.common import ProjectWorkspace

        root = Path(sys.argv[1]).resolve(strict=True)
        ready = Path(sys.argv[2])
        marker = Path(sys.argv[3])
        trigger = Path(sys.argv[4])
        release_state = sys.argv[5]
        metadata = root.stat(follow_symlinks=False)
        workspace = ProjectWorkspace(
            project_id="owner-crash-project",
            root=root,
            kind="managed",
            device_id=int(metadata.st_dev),
            file_id=int(metadata.st_ino),
        )

        status = supervisor.supervised_process_status()
        if not status.available:
            raise RuntimeError(status.reason or "supervisor unavailable")

        @contextlib.contextmanager
        def access(project_id):
            assert project_id == workspace.project_id
            yield workspace

        original_release = supervisor._BubblewrapLifecycle.release

        def stop_at_release(self, cancel_event):
            if release_state == "before":
                ready.write_text("bound", encoding="utf-8")
            else:
                original_release(self, cancel_event)
                ready.write_text("released", encoding="utf-8")
            while True:
                time.sleep(0.05)

        common.project_workspace_access = access
        supervisor._BubblewrapLifecycle.release = stop_at_release
        if release_state == "before":
            payload = (
                "from pathlib import Path; "
                f"Path({str(marker)!r}).write_text('released', encoding='utf-8')"
            )
        else:
            payload = (
                "import time; from pathlib import Path; "
                f"trigger=Path({str(trigger)!r}); marker=Path({str(marker)!r}); "
                "deadline=time.monotonic()+10; "
                'exec("while time.monotonic() < deadline and not trigger.exists():\\n'
                '    time.sleep(0.005)"); '
                "marker.write_text('survived', encoding='utf-8') if trigger.exists() else None"
            )
        supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", payload],
            timeout_seconds=30,
        )
        """
    )
    owner_environment = dict(os.environ)
    owner_environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(backend), owner_environment.get("PYTHONPATH", "")) if part
    )
    owner_environment["UNSLOTH_STUDIO_CHILD_RECORD"] = str(record_dir)
    owner_environment["UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE"] = "1"
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            owner_code,
            str(root),
            str(ready),
            str(marker),
            str(trigger),
            release_state,
        ],
        cwd = str(backend),
        env = owner_environment,
        stderr = subprocess.PIPE,
        stdout = subprocess.PIPE,
        text = True,
    )
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and owner.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if not ready.exists():
            stdout, stderr = owner.communicate(timeout = 2)
            pytest.fail(
                f"owner did not reach the blocked lifecycle: stdout={stdout!r} stderr={stderr!r}"
            )

        with pytest.raises(TimeoutError):
            supervisor._acquire_project_execution_fence(
                "project:owner-crash-project", None, time.monotonic() + 0.02
            )

        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout = 5)
        if release_state == "after":
            trigger.write_text("owner dead", encoding = "utf-8")
        time.sleep(0.3)
        assert not marker.exists()

        records = list(record_dir.glob("*.json"))
        assert len(records) == 1
        record = json.loads(records[0].read_text(encoding = "utf-8"))
        children = record.get("children")
        assert isinstance(children, list) and len(children) == 1
        recorded_pid = children[0]["pid"]
        recorded_pgid = children[0]["pgid"]
        assert recorded_pid >= 2
        assert recorded_pgid == recorded_pid

        reaper_code = (
            "import json; "
            "from utils.process_lifetime import reap_recorded_children; "
            "print(json.dumps(reap_recorded_children(timeout=2)))"
        )
        reaper = subprocess.run(
            [sys.executable, "-c", reaper_code],
            cwd = str(backend),
            env = owner_environment,
            capture_output = True,
            check = True,
            text = True,
            timeout = 10,
        )
        # Parent-death signaling can finish cleanup before startup recovery.
        # The reaper reports only processes it killed; either outcome must
        # leave the recorded group gone and remove its recovery record below.
        recovered_fence = supervisor._acquire_project_execution_fence(
            "project:owner-crash-project", None, time.monotonic() + 5
        )
        supervisor._release_project_execution_fence(recovered_fence)
        assert set(json.loads(reaper.stdout)) <= {recorded_pid}
        group_deadline = time.monotonic() + 5
        while time.monotonic() < group_deadline:
            try:
                os.killpg(recorded_pgid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail("the recorded blocked process group survived startup recovery")
        assert not list(record_dir.glob("*.json"))
        assert not marker.exists()
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout = 5)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from utils.process_lifetime import reap_recorded_children; "
                "reap_recorded_children(timeout=1)",
            ],
            cwd = str(backend),
            env = owner_environment,
            check = False,
            timeout = 10,
        )


def test_project_script_creation_and_cleanup_use_the_pinned_root(local_supervisor, tmp_path):
    workspace, _lease, _boundaries = local_supervisor
    from core.inference import tools

    original = workspace.root
    moved = tmp_path / "moved-root"
    outside = tmp_path / "outside"
    outside.mkdir()
    root_fd = os.open(original, os.O_RDONLY | os.O_DIRECTORY)
    boundary = type("Boundary", (), {"root": original, "_root_fd": root_fd})()
    try:
        original.rename(moved)
        original.symlink_to(outside, target_is_directory = True)
        script = supervisor._create_project_script(boundary, "print('owned')")
        assert (moved / script.name).read_text() == "print('owned')"
        assert not (outside / script.name).exists()
        assert script.path in tools._active_scratch
        script.remove()
        assert not (moved / script.name).exists()
        assert script.path not in tools._active_scratch
    finally:
        os.close(root_fd)


def test_project_script_cleanup_preserves_a_replacement_file(local_supervisor):
    workspace, _lease, _boundaries = local_supervisor
    root_fd = os.open(workspace.root, os.O_RDONLY | os.O_DIRECTORY)
    boundary = type("Boundary", (), {"root": workspace.root, "_root_fd": root_fd})()
    try:
        script = supervisor._create_project_script(boundary, "print('owned')")
        path = workspace.root / script.name
        path.unlink()
        path.write_text("replacement")
        script.remove()
        assert path.read_text() == "replacement"
    finally:
        os.close(root_fd)


@pytest.mark.parametrize("queued", [False, True])
def test_supervisor_shutdown_refuses_prepared_spawn(local_supervisor, monkeypatch, queued):
    workspace, lease_active, boundaries = local_supervisor
    shutting_down = threading.Event()
    if not queued:
        shutting_down.set()
    monkeypatch.setattr(supervisor, "is_process_shutting_down", shutting_down.is_set, raising = False)
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("shutdown must refuse the process launch"),
    )

    with pytest.raises(ProjectExecutionUnavailable, match = "shutting down"):
        supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", "pass"],
            before_start = lambda *_args: shutting_down.set(),
        )

    assert lease_active["value"] is False
    assert all(boundary.closed and not boundary.slot for boundary in boundaries)


@pytest.mark.parametrize("phase", ["adoption", "release"])
def test_supervisor_shutdown_reaps_before_releasing_workspace(local_supervisor, monkeypatch, phase):
    workspace, lease_active, boundaries = local_supervisor
    shutting_down = threading.Event()
    monkeypatch.setattr(supervisor, "is_process_shutting_down", shutting_down.is_set, raising = False)
    observed = {}
    real_popen = supervisor.subprocess.Popen
    release_calls = []
    preparation_calls = []

    def capture_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        observed["process"] = process
        return process

    def before_start(*_args):
        preparation_calls.append(True)
        if phase == "release" and len(preparation_calls) == 2:
            shutting_down.set()

    monkeypatch.setattr(supervisor.subprocess, "Popen", capture_popen)
    if phase == "adoption":
        monkeypatch.setattr(supervisor, "adopt_pid", lambda _pid: shutting_down.set())
    monkeypatch.setattr(
        _LocalLifecycle,
        "release",
        lambda *_args: release_calls.append(True) or True,
    )

    with pytest.raises(ProjectExecutionUnavailable, match = "shutting down"):
        supervisor.run_project_process(
            workspace.project_id,
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds = 0.05,
            before_start = before_start,
        )

    assert observed["process"].poll() is not None
    assert release_calls == []
    assert lease_active["value"] is False
    assert boundaries[-1].closed is True
    assert boundaries[-1].slot is False
    assert all(lifecycle.closed for lifecycle in _LocalLifecycle.instances)
