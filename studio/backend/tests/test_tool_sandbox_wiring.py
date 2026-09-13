# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two tool launches, now routed through the OS-isolation planner.

Every assertion here must hold on BOTH paths, isolated and fallback. One that
only passes when the sandbox is unavailable is a check on the machine.
"""

from __future__ import annotations

import ast
import errno
import importlib
import inspect
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import time

import pytest

from core.inference import os_sandbox, tools
from core.inference.os_sandbox import (
    PreparedSandboxLaunch,
    SandboxUnavailableError,
    ToolLaunchPlan,
)

_SESSION = "__LOCALID_sandbox_wiring"


def test_execute_tool_keeps_every_parameter_it_had_in_the_same_order():
    parameters = inspect.signature(tools.execute_tool).parameters
    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == [
        "name",
        "arguments",
        "cancel_event",
        "timeout",
        "session_id",
        "thread_id",
        "rag_scope",
        "disable_sandbox",
        "output_callback",
        "website_policy",
        "conversation_branch",
        "conversation_budget_tokens",
        "conversation_token_counter",
        "context_tokens",
        "search_images",
        "result_budget_tokens",
    ]
    assert parameters["tool_execution_mode"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["tool_execution_mode"].default == "auto"


@pytest.mark.parametrize("function", [tools._python_exec, tools._bash_exec])
def test_the_executors_take_the_mode_keyword_only_and_default_it(function):
    parameters = inspect.signature(function).parameters
    assert parameters["tool_execution_mode"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["tool_execution_mode"].default == "auto"
    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional[:5] == [
        positional[0],
        "cancel_event",
        "timeout",
        "session_id",
        "disable_sandbox",
    ]


def test_a_caller_that_passes_nothing_new_behaves_as_auto():
    tools._last_tool_execution_record = None
    assert "2" in tools._python_exec("print(1 + 1)", None, 60, _SESSION)
    assert tools._last_tool_execution_record.requested_mode == "auto"


def test_disable_sandbox_still_means_full_access():
    tools._last_tool_execution_record = None
    assert "7" in tools._python_exec("print(7)", None, 60, _SESSION, disable_sandbox = True)
    record = tools._last_tool_execution_record
    assert record.requested_mode == "full"
    assert record.effective_mode == "full"
    assert record.limitations == ("security_restrictions_disabled",)


def test_full_access_is_not_turned_into_a_refusal_by_a_stale_required():
    tools._last_tool_execution_record = None
    out = tools._python_exec(
        "print(11)", None, 60, _SESSION, disable_sandbox = True, tool_execution_mode = "required"
    )
    assert "11" in out
    assert tools._last_tool_execution_record.effective_mode == "full"


def _fallback_host() -> bool:
    return not os_sandbox.capability_snapshot().available


@pytest.mark.skipif(
    not _fallback_host(), reason = "this host can isolate, so there is no fallback to observe"
)
class TestAutoFallsBackOnAHostThatCannotIsolate:
    def test_the_capability_is_unavailable_with_an_actionable_remediation(self):
        capability = os_sandbox.capability_snapshot()
        assert capability.available is False
        assert capability.remediation
        if sys.platform == "linux" and os.path.exists(
            "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
        ):
            blocked = (
                subprocess.run(
                    ["unshare", "--user", "--map-root-user", "true"],
                    stdin = subprocess.DEVNULL,
                    stdout = subprocess.DEVNULL,
                    stderr = subprocess.DEVNULL,
                ).returncode
                != 0
            )
            if blocked:
                assert "apparmor_restrict_unprivileged_userns" in capability.remediation
                assert "bwrap-userns-restrict" in capability.remediation

    @pytest.mark.parametrize(
        "run,expected",
        [
            (lambda: tools._python_exec("print(6 * 7)", None, 60, _SESSION), "42"),
            (lambda: tools._bash_exec("echo 42", None, 60, _SESSION), "42"),
        ],
        ids = ["python", "terminal"],
    )
    def test_auto_refuses_nothing(self, run, expected):
        tools._last_tool_execution_record = None
        assert expected in run()
        record = tools._last_tool_execution_record
        assert record.requested_mode == "auto"
        assert record.effective_mode == "software_safeguards"
        assert record.os_isolation is False
        assert "process_guard" in record.retained_safeguards
        assert "no_os_isolation" in record.limitations
        assert record.network_policy == "unrestricted"

    def test_required_refuses_and_runs_nothing(self):
        out = tools._python_exec(
            "print('SHOULD_NOT_RUN')", None, 60, _SESSION, tool_execution_mode = "required"
        )
        assert "SHOULD_NOT_RUN" not in out
        assert "OS_ISOLATION_UNAVAILABLE" in out
        assert os_sandbox.capability_snapshot().remediation.split(".")[0] in out

    def test_required_raises_out_of_the_planner_itself(self):
        plan = ToolLaunchPlan(
            argv = (sys.executable, "-c", "pass"),
            workdir = os.getcwd(),
            env = {},
            requested_mode = "required",
        )
        with pytest.raises(SandboxUnavailableError) as excinfo:
            os_sandbox.prepare_tool_launch(plan)
        assert excinfo.value.remediation

    def test_terminal_required_refuses_too(self):
        out = tools._bash_exec(
            "echo SHOULD_NOT_RUN", None, 60, _SESSION, tool_execution_mode = "required"
        )
        assert "SHOULD_NOT_RUN" not in out
        assert "OS_ISOLATION_UNAVAILABLE" in out


def test_an_unknown_mode_is_reported_rather_than_silently_downgraded():
    out = tools._python_exec(
        "print('SHOULD_NOT_RUN')", None, 60, _SESSION, tool_execution_mode = "nonsense"
    )
    assert "SHOULD_NOT_RUN" not in out
    assert "nonsense" in out


@pytest.mark.skipif(
    sys.platform == "win32", reason = "pre-exec and pass_fds are POSIX; Windows keeps today's path"
)
def test_the_process_unsloth_holds_still_lands_in_its_own_session():
    """Asserted about the OUTER process: under bubblewrap the payload is not a
    session leader, so asking it about its own sid only passes on a fallback."""
    # One call OUTSIDE the window first. Everything this platform initialises
    # lazily then happens before anything is counted: the capability probe spawns
    # a real launch, and on macOS _developer_paths() shells out to xcode-select.
    # Both are one-time, and counting them made this fail 4 == 2 on macos-14
    # while passing on Linux. What the assertion is for is per-CALL behaviour, so
    # a warm-up is what separates the two: a genuine per-call spawn survives it,
    # which is how the extra fork this test caught before was found.
    # Both kinds, since the terminal path initialises its own shell lookup and a
    # python-only warm-up left that inside the window.
    tools._python_exec("pass", None, 60, _SESSION)
    tools._bash_exec("true", None, 60, _SESSION)

    seen = []
    real = subprocess.Popen

    def capture(argv, **kwargs):
        seen.append((argv, kwargs.get("preexec_fn")))
        return real(argv, **kwargs)

    subprocess.Popen = capture
    try:
        assert "5" in tools._python_exec("print(2 + 3)", None, 60, _SESSION)
        assert "6" in tools._bash_exec("echo 6", None, 60, _SESSION)
    finally:
        subprocess.Popen = real
    # Tool launches carry a pre-exec; the bookkeeping spawns do not. macOS adds a
    # `ps` liveness check per call, which no warm-up removes because it is not a
    # one-time cost, and counting it made this fail 4 == 2 there while passing on
    # Linux. Both halves are asserted, so an extra TOOL launch still fails the
    # count and an extra bookkeeping spawn has to be a known one.
    launches = [preexec for _, preexec in seen if preexec is not None]
    bookkeeping = [argv for argv, preexec in seen if preexec is None]
    # The argv is in the message because a count alone cannot say WHICH extra
    # spawn appeared, and this only ever fails on a runner nobody can attach to.
    assert len(launches) == 2, [argv for argv, _ in seen]
    assert all(tuple(argv)[:1] == ("ps",) for argv in bookkeeping), bookkeeping
    seen = launches
    # Asked by result, not identity: an isolated launch composes the plan's
    # pre-exec with the backend's, so the object differs either way.
    for preexec in seen:
        read_fd, write_fd = os.pipe()
        child = os.fork()
        if child == 0:  # pragma: no cover - runs in the forked child
            try:
                os.close(read_fd)
                preexec()
                os.write(write_fd, b"1" if os.getsid(0) == os.getpid() else b"0")
            finally:
                os._exit(0)
        os.close(write_fd)
        with os.fdopen(read_fd, "rb") as stream:
            leads_its_session = stream.read()
        os.waitpid(child, 0)
        assert leads_its_session == b"1"


def test_a_timeout_kills_the_tool_and_leaves_the_server_running():
    start = time.monotonic()
    out = tools._python_exec("import time; time.sleep(120)", None, 5, _SESSION)
    assert "timed out" in out.lower(), out
    assert time.monotonic() - start < 60
    assert "9" in tools._python_exec("print(4 + 5)", None, 60, _SESSION)


@pytest.mark.skipif(
    sys.platform == "win32", reason = "pre-exec and pass_fds are POSIX; Windows keeps today's path"
)
def test_the_plan_carries_the_pre_exec_the_kill_paths_depend_on():
    seen = []
    real = os_sandbox.prepare_tool_launch

    def capture(plan):
        seen.append(plan)
        return real(plan)

    os_sandbox.prepare_tool_launch = capture
    try:
        tools._python_exec("print(1)", None, 60, _SESSION)
        tools._bash_exec("echo 1", None, 60, _SESSION)
        tools._python_exec("print(1)", None, 60, _SESSION, disable_sandbox = True)
    finally:
        os_sandbox.prepare_tool_launch = real
    assert [plan.preexec_fn for plan in seen] == [
        tools._sandbox_preexec,
        tools._sandbox_preexec,
        tools._bypass_preexec,
    ]
    assert [plan.execution_kind for plan in seen] == ["python", "terminal", "python"]


def test_sandbox_preexec_runs_no_imports_after_the_fork():
    for function in (tools._sandbox_preexec, tools._bypass_preexec):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        offenders = [
            node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert not offenders, f"{function.__name__} imports after the fork"


@pytest.mark.parametrize("disable_sandbox", [False, True], ids = ["sandboxed", "full"])
def test_the_environment_the_shim_depends_on_survives(disable_sandbox):
    workdir = tools._get_workdir(_SESSION)
    out = tools._python_exec(
        "import os\n"
        "print('PYTHONPATH', os.environ.get('PYTHONPATH'))\n"
        "print('HOME', os.environ.get('HOME'))\n"
        "print('TMPDIR', os.environ.get('TMPDIR'))\n",
        None,
        60,
        _SESSION,
        disable_sandbox = disable_sandbox,
    )
    lines = dict(line.split(" ", 1) for line in out.splitlines() if " " in line)
    assert tools._SANDBOX_SITE_DIR in lines["PYTHONPATH"]
    assert lines["HOME"] == workdir
    assert lines["TMPDIR"].startswith(workdir)


def test_the_path_remap_shim_still_heals_an_invented_absolute_path():
    result = tools._python_exec(
        "open('/mnt/data/wiring_probe.csv', 'w').write('a,b\\n')", None, 60, _SESSION
    )
    assert "Error" not in result
    assert os.path.exists(os.path.join(tools._get_workdir(_SESSION), "wiring_probe.csv"))


class _Recorder:
    def __init__(self):
        self.cleaned = 0
        self.spawn_kwargs = None


def _echoing_prepare(recorder, **overrides):
    def prepare(plan):
        prepared = PreparedSandboxLaunch(
            argv = plan.argv,
            workdir = plan.workdir,
            env = plan.env,
            preexec_fn = plan.preexec_fn,
            backend = "test-double",
            **overrides,
        )
        prepared.cleanup_callbacks.append(
            lambda: setattr(recorder, "cleaned", recorder.cleaned + 1)
        )
        return prepared

    return prepare


@pytest.mark.parametrize(
    "run",
    [
        lambda: tools._python_exec("print(1)", None, 60, _SESSION),
        lambda: tools._bash_exec("echo 1", None, 60, _SESSION),
    ],
    ids = ["python", "terminal"],
)
def test_the_launch_is_released_when_the_process_finishes(monkeypatch, run):
    recorder = _Recorder()
    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", _echoing_prepare(recorder))
    run()
    assert recorder.cleaned == 1


def test_the_launch_is_released_when_the_spawn_raises(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", _echoing_prepare(recorder))

    def explode(prepared, **kwargs):
        raise OSError("no fork for you")

    monkeypatch.setattr(os_sandbox, "spawn_prepared_launch", explode)
    assert "no fork for you" in tools._python_exec("print(1)", None, 60, _SESSION)
    assert recorder.cleaned == 1


def test_the_launch_is_released_when_required_refuses(monkeypatch):
    def refuse(plan):
        raise SandboxUnavailableError("OS_ISOLATION_UNAVAILABLE: nope", remediation = "install it")

    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", refuse)
    out = tools._python_exec("print(1)", None, 60, _SESSION, tool_execution_mode = "required")
    assert "install it" in out


@pytest.mark.skipif(
    sys.platform == "win32", reason = "pre-exec and pass_fds are POSIX; Windows keeps today's path"
)
def test_pass_fds_and_owned_files_reach_the_spawn(monkeypatch):
    read_fd, write_fd = os.pipe()
    holder = os.fdopen(write_fd, "wb")
    recorder = _Recorder()
    monkeypatch.setattr(
        os_sandbox,
        "prepare_tool_launch",
        _echoing_prepare(recorder, pass_fds = (read_fd,)),
    )

    real_spawn = os_sandbox.spawn_prepared_launch

    def capture(prepared, **kwargs):
        prepared.owned_files.append(holder)
        recorder.spawn_kwargs = dict(kwargs)
        return real_spawn(prepared, **kwargs)

    monkeypatch.setattr(os_sandbox, "spawn_prepared_launch", capture)
    try:
        out = tools._python_exec(
            f"import os; os.fstat({read_fd}); print('INHERITED')", None, 60, _SESSION
        )
    finally:
        os.close(read_fd)
    assert "INHERITED" in out, out
    assert recorder.spawn_kwargs["pass_fds"] == (read_fd,)
    assert recorder.spawn_kwargs["cwd"] == tools._get_workdir(_SESSION)
    assert recorder.spawn_kwargs["close_fds"] is True
    assert recorder.cleaned == 1
    assert holder.closed


def test_auto_still_runs_when_the_planner_itself_breaks(monkeypatch):
    def explode(plan):
        raise ImportError("no module named sandbox_linux")

    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", explode)
    tools._last_tool_execution_record = None
    assert "5" in tools._python_exec("print(5)", None, 60, _SESSION)
    record = tools._last_tool_execution_record
    assert record.effective_mode == "software_safeguards"
    assert "sandbox_planner_error" in record.limitations


def test_full_access_keeps_its_own_label_even_when_the_planner_breaks(monkeypatch):
    def explode(plan):
        raise RuntimeError("planner down")

    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", explode)
    tools._last_tool_execution_record = None
    assert "9" in tools._python_exec("print(9)", None, 60, _SESSION, disable_sandbox = True)
    record = tools._last_tool_execution_record
    assert record.effective_mode == "full"
    assert "security_restrictions_disabled" in record.limitations
    assert "command_and_code_analysis" not in record.retained_safeguards


@pytest.mark.skipif(
    sys.platform == "win32", reason = "pre-exec and pass_fds are POSIX; Windows keeps today's path"
)
def test_a_backend_that_drops_the_pre_exec_has_it_put_back(monkeypatch):
    def forgetful(plan):
        return PreparedSandboxLaunch(
            argv = plan.argv,
            workdir = plan.workdir,
            env = plan.env,
            preexec_fn = None,  # the bug
            backend = "forgetful",
        )

    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", forgetful)
    out = tools._python_exec(
        "import os; print('SID_MATCHES', os.getsid(0) == os.getpid())", None, 60, _SESSION
    )
    assert "SID_MATCHES True" in out, out


def test_required_still_refuses_when_the_planner_itself_breaks(monkeypatch):
    def explode(plan):
        raise ImportError("no module named sandbox_linux")

    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", explode)
    out = tools._python_exec(
        "print('SHOULD_NOT_RUN')", None, 60, _SESSION, tool_execution_mode = "required"
    )
    assert "SHOULD_NOT_RUN" not in out
    assert "OS_ISOLATION_UNAVAILABLE" in out


@pytest.mark.parametrize("platform", ["win32", "cygwin", "aix"])
def test_a_platform_with_no_backend_gets_exactly_the_plan_it_handed_in(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)

    def marker():
        return None

    env = {"PATH": "/usr/bin"}
    plan = ToolLaunchPlan(
        argv = ("prog", "arg"),
        workdir = "/work",
        env = env,
        preexec_fn = marker,
        requested_mode = "auto",
        timeout_seconds = 300,
    )
    prepared = os_sandbox.prepare_tool_launch(plan)
    assert prepared.argv == plan.argv
    assert prepared.env is env
    assert prepared.preexec_fn is marker
    assert prepared.workdir == "/work"
    assert prepared.pass_fds == ()
    assert prepared.close_fds is True
    assert prepared.execution_record.effective_mode == "software_safeguards"
    assert prepared.execution_record.os_isolation is False


def test_a_platform_with_no_backend_still_refuses_in_required(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(SandboxUnavailableError):
        os_sandbox.prepare_tool_launch(
            ToolLaunchPlan(argv = ("prog",), workdir = "/work", env = {}, requested_mode = "required")
        )


def test_the_tool_descriptions_are_untouched_by_this_change():
    """A sentence about OS isolation here would be decided at import time, putting a
    bwrap launch in the import path of tools.py, and would need a matching
    _FULL_ACCESS_SUBSTITUTIONS entry or Full access advertises a sandbox it
    disabled."""
    note = tools._build_sandbox_paths_note()
    assert "isolation" not in note.lower()
    full = tools._to_full_access(
        "Execute Python code in a sandbox and return stdout/stderr." + note, "python"
    )
    assert "in a sandbox" not in full
    assert "sandbox is disabled" in full


# A backend can decline a specific launch long after the probe said yes, and an
# ML project workdir crosses the 50,000-entry limit as a matter of course. In
# auto that must be a fallback, not the end of the session.


def _declining_backend(
    monkeypatch,
    reason: str,
    unsafe: bool = True,
) -> None:
    """*unsafe* picks WHICH refusal; the two are told apart by type."""
    error = os_sandbox.WorkdirUnsafeError if unsafe else SandboxUnavailableError

    def decline(plan):
        raise error(reason)

    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", decline)


def test_an_unsafe_workdir_fails_the_call_rather_than_de_isolating_it(monkeypatch):
    _declining_backend(monkeypatch, "the session workdir contains a device node")
    tools._last_tool_execution_record = None
    out = tools._python_exec("print('SHOULD_NOT_RUN')", None, 60, _SESSION)
    assert "SHOULD_NOT_RUN" not in out
    assert "device node" in out
    assert tools._last_tool_execution_record is None


def test_required_still_refuses_when_the_backend_declines_this_launch(monkeypatch):
    _declining_backend(monkeypatch, "the session workdir contains a device node")
    out = tools._python_exec(
        "print('SHOULD_NOT_RUN')", None, 60, _SESSION, tool_execution_mode = "required"
    )
    assert "SHOULD_NOT_RUN" not in out
    assert "device node" in out


@pytest.mark.skipif(sys.platform == "win32", reason = "the workdir scan is POSIX only")
def test_tool_code_cannot_switch_the_boundary_off_for_the_next_call(monkeypatch):
    if not os_sandbox.capability_snapshot().available:
        pytest.skip("this host cannot isolate, so there is no boundary to switch off")
    workdir = tools._get_workdir(_SESSION)
    planted = os.path.join(workdir, "planted.sock")
    holder = socket.socket(socket.AF_UNIX)
    # Bound RELATIVE: an AF_UNIX address is capped at ~108 bytes, and under
    # `pytest -n 4` the studio home is a per-worker tmp_path that alone exceeds
    # it, so the absolute spelling raised "AF_UNIX path too long" instead of
    # planting anything. Backend CI runs -n 4.
    monkeypatch.chdir(workdir)
    holder.bind("planted.sock")
    try:
        tools._last_tool_execution_record = None
        out = tools._python_exec("print('SHOULD_NOT_RUN')", None, 60, _SESSION)
        assert "SHOULD_NOT_RUN" not in out
        assert "device or IPC node" in out
        assert tools._last_tool_execution_record is None
    finally:
        holder.close()
        os.unlink(planted)
    assert "42" in tools._python_exec("print(6 * 7)", None, 60, _SESSION)
    assert tools._last_tool_execution_record.os_isolation is True


@pytest.mark.parametrize(
    "function,payload",
    [(tools._python_exec, "print('SHOULD_NOT_RUN')"), (tools._bash_exec, "echo SHOULD_NOT_RUN")],
    ids = ["python", "terminal"],
)
def test_full_is_not_requestable_through_the_mode(function, payload):
    tools._last_tool_execution_record = None
    out = function(payload, None, 60, _SESSION, tool_execution_mode = "full")
    assert "SHOULD_NOT_RUN" not in out
    assert "TOOL_EXECUTION_MODE_INVALID" in out
    assert "disable_sandbox" in out
    assert tools._last_tool_execution_record is None


def test_disable_sandbox_is_still_the_way_to_full_access():
    tools._last_tool_execution_record = None
    assert "7" in tools._python_exec(
        "print(7)", None, 60, _SESSION, disable_sandbox = True, tool_execution_mode = "full"
    )
    assert tools._last_tool_execution_record.effective_mode == "full"


@pytest.mark.parametrize(
    "run",
    [
        lambda: tools._python_exec(
            "import sys; print('IN', repr(sys.stdin.read()))", None, 60, _SESSION
        ),
        lambda: tools._bash_exec("printf 'IN %s\\n' \"$(cat)\"", None, 60, _SESSION),
    ],
    ids = ["python", "terminal"],
)
def test_a_tool_call_cannot_read_the_servers_stdin(run):
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"OPERATOR SECRET\n")
    os.close(write_fd)
    saved = os.dup(0)
    try:
        os.dup2(read_fd, 0)
        out = run()
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        os.close(read_fd)
    assert "OPERATOR SECRET" not in out
    assert "IN" in out


def test_the_fallback_never_claims_a_descendant_sweep_it_does_not_perform():
    limitations = os_sandbox._software_only_limitations()
    if sys.platform == "win32":
        assert "detached_descendant_cleanup_unverified" not in limitations
    else:
        assert "detached_descendant_cleanup_unverified" in limitations
    assert not hasattr(os_sandbox, "descendant_sweep_supported")


def test_a_backend_that_has_just_stopped_being_available_still_falls_back(monkeypatch):
    _declining_backend(
        monkeypatch, "bubblewrap (bwrap) is not installed on this host", unsafe = False
    )
    tools._last_tool_execution_record = None
    assert "42" in tools._python_exec("print(6 * 7)", None, 60, _SESSION)
    record = tools._last_tool_execution_record
    assert record.os_isolation is False
    assert "sandbox_became_unavailable" in record.limitations


def test_a_backend_that_fails_at_launch_drops_the_cached_verdict(monkeypatch):
    reset = []
    monkeypatch.setattr(
        "core.inference.sandbox_probe.reset_probe_cache", lambda: reset.append(True)
    )
    prepared = PreparedSandboxLaunch(
        argv = ("bwrap",), workdir = "/work", env = {}, preexec_fn = None, backend = "bubblewrap"
    )
    tools._forget_sandbox_capability_if_the_backend_failed(
        prepared, "Exit code 1:\nbwrap: setting up uid map: Permission denied\n"
    )
    assert reset == [True]

    tools._forget_sandbox_capability_if_the_backend_failed(
        prepared, "Exit code 1:\nTraceback (most recent call last):\n"
    )
    prepared.backend = "software-safeguards"
    tools._forget_sandbox_capability_if_the_backend_failed(
        prepared, "Exit code 1:\nbwrap: setting up uid map: Permission denied\n"
    )
    assert reset == [True]


def test_a_planner_os_error_refuses_rather_than_running_unisolated(monkeypatch):
    backend = importlib.import_module(
        "core.inference.sandbox_linux"
        if sys.platform == "linux"
        else "core.inference.sandbox_macos"
    )

    def full_disk(plan):
        raise OSError(errno.ENOSPC, "No space left on device")

    # Through the real prepare_tool_launch, because the wrap that types this
    # lives in it.
    monkeypatch.setattr(
        os_sandbox,
        "capability_snapshot",
        lambda **kwargs: os_sandbox.SandboxCapability(
            backend = backend.BACKEND_NAME,
            available = True,
            reason = "probe passed",
            profile_id = backend.PROFILE_ID,
        ),
    )
    monkeypatch.setattr(backend, "prepare", full_disk)
    tools._last_tool_execution_record = None
    out = tools._python_exec("print('SHOULD_NOT_RUN')", None, 60, _SESSION)
    assert "SHOULD_NOT_RUN" not in out
    assert tools._last_tool_execution_record is None


def test_an_unisolated_launch_cannot_be_hooked_by_a_planted_usercustomize(tmp_path):
    """site imports `usercustomize` from sys.path at interpreter startup whenever
    ENABLE_USER_SITE is on, which it is for any non-venv interpreter. The session
    packages directory is writable by the tool call and goes on PYTHONPATH, so
    without PYTHONNOUSERSITE a call could leave a payload that runs on the host at
    the start of every later unisolated call, ahead of that call's own analysed
    script. Measured on a system python3 before this was set."""
    workdir = tmp_path / "session"
    (workdir / os_sandbox.SESSION_PACKAGES_RELPATH).mkdir(parents = True)
    env = tools._with_session_packages({"PATH": "/usr/bin"}, str(workdir))
    assert env["PYTHONNOUSERSITE"] == "1"


def test_the_shipped_sitecustomize_is_found_before_the_session_packages(tmp_path):
    """The other half, and the reason no separate guard is needed for it: site
    always imports `sitecustomize`, and PYTHONNOUSERSITE does not stop that. What
    stops a planted one is ordering, so the ordering is pinned here. The shim
    directory _build_safe_env sets must stay AHEAD of the writable directory."""
    workdir = tmp_path / "session"
    (workdir / os_sandbox.SESSION_PACKAGES_RELPATH).mkdir(parents = True)
    env = tools._with_session_packages({"PYTHONPATH": tools._SANDBOX_SITE_DIR}, str(workdir))
    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries.index(tools._SANDBOX_SITE_DIR) < entries.index(
        str(workdir / os_sandbox.SESSION_PACKAGES_RELPATH)
    )
