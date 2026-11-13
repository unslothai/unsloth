# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two tool launches, now routed through the OS-isolation planner."""

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
import sysconfig
import textwrap
import time
import zipfile

import pytest

from core.inference import os_sandbox, tools
from core.inference.os_sandbox import (
    PreparedSandboxLaunch,
    SandboxUnavailableError,
    ToolLaunchPlan,
)

_SESSION = "__LOCALID_sandbox_wiring"


class _PassthroughAccountConfinement:
    preexec = None
    # tools.py only takes the managed path when `confines` is true.
    confines = True

    @staticmethod
    def wrap(argv):
        return argv


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


@pytest.mark.parametrize(
    "function,payload",
    [
        (tools._python_exec, "print('MANAGED_OK')"),
        (tools._bash_exec, "echo MANAGED_OK"),
    ],
    ids = ["python", "terminal"],
)
def test_managed_account_launch_does_not_require_a_generic_sandbox_record(
    monkeypatch, function, payload
):
    monkeypatch.setattr(tools, "_account_confinement", lambda: _PassthroughAccountConfinement())
    tools._last_tool_execution_record = None

    assert "MANAGED_OK" in function(payload, None, 60, _SESSION)
    assert tools._last_tool_execution_record is None


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
    assert record.execution_status == "completed"
    assert record.completion_status == "finished"
    assert record.cleanup_status == "complete"


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
    """Asserted about the OUTER process: under bubblewrap the payload is not a session leader, so asking it about its own sid only passes on a fallback."""
    tools._python_exec("pass", None, 60, _SESSION)
    tools._bash_exec("true", None, 60, _SESSION)

    seen = []
    real = subprocess.Popen

    def capture(argv, **kwargs):
        seen.append((argv, kwargs.get("preexec_fn"), kwargs.get("start_new_session", False)))
        return real(argv, **kwargs)

    subprocess.Popen = capture
    try:
        assert "5" in tools._python_exec("print(2 + 3)", None, 60, _SESSION)
        assert "6" in tools._bash_exec("echo 6", None, 60, _SESSION)
    finally:
        subprocess.Popen = real
    launches = [
        (argv, preexec) for argv, preexec, new_session in seen if preexec is not None or new_session
    ]
    bookkeeping = [
        argv for argv, preexec, new_session in seen if preexec is None and not new_session
    ]
    assert len(launches) == 2, [entry[0] for entry in seen]
    assert all(tuple(argv)[:1] == ("ps",) for argv in bookkeeping), bookkeeping
    seen = []
    for argv, preexec in launches:
        if preexec is None:
            # No Python pre-exec: Popen runs setsid itself, then setpriv and prlimit exec in place.
            assert os.path.basename(argv[0]) == "setpriv", argv
            continue
        seen.append(preexec)
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


def test_auto_refuses_when_the_planner_itself_breaks(monkeypatch):
    def explode(plan):
        raise ImportError("no module named sandbox_linux")

    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", explode)
    out = tools._python_exec("print('SHOULD_NOT_RUN')", None, 60, _SESSION)
    assert "SHOULD_NOT_RUN" not in out
    assert "without host fallback" in out


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
            preexec_fn = None,
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
    assert "without host fallback" in out


@pytest.mark.parametrize("platform", ["cygwin", "aix"])
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
    monkeypatch.setattr(sys, "platform", "aix")
    with pytest.raises(SandboxUnavailableError):
        os_sandbox.prepare_tool_launch(
            ToolLaunchPlan(argv = ("prog",), workdir = "/work", env = {}, requested_mode = "required")
        )


def test_the_tool_descriptions_are_untouched_by_this_change():
    note = tools._build_sandbox_paths_note()
    assert "isolation" not in note.lower()
    full = tools._to_full_access(
        "Execute Python code in a sandbox and return stdout/stderr." + note, "python"
    )
    assert "in a sandbox" not in full
    assert "sandbox is disabled" in full


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
        "core.inference.sandbox_windows_mxc"
        if sys.platform == "win32"
        else (
            "core.inference.sandbox_linux"
            if sys.platform == "linux"
            else "core.inference.sandbox_macos"
        )
    )

    def full_disk(plan, *_args):
        raise OSError(errno.ENOSPC, "No space left on device")

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
    workdir = tmp_path / "session"
    (workdir / os_sandbox.SESSION_PACKAGES_RELPATH).mkdir(parents = True)
    env = tools._with_session_packages({"PATH": "/usr/bin"}, str(workdir))
    assert env["PYTHONNOUSERSITE"] == "1"


def test_the_shipped_sitecustomize_is_found_before_the_session_packages(tmp_path):
    workdir = tmp_path / "session"
    (workdir / os_sandbox.SESSION_PACKAGES_RELPATH).mkdir(parents = True)
    env = tools._with_session_packages({"PYTHONPATH": tools._SANDBOX_SITE_DIR}, str(workdir))
    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries.index(tools._SANDBOX_SITE_DIR) < entries.index(
        str(workdir / os_sandbox.SESSION_PACKAGES_RELPATH)
    )


@pytest.mark.parametrize(("os_name", "scripts"), [("nt", "Scripts"), ("posix", "bin")])
def test_session_package_commands_stay_on_path(monkeypatch, tmp_path, os_name, scripts):
    """pip --target writes console scripts to Scripts on Windows, so bin alone lost them there."""
    workdir = tmp_path / "session"
    packages = workdir / os_sandbox.SESSION_PACKAGES_RELPATH
    packages.mkdir(parents = True)
    monkeypatch.setattr(os_sandbox.os, "name", os_name)
    monkeypatch.setattr(os_sandbox, "_PIP_PREFIX_SCRIPTS_DIR", scripts)
    env = os_sandbox.with_session_packages({"PATH": "/usr/bin"}, str(workdir))
    assert env["PATH"].split(os.pathsep) == ["/usr/bin", str(packages / scripts)]


def _wheel(workdir, name, files):
    info = f"{name}-1.0.dist-info"
    wheel = workdir / f"{name}-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as whl:
        for path, text in files.items():
            whl.writestr(path, text)
        whl.writestr(f"{info}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n")
        whl.writestr(
            f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        whl.writestr(f"{info}/RECORD", "")
    return wheel.name


_PIP = "python -m pip install -q --no-index --no-deps"


@pytest.mark.parametrize(
    "case",
    [
        "chat",
        "same_process",
        "editable",
        "project_venv",
        "target",
        "env_target",
        "pip_user_off",
        "isolated",
    ],
)
@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_a_pip_install_without_os_isolation_stays_out_of_studio(monkeypatch, tmp_path, case):
    """Studio's interpreter installs into the chat folder; another venv's pip still installs into that venv."""
    name = f"unsloth_chat_only_{case}"
    workdir = tmp_path / "chat"
    workdir.mkdir()
    packages = str(workdir / os_sandbox.SESSION_PACKAGES_RELPATH)
    files = {f"{name}.py": ""}
    if case == "editable":
        (workdir / "src").mkdir()
        (workdir / "src" / f"{name}.py").write_text("")
        files = {f"__editable__.{name}-1.0.pth": str(workdir / "src") + "\n"}
    wheel = _wheel(workdir, name, files)
    show = f"python -c 'import {name}; print({name}.__file__)'"
    command = {
        "chat": f"{_PIP} {wheel} && {show}",
        # An explicit --target keeps its own location.
        "target": f"{_PIP} --target vendor {wheel} && PYTHONPATH=vendor {show}",
        "env_target": f"PIP_TARGET=vendor {_PIP} {wheel} && PYTHONPATH=vendor {show}",
        "pip_user_off": f"PIP_USER=0 {_PIP} {wheel} && {show}",
        "isolated": f"python -m pip --isolated install -q --no-index --no-deps {wheel} && {show}",
        "editable": f"{_PIP} {wheel} && {show}",
        # Python caches the session dirs as missing at startup, before pip creates them.
        "same_process": 'python -c "import subprocess, sys; subprocess.check_call(sys.argv[1:]); '
        f'import {name}; print({name}.__file__)" python -m pip install -q --no-index --no-deps {wheel}',
        # The chat's copy must not satisfy the venv's pip.
        "project_venv": f"{_PIP} {wheel} && python -m venv venv && venv/bin/{_PIP} {wheel} && "
        f"env -u PYTHONPATH venv/bin/{show}",
    }[case]
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    try:
        out = tools._bash_exec(
            command,
            None,
            120,
            _SESSION,
            tool_execution_mode = "software",
        )
    finally:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "-q", name], capture_output = True
        )
    assert tools._last_tool_execution_record.os_isolation is False
    expected = {
        "chat": packages,
        "target": str(workdir / "vendor"),
        "env_target": str(workdir / "vendor"),
        "pip_user_off": packages,
        "isolated": packages,
        "same_process": packages,
        "editable": str(workdir / "src"),
        "project_venv": str(workdir / "venv"),
    }[case]
    assert f"{expected}{os.sep}" in out, out


@pytest.mark.parametrize(
    "case",
    [
        "editable_fork",
        "editable_finder",
        "editable_finder_module",
        "editable_finder_namespace",
        "editable_finder_virtual",
        "upgrade",
        "namespace",
        "pkgutil_namespace",
    ],
)
@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_a_chat_install_wins_over_what_studio_already_has(monkeypatch, tmp_path, case):
    """Editable forks, an upgrade imported in the installing process, and a new namespace portion all load."""
    template = os.path.join(
        sysconfig.get_paths()["purelib"], "setuptools", "command", "editable_wheel.py"
    )
    if case.startswith("editable_finder") and not os.path.exists(template):
        pytest.skip("finder-based editable installs need setuptools in Studio's environment")
    name = f"unsloth_chat_shadow_{case}"
    workdir = tmp_path / "chat"
    workdir.mkdir()
    packages = str(workdir / os_sandbox.SESSION_PACKAGES_RELPATH)
    studio_files = {f"{name}.py": ""}
    show = f"import {name}; print({name}.__file__)"
    # Imported in the process that installs: the session dirs did not exist when it started.
    in_process = (
        'python -c "import subprocess, sys; {pre}subprocess.check_call(sys.argv[1:]); {post}" '
        + _PIP
    )
    if case == "editable_fork":
        (workdir / "src").mkdir()
        (workdir / "src" / f"{name}.py").write_text("")
        wheel = _wheel(workdir, name, {f"__editable__.{name}-1.0.pth": str(workdir / "src") + "\n"})
        command, expected = f'{_PIP} {wheel} && python -c "{show}"', str(workdir / "src")
    elif case.startswith("editable_finder"):
        # setuptools' editable wheel: an import-only .pth and a finder module with literal MAPPING / NAMESPACES.
        proj = workdir / "proj"
        proj.mkdir()
        finder = f"__editable___{name}_1_0_finder"
        mapping, namespaces = {name: str(proj / "implementation")}, {}
        if case == "editable_finder":
            # package_dir remap: the source folder is not named after the package.
            (proj / "implementation").mkdir()
            (proj / "implementation" / "__init__.py").write_text("")
            expected = str(proj / "implementation")
        elif case == "editable_finder_module":
            # py_modules: MAPPING holds the module path without its suffix.
            (proj / "implementation.py").write_text("")
            expected = str(proj)
        elif case == "editable_finder_namespace":
            # A PEP 420 namespace root, with Studio holding another portion of it.
            (proj / "implementation").mkdir()
            (proj / "implementation" / "part.py").write_text("")
            namespaces = {name: [str(proj / "implementation")]}
            studio_files = {f"{name}/a.py": ""}
            show = (
                f"import importlib, {name}.a; importlib.invalidate_caches(); "
                f"import {name}.part; print({name}.part.__file__)"
            )
            expected = str(proj / "implementation")
        else:
            # Only name.part is mapped; name is a virtual namespace with no folder of its own.
            (proj / "part").mkdir()
            (proj / "part" / "__init__.py").write_text("")
            mapping, namespaces = {f"{name}.part": str(proj / "part")}, {name: []}
            studio_files = {f"{name}_unrelated.py": ""}
            show = f"import {name}.part; print({name}.part.__file__)"
            expected = str(proj / "part")
        wheel = _wheel(
            workdir,
            name,
            {
                f"__editable__.{name}-1.0.pth": f"import {finder}; {finder}.install()\n",
                f"{finder}.py": f"MAPPING: dict = {mapping!r}\nNAMESPACES: dict = {namespaces!r}\n"
                "def install():\n    raise SystemExit(9)\n",
            },
        )
        command = f'{_PIP} {wheel} && python -c "{show}"'
    elif case == "upgrade":
        wheel = _wheel(workdir, name, {f"{name}.py": ""})
        command, expected = in_process.format(pre = "", post = show) + f" {wheel}", packages
    else:
        init = "from pkgutil import extend_path; __path__ = extend_path(__path__, __name__)\n"
        studio_files = {f"{name}/a.py": ""}
        if case == "pkgutil_namespace":
            studio_files[f"{name}/__init__.py"] = init
        chat_files = {f"{name}/b.py": ""}
        if case == "pkgutil_namespace":
            chat_files[f"{name}/__init__.py"] = init
        wheel = _wheel(workdir, f"{name}_b", chat_files)
        post = (
            f"import pkgutil, {name}.b; print({name}.b.__file__); "
            f"print(sorted(m.name for m in pkgutil.iter_modules({name}.__path__)))"
        )
        command = in_process.format(pre = f"import {name}.a; ", post = post) + f" {wheel}"
        expected = packages
    studio_dir = tmp_path / "studio"
    studio_dir.mkdir()
    studio_wheel = studio_dir / _wheel(studio_dir, name, studio_files)
    subprocess.run([sys.executable, *_PIP.split()[1:], str(studio_wheel)], check = True)
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    try:
        out = tools._bash_exec(command, None, 120, _SESSION, tool_execution_mode = "software")
    finally:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "-q", name, f"{name}_b"],
            capture_output = True,
        )
    assert f"{expected}{os.sep}" in out, out
    if case in ("namespace", "pkgutil_namespace"):
        assert "['a', 'b']" in out, out


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_an_editable_finder_never_loads_a_chat_planted_stdlib_module(monkeypatch, tmp_path):
    """Building an editable finder imports its helpers from the stdlib, not from the tool-writable chat folder."""
    workdir = tmp_path / "chat"
    workdir.mkdir()
    marker = tmp_path / "planted_ran"
    site_dir = workdir / os_sandbox.SESSION_PACKAGES_RELPATH / os_sandbox._PIP_PREFIX_LIB_DIRS[0]
    site_dir.mkdir(parents = True)
    finder = "__editable___planted_1_0_finder"
    (site_dir / f"__editable__.planted-1.0.pth").write_text(
        f"import {finder}; {finder}.install()\n"
    )
    (site_dir / f"{finder}.py").write_text("MAPPING = {}\nNAMESPACES = {}\n")
    # What reading the finder data imports; types and pathlib are shadowable on main already.
    for helper in ("sysconfig", "ast"):
        (site_dir / f"{helper}.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    out = tools._bash_exec(
        "python -c 'print(41 + 1)'", None, 60, _SESSION, tool_execution_mode = "software"
    )
    assert "42" in out, out
    assert not marker.exists()


def test_only_studios_pip_gets_the_chat_pip_prefix(tmp_path):
    """Every OS: Studio's pip installs into the chat folder; nothing else it starts, nor another venv, inherits that."""
    packages = str(tmp_path / os_sandbox.SESSION_PACKAGES_RELPATH)
    env = {
        **os.environ,
        "PYTHONPATH": tools._SANDBOX_SITE_DIR,
        **os_sandbox.studio_pip_prefix(packages),
        "PIP_PREFIX": packages,
    }
    probe = ["-c", "import os; print(os.environ.get('PIP_PREFIX'))"]
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(tmp_path / "venv")], check = True
    )
    venv_python = tmp_path / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def run(*argv):
        result = subprocess.run(argv, env = env, capture_output = True, text = True)
        return result.stdout + result.stderr

    assert f"PIP_PREFIX={packages!r}" in run(sys.executable, "-m", "pip", "config", "debug")
    assert run(sys.executable, *probe).strip() == "None"
    assert run(str(venv_python), *probe).strip() == "None"


def test_a_seatbelt_launch_failure_also_drops_the_cached_verdict(monkeypatch):
    """A rejected Seatbelt profile is reported as `sandbox-exec:`, not `bwrap:`."""
    reset = []
    monkeypatch.setattr(
        "core.inference.sandbox_probe.reset_probe_cache", lambda: reset.append(True)
    )
    prepared = PreparedSandboxLaunch(
        argv = ("/usr/bin/sandbox-exec",),
        workdir = "/work",
        env = {},
        preexec_fn = None,
        backend = "macos-seatbelt",
    )

    tools._forget_sandbox_capability_if_the_backend_failed(
        prepared, "Exit code 1:\nsandbox-exec: sandbox_apply: Operation not permitted\n"
    )
    assert reset == [True]

    tools._forget_sandbox_capability_if_the_backend_failed(
        prepared, "Exit code 1:\nTraceback (most recent call last):\n"
    )
    assert reset == [True]


def test_an_unknown_execution_mode_is_refused_whatever_the_account(monkeypatch):
    """A managed account's confinement is the outer launch contract and skips the planner, which was the only place the mode was checked."""
    import pytest

    with pytest.raises(os_sandbox.SandboxUnavailableError) as refusal:
        tools._requested_execution_mode("nonsense", False)

    assert "TOOL_EXECUTION_MODE_INVALID" in str(refusal.value)
    assert tools._requested_execution_mode("auto", False) == "auto"
    assert tools._requested_execution_mode("required", False) == "required"
    assert tools._requested_execution_mode("auto", True) == "full"


def test_the_unconfined_placeholder_is_not_treated_as_a_boundary():
    """`unconfined-by-owner` records that the owner ALLOWED unconfined tools on a host that cannot confine them."""
    from core.inference.tool_confinement import Confinement

    assert Confinement(mechanism = "unconfined-by-owner").confines is False
    assert Confinement(mechanism = "landlock", preexec = lambda: None).confines is True
    assert Confinement(mechanism = "sandbox-exec", wrapper = ("sandbox-exec",)).confines is True


def test_a_managed_account_without_confinement_still_refuses_required(monkeypatch):
    """The whole point of `required`: no boundary means no execution."""
    from core.inference.tool_confinement import Confinement

    monkeypatch.setattr(
        tools,
        "_account_confinement",
        lambda: Confinement(mechanism = "unconfined-by-owner"),
    )
    _declining_backend(
        monkeypatch, "bubblewrap (bwrap) is not installed on this host", unsafe = False
    )

    result = tools._python_exec("print(1)", None, 60, _SESSION, tool_execution_mode = "required")

    assert "1" not in result.splitlines()[:1], f"the call ran unisolated: {result!r}"
    assert "Execution error" in result or "OS_ISOLATION_UNAVAILABLE" in result


@pytest.mark.parametrize(
    "run",
    [
        lambda **kw: tools._python_exec(
            "import os\nprint('ran', os.path.exists('/srv/pr-approved/data.csv'))",
            None,
            60,
            _SESSION,
            **kw,
        ),
        lambda **kw: tools._bash_exec(
            "ls /srv/pr-approved/data.csv; echo ran", None, 60, _SESSION, **kw
        ),
    ],
    ids = ["python", "terminal"],
)
@pytest.mark.parametrize(
    "approved, mode, jailed",
    [
        (True, "auto", False),
        (False, "auto", True),
        (True, "required", True),
    ],
)
def test_an_approved_host_path_call_runs_as_it_does_without_the_jail(
    monkeypatch, run, approved, mode, jailed
):
    planned: list = []

    def prepare(plan):
        planned.append(plan)
        return _echoing_prepare(_Recorder())(plan)

    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", prepare)
    tools._last_tool_execution_record = None
    out = run(host_access_approved = approved, tool_execution_mode = mode)
    assert "ran" in out
    assert bool(planned) is jailed
    if not jailed:
        record = tools._last_tool_execution_record
        assert record.effective_mode == "software_safeguards"
        assert "user_approved_host_access" in record.limitations


def test_an_approval_does_not_lift_the_jail_for_a_call_that_needs_no_host_path(monkeypatch):
    planned: list = []

    def prepare(plan):
        planned.append(plan)
        return _echoing_prepare(_Recorder())(plan)

    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", prepare)
    assert "3" in tools._python_exec("print(1 + 2)", None, 60, _SESSION, host_access_approved = True)
    assert len(planned) == 1


@pytest.mark.parametrize("executor", ["python", "terminal"])
def test_a_stop_during_the_sandbox_probe_reads_as_a_cancel(monkeypatch, executor):
    # The Windows DACL probe takes seconds; a stop that lands in it is not a sandbox error.
    import threading

    cancel = threading.Event()

    def probe_then_stop(**_kwargs):
        cancel.set()
        return os_sandbox.SandboxCapability(
            backend = "test", available = True, reason = "ok", environment = sys.platform
        )

    monkeypatch.setattr(os_sandbox, "capability_snapshot", probe_then_stop)
    if executor == "python":
        out = tools._python_exec("print('never')", cancel, 60, _SESSION)
    else:
        out = tools._bash_exec("echo never", cancel, 60, _SESSION)
    assert out == "Execution cancelled."


@pytest.mark.parametrize(
    "command",
    [
        "cat /srv/private/report.txt",
        "cat $'/srv/private/report.txt'",
        "cat $'/srv/private/rep\\x6frt.txt'",
    ],
)
def test_an_approved_ansi_c_quoted_host_path_is_recognised(command):
    """The approval classifier decodes $'...' and prompts; the reach check must agree, or the approved call stays jailed."""
    assert tools._reaches_host_paths("terminal", command)


@pytest.mark.skipif(sys.platform != "linux", reason = "the cap only applies on Linux")
@pytest.mark.parametrize(
    "env, saved, expected",
    [
        (None, None, 8),
        (None, 24, 24),
        ("12", 24, 12),
    ],
    ids = ["default", "saved", "env-wins"],
)
def test_a_sandboxed_run_gets_the_configured_memory_limit(monkeypatch, env, saved, expected):
    from utils import sandbox_memory_limit

    if env is None:
        monkeypatch.delenv(sandbox_memory_limit.MEMORY_LIMIT_ENV, raising = False)
    else:
        monkeypatch.setenv(sandbox_memory_limit.MEMORY_LIMIT_ENV, env)
    monkeypatch.setattr(
        sandbox_memory_limit,
        "saved_memory_limit_gb",
        lambda: saved or sandbox_memory_limit.DEFAULT_MEMORY_LIMIT_GB,
    )
    code = "import resource\nprint('AS', resource.getrlimit(resource.RLIMIT_AS)[0])"
    for out in (
        tools._python_exec(code, None, 60, _SESSION),
        tools._bash_exec(f"python3 -c \"{code.replace(chr(10), ';')}\"", None, 60, _SESSION),
    ):
        lines = dict(line.split(" ", 1) for line in out.splitlines() if line.startswith("AS "))
        assert int(lines["AS"]) == expected * 1024**3, out
