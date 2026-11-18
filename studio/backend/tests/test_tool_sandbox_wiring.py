# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two tool launches, now routed through the OS-isolation planner."""

from __future__ import annotations

import ast
import errno
import importlib
import inspect
import io
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


def test_a_fresh_chat_keeps_the_user_site_for_pip_user(tmp_path):
    """No chat folder yet: user site stays on, so pip --user (left unredirected) still installs."""
    env = tools._with_session_packages({"PATH": "/usr/bin"}, str(tmp_path / "session"))
    assert "PYTHONNOUSERSITE" not in env


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows known folders")
def test_windows_pip_config_dirs_ignore_a_repointed_appdata(tmp_path):
    """pip finds its user pip.ini through the shell, so a repointed %APPDATA% must not hide it."""
    from pip._vendor.platformdirs.windows import get_win_folder

    code = "import sys; sys.path.insert(0, %r); import sitecustomize as s; print(s._win_folder(26, 'APPDATA', 'APPDATA', ''))"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH", "PYTHON"))}
    out = subprocess.run(
        [sys.executable, "-c", code % tools._SANDBOX_SITE_DIR],
        env = {**env, "APPDATA": str(tmp_path)},
        capture_output = True,
        text = True,
    )
    assert out.stdout.strip() == get_win_folder("CSIDL_APPDATA"), out.stderr


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


def _wheel(
    workdir,
    name,
    files,
    version = "1.0",
):
    info = f"{name}-{version}.dist-info"
    wheel = workdir / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as whl:
        for path, text in files.items():
            whl.writestr(path, text)
        whl.writestr(
            f"{info}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        )
        whl.writestr(
            f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        if f"{info}/RECORD" not in files:
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
        "pip_python",
        "pth_zip",
        "own_prefix",
        "abbrev_target",
        "clustered_target",
        "config_target",
        "flat_then_venv",
        "config_devnull",
        "config_file_drops_user",
        "isolated_global_config",
        "config_user_overridden",
        "no_user",
        "config_user_no_user",
        "empty_env_target",
        "no_user_abbrev",
        "option_operand",
        "config_no_user_key",
        "upgrade_isolated",
        "download_into_install",
        "isolated_terminator",
        "pip_home_env",
        "isolated_as_log_operand",
        "config_default_per_file",
        "config_python",
        "empty_cli_target",
        "pip_python_env",
        "last_target_empty",
        "cli_empties_env_target",
        "config_dashed_user",
        "user_config_python_isolated",
        "python_main_parser",
        "log_takes_dashdash",
        "install_section_python",
    ],
)
@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_a_pip_install_without_os_isolation_stays_out_of_studio(monkeypatch, tmp_path, case):
    """Studio's interpreter installs into the chat folder; another venv's pip still installs into that venv."""
    if case == "isolated_global_config" and sys.platform != "linux":
        pytest.skip("XDG_CONFIG_DIRS names pip's global config on Linux only")
    name = f"unsloth_chat_only_{case}"
    workdir = tmp_path / "chat"
    workdir.mkdir()
    packages = str(workdir / os_sandbox.SESSION_PACKAGES_RELPATH)
    files = {f"{name}.py": ""}
    if case == "pth_zip":
        # A .pth line naming a zip archive, which zipimport serves like site.addpackage would.
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as inner:
            inner.writestr(f"{name}.py", "")
        files = {f"{name}.pth": f"{name}.zip\n", f"{name}.zip": archive.getvalue()}
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
        # --isolated ignores PIP_TARGET, so it chooses nothing here.
        "isolated": f"PIP_TARGET=vendor python -m pip --isolated install -q --no-index --no-deps {wheel} && {show}",
        "pth_zip": f"{_PIP} {wheel} && {show}",
        # optparse abbreviations and short-option clusters choose a destination too.
        "abbrev_target": f"{_PIP} --targ vendor {wheel} && PYTHONPATH=vendor {show}",
        # PIP_CONFIG_FILE=/dev/null switches every config file off, the chat's user file included.
        "config_devnull": "mkdir -p .config/pip && printf '[install]\\ntarget = vendor\\n' > "
        f".config/pip/pip.conf && PIP_CONFIG_FILE=/dev/null {_PIP} {wheel} && {show}",
        # An existing PIP_CONFIG_FILE makes pip skip the user file.
        "config_file_drops_user": "mkdir -p .config/pip && printf '[install]\\ntarget = vendor\\n' > "
        f".config/pip/pip.conf && touch empty.ini && PIP_CONFIG_FILE=empty.ini {_PIP} {wheel} && {show}",
        "config_user_no_user": "printf '[install]\\nuser = true\\n' > pip.ini && "
        f"PIP_CONFIG_FILE=pip.ini {_PIP} --no-user {wheel} && {show}",
        # An empty PIP_TARGET is skipped, as pip does: the config's target still applies.
        "empty_env_target": "printf '[install]\\ntarget = vendor\\n' > pip.ini && PIP_CONFIG_FILE=pip.ini "
        f"PIP_TARGET= {_PIP} {wheel} && PYTHONPATH=vendor {show}",
        # --no-user on the command line cancels PIP_USER.
        "no_user": f"PIP_USER=1 {_PIP} --no-user {wheel} && {show}",
        "no_user_abbrev": f"PIP_USER=1 {_PIP} --no-u {wheel} && {show}",
        # pip assigns no-user's value to the same option as user: false here turns the user site off.
        "config_no_user_key": "printf '[global]\\nuser = true\\n[install]\\nno-user = false\\n' > pip.ini && "
        f"PIP_CONFIG_FILE=pip.ini {_PIP} {wheel} && {show}",
        # --upgrade is a flag of its own, not an abbreviation of --upgrade-strategy eating --isolated.
        "upgrade_isolated": f"python -m pip install -q --no-index --no-deps --upgrade --isolated {wheel} && {show}",
        # pip --isolated -- install: the word after a leading -- is still the subcommand.
        "isolated_terminator": f"python -m pip --isolated -- install -q --no-index --no-deps {wheel} && {show}",
        # A [DEFAULT] counts inside its own file only: the env file's user=true never reaches the global [install].
        "config_default_per_file": "mkdir -p xdg/pip && printf '[DEFAULT]\\nuser = false\\n[install]\\n' > xdg/pip/pip.conf"
        " && printf '[DEFAULT]\\nuser = true\\n[uninstall]\\n' > env.ini && "
        f'XDG_CONFIG_DIRS="$PWD/xdg" PIP_CONFIG_FILE=env.ini {_PIP} {wheel} && {show}',
        # A configured python re-runs pip there, like --python: it installs into that venv.
        "config_python": f"python -m venv venv && printf '[global]\\npython = %s\\n' \"$PWD/venv/bin/python\" > pip.ini"
        f" && PIP_CONFIG_FILE=pip.ini python -m pip --isolated install -q --no-index --no-deps {wheel}"
        f" && env -u PYTHONPATH venv/bin/{show}",
        # An empty --target (an unset $DEST) chooses nothing, as pip tests it for truthiness.
        "empty_cli_target": f"{_PIP} --target= {wheel} && {show}",
        # pip normalises config keys (a leading -- dropped): [install] --user = false turns the user site off.
        "config_dashed_user": "printf '[global]\\nuser = true\\n[install]\\n--user = false\\n' > pip.ini && "
        f"PIP_CONFIG_FILE=pip.ini {_PIP} {wheel} && {show}",
        # The user file's python still applies under --isolated: pip's main parser reads it first.
        "user_config_python_isolated": "python -m venv venv && mkdir -p .config/pip && "
        "printf '[global]\\npython = %s\\n' \"$PWD/venv/bin/python\" > .config/pip/pip.conf && "
        f"python -m pip --isolated install -q --no-index --no-deps {wheel} && env -u PYTHONPATH venv/bin/{show}",
        # [install] python is never the main parser's: under --isolated it delegates nowhere.
        "install_section_python": "mkdir -p .config/pip && printf '[install]\\npython = /unused\\n' > .config/pip/pip.conf"
        f" && python -m pip --isolated install -q --no-index --no-deps {wheel} && {show}",
        # --log -- : that -- is the log file name, no terminator to drop before install.
        "log_takes_dashdash": f"python -m pip --isolated --log -- install -q --no-index --no-deps {wheel} && {show}",
        # --python is the main parser's: an install-level --python= cannot undo the one before the subcommand.
        "python_main_parser": 'python -m venv venv && python -m pip --python "$PWD/venv/bin/python" --isolated install'
        f" -q --no-index --no-deps --python= {wheel} && env -u PYTHONPATH venv/bin/{show}",
        # The last --target wins, and an empty one chooses nothing; a CLI value overrides PIP_TARGET too.
        "last_target_empty": f"{_PIP} --target vendor --target= {wheel} && {show}",
        "cli_empties_env_target": f"PIP_TARGET=vendor {_PIP} --target= {wheel} && {show}",
        # PIP_PYTHON re-runs pip in that interpreter even under --isolated.
        "pip_python_env": 'python -m venv venv && PIP_PYTHON="$PWD/venv/bin/python" PIP_CONFIG_FILE=/dev/null '
        f"python -m pip --isolated install -q --no-index --no-deps {wheel} && env -u PYTHONPATH venv/bin/{show}",
        # pip isolates on the raw word, even when it is --log's operand.
        "isolated_as_log_operand": f"{_PIP} --log --isolated {wheel} && {show}",
        # PIP_HOME is no pip option (no --home): it chooses nothing.
        "pip_home_env": f"PIP_HOME=elsewhere {_PIP} {wheel} && {show}",
        # A download destination literally named install is no install subcommand to add --prefix to.
        "download_into_install": f"python -m pip download -q --isolated --dest install --no-index --no-deps {wheel}"
        ' && find "$PWD/install" -name "*.whl"',
        # --prefix=... here is the build-backend setting -C carries, not pip's own --prefix.
        "option_operand": f"{_PIP} -C --prefix=buildtmp {wheel} && {show}",
        # [install] overrides [global], as pip resolves it: user ends up off.
        "config_user_overridden": "printf '[global]\\nuser = true\\n[install]\\nuser = false\\n' > pip.ini && "
        f"PIP_CONFIG_FILE=pip.ini {_PIP} {wheel} && {show}",
        # --isolated still reads the global and interpreter files.
        "isolated_global_config": "mkdir -p xdg/pip && printf '[install]\\ntarget = vendor\\n' > xdg/pip/pip.conf"
        f' && XDG_CONFIG_DIRS="$PWD/xdg" python -m pip --isolated install -q --no-index --no-deps {wheel}'
        f" && PYTHONPATH=vendor {show}",
        "config_target": f"printf '[install]\\ntarget = vendor\\n' > pip.ini && PIP_CONFIG_FILE=pip.ini "
        f"{_PIP} {wheel} && PYTHONPATH=vendor {show}",
        # A --target install into the chat folder itself must not satisfy a project venv's pip either.
        "flat_then_venv": f"{_PIP} --target .unsloth-packages {wheel} && python -m venv venv && "
        f"venv/bin/{_PIP} {wheel} && env -u PYTHONPATH venv/bin/{show}",
        "clustered_target": f"python -m pip install -qt vendor --no-index --no-deps {wheel} && "
        f"PYTHONPATH=vendor {show}",
        # A PIP_PREFIX the command sets itself is kept.
        "own_prefix": f'PIP_PREFIX="$PWD/deps" {_PIP} {wheel} && find "$PWD/deps" -name "{name}.py"',
        "pip_python": f"python -m venv venv && python -m pip --isolated --python venv/bin/python install -q "
        f"--no-index --no-deps {wheel} && env -u PYTHONPATH venv/bin/{show}",
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
        "pth_zip": packages,
        "abbrev_target": str(workdir / "vendor"),
        "config_target": str(workdir / "vendor"),
        "config_devnull": packages,
        "config_file_drops_user": packages,
        "isolated_global_config": str(workdir / "vendor"),
        "config_user_overridden": packages,
        "no_user": packages,
        "no_user_abbrev": packages,
        "option_operand": packages,
        "config_no_user_key": packages,
        "upgrade_isolated": packages,
        "download_into_install": str(workdir / "install"),
        "isolated_terminator": packages,
        "pip_home_env": packages,
        "isolated_as_log_operand": packages,
        "config_default_per_file": packages,
        "config_python": str(workdir / "venv"),
        "empty_cli_target": packages,
        "pip_python_env": str(workdir / "venv"),
        "last_target_empty": packages,
        "cli_empties_env_target": packages,
        "config_dashed_user": packages,
        "user_config_python_isolated": str(workdir / "venv"),
        "python_main_parser": str(workdir / "venv"),
        "log_takes_dashdash": packages,
        "install_section_python": packages,
        "config_user_no_user": packages,
        "empty_env_target": str(workdir / "vendor"),
        "flat_then_venv": str(workdir / "venv"),
        "clustered_target": str(workdir / "vendor"),
        "own_prefix": str(workdir / "deps"),
        "pip_python": str(workdir / "venv"),
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
        "editable_finder_pkgutil",
        "upgrade",
        "namespace",
        "pkgutil_namespace",
        "namespace_upgrade",
        "regular_prepend",
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
        elif case == "editable_finder_pkgutil":
            # Studio's parent is a pkgutil namespace with its own older part; the editable maps only name.part.
            (proj / "part").mkdir()
            (proj / "part" / "__init__.py").write_text("")
            mapping = {f"{name}.part": str(proj / "part")}
            init = "from pkgutil import extend_path; __path__ = extend_path(__path__, __name__)\n"
            studio_files = {f"{name}/__init__.py": init, f"{name}/part/__init__.py": ""}
            show = f"import {name}.part; print({name}.part.__file__)"
            expected = str(proj / "part")
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
    elif case == "regular_prepend":
        # A chat package whose __init__ puts a preferred dir first in its own __path__ keeps that order.
        init = "import os\n__path__.insert(0, os.path.join(__path__[0], 'preferred'))\n"
        files = {
            f"{name}/__init__.py": init,
            f"{name}/child.py": "",
            f"{name}/preferred/child.py": "",
        }
        wheel = _wheel(workdir, name, files)
        command = f'{_PIP} {wheel} && python -c "import {name}.child as c; print(c.__file__)"'
        expected = os.path.join(packages, os_sandbox._PIP_PREFIX_LIB_DIRS[0], name, "preferred")
    elif case == "namespace_upgrade":
        # Studio's namespace is imported first, then a newer child lands in the chat folder.
        studio_files = {f"{name}/child.py": "", f"{name}/other.py": "", f"{name}/data.txt": "v1"}
        wheel = _wheel(workdir, f"{name}_child", {**studio_files, f"{name}/data.txt": "v2"}, "2.0")
        post = (
            f"import importlib.resources, {name}.child as c, {name}.other as o; print(c.__file__); "
            f"print(o.__file__); print(importlib.resources.files({name!r}).joinpath('data.txt').read_text())"
        )
        command = in_process.format(pre = f"import {name}; ", post = post) + f" {wheel}"
        expected = packages
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
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "-q",
                name,
                f"{name}_b",
                f"{name}_child",
            ],
            capture_output = True,
        )
    assert f"{expected}{os.sep}" in out, out
    if case == "namespace_upgrade":
        assert out.count(f"{expected}{os.sep}") == 2 and "v2" in out, out
    if case in ("namespace", "pkgutil_namespace"):
        assert "['a', 'b']" in out, out


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
@pytest.mark.parametrize("editable", [True, False])
def test_the_shim_never_loads_a_chat_planted_stdlib_module(monkeypatch, tmp_path, editable):
    """The shim's helpers come from the stdlib, not the tool-writable chat folder, with or without an editable."""
    workdir = tmp_path / "chat"
    workdir.mkdir()
    marker = tmp_path / "planted_ran"
    site_dir = workdir / os_sandbox.SESSION_PACKAGES_RELPATH / os_sandbox._PIP_PREFIX_LIB_DIRS[0]
    site_dir.mkdir(parents = True)
    if editable:
        finder = "__editable___planted_1_0_finder"
        (site_dir / f"__editable__.planted-1.0.pth").write_text(
            f"import {finder}; {finder}.install()\n"
        )
        # A mapping that names a built-in module must not replace it.
        planted = site_dir / "planted_atexit.py"
        planted.write_text(f"open({str(marker)!r}, 'w').close()\n")
        (site_dir / f"{finder}.py").write_text(
            f"MAPPING = {{'atexit': {str(planted)[:-3]!r}}}\nNAMESPACES = {{}}\n"
        )
    # What reading the finder data imports; types and pathlib are shadowable on main already.
    for helper in ("sysconfig", "ast"):
        (site_dir / f"{helper}.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    command = "python -c 'import collections.abc, xml.dom; print(41 + 1)'"
    if editable:
        # A pip process imports atexit in the shim; pip's own imports already shadow on main.
        command = "python -m pip --version >/dev/null; " + command
    out = tools._bash_exec(command, None, 60, _SESSION, tool_execution_mode = "software")
    assert "42" in out, out
    assert not marker.exists()


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_an_approved_host_access_call_imports_the_chats_editable_installs(monkeypatch, tmp_path):
    """A path-line editable in the chat folder loads in a host-access call; setuptools finders stay inert there."""
    workdir = tmp_path / "chat"
    site_dir = workdir / os_sandbox.SESSION_PACKAGES_RELPATH / os_sandbox._PIP_PREFIX_LIB_DIRS[0]
    site_dir.mkdir(parents = True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "chat_path_editable.py").write_text("")
    (site_dir / "__editable__.chat_path_editable-1.0.pth").write_text(str(tmp_path / "src") + "\n")
    template = os.path.join(
        sysconfig.get_paths()["purelib"], "setuptools", "command", "editable_wheel.py"
    )
    names = ["chat_path_editable"]
    if os.path.exists(template):
        (tmp_path / "flat").mkdir()
        (tmp_path / "flat" / "__init__.py").write_text("")
        finder = "__editable___chat_flat_editable_1_0_finder"
        (site_dir / "__editable__.chat_flat_editable-1.0.pth").write_text(
            f"import {finder}; {finder}.install()\n"
        )
        mapping = {"chat_flat_editable": str(tmp_path / "flat")}
        (site_dir / f"{finder}.py").write_text(f"MAPPING = {mapping!r}\nNAMESPACES = {{}}\n")
        names.append("chat_flat_editable")
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    code = "import sys, chat_path_editable; print(chat_path_editable.__file__); print(sys.path)"
    if len(names) == 2:
        code += "; import importlib.util as u; print(u.find_spec('chat_flat_editable'))"
    command = f'ls /srv/pr-approved >/dev/null 2>&1; python -c "{code}"'
    out = tools._bash_exec(
        command, None, 60, _SESSION, tool_execution_mode = "auto", host_access_approved = True
    )
    assert "user_approved_host_access" in tools._last_tool_execution_record.limitations
    assert str(tmp_path / "src" / "chat_path_editable.py") in out, out
    # A finder would sit outside the host finder's package boundary and plant a path-hook entry on sys.path.
    assert str(tmp_path / "flat") not in out, out


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_a_chat_reinstall_retires_the_files_the_old_version_had(monkeypatch, tmp_path):
    """--ignore-installed never uninstalls: the shim drops what only the replaced chat-local version shipped."""
    name = "unsloth_chat_retire"
    workdir = tmp_path / "chat"
    workdir.mkdir()
    old = _wheel(workdir, name, {f"{name}/__init__.py": "", f"{name}/old/__init__.py": ""}, "1.0")
    new = _wheel(workdir, name, {f"{name}/__init__.py": ""}, "2.0")
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    command = (
        f"{_PIP} {old} && sleep 1 && {_PIP} {new} && "
        f'python -c \'import importlib.util as u; print("OLD", u.find_spec("{name}.old"))\''
    )
    out = tools._bash_exec(command, None, 120, _SESSION, tool_execution_mode = "software")
    assert "OLD None" in out, out


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_a_same_version_reinstall_drops_metadata_the_new_wheel_lacks(monkeypatch, tmp_path):
    """--force-reinstall of the same version: an entry point only the old wheel declared is gone."""
    name = "unsloth_chat_reinstall_meta"
    workdir = tmp_path / "chat"
    (workdir / "again").mkdir(parents = True)
    info = f"{name}-1.0.dist-info"
    first = _wheel(
        workdir,
        name,
        {f"{name}.py": "", f"{info}/entry_points.txt": f"[pytest11]\n{name} = {name}\n"},
    )
    second = "again/" + _wheel(workdir / "again", name, {f"{name}.py": ""})
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    probe = (
        "import importlib.metadata as m; "
        f"print('EP', m.distribution('{name}').read_text('entry_points.txt'))"
    )
    command = (
        f'{_PIP} {first} && sleep 1 && {_PIP} --force-reinstall {second} && python -c "{probe}"'
    )
    out = tools._bash_exec(command, None, 120, _SESSION, tool_execution_mode = "software")
    assert "EP None" in out, out


def test_an_editable_over_a_chat_wheel_of_the_same_version_wins(monkeypatch, tmp_path):
    """The same-version editable rewrites RECORD in place; the wheel's own files go, so the source loads."""
    name = "unsloth_chat_wheel_then_editable"
    workdir = tmp_path / "chat"
    workdir.mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / f"{name}.py").write_text("")
    wheel = _wheel(workdir, name, {f"{name}.py": ""})
    (workdir / "editable").mkdir()
    editable = "editable/" + _wheel(
        workdir / "editable", name, {f"__editable__.{name}-1.0.pth": str(tmp_path / "src") + "\n"}
    )
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    command = (
        f"{_PIP} {wheel} && sleep 1 && {_PIP} {editable} && "
        f"python -c 'import {name}; print({name}.__file__)'"
    )
    out = tools._bash_exec(command, None, 120, _SESSION, tool_execution_mode = "software")
    assert str(tmp_path / "src" / f"{name}.py") in out, out


def test_the_namespace_check_never_loads_ast_from_an_editable_source(monkeypatch, tmp_path):
    """The ast the session finder imports is the stdlib one, not a copy in a path-line editable's folder."""
    workdir = tmp_path / "chat"
    marker = tmp_path / "planted_ran"
    site_dir = workdir / os_sandbox.SESSION_PACKAGES_RELPATH / os_sandbox._PIP_PREFIX_LIB_DIRS[0]
    (site_dir / "email").mkdir(parents = True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "ast.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    (site_dir / "__editable__.planted-1.0.pth").write_text(str(tmp_path / "src") + "\n")
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    command = "python -c 'import email.mime; print(41 + 1)'"
    out = tools._bash_exec(command, None, 60, _SESSION, tool_execution_mode = "software")
    assert "42" in out, out
    assert not marker.exists()


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_a_checkout_in_the_cwd_wins_over_a_chat_editable_finder(monkeypatch, tmp_path):
    """The editable fork beats Studio's copy, not the command's own working directory."""
    template = os.path.join(
        sysconfig.get_paths()["purelib"], "setuptools", "command", "editable_wheel.py"
    )
    if not os.path.exists(template):
        pytest.skip("finder-based editable installs need setuptools in Studio's environment")
    name = "unsloth_chat_cwd_shadow"
    workdir = tmp_path / "chat"
    site_dir = workdir / os_sandbox.SESSION_PACKAGES_RELPATH / os_sandbox._PIP_PREFIX_LIB_DIRS[0]
    site_dir.mkdir(parents = True)
    (tmp_path / "editable" / name).mkdir(parents = True)
    (tmp_path / "editable" / name / "__init__.py").write_text("")
    finder = f"__editable___{name}_1_0_finder"
    (site_dir / f"__editable__.{name}-1.0.pth").write_text(f"import {finder}; {finder}.install()\n")
    mapping = {name: str(tmp_path / "editable" / name)}
    (site_dir / f"{finder}.py").write_text(f"MAPPING = {mapping!r}\nNAMESPACES = {{}}\n")
    (workdir / "checkout").mkdir()
    (workdir / "checkout" / f"{name}.py").write_text("")
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    command = f"cd checkout && python -c 'import {name}; print({name}.__file__)'"
    out = tools._bash_exec(command, None, 60, _SESSION, tool_execution_mode = "software")
    assert str(workdir / "checkout" / f"{name}.py") in out, out


@pytest.mark.skipif(sys.platform == "win32", reason = "posix symlinks and shell")
def test_retiring_a_chat_version_never_follows_a_symlink_out_of_the_prefix(monkeypatch, tmp_path):
    """A recorded folder that links elsewhere: the files behind the link survive the cleanup."""
    workdir = tmp_path / "chat"
    site_dir = workdir / os_sandbox.SESSION_PACKAGES_RELPATH / os_sandbox._PIP_PREFIX_LIB_DIRS[0]
    site_dir.mkdir(parents = True)
    outside = tmp_path / "dataset"
    outside.mkdir()
    (outside / "keep.txt").write_text("data")
    (site_dir / "linked").symlink_to(outside, target_is_directory = True)
    name = "unsloth_chat_linked"
    old = site_dir / f"{name}-1.0.dist-info"
    old.mkdir()
    (old / "RECORD").write_text(f"linked/keep.txt,,\n{name}-1.0.dist-info/RECORD,,\n")
    time.sleep(0.05)
    new = site_dir / f"{name}-2.0.dist-info"
    new.mkdir()
    (new / "RECORD").write_text(
        f"{name}-2.0.dist-info/INSTALLER,,\n{name}-2.0.dist-info/RECORD,,\n"
    )
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    out = tools._bash_exec(
        "python -m pip --version", None, 60, _SESSION, tool_execution_mode = "software"
    )
    assert "pip" in out, out
    assert (outside / "keep.txt").read_text() == "data"
    assert not old.exists()


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_the_namespace_check_never_loads_ast_from_the_cwd(monkeypatch, tmp_path):
    """The cwd joins sys.path after the shim runs; the ast import must not search it either."""
    workdir = tmp_path / "chat"
    marker = tmp_path / "planted_ran"
    site_dir = workdir / os_sandbox.SESSION_PACKAGES_RELPATH / os_sandbox._PIP_PREFIX_LIB_DIRS[0]
    (site_dir / "email").mkdir(parents = True)
    (workdir / "here").mkdir()
    (workdir / "here" / "ast.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    command = "cd here && python -c 'import email.mime; print(41 + 1)'"
    out = tools._bash_exec(command, None, 60, _SESSION, tool_execution_mode = "software")
    assert "42" in out, out
    assert not marker.exists()


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_a_checkout_package_keeps_its_own_submodules_over_a_chat_editable(monkeypatch, tmp_path):
    """The parent came from the checkout, so its child must too, not from the editable's mapping."""
    template = os.path.join(
        sysconfig.get_paths()["purelib"], "setuptools", "command", "editable_wheel.py"
    )
    if not os.path.exists(template):
        pytest.skip("finder-based editable installs need setuptools in Studio's environment")
    name = "unsloth_chat_child_shadow"
    workdir = tmp_path / "chat"
    site_dir = workdir / os_sandbox.SESSION_PACKAGES_RELPATH / os_sandbox._PIP_PREFIX_LIB_DIRS[0]
    site_dir.mkdir(parents = True)
    for root in (tmp_path / "editable" / name, workdir / "checkout" / name):
        root.mkdir(parents = True)
        (root / "__init__.py").write_text("")
        (root / "child.py").write_text("")
    finder = f"__editable___{name}_1_0_finder"
    (site_dir / f"__editable__.{name}-1.0.pth").write_text(f"import {finder}; {finder}.install()\n")
    mapping = {name: str(tmp_path / "editable" / name)}
    (site_dir / f"{finder}.py").write_text(f"MAPPING = {mapping!r}\nNAMESPACES = {{}}\n")
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    command = f"cd checkout && python -c 'import {name}.child as c; print(c.__file__)'"
    out = tools._bash_exec(command, None, 60, _SESSION, tool_execution_mode = "software")
    assert str(workdir / "checkout" / name / "child.py") in out, out


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_an_editable_child_of_a_namespace_wins_over_studios_copy(monkeypatch, tmp_path):
    """Studio's portion of an aggregated namespace is no checkout: the chat's editable child still wins."""
    template = os.path.join(
        sysconfig.get_paths()["purelib"], "setuptools", "command", "editable_wheel.py"
    )
    if not os.path.exists(template):
        pytest.skip("finder-based editable installs need setuptools in Studio's environment")
    ns = "unsloth_chat_ns_parent"
    workdir = tmp_path / "chat"
    site_dir = workdir / os_sandbox.SESSION_PACKAGES_RELPATH / os_sandbox._PIP_PREFIX_LIB_DIRS[0]
    site_dir.mkdir(parents = True)
    (tmp_path / "editable" / "pkg").mkdir(parents = True)
    (tmp_path / "editable" / "pkg" / "__init__.py").write_text("")
    finder = f"__editable___{ns}_pkg_1_0_finder"
    (site_dir / f"__editable__.{ns}_pkg-1.0.pth").write_text(
        f"import {finder}; {finder}.install()\n"
    )
    mapping = {f"{ns}.pkg": str(tmp_path / "editable" / "pkg")}
    (site_dir / f"{finder}.py").write_text(f"MAPPING = {mapping!r}\nNAMESPACES = {{{ns!r}: []}}\n")
    studio_dir = tmp_path / "studio"
    studio_dir.mkdir()
    studio_wheel = studio_dir / _wheel(studio_dir, ns, {f"{ns}/pkg/__init__.py": ""})
    subprocess.run([sys.executable, *_PIP.split()[1:], str(studio_wheel)], check = True)
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    command = f"python -c 'import {ns}.pkg as p; print(p.__file__)'"
    try:
        out = tools._bash_exec(command, None, 60, _SESSION, tool_execution_mode = "software")
    finally:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "-q", ns], capture_output = True
        )
    assert str(tmp_path / "editable" / "pkg" / "__init__.py") in out, out


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_a_coding_cookie_in_a_finder_file_imports_no_chat_codec(monkeypatch, tmp_path):
    """Finder files are read as UTF-8 text before any finder exists, so a cookie cannot import a mapped codec."""
    template = os.path.join(
        sysconfig.get_paths()["purelib"], "setuptools", "command", "editable_wheel.py"
    )
    if not os.path.exists(template):
        pytest.skip("finder-based editable installs need setuptools in Studio's environment")
    workdir = tmp_path / "chat"
    marker = tmp_path / "codec_ran"
    site_dir = workdir / os_sandbox.SESSION_PACKAGES_RELPATH / os_sandbox._PIP_PREFIX_LIB_DIRS[0]
    site_dir.mkdir(parents = True)
    payload = tmp_path / "chatcodec.py"
    payload.write_text(f"open({str(marker)!r}, 'w').close()\n")
    for tag, body in (
        ("a", f"MAPPING = {{'encodings.chatcodec': {str(payload)!r}}}\nNAMESPACES = {{}}\n"),
        ("b", "# coding: chatcodec\nMAPPING = {}\nNAMESPACES = {}\n"),
    ):
        finder = f"__editable___{tag}_1_0_finder"
        (site_dir / f"__editable__.{tag}-1.0.pth").write_text(
            f"import {finder}; {finder}.install()\n"
        )
        (site_dir / f"{finder}.py").write_text(body)
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    out = tools._bash_exec(
        "python -c 'print(41 + 1)'", None, 60, _SESSION, tool_execution_mode = "software"
    )
    assert "42" in out, out
    assert not marker.exists()


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_retirement_keeps_a_file_another_distribution_now_ships(monkeypatch, tmp_path):
    """A module moving from core 1.0 into a new companion package stays installed after the upgrade."""
    workdir = tmp_path / "chat"
    (workdir / "v2").mkdir(parents = True)
    core1 = _wheel(
        workdir,
        "unsloth_split_core",
        {"unsloth_split/__init__.py": "", "unsloth_split/feature.py": ""},
        "1.0",
    )
    core2 = "v2/" + _wheel(
        workdir / "v2", "unsloth_split_core", {"unsloth_split/__init__.py": ""}, "2.0"
    )
    feature = "v2/" + _wheel(
        workdir / "v2", "unsloth_split_feature", {"unsloth_split/feature.py": ""}
    )
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    command = (
        f"{_PIP} {core1} && sleep 1 && {_PIP} {core2} {feature} && "
        "python -c 'import unsloth_split.feature as f; print(f.__file__)'"
    )
    out = tools._bash_exec(command, None, 120, _SESSION, tool_execution_mode = "software")
    assert "feature.py" in out and "Error" not in out, out


def test_another_python_sharing_studios_prefix_is_not_redirected():
    """/usr/local holds 3.11 and 3.12: only Studio's minor version gets the chat prefix."""
    code = "import sitecustomize as s; print('chat', s._chat_prefix)"
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH"))},
        "PYTHONPATH": tools._SANDBOX_SITE_DIR,
        "_UNSLOTH_PIP_PREFIX": "/nonexistent/.unsloth-packages",
        "_UNSLOTH_PIP_PREFIX_FOR": sys.prefix,
    }
    runs = {
        version: subprocess.run(
            [sys.executable, "-c", code],
            env = {**env, "_UNSLOTH_PIP_PREFIX_PY": version},
            capture_output = True,
            text = True,
        ).stdout.strip()
        for version in ("%d.%d" % sys.version_info[:2], "2.7")
    }
    assert list(runs.values()) == ["chat /nonexistent/.unsloth-packages", "chat None"], runs


@pytest.mark.skipif(sys.platform == "win32", reason = "posix paths")
def test_python_m_pip_without_orig_argv_is_routed_once_pip_starts(tmp_path):
    """Python 3.9: -m pip is recognised when runpy loads pip.__main__; -m of anything else is left alone."""
    wheel = tmp_path / _wheel(tmp_path, "unsloth_late_pip", {"unsloth_late_pip.py": ""})
    packages = tmp_path / ".unsloth-packages"
    env = {
        **{
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("PIP_", "_UNSLOTH", "PYTHONPATH"))
        },
        "_UNSLOTH_PIP_PREFIX": str(packages),
        "_UNSLOTH_PIP_PREFIX_FOR": sys.prefix,
    }
    code = (
        "import os, runpy, sys; del sys.orig_argv; sys.argv[:] = ['-m'] + sys.argv[1:]; "
        "sys.path.insert(0, %r); import sitecustomize; "
        "runpy.run_module(sys.argv.pop(1), run_name = '__main__', alter_sys = True)"
    ) % tools._SANDBOX_SITE_DIR
    run = [sys.executable, "-c", code]
    pip = subprocess.run(
        run + ["pip", "install", "-q", "--no-index", "--no-deps", str(wheel)],
        env = env,
        capture_output = True,
        text = True,
    )
    assert list(packages.rglob("unsloth_late_pip.py")), pip.stderr


def test_pip_config_is_read_in_the_locale_encoding(tmp_path):
    """Outside UTF-8 mode pip reads pip.conf in the locale's encoding: a cp1252 comment still yields its target."""
    conf = tmp_path / "pip.ini"
    conf.write_bytes(b"# caf\xe9\n[install]\ntarget = vendor\n")
    code = (
        "import locale, sys; locale.getpreferredencoding = lambda do_setlocale = True: 'cp1252'; "
        "sys.path.insert(0, %r); import sitecustomize as s; print(s._config_chooses_location())"
    ) % tools._SANDBOX_SITE_DIR
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH", "PYTHON"))}
    out = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", code],
        env = {**env, "PIP_CONFIG_FILE": str(conf)},
        capture_output = True,
        text = True,
    )
    assert out.stdout.strip() == "True", out.stderr


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_a_path_line_editable_never_shadows_the_stdlib(monkeypatch, tmp_path):
    """An editable source tree holding shutil.py goes after the stdlib, still ahead of Studio's site-packages."""
    workdir = tmp_path / "chat"
    source = tmp_path / "src"
    source.mkdir()
    marker = tmp_path / "payload_ran"
    for module in ("shutil", "msvcrt"):  # msvcrt: a stdlib name with no file on Linux to come first
        (source / f"{module}.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    site_dir = workdir / os_sandbox.SESSION_PACKAGES_RELPATH / os_sandbox._PIP_PREFIX_LIB_DIRS[0]
    site_dir.mkdir(parents = True)
    (site_dir / "__editable__.planted-1.0.pth").write_text(f"{source}\n")
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    command = (
        "python -m pip --version; python -c 'import shutil; print(shutil.__file__)'; "
        "python -c 'import msvcrt' 2>/dev/null || echo no-msvcrt"
    )
    out = tools._bash_exec(command, None, 60, _SESSION, tool_execution_mode = "software")
    assert "pip" in out and str(source) not in out and "no-msvcrt" in out, out
    assert not marker.exists()


def test_host_session_packages_never_supply_a_stdlib_name(tmp_path):
    """The host finder sits last: a msvcrt.py there must not answer subprocess's probe on Linux."""
    folder = tmp_path / "host_packages"
    folder.mkdir()
    marker = tmp_path / "payload_ran"
    (folder / "msvcrt.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    env = {
        **{
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("PIP_", "_UNSLOTH", "PYTHONPATH"))
        },
        "PYTHONPATH": tools._SANDBOX_SITE_DIR,
        "_UNSLOTH_HOST_SESSION_PACKAGES": str(folder),
    }
    code = "import subprocess; print('ok')"
    out = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, text = True)
    assert out.stdout.strip() == "ok" and not marker.exists(), out.stderr


def test_a_preset_prefix_tracks_python_m_pip_without_orig_argv(tmp_path):
    """OS-isolated launches set PIP_PREFIX themselves: on 3.9 a -m pip still gets the exit-status tracking."""
    code = (
        "import runpy, sys; del sys.orig_argv; sys.argv[:] = ['-m', 'pip', '--version']; sys.path.insert(0, %r); "
        "import sitecustomize as s\n"
        "try:\n    runpy.run_module('pip', run_name = '__main__', alter_sys = True)\n"
        "except SystemExit:\n    pass\n"
        "print('tracked', bool(s._pip_tracked))"
    ) % tools._SANDBOX_SITE_DIR
    env = {
        **{
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("PIP_", "_UNSLOTH", "PYTHONPATH"))
        },
        "PIP_PREFIX": str(tmp_path / "prefix"),
    }
    out = subprocess.run(
        [sys.executable, "-c", code], env = env, capture_output = True, text = True, cwd = tmp_path
    )
    assert "tracked True" in out.stdout, out.stderr


def test_a_module_launch_without_orig_argv_is_not_taken_for_pip():
    """Python 3.9 has no sys.orig_argv: python -m venv --upgrade-deps must not get the chat PIP_PREFIX."""
    code = (
        "import sys; del sys.orig_argv; sys.argv[0] = '-m'; sys.path.insert(0, %r); "
        "import sitecustomize as s; print(s._is_pip_launch())" % tools._SANDBOX_SITE_DIR
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH", "PYTHON"))}
    out = subprocess.run(
        [sys.executable, "-S", "-c", code], env = env, capture_output = True, text = True
    )
    assert out.stdout.strip() == "False", out.stderr


@pytest.mark.skipif(
    sys.platform == "win32", reason = "quote characters are not valid in Windows file names"
)
def test_record_paths_are_read_as_csv(tmp_path):
    """A CSV-quoted RECORD name (the file "victim", quotes included) never retires a neighbour named victim."""
    site_dir = tmp_path / "prefix" / "lib" / "site-packages"
    old = _dist(site_dir, "unsloth_quoted", "1.0", 1, module = '"victim"')
    (old / "RECORD").write_text(f'"""victim""",,\n{old.name}/RECORD,,\n')
    os.utime(old / "RECORD", ns = (10**9, 10**9))
    (site_dir / "victim").write_text("")
    _dist(site_dir, "unsloth_quoted", "2.0", 2)
    _prune(tmp_path / "prefix", site_dir)
    assert (site_dir / "victim").exists() and not (site_dir / '"victim"').exists()


def test_the_stdlib_inventory_has_a_fallback_before_python_3_10():
    """Without sys.stdlib_module_names (3.9), the interpreter's own stdlib dirs still name shutil."""
    code = (
        "import sys; del sys.stdlib_module_names; sys.path.insert(0, %r); "
        "import sitecustomize as s; names = s._stdlib_names(); "
        "print('shutil' in names, 'json' in names, 'msvcrt' in names, 'unsloth_chat_mod' in names)"
        % tools._SANDBOX_SITE_DIR
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH", "PYTHON"))}
    out = subprocess.run(
        [sys.executable, "-S", "-c", code], env = env, capture_output = True, text = True
    )
    assert out.stdout.split() == ["True", "True", "True", "False"], out.stderr


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
@pytest.mark.parametrize("retry", ["same_version", "new_version"])
def test_a_failed_same_version_reinstall_still_retires_on_retry(monkeypatch, tmp_path, retry):
    """The failing reinstall rewrote RECORD before dying: the retry (same build or a new version) still drops
    what only the old build had."""
    name = "unsloth_retry_same"
    workdir = tmp_path / "chat"
    workdir.mkdir()
    (workdir / "again").mkdir()
    first = _wheel(workdir, name, {f"{name}.py": "", f"{name}_gone.py": ""})
    info = f"{name}-1.0.dist-info"
    second = "again/" + _wheel(
        workdir / "again",
        name,
        {
            f"{name}.py": "",
            f"{info}/entry_points.txt": f"[console_scripts]\n{name}_cli = {name}:main\n",
            f"{info}/RECORD": f"{name}.py,,\n{info}/RECORD,,\n",
        },
    )
    packages = workdir / os_sandbox.SESSION_PACKAGES_RELPATH
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    tools._bash_exec(f"{_PIP} {first}", None, 120, _SESSION, tool_execution_mode = "software")
    blocker = packages / "bin" / f"{name}_cli" / "x"
    blocker.mkdir(parents = True)
    failed = tools._bash_exec(
        f"{_PIP} --force-reinstall {second}; echo rc=$?",
        None,
        120,
        _SESSION,
        tool_execution_mode = "software",
    )
    assert "rc=0" not in failed, failed
    blocker.rmdir()
    blocker.parent.rmdir()
    if retry == "new_version":
        (workdir / "newer").mkdir()
        second = "newer/" + _wheel(workdir / "newer", name, {f"{name}.py": ""}, "2.0")
    out = tools._bash_exec(
        f"{_PIP} --force-reinstall {second}; echo rc=$?",
        None,
        120,
        _SESSION,
        tool_execution_mode = "software",
    )
    assert "rc=0" in out and not list(packages.rglob(f"{name}_gone.py")), out


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_a_failed_upgrade_keeps_the_old_versions_files(monkeypatch, tmp_path):
    """pip exits 1 after unpacking a wheel that ships INSTALLER itself: the old version's files stay."""
    name = "unsloth_failed_upgrade"
    workdir = tmp_path / "chat"
    workdir.mkdir()
    old = _wheel(workdir, name, {f"{name}/__init__.py": "", f"{name}/old.py": ""})
    # A companion upgraded in the same run before the failing wheel: its finished 2.0 stays the one in use.
    done = [_wheel(workdir, f"{name}_ok", {f"{name}_ok.py": ""}, v) for v in ("1.0", "2.0")]
    # And one reinstalled at the same version (--force-reinstall) that no longer ships gone.py.
    (workdir / "again").mkdir()
    same = [
        _wheel(workdir, f"{name}_same", {f"{name}_same.py": "", f"{name}_gone.py": ""}),
        "again/" + _wheel(workdir / "again", f"{name}_same", {f"{name}_same.py": ""}),
    ]
    info = f"{name}-2.0.dist-info"
    new = _wheel(
        workdir,
        name,
        {
            f"{name}/__init__.py": "",
            f"{info}/INSTALLER": "pip\n",
            f"{info}/entry_points.txt": f"[console_scripts]\n{name}_cli = {name}:main\n",
            f"{info}/RECORD": f"{name}/__init__.py,,\n{info}/INSTALLER,,\n{info}/RECORD,,\n",
        },
        "2.0",
    )
    packages = workdir / os_sandbox.SESSION_PACKAGES_RELPATH
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    first = f"{_PIP} {old} {done[0]} {same[0]}"
    tools._bash_exec(first, None, 120, _SESSION, tool_execution_mode = "software")
    # A non-empty folder where pip must write the console script makes the install fail late.
    (packages / "bin" / f"{name}_cli" / "x").mkdir(parents = True)
    out = tools._bash_exec(
        f"{_PIP} --force-reinstall {done[1]} {same[1]} {new}; echo rc=$?",
        None,
        120,
        _SESSION,
        tool_execution_mode = "software",
    )
    assert "rc=0" not in out, out
    assert list(packages.rglob(f"{name}/old.py")), out
    assert not list(packages.rglob(f"{name}_gone.py")), out
    assert not list(packages.rglob(f"{name}_ok-1.0.dist-info")), out
    assert not list(packages.glob(".unsloth-installing-*")), out
    assert not list(packages.glob(".unsloth-old-files-*")), out
    # Nor does a later process take the half-written 2.0 for a finished replacement.
    later = tools._bash_exec(
        f"python -m pip --version; python -c \"import importlib.metadata as m; print('ok', m.version('{name}_ok'))\"",
        None,
        120,
        _SESSION,
        tool_execution_mode = "software",
    )
    assert list(packages.rglob(f"{name}/old.py")), out
    assert "ok 2.0" in later and not list(packages.rglob(f"{name}_ok-1.0.dist-info")), later


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_an_editable_mapping_never_replaces_a_stdlib_module(monkeypatch, tmp_path):
    """A finder mapping shutil to chat code: the cleanup pass, and anything else, still gets the stdlib one."""
    template = os.path.join(
        sysconfig.get_paths()["purelib"], "setuptools", "command", "editable_wheel.py"
    )
    if not os.path.exists(template):
        pytest.skip("finder-based editable installs need setuptools in Studio's environment")
    workdir = tmp_path / "chat"
    marker = tmp_path / "payload_ran"
    site_dir = workdir / os_sandbox.SESSION_PACKAGES_RELPATH / os_sandbox._PIP_PREFIX_LIB_DIRS[0]
    site_dir.mkdir(parents = True)
    payload = tmp_path / "shutil.py"
    payload.write_text(f"open({str(marker)!r}, 'w').close()\n")
    finder = "__editable___planted_1_0_finder"
    (site_dir / "__editable__.planted-1.0.pth").write_text(f"import {finder}; {finder}.install()\n")
    # msvcrt: a stdlib name absent on Linux, so only setuptools' own finder behind PathFinder could supply it.
    mapped = {"shutil": str(payload)[:-3], "msvcrt": str(payload)[:-3]}
    (site_dir / f"{finder}.py").write_text(f"MAPPING = {mapped!r}\nNAMESPACES = {{}}\n")
    for version in ("1.0", "2.0"):
        info = site_dir / f"unsloth_twice-{version}.dist-info"
        info.mkdir()
        (info / "RECORD").write_text(f"{info.name}/RECORD,,\n")
        time.sleep(0.05)
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    command = (
        "python -m pip --version; python -c 'import shutil; print(shutil.__file__)'; "
        "python -c 'import msvcrt' 2>/dev/null || echo no-msvcrt"
    )
    out = tools._bash_exec(command, None, 60, _SESSION, tool_execution_mode = "software")
    assert "pip" in out and str(payload) not in out and "no-msvcrt" in out, out
    assert not marker.exists()


def test_retirement_keeps_what_a_distribution_in_the_other_site_dir_ships(tmp_path):
    """purelib and platlib can be separate dirs: a companion's RECORD there still protects its files."""
    prefix = tmp_path / "prefix"
    pure, plat = prefix / "lib" / "site-packages", prefix / "lib64" / "site-packages"
    (prefix / "bin").mkdir(parents = True)
    (prefix / "bin" / "tool").write_text("")
    for version, stamp in (("1.0", 1), ("2.0", 2)):
        info = pure / f"unsloth_pure-{version}.dist-info"
        info.mkdir(parents = True)
        rows = "../../bin/tool,,\n" if version == "1.0" else ""
        (info / "RECORD").write_text(rows + f"{info.name}/INSTALLER,,\n{info.name}/RECORD,,\n")
        os.utime(info / "RECORD", ns = (stamp * 10**9, stamp * 10**9))
    companion = plat / "unsloth_plat-1.0.dist-info"
    companion.mkdir(parents = True)
    (companion / "RECORD").write_text(f"../../bin/tool,,\n{companion.name}/RECORD,,\n")
    code = "import sitecustomize as s; s._prune_stale_dist_info()"
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH"))},
        "PYTHONPATH": os.pathsep.join([tools._SANDBOX_SITE_DIR, str(pure), str(plat)]),
        "PIP_PREFIX": str(prefix),
    }
    subprocess.run([sys.executable, "-c", code], env = env, check = True)
    assert not (pure / "unsloth_pure-1.0.dist-info").exists()
    assert (prefix / "bin" / "tool").exists()


def _prune(prefix, *site_dirs):
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH"))},
        "PYTHONPATH": os.pathsep.join([tools._SANDBOX_SITE_DIR, *map(str, site_dirs)]),
        "PIP_PREFIX": str(prefix),
    }
    code = "import sitecustomize as s; s._prune_stale_dist_info()"
    subprocess.run([sys.executable, "-c", code], env = env, check = True)


def _dist(
    site_dir,
    project,
    version,
    stamp,
    finished = True,
    module = None,
):
    info = site_dir / f"{project}-{version}.dist-info"
    info.mkdir(parents = True)
    module = module or f"{project}.py"
    (site_dir / module).write_text("")
    installer = f"{info.name}/INSTALLER,,\n" if finished else ""
    (info / "RECORD").write_text(f"{module},,\n{installer}{info.name}/RECORD,,\n")
    os.utime(info / "RECORD", ns = (stamp * 10**9, stamp * 10**9))
    return info


@pytest.mark.skipif(sys.platform == "win32", reason = "symlinked site dirs")
def test_one_site_dir_on_two_paths_is_one_install(tmp_path):
    """lib64 -> lib with both on sys.path: the single installed version is not its own stale copy."""
    prefix = tmp_path / "prefix"
    site_dir = prefix / "lib" / "site-packages"
    info = _dist(site_dir, "unsloth_aliased", "1.0", 1)
    (prefix / "lib64").symlink_to(prefix / "lib")
    _prune(prefix, site_dir, prefix / "lib64" / "site-packages")
    assert info.exists() and (site_dir / "unsloth_aliased.py").exists()


@pytest.mark.skipif(sys.platform == "win32", reason = "symlinked site dirs")
def test_a_site_dir_linked_outside_the_prefix_is_left_alone(tmp_path):
    """An older version behind a symlink out of the prefix is never deleted, metadata included."""
    prefix, outside = tmp_path / "prefix", tmp_path / "outside"
    old = _dist(outside, "unsloth_linked", "1.0", 1)
    (prefix / "lib").mkdir(parents = True)
    (prefix / "lib" / "site-packages").symlink_to(outside)
    _dist(prefix / "lib64" / "site-packages", "unsloth_linked", "2.0", 2)
    _prune(prefix, prefix / "lib" / "site-packages", prefix / "lib64" / "site-packages")
    assert old.exists() and (outside / "unsloth_linked.py").exists()


def test_a_killed_install_retires_nothing(tmp_path):
    """pip SIGKILLed mid-wheel leaves only its pending note: the new dist-info, INSTALLER row and all, is not trusted."""
    prefix = tmp_path / "prefix"
    site_dir = prefix / "lib" / "site-packages"
    old = _dist(site_dir, "unsloth_killed", "1.0", 1, module = "old.py")
    _dist(site_dir, "unsloth_killed", "2.0", 2, module = "new.py")
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    note = prefix / f".unsloth-installing-{dead.pid}"
    note.write_text(f"{os.path.abspath(site_dir)}\t{old.name}\t{10**9}\n")
    _prune(prefix, site_dir)
    assert old.exists() and (site_dir / "old.py").exists()
    assert (
        not note.exists() and "unsloth_killed-2.0" in (prefix / ".unsloth-incomplete").read_text()
    )


def test_a_killed_reinstall_note_is_read_before_the_next_upgrade(tmp_path):
    """A dead note with nothing to prune yet is consumed at startup, so a later finished upgrade still counts."""
    prefix = tmp_path / "prefix"
    site_dir = prefix / "lib" / "site-packages"
    old = _dist(site_dir, "unsloth_noted", "1.0", 1, module = "old.py")
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    note = prefix / f".unsloth-installing-{dead.pid}"
    note.write_text(f"{os.path.abspath(site_dir)}\t{old.name}\t{10**9}\n")
    _prune(prefix, site_dir)
    assert not note.exists()
    _dist(site_dir, "unsloth_noted", "2.0", 2, module = "new.py")
    _prune(prefix, site_dir)
    assert not old.exists()


@pytest.mark.skipif(sys.platform == "win32", reason = "symlinked folders")
def test_a_saved_old_file_list_is_rechecked_before_use(tmp_path):
    """A folder of a saved list since replaced by a symlink out of the prefix drops that entry."""
    prefix = tmp_path / "prefix"
    site_dir = prefix / "lib" / "site-packages"
    site_dir.mkdir(parents = True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.py").write_text("")
    (site_dir / "sub").symlink_to(outside)
    inside = site_dir / "kept.py"
    rows = [
        f"{os.path.abspath(site_dir)}\tp-1.0.dist-info\t{os.path.normcase(str(p))}\n"
        for p in (site_dir / "sub" / "x.py", inside)
    ]
    (prefix / ".unsloth-old-files").write_text("".join(rows))
    code = "import sitecustomize as s; print(sorted(p for v in s._saved_inventory().values() for p in v))"
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH"))},
        "PYTHONPATH": os.pathsep.join([tools._SANDBOX_SITE_DIR, str(site_dir)]),
        "PIP_PREFIX": str(prefix),
    }
    out = subprocess.run(
        [sys.executable, "-c", code], env = env, capture_output = True, text = True, cwd = tmp_path
    )
    assert out.stdout.strip() == repr([os.path.normcase(str(inside))]), out.stderr


def test_a_killed_reinstall_keeps_its_old_file_list(tmp_path):
    """SIGKILL after RECORD was rewritten: the run's on-disk inventory still feeds the next retirement."""
    prefix = tmp_path / "prefix"
    site_dir = prefix / "lib" / "site-packages"
    info = _dist(site_dir, "unsloth_killed_same", "1.0", 2, module = "kept.py")
    # Reinstalled at the same version earlier in the run, before the kill: its note stamp is already current.
    done = _dist(site_dir, "unsloth_done_same", "1.0", 3, module = "done.py")
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    site = os.path.abspath(site_dir)
    (prefix / f".unsloth-installing-{dead.pid}").write_text(
        f"{site}\t{info.name}\t{10**9}\n{site}\t{done.name}\t{3 * 10**9}\n"
    )
    gone = os.path.normcase(str(site_dir / "gone.py"))
    gone_done = os.path.normcase(str(site_dir / "gone_done.py"))
    (prefix / f".unsloth-old-files-{dead.pid}").write_text(
        f"{site}\t{info.name}\t{gone}\n{site}\t{done.name}\t{gone_done}\n"
    )
    code = "import sitecustomize as s; print(sorted(p for v in s._saved_inventory().values() for p in v))"
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH"))},
        "PYTHONPATH": os.pathsep.join([tools._SANDBOX_SITE_DIR, str(site_dir)]),
        "PIP_PREFIX": str(prefix),
    }
    out = subprocess.run(
        [sys.executable, "-c", code], env = env, capture_output = True, text = True, cwd = tmp_path
    )
    assert out.stdout.strip() == repr(sorted([gone, gone_done])), out.stderr
    assert not (prefix / f".unsloth-old-files-{dead.pid}").exists()


def test_a_planted_importlib_never_disables_the_metadata_hook(tmp_path):
    """The shim's own helper imports come after the hook and from the stdlib: a chat importlib/ changes nothing."""
    planted = tmp_path / "planted"
    (planted / "importlib").mkdir(parents = True)
    (planted / "importlib" / "__init__.py").write_text("raise RuntimeError('planted')\n")
    code = (
        "import socket\ns = socket.socket()\ns.settimeout(0.1)\n"
        "try:\n    s.connect(('169.254.169.254', 80))\nexcept PermissionError:\n    print('blocked')\n"
        "except OSError:\n    print('unblocked')"
    )
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH"))},
        "PYTHONPATH": os.pathsep.join([str(planted), tools._SANDBOX_SITE_DIR]),
        "_UNSLOTH_SANDBOX_BLOCK_METADATA": "1",
    }
    out = subprocess.run(
        [sys.executable, "-c", code], env = env, capture_output = True, text = True, cwd = tmp_path
    )
    assert out.stdout.strip() == "blocked", out.stderr


def test_malformed_cleanup_state_never_disables_the_metadata_hook(tmp_path):
    """A non-UTF-8 byte in the chat-writable bookkeeping must not stop the shim before the audit hook."""
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    for name in (".unsloth-incomplete", ".unsloth-old-files"):
        (prefix / name).write_bytes(b"\xff\xfe\n")
    code = (
        "import socket\ns = socket.socket()\ns.settimeout(0.1)\n"
        "try:\n    s.connect(('169.254.169.254', 80))\nexcept PermissionError:\n    print('blocked')\n"
        "except OSError:\n    print('unblocked')"
    )
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH"))},
        "PYTHONPATH": tools._SANDBOX_SITE_DIR,
        "PIP_PREFIX": str(prefix),
        "_UNSLOTH_SANDBOX_BLOCK_METADATA": "1",
    }
    out = subprocess.run(
        [sys.executable, "-c", code], env = env, capture_output = True, text = True, cwd = tmp_path
    )
    assert out.stdout.strip() == "blocked", out.stderr


def test_a_run_killed_between_wheels_keeps_its_file_lists(tmp_path):
    """No pending note (killed between wheels): the dead run's inventory is still kept, not dropped."""
    prefix = tmp_path / "prefix"
    site_dir = prefix / "lib" / "site-packages"
    info = _dist(site_dir, "unsloth_between", "1.0", 2, module = "kept.py")
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    gone = os.path.normcase(str(site_dir / "gone.py"))
    (prefix / f".unsloth-old-files-{dead.pid}").write_text(
        f"{os.path.abspath(site_dir)}\t{info.name}\t{gone}\n"
    )
    code = "import sitecustomize as s; print(sorted(p for v in s._saved_inventory().values() for p in v))"
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH"))},
        "PYTHONPATH": os.pathsep.join([tools._SANDBOX_SITE_DIR, str(site_dir)]),
        "PIP_PREFIX": str(prefix),
    }
    out = subprocess.run(
        [sys.executable, "-c", code], env = env, capture_output = True, text = True, cwd = tmp_path
    )
    assert out.stdout.strip() == repr([gone]), out.stderr


@pytest.mark.skipif(sys.platform == "win32", reason = "flock liveness is POSIX only")
def test_a_live_pip_mid_wheel_is_not_pruned_around(tmp_path):
    """Another interpreter starting while pip is mid-wheel leaves that pip's half-written version alone."""
    prefix = tmp_path / "prefix"
    site_dir = prefix / "lib" / "site-packages"
    old = _dist(site_dir, "unsloth_live", "1.0", 1, module = "old.py")
    _dist(site_dir, "unsloth_live", "2.0", 2, module = "new.py")
    note = prefix / ".unsloth-installing-2-live"
    note.write_text(f"{os.path.abspath(site_dir)}\t{old.name}\t{10**9}\n")
    # The other pip holds its note's lock while it runs (its pid may be 2 in every bwrap namespace).
    hold = "import fcntl, os, sys, time; fd = os.open(sys.argv[1], os.O_RDONLY); fcntl.flock(fd, fcntl.LOCK_EX); print('held', flush = True); time.sleep(60)"
    live = subprocess.Popen(
        [sys.executable, "-c", hold, str(note)], stdout = subprocess.PIPE, text = True
    )
    try:
        assert live.stdout.readline().strip() == "held"
        _prune(prefix, site_dir)
        assert old.exists() and (site_dir / "old.py").exists() and note.exists()
    finally:
        live.kill()
        live.wait()
    # Once it is gone the same note reads as a killed run's.
    _prune(prefix, site_dir)
    assert not note.exists() and old.exists()


def test_path_line_files_drop_a_utf8_bom(tmp_path):
    """site.addpackage decodes utf-8-sig: a BOM before the first path is not part of it."""
    path = tmp_path / "x.pth"
    path.write_bytes(b"\xef\xbb\xbf" + str(tmp_path).encode() + b"\n")
    code = (
        "import sys; sys.path.insert(0, %r); import sitecustomize as s; print(s._read_pth(%r) == [%r])"
        % (
            tools._SANDBOX_SITE_DIR,
            str(path),
            str(tmp_path),
        )
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH", "PYTHON"))}
    out = subprocess.run(
        [sys.executable, "-S", "-c", code], env = env, capture_output = True, text = True
    )
    assert out.stdout.strip() == "True", out.stderr


@pytest.mark.skipif(sys.platform == "win32", reason = "posix shell commands")
def test_a_replaced_editable_finder_stops_answering(monkeypatch, tmp_path):
    """An editable replaced by a regular wheel in this process: the retired finder no longer supplies 1.0."""
    template = os.path.join(
        sysconfig.get_paths()["purelib"], "setuptools", "command", "editable_wheel.py"
    )
    if not os.path.exists(template):
        pytest.skip("finder-based editable installs need setuptools in Studio's environment")
    name = "unsloth_retired_editable"
    workdir = tmp_path / "chat"
    proj = workdir / "proj" / "impl"
    proj.mkdir(parents = True)
    (proj / "__init__.py").write_text("VERSION = 'editable'\n")
    finder = f"__editable___{name}_1_0_finder"
    editable = _wheel(
        workdir,
        name,
        {
            f"__editable__.{name}-1.0.pth": f"import {finder}; {finder}.install()\n",
            f"{finder}.py": f"MAPPING: dict = {{{name!r}: {str(proj)!r}}}\nNAMESPACES: dict = {{}}\n",
        },
    )
    (workdir / "newer").mkdir()
    regular = "newer/" + _wheel(
        workdir / "newer", name, {f"{name}/__init__.py": "VERSION = 'wheel'\n"}, "2.0"
    )
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(workdir))
    tools._bash_exec(f"{_PIP} {editable}", None, 120, _SESSION, tool_execution_mode = "software")
    command = (
        'python -c "import subprocess, sys; subprocess.check_call(sys.argv[1:]); '
        f'import {name}; print({name}.VERSION)" ' + _PIP + f" {regular}"
    )
    out = tools._bash_exec(command, None, 120, _SESSION, tool_execution_mode = "software")
    assert out.strip().endswith("wheel"), out


def test_path_line_files_fall_back_to_the_locale_encoding(tmp_path):
    """As site.addpackage: a path-line file that is not UTF-8 is read in the locale's encoding."""
    source = tmp_path / "caf\u00e9"
    path = tmp_path / "x.pth"
    path.write_bytes(str(source).encode("cp1252") + b"\n")
    code = (
        "import locale, sys; locale.getpreferredencoding = lambda do_setlocale = True: 'cp1252'; "
        "sys.path.insert(0, %r); import sitecustomize as s; print(s._read_pth(%r) == [%r])"
    ) % (tools._SANDBOX_SITE_DIR, str(path), str(source))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH", "PYTHON"))}
    out = subprocess.run(
        [sys.executable, "-S", "-c", code], env = env, capture_output = True, text = True
    )
    assert out.stdout.strip() == "True", out.stderr


@pytest.mark.skipif(sys.platform == "win32", reason = "symlinked folders")
def test_retirement_rechecks_each_path_before_removing_it(tmp_path):
    """A folder swapped for a symlink out of the prefix after the list was taken: nothing outside is removed."""
    prefix = tmp_path / "prefix"
    site_dir = prefix / "lib" / "site-packages"
    (site_dir / "pkg").mkdir(parents = True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "old_only.py").write_text("")
    victim = os.path.normcase(str(site_dir / "pkg" / "old_only.py"))
    (site_dir / "pkg").rmdir()
    (site_dir / "pkg").symlink_to(outside)
    code = "import sitecustomize as s; s._remove_files({%r}, %r)" % (victim, str(site_dir))
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH"))},
        "PYTHONPATH": os.pathsep.join([tools._SANDBOX_SITE_DIR, str(site_dir)]),
        "PIP_PREFIX": str(prefix),
    }
    subprocess.run([sys.executable, "-c", code], env = env, check = True, cwd = tmp_path)
    assert (outside / "old_only.py").exists()


def test_a_note_named_with_this_processes_pid_is_still_read(tmp_path):
    """bwrap's --unshare-pid makes every call PID 2: a dead run's note under the same pid is not ours."""
    prefix = tmp_path / "prefix"
    site_dir = prefix / "lib" / "site-packages"
    old = _dist(site_dir, "unsloth_same_pid", "1.0", 1, module = "old.py")
    _dist(site_dir, "unsloth_same_pid", "2.0", 2, module = "new.py")
    code = (
        "import os, sys; prefix = sys.argv[1]; site = sys.argv[2]\n"
        "open(os.path.join(prefix, '.unsloth-installing-' + str(os.getpid())), 'w').write("
        "os.path.abspath(site) + '\\tunsloth_same_pid-1.0.dist-info\\t1000000000\\n')\n"
        "sys.path.insert(0, %r)\nimport sitecustomize"
    ) % tools._SANDBOX_SITE_DIR
    env = {
        **{
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("PIP_", "_UNSLOTH", "PYTHONPATH"))
        },
        "PIP_PREFIX": str(prefix),
        "PYTHONPATH": str(site_dir),
    }
    subprocess.run(
        [sys.executable, "-S", "-c", code, str(prefix), str(site_dir)],
        env = env,
        check = True,
        cwd = tmp_path,
    )
    assert old.exists() and (site_dir / "old.py").exists()
    assert not list(prefix.glob(".unsloth-installing-*"))


@pytest.mark.skipif(sys.platform == "win32", reason = "flock")
def test_an_unheld_note_on_nfs_reads_as_dead(tmp_path):
    """NFS emulates flock with record locks: an exclusive one on a read-only descriptor fails EBADF."""
    note = tmp_path / ".unsloth-installing-1-ab"
    note.write_text("")
    code = (
        "import errno, fcntl, os, sys, types; sys.path.insert(0, %r); import sitecustomize as s\n"
        "def flock(fd, op):\n"
        "    if op & fcntl.LOCK_EX and fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY:\n"
        "        raise OSError(errno.EBADF, 'Bad file descriptor')\n"
        "    fcntl.flock(fd, op)\n"
        "nfs = types.SimpleNamespace(flock = flock, LOCK_EX = fcntl.LOCK_EX, LOCK_SH = fcntl.LOCK_SH, LOCK_NB = fcntl.LOCK_NB)\n"
        "s._fcntl = lambda: nfs; print(s._alive(%r))"
    ) % (tools._SANDBOX_SITE_DIR, str(note))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH", "PYTHON"))}
    out = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, text = True)
    assert out.stdout.strip() == "False", out.stderr


@pytest.mark.skipif(sys.platform == "win32", reason = "flock liveness is POSIX only")
def test_each_prune_pass_rereads_live_notes(tmp_path):
    """An interpreter started before another pip: its exit-time pass still sees that pip's live note."""
    prefix = tmp_path / "prefix"
    site_dir = prefix / "lib" / "site-packages"
    old = _dist(site_dir, "unsloth_later", "1.0", 1, module = "old.py")
    note = prefix / ".unsloth-installing-2-later"
    hold = (
        "import fcntl, os, sys, time; open(sys.argv[1], 'w').write(sys.argv[2]); fd = os.open(sys.argv[1], os.O_RDONLY); "
        "fcntl.flock(fd, fcntl.LOCK_EX); print('held', flush = True); time.sleep(60)"
    )
    code = (
        "import subprocess, sys; sys.path.insert(0, %r); import sitecustomize as s\n"
        "exec(sys.argv[1])\n"
        "p = subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[3], sys.argv[4]], stdout = subprocess.PIPE, text = True)\n"
        "assert p.stdout.readline().strip() == 'held'\n"
        "s._prune_stale_dist_info()\np.kill()"
    ) % tools._SANDBOX_SITE_DIR
    make_new = (
        "import os; d = %r; i = os.path.join(d, 'unsloth_later-2.0.dist-info'); os.makedirs(i); "
        "open(os.path.join(d, 'new.py'), 'w').close(); "
        "open(os.path.join(i, 'RECORD'), 'w').write('new.py,,\\nunsloth_later-2.0.dist-info/INSTALLER,,\\n')"
    ) % str(site_dir)
    env = {
        **{
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("PIP_", "_UNSLOTH", "PYTHONPATH"))
        },
        "PIP_PREFIX": str(prefix),
        "PYTHONPATH": str(site_dir),
    }
    row = f"{os.path.abspath(site_dir)}\t{old.name}\t{10**9}\n"
    subprocess.run(
        [sys.executable, "-S", "-c", code, make_new, hold, str(note), row],
        env = env,
        check = True,
        cwd = tmp_path,
    )
    assert old.exists() and (site_dir / "old.py").exists()


def test_a_preloaded_namespace_gets_a_path_line_editable_portion(tmp_path):
    """A pkgutil namespace imported before the shim still finds a child in a path-line editable root."""
    studio, root = tmp_path / "studio", tmp_path / "src"
    for base in (studio, root):
        (base / "unsloth_preloaded_ns").mkdir(parents = True)
    init = "from pkgutil import extend_path; __path__ = extend_path(__path__, __name__)\n"
    (studio / "unsloth_preloaded_ns" / "__init__.py").write_text(init)
    (root / "unsloth_preloaded_ns" / "child.py").write_text("")
    code = (
        "import sys; sys.path.insert(0, %r); import unsloth_preloaded_ns as ns\n"
        "sys.path.insert(0, %r); import sitecustomize as s; sys.path.append(%r)\n"
        "f = s._SessionPackagesFinder([], [], [%r])\n"
        "spec = f.find_spec('unsloth_preloaded_ns.child', ns.__path__)\nprint(spec.origin)"
    ) % (str(studio), tools._SANDBOX_SITE_DIR, str(root), str(root))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH", "PYTHON"))}
    out = subprocess.run(
        [sys.executable, "-S", "-c", code], env = env, capture_output = True, text = True, cwd = tmp_path
    )
    assert out.stdout.strip() == str(root / "unsloth_preloaded_ns" / "child.py"), out.stderr


def test_notes_without_flock_are_written_in_place_and_removed(tmp_path):
    """Windows has no flock: notes go straight to their name (no .tmp left behind) and _release drops them."""
    code = (
        "import sys; sys.path.insert(0, %r); import sitecustomize as s; s._fcntl = lambda: None\n"
        "p = sys.argv[1]; fd = s._write_held(p, ['x\\n']); import os; print(fd, os.path.exists(p), os.listdir(os.path.dirname(p)))\n"
        "s._release(fd, p); print(os.listdir(os.path.dirname(p)))"
    ) % tools._SANDBOX_SITE_DIR
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH", "PYTHON"))}
    note = tmp_path / ".unsloth-installing-1-x"
    out = subprocess.run(
        [sys.executable, "-S", "-c", code, str(note)], env = env, capture_output = True, text = True
    )
    assert out.stdout.splitlines() == ["None True ['.unsloth-installing-1-x']", "[]"], out.stderr


def test_an_unfinished_install_retires_nothing(tmp_path):
    """A RECORD without pip's INSTALLER row is a wheel's own copy from an install that died halfway."""
    site_dir = tmp_path / "prefix" / "lib" / "site-packages"
    old = _dist(site_dir, "unsloth_halfway", "1.0", 1, module = "old.py")
    _dist(site_dir, "unsloth_halfway", "2.0", 2, finished = False, module = "new.py")
    _prune(tmp_path / "prefix", site_dir)
    assert old.exists() and (site_dir / "old.py").exists()


def test_a_replaced_version_in_the_other_site_dir_is_pruned(tmp_path):
    """purelib and platlib apart: version 1.0 in one and 2.0 in the other still leaves only 2.0."""
    prefix = tmp_path / "prefix"
    pure, plat = prefix / "lib" / "site-packages", prefix / "lib64" / "site-packages"
    for site_dir, version, stamp in ((pure, "1.0", 1), (plat, "2.0", 2)):
        info = site_dir / f"unsloth_split_scheme-{version}.dist-info"
        info.mkdir(parents = True)
        (site_dir / f"unsloth_split_scheme_{version[0]}.py").write_text("")
        (info / "RECORD").write_text(
            f"unsloth_split_scheme_{version[0]}.py,,\n{info.name}/INSTALLER,,\n{info.name}/RECORD,,\n"
        )
        os.utime(info / "RECORD", ns = (stamp * 10**9, stamp * 10**9))
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "_UNSLOTH"))},
        "PYTHONPATH": os.pathsep.join([tools._SANDBOX_SITE_DIR, str(pure), str(plat)]),
        "PIP_PREFIX": str(prefix),
    }
    code = "import sitecustomize as s; s._prune_stale_dist_info()"
    subprocess.run([sys.executable, "-c", code], env = env, check = True)
    assert not (pure / "unsloth_split_scheme-1.0.dist-info").exists()
    assert not (pure / "unsloth_split_scheme_1.py").exists()
    assert (plat / "unsloth_split_scheme-2.0.dist-info").exists()


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
    # python -mvenv: a module glued to -m is still not pip.
    (tmp_path / "probe_mod.py").write_text(probe[1])
    glued = subprocess.run(
        [sys.executable, "-mprobe_mod"], env = env, cwd = tmp_path, capture_output = True, text = True
    )
    assert glued.stdout.strip() == "None", glued.stderr
    # An interpreter option with its own operand comes first: still the module that counts.
    probe_x = [sys.executable, "-X", "utf8", "-mprobe_mod"]
    pip_x = [sys.executable, "-X", "utf8", "-m", "pip", "config", "debug"]
    result = subprocess.run(probe_x, env = env, cwd = tmp_path, capture_output = True, text = True)
    assert result.stdout.strip() == "None", result.stdout + result.stderr
    result = subprocess.run(pip_x, env = env, cwd = tmp_path, capture_output = True, text = True)
    assert f"PIP_PREFIX={packages!r}" in result.stdout, result.stdout + result.stderr
    # Clustered short options: -um pip is still pip.
    clustered = [sys.executable, "-um", "pip", "config", "debug"]
    result = subprocess.run(clustered, env = env, cwd = tmp_path, capture_output = True, text = True)
    assert f"PIP_PREFIX={packages!r}" in result.stdout, result.stdout + result.stderr


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
