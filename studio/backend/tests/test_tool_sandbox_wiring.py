# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two tool launches, now routed through the OS-isolation planner.

``_python_exec`` and ``_bash_exec`` used to build their own ``popen_kwargs`` and
call ``subprocess.Popen``. They now build a ``ToolLaunchPlan`` and spawn what the
planner hands back, which on a host that can isolate is a bubblewrap or Seatbelt
command and on every other host is the same argv main always ran.

That "every other host" clause is the whole claim. These tests must therefore
hold on BOTH paths, and they are written so they do: a host with a working
bubblewrap runs them through the jail, a host that denies the user namespace
runs the same assertions through the fallback. An assertion that only passes
when the sandbox is unavailable is not a check on this PR, it is a check on the
machine it happened to run on. What they pin is the three things that would
break silently:

1. the process tools.py holds still leads its own session, because every kill
   path there is ``killpg`` based and a launch sharing Unsloth's group would take
   the server down with it on a timeout;
2. ``_sandbox_preexec`` still runs no imports after the fork;
3. ``PYTHONPATH``/``HOME``/``TMPDIR`` still point where ``sitecustomize.py`` and
   the download-card flow expect them -- ``TMPDIR`` included, because a temp file
   a tool call writes is served to the user out of the workdir.

Plus the compatibility surface: every parameter added is keyword-only and
defaulted, and ``disable_sandbox`` keeps exactly the meaning it had.
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


# ── backwards compatibility ───────────────────────────────────────────


def test_execute_tool_keeps_every_parameter_it_had_in_the_same_order():
    """A caller passing these positionally must not be silently rebound. The new
    mode is keyword-only, which is what makes that guarantee mechanical."""
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
    # Everything that existed before stays positional-or-keyword, so the four
    # positional call sites in the tests and the loops keep working.
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
    """Its meaning is unchanged, and it wins over the requested mode: Full access
    is the operator having already decided."""
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


# ── auto falls back, and says so ──────────────────────────────────────


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
                # The condition itself, not just "it did not work": an operator
                # can act on the profile name and cannot act on a refusal.
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
        # The software safeguards are still all there; only the OS boundary is not.
        assert "process_guard" in record.retained_safeguards
        assert "no_os_isolation" in record.limitations
        # Never quietly implied: the network is not confined in either mode.
        assert record.network_policy == "unrestricted"

    def test_required_refuses_and_runs_nothing(self):
        out = tools._python_exec(
            "print('SHOULD_NOT_RUN')", None, 60, _SESSION, tool_execution_mode = "required"
        )
        assert "SHOULD_NOT_RUN" not in out
        assert "OS_ISOLATION_UNAVAILABLE" in out
        # The remediation is part of the answer: the person reading it is the one
        # who can fix the host.
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


# ── the three invariants ──────────────────────────────────────────────


def test_the_process_unsloth_holds_still_lands_in_its_own_session():
    """Invariant 1. _capture_process_group / _kill_process_tree / _killpg_captured
    are all killpg based, so a launch sharing Unsloth's group would mean a timeout
    signalling the server.

    Asserted about the process tools.py actually holds and signals, which is the
    OUTER one. Under bubblewrap the payload runs in a fresh PID namespace where
    bwrap is pid 1 and the interpreter is not a session leader, so asking the
    payload about its own sid only ever passes on a host that fell back.
    """
    seen = []
    real = subprocess.Popen

    def capture(argv, **kwargs):
        seen.append(kwargs.get("preexec_fn"))
        return real(argv, **kwargs)

    subprocess.Popen = capture
    try:
        assert "5" in tools._python_exec("print(2 + 3)", None, 60, _SESSION)
        assert "6" in tools._bash_exec("echo 6", None, 60, _SESSION)
    finally:
        subprocess.Popen = real
    assert len(seen) == 2 and all(preexec is not None for preexec in seen)
    # Run what Popen was given, in a fork of our own, and ask the result rather
    # than the identity: an isolated launch composes the plan's pre-exec with the
    # backend's, so the object differs while the setsid that killpg needs is the
    # invariant either way.
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
    """The visible consequence of invariant 1, and the one that holds on the
    isolated path too: a tool outliving its timeout is torn down, and the process
    that launched it is still there to run the next call."""
    start = time.monotonic()
    out = tools._python_exec("import time; time.sleep(120)", None, 5, _SESSION)
    assert "timed out" in out.lower(), out
    assert time.monotonic() - start < 60
    assert "9" in tools._python_exec("print(4 + 5)", None, 60, _SESSION)


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
    """Invariant 2. _libc and _resource are resolved at module import precisely so
    the forked child imports nothing; an import here can deadlock on the import
    lock a thread held at fork time."""
    for function in (tools._sandbox_preexec, tools._bypass_preexec):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        offenders = [
            node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert not offenders, f"{function.__name__} imports after the fork"


@pytest.mark.parametrize("disable_sandbox", [False, True], ids = ["sandboxed", "full"])
def test_the_environment_the_shim_depends_on_survives(disable_sandbox):
    """Invariant 3. sandbox_site/sitecustomize.py is a PYTHONPATH startup hook, and
    the download-card flow reads files out of the workdir HOME points at."""
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
    """The visible consequence of invariant 3: a model writing to /mnt/data still
    gets its file in the workdir and a download card for it."""
    result = tools._python_exec(
        "open('/mnt/data/wiring_probe.csv', 'w').write('a,b\\n')", None, 60, _SESSION
    )
    assert "Error" not in result
    assert os.path.exists(os.path.join(tools._get_workdir(_SESSION), "wiring_probe.csv"))


# ── resources the planner owns ────────────────────────────────────────


class _Recorder:
    """A prepared launch that reports when it was cleaned up."""

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
    """Nothing is prepared in that case, so the point is that the finally block
    does not itself raise on a launch that never existed."""

    def refuse(plan):
        raise SandboxUnavailableError("OS_ISOLATION_UNAVAILABLE: nope", remediation = "install it")

    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", refuse)
    out = tools._python_exec("print(1)", None, 60, _SESSION, tool_execution_mode = "required")
    assert "install it" in out


def test_pass_fds_and_owned_files_reach_the_spawn(monkeypatch):
    """A backend that keeps a control descriptor open across the exec (the macOS
    helper channel) needs both: the fd inherited, and the file object held open
    until the process is done with it."""
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
    # Held open for the whole run, then closed by the cleanup sweep.
    assert recorder.cleaned == 1
    assert holder.closed


# ── auto never fails closed, whatever the planner does ────────────────


def test_auto_still_runs_when_the_planner_itself_breaks(monkeypatch):
    """A backend module that is not importable on this build, a probe raising
    something nobody anticipated: auto's promise is that the tool still runs."""

    def explode(plan):
        raise ImportError("no module named sandbox_linux")

    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", explode)
    tools._last_tool_execution_record = None
    assert "5" in tools._python_exec("print(5)", None, 60, _SESSION)
    record = tools._last_tool_execution_record
    assert record.effective_mode == "software_safeguards"
    assert "sandbox_planner_error" in record.limitations


def test_full_access_keeps_its_own_label_even_when_the_planner_breaks(monkeypatch):
    """A record saying "software safeguards" about a launch that skipped the
    analysis and the rlimits would be a badge claiming more than the run got."""

    def explode(plan):
        raise RuntimeError("planner down")

    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", explode)
    tools._last_tool_execution_record = None
    assert "9" in tools._python_exec("print(9)", None, 60, _SESSION, disable_sandbox = True)
    record = tools._last_tool_execution_record
    assert record.effective_mode == "full"
    assert "security_restrictions_disabled" in record.limitations
    assert "command_and_code_analysis" not in record.retained_safeguards


def test_a_backend_that_drops_the_pre_exec_has_it_put_back(monkeypatch):
    """Silent until the first timeout, and then fatal: without setsid the child
    shares Unsloth's process group and killpg takes the server with it."""

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


# ── platforms with no backend at all ──────────────────────────────────


@pytest.mark.parametrize("platform", ["win32", "cygwin", "aix"])
def test_a_platform_with_no_backend_gets_exactly_the_plan_it_handed_in(monkeypatch, platform):
    """Windows and anything else keep main's behaviour byte for byte: the planner
    returns the same argv, the same env object and the same pre-exec, so the
    popen kwargs built from them are the ones main built."""
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


# ── the tool descriptions were deliberately left alone ────────────────


def test_the_tool_descriptions_are_untouched_by_this_change():
    """_build_sandbox_paths_note() feeds a module-level constant, so a sentence
    about OS isolation there would have to be decided at import time -- before any
    probe has run, and by running one it would put a bwrap launch in the import
    path of tools.py. It would also need a matching _FULL_ACCESS_SUBSTITUTIONS
    entry or Full access would keep advertising a sandbox it disabled. Left for a
    change that can carry the capability into the schema per request."""
    note = tools._build_sandbox_paths_note()
    assert "isolation" not in note.lower()
    # The claim that matters, and the one a new sentence would have broken: Full
    # access still strips every sandbox claim out of the description it ships.
    full = tools._to_full_access(
        "Execute Python code in a sandbox and return stdout/stderr." + note, "python"
    )
    assert "in a sandbox" not in full
    assert "sandbox is disabled" in full


# ── auto never fails closed on the backend's own refusal either ────────

# The planner's OTHER refusal path, and the one that reaches a real user first.
# A backend can decline a specific launch long after the capability probe said
# yes: the Linux one refuses a workdir holding a socket, a nested mount, an
# external hard link or more than 50,000 entries, and an ML project workdir hits
# that last one as a matter of course. In auto that must be a fallback, not the
# end of Python and Terminal for the session.


def _declining_backend(
    monkeypatch,
    reason: str,
    unsafe: bool = True,
) -> None:
    """A planner that refuses. *unsafe* picks WHICH refusal: a workdir the scan
    rejected, or a backend that has stopped being available. The two are told
    apart by type, and they get opposite answers."""
    error = os_sandbox.WorkdirUnsafeError if unsafe else SandboxUnavailableError

    def decline(plan):
        raise error(reason)

    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", decline)


def test_an_unsafe_workdir_fails_the_call_rather_than_de_isolating_it(monkeypatch):
    """The refusal that closes the escalation. Told apart by TYPE, not by asking
    the probe again: a transient probe failure would otherwise re-open the very
    channel the scan just found."""
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
def test_tool_code_cannot_switch_the_boundary_off_for_the_next_call():
    """The escalation the fallback opened: a refusal used to be answered by
    running the next call on the host, so anything sandboxed code could plant in
    its own workdir was a two-line way to get the host back. A refusal now fails
    the CALL, so the worst a littering tool call achieves is breaking its own next
    one, visibly, with the path named."""
    if not os_sandbox.capability_snapshot().available:
        pytest.skip("this host cannot isolate, so there is no boundary to switch off")
    workdir = tools._get_workdir(_SESSION)
    planted = os.path.join(workdir, "planted.sock")
    holder = socket.socket(socket.AF_UNIX)
    holder.bind(planted)
    try:
        tools._last_tool_execution_record = None
        out = tools._python_exec("print('SHOULD_NOT_RUN')", None, 60, _SESSION)
        assert "SHOULD_NOT_RUN" not in out
        assert "device or IPC node" in out
        # And above all: no unisolated launch happened in its place.
        assert tools._last_tool_execution_record is None
    finally:
        holder.close()
        os.unlink(planted)
    # With the litter gone the session is isolated again, not wedged.
    assert "42" in tools._python_exec("print(6 * 7)", None, 60, _SESSION)
    assert tools._last_tool_execution_record.os_isolation is True


# ── full access has exactly one door ──────────────────────────────────


@pytest.mark.parametrize(
    "function,payload",
    [(tools._python_exec, "print('SHOULD_NOT_RUN')"), (tools._bash_exec, "echo SHOULD_NOT_RUN")],
    ids = ["python", "terminal"],
)
def test_full_is_not_requestable_through_the_mode(function, payload):
    """The safe environment, the safety analysis and the resource-limited pre-exec
    are all chosen from disable_sandbox before the mode is read, so an accepted
    tool_execution_mode="full" would skip the OS sandbox with every software
    safeguard still on and then label the run "security restrictions disabled".
    Refused rather than silently downgraded."""
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


def test_an_unknown_mode_is_still_refused_rather_than_run_unisolated():
    """The one SandboxUnavailableError that must never become a fallback launch:
    it is a caller error, not a host that cannot isolate."""
    out = tools._python_exec(
        "print('SHOULD_NOT_RUN')", None, 60, _SESSION, tool_execution_mode = "nonsense"
    )
    assert "SHOULD_NOT_RUN" not in out
    assert "nonsense" in out


# ── what the launch does not carry in from the server ─────────────────


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
    """close_fds leaves 0, 1 and 2 alone, so an unset stdin is whatever Studio was
    started with: an operator terminal, or the file `unsloth studio < f` redirected
    in. That descriptor is already open, so no path rule in either sandbox applies
    to it, and neither tool has an API for supplying input."""
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
    """Nothing stamps the per-call marker and nothing signals through a pidfd:
    teardown is killpg on the captured group, which a tool that calls setsid and
    closes stdout survives. The limitation comes off when the sweep is written."""
    limitations = os_sandbox._software_only_limitations()
    if sys.platform == "win32":
        assert "detached_descendant_cleanup_unverified" not in limitations
    else:
        assert "detached_descendant_cleanup_unverified" in limitations
    assert not hasattr(os_sandbox, "descendant_sweep_supported")


def test_a_backend_that_has_just_stopped_being_available_still_falls_back(monkeypatch):
    """Not every refusal after a successful probe is about the workdir: bwrap
    removed by a package update raises one too, and that is the fallback's own
    case. Told apart by type rather than by a second probe, whose transient
    failure would otherwise be enough to run in a workdir the scan rejected."""
    _declining_backend(
        monkeypatch, "bubblewrap (bwrap) is not installed on this host", unsafe = False
    )
    tools._last_tool_execution_record = None
    assert "42" in tools._python_exec("print(6 * 7)", None, 60, _SESSION)
    record = tools._last_tool_execution_record
    assert record.os_isolation is False
    assert "sandbox_became_unavailable" in record.limitations


def test_a_backend_that_fails_at_launch_drops_the_cached_verdict(monkeypatch):
    """prepare() only builds an argv, so a probe verdict that goes stale under a
    running Studio is not discovered until bwrap exits at exec. Nothing rescues
    the call that already ran, but re-probing bounds the damage to that one call
    instead of every call for the rest of the cache's life."""
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

    # An ordinary tool failure is not one, and neither is a fallback launch.
    tools._forget_sandbox_capability_if_the_backend_failed(
        prepared, "Exit code 1:\nTraceback (most recent call last):\n"
    )
    prepared.backend = "software-safeguards"
    tools._forget_sandbox_capability_if_the_backend_failed(
        prepared, "Exit code 1:\nbwrap: setting up uid map: Permission denied\n"
    )
    assert reset == [True]


def test_a_planner_os_error_refuses_rather_than_running_unisolated(monkeypatch):
    """The type above has to cover a build failure too, and this is why.

    The fallback belongs to a host that cannot isolate. A host whose planner hit
    an OS error can: the backend is still installed and the probe still passes.
    And that errno is reachable from inside the jail, because filling the disk
    makes the next call's seccomp temporary file fail with ENOSPC, so letting it
    reach the general `except Exception` would sell an unisolated launch for the
    price of writing enough data. Measured in a container before the wrap existed:
    "the call ran on the host with the boundary silently off".
    """
    backend = importlib.import_module(
        "core.inference.sandbox_linux"
        if sys.platform == "linux"
        else "core.inference.sandbox_macos"
    )

    def full_disk(plan):
        raise OSError(errno.ENOSPC, "No space left on device")

    # Through the real prepare_tool_launch, because the wrap that types this lives
    # in it. Patching the entry point instead would test nothing.
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
