# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sandboxed tool launches without a Python pre-exec: the child must get exactly what _sandbox_preexec gave it."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

import core.inference.tools as tools
from core.inference import os_sandbox

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason = "the shims are Linux only")

_HAVE_SHIMS = sys.version_info >= (3, 10) and all(
    tools._trusted_system_binary(name) for name in ("setpriv", "prlimit")
)
_needs_shims = pytest.mark.skipif(
    not _HAVE_SHIMS,
    reason = "util-linux setpriv/prlimit not installed, or Python 3.9 (keeps the pre-exec)",
)

_REPORT = textwrap.dedent(
    """
    import ctypes, json, os, resource as r
    libc = ctypes.CDLL(None, use_errno=True)
    sig = ctypes.c_int(-1)
    libc.prctl(2, ctypes.byref(sig), 0, 0, 0)  # PR_GET_PDEATHSIG
    um = os.umask(0)
    nnp = [l.split()[1] for l in open('/proc/self/status') if l.startswith('NoNewPrivs')][0]
    names = ['RLIMIT_NPROC', 'RLIMIT_FSIZE', 'RLIMIT_AS', 'RLIMIT_CPU', 'RLIMIT_NOFILE']
    print(json.dumps(dict(pid=os.getpid(), session_leader=os.getsid(0) == os.getpid(),
        group_leader=os.getpgid(0) == os.getpid(), umask=um, nnp=nnp, pdeathsig=sig.value,
        limits={n: r.getrlimit(getattr(r, n)) for n in names})))
    """
)


@pytest.fixture(autouse = True)
def _fresh_probe(monkeypatch):
    monkeypatch.setattr(tools, "_sandbox_shims", False)


def _launch(tmp_path, argv, **popen_kwargs):
    plan = os_sandbox.ToolLaunchPlan(
        argv = tuple(argv),
        workdir = str(tmp_path),
        env = tools._build_safe_env(str(tmp_path)),
        preexec_fn = tools._sandbox_preexec,
        requested_mode = "software",
        timeout_seconds = 30,
        execution_kind = "python",
    )
    tools._refresh_sandbox_memory_limit()
    prepared = tools._prepare_tool_launch(plan)
    kwargs = dict(
        stdin = subprocess.DEVNULL, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True
    )
    kwargs.update(popen_kwargs)
    kwargs = tools._apply_prepared_launch(prepared, kwargs)
    return os_sandbox.spawn_prepared_launch(prepared, **kwargs), prepared, kwargs


def _report(tmp_path):
    proc, prepared, kwargs = _launch(tmp_path, [sys.executable, "-c", _REPORT])
    out, _ = proc.communicate(timeout = 30)
    seen = json.loads(out.strip().splitlines()[-1])
    assert (
        seen.pop("pid") == proc.pid
    ), "the tool must keep the Popen pid: every kill path signals it"
    return seen, prepared, kwargs


def _alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _gone(pid, budget = 5.0):
    end = time.monotonic() + budget
    while time.monotonic() < end and _alive(pid):
        time.sleep(0.05)
    return not _alive(pid)


@_needs_shims
def test_sandboxed_launch_has_no_python_preexec(tmp_path):
    _, prepared, kwargs = _report(tmp_path)
    assert "preexec_fn" not in kwargs
    assert kwargs["start_new_session"] is True
    assert kwargs["umask"] == 0o077
    assert prepared.argv[0].endswith("/setpriv")


@_needs_shims
def test_child_sees_what_the_preexec_gave_it(tmp_path, monkeypatch):
    shimmed, _, kwargs = _report(tmp_path)
    assert "preexec_fn" not in kwargs
    monkeypatch.setattr(tools, "_sandbox_shims", None)
    reference, _, kwargs = _report(tmp_path)
    assert kwargs["preexec_fn"] is tools._sandbox_preexec
    assert shimmed == reference
    assert reference["session_leader"] and reference["group_leader"]
    assert reference["umask"] == 0o077 and reference["nnp"] == "1"
    assert reference["pdeathsig"] == signal.SIGKILL


@_needs_shims
@pytest.mark.parametrize(
    "env",
    [
        {"UNSLOTH_STUDIO_SANDBOX_NPROC": "not-a-number", "UNSLOTH_STUDIO_SANDBOX_CPU_S": "30"},
        {"UNSLOTH_STUDIO_SANDBOX_NOFILE": str(2**40)},
    ],
)
def test_limit_overrides_match_the_preexec(tmp_path, monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    shimmed, _, kwargs = _report(tmp_path)
    assert "preexec_fn" not in kwargs
    monkeypatch.setattr(tools, "_sandbox_shims", None)
    reference, _, _ = _report(tmp_path)
    assert shimmed["limits"] == reference["limits"]


def test_a_hard_limit_it_could_only_raise_with_privilege_keeps_the_preexec(tmp_path, monkeypatch):
    import resource

    monkeypatch.setattr(tools, "_has_cap_sys_resource", lambda: True)
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_NPROC", str(2**62))
    assert tools._sandbox_rlimit_args() is None
    monkeypatch.setattr(tools, "_has_cap_sys_resource", lambda: False)
    args = tools._sandbox_rlimit_args()
    if resource.getrlimit(resource.RLIMIT_NPROC)[1] != resource.RLIM_INFINITY:
        # Unprivileged, the pre-exec's setrlimit fails and is skipped, so the flag is left out.
        assert not any(arg.startswith("--nproc=") for arg in args)
    assert f"--nofile={min(16384, hard)}:{min(16384, hard)}" in args


def test_missing_shims_keep_the_preexec(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_sandbox_shims", None)
    _, prepared, kwargs = _report(tmp_path)
    assert kwargs["preexec_fn"] is tools._sandbox_preexec
    assert prepared.argv[0] == sys.executable


def test_an_untrusted_shim_is_not_used(tmp_path, monkeypatch):
    fake = tmp_path / "setpriv"
    fake.write_text('#!/bin/sh\nexec "$@"\n')
    fake.chmod(0o755)
    monkeypatch.setattr(tools, "_SANDBOX_SHIM_DIRS", (str(tmp_path),))
    # Not root-owned (the test does not run as root), so it never stands in for the system binary.
    if os.geteuid() != 0:
        assert tools._trusted_system_binary("setpriv") is None


def test_a_composed_backend_preexec_is_left_alone(tmp_path):
    def composed():
        tools._sandbox_preexec()

    prepared = os_sandbox.PreparedSandboxLaunch(
        argv = (sys.executable, "-c", "pass"),
        workdir = str(tmp_path),
        env = {},
        preexec_fn = composed,
        backend = "bubblewrap",
    )
    kwargs = tools._apply_prepared_launch(prepared, {})
    assert kwargs["preexec_fn"] is composed
    assert prepared.argv[0] == sys.executable


@_needs_shims
def test_a_missing_program_still_raises_from_popen(tmp_path):
    with pytest.raises(FileNotFoundError):
        _launch(tmp_path, ["/nonexistent/studio-tool"])


@_needs_shims
def test_the_death_signal_is_armed_before_the_launch_returns(tmp_path):
    """The forking thread exits right after the launch: the pre-exec armed the signal before Popen returned, so
    the child dies with that thread, and the shim launch must not leave a window where it does not."""
    holder = {}

    def launch():
        holder["proc"] = _launch(tmp_path, [sys.executable, "-c", "import time; time.sleep(60)"])[0]

    worker = threading.Thread(target = launch)
    worker.start()
    worker.join()
    proc = holder["proc"]
    try:
        assert _gone(proc.pid)
    finally:
        tools._kill_process_tree(proc)
        proc.wait()


@_needs_shims
def test_the_tool_dies_with_a_killed_server(tmp_path):
    backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server = textwrap.dedent(
        f"""
        import os, subprocess, sys, time
        sys.path.insert(0, {backend!r})
        from core.inference import os_sandbox, tools
        plan = os_sandbox.ToolLaunchPlan(argv=(sys.executable, '-c', 'import time; time.sleep(60)'),
            workdir={str(tmp_path)!r}, env=tools._build_safe_env({str(tmp_path)!r}),
            preexec_fn=tools._sandbox_preexec, requested_mode='software', timeout_seconds=60,
            execution_kind='python')
        prepared = tools._prepare_tool_launch(plan)
        kw = tools._apply_prepared_launch(prepared, dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL))
        assert 'preexec_fn' not in kw
        print(os_sandbox.spawn_prepared_launch(prepared, **kw).pid, flush=True)
        time.sleep(60)
        """
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", server],
        stdout = subprocess.PIPE,
        text = True,
        start_new_session = True,
        cwd = backend,
    )
    try:
        tool_pid = int(parent.stdout.readline())
        assert _alive(tool_pid)
        parent.kill()
        parent.wait()
        assert _gone(tool_pid), "the tool outlived the server that launched it"
    finally:
        parent.kill()
        parent.wait()
        parent.stdout.close()


@_needs_shims
def test_kill_tree_takes_the_whole_group(tmp_path):
    proc, _, _ = _launch(tmp_path, ["bash", "-c", "sleep 100 & sleep 100 & echo $(jobs -p); wait"])
    children = [int(pid) for pid in proc.stdout.readline().split()]
    assert children and all(os.getpgid(pid) == proc.pid for pid in children)
    tools._kill_process_tree(proc)
    proc.wait()
    proc.stdout.close()
    assert all(_gone(pid) for pid in (proc.pid, *children))
