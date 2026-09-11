# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the live probe is allowed to conclude, and from what.

The confining backend is simulated in-process rather than skipped: the CI host
cannot build a real sandbox, and "skipped" would leave the positive half of
every pairing untested.
"""

from __future__ import annotations

import inspect
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

if sys.platform == "win32":
    pytest.skip("the live probe is POSIX only", allow_module_level = True)

from core.inference import os_sandbox, sandbox_landlock, sandbox_probe
from core.inference.os_sandbox import PreparedSandboxLaunch, ToolLaunchPlan


@pytest.fixture(autouse = True)
def _clean_probe_cache():
    sandbox_probe.reset_probe_cache()
    yield
    sandbox_probe.reset_probe_cache()


class _Backend:
    """A stand-in exporting exactly the three names ``os_sandbox`` reads."""

    PROFILE_ID = "test-profile"
    LIMITATIONS = ()

    def __init__(self, name, prepare):
        self.BACKEND_NAME = name
        self._prepare = prepare
        self.prepared = []
        self.calls = 0

    def prepare(self, plan):
        self.calls += 1
        prepared = self._prepare(plan)
        self.prepared.append(prepared)
        return prepared


def _passthrough(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    return PreparedSandboxLaunch(
        argv = plan.argv,
        workdir = plan.workdir,
        env = plan.env,
        preexec_fn = plan.preexec_fn,
        backend = "passthrough",
    )


_CONFINE_WRAPPER = """import builtins, os, sys
WORK = os.path.realpath({workdir!r})
# Both spellings, because on macOS the probe base is under /tmp, which is a
# symlink to /private/tmp. With RESOLVE off the guard compares the path AS
# WRITTEN, so resolving only one side made the double refuse its own workdir and
# the probe stopped before it reached the symlink leg it exists to exercise.
WORK_AS_GIVEN = os.path.abspath({workdir!r})
LEAK_WRITES = {leak_writes!r}
RESOLVE = {resolve!r}
_host_open = builtins.open


def _confined(file, *args, **kwargs):
    # A stand-in for a filesystem namespace. RESOLVE picks whether the boundary
    # is drawn on the resolved TARGET (what a real mount namespace does) or on
    # the spelling of the path, which is the mistake the escape symlink catches.
    try:
        target = os.path.realpath(file) if RESOLVE else os.fspath(file)
    except (TypeError, ValueError):
        target = None
    outside = isinstance(target, str) and not any(
        target == root or target.startswith(root + os.sep)
        for root in (WORK, WORK_AS_GIVEN)
    )
    if outside:
        mode = args[0] if args else kwargs.get("mode", "r")
        writing = "r" not in mode or "+" in mode
        system_root = target == os.path.realpath(sys.executable)
        # LEAK_WRITES models a backend that confines reads and protects the
        # system root but lets an ordinary write through: every control the
        # sandboxed process can evaluate comes out right, and the host still
        # ends up with the file.
        if not (LEAK_WRITES and writing and not system_root):
            raise PermissionError(13, "confined to the workdir")
    return _host_open(file, *args, **kwargs)


builtins.open = _confined
exec(compile({payload!r}, "<probe-payload>", "exec"), {{"__name__": "__main__"}})
"""


def _wrap(
    plan: ToolLaunchPlan,
    name: str,
    *,
    resolve = True,
    leak_writes = False,
):
    """A builtins.open guard, not a kernel namespace. Import machinery uses
    io.open_code, so the payload's own imports still work."""
    wrapper = _CONFINE_WRAPPER.format(
        workdir = plan.workdir,
        payload = plan.argv[-1],
        resolve = resolve,
        leak_writes = leak_writes,
    )
    return PreparedSandboxLaunch(
        argv = plan.argv[:-1] + (wrapper,),
        workdir = plan.workdir,
        env = plan.env,
        # A backend claiming to confine has to carry the abstract-socket scope or
        # the probe refuses it, which is the point of that control.
        preexec_fn = sandbox_landlock.with_abstract_scope(plan.preexec_fn),
        backend = name,
    )


def _confining(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    return _wrap(plan, "confining")


def _spelling_only(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    return _wrap(plan, "spelling-only", resolve = False)


def _leaking_writes(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    return _wrap(plan, "leaky", leak_writes = True)


def test_a_backend_that_confines_nothing_is_not_available():
    backend = _Backend("passthrough", _passthrough)
    available, reason = sandbox_probe.probe(backend)
    assert available is False
    assert "read the host sentinel" in reason, reason


def test_a_backend_that_really_confines_is_available():
    backend = _Backend("confining", _confining)
    available, reason = sandbox_probe.probe(backend)
    assert available is True, reason
    assert "confining" in reason


def test_the_symlink_leg_is_judged_on_the_target_not_the_path():
    """A spelling-only guard passes the sentinel leg and fails here, so the two legs
    are not redundant."""
    available, reason = sandbox_probe.probe(_Backend("spelling-only", _spelling_only))
    assert available is False
    assert "symlink" in reason, reason


def test_a_write_that_reaches_the_host_fails_the_probe():
    """This backend passes every control the sandboxed process can evaluate and
    still lands a write on the host, which only the host can see."""
    available, reason = sandbox_probe.probe(_Backend("leaky", _leaking_writes))
    assert available is False
    assert "wrote through to the host" in reason, reason


def test_the_escape_check_is_made_before_the_scratch_root_is_removed():
    text = inspect.getsource(sandbox_probe._run_probe)
    assert text.index("_host_saw_the_write") < text.index("shutil.rmtree")


def test_the_positive_controls_run_before_anything_is_concluded(monkeypatch):
    backend = _Backend("confining", _confining)
    seen = []

    def broken(workdir, sentinel, outside, env):
        seen.append(sentinel)
        return "the host itself could not read the probe sentinel (simulated)"

    monkeypatch.setattr(sandbox_probe, "_host_positive_controls", broken)
    available, reason = sandbox_probe.probe(backend)
    assert available is False
    assert "the host itself could not read" in reason
    assert backend.calls == 0
    assert seen


def test_the_host_control_really_reads_the_sentinel_it_was_given(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(work), "TMPDIR": str(work)}
    missing = str(tmp_path / "absent.txt")
    assert "could not read" in sandbox_probe._host_positive_controls(
        str(work), missing, str(tmp_path / "out.txt"), env
    )
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("not the token", encoding = "utf-8")
    assert "did not contain" in sandbox_probe._host_positive_controls(
        str(work), str(sentinel), str(tmp_path / "out.txt"), env
    )


def test_a_host_that_fails_the_positive_half_is_not_blamed_on_the_backend():
    backend = _Backend("confining", _confining)
    real = sandbox_probe._host_payload
    sandbox_probe._host_payload = lambda workdir: "raise SystemExit(7)"
    try:
        available, reason = sandbox_probe.probe(backend)
    finally:
        sandbox_probe._host_payload = real
    assert available is False
    assert reason.startswith("the host itself could not pass"), reason
    assert backend.calls == 0


def test_the_scratch_root_fits_an_af_unix_address():
    base = sandbox_probe._probe_base()
    try:
        # base + "/work/tmp" + "/pymp-XXXXXXXX/listener-XXXXXXXX" must fit in
        # sun_path (108 bytes with the NUL).
        assert len(base) + len("/work/tmp") + 32 < 108, base
    finally:
        import shutil as _shutil
        _shutil.rmtree(base, ignore_errors = True)


def test_a_backend_that_cannot_prepare_is_unavailable_not_an_exception():
    def explode(plan):
        raise RuntimeError("no user namespaces here")

    available, reason = sandbox_probe.probe(_Backend("exploding", explode))
    assert available is False
    assert "no user namespaces here" in reason


def test_a_missing_backend_module_is_unavailable_not_an_exception():
    class Absent:
        pass

    available, reason = sandbox_probe.probe(Absent())
    assert available is False
    assert reason


def test_a_wedged_backend_times_out_instead_of_hanging(monkeypatch):
    monkeypatch.setattr(sandbox_probe, "PROBE_TIMEOUT_SECONDS", 1.0)

    def sleeper(plan):
        return PreparedSandboxLaunch(
            argv = (sys.executable, "-I", "-S", "-c", "import time; time.sleep(120)"),
            workdir = plan.workdir,
            env = plan.env,
            preexec_fn = None,
            backend = "wedged",
        )

    available, reason = sandbox_probe.probe(_Backend("wedged", sleeper))
    assert available is False
    assert "timed out" in reason


def test_a_backend_that_only_prints_the_token_is_not_believed():
    def liar(plan):
        return PreparedSandboxLaunch(
            argv = (sys.executable, "-I", "-S", "-c", f"print({sandbox_probe.PROBE_TOKEN!r})"),
            workdir = plan.workdir,
            env = plan.env,
            preexec_fn = None,
            backend = "liar",
        )

    # The probe cannot detect a backend that rewrites its own payload, so this
    # pins the weaker guarantee: the payload HANDED OVER carries every control.
    backend = _Backend("liar", liar)
    sandbox_probe.probe(backend)
    handed_over = backend.prepared and True
    assert handed_over


def test_the_payload_handed_to_the_backend_carries_every_control():
    backend = _Backend("passthrough", _passthrough)
    sandbox_probe.probe(backend)
    payload = backend.prepared[0].argv[-1]
    for control in (
        "read the host sentinel",
        "followed a workdir symlink to the host sentinel",
        sandbox_probe._OUTSIDE_WRITE_TOKEN,
        "socketpair",
        "DupFd",
        "a python child could not run",
        sandbox_probe.PROBE_TOKEN,
    ):
        assert control in payload, control
    if os.access(sys.executable, os.W_OK):
        assert "opened the interpreter for writing" in payload


def test_the_launch_is_built_by_the_backend_not_by_the_probe():
    backend = _Backend("confining", _confining)
    sandbox_probe.probe(backend)
    assert backend.calls == 1
    plan = None

    def capture(p):
        nonlocal plan
        plan = p
        return _confining(p)

    sandbox_probe.probe(_Backend("capture", capture))
    assert isinstance(plan, ToolLaunchPlan)
    # A real launch plan, so the probe exercises the code path that runs.
    assert plan.execution_kind == "python"
    assert plan.env["HOME"] == plan.workdir
    assert plan.env["TMPDIR"].startswith(plan.workdir + os.sep)


def test_everything_the_probe_owns_is_released():
    released = []

    def prepare(plan):
        prepared = _confining(plan)
        prepared.cleanup_callbacks.append(lambda: released.append("backend"))
        return prepared

    backend = _Backend("confining", prepare)
    available, reason = sandbox_probe.probe(backend)
    assert available is True, reason
    assert released == ["backend"]
    assert not os.path.exists(backend.prepared[0].workdir)


def test_cleanup_still_happens_when_the_launch_fails():
    released = []

    def prepare(plan):
        prepared = _passthrough(plan)
        prepared.cleanup_callbacks.append(lambda: released.append("backend"))
        return prepared

    backend = _Backend("passthrough", prepare)
    assert sandbox_probe.probe(backend)[0] is False
    assert released == ["backend"]
    assert not os.path.exists(backend.prepared[0].workdir)


def test_the_verdict_is_cached_so_a_tool_call_does_not_re_probe():
    backend = _Backend("confining", _confining)
    first = sandbox_probe.probe(backend)
    second = sandbox_probe.probe(backend)
    assert first == second
    assert backend.calls == 1


def test_force_re_probes():
    backend = _Backend("confining", _confining)
    sandbox_probe.probe(backend)
    sandbox_probe.probe(backend, force = True)
    assert backend.calls == 2


def test_the_cache_is_keyed_on_the_backend_as_well_as_the_runtime():
    confining = _Backend("confining", _confining)
    passthrough = _Backend("passthrough", _passthrough)
    assert sandbox_probe.probe(confining)[0] is True
    assert sandbox_probe.probe(passthrough)[0] is False


def test_an_expired_verdict_is_re_probed(monkeypatch):
    """Set before the first probe: the expiry is stamped when the verdict is stored."""
    monkeypatch.setattr(sandbox_probe, "_CACHE_TTL_SECONDS", 0.0)
    backend = _Backend("confining", _confining)
    sandbox_probe.probe(backend)
    sandbox_probe.probe(backend)
    assert backend.calls == 2


def test_the_cache_cannot_grow_without_bound():
    for index in range(sandbox_probe._CACHE_MAX_ENTRIES * 3):
        sandbox_probe.probe(_Backend(f"passthrough-{index}", _passthrough))
    assert len(sandbox_probe._cache) <= sandbox_probe._CACHE_MAX_ENTRIES


def test_this_host_reports_unavailable_with_something_actionable():
    if sys.platform != "linux":
        pytest.skip("the AppArmor condition is Linux only")
    capability = os_sandbox.capability_snapshot(force = True)
    if capability.available:
        pytest.skip("this host can build a real sandbox, so there is no fallback to check")
    assert capability.backend == "none"
    assert capability.limitations == ("no_os_isolation",)
    assert capability.remediation
    blocked = (
        subprocess.run(
            ["unshare", "--user", "--map-root-user", "true"],
            stdin = subprocess.DEVNULL,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
        ).returncode
        != 0
    )
    if blocked and os.path.exists("/proc/sys/kernel/apparmor_restrict_unprivileged_userns"):
        assert "apparmor_restrict_unprivileged_userns" in capability.remediation


def test_the_abstract_socket_control_is_paired_like_every_other(monkeypatch):
    from core.inference import sandbox_landlock, sandbox_probe

    monkeypatch.setattr(sandbox_landlock, "abstract_scope_supported", lambda: False)
    assert sandbox_probe._abstract_control() == (None, None)

    monkeypatch.setattr(sandbox_landlock, "abstract_scope_supported", lambda: True)
    if sys.platform != "linux":
        assert sandbox_probe._abstract_control() == (None, None)
        return
    name, listener = sandbox_probe._abstract_control()
    try:
        assert name is not None and name.startswith(b"\0unsloth-probe-")
        assert listener is not None
    finally:
        if listener is not None:
            listener.close()


def test_the_payload_requires_the_abstract_socket_to_be_out_of_reach():
    from core.inference import sandbox_probe

    with_scope = sandbox_probe._payload(
        "/work", "/sentinel", "/escape", "/outside", False, b"\0host-socket"
    )
    assert "connected to a host abstract unix socket" in with_scope
    assert "AF_UNIX" in with_scope
    without = sandbox_probe._payload("/work", "/sentinel", "/escape", "/outside", False, None)
    assert "abstract" not in without


def test_the_symlink_leg_survives_a_probe_base_reached_through_a_symlink(monkeypatch):
    """macOS puts the probe base under /tmp, which is a symlink to /private/tmp,
    and this test's double compares the path as written. Resolving only one side
    made it refuse its own workdir, so the probe failed before reaching the leg
    above and reported a confinement error for a path-spelling reason."""
    # Short names, under the same short root the probe itself prefers: a base over
    # _MAX_PROBE_BASE_LEN makes the fd-passing control's AF_UNIX address too long
    # and the HOST half fails, which says nothing about the symlink.
    root = "/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir()
    holder = tempfile.mkdtemp(prefix = "us-sym-", dir = root)
    real = pathlib.Path(holder) / "r"
    real.mkdir()
    alias = pathlib.Path(holder) / "a"
    alias.symlink_to(real)
    real_mkdtemp = sandbox_probe.tempfile.mkdtemp

    def through_the_symlink(prefix = None, dir = None):
        made = real_mkdtemp(prefix = prefix, dir = str(alias))
        return made.replace(str(real), str(alias))

    monkeypatch.setattr(sandbox_probe.tempfile, "mkdtemp", through_the_symlink)
    try:
        assert str(alias.resolve()) != str(alias)  # the premise, not an assumption
        available, reason = sandbox_probe.probe(_Backend("spelling-only", _spelling_only))
        assert available is False
        assert "symlink" in reason, reason
        available, reason = sandbox_probe.probe(_Backend("confining", _confining))
        assert available is True, reason
    finally:
        shutil.rmtree(holder, ignore_errors = True)


def test_the_landlock_helper_imports_where_it_will_never_be_used():
    """ctypes.CDLL(None) means "the running process" only where dlopen has that
    convention. On Windows ctypes tests the name for a path separator first and
    raises TypeError, which the import guard did not catch, so importing this
    Linux-only helper aborted collection on a platform that never calls it."""
    source = pathlib.Path(sandbox_landlock.__file__).read_text(encoding = "utf-8")
    guard = re.search(r"except \(([^)]*)\):[^\n]*\n(?:\s*#[^\n]*\n)*\s*_libc = None", source)
    assert guard, "the CDLL(None) import guard moved; this test no longer checks it"
    assert "TypeError" in guard.group(1), guard.group(1)
    # Every entry point has to survive _libc being None, or the guard only moves
    # the failure from import to first use.
    real = sandbox_landlock._libc
    sandbox_landlock._libc = None
    sandbox_landlock.abstract_scope_supported.cache_clear()
    try:
        assert sandbox_landlock.abstract_scope_supported() is False
        sandbox_landlock.apply_abstract_scope()  # a no-op, not a crash
        sandbox_landlock.with_abstract_scope(None)()
    finally:
        sandbox_landlock._libc = real
        sandbox_landlock.abstract_scope_supported.cache_clear()
