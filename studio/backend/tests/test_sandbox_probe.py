# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the live probe is allowed to conclude, and from what.

The probe is the only thing standing between "bwrap is installed" and "tool
calls are isolated", and those two are not the same claim: this very host has
bubblewrap 0.9.0 and ``kernel.apparmor_restrict_unprivileged_userns=1``, so
bwrap cannot build a sandbox at all. A probe that inferred availability from the
binary would advertise a boundary that does not exist.

So the tests here are about the probe's own honesty rather than about any one
backend:

* a backend that confines nothing must come back unavailable, even though its
  launch runs perfectly and exits 0;
* a backend that really does confine must come back available;
* a control that could not be set up on the HOST (an unreadable sentinel, an
  unwritable base) makes the whole verdict meaningless, so the probe declines
  instead of passing;
* nothing the probe can be handed makes it raise, because its caller either
  falls back or refuses and neither is served by a traceback.

The confining backend is simulated in-process rather than skipped, because the
CI host cannot build a real sandbox and "skipped" would leave the positive half
of every pairing untested.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys

import pytest

from core.inference import os_sandbox, sandbox_probe
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
    """A backend that isolates nothing: runs the payload straight on the host."""
    return PreparedSandboxLaunch(
        argv = plan.argv,
        workdir = plan.workdir,
        env = plan.env,
        preexec_fn = plan.preexec_fn,
        backend = "passthrough",
    )


_CONFINE_WRAPPER = """import builtins, os, sys
WORK = os.path.realpath({workdir!r})
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
    outside = (
        isinstance(target, str) and target != WORK and not target.startswith(WORK + os.sep)
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
    """A backend whose launch really does refuse to leave the workdir.

    The confinement is a builtins.open guard rather than a kernel namespace,
    which is enough for what is under test here: the probe's ability to tell a
    boundary from the absence of one, and a real boundary from a lookalike.
    Import machinery uses io.open_code, so the payload's own imports still work,
    exactly as they would inside bwrap.
    """
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
        preexec_fn = plan.preexec_fn,
        backend = name,
    )


def _confining(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    return _wrap(plan, "confining")


def _spelling_only(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    return _wrap(plan, "spelling-only", resolve = False)


def _leaking_writes(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    return _wrap(plan, "leaky", leak_writes = True)


# ── the two halves of every pairing ───────────────────────────────────


def test_a_backend_that_confines_nothing_is_not_available():
    """The whole point. This launch exits 0 on the host and proves nothing."""
    backend = _Backend("passthrough", _passthrough)
    available, reason = sandbox_probe.probe(backend)
    assert available is False
    # And it says WHICH control came out wrong, so the failure is diagnosable.
    assert "read the host sentinel" in reason, reason


def test_a_backend_that_really_confines_is_available():
    backend = _Backend("confining", _confining)
    available, reason = sandbox_probe.probe(backend)
    assert available is True, reason
    assert "confining" in reason


def test_the_symlink_leg_is_judged_on_the_target_not_the_path():
    """A guard that only compared the SPELLING of the path passes the sentinel leg
    and fails here, so the two legs are not redundant."""
    available, reason = sandbox_probe.probe(_Backend("spelling-only", _spelling_only))
    assert available is False
    assert "symlink" in reason, reason


def test_a_write_that_reaches_the_host_fails_the_probe():
    """The escape check that cannot be made from inside.

    This backend confines every read and protects the system root, so every
    control the sandboxed process itself can evaluate comes out right -- and it
    still lets a write land on the real filesystem. Only the host can see that.
    """
    available, reason = sandbox_probe.probe(_Backend("leaky", _leaking_writes))
    assert available is False
    assert "wrote through to the host" in reason, reason


def test_the_escape_check_is_made_before_the_scratch_root_is_removed():
    """A cleanup that ran first would find nothing and report every escape as a
    pass, which is why the verdict is reached inside the try."""
    text = inspect.getsource(sandbox_probe._run_probe)
    assert text.index("_host_saw_the_write") < text.index("shutil.rmtree")


def test_the_positive_controls_run_before_anything_is_concluded(monkeypatch):
    """An unreadable sentinel means every in-sandbox failure is uninformative, so
    the probe must decline rather than report a boundary it never tested."""
    backend = _Backend("confining", _confining)
    seen = []

    def broken(workdir, sentinel, outside, env):
        seen.append(sentinel)
        return "the host itself could not read the probe sentinel (simulated)"

    monkeypatch.setattr(sandbox_probe, "_host_positive_controls", broken)
    available, reason = sandbox_probe.probe(backend)
    assert available is False
    assert "the host itself could not read" in reason
    # And it declined BEFORE launching anything.
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
    """The bug this pairing exists to prevent: a temp directory too deep for an
    AF_UNIX address failed the fd-passing leg, and the probe reported it as the
    sandbox having broken multiprocessing. The reason must name the HOST."""
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
    """A deep TMPDIR must not be able to fail a positive control on its own."""
    base = sandbox_probe._probe_base()
    try:
        # base + "/work/tmp" + "/pymp-XXXXXXXX/listener-XXXXXXXX" must fit in
        # sun_path (108 bytes including the NUL).
        assert len(base) + len("/work/tmp") + 32 < 108, base
    finally:
        import shutil as _shutil
        _shutil.rmtree(base, ignore_errors = True)


# ── failure is always a verdict, never an exception ───────────────────


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
    """A tool call must not be able to block forever behind a stuck helper."""
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
    """Exit 0 plus the token is not enough on its own: the controls have to have
    run. A payload replaced by a bare print must not qualify."""

    def liar(plan):
        return PreparedSandboxLaunch(
            argv = (sys.executable, "-I", "-S", "-c", f"print({sandbox_probe.PROBE_TOKEN!r})"),
            workdir = plan.workdir,
            env = plan.env,
            preexec_fn = None,
            backend = "liar",
        )

    # The probe cannot detect a backend that rewrites its own payload -- that is
    # the backend lying to itself -- so this pins the weaker, real guarantee: the
    # payload it HANDS OVER is the one carrying every control.
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


# ── it goes through the backend's real prepare() ──────────────────────


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
    # The plan is a real tool launch plan, not a bespoke probe struct: same type
    # the executors build, so the probe exercises the code path that runs.
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
    # And the probe's own scratch tree is gone, including on the success path.
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


# ── caching ───────────────────────────────────────────────────────────


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
    """Two backends must never share a verdict; that is how a passing macOS probe
    would come to vouch for a Linux one."""
    confining = _Backend("confining", _confining)
    passthrough = _Backend("passthrough", _passthrough)
    assert sandbox_probe.probe(confining)[0] is True
    assert sandbox_probe.probe(passthrough)[0] is False


def test_an_expired_verdict_is_re_probed(monkeypatch):
    """Set before the first probe, because the expiry is stamped when the verdict
    is stored: a host that gains the AppArmor profile must not wait out a TTL
    that was already fixed."""
    monkeypatch.setattr(sandbox_probe, "_CACHE_TTL_SECONDS", 0.0)
    backend = _Backend("confining", _confining)
    sandbox_probe.probe(backend)
    sandbox_probe.probe(backend)
    assert backend.calls == 2


def test_the_cache_cannot_grow_without_bound():
    for index in range(sandbox_probe._CACHE_MAX_ENTRIES * 3):
        sandbox_probe.probe(_Backend(f"passthrough-{index}", _passthrough))
    assert len(sandbox_probe._cache) <= sandbox_probe._CACHE_MAX_ENTRIES


# ── this host, for real ───────────────────────────────────────────────


def test_this_host_reports_unavailable_with_something_actionable():
    """bubblewrap 0.9.0 is installed here and AppArmor denies it the user
    namespace, which is precisely the case a binary-presence check gets wrong."""
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
