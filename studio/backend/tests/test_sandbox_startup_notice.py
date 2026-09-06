# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two operator-facing halves of core/inference/sandbox_startup.py.

Tools default to os_isolation_required and fail closed, so a host whose sandbox
is unavailable refuses every Python and Terminal call. The startup notice is the
only place that says so before the first refusal, and the warm probe keeps the
cold-host system-root scan (up to two minutes) off that first call.
"""

import threading
import time
from dataclasses import replace

import pytest

from core.inference import sandbox_startup
from core.inference.os_sandbox import SandboxCapability, _apparmor_userns_remediation

_UNAVAILABLE = SandboxCapability(
    backend = "linux-bubblewrap",
    qualified = False,
    reason = (
        "AppArmor restricts unprivileged user namespaces on this host "
        "(kernel.apparmor_restrict_unprivileged_userns=1): bwrap: setting up uid map: "
        "Permission denied"
    ),
    available = False,
    environment = "native_linux",
    protection_state = "unavailable",
    remediation = _apparmor_userns_remediation(),
)

_AVAILABLE = replace(
    _UNAVAILABLE,
    qualified = True,
    available = True,
    reason = "Bubblewrap qualified",
    protection_state = "protected",
    remediation = "",
)


@pytest.fixture(autouse = True)
def _clean_warmup_state():
    sandbox_startup.reset_sandbox_warmup_state()
    yield
    sandbox_startup.reset_sandbox_warmup_state()


# --- notice --------------------------------------------------------------------


def test_unavailable_capability_prints_the_full_notice(capsys):
    sandbox_startup.print_sandbox_startup_notice(_UNAVAILABLE)
    out = capsys.readouterr().out

    assert "OS isolation for tool calls is unavailable on this machine." in out
    # What it means under the fail-closed default, and the way out.
    assert "Python and Terminal tool calls refuse by default" in out
    assert "Limited or Full" in out
    # What was detected, and why it failed.
    assert "Detected: linux-bubblewrap backend, native_linux environment." in out
    assert f"Reason: {_UNAVAILABLE.reason}" in out
    # The remediation must survive verbatim: it carries a copy-pasteable AppArmor
    # profile whose own indentation matters.
    assert _UNAVAILABLE.remediation in out
    assert "/etc/apparmor.d/bwrap" in out


def test_available_capability_says_nothing(capsys):
    sandbox_startup.print_sandbox_startup_notice(_AVAILABLE)
    assert capsys.readouterr().out == ""
    assert sandbox_startup.format_sandbox_startup_notice(_AVAILABLE) == ""


def test_qualified_but_unavailable_still_warns():
    # available is authoritative; a qualified probe the identity pass marked
    # unavailable must not read as healthy.
    capability = replace(_AVAILABLE, available = False)
    assert sandbox_startup.format_sandbox_startup_notice(capability) != ""


def test_notice_stays_compact_and_plain():
    notice = sandbox_startup.format_sandbox_startup_notice(_UNAVAILABLE)
    lines = notice.splitlines()
    # The remediation is 8 lines on its own; the wrapper around it must stay small.
    assert len(lines) - len(_UNAVAILABLE.remediation.splitlines()) <= 6
    assert "\033[" not in notice
    assert "—" not in notice


def test_notice_never_raises_when_the_probe_fails(capsys, monkeypatch):
    def boom() -> SandboxCapability:
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(sandbox_startup, "_capability_snapshot", boom)
    sandbox_startup.print_sandbox_startup_notice()
    assert capsys.readouterr().out == ""


def test_notice_waits_for_an_in_flight_warm_probe(capsys, monkeypatch):
    """No interleaving: the notice joins the warm probe instead of racing it."""
    release = threading.Event()

    def slow() -> SandboxCapability:
        release.wait(10)
        return _UNAVAILABLE

    monkeypatch.setattr(sandbox_startup, "_capability_snapshot", slow)
    thread = sandbox_startup.start_sandbox_capability_warmup()
    assert thread is not None

    # Still probing: the notice gives up rather than stalling the banner.
    sandbox_startup.print_sandbox_startup_notice(wait = 0.1)
    assert capsys.readouterr().out == ""

    release.set()
    thread.join(10)
    assert not thread.is_alive()

    sandbox_startup.print_sandbox_startup_notice(wait = 5)
    assert "OS isolation for tool calls is unavailable" in capsys.readouterr().out


# --- warm probe ----------------------------------------------------------------


def test_warmup_returns_immediately_with_a_daemon_thread(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def slow() -> SandboxCapability:
        entered.set()
        release.wait(10)
        return _UNAVAILABLE

    monkeypatch.setattr(sandbox_startup, "_capability_snapshot", slow)

    started = time.perf_counter()
    thread = sandbox_startup.start_sandbox_capability_warmup()
    elapsed = time.perf_counter() - started

    try:
        assert thread is not None
        assert elapsed < 1.0, f"warm-up blocked startup for {elapsed:.2f}s"
        assert thread.daemon, "a live probe thread must not hold the process open"
        assert entered.wait(10), "the probe never ran"
        # A second call while the first is in flight reuses it.
        assert sandbox_startup.start_sandbox_capability_warmup() is thread
    finally:
        release.set()
        thread.join(10)


def test_repeated_startups_do_not_leak_threads(monkeypatch):
    calls = []

    def counted() -> SandboxCapability:
        calls.append(1)
        return _UNAVAILABLE

    monkeypatch.setattr(sandbox_startup, "_capability_snapshot", counted)

    before = threading.active_count()
    first = sandbox_startup.start_sandbox_capability_warmup()
    assert first is not None
    first.join(10)

    for _ in range(5):
        assert sandbox_startup.start_sandbox_capability_warmup() is None

    assert len(calls) == 1
    assert threading.active_count() <= before


def test_warmup_swallows_probe_failures_and_marks_done(monkeypatch):
    def boom() -> SandboxCapability:
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(sandbox_startup, "_capability_snapshot", boom)
    thread = sandbox_startup.start_sandbox_capability_warmup()
    assert thread is not None
    thread.join(10)
    assert not thread.is_alive()
    # Marked done even after a failure, so a restart does not spawn another.
    assert sandbox_startup.start_sandbox_capability_warmup() is None


# --- wiring --------------------------------------------------------------------


def test_banner_emits_the_notice_and_respects_disable_tools(monkeypatch, capsys):
    import run

    seen = []
    monkeypatch.setattr(
        sandbox_startup,
        "print_sandbox_startup_notice",
        lambda *a, **k: seen.append(True) or print("SANDBOX NOTICE"),
    )

    run._emit_sandbox_capability_notice(None)
    run._emit_sandbox_capability_notice(True)
    assert seen == [True, True]
    assert capsys.readouterr().out.count("SANDBOX NOTICE") == 2

    # --disable-tools: no tool can run, so the OS-isolation warning is noise.
    run._emit_sandbox_capability_notice(False)
    assert seen == [True, True]
    assert capsys.readouterr().out == ""


def test_startup_paths_call_the_notice():
    """Both banners (plain and secure) must reach it, or a whole launch mode is silent."""
    import inspect

    import run

    for fn in (run._emit_startup_output, run._emit_secure_startup_output):
        assert "_emit_sandbox_capability_notice" in inspect.getsource(fn)


def test_lifespan_starts_the_warm_probe():
    """Next to start_sandbox_recovery, so no tool call pays the cold scan."""
    import inspect

    import main

    source = inspect.getsource(main.lifespan)
    assert "start_sandbox_capability_warmup" in source
