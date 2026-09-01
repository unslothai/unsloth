# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The freeze reporter must not answer confidently and wrongly.

`studio/scripts/unsloth_freeze_report.py` exists to end a guessing game: someone whose
interface freezes runs it once and gets a verdict per candidate workaround. That makes a
wrong verdict worse than no verdict, because the reader has no way to doubt it and spends
their next day on the candidate the report told them worked.

Each test here is one wrong verdict the script gave before, driven through `classify()`,
which is the whole oracle and is pure for exactly this reason:

  * a cold start, where the native watchdog answers a few probes before the webview has
    finished loading, read as a freeze at the moment startup finished;
  * a backend that stopped answering halfway through, read as a healthy run because both
    counters stopped together and neither could contradict the other;
  * a run with no watchdog at all, read as healthy although the freeze oracle needs two
    signals and only had one;
  * Ctrl-C, which the script prompts as "skips to the next candidate", scored as a real
    measurement of the partial window.

Plus the two ways a run can be measured against the wrong thing at all: an inherited
workaround variable that silently makes the control not a control, and a heartbeat counted
from a single log path that the backend's access log deduplicates away.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "studio" / "scripts" / "unsloth_freeze_report.py"


def _load():
    spec = importlib.util.spec_from_file_location("unsloth_freeze_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


freeze = _load()


def verdict(samples, **kwargs):
    """classify() with the arguments of an ordinary run, overridable one at a time."""
    n_mon = samples[-1][1] if samples else 0
    n_live = samples[-1][2] if samples else 0
    args = dict(
        samples = samples,
        n_mon = n_mon,
        n_live = n_live,
        exited = None,
        ran_for = samples[-1][0] if samples else 0,
        interrupted = False,
        preflight = "desktop_preflight completed disposition=OwnedReady port=8888 in 40ms",
        shell_started = True,
        has_display = True,
        warmup = 90,
    )
    args.update(kwargs)
    return freeze.classify(**args)


def healthy_samples(end = 240, warmup_lag = 60):
    """A run that is fine: the watchdog answers from the start, the webview joins once it
    has loaded, and from then on both keep going to the end of the window."""
    samples, mon, live = [], 0, 0
    for t in range(15, end + 1, 15):
        live += 1
        if t >= warmup_lag:
            mon += 1
        samples.append((t, mon, live))
    return samples


def test_cold_start_is_not_a_freeze():
    """The webview polls late, not never. Comparing samples from before the warmup
    boundary made every healthy candidate FROZE at about the moment it started up."""
    assert verdict(healthy_samples()).startswith("OK")


def test_a_real_stall_after_warmup_is_still_caught():
    """The warmup exclusion must not blunt the thing the script is for."""
    samples, mon, live = [], 0, 0
    for t in range(15, 241, 15):
        live += 1
        if 60 <= t <= 120:
            mon += 1
        samples.append((t, mon, live))
    assert verdict(samples).startswith("FROZE")
    assert "135s" in verdict(samples)


def test_both_loops_going_quiet_is_not_healthy():
    """Backend gone at 120s, shell still up. Both counters freeze together, so no
    comparison between them can fail, and the totals from the first half are large enough
    that every other branch declines: this used to fall through to OK."""
    samples, mon, live = [], 0, 0
    for t in range(15, 241, 15):
        if t <= 120:
            mon += 1
            live += 1
        samples.append((t, mon, live))
    assert verdict(samples).startswith("NO SIGNAL")


def test_no_watchdog_is_a_measurement_failure_not_a_pass():
    """The oracle is "watchdog ticking, interface silent". With no watchdog the script
    cannot tell a frozen interface from a healthy one, so it must not claim either."""
    samples = [(t, t // 15, 0) for t in range(15, 241, 15)]
    assert verdict(samples, n_live = 0).startswith("NO SIGNAL")


def test_interrupted_candidate_is_skipped_not_scored():
    """Ctrl-C is documented as skipping the candidate. Scoring the truncated window puts
    an OK (or a FROZE) into the summary for a run nobody measured."""
    result = verdict(healthy_samples(end = 120), interrupted = True, ran_for = 120)
    assert result.startswith("SKIPPED")
    assert "120s" in result


def test_clean_exit_late_in_the_window_is_not_a_single_instance_handoff():
    """Immediate clean exit means another copy took over; a clean exit at 200s means the
    window was closed, and saying "another copy is already running" sends the reporter
    hunting a copy that is not there."""
    assert verdict(healthy_samples(), exited = 0, ran_for = 200).startswith("ENDED EARLY")
    assert verdict([], exited = 0, ran_for = 15).startswith("SKIPPED")


def test_candidate_env_drops_an_inherited_workaround():
    """The reporter has already been told to try these by hand, so one is very likely
    still exported. Overlaying on top of it runs the control through the workaround and
    the report then says the control was fine."""
    base = {"GDK_BACKEND": "x11", "PATH": "/usr/bin", "DISPLAY": ":0"}
    control = freeze.candidate_env(base, {})
    assert "GDK_BACKEND" not in control
    assert control["PATH"] == "/usr/bin"

    candidate = freeze.candidate_env(base, {"WEBKIT_DISABLE_COMPOSITING_MODE": "1"})
    assert "GDK_BACKEND" not in candidate
    assert candidate["WEBKIT_DISABLE_COMPOSITING_MODE"] == "1"

    x11 = freeze.candidate_env(base, {"GDK_BACKEND": "x11"})
    assert x11["GDK_BACKEND"] == "x11"


def test_display_check_reads_the_candidate_environment():
    """With GDK_BACKEND stripped from the control, the control's display question is
    "is there any display", not "is there an X display" inherited from the caller."""
    assert freeze._has_display({"WAYLAND_DISPLAY": "wayland-0"})
    assert not freeze._has_display({"GDK_BACKEND": "x11", "WAYLAND_DISPLAY": "wayland-0"})
    assert freeze._has_display({"GDK_BACKEND": "x11", "DISPLAY": ":0"})


def test_the_heartbeat_survives_the_backend_keeping_its_suppressors_on():
    """The widened access log is a request, not a guarantee: a run that attached to a
    backend it did not start never delivered those variables, so that backend still
    collapses the UI liveness group into one shared 10s bucket and writes down whichever
    member won it (studio/backend/loggers/handlers.py). Counting the whole group rather
    than one path is what keeps a count above zero in that case, and the verdict then
    lands on NO SIGNAL through the branch above rather than on a wrong FROZE."""
    log = (
        '127.0.0.1 "GET /api/inference/images/status" 200\n'
        '127.0.0.1 "GET /api/inference/audio/stt/status" 200\n'
    )
    assert len(freeze.INTERFACE.findall(log)) == 2


def test_a_silent_interface_is_not_a_freeze_when_silence_proves_nothing():
    """The scenario that defeated the previous fix, end to end through the verdict.

    Every repeating poll the previous heartbeat matched is behind a user preference, and
    the loaded-model ones are behind one that is OFF until somebody turns it on
    (show-loaded-models-pref.ts: `localStorage.getItem(KEY) === "true"`). Turn the API
    monitor off in Settings on top of that and the whole group goes quiet on an app that
    is working perfectly, while /api/liveness carries on because the native shell owns it.
    The old classifier read exactly that as "the interface never polled at all while the
    app kept running" and told the reporter their app froze, for every candidate.

    A count of zero cannot distinguish that from a real freeze, so the only correct answer
    is that there was nothing to measure.
    """
    healthy_watchdog = [(t, 0, t // 15) for t in range(15, 241, 15)]
    result = verdict(healthy_watchdog, n_mon = 0)
    assert not result.startswith("FROZE")
    assert result.startswith("NO SIGNAL")


def test_the_heartbeat_includes_a_poll_no_preference_can_switch_off():
    """So that a zero above is rare rather than routine.

    use-export-runtime-lifecycle.ts polls /api/export/status every 5s from an effect with
    an empty dependency list, mounted at the app root on every route, gated on nothing but
    hasAuthToken(). There is no setting for it, and unlike every other poll in the app it
    has no document.hidden check either, so a minimised window keeps it going.
    """
    assert freeze.INTERFACE.findall('127.0.0.1 "GET /api/export/status" 200')
    # The preference-gated polls still count when they are running:
    optional = (
        '127.0.0.1 "GET /api/inference/monitor" 200\n'
        '127.0.0.1 "GET /api/inference/status" 200\n'
        '127.0.0.1 "GET /api/inference/images/status" 200\n'
        '127.0.0.1 "GET /api/inference/video/status" 200\n'
        '127.0.0.1 "GET /api/inference/audio/stt/status" 200\n'
    )
    assert len(freeze.INTERFACE.findall(optional)) == 5
    # /api/auth/status is navigation-driven with a 30s TTL (app/auth-guards.ts), never a timer.
    # Scoring it flatlines the moment the reporter stops clicking, which is the false FROZE this test exists to keep
    # out.
    assert not freeze.INTERFACE.findall('"GET /api/auth/status" 200')
    # The watchdog stays its own signal; it must not be swept into the interface count.
    assert not freeze.INTERFACE.findall('"GET /api/liveness" 200')


def test_the_run_widens_the_access_log_so_the_heartbeat_is_written_down():
    """The heartbeat above is invisible by default: /api/export/status is in
    _QUIET_SUCCESS_PATHS, so the backend drops its 2xx line outright, and the loaded-model
    polls share one 10s dedup bucket. Both suppressors are off when the two window
    variables are 0, which is what --verbose sets."""
    env = freeze.candidate_env({"PATH": "/usr/bin"}, {})
    assert env["UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS"] == "0"
    assert env["UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS"] == "0"
    # It is a logging window, not a rendering setting: it must not vary by candidate.
    for _, extra, _ in freeze.CANDIDATES:
        assert freeze.candidate_env({}, extra)["UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS"] == "0"


def test_control_is_not_pinned_by_an_override_the_candidates_never_name():
    """linux_webkit.rs returns PreserveEnvironment on an inherited
    WEBKIT_DISABLE_DMABUF_RENDERER or WEBKIT_DMABUF_RENDERER_FORCE_SHM, honours
    WEBKIT_FORCE_DMABUF_RENDERER as the NVIDIA patch's opt-out, and reads
    UNSLOTH_WEBKIT_RENDERER_WORKAROUND as its own claim on values it set itself, and takes
    UNSLOTH_WEBKIT_DISABLE_COMPOSITING as an instruction either way. None of them are in
    CANDIDATES, so a set derived from CANDIDATES leaves them active and they pin every
    launch including the control."""
    base = {
        "WEBKIT_DISABLE_DMABUF_RENDERER": "1",
        "WEBKIT_DMABUF_RENDERER_FORCE_SHM": "1",
        "WEBKIT_FORCE_DMABUF_RENDERER": "1",
        "UNSLOTH_WEBKIT_RENDERER_WORKAROUND": "WEBKIT_DISABLE_DMABUF_RENDERER",
        "UNSLOTH_WEBKIT_DISABLE_COMPOSITING": "1",
        "GDK_BACKEND": "x11",
        "PATH": "/usr/bin",
    }
    control = freeze.candidate_env(base, {})
    for name in base:
        if name != "PATH":
            assert name not in control, f"{name} still pins the control"
    assert control["PATH"] == "/usr/bin"
    # A candidate still gets its own value, and still sheds everything else.
    shm = freeze.candidate_env(base, {"WEBKIT_DISABLE_COMPOSITING_MODE": "1"})
    assert shm["WEBKIT_DISABLE_COMPOSITING_MODE"] == "1"
    assert "WEBKIT_DMABUF_RENDERER_FORCE_SHM" not in shm
    assert "UNSLOTH_WEBKIT_RENDERER_WORKAROUND" not in shm


def test_the_app_marker_is_recorded_so_a_stale_claim_is_visible():
    """UNSLOTH_WEBKIT_RENDERER_WORKAROUND decides whether the app reads a renderer
    variable as its own output or as an instruction, so a report that omits it cannot
    explain a launch that preserved the environment."""
    import inspect
    assert "UNSLOTH_WEBKIT_RENDERER_WORKAROUND" in inspect.getsource(freeze.exec_env)


def test_every_setting_the_app_reads_is_one_the_reporter_clears():
    """The list above is hand-maintained, so it goes stale the moment linux_webkit.rs learns
    a new UNSLOTH_WEBKIT_* setting: an inherited value pins every launch including the
    control, and the report reads clean without having compared anything. Reading the names
    back out of the Rust, rather than restating them, is what makes adding one fail here."""
    import re

    source = (REPO_ROOT / "studio" / "src-tauri" / "src" / "linux_webkit.rs").read_text(
        encoding = "utf-8"
    )
    settings = set(re.findall(r'"(UNSLOTH_WEBKIT_[A-Z_]+)"', source))
    assert settings, "no settings found; the pattern above stopped matching the Rust"
    missing = sorted(settings - set(freeze.RENDERER_OVERRIDE_VARS))
    assert missing == [], (
        f"{missing} steer the app but survive candidate_env(), so an inherited value pins "
        f"every launch including the control. Add them to RENDERER_OVERRIDE_VARS."
    )


def test_ports_cover_the_range_the_desktop_actually_uses():
    """desktop_candidate_ports() walks 8888..=8908. A leftover backend on any port outside
    the checked set is neither stopped nor waited for, and the next candidate adopts it."""
    assert freeze.PORTS[0] == 8888
    assert freeze.PORTS[-1] == 8908
    assert 8889 in freeze.PORTS


def test_renderer_workaround_comes_from_the_app_not_from_proc():
    """/proc/<pid>/environ is the environment as it was at execve; the app applies its
    renderer choice with set_var afterwards, so that file can never show it."""
    shell_out = (
        "12:00:00 [INFO] Unsloth desktop app starting\n"
        "12:00:00 [INFO] NVIDIA on Wayland with an AppImage that cannot load GLES; "
        "set WEBKIT_DISABLE_DMABUF_RENDERER=1 for WebKitGTK compatibility\n"
    )
    assert freeze.renderer_applied(shell_out) == {"WEBKIT_DISABLE_DMABUF_RENDERER": "1"}
    assert freeze.renderer_applied("no renderer line here") == {}


def test_shell_started_marker_separates_a_dead_shell_from_a_dead_backend():
    """ "The desktop shell never started" is the one verdict that tells the reporter they
    ran the wrong program, so it must not be reached by a shell that did start."""
    started = "12:00:00 [INFO] Unsloth desktop app starting"
    assert freeze.SHELL_STARTED.search(started)
    silent = verdict([(15, 0, 0), (30, 0, 0)], n_mon = 0, n_live = 0, preflight = "", shell_started = False)
    assert "the desktop shell never started" in silent
    live_shell = verdict(
        [(15, 0, 0), (30, 0, 0)], n_mon = 0, n_live = 0, preflight = "", shell_started = True
    )
    assert "the desktop shell never started" not in live_shell


def test_bare_relative_executable_is_resolved_before_launch(tmp_path, monkeypatch):
    """Popen looks a slashless argument up on PATH, not in the current directory, so
    `Unsloth.AppImage` typed in ~/Downloads passed validation and then failed to launch."""
    app = tmp_path / "Unsloth.AppImage"
    app.write_text("#!/bin/sh\n")
    app.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    resolved = freeze.resolve_command(["Unsloth.AppImage"])
    assert resolved == [str(app)]
    assert Path(resolved[0]).is_absolute()
    assert freeze.resolve_command(["definitely-not-installed-xyz"]) is None


def test_candidate_rationale_survives_a_no_signal_result():
    """The reason a candidate was tried is a field of the report. A no-signal run used to
    overwrite it with the measurement-failure explanation, so the one result that most
    needs context lost it."""
    import inspect

    source = inspect.getsource(freeze.run_candidate)
    assert '"why": why' in source
    body = source[source.index("def run_candidate") :]
    assert "\n        why = " not in body and "\n    why = " not in body


def test_busy_port_is_not_taken_as_permission_to_stop_a_live_studio(monkeypatch):
    """stop_leftover_backend() cannot tell an orphan from the backend serving the Unsloth
    the reporter has open, so an unattended run must refuse rather than SIGTERM it."""
    monkeypatch.delenv("UNSLOTH_FREEZE_STOP_RUNNING", raising = False)
    monkeypatch.setattr(freeze.sys, "stdin", None)
    assert freeze.confirm_stop_running_studio() is False
    monkeypatch.setenv("UNSLOTH_FREEZE_STOP_RUNNING", "1")
    assert freeze.confirm_stop_running_studio() is True


def test_main_aborts_instead_of_killing_a_running_studio(monkeypatch, capsys, tmp_path):
    """The end to end shape of the above: an Unsloth backend already answering, with nobody
    to ask, must reach no candidate at all, so nothing is ever killed."""
    launched = []
    monkeypatch.setattr(freeze, "studio_backend_pids", lambda: [4321])
    monkeypatch.setattr(freeze, "confirm_stop_running_studio", lambda: False)
    monkeypatch.setattr(freeze, "stop_leftover_backend", lambda: launched.append("killed"))
    monkeypatch.setattr(freeze, "run_candidate", lambda *a, **k: launched.append("ran"))
    app = tmp_path / "unsloth-studio"
    app.write_text("#!/bin/sh\n")
    app.chmod(0o755)
    monkeypatch.setattr(freeze.sys, "argv", ["unsloth_freeze_report.py", str(app)])
    # main() writes its report to the working directory. Without this, a regression that
    monkeypatch.chdir(tmp_path)
    assert freeze.main() == 2
    assert launched == []
    assert "no report was written" in capsys.readouterr().out
    assert not list(tmp_path.glob("unsloth-freeze-report-*.json"))


def test_a_listener_that_is_not_ours_does_not_refuse_the_run(monkeypatch, tmp_path):
    """The abort above must fire on something of ours to stop, not on any listener at all.

    Somebody else's Jupyter on 8888 needs nothing done about it: run.py falls back to the
    next free port, and stop_leftover_backend() would refuse to touch it anyway. Refusing
    on it turned every unattended invocation into an immediate exit 2, because
    confirm_stop_running_studio() returns False when there is nobody to ask.
    """
    ran = []
    monkeypatch.setattr(freeze, "studio_backend_pids", lambda: [])
    monkeypatch.setattr(freeze, "port_busy", lambda: True)
    monkeypatch.setattr(freeze, "confirm_stop_running_studio", lambda: pytest.fail("must not ask"))
    monkeypatch.setattr(freeze, "stop_leftover_backend", lambda: None)
    monkeypatch.setattr(
        freeze,
        "host_facts",
        lambda: {"session_type": "wayland", "desktop": "GNOME", "gpus": [], "nvidia_driver": ""},
    )
    monkeypatch.setattr(
        freeze,
        "run_candidate",
        lambda label, extra, why, cmd: ran.append(label)
        or {"candidate": label, "verdict": "OK: x"},
    )
    app = tmp_path / "unsloth-studio"
    app.write_text("#!/bin/sh\n")
    app.chmod(0o755)
    monkeypatch.setattr(freeze.sys, "argv", ["unsloth_freeze_report.py", str(app)])
    monkeypatch.chdir(tmp_path)
    assert freeze.main() == 0
    assert len(ran) == len(freeze.CANDIDATES)


def test_the_gate_and_the_cleanup_share_one_attribution_rule():
    """Two rules for "is this ours" is how the gate ends up refusing to run against a
    process the cleanup would then decline to touch. stop_leftover_backend() must ask
    studio_backend_pids() rather than re-deriving it."""
    import inspect

    assert "studio_backend_pids()" in inspect.getsource(freeze.stop_leftover_backend)
    # And the rule itself is the command line, not /proc/<pid>/exe:
    source = inspect.getsource(freeze.studio_backend_pids)
    assert "/cmdline" in source
    assert "{pid}/exe" not in source


def test_exactly_three_silent_intervals_is_already_stale():
    """STALE_AFTER is three poll intervals, and the boundary belongs on the stale side.

    A run whose counters last moved at 195s and ended at 240s has been silent for exactly
    45s, which is the three intervals the constant is picked to name ("so a single missed
    sample is not it"). The strict comparison let that exact case fall through to OK and
    reported a backend that had stopped being recorded as a healthy run.
    """
    samples, mon, live = [], 0, 0
    for t in range(15, 241, 15):
        if t <= 195:
            mon += 1
            live += 1
        samples.append((t, mon, live))
    assert samples[-1][0] - 195 == freeze.STALE_AFTER
    got = verdict(samples)
    assert got.startswith("NO SIGNAL"), got
    assert "45s" in got


def test_a_non_executable_appimage_is_refused_with_the_command_that_fixes_it(
    monkeypatch, tmp_path, capsys
):
    """An AppImage downloaded through a browser has no execute bit. Popen raises
    PermissionError on it and nothing between run_candidate() and the candidate loop
    catches it, so the run ended in a traceback with nothing measured and no report."""
    app = tmp_path / "Unsloth-Desktop.AppImage"
    app.write_text("#!/bin/sh\n")
    app.chmod(0o644)
    monkeypatch.setattr(freeze.sys, "argv", ["unsloth_freeze_report.py", str(app)])
    monkeypatch.setattr(
        freeze, "studio_backend_pids", lambda: pytest.fail("must not reach the port gate")
    )
    monkeypatch.chdir(tmp_path)
    assert freeze.main() == 2
    out = capsys.readouterr().out
    assert str(app) in out
    assert f"chmod +x {app}" in out
    assert not list(tmp_path.glob("unsloth-freeze-report-*.json"))


def test_discovery_prefers_an_appimage_that_can_actually_be_started(monkeypatch, tmp_path):
    """Downloads/ is where a non-executable copy lands, and it is usually the newest, so
    picking purely by mtime handed the launch a file that cannot run."""
    (tmp_path / "Downloads").mkdir()
    (tmp_path / "Applications").mkdir()
    runnable = tmp_path / "Applications" / "Unsloth-Desktop.AppImage"
    runnable.write_text("#!/bin/sh\n")
    runnable.chmod(0o755)
    downloaded = tmp_path / "Downloads" / "Unsloth-Desktop.AppImage"
    downloaded.write_text("#!/bin/sh\n")
    downloaded.chmod(0o644)
    import os as _os

    _os.utime(downloaded, (2_000_000_000, 2_000_000_000))
    monkeypatch.setattr(freeze, "HOME", tmp_path)
    monkeypatch.setattr(freeze.shutil, "which", lambda name: None)
    if Path("/usr/bin/unsloth-studio").is_file() or Path("/opt/Unsloth/unsloth-studio").is_file():
        pytest.skip("a system-wide Unsloth Desktop outranks the discovery being tested")
    assert freeze.find_desktop_app() == [str(runnable)]
    # With nothing runnable at all it still names the file, so the caller can say which one needs chmod rather than
    runnable.chmod(0o644)
    assert freeze.find_desktop_app() == [str(downloaded)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_one_flat_interval_that_recovers_is_not_a_freeze():
    """A freeze does not recover.

    The interface polls every 5s and the script samples every 15s, so a healthy window
    carries about three heartbeats. A delayed request, a pause, or a backend hiccup can
    still leave one window flat. Reporting FROZE on the first gap made a run that polled
    normally for the rest of its life unreadable, and no later evidence could clear it.
    """
    samples = [
        (0, 0, 0),
        (15, 3, 3),
        (30, 6, 6),
        (45, 9, 9),
        (60, 12, 12),
        (75, 12, 15),  # the gap: interface flat, watchdog carries on and it comes straight back
        (90, 15, 18),
        (105, 18, 21),
        (120, 21, 24),
    ]
    got = verdict(samples, warmup = 0)
    assert not got.startswith("FROZE"), got


def test_an_interface_that_stops_and_stays_stopped_is_still_a_freeze():
    """The other direction, so the fix above cannot be satisfied by never reporting."""
    samples = [
        (0, 0, 0),
        (15, 3, 3),
        (30, 6, 6),
        (45, 9, 9),
        (60, 9, 12),
        (75, 9, 15),
        (90, 9, 18),
        (105, 9, 21),
        (120, 9, 24),
    ]
    got = verdict(samples, warmup = 0)
    assert got.startswith("FROZE"), got


def test_a_stall_that_only_starts_as_the_window_closes_is_not_called_a_freeze():
    """ "It never polled again" is free when there is no "again" left.

    The interface goes flat on the final sample only. The rule that a freeze does not
    recover is satisfied vacuously, because the run ended before the interface had any
    chance to come back, so this reported FROZE on exactly the one delayed interval the
    comment above it calls insufficient. The run ending is not evidence the stall lasted.
    """
    samples = [
        (0, 0, 0),
        (15, 3, 3),
        (30, 6, 6),
        (45, 9, 9),
        (60, 12, 12),
        (75, 15, 15),
        (90, 15, 18),  # flat for the first time, and the window ends here
    ]
    got = verdict(samples, warmup = 0)
    assert not got.startswith("FROZE"), got
    # And it must not be waved through as healthy either: something was seen, it just was
    assert not got.startswith("OK"), got
    assert got.startswith("SUSPECT"), got
    assert "90s" in got
    assert "UNSLOTH_FREEZE_WINDOW" in got


def test_a_stall_watched_for_exactly_stale_after_is_a_freeze():
    """The other side of the same boundary, so the fix above cannot be met by refusing.

    The interface stops at 90s and the watchdog keeps answering to 135s, which is the three
    poll intervals STALE_AFTER exists to name. That is a watched stall, not a tail artefact.
    """
    samples = [
        (0, 0, 0),
        (15, 3, 3),
        (30, 6, 6),
        (45, 9, 9),
        (60, 12, 12),
        (75, 15, 15),
        (90, 15, 18),
        (105, 15, 21),
        (120, 15, 24),
        (135, 15, 27),
    ]
    assert samples[-1][0] - 90 == freeze.STALE_AFTER
    got = verdict(samples, warmup = 0)
    assert got.startswith("FROZE"), got
    assert "90s" in got


class _Clock:
    """A monotonic clock that only moves when the code under test sleeps, so a run that
    takes four minutes of wall time takes none here and its sample times are exact."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class _FakeApp:
    """A launched app that answers `poll()` alive for the first `dies_after` asks."""

    pid = 4242

    def __init__(
        self,
        dies_after,
        code = 9,
    ):
        self.polls, self.dies_after, self.code = 0, dies_after, code
        self.returncode = None

    def poll(self):
        self.polls += 1
        if self.polls > self.dies_after:
            self.returncode = self.code
            return self.code
        return None

    def wait(self, timeout = None):
        return self.returncode


def _drive_candidate(
    monkeypatch,
    tmp_path,
    log_at_sample,
    dies_after = 10_000,
):
    """One real run_candidate() over a scripted access log, one entry per 15s sample.

    Everything outside the script is faked and nothing is launched, so what is exercised is
    the loop, the cleanup and the handoff to classify() as they are actually written, rather
    than an argument list a test made up.
    """
    proc = _FakeApp(dies_after)
    # Before anything else: the cleanup SIGTERMs proc.pid's process group, and a fake pid is a real pid to the kernel.
    signalled = []
    monkeypatch.setattr(freeze.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(freeze.os, "killpg", lambda pgid, sig: signalled.append((pgid, sig)))
    monkeypatch.setattr(freeze, "time", _Clock())
    monkeypatch.setattr(freeze, "WARMUP", 0)
    monkeypatch.setattr(freeze, "WINDOW", 15 * len(log_at_sample))
    monkeypatch.setattr(freeze, "POLL_EVERY", 15)
    monkeypatch.setattr(freeze, "stop_leftover_backend", lambda: None)
    monkeypatch.setattr(freeze, "wait_for_leftover_backend_to_stop", lambda *a, **k: True)
    monkeypatch.setattr(freeze, "backend_offsets", lambda: {})
    monkeypatch.setattr(freeze, "exec_env", lambda pid: {})
    monkeypatch.setattr(freeze.subprocess, "Popen", lambda *a, **k: proc)
    calls = {"n": 0}

    def tail(_before):
        text = log_at_sample[min(calls["n"], len(log_at_sample) - 1)]
        calls["n"] += 1
        return text

    monkeypatch.setattr(freeze, "backend_tail", tail)
    monkeypatch.chdir(tmp_path)
    return freeze.run_candidate("control (no override)", {}, "baseline", ["/bin/true"])


def _log(
    heartbeats,
    watchdogs,
    session = 0,
):
    """A cumulative access log with that many of each request in it."""
    return (
        '127.0.0.1 "GET /api/export/status" 200\n' * heartbeats
        + '127.0.0.1 "GET /api/liveness" 200\n' * watchdogs
        + '127.0.0.1 "POST /api/auth/logout" 200\n' * session
    )


def test_an_exit_seen_only_by_the_cleanup_poll_is_still_recorded(monkeypatch, tmp_path):
    """The app can die in the gap between the loop's last poll and the cleanup's.

    That gap is seconds wide and the difference in meaning is the whole report: the samples
    up to it look like a healthy run, because they are the samples of a run that was healthy
    until it crashed. The cleanup saw the dead process, killed nothing, and left `exited` as
    None, so the classifier skipped both exit branches and judged the samples on their own,
    and the crash came back as "OK: the interface kept polling for the whole run".
    """
    # A log growing at a healthy rate throughout, so nothing in the samples hints at the crash and the verdict has to
    healthy = [_log(3 * n, 3 * n) for n in range(1, 5)]
    result = _drive_candidate(monkeypatch, tmp_path, healthy, dies_after = 4)
    assert result["samples"], "the loop must have run, or this proves nothing"
    assert result["exit_code"] == 9, "the cleanup poll saw it dead; that must be recorded"
    assert not result["verdict"].startswith("OK"), result["verdict"]
    assert "9" in result["verdict"]


def test_signing_out_midway_is_not_reported_as_a_freeze():
    """Losing the session stops the heartbeat as thoroughly as a freeze does.

    pollStatus() in use-export-runtime-lifecycle.ts opens with `if (!hasAuthToken()) return;`
    (:156) and the interval that calls it (:192) keeps firing regardless, so a session
    cleared mid-run stops /api/export/status while the native watchdog carries on. Round 2's
    fix does not help here: the heartbeat WAS heard first, and then stopped, which is exactly
    the shape the FROZE arm was narrowed to. A perfectly healthy login screen was reported
    as a freeze, for every candidate after the sign-out.

    What separates the two is that the webview went on making requests. It cannot do that if
    it is frozen.
    """
    samples = [
        (0, 0, 0),
        (15, 3, 3),
        (30, 6, 6),
        (45, 9, 9),
        (60, 12, 12),
        (75, 15, 15),
        (90, 15, 18),
        (105, 15, 21),
        (120, 15, 24),
        (135, 15, 27),
    ]
    frozen = verdict(samples, warmup = 0)
    assert frozen.startswith("FROZE"), "the stall itself is real and still reads as a freeze"
    got = verdict(samples, warmup = 0, session_at = 90)
    assert not got.startswith("FROZE"), got
    assert got.startswith("SIGNED OUT"), got
    assert "90s" in got


def test_sign_in_traffic_from_before_the_stall_does_not_clear_a_freeze():
    """The evidence is a request the webview made AFTER it went quiet. Every run has auth
    traffic at startup, and if that counted, the FROZE arm would never fire again."""
    samples = [
        (0, 0, 0),
        (15, 3, 3),
        (30, 6, 6),
        (45, 9, 9),
        (60, 12, 12),
        (75, 15, 15),
        (90, 15, 18),
        (105, 15, 21),
        (120, 15, 24),
        (135, 15, 27),
    ]
    got = verdict(samples, warmup = 0, session_at = 30)
    assert got.startswith("FROZE"), got
    # And a session cleared without any request reaching the backend is still indistinguishable from a freeze, which is
    assert verdict(samples, warmup = 0, session_at = None).startswith("FROZE")


def test_the_session_signal_is_the_webview_talking_not_the_shell():
    """/api/auth/desktop-login is posted by the native shell itself
    (src-tauri/src/desktop_auth.rs:194), so counting it would let the shell vouch for a
    webview that is not running, which is the one thing this signal must never do."""
    assert freeze.SESSION.findall('127.0.0.1 "POST /api/auth/logout" 200')
    assert freeze.SESSION.findall('127.0.0.1 "POST /api/auth/refresh" 401')
    assert freeze.SESSION.findall('127.0.0.1 "GET /api/auth/status" 200')
    assert freeze.SESSION.findall('127.0.0.1 "POST /api/auth/login" 200')
    assert not freeze.SESSION.findall('127.0.0.1 "POST /api/auth/desktop-login" 200')
    # And it is not a heartbeat: it must not be swept into the interface count, or a run
    assert not freeze.INTERFACE.findall('127.0.0.1 "GET /api/auth/status" 200')


def test_the_run_records_when_the_session_was_last_asked_about(monkeypatch, tmp_path):
    """The classifier cannot use a signal the run does not collect. This is the loop and the
    handoff to classify() as written, not an argument list invented by a test."""
    # Heartbeat for the first four samples, then a sign-out at 75s and silence after it, while the watchdog keeps going
    scripted = [
        _log(3, 3),
        _log(6, 6),
        _log(9, 9),
        _log(12, 12),
        _log(12, 15, session = 1),
        _log(12, 18, session = 1),
        _log(12, 21, session = 1),
        _log(12, 24, session = 1),
    ]
    result = _drive_candidate(monkeypatch, tmp_path, scripted)
    assert result["session_seen_at"] == 75, result["samples"]
    assert result["verdict"].startswith("SIGNED OUT"), result["verdict"]


def test_an_interface_first_heard_as_the_window_closes_is_not_a_measured_run():
    """The start of the series, the same way the end of it was wrong three times over.

    The backend can take most of the window to come up on a first run, and the interface
    cannot poll before it has a session either, so a reporter who signs in near the end
    produces a run whose heartbeat first moves in the last few samples. There is no flat
    interval anywhere (the counter only ever rises), nothing went quiet at the end, and the
    handful of polls from those last samples clear the ratio test, so the bottom line said
    the interface "kept polling for the whole run" about an interface watched for 30s.
    """
    samples, mon, live = [], 0, 0
    for t in range(15, 241, 15):
        live += 1
        if t >= 210:
            mon += 3
        samples.append((t, mon, live))
    assert samples[-1][1] * 3 >= samples[-1][2], "must clear the ratio test, as the real run did"
    got = verdict(samples, warmup = 0)
    assert not got.startswith("OK"), got
    assert got.startswith("SUSPECT"), got
    assert "210s" in got and "30s" in got


def test_an_interface_heard_early_enough_is_still_allowed_to_be_healthy():
    """The other side of it, so the check above cannot be satisfied by never saying OK.

    Warmup lag is normal: the native watchdog answers while the webview is still loading.
    Once the heartbeat has been watched for STALE_AFTER a delayed freeze would have shown,
    so a run that keeps polling to the end is what OK is for.
    """
    assert verdict(healthy_samples()).startswith("OK")
    late, mon, live = [], 0, 0
    for t in range(15, 241, 15):
        live += 1
        if t >= 195:
            mon += 3
        late.append((t, mon, live))
    assert freeze._first_heard(late, 1) == 195
    assert late[-1][0] - 195 == freeze.STALE_AFTER
    assert verdict(late, warmup = 0).startswith("OK"), verdict(late, warmup = 0)


def test_a_launch_that_fails_at_execve_does_not_end_the_whole_run(monkeypatch, tmp_path):
    """The execute bit says the kernel may try, not that the try works.

    A build for the wrong CPU, a truncated AppImage, a missing interpreter or a noexec
    mount all reach execve and fail there. Popen raises OSError, and the candidate loop
    catches only KeyboardInterrupt, so the first bad candidate ended the diagnostic in a
    traceback: nothing measured, no report written, and no line saying what went wrong.
    """
    app = tmp_path / "Unsloth-Desktop.AppImage"
    app.write_text("garbage, not an executable format\n")
    app.chmod(0o755)

    def refuse(*args, **kwargs):
        raise OSError(8, "Exec format error", str(app))

    monkeypatch.setattr(freeze.subprocess, "Popen", refuse)
    monkeypatch.setattr(freeze, "studio_backend_pids", lambda: [])
    monkeypatch.setattr(freeze, "port_busy", lambda: False)
    monkeypatch.setattr(freeze, "stop_leftover_backend", lambda: None)
    monkeypatch.setattr(freeze, "wait_for_leftover_backend_to_stop", lambda *a, **k: True)
    monkeypatch.setattr(
        freeze,
        "host_facts",
        lambda: {"session_type": "wayland", "desktop": "GNOME", "gpus": [], "nvidia_driver": ""},
    )
    monkeypatch.setattr(freeze.sys, "argv", ["unsloth_freeze_report.py", str(app)])
    monkeypatch.chdir(tmp_path)

    assert freeze.main() == 0
    written = list(tmp_path.glob("unsloth-freeze-report-*.json"))
    assert len(written) == 1, "the report is the only output of the run; it must be written"
    import json as _json

    results = _json.loads(written[0].read_text(encoding = "utf-8"))["results"]
    # Every candidate is still attempted and still accounted for.
    assert len(results) == len(freeze.CANDIDATES)
    for r in results:
        assert r["verdict"].startswith("CANNOT RUN"), r["verdict"]
        assert "Exec format error" in r["verdict"]
        # The report's own schema is unchanged, so a reader (and the summary) can treat this result like any other.
        assert r["samples"] == [] and r["interface_polls"] == 0
