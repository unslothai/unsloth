#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Collect evidence about an Unsloth Desktop interface freeze.

Run this instead of trying environment variables by hand. It launches Unsloth Desktop
several times, once per candidate workaround, and after each one says whether the interface
kept running or froze. It also records what your machine is, so the report is usable
without a further round of questions.

  python3 unsloth_freeze_report.py

Takes about 20 minutes. Nothing is uploaded: it writes one file and prints its path, and you
decide whether to send it.

How the freeze is detected
--------------------------
Two independent loops poll the backend, and they have different owners:

  the interface polls      the webview's own requests; only run while the interface is
                           alive. Led by /api/export/status, which is the one repeating
                           request the app makes with no preference, no panel and no
                           window-visibility check in front of it.
  /api/liveness            a native watchdog in the app; keeps running even if the
                           interface is dead

A freeze is the specific pattern where the watchdog keeps ticking and the interface goes
silent AFTER having been heard. An interface that was never heard from at all is reported
as NO SIGNAL rather than as a freeze: a count of zero is what a real freeze looks like,
but it is also what a missing session token looks like, and guessing between them is how
this script would tell somebody their app froze when it did not. For the same reason, an
interface that went silent but went on asking the backend about the session is reported as
SIGNED OUT: the heartbeat needs a session token, so signing out stops it without anything
having frozen.

That is measured here rather than guessed at, which is why this is worth running even
though you can already see the freeze with your own eyes: it distinguishes "the interface
stopped" from "the whole app died" from "the app is fine and it looked stuck", and it says
which of those each workaround produces.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HOME = Path.home()
STUDIO = HOME / ".unsloth" / "studio"
BACKEND_LOGS = STUDIO / "logs"
# The interface's heartbeat. /api/export/status is the load-bearing member: it is polled every 5s
# from an effect with an EMPTY dependency list, mounted unconditionally at the app root, with no
# setting and no `document.hidden` check. Every OTHER poll here is behind something the user
# controls, so a hit from one is good evidence the webview is alive but its SILENCE is not a
# symptom; reading it as one made an earlier version report FROZE on a stock install.
# /api/auth/status is deliberately absent: it fetches on navigation with a 30s TTL, never on a
# timer, so scoring it would invent a freeze out of somebody sitting still. The native shell
# requests only /api/liveness and /api/health, so every hit here comes from the webview.
INTERFACE = re.compile(
    r"/api/(?:export/status|inference/monitor|inference/status"
    r"|inference/images/status|inference/video/status|inference/audio/stt/status)"
)
LIVENESS = re.compile(r"/api/liveness")
# The webview's sign-in traffic.
SESSION = re.compile(r"/api/auth/(?:status|login|logout|refresh)\b")
# Printed by the desktop shell (main.rs) before anything else, through a stderr logger
SHELL_STARTED = re.compile(r"Unsloth desktop app starting")
# `{reason}; set VAR=1 VAR2=1 for WebKitGTK compatibility`, the app's own record
RENDERER_APPLIED = re.compile(r"set ((?:[A-Za-z_][A-Za-z_0-9]*=1\s*)+)for WebKitGTK compatibility")

# Overridable so CI can exercise this script end to end in a couple of minutes.
WARMUP = int(os.environ.get("UNSLOTH_FREEZE_WARMUP", 90))
WINDOW = int(os.environ.get("UNSLOTH_FREEZE_WINDOW", 150))
POLL_EVERY = 15
# Both counters flat for this long, with the app still running, is not a healthy run: the
# backend stopped being recorded. Three poll intervals, so a single missed sample is not it.
STALE_AFTER = 3 * POLL_EVERY
# desktop_candidate_ports() in studio/src-tauri/src/desktop_backend_owner.rs: the shell
# walks 8888..=8908 and takes the first free one, so checking two of them would miss a
# leftover backend on any of the other nineteen and hand the next candidate an orphan.
PORTS = tuple(range(8888, 8909))

# Each entry is (label, extra environment, why it is being tried).
CANDIDATES = [
    ("control (no override)", {}, "baseline; everything below is compared against this"),
    (
        "WEBKIT_DISABLE_COMPOSITING_MODE=1",
        {"WEBKIT_DISABLE_COMPOSITING_MODE": "1"},
        "turns off accelerated compositing entirely, a level above the buffer transport",
    ),
    (
        "__NV_DISABLE_EXPLICIT_SYNC=1",
        {"__NV_DISABLE_EXPLICIT_SYNC": "1"},
        "disables explicit sync on the NVIDIA driver's Wayland path",
    ),
    (
        "GDK_BACKEND=x11",
        {"GDK_BACKEND": "x11"},
        "runs the interface through XWayland instead of Wayland",
    ),
]


CANDIDATE_VARS = tuple(sorted({k for _, extra, _ in CANDIDATES for k in extra}))

# Every renderer override the APP reads.
RENDERER_OVERRIDE_VARS = (
    "GDK_BACKEND",
    "UNSLOTH_WEBKIT_DISABLE_COMPOSITING",
    "UNSLOTH_WEBKIT_RENDERER_WORKAROUND",
    "WEBKIT_DISABLE_DMABUF_RENDERER",
    "WEBKIT_DMABUF_RENDERER_FORCE_SHM",
    "WEBKIT_FORCE_DMABUF_RENDERER",
)
CLEARED_VARS = tuple(sorted(set(CANDIDATE_VARS) | set(RENDERER_OVERRIDE_VARS)))

# Applied to the app this script launches, and to nothing else.
MEASUREMENT_ENV = {
    "UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS": "0",
    "UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS": "0",
}


def candidate_env(base: dict, extra: dict) -> dict:
    """The environment for one candidate: the caller's, minus every renderer override,
    plus the logging this script needs, plus this candidate's own variables.

    The subtraction is the point. These are exactly the variables the reporter has already
    been asked to try by hand, so one of them is quite likely still exported in the shell
    this script is run from. Overlaying on top of that leaves the control running with the
    workaround still applied and every comparison in the report meaningless: an inherited
    GDK_BACKEND=x11 puts all four launches through XWayland, and the report says the
    control was fine. It clears RENDERER_OVERRIDE_VARS and not just the four the candidates
    name, because the app honours a wider set than this script tries.
    """
    env = {k: v for k, v in base.items() if k not in CLEARED_VARS}
    env.update(MEASUREMENT_ENV)
    env.update(extra)
    return env


def _has_display(env: dict) -> bool:
    """Is there somewhere for THIS candidate to draw?

    Candidate specific, because GDK_BACKEND=x11 needs an X DISPLAY in particular: on a
    Wayland-only session it has nowhere to go, and reporting that as a crash sends someone
    hunting a bug that is not there. Reads the environment the candidate is actually
    launched with, which is not the caller's once the candidate variables are stripped.
    """
    backend = env.get("GDK_BACKEND") or ""
    if backend == "x11":
        return bool(env.get("DISPLAY"))
    if backend == "wayland":
        return bool(env.get("WAYLAND_DISPLAY"))
    return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))


def _span(seconds: int) -> str:
    """Human duration. Integer division alone printed "0 minutes" for a short window."""
    if seconds < 60:
        return f"{seconds} seconds"
    m, sec = divmod(seconds, 60)
    return f"{m} minutes" if not sec else f"{m} minutes {sec} seconds"


def find_desktop_app() -> list[str] | None:
    """Locate Unsloth DESKTOP, which is not the same program as `unsloth studio`.

    This matters more than it looks. `unsloth studio` is the pip CLI: it serves the web UI
    and never runs the Tauri shell, so none of the desktop renderer logic executes and no
    `backend-*.log` is written (that file is produced by the shell, see
    studio/src-tauri/src/diagnostics/phase_log.rs). Pointing this script at the CLI
    therefore measures a different program from the one that freezes, and every candidate
    comes back with nothing observed.
    """
    globs = [
        "Unsloth*.AppImage",
        "Applications/Unsloth*.AppImage",
        "Downloads/Unsloth*.AppImage",
        ".local/bin/Unsloth*.AppImage",
    ]
    found = [
        c
        for c in (
            shutil.which("unsloth-studio"),
            "/usr/bin/unsloth-studio",
            "/opt/Unsloth/unsloth-studio",
        )
        if c and Path(c).is_file()
    ]
    hits = [q for g in globs for q in sorted(HOME.glob(g)) if q.is_file()]
    found += [str(q) for q in sorted(hits, key = lambda q: q.stat().st_mtime, reverse = True)]
    if not found:
        return None
    # An AppImage arrives without the execute bit, so prefer a startable candidate but still return one, or Popen raises
    # PermissionError before anything is measured or written.
    return [next((c for c in found if is_executable(c)), found[0])]


def is_executable(path) -> bool:
    return os.access(str(path), os.X_OK)


def sh(args, timeout = 20):
    try:
        r = subprocess.run(
            args,
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = timeout,
        )
        return (r.stdout or r.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def scrub(text: str) -> str:
    """Remove the two things that leak: the home path and anything token-shaped."""
    out = text.replace(str(HOME), "~")
    out = re.sub(r"\b(hf_|ghp_|github_pat_|sk-)[A-Za-z0-9_-]{8,}", r"\1<redacted>", out)
    return re.sub(r"[A-Za-z0-9+/]{40,}={0,2}", "<redacted-blob>", out)


def host_facts() -> dict:
    gpus = [
        l
        for l in sh(["lspci"]).splitlines()
        if re.search(r"VGA|3D controller|Display controller", l)
    ]
    return {
        "when": datetime.now().astimezone().isoformat(timespec = "seconds"),
        "os": sh(["sh", "-c", ". /etc/os-release 2>/dev/null && echo $PRETTY_NAME"]),
        "kernel": platform.release(),
        "session_type": os.environ.get("XDG_SESSION_TYPE", "(unset)"),
        "desktop": os.environ.get("XDG_CURRENT_DESKTOP", "(unset)"),
        "wayland_display": os.environ.get("WAYLAND_DISPLAY", "(unset)"),
        "display": os.environ.get("DISPLAY", "(unset)"),
        "gdk_backend_preset": os.environ.get("GDK_BACKEND", "(unset)"),
        "webkit_vars_preset": {
            k: v for k, v in os.environ.items() if k.startswith("WEBKIT_") or k.startswith("__NV_")
        },
        "gpus": gpus,
        "nvidia_driver": sh(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        ),
        "nvidia_module": (
            Path("/proc/driver/nvidia/version").read_text(encoding = "utf-8").strip()
            if Path("/proc/driver/nvidia/version").exists()
            else "(absent)"
        ),
        "webkit2gtk": sh(
            ["sh", "-c", "dpkg -l 2>/dev/null | awk '/libwebkit2gtk/ {print $2, $3}'"]
        ),
        "unsloth_cli": shutil.which("unsloth") or "(not on PATH)",
    }


def port_busy() -> bool:
    for p in PORTS:
        with socket.socket() as s:
            s.settimeout(0.4)
            if s.connect_ex(("127.0.0.1", p)) == 0:
                return True
    return False


def studio_backend_pids() -> list[int]:
    """PIDs listening on an Unsloth port that are ours, and only those.

    ONE rule, shared by everything that asks the question, so the gate that refuses to run
    and the cleanup that kills cannot disagree about what counts as ours. They did: the
    gate keyed on "is any Unsloth port busy" while the cleanup keyed on attribution, so an
    unrelated Jupyter on 8888 made the gate refuse to start a run in which the cleanup
    would have refused to touch it.

    Identify by the command line, not by /proc/<pid>/exe. The backend runs from a
    virtualenv, and a venv's python is a symlink, so `exe` resolves to the system
    interpreter (/usr/bin/python3.x) and never matches. Checking it skipped our own backend
    every time and left the orphan in place.
    """
    found = []
    for pid in sh(
        [
            "sh",
            "-c",
            f"ss -ltnp 2>/dev/null | grep -E ':({'|'.join(map(str, PORTS))}) ' "
            "| grep -oE 'pid=[0-9]+' | cut -d= -f2",
        ]
    ).split():
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace")
            if str(STUDIO) in argv:
                found.append(int(pid))
        except (OSError, ValueError):
            continue
    return found


def stop_leftover_backend():
    """Stop a backend left behind by the previous candidate.

    Closing the app does not necessarily stop the backend it started, and an orphaned one
    is the worst possible state to measure in: it still answers, so the next launch attaches
    to it (`OwnedReady`) instead of starting its own, but the process that copied its output
    into the log died with the app. The result is a backend that serves and is never written
    down, and every candidate after the first reads zero.

    Only processes attributable to Unsloth are touched. Matching on the port alone would
    happily kill an unrelated program that happens to be listening there.
    """
    for pid in studio_backend_pids():
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"    stopped the previous run's backend (pid {pid})", flush = True)
        except OSError:
            pass


def wait_for_leftover_backend_to_stop(timeout = 30):
    """Wait for OUR backend to go, but say so, and give up early.

    This used to wait 120s per candidate in silence, and on any listener at all. A port held
    by something that is not Unsloth never clears and never needed to: the backend walks on
    to the next free port in the range (_resolve_port in studio/backend/run.py). Waiting on
    it turned an unrelated service into an eight minute hang with no output and no reason.
    """
    if not studio_backend_pids():
        return True
    print(f"    waiting up to {timeout}s for the previous backend to exit", flush = True)
    for _ in range(timeout):
        if not studio_backend_pids():
            return True
        time.sleep(1)
    print(
        "    an Unsloth backend is still holding an Unsloth port and did not stop. Close "
        "whatever is still running, or this candidate will have nothing to measure.",
        flush = True,
    )
    return False


def backend_offsets() -> dict:
    try:
        return {p: p.stat().st_size for p in BACKEND_LOGS.glob("backend-*.log")}
    except OSError:
        return {}


def backend_tail(before: dict) -> str:
    """Everything appended to any supervised backend log since `before`.

    Re-globbed rather than reusing the earlier listing, because the run being measured
    usually creates its own log file, which was not there when the offsets were taken.
    """
    out = []
    try:
        logs = list(BACKEND_LOGS.glob("backend-*.log"))
    except OSError:
        return ""
    for p in logs:
        try:
            with p.open("rb") as fh:
                fh.seek(before.get(p, 0))
                out.append(fh.read().decode("utf-8", "replace"))
        except OSError:
            pass
    return "".join(out)


def renderer_applied(shell_out: str) -> dict:
    """The renderer workaround the app chose FOR ITSELF, from the app's own log line.

    Not from /proc/<pid>/environ. That file is the environment as it was at execve and
    nothing else: "modifications ... after this -- for example, by calling putenv(3), or by
    directly modifying the environ(7) variable -- are not reflected in /proc/[pid]/environ"
    (proc_pid_environ(5)). linux_webkit::configure_renderer() applies its choice with
    std::env::set_var AFTER exec, so reading /proc could never see it: `applied_by_app`
    came back empty on every run that applied a workaround, which reads in the report as
    "the app applied nothing" and is exactly the wrong thing to tell someone chasing a
    renderer bug. main.rs logs the decision instead, through a stderr logger, so it is in
    the captured output.
    """
    found = {}
    for match in RENDERER_APPLIED.finditer(shell_out):
        for assignment in match.group(1).split():
            k, _, v = assignment.partition("=")
            found[k] = v
    return found


def exec_env(pid: int) -> dict:
    """The candidate variables the app was STARTED with, read from the live process.

    UNSLOTH_WEBKIT_RENDERER_WORKAROUND is included because it decides whether the app reads
    a renderer variable as its own earlier output or as an instruction from the operator.
    Without it the report cannot explain a launch that preserved the environment and
    applied nothing.
    """
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().decode("utf-8", "replace")
    except OSError:
        return {}
    found = {}
    for entry in raw.split("\0"):
        k, _, v = entry.partition("=")
        if k.startswith("WEBKIT_") or k.startswith("__NV_") or k in RENDERER_OVERRIDE_VARS:
            found[k] = v
    return found


def _last_rise(samples: list, column: int) -> int | None:
    """Elapsed time of the last sample at which that counter moved, or None if it never did."""
    last = None
    for i in range(1, len(samples)):
        if samples[i][column] > samples[i - 1][column]:
            last = samples[i][0]
    return last


def _first_heard(samples: list, column: int) -> int | None:
    """Elapsed time of the earliest sample that proves that counter was moving.

    A rise between two samples proves it, and so does a first sample that is already above
    zero: that count was accumulated during the interval before it. None means the counter
    never moved while the run was being watched.
    """
    if not samples:
        return None
    if samples[0][column] > 0:
        return samples[0][0]
    for i in range(1, len(samples)):
        if samples[i][column] > samples[i - 1][column]:
            return samples[i][0]
    return None


def classify(
    samples: list,
    n_mon: int,
    n_live: int,
    exited,
    ran_for: int,
    interrupted: bool,
    preflight: str,
    shell_started: bool,
    has_display: bool,
    warmup: int = None,
    session_at: int = None,
) -> str:
    """One candidate's verdict. Pure, so the wrong ones can be caught by a test rather than
    by a reporter following them down a false path."""
    warmup = WARMUP if warmup is None else warmup

    if interrupted:
        return (
            f"SKIPPED: interrupted after {ran_for}s, before the observation window "
            f"finished, so this candidate was not measured"
        )

    if exited == 0 and ran_for <= 20:
        # A clean, immediate exit is almost always the single-instance guard (another copy already open,
        # so this launch handed over and quit), and calling that "crashed" would be wrong and alarming.
        return (
            "SKIPPED: the app exited immediately and cleanly, which usually means "
            "another copy of Unsloth is already running. Close it and re-run"
        )
    if exited == 0:
        return (
            f"ENDED EARLY: the app ran for {ran_for}s and then exited cleanly. If you "
            f"closed the window, just re-run and leave it open"
        )
    if exited is not None and not has_display:
        # Over plain SSH there is nothing to draw on.
        return (
            f"CANNOT RUN: the app exited (code {exited}) and there is no display to draw "
            f"on. Run this from a desktop session, not over plain SSH"
        )
    if exited is not None:
        return f"CRASHED: the app exited on its own (code {exited})"

    if n_mon == 0 and n_live == 0:
        # Do not guess the cause: the preflight line the app already printed says
        if not preflight and not shell_started:
            reason = (
                "the desktop shell never started. If you launched `unsloth studio`, that "
                "is the command line version and not the app that freezes; re-run this "
                "with the path to Unsloth Desktop"
            )
        elif "NotInstalled" in preflight:
            reason = (
                "Unsloth Studio itself is not installed, so there is no backend to "
                "observe. Open the app once and let it finish installing, then re-run"
            )
        elif "AttachedReady" in preflight or "OwnedReady" in preflight:
            reason = (
                "the app attached to a backend that was already running, which nothing "
                "is recording. Close every copy of Unsloth and re-run"
            )
        else:
            reason = "the backend never started, so there was nothing to observe"
        return f"NO SIGNAL: this run measured nothing, because {reason}"

    if n_live == 0:
        # The whole oracle is "watchdog alive, interface silent".
        return (
            "NO SIGNAL: the native watchdog never polled, so there is no independent "
            "signal to tell a frozen interface from a healthy one"
        )
    if n_mon == 0:
        # NOT a freeze, however much it looks like one.
        return (
            "NO SIGNAL: the interface was never heard from at all, so a frozen webview "
            "cannot be told apart from one that was never able to poll. Check that you "
            "are signed in, and that no other copy of Unsloth was already running when "
            "this started"
        )

    # Did the interface stop polling partway through while the watchdog carried on?
    post = [s for s in samples if s[0] >= warmup]
    resumed_at = _last_rise(post, 1)
    watchdog_last = _last_rise(post, 2)
    for i in range(1, len(post)):
        if post[i][1] == post[i - 1][1] and post[i][2] > post[i - 1][2]:
            if resumed_at is not None and resumed_at >= post[i][0]:
                continue
            stalled_at = post[i][0]
            if session_at is not None and session_at >= stalled_at:
                # The heartbeat is gated on holding a session token, so losing the session stops it as thoroughly as a
                # freeze does and the counters look identical. What separates them is that the webview went on making
                # requests after the heartbeat stopped, and a frozen webview cannot ask anything, so this is the app
                # falling back to its login screen. A positive signal rather than a doubt: it does not narrow the FROZE
                # arm, and a session cleared without a single request reaching the backend is still reported as a
                # freeze.
                return (
                    f"SIGNED OUT: the interface stopped polling at about {stalled_at}s, but "
                    f"it was still asking the backend about your session at about "
                    f"{session_at}s, and a frozen interface cannot ask anything. The app "
                    f"signed out, which stops the heartbeat on its own, so this candidate "
                    f"was not measured. Sign in and re-run it, staying signed in throughout"
                )
            sustained = (watchdog_last if watchdog_last is not None else stalled_at) - stalled_at
            if sustained >= STALE_AFTER:
                return (
                    f"FROZE: the interface stopped polling at about {stalled_at}s "
                    f"while the watchdog kept going"
                )
            # Short of that, this candidate is unsettled.
            return (
                f"SUSPECT: the interface stopped polling at about {stalled_at}s, but the "
                f"watchdog only kept going for another {sustained}s after that, short of "
                f"the {STALE_AFTER}s needed to tell a freeze from a delayed poll. This "
                f"candidate is unsettled rather than healthy: re-run it with "
                f"UNSLOTH_FREEZE_WINDOW above {WINDOW} to give the stall room to prove "
                f"itself"
            )

    # Both loops stopped together while the shell stayed up: neither counter moving means neither can be compared, and
    # this used to fall through to OK and report a dead run as healthy.
    end = samples[-1][0] if samples else 0
    mon_rise, live_rise = _last_rise(samples, 1), _last_rise(samples, 2)
    if len(samples) >= 4 and end >= warmup:
        quiet_for = end - max(mon_rise or 0, live_rise or 0)
        # Inclusive: STALE_AFTER is three poll intervals, and the strict comparison let the exact boundary case through
        # to OK, which is the one case the constant was picked to name.
        if quiet_for >= STALE_AFTER:
            return (
                f"NO SIGNAL: nothing was recorded for the last {quiet_for}s of the run, "
                f"neither from the interface nor from the watchdog, so the backend stopped "
                f"answering or stopped being logged before the window ended"
            )

    # The interface cannot be observed until it has a backend and a session, so a run whose counters first move in the
    # last few samples has no flat interval to find and passed the ratio test as OK.
    heard_from = _first_heard(samples, 1)
    watched = (end - heard_from) if heard_from is not None else 0
    if heard_from is not None and watched < STALE_AFTER:
        return (
            f"SUSPECT: the interface was not heard from until about {heard_from}s, so it "
            f"was only watched for {watched}s before the run ended, short of the "
            f"{STALE_AFTER}s a stall has to last to be called one. This candidate is "
            f"unsettled rather than healthy: sign in and let the app finish starting "
            f"before the next run, or re-run it with UNSLOTH_FREEZE_WINDOW above {WINDOW}"
        )

    if n_live >= 3 and n_mon * 3 < n_live:
        return "SUSPECT: the interface polled far less than the watchdog"
    return "OK: the interface kept polling for the whole run"


def run_candidate(label, extra, why, cmd) -> dict:
    print(f"\n=== {label} ===", flush = True)
    print(f"    ({why})", flush = True)
    stop_leftover_backend()
    if not wait_for_leftover_backend_to_stop():
        print(
            "    the previous run has not released its port; this candidate will attach "
            "to it and is likely to report NO SIGNAL",
            flush = True,
        )

    env = candidate_env(dict(os.environ), extra)
    cleared = sorted(k for k in CLEARED_VARS if k in os.environ and k not in extra)
    if cleared:
        print(f"    unset for this candidate: {', '.join(cleared)}", flush = True)
    before = backend_offsets()
    # To a FILE, never subprocess.PIPE: nothing reads the pipe while the app runs, so once it filled
    # the 64 KiB buffer it would block on its own stdout and this script would hang the app it is
    # measuring.
    app_log = Path(tempfile.mkstemp(suffix = ".log", prefix = "unsloth-freeze-")[1])
    try:
        proc = subprocess.Popen(
            cmd,
            env = env,
            stdout = app_log.open("w", encoding = "utf-8", errors = "replace"),
            stderr = subprocess.STDOUT,
            start_new_session = True,
        )
    except OSError as exc:
        # The execute bit says the kernel may try, not that the try succeeds: a wrong-arch build, a
        # truncated AppImage, a missing `#!` interpreter or a noexec mount all fail at execve, and Popen
        # raises OSError, which the candidate loop does not catch, ending the whole diagnostic with
        # nothing measured. Record it as this candidate's result so the rest still run.
        why_failed = scrub(str(exc.strerror or exc))
        print(f"    CANNOT RUN: {why_failed}", flush = True)
        return {
            "candidate": label,
            "why": why,
            "env": extra,
            "cleared_env": cleared,
            "verdict": (
                f"CANNOT RUN: the app could not be launched ({why_failed}). It has its "
                f"execute bit, so this is the launch itself failing: a build for another "
                f"CPU architecture, a corrupt or partly downloaded AppImage, a missing "
                f"interpreter, or a filesystem mounted noexec"
            ),
            "preflight": "(not seen)",
            "applied_by_app": {},
            "env_at_exec": {},
            "interface_polls": 0,
            "watchdog_polls": 0,
            "session_seen_at": None,
            "exit_code": None,
            "samples": [],
            "backend_log_excerpt": "",
        }
    started = time.monotonic()
    at_exec, samples, exited, ran_for = {}, [], None, 0
    interrupted = False
    # When the webview was last seen asking about the session, and how many such requests
    # that was. Kept beside the samples rather than in them: it is not a heartbeat, it is
    # the one thing that can tell a sign-out apart from a freeze. See SESSION.
    session_seen, session_at = 0, None

    print(f"    launching, then watching for {_span(WARMUP + WINDOW)}.", flush = True)
    print("    Use the window normally while this runs.", flush = True)
    try:
        while time.monotonic() - started < WARMUP + WINDOW:
            time.sleep(POLL_EVERY)
            if proc.poll() is not None:
                exited = proc.returncode
                ran_for = round(time.monotonic() - started)
                print(
                    f"    the app EXITED (code {exited}) after "
                    f"{time.monotonic() - started:.0f}s",
                    flush = True,
                )
                break
            if not at_exec:
                at_exec = exec_env(proc.pid)
            text = backend_tail(before)
            n_mon, n_live = len(INTERFACE.findall(text)), len(LIVENESS.findall(text))
            elapsed = round(time.monotonic() - started)
            samples.append((elapsed, n_mon, n_live))
            n_session = len(SESSION.findall(text))
            if n_session > session_seen:
                session_seen, session_at = n_session, elapsed
            if len(samples) % 2 == 0:
                print(
                    f"    t={samples[-1][0]:4}s  interface={n_mon:3}  watchdog={n_live:3}",
                    flush = True,
                )
    except KeyboardInterrupt:
        interrupted = True
        ran_for = round(time.monotonic() - started)
        print("    interrupted; this candidate is recorded as skipped", flush = True)
    finally:
        alive = proc.poll() is None
        if not alive and exited is None:
            # It died between the loop's last poll and this one.
            exited = proc.returncode
            if not ran_for:
                ran_for = round(time.monotonic() - started)
            print(f"    the app EXITED (code {exited}) during cleanup", flush = True)
        if alive:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(5)
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        try:
            proc.wait(timeout = 30)
        except subprocess.TimeoutExpired:
            pass

    text = backend_tail(before)
    shell_out = app_log.read_text(encoding = "utf-8", errors = "replace")
    n_mon, n_live = len(INTERFACE.findall(text)), len(LIVENESS.findall(text))

    pre_lines = [l for l in (text + shell_out).splitlines() if "desktop_preflight completed" in l]
    preflight = pre_lines[-1].strip() if pre_lines else ""
    applied = renderer_applied(shell_out)

    if not ran_for:
        ran_for = samples[-1][0] if samples else 0

    verdict = classify(
        samples = samples,
        n_mon = n_mon,
        n_live = n_live,
        exited = exited,
        ran_for = ran_for,
        interrupted = interrupted,
        preflight = preflight,
        shell_started = bool(SHELL_STARTED.search(shell_out)) or bool(applied),
        has_display = _has_display(env),
        session_at = session_at,
    )

    print(f"    VERDICT: {verdict}", flush = True)
    return {
        "candidate": label,
        "why": why,
        "env": extra,
        # What was taken away, not just what was added.
        "cleared_env": cleared,
        "verdict": verdict,
        "preflight": scrub(preflight) if preflight else "(not seen)",
        "applied_by_app": applied,
        "env_at_exec": at_exec,
        "interface_polls": n_mon,
        "watchdog_polls": n_live,
        # When the webview last asked about the session, so a reader can see for
        # themselves why a stall was or was not read as a sign-out. See SESSION.
        "session_seen_at": session_at,
        "exit_code": exited,
        "samples": samples,
        "backend_log_excerpt": scrub("\n".join(text.splitlines()[-40:])),
    }


def resolve_command(cmd: list[str]) -> list[str] | None:
    """The command as `subprocess.Popen` will resolve it, or None if it does not exist.

    Popen uses `os.execvpe()`-like behaviour, so a first argument with no slash in it is
    looked up on PATH and NOT in the current directory. `Unsloth.AppImage`, typed while
    sitting in ~/Downloads, passes an `is_file()` check and then raises FileNotFoundError
    at launch, which is the one place a plain mistake looks like a broken script. Anything
    that names an existing file becomes an absolute path here so the two agree.
    """
    on_path = shutil.which(cmd[0])
    if on_path and not Path(cmd[0]).is_file():
        return [on_path] + cmd[1:]
    if Path(cmd[0]).is_file():
        return [str(Path(cmd[0]).resolve())] + cmd[1:]
    if on_path:
        return [on_path] + cmd[1:]
    return None


def confirm_stop_running_studio() -> bool:
    """A backend is already answering before the first candidate. Ask before touching it.

    `stop_leftover_backend()` cannot tell a backend orphaned by a previous run from the one
    serving the Unsloth the reporter has open right now: both are Unsloth processes on an
    Unsloth port. Running into it unasked SIGTERMs a live session, interrupting whatever it
    was doing, to gather a report about a freeze. Printing a note and carrying on is not
    consent, so require an answer, and refuse rather than guess when there is nobody to ask.
    """
    if os.environ.get("UNSLOTH_FREEZE_STOP_RUNNING") == "1":
        return True
    if not sys.stdin or not sys.stdin.isatty():
        return False
    try:
        return input("Stop it and continue? [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def main() -> int:
    cmd = sys.argv[1:] or find_desktop_app()
    if not cmd:
        print(
            "Could not find Unsloth Desktop.\n\n"
            "Note this is NOT the same as `unsloth studio`, which is the command line\n"
            "version: it serves the web interface but does not run the desktop shell, so\n"
            "it cannot show the freeze you are seeing and there is nothing to measure.\n\n"
            "Pass the desktop app explicitly, for example:\n"
            f"  python3 {Path(__file__).name} ~/Applications/Unsloth-Desktop.AppImage\n"
            "  python3 {} /usr/bin/unsloth-studio".format(Path(__file__).name)
        )
        return 2
    asked_for = cmd[0]
    cmd = resolve_command(cmd)
    if cmd is None:
        print(
            f"cannot find {asked_for!r} on PATH. Pass the command explicitly, for example:\n"
            f"  python3 {Path(__file__).name} ~/Applications/Unsloth-Desktop.AppImage"
        )
        return 2
    # Checked here, not left to the launch. subprocess.Popen raises PermissionError
    if not is_executable(cmd[0]):
        print(
            f"{cmd[0]} is not executable, so it cannot be launched.\n\n"
            "An AppImage downloaded through a browser arrives without the execute bit. "
            "Set it and re-run:\n"
            f"  chmod +x {cmd[0]}"
        )
        return 2

    print("Unsloth Desktop freeze report")
    print("=" * 60)
    print(
        f"This runs {len(CANDIDATES)} launches of about "
        f"{_span(WARMUP + WINDOW)} each, so roughly "
        f"{_span(len(CANDIDATES) * (WARMUP + WINDOW))} in total."
    )
    print("Use the app normally during each one. Ctrl-C skips to the next candidate.\n")

    # Only refuse over a listener this script would actually stop.
    if studio_backend_pids():
        print("An Unsloth backend is already listening on an Unsloth port. That is either")
        print("Unsloth running right now, or a backend left behind by an earlier run, and")
        print("this script cannot tell them apart: continuing STOPS it, which interrupts")
        print("whatever that Unsloth is doing.\n")
        print("Close any running Unsloth (including `unsloth studio` in another terminal)")
        print("and start again, or answer below to stop it from here.\n")
        if not confirm_stop_running_studio():
            print("Nothing was stopped and no report was written.")
            return 2
        print()
    elif port_busy():
        print("Something that is not Unsloth is listening on an Unsloth port. Nothing needs")
        print("to be done about it: the backend falls back to the next free port, and this")
        print("script only ever stops a process it can attribute to Unsloth.\n")

    print("The app is launched with the backend's access log suppressors turned off, so")
    print("its own liveness polls get written down. That changes what is recorded, not")
    print("how the interface renders.\n")

    facts = host_facts()
    print(f"  session : {facts['session_type']}   desktop: {facts['desktop']}")
    print(f"  gpus    : {'; '.join(facts['gpus']) or '(none reported)'}")
    print(f"  driver  : {facts['nvidia_driver'] or '(no nvidia-smi)'}")

    results = []
    try:
        for label, extra, why in CANDIDATES:
            try:
                results.append(run_candidate(label, extra, why, cmd))
            except KeyboardInterrupt:
                print("\n  skipped by user", flush = True)
    finally:
        # Closing the app does not stop the backend it started, so without this the reporter's next launch attaches to a
        # backend nothing is recording.
        print("\n  stopping any backend left behind by the last candidate", flush = True)
        stop_leftover_backend()

    out = Path.cwd() / f"unsloth-freeze-report-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(
        json.dumps(
            {
                "host": json.loads(scrub(json.dumps(facts))),
                # Stated, not just commented: every launch ran with the access-log suppressors off, which is the only
                # reason the interface heartbeat appears at all.
                "measurement_env": MEASUREMENT_ENV,
                "results": results,
            },
            indent = 2,
        ),
        encoding = "utf-8",
    )

    print("\n" + "=" * 60)
    print("Summary")
    for r in results:
        print(f"  {r['verdict'].split(':')[0]:10} {r['candidate']}")
    print(f"\nReport written to:\n  {out}")
    print("\nIt contains your OS, GPU model, driver version and the app's own log lines.")
    print("Home paths and anything token-shaped are already removed. Please look it over")
    print("before sending it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
