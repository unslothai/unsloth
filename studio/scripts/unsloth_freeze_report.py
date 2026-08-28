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
# The interface's heartbeat. /api/export/status is the load-bearing member and the reason
# this list is trustworthy at all: use-export-runtime-lifecycle.ts polls it every 5s from
# an effect with an EMPTY dependency list, mounted unconditionally at the app root
# (studio/frontend/src/app/routes/__root.tsx), and the only thing in front of it is
# `if (!hasAuthToken()) return;`. There is no setting for it anywhere in the UI, and unlike
# every other poll in the app it has no `document.hidden` check either, so minimising the
# window does not stop it.
#
# That property is what the rest of this list lacks. Every other repeating webview poll is
# behind something the user controls:
#
#   /api/inference/monitor          api-monitor-overlay.tsx stands its loop down on
#                                   `onFullPage || (!autoOpen && !isOpen)`. autoOpen starts
#                                   out true, but Settings > Resources turns it off and the
#                                   panel's own "stop opening this automatically" does too.
#   /api/inference/status and the images / video / audio-stt status polls
#                                   use-loaded-models.ts returns early on `!track`, and
#                                   track is show-loaded-models-pref.ts, which is
#                                   `localStorage.getItem(KEY) === "true"` and therefore
#                                   OFF until somebody explicitly turns the indicator on.
#
# They are still counted, because a hit from any of them is equally good evidence that the
# webview is alive. What they cannot support is the opposite inference: their silence is
# the default state of a healthy app, not a symptom. Reading it as one is what made an
# earlier version of this script report FROZE for every candidate on a stock install, and
# no amount of grouping them together fixes that, because the whole group is optional.
#
# /api/auth/status is deliberately NOT here, even though the backend files it in the same
# log bucket as the loaded-model polls. app/auth-guards.ts fetches it on navigation with a
# 30s TTL, never on a timer, so it flatlines as soon as the reporter stops clicking around
# and scoring it would invent a freeze out of somebody sitting still.
#
# None of these are called by the native shell, which only ever requests /api/liveness and
# /api/health (studio/src-tauri/src/commands.rs), so every hit here comes from the webview.
INTERFACE = re.compile(
    r"/api/(?:export/status|inference/monitor|inference/status"
    r"|inference/images/status|inference/video/status|inference/audio/stt/status)"
)
LIVENESS = re.compile(r"/api/liveness")
# The webview's sign-in traffic, which is NOT a heartbeat and is never counted as one: it
# is how this script tells a sign-out apart from a freeze.
#
# The heartbeat above stops for a reason that has nothing to do with rendering:
# `if (!hasAuthToken()) return;` in use-export-runtime-lifecycle.ts (:156). The interval
# keeps firing, the request stops being made, and a session cleared mid-run (a sign-out, or
# a refresh that failed) therefore produces exactly the pattern this script calls a freeze,
# on an app whose login screen is drawing perfectly.
#
# Nothing in the counters can separate those two, but the log can, because clearing a
# session is not silent. A sign-out POSTs /api/auth/logout (features/auth/api.ts:296), an
# expired session POSTs /api/auth/refresh (:174) and gets a 401, and the redirect to the
# login screen that follows either one GETs /api/auth/status (app/auth-guards.ts:42). All
# three are requests, and a frozen webview cannot make a request: one of them landing at or
# after the moment the heartbeat stopped is positive evidence that the interface was alive.
#
# /api/auth/desktop-login is excluded deliberately. It is the one auth route the NATIVE
# shell posts by itself (src-tauri/src/desktop_auth.rs:194), so counting it would let the
# shell vouch for a webview that is not running. The rest of /api/auth is webview-only.
#
# All of these are logged verbatim under this script's MEASUREMENT_ENV: the dedup windows
# are zero, which sets _VERBOSE_ACCESS_LOG in studio/backend/loggers/handlers.py and turns
# the 2xx poll suppressor off, and the mutations were never suppressed to begin with.
SESSION = re.compile(r"/api/auth/(?:status|login|logout|refresh)\b")
# Printed by the desktop shell (main.rs) before anything else, through a stderr logger, so
# it lands in the captured shell output. Distinguishes "the shell never started" from "the
# shell started and its backend did not".
SHELL_STARTED = re.compile(r"Unsloth desktop app starting")
# `{reason}; set VAR=1 VAR2=1 for WebKitGTK compatibility`, the app's own record of the
# renderer workaround it chose for itself.
RENDERER_APPLIED = re.compile(r"set ((?:[A-Za-z_][A-Za-z_0-9]*=1\s*)+)for WebKitGTK compatibility")

# Overridable so CI can exercise this script end to end in a couple of minutes. A reporter
# should never need to set them: the defaults are what make a slow freeze visible.
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

# Every renderer override the APP reads, which is not the same set as the ones this script
# tries, and must not be re-derived from CANDIDATES. studio/src-tauri/src/linux_webkit.rs:
#
#   WEBKIT_DISABLE_DMABUF_RENDERER    either one present and unclaimed returns
#   WEBKIT_DMABUF_RENDERER_FORCE_SHM  RenderingPlan::PreserveEnvironment, so the app
#                                     applies nothing and the inherited value decides the
#                                     renderer for the whole launch.
#   WEBKIT_FORCE_DMABUF_RENDERER      the NVIDIA dmabuf patch's own opt-out, honoured
#                                     explicitly because WebKit returns on DISABLE_DMABUF
#                                     before it would ever be read.
#   UNSLOTH_WEBKIT_RENDERER_WORKAROUND  the comma-joined marker naming the variables the
#                                     app set for itself, so a relaunch can tell its own
#                                     inherited output from an operator's value. Inherited
#                                     from an unrelated earlier launch it makes the app
#                                     read a stale claim as its own and skip the override
#                                     test above.
#   GDK_BACKEND                       selects the display backend, which is what decides
#                                     between the shared-memory switch and no workaround.
#   UNSLOTH_WEBKIT_DISABLE_COMPOSITING  the app's own on/off switch for the compositing
#                                     workaround, and the one a freezing host is told to
#                                     export, so of all of these it is the likeliest to
#                                     still be set in the shell that runs this script.
#
# None of these appear in CANDIDATES, so a cleared set derived from CANDIDATES left every
# one of them active. Any of them still exported in the reporter's shell then pins all four
# launches, INCLUDING the control, and a control that cannot produce the other answer is not
# a control: the report comes back saying the comparison was clean when nothing was ever
# compared.
RENDERER_OVERRIDE_VARS = (
    "GDK_BACKEND",
    "UNSLOTH_WEBKIT_DISABLE_COMPOSITING",
    "UNSLOTH_WEBKIT_RENDERER_WORKAROUND",
    "WEBKIT_DISABLE_DMABUF_RENDERER",
    "WEBKIT_DMABUF_RENDERER_FORCE_SHM",
    "WEBKIT_FORCE_DMABUF_RENDERER",
)
CLEARED_VARS = tuple(sorted(set(CANDIDATE_VARS) | set(RENDERER_OVERRIDE_VARS)))

# Applied to the app this script launches, and to nothing else. The backend's access log
# suppresses precisely the line the verdict now depends on: /api/export/status is in
# _QUIET_SUCCESS_PATHS (studio/backend/loggers/handlers.py), so its 2xx is dropped
# outright, and the loaded-model polls share one 10s dedup bucket. Both suppressors read
# these two variables once at import, and both are off at 0, which is exactly what
# `--verbose` sets.
#
# This widens what gets written down, not what is being measured: neither variable reaches
# the renderer, the webview or any user preference, so the app under test is still the app
# the reporter runs. The backend inherits them because nothing on the spawn path calls
# env_clear and only UNSLOTH_STUDIO_HOME, STUDIO_HOME and STUDIO_LOCAL_REPO are scrubbed
# (MANAGED_CHILD_SCRUBBED_ENV in studio/src-tauri/src/process.rs). A run that ATTACHES to a
# backend it did not start never delivers them, which is why the verdict below treats a
# heartbeat of zero as "not measured" rather than as a freeze.
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
    # An AppImage arrives from the browser without the execute bit, and a downloaded one is
    # the likeliest thing this glob picks up. Prefer a candidate that can actually be
    # started; if none can, still return one, so the check in main() can name the file and
    # the command that fixes it instead of leaving Popen to raise PermissionError out of
    # the first candidate, before anything is measured and before a report is written.
    return [next((c for c in found if is_executable(c)), found[0])]


def is_executable(path) -> bool:
    return os.access(str(path), os.X_OK)


def sh(args, timeout = 20):
    try:
        r = subprocess.run(args, capture_output = True, text = True, timeout = timeout)
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
        # Ctrl-C is documented as "skips to the next candidate", so the samples are a
        # truncated window, not a measurement. Judging them says OK for a run that was
        # stopped while healthy and NO SIGNAL for one stopped during startup, and both of
        # those go into the summary looking like findings.
        return (
            f"SKIPPED: interrupted after {ran_for}s, before the observation window "
            f"finished, so this candidate was not measured"
        )

    if exited == 0 and ran_for <= 20:
        # A clean, immediate exit is almost always the single-instance guard: another copy
        # of Unsloth is already open, so this launch handed over and quit. Calling that
        # "crashed" would be both wrong and alarming, and it is the likeliest thing to go
        # wrong for someone running this on their own desktop.
        return (
            "SKIPPED: the app exited immediately and cleanly, which usually means "
            "another copy of Unsloth is already running. Close it and re-run"
        )
    if exited == 0:
        # Ran a while and then exited cleanly. Single instance handover is immediate, so
        # this is not that; the likeliest cause is simply that the window was closed.
        return (
            f"ENDED EARLY: the app ran for {ran_for}s and then exited cleanly. If you "
            f"closed the window, just re-run and leave it open"
        )
    if exited is not None and not has_display:
        # Over plain SSH there is nothing to draw on. Calling that a crash starts a bug
        # hunt for a bug that is not there.
        return (
            f"CANNOT RUN: the app exited (code {exited}) and there is no display to draw "
            f"on. Run this from a desktop session, not over plain SSH"
        )
    if exited is not None:
        return f"CRASHED: the app exited on its own (code {exited})"

    if n_mon == 0 and n_live == 0:
        # Do not guess the cause: the preflight line the app already printed says which
        # of the two it is, and naming the wrong one sends the user off fixing nothing.
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
        # The whole oracle is "watchdog alive, interface silent". Without the watchdog
        # there is no second opinion, so a webview that polled at startup and then froze
        # is indistinguishable here from one that never stopped. Say so instead of
        # passing it.
        return (
            "NO SIGNAL: the native watchdog never polled, so there is no independent "
            "signal to tell a frozen interface from a healthy one"
        )
    if n_mon == 0:
        # NOT a freeze, however much it looks like one. A heartbeat of zero is what a real
        # freeze produces, but it is also what several perfectly healthy runs produce, and
        # nothing in the samples separates them:
        #
        #   * the webview only starts polling /api/export/status once it holds a session
        #     token, so a run sitting on the sign-in screen reads zero;
        #   * a launch that ATTACHED to a backend it did not start never delivered
        #     MEASUREMENT_ENV to that backend, which therefore still drops the 2xx line for
        #     that path and collapses the optional polls into one 10s bucket;
        #   * every other interface poll is behind a user preference, and the loaded-model
        #     ones are behind one that is off until somebody turns it on.
        #
        # The last of those is why this branch used to be wrong for everybody: on a stock
        # install with the API monitor switched off, every counted path is silent while
        # /api/liveness ticks away, and calling that FROZE told a reporter whose app was
        # fine that all four candidates froze. A verdict the reader has no way to doubt has
        # to decline when it cannot tell.
        return (
            "NO SIGNAL: the interface was never heard from at all, so a frozen webview "
            "cannot be told apart from one that was never able to poll. Check that you "
            "are signed in, and that no other copy of Unsloth was already running when "
            "this started"
        )

    # Did the interface stop polling partway through while the watchdog carried on? That is
    # the reported symptom, and a total that looks healthy can still hide it.
    #
    # Only after the warmup boundary. On a cold launch the native watchdog is answering
    # before the webview has finished loading, so the very first samples always show a
    # still interface count and a rising watchdog count. Comparing those samples set
    # `stalled_at` on the startup of a run that then went on to poll happily for four
    # minutes, and no later evidence could clear it: every healthy candidate was reported
    # FROZE at about the moment it finished starting up.
    # A freeze does not recover. One flat interval is a delayed request, a pause, or a
    # backend hiccup, and the reported symptom is an interface that stops and stays
    # stopped, so require that it never polls again rather than reporting the first gap.
    # STALE_AFTER already applies this reasoning to the both-counters-flat case below
    # ("a single missed sample is not it"); this arm was the one place it did not.
    # "It never polls again" is not enough on its own, because a run that ends one sample
    # after the interface goes quiet has no later samples in which it COULD poll again. The
    # stall has to have been watched for long enough to mean something, which is what
    # STALE_AFTER names, and the watch only counts while the watchdog is still answering:
    # once both counters stop there is no longer a second signal to contradict the first.
    post = [s for s in samples if s[0] >= warmup]
    resumed_at = _last_rise(post, 1)
    watchdog_last = _last_rise(post, 2)
    for i in range(1, len(post)):
        if post[i][1] == post[i - 1][1] and post[i][2] > post[i - 1][2]:
            if resumed_at is not None and resumed_at >= post[i][0]:
                continue
            stalled_at = post[i][0]
            if session_at is not None and session_at >= stalled_at:
                # The heartbeat is gated on holding a session token, so losing the session
                # stops it just as thoroughly as a freeze does, and the counters look
                # identical. What separates them is that the webview went on making
                # requests: it asked the backend about the session at or after the moment
                # the heartbeat stopped, and a frozen webview cannot ask anything. So this
                # is the app falling back to its login screen, not a freeze.
                #
                # This is a positive signal rather than a doubt, which is why it does not
                # narrow the FROZE arm any further: a stall with no sign-in traffic after it
                # is still called a freeze, exactly as before. A session cleared without a
                # single request reaching the backend would still be indistinguishable, and
                # would still be reported as a freeze.
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
            # Short of that, this candidate is unsettled, and it must not be allowed to
            # fall through to the OK at the bottom: a stall that started just before the
            # window closed would then be reported as a healthy run, which is the same
            # confident wrong answer in the other direction. Say what was seen, say why it
            # is not conclusive, and say what to change to settle it.
            return (
                f"SUSPECT: the interface stopped polling at about {stalled_at}s, but the "
                f"watchdog only kept going for another {sustained}s after that, short of "
                f"the {STALE_AFTER}s needed to tell a freeze from a delayed poll. This "
                f"candidate is unsettled rather than healthy: re-run it with "
                f"UNSLOTH_FREEZE_WINDOW above {WINDOW} to give the stall room to prove "
                f"itself"
            )

    # Both loops stopped together while the shell stayed up: the backend went away, or its
    # output stopped being recorded. Neither counter moving means neither can be compared,
    # and the totals from earlier in the run are large enough that nothing above matches,
    # so this used to fall through to OK and report a dead run as a healthy one.
    end = samples[-1][0] if samples else 0
    mon_rise, live_rise = _last_rise(samples, 1), _last_rise(samples, 2)
    if len(samples) >= 4 and end >= warmup:
        quiet_for = end - max(mon_rise or 0, live_rise or 0)
        # Inclusive: STALE_AFTER is three poll intervals, and a run whose counters last
        # moved three intervals before the end has been silent for exactly that long. The
        # strict comparison let the boundary case through to OK, which is the one case the
        # constant was picked to name.
        if quiet_for >= STALE_AFTER:
            return (
                f"NO SIGNAL: nothing was recorded for the last {quiet_for}s of the run, "
                f"neither from the interface nor from the watchdog, so the backend stopped "
                f"answering or stopped being logged before the window ended"
            )

    # How long was the interface actually watched? Everything above reasons about the
    # INTERIOR of the sample series; this is its start. The run begins when the app is
    # launched, but the interface cannot be observed until it has a backend to talk to and a
    # session to talk with, and neither is guaranteed to arrive early: a slow first install,
    # a backend that takes most of the window to come up, or a reporter who signs in near
    # the end all produce a run whose counters first move in the last few samples. There is
    # then no flat interval to find, the totals from those last samples pass the ratio test
    # below, and the bottom line says the interface "kept polling for the whole run" about
    # an interface that was seen for seconds. A freeze that takes a minute to arrive cannot
    # be ruled out in that time, so this is the same unsettled case as a stall that begins
    # as the window closes, and it gets the same answer rather than a false OK.
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
    # To a FILE, never subprocess.PIPE. Nothing here reads the pipe while the app runs, so
    # once the app had written enough to fill the 64 KiB buffer it would block on its own
    # stdout: this script would hang the app it is measuring, and the user would see a
    # freeze that the script itself caused.
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
        # The execute bit checked in main() says the kernel is allowed to try, not that the
        # try succeeds: a build for another CPU architecture, a truncated AppImage, a
        # missing `#!` interpreter and a noexec mount all get as far as execve and fail
        # there. Popen raises OSError, and the candidate loop in main() catches only
        # KeyboardInterrupt, so the first such candidate ended the whole diagnostic in a
        # traceback with nothing measured and no report written. Record it as this
        # candidate's result instead, so the rest still run and the report still lands.
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
            # It died between the loop's last poll and this one, which is a narrow gap in
            # seconds and a wide one in meaning: an app that crashes right at the end of the
            # window still crashed. Without recording it here the cleanup saw a dead process
            # and left `exited` as None, so classify() skipped both exit branches and judged
            # the run on its samples alone, which look healthy right up to the moment the
            # app disappeared. The crash the reporter ran this to catch was then reported
            # back to them as OK.
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
        # What was taken away, not just what was added. A renderer override inherited from
        # the reporter's shell would otherwise pin this launch invisibly, and the reader of
        # the report has no other way to see that it was dealt with.
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
    # Checked here, not left to the launch. subprocess.Popen raises PermissionError, and
    # nothing on the path from run_candidate() back up to the candidate loop catches it, so
    # a browser-downloaded AppImage that never got its execute bit ended the whole run with
    # a traceback on the first candidate: nothing measured, no report written, and the one
    # line that says what to do about it absent.
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

    # Only refuse over a listener this script would actually stop. Asking about any busy
    # port at all meant somebody else's Jupyter on 8888 turned every unattended run into an
    # immediate exit 2, because confirm_stop_running_studio() answers no when there is
    # nobody to ask, over a process stop_leftover_backend() would then have declined to
    # touch. An unrelated listener needs nothing done: _resolve_port() in
    # studio/backend/run.py walks on to the next free port in the range.
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
        # The last candidate's backend is cleaned up by the NEXT candidate, and there is no
        # next one. Closing the app does not stop the backend it started, so without this
        # the script exits leaving Unsloth quietly serving: the reporter's next real launch
        # attaches to a backend nothing is recording, which this script's own docstring
        # calls the worst possible state to be in, and a second run of it would refuse to
        # start against a port it cannot explain.
        print("\n  stopping any backend left behind by the last candidate", flush = True)
        stop_leftover_backend()

    out = Path.cwd() / f"unsloth-freeze-report-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(
        json.dumps(
            {
                "host": json.loads(scrub(json.dumps(facts))),
                # Stated, not just commented: every launch below ran with the backend's
                # access-log suppressors off, which is the only reason the interface
                # heartbeat appears at all. Anyone comparing this against their own logs
                # needs to know the recording was widened.
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
