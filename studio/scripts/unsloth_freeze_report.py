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

  /api/inference/monitor   the web interface itself; only runs while the interface is alive
  /api/liveness            a native watchdog in the app; keeps running even if the
                           interface is dead

A freeze is the specific pattern where the watchdog keeps ticking and the interface goes
silent. That is measured here rather than guessed at, which is why this is worth running
even though you can already see the freeze with your own eyes: it distinguishes "the
interface stopped" from "the whole app died" from "the app is fine and it looked stuck",
and it says which of those each workaround produces.
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
MONITOR = re.compile(r"/api/inference/monitor")
LIVENESS = re.compile(r"/api/liveness")

# Overridable so CI can exercise this script end to end in a couple of minutes. A reporter
# should never need to set them: the defaults are what make a slow freeze visible.
WARMUP = int(os.environ.get("UNSLOTH_FREEZE_WARMUP", 90))
WINDOW = int(os.environ.get("UNSLOTH_FREEZE_WINDOW", 150))
PORTS = (8888, 8890)

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


def _has_display(extra: dict) -> bool:
    """Is there somewhere for THIS candidate to draw?

    Candidate specific, because GDK_BACKEND=x11 needs an X DISPLAY in particular: on a
    Wayland-only session it has nowhere to go, and reporting that as a crash sends someone
    hunting a bug that is not there.
    """
    backend = extra.get("GDK_BACKEND") or os.environ.get("GDK_BACKEND") or ""
    if backend == "x11":
        return bool(os.environ.get("DISPLAY"))
    if backend == "wayland":
        return bool(os.environ.get("WAYLAND_DISPLAY"))
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


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
    for cand in (
        shutil.which("unsloth-studio"),
        "/usr/bin/unsloth-studio",
        "/opt/Unsloth/unsloth-studio",
    ):
        if cand and Path(cand).is_file():
            return [str(cand)]
    globs = [
        "Unsloth*.AppImage",
        "Applications/Unsloth*.AppImage",
        "Downloads/Unsloth*.AppImage",
        ".local/bin/Unsloth*.AppImage",
    ]
    hits = [q for g in globs for q in sorted(HOME.glob(g)) if q.is_file()]
    if hits:
        return [str(max(hits, key = lambda q: q.stat().st_mtime))]
    return None


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
            Path("/proc/driver/nvidia/version").read_text().strip()
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


def stop_leftover_backend():
    """Stop a backend left behind by the previous candidate.

    Closing the app does not necessarily stop the backend it started, and an orphaned one
    is the worst possible state to measure in: it still answers, so the next launch attaches
    to it (`OwnedReady`) instead of starting its own, but the process that copied its output
    into the log died with the app. The result is a backend that serves and is never written
    down, and every candidate after the first reads zero.

    Only processes whose executable lives under ~/.unsloth are touched. Matching on the port
    alone would happily kill an unrelated program that happens to be listening there.
    """
    for pid in sh(
        [
            "sh",
            "-c",
            f"ss -ltnp 2>/dev/null | grep -E ':({'|'.join(map(str, PORTS))}) ' "
            "| grep -oE 'pid=[0-9]+' | cut -d= -f2",
        ]
    ).split():
        # Identify it by its command line, not by /proc/<pid>/exe. The backend runs from a
        # virtualenv, and a venv's python is a symlink, so `exe` resolves to the system
        # interpreter (/usr/bin/python3.x) and never matches. Checking `exe` skipped our own
        # backend every time and left the orphan in place.
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if str(STUDIO) not in argv:
            print(f"    leaving pid {pid} alone; not started by Unsloth", flush = True)
            continue
        try:
            os.kill(int(pid), signal.SIGTERM)
            print(f"    stopped the previous run's backend (pid {pid})", flush = True)
        except (OSError, ValueError):
            pass


def wait_for_free_ports(timeout = 30):
    """Wait, but say so, and give up early.

    This used to wait 120s per candidate in silence. When the port is held by a program
    this script cannot stop (another user's process, or anything that is not Unsloth), it
    never clears, so a run looked like an eight minute hang with no output and no reason.
    """
    if not port_busy():
        return True
    print(f"    waiting up to {timeout}s for the Studio port to be released", flush = True)
    for _ in range(timeout):
        if not port_busy():
            return True
        time.sleep(1)
    print(
        "    the port is still in use by a program this script cannot stop. Close whatever "
        "is using it, or this candidate will have nothing to measure.",
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


def applied_env(pid: int) -> dict:
    """What the app set on ITSELF, read from the live process.

    Not from a log line: the app only logs a renderer decision when it applies one, so an
    absent line and an absent workaround look identical in the log and do not here.
    """
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().decode("utf-8", "replace")
    except OSError:
        return {}
    found = {}
    for entry in raw.split("\0"):
        k, _, v = entry.partition("=")
        if k.startswith("WEBKIT_") or k.startswith("__NV_") or k == "GDK_BACKEND":
            found[k] = v
    return found


def run_candidate(label, extra, why, cmd) -> dict:
    print(f"\n=== {label} ===", flush = True)
    print(f"    ({why})", flush = True)
    stop_leftover_backend()
    if not wait_for_free_ports():
        print(
            "    the previous run has not released its port; this candidate will attach "
            "to it and is likely to report NO SIGNAL",
            flush = True,
        )

    env = {**os.environ, **extra}
    before = backend_offsets()
    # To a FILE, never subprocess.PIPE. Nothing here reads the pipe while the app runs, so
    # once the app had written enough to fill the 64 KiB buffer it would block on its own
    # stdout: this script would hang the app it is measuring, and the user would see a
    # freeze that the script itself caused.
    app_log = Path(tempfile.mkstemp(suffix = ".log", prefix = "unsloth-freeze-")[1])
    proc = subprocess.Popen(
        cmd,
        env = env,
        stdout = app_log.open("w"),
        stderr = subprocess.STDOUT,
        start_new_session = True,
    )
    started = time.time()
    applied, samples, exited, ran_for = {}, [], None, 0

    print(f"    launching, then watching for {_span(WARMUP + WINDOW)}.", flush = True)
    print("    Use the window normally while this runs.", flush = True)
    try:
        while time.time() - started < WARMUP + WINDOW:
            time.sleep(15)
            if proc.poll() is not None:
                exited = proc.returncode
                ran_for = round(time.time() - started)
                print(
                    f"    the app EXITED (code {exited}) after " f"{time.time() - started:.0f}s",
                    flush = True,
                )
                break
            if not applied:
                applied = applied_env(proc.pid)
            text = backend_tail(before)
            n_mon, n_live = len(MONITOR.findall(text)), len(LIVENESS.findall(text))
            samples.append((round(time.time() - started), n_mon, n_live))
            if len(samples) % 2 == 0:
                print(
                    f"    t={samples[-1][0]:4}s  interface={n_mon:3}  watchdog={n_live:3}",
                    flush = True,
                )
    except KeyboardInterrupt:
        print("    interrupted; recording what was collected so far", flush = True)
    finally:
        alive = proc.poll() is None
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
    shell_out = app_log.read_text(errors = "replace")
    n_mon, n_live = len(MONITOR.findall(text)), len(LIVENESS.findall(text))

    # Did the interface stop polling partway through while the watchdog carried on? That is
    # the reported symptom, and a total that looks healthy can still hide it.
    stalled_at = None
    for i in range(1, len(samples)):
        if samples[i][1] == samples[i - 1][1] and samples[i][2] > samples[i - 1][2]:
            stalled_at = samples[i][0]
            break

    pre_lines = [l for l in (text + shell_out).splitlines() if "desktop_preflight completed" in l]
    preflight = pre_lines[-1].strip() if pre_lines else ""

    if not ran_for:
        ran_for = samples[-1][0] if samples else 0

    if exited == 0 and ran_for <= 20:
        # A clean, immediate exit is almost always the single-instance guard: another copy
        # of Unsloth is already open, so this launch handed over and quit. Calling that
        # "crashed" would be both wrong and alarming, and it is the likeliest thing to go
        # wrong for someone running this on their own desktop.
        verdict = (
            "SKIPPED: the app exited immediately and cleanly, which usually means "
            "another copy of Unsloth is already running. Close it and re-run"
        )
    elif exited == 0:
        # Ran a while and then exited cleanly. Single instance handover is immediate, so
        # this is not that; the likeliest cause is simply that the window was closed.
        verdict = (
            f"ENDED EARLY: the app ran for {ran_for}s and then exited cleanly. If you "
            f"closed the window, just re-run and leave it open"
        )
    elif exited is not None and not _has_display(extra):
        # Over plain SSH there is nothing to draw on. Calling that a crash starts a bug
        # hunt for a bug that is not there.
        verdict = (
            f"CANNOT RUN: the app exited (code {exited}) and there is no display to draw "
            f"on. Run this from a desktop session, not over plain SSH"
        )
    elif exited is not None:
        verdict = f"CRASHED: the app exited on its own (code {exited})"
    elif n_mon == 0 and n_live == 0:
        # Do not guess the cause: the preflight line the app already printed says which
        # of the two it is, and naming the wrong one sends the user off fixing nothing.
        if not preflight and not applied:
            why = (
                "the desktop shell never started. If you launched `unsloth studio`, that "
                "is the command line version and not the app that freezes; re-run this "
                "with the path to Unsloth Desktop"
            )
        elif "NotInstalled" in preflight:
            why = (
                "Unsloth Studio itself is not installed, so there is no backend to "
                "observe. Open the app once and let it finish installing, then re-run"
            )
        elif "AttachedReady" in preflight or "OwnedReady" in preflight:
            why = (
                "the app attached to a backend that was already running, which nothing "
                "is recording. Close every copy of Unsloth and re-run"
            )
        else:
            why = "the backend never started, so there was nothing to observe"
        verdict = f"NO SIGNAL: this run measured nothing, because {why}"
    elif n_live > 0 and n_mon == 0:
        verdict = "FROZE: the interface never polled at all while the app kept running"
    elif stalled_at is not None:
        verdict = (
            f"FROZE: the interface stopped polling at about {stalled_at}s "
            f"while the watchdog kept going"
        )
    elif n_live >= 3 and n_mon * 3 < n_live:
        verdict = "SUSPECT: the interface polled far less than the watchdog"
    else:
        verdict = "OK: the interface kept polling for the whole run"

    print(f"    VERDICT: {verdict}", flush = True)
    return {
        "candidate": label,
        "why": why,
        "env": extra,
        "verdict": verdict,
        "preflight": scrub(preflight) if preflight else "(not seen)",
        "applied_by_app": applied,
        "interface_polls": n_mon,
        "watchdog_polls": n_live,
        "exit_code": exited,
        "samples": samples,
        "backend_log_excerpt": scrub("\n".join(text.splitlines()[-40:])),
    }


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
    if not shutil.which(cmd[0]) and not Path(cmd[0]).is_file():
        print(
            f"cannot find {cmd[0]!r} on PATH. Pass the command explicitly, for example:\n"
            f"  python3 {Path(__file__).name} ~/Applications/Unsloth-Desktop.AppImage"
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

    if port_busy():
        print("NOTE: something is already listening on the Studio port. Close any running")
        print("Unsloth (including `unsloth studio` in another terminal) first, or the app")
        print("will attach to it and this script will not be able to measure anything.\n")

    facts = host_facts()
    print(f"  session : {facts['session_type']}   desktop: {facts['desktop']}")
    print(f"  gpus    : {'; '.join(facts['gpus']) or '(none reported)'}")
    print(f"  driver  : {facts['nvidia_driver'] or '(no nvidia-smi)'}")

    results = []
    for label, extra, why in CANDIDATES:
        try:
            results.append(run_candidate(label, extra, why, cmd))
        except KeyboardInterrupt:
            print("\n  skipped by user", flush = True)

    out = Path.cwd() / f"unsloth-freeze-report-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(
        json.dumps({"host": json.loads(scrub(json.dumps(facts))), "results": results}, indent = 2)
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
