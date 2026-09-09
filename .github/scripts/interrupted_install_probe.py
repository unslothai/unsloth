#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""After an install is interrupted, decide whether the desktop app WOULD report the
resulting venv as healthy -- reproducing the Tauri preflight probes so the regression
is testable without building the app.

The reported bug: quitting the app during the dependency pass SIGTERMs the installer
(install.rs stop_install), and landing in "studio deps" drops
studio/backend/requirements/studio.txt, where structlog is declared. Preflight probes
`unsloth -h` (managed.rs:419) and `studio desktop-capabilities` (managed.rs:318); both
SUCCEED because typer/click/rich are core, so the app reports ManagedReady with
can_auto_repair=false while the backend dies on `import structlog`.

ONE implementation for all three platforms. The bespoke inline PowerShell probe it
replaced ran only `-h` and `desktop-capabilities`, so it could not observe
`studio_install_ok`, `verify-install` or `desktop-runtime-check`, and would have failed
the very PRs that add them. A probe that cannot see the fix is worse than no probe.

Verdicts:
  HEALTHY      the backend boots AND desktop-capabilities reports the install
               complete -- preflight would report ManagedReady and be right
  REPAIRABLE   the backend is broken AND a probe the DESKTOP consumes reports it,
               so the app can offer a repair
  FALSE_READY  the backend is broken and every probe says ready -> THE BUG

Exit: 0 for HEALTHY/REPAIRABLE/NO_CLI, 1 for FALSE_READY, 2 for a usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def run(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """(rc, stdout, stderr), kept SEPARATE: preflight pipes stdout and sends stderr to
    /dev/null (managed.rs:358), so anything folded in here is text the desktop never sees."""
    try:
        p = subprocess.run(cmd, capture_output = True, text = True, timeout = timeout)
        return p.returncode, p.stdout or "", p.stderr or ""
    except (subprocess.TimeoutExpired, OSError) as e:
        return 127, "", f"{type(e).__name__}: {e}"


def merged(rc_out_err: tuple[int, str, str]) -> str:
    """Both streams, for artefact logs only -- never for parsing."""
    return rc_out_err[1] + rc_out_err[2]


def has_subcommand(bin_path: str, args: list[str]) -> bool:
    """Whether the CLI knows the subcommand: older builds lack the verify commands, and
    'absent' must not read as 'reported failure'."""
    rc, _, _ = run([bin_path, *args, "--help"], timeout = 60)
    return rc == 0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description = __doc__)
    ap.add_argument("bin", help = "path to the unsloth CLI")
    ap.add_argument("--port", type = int, default = 0, help = "0 picks a free port")
    ap.add_argument("--out", default = "probe", help = "directory for probe artefacts")
    # The desktop's own grace, BACKEND_STARTUP_GRACE_PERIOD = 5 min (commands.rs:9): less fails a leg the app would wait
    # out. It cannot mask the bug: a missing import kills the backend and the loop breaks on proc.poll(), so this
    # only bounds a LIVE backend.
    ap.add_argument("--boot-timeout", type = int, default = 300)
    a = ap.parse_args(argv)

    binp = a.bin
    if not Path(binp).exists():
        print(f"::error::unsloth bin not found: {binp}")
        return 2
    out = Path(a.out)
    out.mkdir(parents = True, exist_ok = True)
    port = a.port or free_port()
    facts: dict[str, object] = {}

    def say(k: str, v: object) -> None:
        facts[k] = v
        print(f"[probe] {k:28} = {v}")

    # ── the two probes Tauri preflight actually runs ─────────────────────────
    # The DESKTOP's deadline, not a generous CI one: preflight allows each call 10s
    # (managed.rs:337 for `-h`, :390 for desktop-capabilities) then reports Stale
    # (managed.rs:471, :521). Longer would call a slow torn venv HEALTHY and skip the re-run
    # assertion; run() reports a timeout as a non-zero rc, the same REPAIRABLE arm.
    PREFLIGHT_TIMEOUT = 10

    t0 = time.time()
    r = run([binp, "-h"], timeout = PREFLIGHT_TIMEOUT)
    (out / "cli-h.log").write_text(merged(r), encoding = "utf-8", errors = "replace")
    say("cli_h_ok", r[0] == 0)
    say("cli_h_seconds", round(time.time() - t0, 2))

    t0 = time.time()
    caps_rc, caps_out, caps_err = run(
        [binp, "studio", "desktop-capabilities", "--json"], timeout = PREFLIGHT_TIMEOUT
    )
    (out / "desktop-capabilities.json").write_text(caps_out, encoding = "utf-8", errors = "replace")
    (out / "desktop-capabilities.stderr.log").write_text(
        caps_err, encoding = "utf-8", errors = "replace"
    )
    say("capabilities_ok", caps_rc == 0)
    say("capabilities_seconds", round(time.time() - t0, 2))

    # Parse EXACTLY as the desktop does: managed.rs:414 hands the whole stdout buffer to serde_json, which rejects
    # leading or trailing non-JSON, and stderr was already discarded at managed.rs:358.
    # Folding stderr in made one warning line enough to fail the parse and report FALSE_READY over an install the real
    # app offers to repair.
    # "absent" (studio_install_ok predates the install manifest) and "unparseable" split only for a readable artefact:
    # the desktop reports Stale for both ("desktop_capability_probe_failed", managed.rs:521).
    # The field is Option<bool> (managed.rs:43), so serde rejects a non-boolean and the whole payload fails to
    # deserialize -> Stale; bool() instead read the JSON string "false" as True and reported HEALTHY over a torn
    # install. Only a literal JSON true counts.
    install_ok: object = "absent"
    try:
        parsed = json.loads(caps_out)
        if not isinstance(parsed, dict):
            install_ok = "unparseable"
        else:
            v = parsed.get("studio_install_ok")
            if v is None:
                install_ok = "absent"
            elif isinstance(v, bool):
                install_ok = v
            else:
                install_ok = "non-boolean"
    except json.JSONDecodeError:
        install_ok = "unparseable"
    say("capabilities.studio_install_ok", install_ok)

    # The desktop's own conclusion: Ready only on rc 0 plus a true studio_install_ok.
    # The predicate is `!= Some(true)` (managed.rs:445), so an ABSENT field is Stale exactly like a false one;
    # a CLI too old to answer is rejected one check earlier on desktop_manageability_version.
    # Leaving "absent" undecided reported HEALTHY on every booting leg and skipped the repair assertion: the
    # regression `unsloth_cli/commands/studio.py` sits in the path filter to catch.
    caps_ready = caps_rc == 0 and install_ok is True
    say("desktop_would_call_install_ok", caps_ready)

    # ── the deeper probes the fix PRs add ────────────────────────────────────
    # RECORDED, not repair evidence: preflight runs only `-h` and
    # `studio desktop-capabilities --json` (managed.rs:357, :445), so counting these would
    # pass a leg while the real app still reports ManagedReady over a torn install.
    for label, args in (
        ("verify_install", ["studio", "verify-install"]),
        ("desktop_runtime_check", ["studio", "desktop-runtime-check"]),
    ):
        if not has_subcommand(binp, args):
            say(label, "absent")
            continue
        r = run([binp, *args], timeout = 300)
        (out / f"{label}.log").write_text(merged(r), encoding = "utf-8", errors = "replace")
        say(label, "ok" if r[0] == 0 else "failed")

    # The in-progress marker #7490 writes before spawning the installer.
    # RECORDED ONLY: both drivers seed it and never clear it, so it is true on every leg by construction, and using it
    # in the verdict would make FALSE_READY, the one failing outcome, unreachable.
    home = Path(os.environ.get("UNSLOTH_STUDIO_HOME") or (Path.home() / ".unsloth" / "studio"))
    say("install_in_progress_marker", (home / ".desktop-install-in-progress").exists())

    # ── ground truth: does the backend actually boot? ────────────────────────
    # Own the whole process tree: the CLI spawns uvicorn/python children that would hold the port and hang the next
    # leg's probe. Same reason the driver kills the group.
    popen_kw: dict = {}
    if os.name == "posix":
        popen_kw["start_new_session"] = True
    else:
        popen_kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    # Straight to the artefact file, never a PIPE: nothing drains a pipe until after the polling loop, so a backend
    # whose imports outrun the OS buffer (64 KiB on Linux and macOS, one page on Windows) blocks BEFORE binding the
    # port, and backend_ok -- what the verdict pivots on -- would be false for a perfectly good install.
    blog_path = out / "backend.log"
    blog_fh = blog_path.open("w", encoding = "utf-8", errors = "replace")
    # An interrupted install can leave the console script with its venv interpreter gone.
    # An unguarded spawn raises, so no verdict.json is written and both workflows die on json.load.
    # An unlaunchable CLI is a broken backend that `-h` flags.
    proc = None
    try:
        proc = subprocess.Popen(
            [binp, "studio", "--api-only", "-H", "127.0.0.1", "-p", str(port)],
            stdout = blog_fh,
            stderr = subprocess.STDOUT,
            text = True,
            **popen_kw,
        )
    except OSError as e:
        say("backend_spawn_error", f"{type(e).__name__}: {e}")
    backend_ok = False
    deadline = time.time() + a.boot_timeout
    while proc is not None and time.time() < deadline:
        if proc.poll() is not None:
            break
        for path in ("/api/health", "/healthz"):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout = 2) as r:
                    if r.status == 200:
                        backend_ok = True
                        break
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
        if backend_ok:
            break
        time.sleep(1)

    def reap() -> None:
        if proc is None:
            return
        if os.name == "posix":
            import signal

            # start_new_session made this child its own group leader.
            # Read the pgid BEFORE the reap: once waited on, os.getpgid() raises and escalation hits nothing.
            try:
                pgid = os.getpgid(proc.pid)
            except OSError:
                pgid = proc.pid
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(pgid, sig)
                except OSError:
                    pass
                try:
                    proc.wait(timeout = 10)
                    break
                except subprocess.TimeoutExpired:
                    continue
            # Unconditional, and to the GROUP, the same escalation interrupt-install.sh makes. The leader exits
            # promptly on SIGTERM while a uvicorn worker does not, so returning once proc.wait() succeeded left that
            # worker holding the port and venv while the repair reinstalled underneath. Signalling an empty group is
            # a no-op.
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
        else:
            # On win32 the CLI re-spawns the server as a CHILD and waits on it (unsloth_cli/commands/studio.py:1543),
            # and CREATE_NEW_PROCESS_GROUP does not make terminate() reach descendants, so killing the wrapper alone
            # leaves the venv locked against the repair. taskkill /T takes the tree.
            run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], timeout = 30)
            try:
                proc.wait(timeout = 10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout = 10)
                except subprocess.TimeoutExpired:
                    proc.kill()

    reap()
    blog_fh.close()
    blog = blog_path.read_text(encoding = "utf-8", errors = "replace")
    say("backend_ok", backend_ok)

    missing = ""
    for line in blog.splitlines():
        if "ModuleNotFoundError" in line:
            missing = line.strip()
    if missing:
        say("backend_error", missing)

    # ── verdict ──────────────────────────────────────────────────────────────
    # A booting backend is not enough. The manifest is written LAST
    # (install_python_stack.py:3255), so the data-designer leg boots while
    # desktop-capabilities still says studio_install_ok=false and preflight reports Stale
    # (managed.rs:445); calling that HEALTHY skipped the re-run step. `-h` gates it for the
    # same reason: probe_managed_bin runs it FIRST and returns Stale "cli_unusable" without
    # reaching the capability probe (managed.rs:465-478), so consulting cli_h_ok only in the
    # repairable arm called a help-less CLI HEALTHY.
    if backend_ok and caps_ready and facts.get("cli_h_ok"):
        verdict = "HEALTHY"
    elif not caps_ready or not facts.get("cli_h_ok"):
        verdict = "REPAIRABLE"
    else:
        verdict = "FALSE_READY"

    facts["verdict"] = verdict
    (out / "verdict.json").write_text(json.dumps(facts, indent = 2), encoding = "utf-8")
    print(f"[probe] VERDICT = {verdict}")

    if verdict == "FALSE_READY":
        print(
            "::error::Interrupted install reports READY but the backend cannot boot"
            f" ({missing or 'import failure'}). Preflight sees -h ok + desktop-capabilities"
            " ok, so the app shows ManagedReady with can_auto_repair=false and the user"
            " is stuck."
        )
        return 1
    if verdict == "REPAIRABLE":
        print("[probe] incomplete install is detectable -> the desktop app can auto-repair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
