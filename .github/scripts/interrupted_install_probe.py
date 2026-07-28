#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""After an install is interrupted, decide whether the desktop app WOULD report the
resulting venv as healthy -- reproducing the Tauri preflight probes so the regression
is testable without building the app.

One implementation for all three platforms. There were briefly two (a shell probe and
an inline PowerShell one), and they diverged: the PowerShell version only ran `-h` and
`desktop-capabilities`, so it could not see the `studio_install_ok`, `verify-install`
or `desktop-runtime-check` signals that the fix PRs introduce -- it would have
reported those PRs as failing no matter how well they worked. A probe that cannot
observe the fix is worse than no probe, hence a single shared one.

The reported bug: quitting the app during the dependency pass SIGTERMs the installer
(install.rs stop_install). Landing in the "studio deps" step drops
studio/backend/requirements/studio.txt, where structlog is declared. Preflight then
probes `unsloth -h` (preflight/managed.rs:419) and `studio desktop-capabilities`
(managed.rs:318); both SUCCEED because typer/click/rich are core, so the app reports
ManagedReady with can_auto_repair=false and the backend dies on `import structlog`.

Verdicts:
  HEALTHY      the backend boots -- the interruption did no lasting harm
  REPAIRABLE   the backend is broken AND something reports it, so the app can repair
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


def run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output = True, text = True, timeout = timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (subprocess.TimeoutExpired, OSError) as e:
        return 127, f"{type(e).__name__}: {e}"


def has_subcommand(bin_path: str, args: list[str]) -> bool:
    """Whether the CLI understands a subcommand at all. Older builds do not have the
    newer verify commands, and 'absent' must not be confused with 'reported failure'."""
    rc, _ = run([bin_path, *args, "--help"], timeout = 60)
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
    ap.add_argument("--boot-timeout", type = int, default = 120)
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
    rc, log = run([binp, "-h"], timeout = 180)
    (out / "cli-h.log").write_text(log, encoding = "utf-8", errors = "replace")
    say("cli_h_ok", rc == 0)

    rc, caps_raw = run([binp, "studio", "desktop-capabilities", "--json"], timeout = 180)
    (out / "desktop-capabilities.json").write_text(caps_raw, encoding = "utf-8", errors = "replace")
    say("capabilities_ok", rc == 0)

    # studio_install_ok is added by the install-manifest work; absent on older trees,
    # which is different from present-and-false.
    install_ok: object = "absent"
    try:
        # The CLI may print a banner before the JSON, so start at the first brace.
        brace = caps_raw.find("{")
        if brace >= 0:
            v = json.loads(caps_raw[brace:]).get("studio_install_ok")
            install_ok = "absent" if v is None else bool(v)
    except (json.JSONDecodeError, AttributeError):
        pass
    say("capabilities.studio_install_ok", install_ok)

    # ── the deeper probes the fix PRs add ────────────────────────────────────
    for label, args in (
        ("verify_install", ["studio", "verify-install"]),
        ("desktop_runtime_check", ["studio", "desktop-runtime-check"]),
    ):
        if not has_subcommand(binp, args):
            say(label, "absent")
            continue
        rc, log = run([binp, *args], timeout = 300)
        (out / f"{label}.log").write_text(log, encoding = "utf-8", errors = "replace")
        say(label, "ok" if rc == 0 else "failed")

    # The in-progress marker #7490 writes before spawning the installer. RECORDED ONLY,
    # never used as repair evidence: both interrupt drivers seed it before every install
    # and deliberately never clear it, so it is true on every leg by construction. Using
    # it in the verdict below would make REPAIRABLE unconditional and FALSE_READY -- the
    # single outcome this workflow exists to catch -- unreachable.
    home = Path(os.environ.get("UNSLOTH_STUDIO_HOME") or (Path.home() / ".unsloth" / "studio"))
    say("install_in_progress_marker", (home / ".desktop-install-in-progress").exists())

    # ── ground truth: does the backend actually boot? ────────────────────────
    # Own the whole process tree: the CLI spawns uvicorn/python children, and
    # terminating only the parent leaves them holding the port, so the next leg's
    # probe would hang. Same reason the interrupt driver kills the group.
    popen_kw: dict = {}
    if os.name == "posix":
        popen_kw["start_new_session"] = True
    else:
        popen_kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    # Straight to the artefact file, never a PIPE: nothing reads that pipe until after
    # the polling loop, so a backend whose imports emit more than the OS pipe buffer
    # (64 KiB on Linux and macOS, a single page by default on Windows) blocks on write
    # BEFORE it binds the port. backend_ok, which this whole verdict pivots on, would
    # then be false for a perfectly good install.
    blog_path = out / "backend.log"
    blog_fh = blog_path.open("w", encoding = "utf-8", errors = "replace")
    proc = subprocess.Popen(
        [binp, "studio", "--api-only", "-H", "127.0.0.1", "-p", str(port)],
        stdout = blog_fh,
        stderr = subprocess.STDOUT,
        text = True,
        **popen_kw,
    )
    backend_ok = False
    deadline = time.time() + a.boot_timeout
    while time.time() < deadline:
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
        if os.name == "posix":
            import signal
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                try:
                    proc.wait(timeout = 10)
                    return
                except subprocess.TimeoutExpired:
                    continue
        else:
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
    if backend_ok:
        verdict = "HEALTHY"
    elif (
        facts.get("verify_install") == "failed"
        or facts.get("desktop_runtime_check") == "failed"
        or facts.get("capabilities.studio_install_ok") is False
        or not facts.get("cli_h_ok")
        or not facts.get("capabilities_ok")
    ):
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
        print("[probe] broken install is detectable -> the desktop app can auto-repair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
