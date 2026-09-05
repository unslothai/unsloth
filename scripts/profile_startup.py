#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Measure where Unsloth Studio's startup time goes, per platform.

Nothing measured this before: the backend logs "lifespan startup completed in X ms"
but no test or CI job asserted a budget, and studio_test_kit discards the elapsed
time of its /healthz poll. A first local run (Linux, warm cache, fast server CPU)
found `import main` alone costs 6.6s before the server can bind, dominated by eager
module-level imports pulled in by the `routes` package:

    torch          1930 ms self
    unsloth_zoo     914 ms self
    routes          779 ms self
    transformers    524 ms self

Phases measured:
  import   `python -X importtime -c "import main"`, top cumulative + per-package self
  spawn    process start -> first byte on stdout
  healthz  process start -> /api/health (or /healthz) answers 200
  lifespan the backend's own "lifespan startup completed in X ms" log line

Usage:
    python scripts/profile_startup.py --repeats 3 --json out.json
    python scripts/profile_startup.py --import-only     # no server, no port needed

Exit code is 0 unless --max-healthz-seconds is given and exceeded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "studio" / "backend"

_IMPORTTIME_RE = re.compile(r"import time:\s+(\d+)\s+\|\s+(\d+)\s+\|(\s*)(\S.*)")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def profile_imports(python: str, top: int = 15) -> dict:
    """Cumulative and self import cost for the backend's module graph.

    Run in a subprocess with -X importtime: the numbers are only meaningful for a
    cold interpreter, and importing in-process would measure a warm sys.modules.
    """
    proc = subprocess.run(
        [python, "-X", "importtime", "-c", "import sys; sys.path.insert(0, '.'); import main"],
        cwd = BACKEND,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    rows = []
    for line in proc.stderr.splitlines():
        m = _IMPORTTIME_RE.match(line)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), m.group(4).strip()))
    if not rows:
        return {"ok": False, "error": (proc.stderr or proc.stdout)[-2000:]}
    if proc.returncode != 0:
        # Rows survive up to the failure, so any total from a partial graph is wrong.
        return {
            "ok": False,
            "error": (proc.stderr or proc.stdout)[-2000:],
            "partial_rows": len(rows),
        }

    by_cum = sorted(rows, key = lambda r: -r[1])
    # Total comes from the `main` row, not by_cum[0]: -X importtime also prints the interpreter's
    # own startup graph (`site`), which can outrank a trivial main.
    main_row = next((r for r in reversed(rows) if r[2] == "main"), None)
    if main_row is None:
        return {
            "ok": False,
            "error": "no `import main` row in -X importtime output\n"
            + (proc.stderr or proc.stdout)[-2000:],
        }
    self_by_pkg: dict[str, int] = {}
    for self_us, _cum, name in rows:
        pkg = name.split(".")[0]
        self_by_pkg[pkg] = self_by_pkg.get(pkg, 0) + self_us

    return {
        "ok": True,
        "total_seconds": round(main_row[1] / 1e6, 3),
        "top_cumulative": [
            {"module": n, "seconds": round(c / 1e6, 3)} for _s, c, n in by_cum[:top]
        ],
        "self_by_package_ms": {
            k: round(v / 1000) for k, v in sorted(self_by_pkg.items(), key = lambda x: -x[1])[:top]
        },
    }


def _terminate_tree(proc: subprocess.Popen) -> None:
    """Stop the server AND its children, which on Windows are a separate process.

    CI profiles `Scripts/unsloth.exe`, a distlib launcher stub that CreateProcess's
    the venv python and waits, so terminate() reaps the stub only: the real backend
    keeps the inherited stdout handle, the reader thread never sees EOF, and
    --repeats strands one server per iteration on the shared UNSLOTH_STUDIO_HOME.
    taskkill /T walks the tree, as unsloth_cli/commands/start.py already does.
    """
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            killed = subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output = True,
                timeout = 30,
                check = False,
            )
            if killed.returncode == 0:
                return
        except Exception:
            # taskkill missing or timed out; fall through so the stub still dies.
            pass
        # check=False: a nonzero taskkill does not raise, so fall through as well.
    proc.terminate()


def profile_launch(
    bin_path: str,
    port: int,
    timeout_s: int = 300,
) -> dict:
    """Spawn the backend the way the desktop app does and time it to first 200."""
    log_lines: list[str] = []
    first_byte: list[float] = []
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [bin_path, "studio", "--api-only", "-H", "127.0.0.1", "-p", str(port)],
        cwd = REPO_ROOT,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        bufsize = 1,
    )

    def _drain() -> None:
        # Runs alongside the health polling: the first read timestamps the spawn phase, and an undrained pipe
        # blocks the backend before it binds.
        for line in proc.stdout:
            if not first_byte:
                first_byte.append(time.perf_counter() - t0)
            log_lines.append(line.rstrip("\n"))

    reader = threading.Thread(target = _drain, daemon = True)
    reader.start()

    t_healthz = None
    deadline = t0 + timeout_s
    try:
        while time.perf_counter() < deadline:
            if proc.poll() is not None:
                break
            if t_healthz is None:
                for url in (
                    f"http://127.0.0.1:{port}/api/health",
                    f"http://127.0.0.1:{port}/healthz",
                ):
                    try:
                        with urllib.request.urlopen(url, timeout = 2) as r:
                            if r.status == 200:
                                t_healthz = time.perf_counter() - t0
                                break
                    except (urllib.error.URLError, OSError, TimeoutError):
                        pass
            if t_healthz is not None:
                break
            time.sleep(0.25)
    finally:
        _terminate_tree(proc)
        try:
            # Safe: the reader drains the pipe, so the child cannot block on write().
            proc.wait(timeout = 30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        reader.join(timeout = 10)

    t_first_byte = first_byte[0] if first_byte else None
    lifespan_ms = None
    for line in log_lines:
        m = re.search(r"lifespan startup completed in ([\d.]+)ms", line)
        if m:
            lifespan_ms = float(m.group(1))
    return {
        "spawn_seconds": round(t_first_byte, 3) if t_first_byte is not None else None,
        "healthz_seconds": round(t_healthz, 3) if t_healthz is not None else None,
        "lifespan_ms": lifespan_ms,
        "reached_healthz": t_healthz is not None,
        "log_tail": log_lines[-25:],
    }


def python_version_of(python: str) -> str:
    """Version of the interpreter that runs the imports, not the one running us.

    --python points at the installed Unsloth venv while this script runs under the
    runner's system python, so platform.python_version() would label it wrong.
    """
    if python == sys.executable:
        return platform.python_version()
    try:
        proc = subprocess.run(
            [python, "-c", "import platform; print(platform.python_version())"],
            capture_output = True,
            text = True,
            timeout = 60,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def find_bin() -> str | None:
    home = os.environ.get("UNSLOTH_STUDIO_HOME") or str(Path.home() / ".unsloth" / "studio")
    names = ["unsloth.exe", "unsloth"] if platform.system() == "Windows" else ["unsloth"]
    subdirs = ["unsloth_studio/Scripts", "unsloth_studio/bin", "bin", "Scripts"]
    for sd in subdirs:
        for n in names:
            p = Path(home) / sd / n
            if p.exists():
                return str(p)
    return shutil.which("unsloth")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--repeats",
        type = int,
        default = 1,
        help = "launch repeats; the median is reported (imports are measured once)",
    )
    ap.add_argument(
        "--python",
        default = sys.executable,
        help = "interpreter used for the import profile (default: this one)",
    )
    ap.add_argument("--bin", help = "path to the unsloth CLI (default: autodetect)")
    ap.add_argument(
        "--import-only",
        action = "store_true",
        help = "skip the server phases (no install needed beyond the deps)",
    )
    ap.add_argument(
        "--max-healthz-seconds",
        type = float,
        help = "fail if the median time to a healthy port exceeds this",
    )
    ap.add_argument("--json", help = "write the full report here")
    a = ap.parse_args(argv)
    # range(0) launches nothing, leaving the budget check with nothing to fail on.
    if a.repeats < 1:
        ap.error("--repeats must be at least 1")
    # Same reason: --import-only never launches anything.
    if a.import_only and a.max_healthz_seconds is not None:
        ap.error("--max-healthz-seconds cannot be combined with --import-only")
    # nan and inf parse fine as floats but `med > budget` is then always False, so the gate would report success without
    # ever bounding anything.
    if a.max_healthz_seconds is not None and not math.isfinite(a.max_healthz_seconds):
        ap.error("--max-healthz-seconds must be a finite number")

    report: dict = {
        "platform": platform.system().lower(),
        "machine": platform.machine(),
        "python": python_version_of(a.python),
        "cpu_count": os.cpu_count(),
    }

    print("== import graph ==")
    report["imports"] = profile_imports(a.python)
    imp = report["imports"]
    if imp.get("ok"):
        print(f"  import main: {imp['total_seconds']}s")
        for row in imp["top_cumulative"][:8]:
            print(f"    {row['seconds']:7.3f}s  {row['module']}")
        print("  self time by package (ms):")
        for k, v in list(imp["self_by_package_ms"].items())[:8]:
            print(f"    {v:8} ms  {k}")
    else:
        print(f"  FAILED: {imp.get('error', '')[:400]}")

    if not a.import_only:
        bin_path = a.bin or find_bin()
        if not bin_path:
            print(
                "== launch == skipped: no unsloth CLI found "
                "(set UNSLOTH_STUDIO_HOME or pass --bin)"
            )
            report["launch"] = {"skipped": "no unsloth CLI found"}
        else:
            print(f"== launch == {bin_path}")
            runs = []
            for i in range(a.repeats):
                r = profile_launch(bin_path, _free_port())
                runs.append(r)
                print(
                    f"  run {i + 1}: healthz={r['healthz_seconds']}s "
                    f"lifespan={r['lifespan_ms']}ms reached={r['reached_healthz']}"
                )
            got = [r["healthz_seconds"] for r in runs if r["healthz_seconds"] is not None]
            report["launch"] = {
                "runs": runs,
                "failed_runs": sum(1 for r in runs if not r["reached_healthz"]),
                "healthz_median_seconds": round(statistics.median(got), 3) if got else None,
                "healthz_max_seconds": round(max(got), 3) if got else None,
            }
            if got:
                print(
                    f"  median time to healthy port: {report['launch']['healthz_median_seconds']}s"
                )

    if a.json:
        Path(a.json).write_text(json.dumps(report, indent = 2), encoding = "utf-8")
        print(f"\nwrote {a.json}")

    if a.max_healthz_seconds is not None:
        launch = report.get("launch") or {}
        med = launch.get("healthz_median_seconds")
        failed = launch.get("failed_runs") or 0
        if failed:
            # Failed launches fail the budget; dropping them would keep only the fast ones.
            print(
                f"::error::startup regression: {failed} of {len(launch.get('runs') or [])} "
                f"launches never became healthy within the timeout"
            )
            return 1
        if med is None:
            # Nothing measured: exiting 0 would pass a requested budget without a single health request, so fail closed.
            print(
                "::error::startup regression: no healthz measurement, so the "
                f"{a.max_healthz_seconds}s budget was never checked "
                f"({launch.get('skipped') or 'launch phase produced no runs'})"
            )
            return 1
        elif med > a.max_healthz_seconds:
            print(
                f"::error::startup regression: {med}s median to a healthy port "
                f"exceeds the {a.max_healthz_seconds}s budget"
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
