#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Measure where Unsloth Studio's startup time goes, per platform.

Motivation: nothing measured this. The backend logs "lifespan startup completed in
X ms" (studio/backend/main.py) but no test or CI job ever asserted a budget, and
studio_test_kit polls /healthz in a loop that discards the elapsed time -- its
default healthz_timeout_s of 180 was the only recorded expectation.

A first local run (Linux, warm cache, fast server CPU) found `import main` alone
costs 6.6s before the server can bind, dominated by eager module-level imports:

    torch          1930 ms self
    unsloth_zoo     914 ms self
    routes          779 ms self
    transformers    524 ms self

pulled in transitively by the `routes` package (routes.training ->
utils.models.model_config, routes.models -> unsloth_zoo). A laptop is slower. That
is the number this script exists to track, alongside the phases around it.

Phases measured:
  import   `python -X importtime -c "import main"`, top cumulative + per-package self
  spawn    process start -> first byte on stdout
  healthz  process start -> /api/health (or /healthz) answers 200
  lifespan the backend's own "lifespan startup completed in X ms" log line

Usage:
    python scripts/profile_startup.py --repeats 3 --json out.json
    python scripts/profile_startup.py --import-only     # no server, no port needed

Exit code is 0 unless --max-healthz-seconds is given and exceeded, so this can be
turned into a regression gate once budgets are agreed.
"""

from __future__ import annotations

import argparse
import json
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

    Run in a subprocess with -X importtime because the numbers are only meaningful
    for a cold interpreter -- importing in-process would measure an already-warm
    sys.modules and report near zero.
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
        # -X importtime still prints a row for every module that finished
        # importing before the failure -- including `main` itself when its own
        # body raises. The surviving rows therefore describe a partial graph,
        # and the largest cumulative one need not be `import main` at all, so
        # reporting a total here would publish a plausible but wrong number.
        return {
            "ok": False,
            "error": (proc.stderr or proc.stdout)[-2000:],
            "partial_rows": len(rows),
        }

    by_cum = sorted(rows, key = lambda r: -r[1])
    # The total has to come from the row named `main`, not from the largest row.
    # -X importtime also prints the interpreter's own startup graph (`site`,
    # `encodings`, and anything a venv's sitecustomize drags in), which is not
    # part of `import main`. Today main dwarfs those, but the two are not ordered
    # by construction: with a trivial main this file's old `by_cum[0]` reported
    # `site`'s 0.027s as `import main` while main actually cost 0.000249s. Read
    # the labelled row so the headline number cannot silently become a different
    # module's cost as backend imports get optimized.
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

    CI profiles `Scripts/unsloth.exe`, and a pip console-script .exe is a distlib
    launcher stub: it parses its own shebang, CreateProcess's the venv python on
    the zip appended to itself, and just waits. terminate() therefore reaps the
    stub only, leaving the real backend holding the inherited stdout handle -- so
    the reader thread never sees EOF and burns the full reader.join(timeout=10),
    and with --repeats every iteration strands another server on the shared
    UNSLOTH_STUDIO_HOME. taskkill /T walks the tree, matching the cleanup already
    used in unsloth_cli/commands/start.py and unsloth/dataprep/synthetic.py.
    """
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output = True,
                timeout = 30,
                check = False,
            )
            return
        except Exception:
            # taskkill missing or timed out; fall through so the stub still dies.
            pass
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
        # Has to run concurrently with the health polling, for two reasons: the
        # first read is what timestamps the spawn phase, and nothing else reads
        # this pipe, so a backend that logs more than the OS pipe buffer (64 KiB
        # on Linux) would block in write() before it ever binds the port.
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
            # Safe to wait() rather than communicate(): the reader thread is
            # already draining the pipe, so the child cannot block on write().
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

    The workflow deliberately points --python at the installed Studio venv while
    the script itself runs under the runner's system python, so reporting
    platform.python_version() would label the timings with the wrong version.
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
    # range(0) launches nothing, so an empty runs list reaches the budget check as
    # "no healthz measurement", warns and exits 0: a gate that cannot fail. Reject
    # the value instead, since --repeats comes straight from a dispatch input.
    if a.repeats < 1:
        ap.error("--repeats must be at least 1")

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
            # A launch that never became healthy has to fail the budget, not be
            # filtered out of it: dropping it would leave the median and max
            # computed from the surviving (faster) runs, and dropping all of
            # them would make the gate exit 0 no matter how broken startup is.
            print(
                f"::error::startup regression: {failed} of {len(launch.get('runs') or [])} "
                f"launches never became healthy within the timeout"
            )
            return 1
        if med is None:
            print("::warning::no healthz measurement; not enforcing the budget")
        elif med > a.max_healthz_seconds:
            print(
                f"::error::startup regression: {med}s median to a healthy port "
                f"exceeds the {a.max_healthz_seconds}s budget"
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
