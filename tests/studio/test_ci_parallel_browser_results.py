# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest
import yaml


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/studio-frontend-ci.yml"


def _workflow_bash():
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")
    if not git:
        return None
    candidate = Path(git).resolve().parent.parent / "bin" / "bash.exe"
    return str(candidate) if candidate.is_file() else None


BASH = _workflow_bash()
pytestmark = pytest.mark.skipif(not BASH, reason = "requires the workflow's bash shell")


@pytest.mark.parametrize("failing", ["", "chromium", "firefox"])
def test_browser_children_are_isolated_and_both_results_gate(tmp_path, failing):
    steps = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"]["windows"]["steps"]
    script = next(
        s["run"]
        for s in steps
        if s.get("name") == "Data settings deletion choices in Chromium and Firefox"
    )
    stub = r"""
python() {
  printf '%s %s %s %s\n' "$PW_ENGINE" "$PW_PORT" "$PW_OUT" "$VITE_TEST_CACHE_DIR"
  touch "$PW_ENGINE.ready"
  for attempt in $(seq 1 100); do
    [ -e chromium.ready ] && [ -e firefox.ready ] && break
    sleep 0.02
  done
  [ -e chromium.ready ] && [ -e firefox.ready ] || return 9
  [ "$PW_ENGINE" != "$FAIL_ENGINE" ] || return 7
}
"""
    result = subprocess.run(
        [BASH, "-e", "-o", "pipefail", "-c", stub + script],
        cwd = tmp_path,
        env = {**os.environ, "RUNNER_TEMP": str(tmp_path), "FAIL_ENGINE": failing},
        text = True,
        capture_output = True,
        timeout = 15,
    )
    assert result.returncode == bool(failing), result.stdout + result.stderr
    assert "chromium 5420 logs/data_settings_report.json" in result.stdout
    assert "firefox 5421 logs/data_settings_firefox_report.json" in result.stdout
    assert f"{tmp_path}/vite-data-chromium" in result.stdout
    assert f"{tmp_path}/vite-data-firefox" in result.stdout
    assert (tmp_path / "chromium.ready").exists()
    assert (tmp_path / "firefox.ready").exists()


def _linux_script():
    steps = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"]["build"]["steps"]
    return next(
        s["run"]
        for s in steps
        if s.get("name") == "Browser smoke groups for data settings and frontend controls"
    )


CHECKS = [
    "data_settings",
    "settings_tabs",
    "settings_tabs_blocked",
    "find_in_page",
    "keyboard_shortcuts",
    "strip_ansi_smoke",
    "composer_icons",
    "research_review",
]


def _linux_stubs(tmp_path):
    stub = r"""
import json
import os
from pathlib import Path
import signal
import sys
import time

name = Path(sys.argv[1]).stem.removeprefix("playwright_")
if os.environ.get("PW_CHUNK_FAIL"):
    name += "_blocked"
record = {"args": sys.argv[1:], "env": {key: value for key, value in os.environ.items()
          if key.startswith(("PW_", "VITE_"))}, "pid": os.getpid()}

def cancel(signum, frame):
    Path(f"{name}.stopped").touch()
    sys.exit(128 + signum)

signal.signal(signal.SIGTERM, cancel)
Path(f"{name}.tmp").write_text(json.dumps(record))
Path(f"{name}.tmp").replace(f"{name}.json")
if name in {"data_settings", "settings_tabs"}:
    deadline = time.monotonic() + 5
    while not all(Path(f"{other}.json").exists() for other in ("data_settings", "settings_tabs")):
        if time.monotonic() > deadline:
            sys.exit(9)
        time.sleep(0.01)
if os.environ.get("CANCEL_TEST"):
    time.sleep(60)
with open("order.log", "a") as out:
    out.write(name + "\n")
print(name)
sys.exit(7 if name == os.environ.get("FAIL_CHECK") else 0)
"""
    python = tmp_path / "python3"
    python.write_text(f"#!{sys.executable}\n" + stub, encoding = "utf-8")
    python.chmod(0o755)
    timeout = tmp_path / "timeout"
    timeout.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" > timeout.args\nshift 2\nexec "$@"\n',
        encoding = "utf-8",
    )
    timeout.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "RUNNER_TEMP": str(tmp_path),
    }


@pytest.mark.skipif(os.name == "nt", reason = "Linux workflow uses POSIX tools")
@pytest.mark.parametrize("failing", ["", *CHECKS])
def test_linux_groups_overlap_and_collect_every_result(tmp_path, failing):
    env = {**_linux_stubs(tmp_path), "FAIL_CHECK": failing}
    # Reuse the same logs and cache paths to cover a repeated local run.
    for _ in range(2 if not failing else 1):
        for record in tmp_path.glob("*.json"):
            record.unlink()
        (tmp_path / "order.log").unlink(missing_ok = True)
        result = subprocess.run(
            [BASH, "-e", "-o", "pipefail", "-c", _linux_script()],
            cwd = tmp_path,
            env = env,
            text = True,
            capture_output = True,
            timeout = 15,
        )
        assert result.returncode == bool(failing), result.stdout + result.stderr
        # a failing check is named in an annotation, not only inside its group's log
        label = failing.removesuffix("_smoke")
        assert (f"::error title=Browser smoke {label}::" in result.stdout) == bool(
            failing
        ), result.stdout
        records = {name: json.loads((tmp_path / f"{name}.json").read_text()) for name in CHECKS}
        order = (tmp_path / "order.log").read_text().splitlines()
        assert sorted(order) == sorted(CHECKS)
        assert [name for name in order if name != "data_settings"] == CHECKS[1:]
        for name, record in records.items():
            group = "data" if name == "data_settings" else "controls"
            assert record["env"]["VITE_TEST_CACHE_DIR"] == str(tmp_path / f"vite-frontend-{group}")
            label = "strip_ansi" if name == "strip_ansi_smoke" else name
            assert f"{label}: exit {7 if name == failing else 0}" in result.stdout
        blocked = records["settings_tabs_blocked"]["env"]
        assert (blocked["PW_CHUNK_FAIL"], blocked["PW_PORT"], blocked["PW_OUT"]) == (
            "data",
            "5400",
            "logs/settings_tabs_blocked_report.json",
        )
        assert "PW_CHUNK_FAIL" not in records["settings_tabs"]["env"]
        assert records["strip_ansi_smoke"]["env"]["PW_BROWSER"] == "chromium"
        assert records["composer_icons"]["args"][-1] == "chromium"
        assert (tmp_path / "timeout.args").read_text().strip() == (
            "--kill-after=10s 180s python3 tests/studio/playwright_composer_icons.py chromium"
        )
        assert (tmp_path / "logs/playwright-frontend-data.log").is_file()
        assert (tmp_path / "logs/playwright-frontend-controls.log").is_file()


@pytest.mark.skipif(os.name == "nt", reason = "Linux cancellation uses POSIX signals")
def test_linux_cancel_stops_both_active_harnesses(tmp_path):
    env = {**_linux_stubs(tmp_path), "CANCEL_TEST": "1"}
    process = subprocess.Popen(
        [BASH, "-e", "-o", "pipefail", "-c", _linux_script()],
        cwd = tmp_path,
        env = env,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        start_new_session = True,
    )
    records = []
    try:
        deadline = time.monotonic() + 10
        while not all((tmp_path / f"{name}.json").exists() for name in CHECKS[:2]):
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)
        records = [json.loads((tmp_path / f"{name}.json").read_text()) for name in CHECKS[:2]]
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout = 10)
        assert process.returncode == 143, stdout + stderr
        # the groups write to files: a cancelled or timed-out step still prints what both had run
        assert "::group::data_settings" in stdout and "::group::settings_tabs" in stdout, (
            stdout + stderr
        )
        for name, record in zip(CHECKS[:2], records):
            assert (tmp_path / f"{name}.stopped").exists()
            with pytest.raises(ProcessLookupError):
                os.kill(record["pid"], 0)
        assert not (tmp_path / "settings_tabs_blocked.json").exists()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout = 5)
        for record_path in tmp_path.glob("*.json"):
            record = json.loads(record_path.read_text())
            try:
                os.kill(record["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_linux_groups_step_times_out_before_the_job():
    """A job timeout is a cancellation, which skips the failure-only artifact upload; the step's
    own bound fires first, so a hung check fails the step and its logs are uploaded."""
    job = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"]["build"]
    step = next(
        s
        for s in job["steps"]
        if s.get("name") == "Browser smoke groups for data settings and frontend controls"
    )
    assert 0 < step["timeout-minutes"] < job["timeout-minutes"]
