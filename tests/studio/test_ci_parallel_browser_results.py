# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/studio-frontend-ci.yml"
pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason = "requires bash")


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
        ["bash", "-e", "-o", "pipefail", "-c", stub + script],
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
