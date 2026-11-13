# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Workflows that boot Studio and click through it run with update checks off.

A llama.cpp, whisper.cpp or audio.cpp release published while the job runs makes the prebuilt it just
installed one behind, and Studio then shows an "update available" card in the bottom-right overlay rail,
over the controls the scripts click. The click times out and the job fails for a reason unrelated to the
change under test. Seen in studiobench (llama.cpp b11681), the Chat UI smoke (audio.cpp v0.9.2), and the Windows UI
lane and the UI parity null control.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
SWITCH = "UNSLOTH_DISABLE_UPDATE_CHECK"
# A job drives a browser if it installs Playwright's browsers or runs studiobench; booting Studio through the
# shared boot-studio step counts too. Windows lanes boot Studio inside their own script, so the browser half
# is what finds them.
_BROWSER = re.compile(r"playwright install|-m studiobench|studiobench\.")
# Lanes that must stay covered: if discovery stops finding one, the guard has gone blind, not green.
KNOWN = {
    ("studio-ui-smoke.yml", "ui-smoke"),
    ("studio-windows-ui-smoke.yml", "ui-smoke"),
    ("studio-mac-ui-smoke.yml", "ui-smoke"),
    ("studiobench-ci.yml", "liveness"),
    ("studiobench-ui-parity.yml", "arms"),
}


def _browser_jobs():
    found = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        jobs = yaml.safe_load(path.read_text(encoding = "utf-8")).get("jobs") or {}
        for name, job in jobs.items():
            runs = " ".join(str(step.get("run", "")) for step in job.get("steps") or [])
            boots = "boot-studio" in runs and "playwright" in runs.lower()
            if boots or _BROWSER.search(runs):
                found[(path.name, name)] = job
    return found


def test_discovery_still_finds_the_ui_lanes():
    missing = KNOWN - set(_browser_jobs())
    assert (
        not missing
    ), f"no longer recognised as browser jobs: {sorted(missing)}; update this guard"


@pytest.mark.parametrize("key", sorted(_browser_jobs()), ids = lambda k: f"{k[0]}:{k[1]}")
def test_every_browser_job_turns_update_checks_off(key):
    workflow, name = key
    job = _browser_jobs()[key]
    env = job.get("env") or {}
    assert str(env.get(SWITCH)) == "1", (
        f"{workflow} job {name!r} drives a browser without {SWITCH}=1, so a release published "
        "mid-run can put an update card over the controls its scripts click"
    )
    for step in job.get("steps", []):
        # A step-level override would quietly turn the checks back on for that step.
        step_env = step.get("env") or {}
        assert (
            SWITCH not in step_env or str(step_env[SWITCH]) == "1"
        ), f"{workflow} job {name!r} step {step.get('name')!r} sets {SWITCH} back on"


def test_the_switch_is_what_the_backend_reads():
    # A renamed variable would leave the job green and the banner back.
    source = (REPO / "studio" / "backend" / "utils" / "update_status.py").read_text(
        encoding = "utf-8"
    )
    assert f'"{SWITCH}"' in source
