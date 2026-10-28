# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Workflows that boot Studio and click through it run with update checks off.

A llama.cpp, whisper.cpp or audio.cpp release published while the job runs makes the prebuilt it just
installed one behind, and Studio then shows an "update available" card in the bottom-right overlay rail,
over the controls the scripts click. The click times out and the job fails for a reason unrelated to the
change under test. Seen in studiobench (llama.cpp b11681) and in the Chat UI smoke (audio.cpp v0.9.2).
"""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
GUARDED = ("studiobench-ci.yml", "studio-ui-smoke.yml")
SWITCH = "UNSLOTH_DISABLE_UPDATE_CHECK"


def _jobs_that_boot_studio(workflow):
    jobs = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding = "utf-8"))["jobs"]
    booting = {
        name: job
        for name, job in jobs.items()
        if any("boot-studio" in str(step.get("run", "")) for step in job.get("steps", []))
    }
    assert booting, f"no {workflow} job boots Studio any more; update this guard"
    return booting


@pytest.mark.parametrize("workflow", GUARDED)
def test_every_studio_booting_job_turns_update_checks_off(workflow):
    for name, job in _jobs_that_boot_studio(workflow).items():
        env = job.get("env") or {}
        assert str(env.get(SWITCH)) == "1", (
            f"{workflow} job {name!r} boots Studio without {SWITCH}=1, so a release published "
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
