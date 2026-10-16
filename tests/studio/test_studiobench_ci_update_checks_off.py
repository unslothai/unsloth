# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""studiobench's real-path session boots Studio with update checks off.

A llama.cpp release published while the job runs makes the prebuilt it just installed one behind, and
Studio then shows an "update available" card in the bottom-right overlay rail, over the controls the
scene clicks. The click times out and the session fails for a reason unrelated to the change under test.
"""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "studiobench-ci.yml"
SWITCH = "UNSLOTH_DISABLE_UPDATE_CHECK"


def _jobs_that_boot_studio():
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"]
    booting = {
        name: job
        for name, job in jobs.items()
        if any("boot-studio" in str(step.get("run", "")) for step in job.get("steps", []))
    }
    assert booting, "no studiobench job boots Studio any more; update this guard"
    return booting


def test_every_studio_booting_job_turns_update_checks_off():
    for name, job in _jobs_that_boot_studio().items():
        env = job.get("env") or {}
        assert str(env.get(SWITCH)) == "1", (
            f"studiobench job {name!r} boots Studio without {SWITCH}=1, so a llama.cpp release "
            "published mid-run can put an update card over the controls the scene clicks"
        )


def test_the_switch_is_what_the_backend_reads():
    # A renamed variable would leave the job green and the banner back.
    source = (REPO / "studio" / "backend" / "utils" / "update_status.py").read_text(
        encoding = "utf-8"
    )
    assert f'"{SWITCH}"' in source
