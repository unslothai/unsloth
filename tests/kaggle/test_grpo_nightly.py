# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO runs nightly, and the reason is a measurement rather than a preference.

Over nine T4 sessions the leg hit an intermittent illegal memory access in
vLLM's standby sleep on Turing in FOUR of them. Everything else about it is
sound -- 0.95 utilisation confirmed on both Colab and Kaggle, sleep/wake
surviving three cycles, a non-zero ``reward_std`` on every step once the reward
function stopped saturating -- but a 44% red in front of every PR, for a race no
reader can act on, is exactly how a check gets switched off before the day it is
right. At nightly cadence a clean run still arrives most days and a red one
costs nobody a merge.

So the rules here are about the SHAPE that makes that possible:

* the schedule exists and fires the GRPO leg specifically;
* a leg list REPLACES ``--all-kernels`` rather than filtering after it, because
  the kernel plan and the expected payload count come out of the same call and
  a filter applied afterwards leaves the launcher waiting on payloads nobody
  built;
* the schedule bypasses the sampling gate. A nightly sampled at 15% is a
  weekly, and the difference is invisible until someone goes looking for a
  result that never existed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "kaggle-t4-notebook-ci.yml"
TEXT = WORKFLOW.read_text(encoding = "utf-8")
DOC = yaml.safe_load(TEXT)
TRIGGERS = DOC.get(True) or DOC.get("on")

sys.path.insert(0, str(ROOT / ".github" / "scripts" / "kaggle_t4_ci"))
import legs  # noqa: E402


def test_there_is_a_nightly_schedule():
    assert "schedule" in TRIGGERS, "nothing runs the legs that cannot be per-PR"
    crons = [entry["cron"] for entry in TRIGGERS["schedule"]]
    assert len(crons) == 1, crons
    minute, hour = crons[0].split()[:2]
    assert minute.isdigit() and hour.isdigit(), "the nightly must be a fixed time"


def test_the_nightly_does_not_pile_onto_the_hour_mark():
    """Every scheduled workflow on GitHub asks for :00, and this one has no
    reason to join that queue."""
    minute = TRIGGERS["schedule"][0]["cron"].split()[0]
    assert minute not in ("0", "00", "30"), f"minute {minute} is the mark everyone else picks"


def test_the_schedule_runs_the_grpo_leg():
    assert "github.event_name == 'schedule' && 'grpo'" in TEXT, (
        "the schedule must select the leg; a nightly that runs the wired set "
        "is just another copy of the per-PR run"
    )


def test_the_leg_the_nightly_names_exists():
    """A typo here produces a build that selects nothing and a run that proves
    nothing, with no error anywhere."""
    named = set(re.findall(r"&& '([a-z_]+)' \|\| ''", TEXT))
    assert named, "no scheduled leg name found at all"
    for name in named:
        assert name in legs.LEGS, f"the nightly names {name!r}, which is not a leg"


def test_the_nightly_leg_is_NOT_in_the_per_pr_set():
    """The whole point. If grpo were also wired into KERNELS the 44% crash rate
    would be back in front of every PR and the nightly would be redundant."""
    wired = {name for kernel in legs.KERNELS for name in kernel}
    assert "grpo" not in wired, (
        "grpo is in the per-PR set, so the nightly is pointless and every PR carries a 44% red"
    )


def test_a_leg_list_replaces_all_kernels_rather_than_filtering_after_it():
    """``--all-kernels`` derives BOTH the kernel plan and the payload count the
    launcher waits on. A filter applied afterwards leaves it expecting payloads
    that were never built, which times out rather than failing."""
    assert 'KERNEL_SELECT="--legs $LEG_LIST"' in TEXT
    assert 'KERNEL_SELECT="--all-kernels"' in TEXT
    assert "$KERNEL_SELECT \\" in TEXT
    assert (
        "--all-kernels \\" not in TEXT
    ), "--all-kernels is still hardcoded, so the override cannot take effect"


def test_the_schedule_bypasses_the_sampling_gate():
    """A nightly sampled at 15% is a weekly, and the difference is invisible
    until someone goes looking for a result that never existed."""
    assert "github.event_name == 'schedule' && 'true'" in TEXT, (
        "the schedule does not force the gate, so most nights it will draw a "
        "stand-down and report nothing"
    )


def test_the_leg_list_default_survives_a_schedule_event():
    """``inputs`` is null on a schedule, so an input default cannot supply the
    value; it has to come from the fallback."""
    assert "inputs.legs || (github.event_name == 'schedule'" in TEXT
