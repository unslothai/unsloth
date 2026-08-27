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


def _nightly_legs():
    """The leg names the schedule actually selects, read off the workflow."""
    # Anchored on LEG_LIST, not on the schedule fallback in general. The gate
    # bypass a few lines above reads `github.event_name == 'schedule' && 'true'`
    # and a loose pattern picks THAT up: the first version of this helper
    # reported the nightly leg set as ["true"].
    match = re.search(r"LEG_LIST:.*github\.event_name == 'schedule' && '([a-z_,]+)'", TEXT)
    assert match, "the schedule selects no leg list at all"
    return [name for name in match.group(1).split(",") if name]


def test_the_schedule_runs_the_grpo_leg():
    assert "grpo" in _nightly_legs(), (
        "the schedule must select the leg; a nightly that runs the wired set "
        "is just another copy of the per-PR run"
    )


def test_the_schedule_also_runs_the_multi_gpu_leg():
    """multi_gpu is here for a different reason from grpo and the distinction
    is worth keeping: grpo is nightly because it CRASHES 44% of the time,
    multi_gpu because it costs makespan. It passes on hardware -- the ab3 A/B
    has it green in both arms -- and the brief it was built to was multi-GPU
    coverage at no wall-clock cost, which it measurably is not (+172.4s and
    +39.7s over two same-commit pairs, slower in both).

    Nightly is where that coverage is free. Without this the DEVICE_COUNT > 1
    bindings in unsloth's kernels go back to being tested nowhere at all, which
    is the state this leg was written to end."""
    assert "multi_gpu" in _nightly_legs(), (
        "the nightly no longer runs multi_gpu, so unsloth's DEVICE_COUNT > 1 "
        "code path is covered by nothing: every other leg is pinned to one card"
    )


def test_the_schedule_also_runs_the_latest_compile_leg():
    """Third reason again, and the distinction is the point: grpo is nightly
    because it crashes 44% of sessions, multi_gpu because it costs makespan at
    the margin, latest_compile because it does not FIT.

    Its DONE record is 1323.0s (unsloth-probe-lcleg-tmpdir-ac53ca) and at
    12.73GB peak it admits no co-tenant, so it wants a whole card for 22
    minutes. The per-PR kernel's only slack is gpu1's 776.3s idle block while
    Studio holds gpu0. 1323 does not go into 776.

    Without this the leg is wired nowhere and gemma-4-E2B-it on the newest
    transformers and trl -- the only thing that caught zoo #1103 -- is tested
    by nothing. That is the state it was built to end, and the reason the leg
    is nightly rather than deleted."""
    assert "latest_compile" in _nightly_legs(), (
        "the nightly no longer runs latest_compile, so nothing anywhere loads "
        "gemma-4-E2B-it on the newest transformers and trl, which is the "
        "pairing that found unsloth-zoo #1103"
    )


def test_every_leg_the_nightly_names_exists():
    """A typo here produces a build that selects nothing and a run that proves
    nothing, with no error anywhere."""
    named = _nightly_legs()
    assert named, "no scheduled leg name found at all"
    for name in named:
        assert name in legs.LEGS, f"the nightly names {name!r}, which is not a leg"


def test_the_nightly_set_fits_in_one_kernel():
    """`--legs` builds ONE kernel, and MAX_LEGS_PER_KERNEL is what the driver's
    scheduling was measured against. A list longer than that silently packs a
    kernel nobody has run."""
    assert len(_nightly_legs()) <= legs.MAX_LEGS_PER_KERNEL


def test_no_nightly_leg_is_ALSO_in_the_per_pr_set():
    """The whole point, and it applies to each of them. If grpo were wired into
    KERNELS the 44% crash rate would be back in front of every PR; if multi_gpu
    were, the makespan it was moved here to avoid would be back too. Either way
    the nightly becomes a second copy of the per-PR run."""
    wired = {name for kernel in legs.KERNELS for name in kernel}
    both = sorted(set(_nightly_legs()) & wired)
    assert not both, (
        f"{both} run nightly AND per-PR, so the nightly is pointless and every "
        f"PR carries whatever these were moved off the critical path to avoid"
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
