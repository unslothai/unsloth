# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The recovery test: does deleting the thread give the performance back?

Ninety seconds, never run before, and it separates three fix classes that every other measurement
in this tool conflates.

    seed 512 turns  ->  measure  ->  delete back to zero  ->  measure again

The user report is "it gets worse the longer you use it". That sentence has two completely
different causes and they need different fixes:

  OCCUPANCY   the cost is proportional to what is currently on the page. Delete the turns and the
              cost goes away. The fix is to stop keeping them present: virtualise, contain, or
              unmount. This is the comfortable case and it is what everyone assumes.
  RETAINED    the cost does not come back down. Something survives the delete: a detached DOM
              tree still referenced by a closure, an observer never disconnected, a growing map
              in a store, a Shiki cache keyed per block. The fix is to release it, and no amount
              of virtualisation will help, because the structure is not on screen in the first
              place.
  HYSTERETIC  partial recovery. Both mechanisms are present, and fixing only the visible one
              leaves a slow drift that reappears in a support ticket six weeks later.

The test is worth its ninety seconds because those three imply different work, and nothing else
in this tool can tell them apart: every other measurement is taken at a fixed thread size, where
occupancy and retention are perfectly correlated.

A FOURTH OUTCOME MATTERS AND IS EASY TO MISREAD. If the loaded measurement is not meaningfully
worse than the baseline, there is nothing to recover, and the recovery fraction is undefined
rather than 100%. Printing "fully recovered" for a load that never cost anything would be the
same defect as printing zero for an instrument that never ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..scoring.schema import Measure

#: Seeded turns for the loaded phase. 512 is chosen to be well past the point where the reported
#: symptom appears while still seeding in under a minute through the messages API.
RECOVERY_TURNS = 512

FULL_RECOVERY = 0.90
NO_RECOVERY = 0.10


@dataclass
class RecoveryResult:
    """Baseline, loaded, and post-delete, plus which of the three classes this is."""

    baseline: Measure
    loaded: Measure
    after_delete: Measure
    turns: int
    noise_floor_ms: float
    recovered_fraction: float | None
    classification: str
    implies_fix: str
    note: str

    def to_json(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_json(),
            "loaded": self.loaded.to_json(),
            "after_delete": self.after_delete.to_json(),
            "turns": int(self.turns),
            "noise_floor_ms": float(self.noise_floor_ms),
            "recovered_fraction": self.recovered_fraction,
            "classification": self.classification,
            "implies_fix": self.implies_fix,
            "note": self.note,
        }

    def render(self) -> str:
        fraction = (
            f"{self.recovered_fraction:.0%}" if self.recovered_fraction is not None else "undefined"
        )
        return "\n".join(
            [
                f"RECOVERY TEST ({self.turns} turns seeded, then deleted back to zero)",
                f"  baseline       {self.baseline.display()}",
                f"  loaded         {self.loaded.display()}",
                f"  after delete   {self.after_delete.display()}",
                f"  recovered      {fraction}",
                f"  CLASSIFICATION: {self.classification}",
                f"  {self.note}",
                f"  implies: {self.implies_fix}",
            ]
        )


def classify_recovery(
    *,
    baseline: Measure,
    loaded: Measure,
    after_delete: Measure,
    noise_floor_ms: float,
    turns: int = RECOVERY_TURNS,
) -> RecoveryResult:
    """Turn three readings into one of four classes, or an explicit refusal to classify."""

    def refuse(reason: str) -> RecoveryResult:
        return RecoveryResult(
            baseline = baseline,
            loaded = loaded,
            after_delete = after_delete,
            turns = turns,
            noise_floor_ms = float(noise_floor_ms),
            recovered_fraction = None,
            classification = "NOT CLASSIFIED",
            implies_fix = "none: this run cannot distinguish the cases",
            note = reason,
        )

    if not (baseline.has_reading and loaded.has_reading and after_delete.has_reading):
        return refuse(
            "one of the three phases produced no reading, so there is no recovery to compute"
        )

    base_v = float(baseline.value)
    loaded_v = float(loaded.value)
    after_v = float(after_delete.value)
    load_cost = loaded_v - base_v

    if load_cost <= noise_floor_ms:
        return RecoveryResult(
            baseline = baseline,
            loaded = loaded,
            after_delete = after_delete,
            turns = turns,
            noise_floor_ms = float(noise_floor_ms),
            recovered_fraction = None,
            classification = "NOTHING TO RECOVER",
            implies_fix = (
                "none from this test. Seeding the thread did not make it measurably slower on "
                "this machine, so the recovery question does not arise here"
            ),
            note = (
                f"the loaded phase is only {load_cost:.3f} ms above baseline, at or below the "
                f"noise floor of {noise_floor_ms:.3f} ms. The recovery fraction is UNDEFINED, "
                "not 100%: there was no cost to give back"
            ),
        )

    fraction = (loaded_v - after_v) / load_cost

    if after_v > loaded_v + noise_floor_ms:
        return RecoveryResult(
            baseline = baseline,
            loaded = loaded,
            after_delete = after_delete,
            turns = turns,
            noise_floor_ms = float(noise_floor_ms),
            recovered_fraction = fraction,
            classification = "WORSE AFTER DELETE",
            implies_fix = (
                "look at the delete path itself. Something about removing the turns costs more "
                "than keeping them, which usually means detached subtrees still observed, or "
                "state rebuilt on every removal"
            ),
            note = (
                "the post-delete reading is worse than the loaded one. This is not recovery "
                "failing, it is the deletion adding cost of its own"
            ),
        )

    if fraction >= FULL_RECOVERY:
        classification = "OCCUPANCY"
        implies = (
            "the cost is proportional to what is currently present. Virtualise, contain or "
            "unmount the retained messages and the cost goes with them"
        )
        note = (
            f"{fraction:.0%} of the loaded cost came back on delete, so nothing meaningful "
            "survives the removal"
        )
    elif fraction <= NO_RECOVERY:
        classification = "RETAINED STRUCTURE"
        implies = (
            "something survives the delete: a detached tree still referenced, an observer never "
            "disconnected, a cache keyed per block, a store that only grows. Virtualisation will "
            "not touch this, because the structure is not on screen to begin with"
        )
        note = (
            f"only {fraction:.0%} of the loaded cost came back. This is the literal shape of "
            "'gets worse the longer you use it and stays worse'"
        )
    else:
        classification = "HYSTERETIC"
        implies = (
            "both mechanisms are present. Fixing the visible one leaves a residual drift that "
            "will come back as a separate report later"
        )
        note = (
            f"{fraction:.0%} of the loaded cost came back, which is neither full recovery nor "
            "none. Do not round it to whichever is more convenient"
        )

    return RecoveryResult(
        baseline = baseline,
        loaded = loaded,
        after_delete = after_delete,
        turns = turns,
        noise_floor_ms = float(noise_floor_ms),
        recovered_fraction = fraction,
        classification = classification,
        implies_fix = implies,
        note = note,
    )
