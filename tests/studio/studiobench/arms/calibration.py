# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NULL and SPIKE: the two arms that decide whether any other arm in the batch may be quoted.

An ablation harness has two failure modes that look exactly like results.

    IDENTICAL WHERE IT SHOULD DIFFER. Every arm reads the same. That is reported as "no mechanism
    dominates", when what actually happened is that the instrument cannot resolve anything at
    this magnitude. The SPIKE arm settles it: a known d milliseconds of work is injected, and if
    the instrument cannot see d, it could not have seen the mechanism either.

    DIFFERENT WHERE IT SHOULD NOT. Two identical configurations read 12% apart, and every arm in
    the batch inherits that 12% as apparent signal. The NULL arm settles it: the same build is
    run twice under two different arm ids, and whatever spread that produces is the noise floor
    for this batch on this machine.

BOTH ARE NON-DROPPABLE, IN EVERY BATCH. Not once at the start of the session, not on the tester's
machine last week. In every batch, because the thing they measure -- the machine's current
ability to resolve a difference -- changes with thermal state, with background load, and with
which browser build got installed this morning.

THE RULE: one arm must read the SAME and one arm must read DIFFERENT, or nothing in that batch is
quotable. Not "quotable with a caveat". A batch where the null control drifted and a batch where
the spike went unseen are both batches in which the numbers are unrelated to the app.

WHAT GETS PRINTED. The RECOVERY FRACTION (observed delta divided by the milliseconds actually
burned, measured inside the page rather than assumed from the requested d) and the DETECTION
FLOOR (the smallest injected cost this batch could actually see). A recovery fraction of 0.4 says
the instrument sees 40% of what is there, which means every ablation number in the batch is an
underestimate by a factor the reader can now apply. That is a far more useful thing to print than
a confidence interval on a number whose scale is unverified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..scoring.schema import Measure
from .manifest import Arm, ArmOutcome, ArmStatus, Invariance, PotencyCounter

#: The spike sizes every batch runs. 0.1 ms is below any plausible per-update mechanism and is
#: expected to be INVISIBLE, so the detection floor has something to sit on; 2.0 ms is comfortably
#: above and must be seen or the instrument is broken.
SPIKE_SIZES_MS: tuple[float, ...] = (0.1, 0.5, 2.0)

#: A spike is considered recovered when the observed delta is this close to the burned cost. Wide on
#: purpose: the point is "roughly the right amount", not calibration to three digits, and a tight
#: band would fail on a merely noisy machine.
RECOVERY_MIN = 0.5
RECOVERY_MAX = 2.0


SPIKE_INIT_JS = """
// SPDX-License-Identifier: AGPL-3.0-only
// Injected calibration spike. Burns a known amount of main-thread time per DOM update batch, so
// the harness can check that it can see a cost it put there itself.
//
// The burn is a BUSY WAIT, not a sleep and not a setTimeout: the mechanisms under test occupy
// the main thread, and a calibration that yields would be recovered by a different path through
// the scheduler than the thing it is calibrating. `performance.now()` is read in a tight loop
// and the ACTUAL elapsed time is accumulated, so the recovery fraction is computed against what
// was really burned rather than against what was requested. On a machine under load those two
// differ by more than the spike itself.
(() => {
  if (window.__sbSpike) { return; }
  const requested = %(spike_ms)f;
  const S = {
    requestedMs: requested,
    invocations: 0,
    burnedMs: 0,
    maxBurnMs: 0,
    installed: false,
    reason: "",
  };
  window.__sbSpike = S;
  const burn = () => {
    const started = performance.now();
    // A tight read loop. Anything cheaper (a counter, a sum) gets optimised into nothing by the
    // JIT and burns no time at all, which reads as an instrument that cannot see the spike.
    while (performance.now() - started < requested) { /* spin */ }
    const actual = performance.now() - started;
    S.invocations += 1;
    S.burnedMs += actual;
    if (actual > S.maxBurnMs) { S.maxBurnMs = actual; }
  };
  const attach = () => {
    const viewport = document.querySelector(".aui-thread-viewport.aui-stream-viewport");
    if (!viewport) { return false; }
    try {
      const observer = new MutationObserver(burn);
      observer.observe(viewport, { childList: true, subtree: true, characterData: true });
      S.installed = true;
      S.observer = observer;
      return true;
    } catch (e) {
      S.reason = "MutationObserver refused: " + e;
      return false;
    }
  };
  const poll = setInterval(() => {
    if (attach()) { clearInterval(poll); }
  }, 50);
  S.stop = () => {
    clearInterval(poll);
    if (S.observer) { S.observer.disconnect(); }
  };
})();
"""


def spike_init_script(spike_ms: float) -> str:
    """Init script that burns `spike_ms` of main-thread time per DOM update batch."""

    return SPIKE_INIT_JS % {"spike_ms": float(spike_ms)}


def null_arm(arm_id: str = "NULL", *, reference_id: str = "shipping") -> Arm:
    """A byte-identical rebuild under a different arm id. It must read EQUAL.

    Nothing about the build changes. The arm exists so that the harness measures the same thing
    twice under two labels and has to admit how far apart the two answers came out. Its potency
    counter is deliberately trivial (the run happened at all), because there is no treatment to
    detect: a NULL arm that "did not fire" is a NULL arm that did not run, which is caught by the
    session count rather than by a knob.
    """

    return Arm(
        arm_id = arm_id,
        title = f"NULL calibration (byte-identical rebuild of {reference_id})",
        mechanism = "none: the same build, the same config, a different arm id",
        invariance = Invariance.EXACT,
        potency = PotencyCounter(
            name = "windows_measured",
            min_delta = 1,
            direction = "increase",
            description = "the cell ran; there is no treatment to detect",
        ),
        implies_fix = (
            "nothing. If this arm reads different from its reference, the batch is measuring the "
            "machine and not the build"
        ),
        kind = "calibration",
        notes = (
            "non-droppable. A batch without a NULL arm has no measured noise floor, and every "
            "difference in it is being compared against a number nobody checked"
        ),
    )


def spike_arm(spike_ms: float) -> Arm:
    """A known d ms of main-thread work per update. It must read +d."""

    return Arm(
        arm_id = f"SPIKE{spike_ms:g}",
        title = f"SPIKE calibration ({spike_ms:g} ms burned per DOM update batch)",
        mechanism = f"an injected busy wait of {spike_ms:g} ms per update batch",
        invariance = Invariance.EXACT,
        potency = PotencyCounter(
            name = "spike_invocations",
            min_delta = 1,
            direction = "increase",
            description = "the spike observer fired at least once",
        ),
        implies_fix = (
            "nothing. This arm measures the instrument: if a known cost cannot be recovered, no "
            "unknown cost in this batch can be either"
        ),
        kind = "calibration",
        init_script = spike_init_script(spike_ms),
        notes = (
            "non-droppable. The 0.1 ms spike is expected to be invisible and is what the "
            "detection floor rests on; the 2.0 ms spike must be seen"
        ),
    )


CALIBRATION_ARM_IDS: tuple[str, ...] = ("NULL",) + tuple(f"SPIKE{d:g}" for d in SPIKE_SIZES_MS)


def calibration_arms() -> list[Arm]:
    return [null_arm()] + [spike_arm(d) for d in SPIKE_SIZES_MS]


class CalibrationMissing(AssertionError):
    """Raised when a batch plan tries to run without its calibration arms."""


def assert_batch_includes_calibration(planned_arm_ids: Iterable[str]) -> None:
    """Calibration arms are non-droppable. A plan that omits one is refused before it runs."""

    planned = set(planned_arm_ids)
    missing = [arm_id for arm_id in CALIBRATION_ARM_IDS if arm_id not in planned]
    if missing:
        raise CalibrationMissing(
            "this batch omits non-droppable calibration arms: "
            + ", ".join(missing)
            + ". One arm must read the same and one must read different, or nothing in the batch "
            "is quotable, and that cannot be established after the fact"
        )


@dataclass
class SpikeRecovery:
    """One spike arm's result: what was burned, what was seen, and the ratio."""

    spike_ms: float
    burned_ms_per_update: Measure
    observed_delta: Measure
    recovery_fraction: float | None
    recovered: bool
    note: str

    def to_json(self) -> dict[str, Any]:
        return {
            "spike_ms": float(self.spike_ms),
            "burned_ms_per_update": self.burned_ms_per_update.to_json(),
            "observed_delta": self.observed_delta.to_json(),
            "recovery_fraction": self.recovery_fraction,
            "recovered": bool(self.recovered),
            "note": self.note,
        }


@dataclass
class CalibrationVerdict:
    """Whether anything in this batch may be quoted, and the two numbers that decide it."""

    quotable: bool
    reason: str
    noise_floor_ms: Measure
    detection_floor_ms: Measure
    null_deltas: list[Measure] = field(default_factory = list)
    spikes: list[SpikeRecovery] = field(default_factory = list)

    def to_json(self) -> dict[str, Any]:
        return {
            "quotable": bool(self.quotable),
            "reason": self.reason,
            "noise_floor_ms": self.noise_floor_ms.to_json(),
            "detection_floor_ms": self.detection_floor_ms.to_json(),
            "null_deltas": [m.to_json() for m in self.null_deltas],
            "spikes": [s.to_json() for s in self.spikes],
        }

    def render(self) -> str:
        lines = ["CALIBRATION (non-droppable, every batch)"]
        lines.append(f"  noise floor       {self.noise_floor_ms.display()}   (NULL arms)")
        lines.append(f"  detection floor   {self.detection_floor_ms.display()}   (SPIKE arms)")
        for null_delta in self.null_deltas:
            lines.append(f"  NULL delta        {null_delta.display()}")
        for spike in self.spikes:
            fraction = (
                f"{spike.recovery_fraction:.2f}" if spike.recovery_fraction is not None else "n/a"
            )
            lines.append(
                f"  SPIKE {spike.spike_ms:>4g} ms    burned {spike.burned_ms_per_update.display()}"
                f", saw {spike.observed_delta.display()}, recovery {fraction}"
                f"  [{'recovered' if spike.recovered else 'not recovered'}]"
            )
        lines.append("")
        lines.append(
            f"  VERDICT: {'quotable' if self.quotable else 'NOT QUOTABLE'} -- {self.reason}"
        )
        return "\n".join(lines)


def evaluate_batch(
    *, null_deltas: Sequence[Measure], spike_observations: Sequence[Mapping[str, Any]]
) -> CalibrationVerdict:
    """Decide whether this batch may be quoted at all.

    `null_deltas` are the differences between each NULL arm and its reference, in ms per update.
    `spike_observations` is one mapping per spike arm with keys `spike_ms`, `burned_ms_per_update`
    (a Measure, read from inside the page) and `observed_delta` (a Measure).

    The noise floor is the largest absolute NULL delta. The detection floor is the smallest spike
    that was RECOVERED, where recovered means the observed delta both cleared the noise floor and
    landed within a factor of two of what was actually burned. A batch where no spike was
    recovered has no detection floor and is not quotable: the instrument is blind at every
    magnitude tested, and an arm reading "no difference" in that batch is not evidence.
    """

    usable_nulls = [m for m in null_deltas if m.has_reading]
    if not usable_nulls:
        return CalibrationVerdict(
            quotable = False,
            reason = (
                "no NULL arm produced a reading, so this batch has no measured noise floor and "
                "every difference in it is being compared against nothing"
            ),
            noise_floor_ms = Measure.failed("ms/update", "no NULL arm produced a reading"),
            detection_floor_ms = Measure.failed("ms/update", "not evaluated without a noise floor"),
            null_deltas = list(null_deltas),
        )

    noise_floor_value = max(abs(float(m.value)) for m in usable_nulls)
    noise_floor = Measure.read(noise_floor_value, "ms/update")

    recoveries: list[SpikeRecovery] = []
    for observation in spike_observations:
        spike_ms = float(observation["spike_ms"])
        burned: Measure = observation["burned_ms_per_update"]
        observed: Measure = observation["observed_delta"]
        if not (burned.has_reading and observed.has_reading) or float(burned.value) <= 0:
            recoveries.append(
                SpikeRecovery(
                    spike_ms = spike_ms,
                    burned_ms_per_update = burned,
                    observed_delta = observed,
                    recovery_fraction = None,
                    recovered = False,
                    note = (
                        "no usable reading. A spike that did not burn is a spike that did not "
                        "run, which says nothing about the instrument"
                    ),
                )
            )
            continue
        fraction = float(observed.value) / float(burned.value)
        clears_noise = abs(float(observed.value)) > noise_floor_value
        in_band = RECOVERY_MIN <= fraction <= RECOVERY_MAX
        recovered = clears_noise and in_band
        if recovered:
            note = "recovered: the instrument saw what was burned, within a factor of two"
        elif not clears_noise:
            note = (
                "below the noise floor of this batch. Expected for the smallest spike; that is "
                "what the detection floor is made of"
            )
        else:
            note = (
                f"seen, but the recovery fraction is {fraction:.2f}, outside "
                f"[{RECOVERY_MIN}, {RECOVERY_MAX}]. The instrument is reading a different "
                "magnitude than was burned, so its scale is not trustworthy here"
            )
        recoveries.append(
            SpikeRecovery(
                spike_ms = spike_ms,
                burned_ms_per_update = burned,
                observed_delta = observed,
                recovery_fraction = fraction,
                recovered = recovered,
                note = note,
            )
        )

    recovered = [r for r in recoveries if r.recovered]
    if not recovered:
        return CalibrationVerdict(
            quotable = False,
            reason = (
                "no SPIKE arm was recovered. The instrument could not see a cost it injected "
                "itself at any magnitude tested, so an arm reading 'no difference' in this batch "
                "is not evidence about the app"
            ),
            noise_floor_ms = noise_floor,
            detection_floor_ms = Measure.failed(
                "ms/update",
                f"no spike up to {max(SPIKE_SIZES_MS):g} ms was recovered",
            ),
            null_deltas = list(null_deltas),
            spikes = recoveries,
        )

    detection_floor_value = min(r.spike_ms for r in recovered)
    detection_floor = Measure.read(detection_floor_value, "ms/update")

    # The other half of the rule: one arm must read the SAME. A NULL arm whose delta exceeds the
    # smallest thing the batch can detect is not reading the same.
    worst_null = max(abs(float(m.value)) for m in usable_nulls)
    if worst_null >= detection_floor_value:
        return CalibrationVerdict(
            quotable = False,
            reason = (
                f"the NULL arm moved by {worst_null:.3f} ms/update, which is at or above the "
                f"detection floor of {detection_floor_value:g} ms/update. Two identical builds "
                "read as different, so a difference between two different builds means nothing "
                "in this batch"
            ),
            noise_floor_ms = noise_floor,
            detection_floor_ms = detection_floor,
            null_deltas = list(null_deltas),
            spikes = recoveries,
        )

    return CalibrationVerdict(
        quotable = True,
        reason = (
            f"the NULL arm read equal to within {worst_null:.3f} ms/update and the "
            f"{detection_floor_value:g} ms spike was recovered. One arm read the same and one "
            "read different, so differences in this batch are differences"
        ),
        noise_floor_ms = noise_floor,
        detection_floor_ms = detection_floor,
        null_deltas = list(null_deltas),
        spikes = recoveries,
    )


def null_delta_from_outcomes(reference: ArmOutcome, null_outcome: ArmOutcome) -> Measure:
    """The NULL arm's difference from its reference, or an explicit non-reading."""

    if null_outcome.status is ArmStatus.VOIDED:
        return Measure.failed("ms/update", f"NULL arm voided: {null_outcome.reason}")
    if not (reference.cost.has_reading and null_outcome.cost.has_reading):
        return Measure.failed(
            "ms/update",
            "the NULL arm or its reference produced no reading",
        )
    return Measure.read(
        float(null_outcome.cost.value) - float(reference.cost.value),
        reference.cost.unit,
    )
