# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Is this run's step count one that can train anything? Answered before paying.

``max_steps`` is a free-text ``workflow_dispatch`` input, and both ways of
getting it wrong cost a Kaggle session and report the pull request RED for
something that is not a code defect:

* **Not an integer.** ``foo`` travels all the way into the payload's argv, where
  argparse rejects it and the process exits 2 with no report written. The
  generated cell turns that into a failing report on purpose (a crashed payload
  must not read as missing evidence), so three legs come back as assertion
  failures -- after the kernels were pushed, the model downloaded and the quota
  spent.
* **Too small to apply an optimizer update.** Under fp16 the dynamic gradient
  scaler starts at 65536, halves on each overflow and SKIPS the step it
  overflowed on. The workflow passes no ``--init-loss-scale``, so the first
  steps of every run are skipped ones, and a run shorter than that prefix
  applies zero updates: ``optimisation_failures`` says so and the leg goes red
  having measured nothing.

Both are stand-downs, the same answer this workflow gives an unresolvable ref: a
warning and a green job, because nothing was learned about the code under test.

THE FLOOR IS MEASURED, NOT DECLARED. It is the shortest prefix of the COMMITTED
reference trace that the payload's own ``optimisation_failures`` accepts, so it
follows the reference and that function rather than a number written here that
would go stale the moment either moved. The prefix is the right comparison
because the payload trains on a constant schedule with no warmup
(``lr_scheduler_type="constant"``, ``warmup_steps=0``) and logs every step, so
an n-step run IS the first n steps of the committed 10-step one.

On the committed trace that comes out at 5: steps 1-3 have grad_norm NaN (all
skipped, no update applied) and step 4's loss is above step 1's, so 4 fails the
"loss did not decrease" check too, which is why the file docstring's "about
five" is a measurement rather than a guess.

Exits 0 whatever it decides. The stand-down travels as the ``stand_down``
output, like gate.py's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from legs import LEGS  # noqa: E402


def _log(msg: str) -> None:
    print(f"[steps] {msg}", flush = True)


def _out(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding = "utf-8") as fh:
            fh.write(f"{key}={value}\n")
    _log(f"{key}={value}")


def parse_steps(raw: str) -> int | None:
    """The dispatched value as a positive integer, or None if it is not one.

    ``int()`` alone is not the test: it accepts ``+7`` and surrounding
    whitespace, which are fine, and returns 0 or a negative for values the
    trainer cannot run at all.
    """
    try:
        steps = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return steps if steps > 0 else None


def reference_steps(payload_dir: Path, leg: str = "control") -> int | None:
    """The step count the committed reference declares, read the payload's way.

    The build step drops the band when this run's count is not the reference's,
    and ``check_reference`` refuses the same pairing from the other side, so the
    two have to mean the same thing by "the reference's count". This calls the
    payload's own ``reference_step_count`` rather than reading the JSON a second
    time, and finds the file through the leg registry rather than naming it.

    None when it cannot be established, which the build step already treats as
    not comparable: the payload's answer for a reference that does not say is
    ``reference_step_count_unknown``, a hard failure, so a band left on here
    would be red for the reference rather than for the code.
    """
    try:
        name = LEGS[leg].reference
        if not name:
            return None
        sys.path.insert(0, str(payload_dir))
        from run_t4_smoke import reference_step_count

        data = json.loads((payload_dir / "references" / name).read_text(encoding = "utf-8"))
        return reference_step_count(data)
    except Exception:  # noqa: BLE001
        return None


def reference_metrics(payload_dir: Path, leg: str = "control") -> list[dict]:
    """The committed trace the floor is measured from.

    Read through the leg registry rather than by naming the file again: the leg
    is what decides which reference this CI runs against.
    """
    name = LEGS[leg].reference
    if not name:
        return []
    path = payload_dir / "references" / name
    data = json.loads(path.read_text(encoding = "utf-8"))
    metrics = data.get("metrics")
    return metrics if isinstance(metrics, list) else []


def minimum_steps(metrics: list[dict], failures) -> int | None:
    """Shortest prefix of ``metrics`` the payload's own verdict accepts.

    None when no prefix does, including the empty trace: an unmeasurable floor
    is not a floor of zero, and answering "any step count will do" is how a
    check that cannot fail gets written.
    """
    for n in range(1, len(metrics) + 1):
        if not failures(metrics[:n]):
            return n
    return None


def decide(raw: str, payload_dir: Path) -> tuple[bool, str]:
    """(stand_down, reason) for the dispatched value."""
    steps = parse_steps(raw)
    if steps is None:
        return True, (
            f"max_steps was dispatched as {str(raw)[:80]!r}, which is not a positive "
            f"integer. Every leg forwards it to the payload, where argparse rejects it "
            f"and the process dies before it can judge itself, so the run would have "
            f"spent a Kaggle session to report the pull request red for a typo."
        )

    sys.path.insert(0, str(payload_dir))
    try:
        from run_t4_smoke import optimisation_failures
    except Exception as exc:  # noqa: BLE001
        return True, (
            f"the payload's optimisation_failures could not be read "
            f"({type(exc).__name__}), so the step count that trains anything is "
            f"unknown and this run cannot be judged before it is paid for."
        )

    try:
        metrics = reference_metrics(payload_dir)
    except Exception as exc:  # noqa: BLE001
        return True, (
            f"the committed reference could not be read ({type(exc).__name__}), so the "
            f"step count below which the fp16 scaler leaves a run with no applied "
            f"optimizer update is unknown."
        )

    floor = minimum_steps(metrics, optimisation_failures)
    if floor is None:
        return True, (
            "no prefix of the committed reference satisfies the payload's own "
            "optimisation checks, so the shortest run that trains anything cannot be "
            "measured. Recapture the reference before dispatching a step count."
        )
    if steps < floor:
        return True, (
            f"max_steps={steps} is below {floor}, the shortest run the committed "
            f"reference shows applying an optimizer update: under fp16 the gradient "
            f"scaler skips the leading steps, and this workflow passes no "
            f"--init-loss-scale, so a shorter run trains nothing and reports red for "
            f"the step count rather than for the code. Dispatch at least {floor}, or "
            f"wire --init-loss-scale and recapture the reference with it."
        )
    return False, f"max_steps={steps} is at or above the measured floor of {floor}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", required = True, help = "the dispatched value, unvalidated")
    ap.add_argument("--payload-dir", required = True)
    args = ap.parse_args()

    stand_down, reason = decide(args.max_steps, Path(args.payload_dir))
    if stand_down:
        print(
            f"::warning title=Stood down on the dispatched step count::{reason}",
            flush = True,
        )
    else:
        _log(reason)
    _out("stand_down", "true" if stand_down else "false")
    _out("reason", reason)

    # The PARSED value, for everything downstream, so that the one place which
    # normalises the dispatched string is the one place that validated it.
    # parse_steps deliberately accepts "+10", "010" and surrounding whitespace
    # as the ten they are, and the payload's argparse agrees, so the build step
    # comparing the raw string against the reference's count used to read those
    # as a different run and drop the reference band from it: a green run with
    # the committed band never applied, announced as a step count that was not
    # actually different. The reference's own count comes from here too, for
    # the same reason -- one definition, the payload's.
    steps = parse_steps(args.max_steps)
    if steps is not None:
        _out("steps", str(steps))
    ref_steps = reference_steps(Path(args.payload_dir))
    _out("reference_steps", "" if ref_steps is None else str(ref_steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
