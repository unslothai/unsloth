# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Is this path linear, asked without the runner answering for it.

A guard against a quadratic blow-up -- a regex that backtracks, a sweep that rescans its
own tail -- reads naturally as "it must finish in under X seconds". On a shared runner
that is a budget, not a property: it says as much about who else is on the box as about
the code, and a value low enough to catch a real regression is low enough to fail on a
quiet branch. `test_pr5624_regressions.py` carried one and cost two unrelated PRs a red
check before #10868 replaced it with the paired-ratio form below.

The property those tests are actually named for is growth: `factor` times the input must
cost about `factor` times the time, not `factor ** 2` times. That is a ratio, and a ratio
of two measurements taken back to back divides the machine out. Lives in tests/_shared so both test trees reach it (see
tests/_shared/real_accelerator.py for the same arrangement) and the next guard of this
shape reuses the statistics instead of inventing another budget.
"""

from __future__ import annotations


def growth(
    run,
    build,
    units: int,
    factor: int = 4,
    repeats: int = 3,
):
    """How much more `factor` times the input costs. Returns (ratio, big_seconds, big_result).

    `build(n)` makes an input of size n and `run(text)` is the thing being measured.

    The MEDIAN of `repeats` PAIRED ratios: each pair times the small input and then the big
    one back to back, and the ratio is formed inside the pair before anything is aggregated.

    This replaces an absolute ``elapsed < 1.0`` budget at one size. That budget read 0.20s on
    a quiet runner and 1.41s on a busy one, so it failed for the wrong reason on unrelated
    PRs, and it also sat close enough to the line that a REAL regression (#10507, which made
    the R1 path quadratic again) only tipped it over some of the time. A ratio answers the
    question these tests are named for: linear is ~`factor`, quadratic is ~`factor ** 2`.

    It also replaces ``min(big over repeats) / min(small over repeats)``, which looked like
    it cancelled the machine out and did not. Those two minima come from batches run at
    different times, so a quiet window during the small batch and a busy one during the big
    batch multiply instead of cancelling, and the quotient of two separately-taken minima is
    not a minimum of anything. Measured on a 2-vCPU box under load, 15 trials per shape:

        strategy      R1        R1-distant  GLM       V3        >= 6.0
        min/min       max 8.69  max 8.23    max 8.27  max 8.59  7 of 60
        paired median max 4.26  max 4.83    max 4.83  max 5.42  0 of 60

    That 12% false-fail rate is not hypothetical: it is why this test failed on #10825 and
    again on #10864 at 6.56, both times on branches that touch none of this code.

    Pairing is what fixes it. Contention hits both halves of a pair roughly equally and
    divides out, which is what the old comment claimed for the unpaired form. The median
    then discards a pair that got unlucky, in either direction, rather than trusting one
    reading.

    Detection power is kept, which is the half worth checking before loosening anything. On
    a synthetic path with a true 16x profile this reports 8 of 8 over the bar, same as the
    old form. On the marginal 6.7x shape (#10832's partially fixed sweep) it reports 9 of 12
    against the old form's 10 of 12, a difference well inside the noise at that sample size,
    and that shape's sibling test measures 12.2x when broken, so the suite still catches it.
    """
    import statistics as _statistics
    import time as _time

    def once(text):
        start = _time.perf_counter()
        result = run(text)
        return _time.perf_counter() - start, result

    small_text, big_text = build(units), build(units * factor)
    ratios, big, result = [], None, None
    for _ in range(repeats):
        small_elapsed, _ = once(small_text)
        big_elapsed, result = once(big_text)
        # The backstop below reads the BEST big time, not the worst. Contention only adds, so
        # the minimum is the closest this size got to its own cost, and the backstop should
        # fire on a path that is genuinely too slow rather than on a runner that stalled once.
        big = big_elapsed if big is None else min(big, big_elapsed)
        # A timer's own resolution must not read as superlinear growth on a very fast machine.
        ratios.append(big_elapsed / max(small_elapsed, 1e-4))

    return _statistics.median(ratios), big, result


def assert_linear(
    run,
    build,
    label: str,
    units: int,
    *,
    factor: int = 4,
    tolerance: float = 6.0,
):
    """`run(build(n))` must cost ~`factor`x, not ~`factor ** 2`x, for `factor`x the input."""
    ratio, big, result = growth(run, build, units, factor)
    # Backstop: a regression bad enough to make the ratio unmeasurable still has to fail, and
    # fail quickly, rather than run until the job's own timeout kills it with no explanation.
    assert big < 60.0, f"{label} path took {big:.1f}s on {units * factor} units"
    assert ratio < tolerance, (
        f"{label} path is not linear: {factor}x the input cost {ratio:.1f}x the time "
        f"(linear is ~{factor}, quadratic is ~{factor ** 2})"
    )
    return result
