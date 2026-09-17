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
    abort_over_s: float = None,
    clock = None,
):
    """How much more `factor` times the input costs.

    Returns (ratio, best_big_seconds, big_result, pairs_completed). `pairs_completed` is
    less than `repeats` only when `abort_over_s` cut the loop short, and a caller that
    treats the ratio as a verdict has to look at it: one aborted pair whose small leg was
    also slow can report a ratio under the bar from a sample that never finished.

    `build(n)` makes an input of size n and `run(text)` is the thing being measured.

    `clock` is `time.perf_counter` unless given. It exists so this module's OWN tests can
    state a timing shape exactly instead of sleeping for it: a test of a flakiness guard
    that needs the scheduler to cooperate is the thing being fixed here, not a way to
    check it. Nothing in the repo passes it outside those tests.

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

    read_clock = _time.perf_counter if clock is None else clock

    def once(text):
        start = read_clock()
        result = run(text)
        return read_clock() - start, result

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
        # Checked here, not after the loop: on the regression these guards exist for, the
        # big leg is the minutes-long one, so finishing all `repeats` of it to report a
        # number the caller will reject anyway is the slow way to reach the same verdict.
        if abort_over_s is not None and big_elapsed > abort_over_s:
            break

    return _statistics.median(ratios), big, result, len(ratios)


def assert_linear(
    run,
    build,
    label: str,
    units: int,
    *,
    factor: int = 4,
    tolerance: float = 6.0,
    repeats_on_retry: int = 7,
    clock = None,
):
    """`run(build(n))` must cost ~`factor`x, not ~`factor ** 2`x, for `factor`x the input.

    `units` is the SMALL size. The largest input actually run is `units * factor`, so when
    this replaces an absolute budget, pass the old size DIVIDED by `factor` -- otherwise the
    big leg is `factor`x bigger than anything that was ever measured, and on the regression
    being guarded against that leg is `factor ** 2`x slower again. A guard whose broken case
    takes a minute at the old size would then take a quarter of an hour, and the job's own
    timeout kills it before the ratio below can say why.
    """
    budget = 60.0
    # Passed down rather than checked here: on a path slow enough to trip it, every repeat
    # is another minute spent measuring something already known to be too slow.
    ratio, big, result, _ = growth(run, build, units, factor, abort_over_s = budget, clock = clock)
    # Backstop: a regression bad enough to make the ratio unmeasurable still has to fail, and
    # fail quickly, rather than run until the job's own timeout kills it with no explanation.
    assert big < budget, f"{label} path took {big:.1f}s on {units * factor} units"
    if ratio >= tolerance:
        # RE-MEASURE rather than loosen. Pairing divides most contention out, but not all of
        # it: the big leg runs `factor`x longer than the small one, so a scheduler stall that
        # lands inside a run is `factor`x more likely to land in the big half, which biases a
        # pair upward and never downward. Three pairs is few enough that two unlucky ones move
        # the median, which is how this reported 7.2x on unslothai/unsloth#11152, a branch that
        # touches none of this code.
        #
        # A second, larger sample is the honest answer, and it is not a second chance: a path
        # that is genuinely quadratic measures ~`factor ** 2` in EVERY pair, so its second
        # median comes back over the bar as surely as its first, while a contention spike does
        # not survive being asked again on a bigger sample. The cost is paid only on the
        # reading that would otherwise have failed, so a green run still takes three pairs.
        confirm, big_again, result, pairs = growth(
            run,
            build,
            units,
            factor,
            repeats = repeats_on_retry,
            abort_over_s = budget,
            clock = clock,
        )
        # The confirmation stands on its OWN reading, not on min(big, big_again). Taking the
        # better of the two samples hid the case that matters: the retry aborts after one pair
        # because its big leg blew the budget, and that same pair's small leg was slow enough
        # to put the ratio under the bar, so a superlinear path passed on a sample that never
        # finished. Both halves are now required of the confirmation itself.
        assert (
            big_again < budget
        ), f"{label} path took {big_again:.1f}s on {units * factor} units while re-measuring"
        assert pairs == repeats_on_retry, (
            f"{label} path: the confirmation stopped after {pairs} of {repeats_on_retry} pairs, "
            "so its ratio is not a reading of anything"
        )
        assert confirm < tolerance, (
            f"{label} path is not linear: {factor}x the input cost {confirm:.1f}x the time "
            f"over {repeats_on_retry} pairs, after {ratio:.1f}x over 3 "
            f"(linear is ~{factor}, quadratic is ~{factor ** 2})"
        )
    return result
