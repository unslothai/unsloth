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
    budget_s: float = None,
    clock = None,
):
    """How much more `factor` times the input costs.

    Returns (ratio, best_big_seconds, big_result, pairs_completed). `pairs_completed` is
    less than `repeats` only when `abort_over_s` or `budget_s` cut the loop short, and a
    caller that treats the ratio as a verdict has to look at it: one aborted pair whose
    small leg was also slow can report a ratio under the bar from a sample that never
    finished.

    `abort_over_s` bounds ONE BIG LEG; `budget_s` bounds THE WHOLE CALL. They are not the
    same guard and neither implies the other: `repeats` legs each just under `abort_over_s`
    is `repeats` times the cost that one leg was allowed, which is how a sample sized from a
    previous reading still overran the runner's per-test timeout. `budget_s` is measured
    rather than predicted, and it is asked BEFORE each pair rather than after, reserving
    room at the cost of the worst pair seen so far -- a bound tested only once a pair is
    already home is not a bound on that pair.

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
    started, pair_costs = read_clock(), []
    for _ in range(repeats):
        # Reserve room for another pair at the worst observed cost, before running it.
        if budget_s is not None and pair_costs:
            if read_clock() - started + max(pair_costs) > budget_s:
                break
        pair_started = read_clock()
        small_elapsed, _ = once(small_text)
        big_elapsed, result = once(big_text)
        pair_costs.append(read_clock() - pair_started)
        # Backstop uses the best big time: contention only adds.
        big = big_elapsed if big is None else min(big, big_elapsed)
        ratios.append(big_elapsed / max(small_elapsed, 1e-4))
        # Abort early: on a regression the big leg takes minutes per repeat.
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
    total_budget_s: float = 240.0,
    clock = None,
):
    """`run(build(n))` must cost ~`factor`x, not ~`factor ** 2`x, for `factor`x the input.

    `units` is the SMALL size. The largest input actually run is `units * factor`, so when
    this replaces an absolute budget, pass the old size DIVIDED by `factor` -- otherwise the
    big leg is `factor`x bigger than anything that was ever measured, and on the regression
    being guarded against that leg is `factor ** 2`x slower again. A guard whose broken case
    takes a minute at the old size would then take a quarter of an hour, and the job's own
    timeout kills it before the ratio below can say why.

    `total_budget_s` caps what this call may SPEND IN TOTAL, for the same reason the paragraph
    above exists: a red run has to arrive as this function's message and not as the runner's
    timeout, or nobody learns which shape was measured. Everything is inside it, first sample
    included -- the first sample has no bound of its own beyond the 60s per big leg, so a
    scheduler stall in one of its small legs can spend most of the runner's patience before
    the re-measurement is even considered, and a retry budget counted fresh from that point
    overruns whatever was left. 240s against the `--timeout=330` those CI invocations pass,
    leaving margin for collection and the rest of the test body.
    """
    import time as _time

    budget = 60.0
    read_clock = _time.perf_counter if clock is None else clock
    started = read_clock()
    ratio, big, result, first_pairs = growth(
        run,
        build,
        units,
        factor,
        abort_over_s = budget,
        budget_s = total_budget_s,
        clock = clock,
    )
    # Backstop: an unmeasurable regression must still fail quickly.
    assert big < budget, f"{label} path took {big:.1f}s on {units * factor} units"
    # The budget covers this sample too; an early stop is not a verdict.
    assert first_pairs == 3, (
        f"{label} path: the first sample stopped after {first_pairs} of 3 pairs, on the "
        f"{total_budget_s:.0f}s this call is allowed, so its ratio is not a reading of anything"
    )
    if ratio >= tolerance:
        # Re-measure rather than loosen: contention biases pairs upward, while quadratic growth repeats.
        # Affordable pairs come from the remaining budget, since CI runs these with --timeout=330.
        remaining = total_budget_s - (read_clock() - started)
        pair_cost = big * (1.0 + 1.0 / max(ratio, 1.0))
        affordable = repeats_on_retry if pair_cost <= 0 else int(remaining // pair_cost)
        # Fewer than three pairs cannot outvote anything, so the first sample stands.
        assert affordable >= 3, (
            f"{label} path is not linear: {factor}x the input cost {ratio:.1f}x the time over 3 "
            f"pairs (linear is ~{factor}, quadratic is ~{factor ** 2}), and at {big:.1f}s per big "
            f"leg a second sample does not fit in the {remaining:.0f}s left of {total_budget_s:.0f}s, "
            "so this reading stands"
        )
        pairs_wanted = min(repeats_on_retry, affordable)
        confirm, big_again, result, pairs = growth(
            run,
            build,
            units,
            factor,
            repeats = pairs_wanted,
            abort_over_s = budget,
            budget_s = remaining,
            clock = clock,
        )
        # The confirmation must pass on its own reading, not min of both samples.
        assert (
            big_again < budget
        ), f"{label} path took {big_again:.1f}s on {units * factor} units while re-measuring"
        assert pairs == pairs_wanted, (
            f"{label} path: the confirmation stopped after {pairs} of {pairs_wanted} pairs, "
            f"either on a big leg over {budget:.0f}s or on the {remaining:.0f}s the whole "
            "sample is allowed, so its ratio is not a reading of anything"
        )
        assert confirm < tolerance, (
            f"{label} path is not linear: {factor}x the input cost {confirm:.1f}x the time "
            f"over {pairs_wanted} pairs, after {ratio:.1f}x over 3 "
            f"(linear is ~{factor}, quadratic is ~{factor ** 2})"
        )
    return result
