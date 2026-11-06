# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Check growth as a paired timing ratio; an absolute time budget flakes on shared runners."""

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
    """Median of paired big/small ratios; if pairs_completed is below repeats the ratio is not a verdict."""
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
    """Fail unless run(build(n)) grows ~factor, not ~factor**2; pass units = old size / factor."""
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
