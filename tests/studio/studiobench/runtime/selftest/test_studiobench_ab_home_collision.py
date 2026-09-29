# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AN A/B THAT COMPARES A BUILD AGAINST ITSELF, AND LOOKS CLEAN DOING IT.

`install_studio` derives its repo checkout from the home directory, so two arms sharing one home
share one checkout. The second install overwrites the first and both arms then serve whichever
build was installed last. Every downstream number is then a comparison of a build with itself:
the parity digest matches, the invariants agree, the timings sit on top of each other, and none
of it means anything.

This was not hypothetical. Two runs of the same pair reported 716 ms and 718 ms for base and
treatment in one, and 2,583 ms and 2,614 ms in the other: nearly equal within each run, 3.6x
apart between them, because each run was internally uniform and the two runs were serving
different builds. Read within a run it says the change does nothing.

It is the same shape as a copy timing that is really a sleep, and it is the reason this refuses
rather than warns.
"""

from __future__ import annotations

import re
from pathlib import Path

_MAIN = Path(__file__).resolve().parents[2] / "__main__.py"


def _source() -> str:
    return _MAIN.read_text(encoding = "utf-8")


def test_a_shared_home_under_ab_is_refused_rather_than_warned():
    source = _source()
    assert "if not args.attach and args.home:" in source, "the A/B path must refuse a shared home"
    # A refusal, not a log line that scrolls past. `return 2` is the CLI's usage-error code.
    guard = source[source.index("if not args.attach and args.home:") :]
    assert "return 2" in guard[:900], "the guard must exit non-zero, not merely print"


def test_the_refusal_says_what_to_do_instead():
    """A refusal a reader cannot act on gets worked around. Naming the replacement is the
    difference between a guard and an obstacle."""
    source = _source()
    guard = source[source.index("--home cannot be used with --ab") :][:700]
    assert "Drop --home" in guard
    assert "studio_home_" in guard, "it must name the per-arm directory it falls back to"


def test_the_guard_sits_before_any_install_runs():
    """Refusing AFTER the first install has already run costs the caller the slow half of the
    mistake and leaves a half-built home behind."""
    source = _source()
    assert source.index("if not args.attach and args.home:") < source.index(
        "side_install = install_studio(ref, home)"
    )


def test_the_guard_sits_before_the_payload_is_archived():
    """A refusal that starts nothing must cost the previous run nothing.

    `prepare_payload` archives an existing `payload.jsonl` for a fresh run, so a guard placed
    after it answers `--ab X --home H --out DIR` by moving DIR's payload off the standard path and
    THEN exiting 2 having run no benchmark: the next `--resume` finds nothing to continue and
    silently re-runs the whole ladder, and `--report` and `--assert-liveness` open the standard
    name and find no rows. `rollback_session_rows` states the rule this keeps -- a refusal has to
    leave the payload it refused exactly as it found it.

    `invalidate_stale_reports` is held to the same line and for the same reason: it REPLACES
    `summary.md` and `ab.md`, so a refusal reaching it would take the previous run's reports down
    along with its payload.

    The archive is pinned as `archived = prepare_payload(`, which is the literal
    `test_the_run_passes_the_archive_result_to_the_invalidation` pins for the wiring, so the two
    move together instead of one going stale the next time that call is rewritten."""

    source = _source()
    run_body = source[source.index("def run(args, ab_ref = None) -> int:") :]
    for sink in ("archived = prepare_payload(", "invalidate_stale_reports(paths.out"):
        for guard in (
            "if not args.attach and args.home:",
            "if args.attach and not args.attach_b:",
            "injection_problem = stream_cost_injection_problem(",
        ):
            assert run_body.index(guard) < run_body.index(sink), f"{guard} vs {sink}"


def test_a_single_arm_run_still_accepts_home():
    """`--home` is legitimate on its own: there is only one build to install, so there is no
    collision. The guard is on the COMBINATION, and narrowing it wrongly would break the
    single-arm path that people use to pin an install."""
    source = _source()
    guard_line = next(
        line for line in source.split("\n") if "if not args.attach and args.home:" in line
    )
    # The condition must mention the attach case; the A/B-only scoping comes from the block it
    # sits inside, which is guarded by `if ab_ref:`.
    assert "args.attach" in guard_line
    block_start = source.index("if ab_ref:")
    assert block_start < source.index("if not args.attach and args.home:")


def test_the_reason_is_recorded_where_the_next_person_reads_it():
    """The mechanism is not guessable from the flag names, so it is written where the guard is."""
    source = _source()
    comment = source[source.index("ONE HOME CANNOT HOLD TWO BUILDS") :][:1400]
    assert "overwrites the first" in comment
    assert re.search(r"716|2,583", comment), "the measured numbers belong beside the claim"
