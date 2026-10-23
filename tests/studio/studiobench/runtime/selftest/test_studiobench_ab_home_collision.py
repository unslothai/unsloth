# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two A/B arms sharing one home share one checkout, so both serve the last build installed."""

from __future__ import annotations

import re
from pathlib import Path

_MAIN = Path(__file__).resolve().parents[2] / "__main__.py"


def _source() -> str:
    return _MAIN.read_text(encoding = "utf-8")


def test_a_shared_home_under_ab_is_refused_rather_than_warned():
    source = _source()
    assert "if not args.attach and args.home:" in source, "the A/B path must refuse a shared home"
    # A refusal, not a log line that scrolls past; return 2 is the CLI usage-error code.
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
    """A refusal must cost the previous run nothing, so the guard sits before the payload is archived."""

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
    assert "args.attach" in guard_line
    block_start = source.index("if ab_ref:")
    assert block_start < source.index("if not args.attach and args.home:")


def test_the_reason_is_recorded_where_the_next_person_reads_it():
    """The mechanism is not guessable from the flag names, so it is written where the guard is."""
    source = _source()
    comment = source[source.index("ONE HOME CANNOT HOLD TWO BUILDS") :][:1400]
    assert "overwrites the first" in comment
    assert re.search(r"716|2,583", comment), "the measured numbers belong beside the claim"
