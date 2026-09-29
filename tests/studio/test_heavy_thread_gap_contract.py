# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two methods the viewport-gap measurement in #9058 drives this harness through.

#9058 targets this page rather than duplicating the harness, and its committed
`probe_compact_tail_gap.py` preflights for these exact names and exits naming whichever is
missing. So renaming either one, or dropping a key out of `gapMetrics`, breaks a probe on
another branch rather than anything here, which is precisely the kind of break nothing local
would catch.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "studio" / "frontend" / "smoke-heavy-thread-main.tsx"


def entry() -> str:
    return ENTRY.read_text(encoding = "utf-8")


def test_the_compact_tail_seed_keeps_its_name_and_arity() -> None:
    assert (
        "seedCompactTail(targetChars: number, tailMessages: number)" in entry()
    ), "#9058's probe calls seedCompactTail(chars, tailMessages) by name"


def test_the_compact_tail_seed_reuses_the_ordinary_fixture_builder() -> None:
    # The census parity #9058 relies on comes from calling the same buildThread() seed() calls.
    # Measured: seedCompactTail(25000, 16) reports 36 messages against seed(25000)'s 20, a tail of exactly 16, with
    # every other count unchanged.
    body = entry()
    tail = body[body.index("seedCompactTail(") : body.index("gapMetrics()")]
    assert "buildThread(targetChars)" in tail, (
        "seedCompactTail must build the heavy part with buildThread(targetChars), or its counts "
        "stop matching seed(targetChars) and #9058's numbers stop being comparable"
    )


def test_gap_metrics_keeps_every_key_the_probe_reads() -> None:
    body = entry()
    gap = body[body.index("gapMetrics()") :]
    gap = gap[: gap.index("\n      },")]
    for key in (
        "ok",
        "mountedRows",
        "clientHeight",
        "scrollHeight",
        "scrollTop",
        "maxScrollTop",
        "mountedHeight",
        "gapTop",
        "gapBottom",
        "spacerHeight",
    ):
        # `key:` or the shorthand `key,` / `key\n`.
        assert re.search(
            rf"\b{key}\s*[:,\n]", gap
        ), f"gapMetrics no longer reports {key}, which #9058's probe reads"


def test_the_gap_below_is_measured_against_the_viewport_edge() -> None:
    # Against the box bottom, not scrollHeight.
    # Measured with a 16 message tail: spacerHeight 165, gapBottom 199.
    body = entry()
    gap = body[body.index("gapMetrics()") :]
    gap = gap[: gap.index("\n      },")]
    assert "box.bottom - last.bottom" in gap, (
        "gapBottom must be measured from the viewport's bottom edge, or #9058's numbers stop "
        "being comparable across sizes"
    )
