# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The reply-length axis and the dollar-bearing corpus, which the rung ladder cannot provide.

The ladder pins the streamed reply at `STREAM_TAIL_CHARS` on every rung, deliberately, so that the
thread is the only thing that varies. The consequence is that a cost scaling with the length of the
reply BEING STREAMED is constant across the whole ladder and reads as a floor rather than as an
effect. These two knobs are the axis that can see one, and the tests below pin the properties that
make them trustworthy: the default is unchanged, the frozen corpus is untouched, and a run that
uses either of them says so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.fixture.corpus import (  # noqa: E402
    STREAM_TAIL_CHARS,
    Corpus,
    dollarise,
    plan_rung,
)

FROZEN = Path(__file__).resolve().parents[1] / "corpus" / "frozen"


@pytest.fixture(scope = "module")
def corpus() -> Corpus:
    return Corpus.load()


def _streamed_text(plan) -> str:
    unit = plan.streamed_unit
    return "" if unit is None else unit.reasoning + unit.content


def test_the_default_is_exactly_what_it_was(corpus: Corpus):
    """No argument means no change, so every earlier payload stays comparable."""
    before = plan_rung(corpus, "100K")
    after = plan_rung(corpus, "100K", stream_tail_chars = None, dollars = False)
    assert before.streamed_chars == after.streamed_chars
    assert _streamed_text(before) == _streamed_text(after)
    assert before.streamed_chars <= STREAM_TAIL_CHARS


def test_the_ladder_really_does_pin_the_reply_length(corpus: Corpus):
    """The premise of the whole axis, asserted rather than described.

    If this ever fails, the rung ladder has started varying reply length and a reply-length
    investigation can use it directly.

    Not exact equality: the tail is clipped at whole BLOCK boundaries, so it lands a little under
    the budget and where it lands depends on the unit. The invariant is that it does not GROW with
    the rung, which is what decides whether the ladder can see a reply-length effect.
    """
    plans = {rung: plan_rung(corpus, rung) for rung in ("1K", "10K", "100K")}
    lengths = {rung: p.streamed_chars for rung, p in plans.items()}
    assert max(lengths.values()) <= STREAM_TAIL_CHARS, lengths

    # The thread grows tenfold between these two rungs; the reply must not.
    thread_ratio = plans["100K"].total_chars / plans["10K"].total_chars
    reply_ratio = lengths["100K"] / lengths["10K"]
    assert thread_ratio > 8, (thread_ratio, plans["10K"].total_chars, plans["100K"].total_chars)
    assert reply_ratio < 1.1, (reply_ratio, lengths)


def test_the_tail_override_moves_the_reply_and_not_the_thread(corpus: Corpus):
    """Reply length is the variable; total thread size stays put, or the two are confounded."""
    base = plan_rung(corpus, "100K")
    for tail in (24_000, 96_000):
        plan = plan_rung(corpus, "100K", stream_tail_chars = tail)
        assert abs(plan.streamed_chars - tail) < tail * 0.1, (tail, plan.streamed_chars)
        # Within a few percent: the seeded prefix is trimmed to compensate, so the cell measures
        # a different SPLIT of the same total rather than a bigger thread.
        assert abs(plan.total_chars - base.total_chars) < base.total_chars * 0.05, (
            tail,
            plan.total_chars,
            base.total_chars,
        )


def test_the_tail_override_grows_monotonically(corpus: Corpus):
    seen = [
        plan_rung(corpus, "100K", stream_tail_chars = t).streamed_chars
        for t in (6_000, 12_000, 24_000, 48_000, 96_000)
    ]
    assert seen == sorted(seen), seen
    assert seen[-1] > seen[0] * 10, seen


def test_the_frozen_corpus_now_carries_math_of_its_own(corpus: Corpus):
    """This test used to assert the opposite, and the change is the point.

    It was written to pin a defect: the frozen corpus contained zero `$`, so `preprocessLaTeX`
    always took its cheap early return and `--corpus-dollars` was the only way past it. Its
    docstring said that if a future corpus gained a `$` of its own, this test would fail and
    whoever changed it would have to decide whether `--corpus-dollars` was still needed rather
    than let the two silently overlap.

    Corpus v2 did exactly that, and the gate fired as designed. The answer to the question it
    forced is in `dollarise`'s docstring: the flag is no longer what reaches the expensive regime,
    but it is not redundant either, because v2's dollars are well-formed math that the currency
    pass SKIPS and its dollars are false positives that the currency pass REWRITES.

    So the assertion is inverted rather than deleted. A future corpus that loses its math would
    silently return every run to the cheap path, and that must fail here too.
    """
    text = "".join(
        json.loads(line)["reasoning"] + json.loads(line)["content"]
        for line in (FROZEN / "units.jsonl").read_text(encoding = "utf-8").splitlines()
        if line.strip()
    )
    assert text.count("$") > 0, "corpus v2 puts math in the frozen units; this found none"
    # Both delimiter families, because `convertLatexDelimiters` handles them on different paths.
    assert text.count("\\[") > 0
    assert text.count("\\(") > 0


def test_dollars_reach_the_streamed_turn_and_nothing_else(corpus: Corpus):
    """The flag ADDS to the streamed turn; it no longer creates the only dollars in it.

    Under corpus v2 the streamed unit is drawn from a corpus that carries math, so it already has
    `$` before the flag is applied. What the flag has to do now is add MORE, and add them only
    where they belong.
    """
    plain = plan_rung(corpus, "100K", stream_tail_chars = 24_000)
    salted = plan_rung(corpus, "100K", stream_tail_chars = 24_000, dollars = True)
    assert _streamed_text(salted).count("$") > _streamed_text(plain).count("$")
    # The seeded prefix is rendered once at mount and never re-preprocessed, so dollars there
    # would change the corpus without changing what the per-frame path is asked to do. v2's own
    # math is expected there; what must not appear is anything THIS added, so the prefix has to
    # be byte-identical with the flag on and off.
    assert [(u.reasoning, u.content) for u in salted.seeded_units] == [
        (u.reasoning, u.content) for u in plain.seeded_units
    ]


def test_the_flag_is_not_a_no_op_under_corpus_v2(corpus: Corpus):
    """FAILURE DIRECTION. A flag that stopped changing anything must not keep shipping quietly.

    v2 defeats `preprocessLaTeX`'s early return on its own, so the flag's original justification is
    gone. It survives because it exercises a different branch of the same function: dollars that
    are NOT math, which the currency pass has to escape or exclude rather than skip. If a future
    corpus makes even that indistinguishable, this fails and the flag should be deleted.
    """
    plain = _streamed_text(plan_rung(corpus, "100K", stream_tail_chars = 24_000))
    salted = _streamed_text(plan_rung(corpus, "100K", stream_tail_chars = 24_000, dollars = True))
    assert salted != plain
    # Shell-shaped and price-shaped, which is what makes them false positives rather than math.
    added = salted.count("$") - plain.count("$")
    assert added > 0, "the flag added no dollars, so it is a no-op and should be removed"
    assert "$HOME/" in salted or ".99" in salted


def test_dollars_are_deterministic(corpus: Corpus):
    a = _streamed_text(plan_rung(corpus, "100K", stream_tail_chars = 24_000, dollars = True))
    b = _streamed_text(plan_rung(corpus, "100K", stream_tail_chars = 24_000, dollars = True))
    assert a == b


def test_dollarise_keeps_shell_dollars_inside_the_fence(corpus: Corpus):
    """The two branches are not interchangeable and the test says which is which.

    A `$` inside a fence must be EXCLUDED by the code-region scan and a `$` in prose must be
    escaped by the currency pass. A generator that only produced one of them would exercise half
    the function while looking like it exercised all of it.
    """
    source = "\n".join(
        [
            "```bash",
            *[f"line {i}" for i in range(30)],
            "```",
            "",
            *[f"prose line {i}" for i in range(30)],
        ]
    )
    out = dollarise(source, "x")
    fenced, prose = [], []
    inside = False
    for line in out.split("\n"):
        if line.lstrip().startswith("```"):
            inside = not inside
            continue
        (fenced if inside else prose).append(line)
    assert any("$" in line for line in fenced), out
    assert any("$" in line for line in prose), out


def test_dollarise_does_not_reduce_the_text(corpus: Corpus):
    """It only adds, so a dollarised cell is never SHORTER than its plain twin."""
    source = "\n".join(f"line {i}" for i in range(50))
    assert len(dollarise(source, "x")) >= len(source)
    assert dollarise("", "x") == ""


# ── the axis has to survive the film, not just the fixture ───────────────────────────────────


def test_a_long_tail_really_does_outlast_the_standard_film(corpus: Corpus):
    """The premise of the test below, taken from the real corpus rather than asserted.

    The films are packed against the DEFAULT tail: `stop_generation` opens at 28 s on the standard
    film and the pinned 6,000 character tail drains in at most 17.8 s, so nothing of the cell's own
    is ever running when that slot opens. Raise the tail and the slot lands mid-stream.
    """
    from studiobench.scene.schedule import STANDARD

    field_chars_per_sec = 24 / 0.073
    stop = next(s for s in STANDARD.slots if s.action == "stop_generation")

    default_s = plan_rung(corpus, "100K").streamed_chars / field_chars_per_sec
    assert default_s < stop.t_start_ms / 1000.0, default_s

    long_s = (
        plan_rung(corpus, "100K", stream_tail_chars = 96_000).streamed_chars / field_chars_per_sec
    )
    assert long_s > STANDARD.duration_ms / 1000.0, long_s
    assert long_s > stop.t_start_ms / 1000.0


class _FakeKeyboard:
    def __init__(self, page) -> None:
        self.pressed: list[str] = []
        self._page = page

    def press(self, key: str) -> None:
        self.pressed.append(key)
        if key == "Enter" and "one more" in self._page.filled:
            # Sending the throwaway turn starts a generation, which is what the own-turn path
            # then waits for and stops.
            self._page.running = True


class _FakePage:
    """The four page calls `stop_generation` makes, and a record of what it reached for."""

    def __init__(self, *, running: bool) -> None:
        self.running = running
        self.filled: list[str] = []
        self.queried: list[str] = []
        self.clicked = 0
        self.keyboard = _FakeKeyboard(self)

    def evaluate(
        self,
        script,
        arg = None,
    ):
        if "isRunning" in script:
            return self.running
        if "composerText" in script:
            return ""
        if "assistantChars" in script:
            return 9_200
        return {}

    def fill(self, _selector: str, text: str) -> None:
        self.filled.append(text)

    def wait_for_timeout(self, _ms) -> None:
        return None

    def query_selector(self, selector: str):
        self.queried.append(selector)
        page = self

        class _Button:
            def click(self_inner) -> None:
                page.clicked += 1
                page.running = False

        return _Button() if "Stop generating" in selector else None


def _stop_ctx(page: _FakePage, budget_ms: int = 3_000):
    """3,000 ms because that is the stop slot on the fast and quick films, and the smallest one
    any film gives this action. The budget decides whether the throwaway turn is affordable at
    all -- see `OWN_TURN_RESERVE_MS` -- so a figure no film uses would test a slot that does not
    exist. The truncation test below keeps a deliberately tiny one for its own reason."""

    from studiobench.runtime.types import ActionContext
    return ActionContext(
        page = page,
        cdp = None,
        cell = None,
        window = None,
        args = {},
        budget_ms = budget_ms,
        dom = None,
        log = lambda _m: None,
    )


def test_stop_refuses_to_truncate_the_cell_s_own_reply():
    """REGRESSION, and the failure it pins is a SILENT one.

    `stop_generation` sends and stops a throwaway turn precisely so that it never truncates the
    reply the rest of the film measures. That guard was written as "if nothing is running, make
    something to stop", so the moment something WAS running the action fell through and clicked
    Stop on it. `--stream-tail-chars 96000` is the supported way to make that happen: the reply
    then streams for 291 s against a 243 s standard film, this slot opens at 28 s, and the reply
    the flag exists to lengthen is cut at about 9,200 characters. Every later action still runs
    against a settled thread, the row still says `ran: true`, and `--assert-liveness` -- which the
    flag's own help text sends the caller to -- still passes, so the reply-length axis reports a
    clean run having measured a reply a tenth of the requested size.
    """
    from studiobench.scene.actions import stop_generation

    page = _FakePage(running = True)
    # 200 ms: too small to hold the throwaway turn at all, so the drain wait is zero and the
    # refusal below is the one this test is about rather than the budget check beside it.
    result = stop_generation(_stop_ctx(page, budget_ms = 200))

    assert result.ran is False, "a stop that would truncate the measured reply must not run"
    assert "truncate" in (result.reason or "")
    assert page.clicked == 0, "the cell's own reply was stopped"
    assert page.queried == [], "the stop button was not even looked for"
    assert "one more" not in page.filled, "a second turn must not be started on top of a live one"


def test_stop_still_sends_its_own_turn_when_nothing_is_running():
    """The default path, unchanged: with the pinned tail nothing is ever running at this slot."""
    from studiobench.scene.actions import stop_generation

    page = _FakePage(running = False)
    result = stop_generation(_stop_ctx(page))

    assert "one more" in page.filled, "stop must still get its own generation to stop"
    assert page.keyboard.pressed == ["Enter"]
    assert page.clicked == 1, "the throwaway turn is what gets stopped"
    assert result.ran is True
    assert result.expect["own_generation"] is True
