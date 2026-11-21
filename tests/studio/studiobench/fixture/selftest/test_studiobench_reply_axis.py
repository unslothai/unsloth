# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rung ladder pins the streamed reply at STREAM_TAIL_CHARS, so only this axis can vary its length."""

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
    """The streamed tail must not grow with the rung; block clipping means it lands under budget."""
    plans = {rung: plan_rung(corpus, rung) for rung in ("1K", "10K", "100K")}
    lengths = {rung: p.streamed_chars for rung, p in plans.items()}
    assert max(lengths.values()) <= STREAM_TAIL_CHARS, lengths

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
        # Prefix is trimmed to compensate, so the cell measures a different split of the same total.
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
    """The frozen corpus must keep its own math; losing it would silently return runs to the cheap path."""
    text = "".join(
        json.loads(line)["reasoning"] + json.loads(line)["content"]
        for line in (FROZEN / "units.jsonl").read_text(encoding = "utf-8").splitlines()
        if line.strip()
    )
    assert text.count("$") > 0, "corpus v2 puts math in the frozen units; this found none"
    # convertLatexDelimiters handles the two families on different paths.
    assert text.count("\\[") > 0
    assert text.count("\\(") > 0


def test_dollars_reach_the_streamed_turn_and_nothing_else(corpus: Corpus):
    """The flag adds dollars to the streamed turn on top of the corpus's own, and nowhere else."""
    plain = plan_rung(corpus, "100K", stream_tail_chars = 24_000)
    salted = plan_rung(corpus, "100K", stream_tail_chars = 24_000, dollars = True)
    assert _streamed_text(salted).count("$") > _streamed_text(plain).count("$")
    # The seeded prefix is never re-preprocessed, so it must be identical with the flag on and off.
    assert [(u.reasoning, u.content) for u in salted.seeded_units] == [
        (u.reasoning, u.content) for u in plain.seeded_units
    ]


def test_the_flag_is_not_a_no_op_under_corpus_v2(corpus: Corpus):
    """The flag must still change non-math dollars the currency pass handles, or it should be deleted."""
    plain = _streamed_text(plan_rung(corpus, "100K", stream_tail_chars = 24_000))
    salted = _streamed_text(plan_rung(corpus, "100K", stream_tail_chars = 24_000, dollars = True))
    assert salted != plain
    added = salted.count("$") - plain.count("$")
    assert added > 0, "the flag added no dollars, so it is a no-op and should be removed"
    assert "$HOME/" in salted or ".99" in salted


def test_dollars_are_deterministic(corpus: Corpus):
    a = _streamed_text(plan_rung(corpus, "100K", stream_tail_chars = 24_000, dollars = True))
    b = _streamed_text(plan_rung(corpus, "100K", stream_tail_chars = 24_000, dollars = True))
    assert a == b


def test_dollarise_keeps_shell_dollars_inside_the_fence(corpus: Corpus):
    """Both branches are needed: fenced dollars must be excluded by the code scan, prose dollars escaped."""
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


def test_a_long_tail_really_does_outlast_the_standard_film(corpus: Corpus):
    """Under the default tail, the standard film's stop_generation slot opens after the reply drains."""
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
    """Budget matches the smallest real stop slot; the throwaway turn's affordability depends on it."""

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
    """stop_generation must not Stop a reply that is already running, or it silently truncates it."""
    from studiobench.scene.actions import stop_generation

    page = _FakePage(running = True)
    # 200 ms cannot hold the throwaway turn, so the drain wait is zero and the refusal is isolated.
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
