# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tool result has to fit the room the conversation has LEFT, not a share of the window.

`test_web_page_cap_fits_window.py` made the cap a share of the loaded window, which fixed
one result being larger than the whole prompt budget. It does not fall as the thread fills,
so the last result before an overflow is allowed exactly as much as the first:

    ctx 4096 -> min(16000, max(2000, 4096 * 4 * 0.35)) = 5734 chars

which is roughly 1,900 tokens of the dense ASCII these tools print, about 47% of the whole
window, granted no matter how little is left. Reported live: `Qwen3.6-35B-A3B-GGUF` at 4096
with code execution, "Create a Flappy Bird and save it to game.html", then "Print and show
game.html verbatim" repeatedly. Each `cat` is individually under the cap and they accumulate
until the turn cannot be served, and nothing downstream recovers -- the fit protects the
newest turn, so compaction may not drop the very result that does not fit.
"""

from __future__ import annotations

import os

import pytest

from core.inference import tools
from core.inference.context_window import prompt_budget, tool_result_budget


@pytest.fixture(autouse = True)
def _unknown_window(monkeypatch):
    """Default to "no model loaded", and to "the caller did not price the thread"."""
    monkeypatch.setattr(tools, "_loaded_context_tokens", lambda: None)
    ctx_token = tools._REQUEST_CONTEXT_TOKENS.set(tools._UNSET_CONTEXT_TOKENS)
    room_token = tools._REQUEST_RESULT_BUDGET.set(None)
    yield
    tools._REQUEST_CONTEXT_TOKENS.reset(ctx_token)
    tools._REQUEST_RESULT_BUDGET.reset(room_token)


def _window(monkeypatch, ctx):
    monkeypatch.setattr(tools, "_loaded_context_tokens", lambda: ctx)


def _room(value):
    tools._REQUEST_RESULT_BUDGET.set(value)


# Dense ASCII, which is what a printed file is: no spaces to tokenise cheaply.
def _dense(chars: int) -> str:
    line = "x" * 79
    out = "\n".join(line for _ in range(chars // 80 + 1))
    return out[:chars]


class TestTheBudgetFollowsTheRoom:
    def test_it_shrinks_as_the_conversation_grows(self):
        """The whole point: the same window, three different answers."""
        ctx, target = 4096, prompt_budget(4096, None)
        early = tool_result_budget(ctx, None, 200)
        middle = tool_result_budget(ctx, None, target // 2)
        late = tool_result_budget(ctx, None, target - 100)

        assert early > middle > late
        # And never more than the budget it is measured against.
        assert early <= target

    def test_a_full_thread_gets_nothing_rather_than_a_share(self):
        """At or past the budget the answer is zero, not a floor. A floor here is the
        overflow: the result lands in the newest turn, which the next fit protects."""
        target = prompt_budget(4096, None)
        assert tool_result_budget(4096, None, target) == 0
        assert tool_result_budget(4096, None, target + 5_000) == 0

    def test_it_reserves_the_reply_room_rather_than_filling_the_window(self):
        """Sized against prompt_budget, not context_length. Filling to 99% of the physical
        window leaves the prompt fitting and nothing to answer in, which is the
        reserve-missing case the rolling fit already has to rescue."""
        room = tool_result_budget(4096, None, 0)
        assert room < prompt_budget(4096, None) <= 4096
        # The gap is the reply reserve, and it is most of the difference from the window.
        assert 4096 - room > 1_000


class TestTheCharacterCapHonoursIt:
    def test_a_nearly_full_thread_cannot_spend_the_window_share(self, monkeypatch):
        """5,734 characters is what the window share allows at ctx 4096. With 120 tokens
        of room left it must not be handed anything close to that."""
        _window(monkeypatch, 4096)
        text = _dense(40_000)

        wide = tools._dense_char_limit(text, tools._MAX_OUTPUT_CHARS)
        _room(120)
        narrow = tools._dense_char_limit(text, tools._MAX_OUTPUT_CHARS)

        assert narrow < wide
        # 120 tokens of dense ASCII is a few hundred characters, nowhere near the share.
        assert narrow < 2_000

    def test_the_floor_yields_when_the_room_is_smaller_than_it(self, monkeypatch):
        """_MIN_PAGE_CHARS keeps a result worth reading, but it is a comfort and not a
        right: holding 2,000 characters of dense output when 30 tokens remain reinstates
        the overflow the budget exists to prevent."""
        _window(monkeypatch, 4096)
        _room(30)

        limit = tools._dense_char_limit(_dense(40_000), tools._MAX_OUTPUT_CHARS)

        assert limit < tools._MIN_PAGE_CHARS

    def test_an_unpriced_call_behaves_exactly_as_before(self, monkeypatch):
        """The blast radius. External providers and the hosted path never set a room, and
        must see the window-share cap they saw before this existed."""
        _window(monkeypatch, 4096)
        text = _dense(40_000)

        before = tools._dense_char_limit(text, tools._MAX_OUTPUT_CHARS)
        _room(None)
        after = tools._dense_char_limit(text, tools._MAX_OUTPUT_CHARS)

        assert after == before

    def test_an_unknown_window_still_keeps_the_caller_cap(self, monkeypatch):
        """No window means no share to take, with or without a room."""
        _room(50)
        assert tools._dense_char_limit(_dense(40_000), 9_000) == 9_000


class TestPagingTheRest:
    def test_the_hint_names_a_range_that_resumes_where_the_head_stopped(self, tmp_path):
        """A truncation the model can act on. The line the hint names must be the line
        after the last one shown -- off by one either repeats a line of the user's file or
        loses it."""
        text = "\n".join(f"line {i}" for i in range(1, 501))

        out = tools._truncate(text, 200, workdir = str(tmp_path))

        head = out.split("\n\n... (truncated")[0]
        shown = head.count("\n") + 1
        assert f"showing lines 1-{shown} of 500" in out
        assert f"sed -n '{shown + 1}," in out
        # The named line really is the next one, read back off the spill.
        spill = out.split("saved to ")[1].split(" ")[0]
        full = (tmp_path / spill).read_text().splitlines()
        assert full[shown] == f"line {shown + 1}"

    def test_the_head_stops_on_a_line_boundary(self, tmp_path):
        """So the hint's line number is exact rather than approximate."""
        text = "\n".join(f"line {i}" for i in range(1, 501))

        out = tools._truncate(text, 200, workdir = str(tmp_path))

        head = out.split("\n\n... (truncated")[0]
        assert not head.endswith("\n")
        assert head.splitlines()[-1] == f"line {head.count(chr(10)) + 1}"

    def test_one_enormous_line_is_still_cut_rather_than_dropped(self, tmp_path):
        """Minified JS and base64 have no newline to rewind to. Rewinding anyway would
        throw the whole result away to keep the hint tidy."""
        out = tools._truncate("A" * 40_000, 500, workdir = str(tmp_path))

        head = out.split("\n\n... (truncated")[0]
        assert len(head) == 500

    def test_the_full_output_is_recoverable_from_the_spill(self, tmp_path):
        text = "\n".join(f"line {i}" for i in range(1, 2_001))

        out = tools._truncate(text, 300, workdir = str(tmp_path))

        spill = out.split("saved to ")[1].split(" ")[0]
        assert (tmp_path / spill).read_text() == text

    def test_the_spill_is_hidden_from_the_created_files_card(self, tmp_path):
        """It lives in a dot-directory, which `_snapshot_workdir_files` skips. Without
        that, every truncated result would grow a phantom download beside it."""
        before = tools._snapshot_workdir_files(str(tmp_path))
        tools._truncate("\n".join(str(i) for i in range(5_000)), 200, workdir = str(tmp_path))
        after = tools._snapshot_workdir_files(str(tmp_path))

        assert after == before

    def test_no_workdir_falls_back_to_the_plain_notice(self):
        """A hint naming a file that is not there is worse than admitting it is gone."""
        out = tools._truncate("\n".join(str(i) for i in range(5_000)), 200)

        assert "truncated to" in out
        assert "sed -n" not in out
        assert "saved to" not in out

    def test_spills_do_not_accumulate_without_bound(self, tmp_path):
        text = "\n".join(str(i) for i in range(5_000))
        for _ in range(tools._SPILL_KEEP + 6):
            tools._truncate(text, 200, workdir = str(tmp_path))

        kept = os.listdir(tmp_path / tools._SPILL_DIR)
        assert len(kept) <= tools._SPILL_KEEP


# Three characters per token, which is what the code tools actually print: minified HTML,
# base64 and hexdumps all run nearer three than the four the character estimate assumes.
# `_loaded_token_counter` is the same seam llama_cpp fills with the serving model.
_CHARS_PER_TOKEN = 3


def _tokenizer(monkeypatch):
    monkeypatch.setattr(
        tools, "_loaded_token_counter", lambda ctx: (lambda chunk: len(chunk) // _CHARS_PER_TOKEN)
    )


def _cat_game_html(monkeypatch, page, *, price_the_room):
    """Run `cat game.html` three times and return what the thread ends up spending."""
    ctx = 4096
    spent = 300  # system turn plus the first question
    for _ in range(3):
        _room(tool_result_budget(ctx, None, spent) if price_the_room else None)
        limit = tools._dense_char_limit(page, tools._tool_result_char_budget())
        served = tools._truncate(page, limit, workdir = None)
        spent += len(served) // _CHARS_PER_TOKEN + 40  # the result plus the turn's framing
    return spent


class TestTheReportedScenario:
    def test_repeated_prints_of_one_file_stay_inside_the_window(self, monkeypatch):
        """The user's repro, as arithmetic. `cat game.html` three times at ctx 4096.

        Reported against Qwen3.6-35B-A3B at 4096: each result is individually under the
        cap, and the third turn cannot be served. Priced against the room left, and with
        the serving model pricing the characters, the three together never pass the budget.
        """
        _window(monkeypatch, 4096)
        _tokenizer(monkeypatch)
        target = prompt_budget(4096, None)

        spent = _cat_game_html(monkeypatch, _dense(40_000), price_the_room = True)

        assert spent < target, (
            f"three printed results spent {spent} tokens of a {target}-token budget; "
            "the thread can no longer be served and the fit cannot evict the newest turn"
        )

    def test_the_same_loop_overflows_when_the_room_is_ignored(self, monkeypatch):
        """The control, and what makes the test above non-vacuous: with the room unset --
        exactly the old behaviour -- the same three prints do overflow."""
        _window(monkeypatch, 4096)
        _tokenizer(monkeypatch)
        target = prompt_budget(4096, None)

        spent = _cat_game_html(monkeypatch, _dense(40_000), price_the_room = False)

        assert spent > target

    def test_without_a_tokenizer_the_estimate_alone_is_optimistic(self, monkeypatch):
        """A limitation worth stating rather than discovering later.

        With no model able to price a string, the room is still converted to characters at
        the four-per-token ENGLISH rate, so output that really costs three per token
        overspends by about a third. Every local GGUF path has a tokenizer (llama_cpp
        hands it to `_loaded_token_counter`); this is the shape of the residual risk on a
        path that does not, and the reason the conversion is measured when it can be.
        """
        _window(monkeypatch, 4096)  # deliberately NO _tokenizer()
        target = prompt_budget(4096, None)

        spent = _cat_game_html(monkeypatch, _dense(40_000), price_the_room = True)

        assert spent > target
