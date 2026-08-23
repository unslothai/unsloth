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
from core.inference.context_window import (
    _RESULT_NOTICE_RESERVE,
    prompt_budget,
    tool_result_budget,
)


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


def _spill_path(out: str) -> str:
    """The spill path out of a notice, without the punctuation that follows it."""
    return out.split("saved to ")[1].split(" ")[0].rstrip(".,)")


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

    def test_the_floor_yields_on_the_measured_leg_too(self, monkeypatch):
        """The leg the test above misses, and the one that matters most.

        `_exact_prefix_chars` bottoms out on the floor, so with a tokenizer serving the
        request it returned the legacy 2,000 characters however little room was left:
        100 tokens of room bought 2,000 characters, 666 tokens of dense output, 6.6x the
        budget. The accurate leg was the leaky one.
        """
        _window(monkeypatch, 4096)
        _tokenizer(monkeypatch)
        _room(100)

        limit = tools._dense_char_limit(_dense(40_000), tools._MAX_OUTPUT_CHARS)

        assert limit // _CHARS_PER_TOKEN <= 100, "the measured leg overshot the room"
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
        spill = _spill_path(out)
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

    def test_a_mid_line_cut_resumes_by_bytes_rather_than_by_line(self, tmp_path):
        """A line number cannot describe a cut that landed inside a line.

        `shown` counts the partial line as complete, so a line-based resume starts at the
        line AFTER the one the reader is standing in the middle of and skips everything
        still unread in it. On single-line output -- minified JS, base64, one long JSON --
        that is the entire remainder, and `sed` returns nothing at all.
        """
        out = tools._truncate("A" * 40_000, 500, workdir = str(tmp_path))

        assert "sed -n" not in out, "a line number cannot resume a mid-line cut"
        assert "tail -c +501" in out
        # And it really does resume at the first unseen byte.
        spill = _spill_path(out)
        assert (tmp_path / spill).read_text()[500:501] == "A"

    def test_a_boundary_cut_still_resumes_by_line(self, tmp_path):
        """The byte fallback must not swallow the readable case."""
        out = tools._truncate(
            "\n".join(f"line {i}" for i in range(1, 501)), 200, workdir = str(tmp_path)
        )

        assert "sed -n" in out
        assert "tail -c" not in out

    def test_a_multibyte_mid_line_cut_counts_bytes_not_characters(self, tmp_path):
        """`tail -c` counts bytes. Handing it a character count resumes in the middle of a
        codepoint on any non-ASCII output, which is most of a CJK result."""
        out = tools._truncate("你好" * 10_000, 300, workdir = str(tmp_path))

        head = out.split("\n\n... (truncated")[0]
        offset = int(out.split("tail -c +")[1].split(" ")[0]) - 1
        assert offset == len(head.encode("utf-8"))
        assert offset > len(head), "bytes, not characters, for multibyte text"

    def test_the_full_output_is_recoverable_from_the_spill(self, tmp_path):
        text = "\n".join(f"line {i}" for i in range(1, 2_001))

        out = tools._truncate(text, 300, workdir = str(tmp_path))

        spill = _spill_path(out)
        assert (tmp_path / spill).read_text() == text

    def test_the_spill_is_hidden_from_the_created_files_card(self, tmp_path):
        """It lives in a dot-directory, which `_snapshot_workdir_files` skips. Without
        that, every truncated result would grow a phantom download beside it."""
        before = tools._snapshot_workdir_files(str(tmp_path))
        tools._truncate("\n".join(str(i) for i in range(5_000)), 200, workdir = str(tmp_path))
        after = tools._snapshot_workdir_files(str(tmp_path))

        assert after == before

    def test_a_cmd_only_windows_host_gets_no_command_it_cannot_run(self, tmp_path, monkeypatch):
        """`_get_shell_cmd` falls back to `cmd /c` when the host has no trusted bash, and
        cmd has no sed, tail or head. Naming one anyway hands the model a command that
        fails, and the likely next move is re-running the call that truncated."""
        monkeypatch.setattr(tools.sys, "platform", "win32")
        monkeypatch.setattr(tools, "_windows_bash", lambda: None)

        out = tools._truncate(
            "\n".join(f"line {i}" for i in range(1, 501)), 200, workdir = str(tmp_path)
        )

        assert "saved to" in out, "the spill is still worth naming"
        assert "continue with" not in out
        for tool in ("sed -n", "tail -c", "head -c"):
            assert tool not in out

    def test_a_windows_host_with_bash_still_gets_the_command(self, tmp_path, monkeypatch):
        """The guard must not strip paging from the Windows hosts that can page."""
        monkeypatch.setattr(tools.sys, "platform", "win32")
        monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")

        out = tools._truncate(
            "\n".join(f"line {i}" for i in range(1, 501)), 200, workdir = str(tmp_path)
        )

        assert "continue with" in out
        assert "sed -n" in out

    def test_the_spill_is_written_without_newline_translation(self, tmp_path, monkeypatch):
        """The byte offset in the hint is counted from the untranslated text, so the file
        has to hold those same bytes. The default text mode writes os.linesep, which on
        Windows adds a byte per line and moves every later offset, resuming early and
        repeating output.

        Asserted on the open() call rather than on the bytes, because this platform does
        not translate: comparing the file here would pass with or without the fix and
        prove nothing about the platform the bug is on.
        """
        import builtins

        seen = {}
        real_open = builtins.open

        def _recording_open(
            path,
            mode = "r",
            *args,
            **kwargs,
        ):
            if "w" in mode and str(path).endswith(".txt"):
                seen.update(kwargs)
            return real_open(path, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", _recording_open)
        text = "\n".join(f"line {i}" for i in range(1, 200))
        tools._truncate(text, 120, workdir = str(tmp_path))

        assert seen.get("newline") == "", f"spill opened with newline={seen.get('newline')!r}"
        spill = next((tmp_path / tools._SPILL_DIR).iterdir())
        assert spill.read_bytes() == text.encode("utf-8")

    def test_no_workdir_falls_back_to_the_plain_notice(self):
        """A hint naming a file that is not there is worse than admitting it is gone."""
        out = tools._truncate("\n".join(str(i) for i in range(5_000)), 200)

        assert "truncated to" in out
        assert "sed -n" not in out
        assert "saved to" not in out

    def test_spills_do_not_accumulate_without_bound(self, tmp_path):
        # Distinct bodies, so each really is a new spill: identical output is content
        # addressed onto one file and would never exercise the prune at all.
        for n in range(tools._SPILL_KEEP + 6):
            tools._truncate(
                f"run {n}\n" + "\n".join(str(i) for i in range(5_000)), 200, workdir = str(tmp_path)
            )

        kept = os.listdir(tmp_path / tools._SPILL_DIR)
        assert len(kept) <= tools._SPILL_KEEP

    def test_the_same_output_twice_reuses_one_spill(self, tmp_path):
        """Content addressed, so the notice is identical between the streaming and
        non-streaming runs of one call, and printing the same file twice does not fill
        the sandbox with copies."""
        text = "\n".join(f"line {i}" for i in range(1, 2_000))

        first = tools._truncate(text, 200, workdir = str(tmp_path))
        second = tools._truncate(text, 200, workdir = str(tmp_path))

        assert first == second
        assert len(os.listdir(tmp_path / tools._SPILL_DIR)) == 1


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


def _within_room(out: str, room: int) -> None:
    """The body fits the room, and the notice fits the reserve held back for it.

    `tool_result_budget` subtracts `_RESULT_NOTICE_RESERVE` before reporting the room, so
    the notice explaining the cut is already paid for and only the body is measured
    against the number itself.
    """
    body = out.split("\n\n... (")[0]
    assert len(body) // _CHARS_PER_TOKEN <= room, "the body alone overruns the room"
    assert (
        len(out) // _CHARS_PER_TOKEN <= room + _RESULT_NOTICE_RESERVE
    ), "the notice costs more than the reserve held back for it"


class TestEveryToolIsHeldToTheRoom:
    """`python` and `terminal` truncate themselves; the rest hand their string back whole.

    An MCP response is unbounded and an edit receipt or a search result runs to thousands
    of characters, so a tool that is not the code sandbox can overflow the same nearly full
    thread and land in the newest turn, which the next fit protects. Whatever the tool, the
    result has to be held to the room.
    """

    def test_a_web_search_result_is_capped_by_the_room(self, monkeypatch):
        _window(monkeypatch, 4096)
        _tokenizer(monkeypatch)
        monkeypatch.setattr(tools, "_web_search", lambda *a, **k: _dense(40_000))
        out = tools.execute_tool("web_search", {"query": "flappy bird"}, result_budget_tokens = 120)

        assert len(out) < 40_000
        _within_room(out, 120)

    def test_an_mcp_response_is_capped_by_the_room(self, monkeypatch):
        _window(monkeypatch, 4096)
        _tokenizer(monkeypatch)
        monkeypatch.setattr(
            tools.mcp_servers_db,
            "get_server",
            lambda _id: {"url": "https://example.invalid/mcp", "is_enabled": True},
        )
        monkeypatch.setattr(tools, "parse_server_headers", lambda _s: {})
        monkeypatch.setattr(tools, "is_stdio", lambda _u: False)
        monkeypatch.setattr(tools, "call_tool_sync", lambda **k: _dense(40_000))
        out = tools.execute_tool(
            f"{tools.MCP_TOOL_PREFIX}srv__read_file", {}, result_budget_tokens = 120
        )

        assert len(out) < 40_000
        _within_room(out, 120)

    def test_an_edit_receipt_is_capped_by_the_room(self, monkeypatch):
        _window(monkeypatch, 4096)
        _tokenizer(monkeypatch)
        monkeypatch.setattr(tools, "_edit_file", lambda *a, **k: _dense(8_000))
        out = tools.execute_tool(
            "edit_file", {"path": "game.html"}, session_id = None, result_budget_tokens = 120
        )

        assert len(out) < 8_000
        _within_room(out, 120)

    def test_an_unpriced_thread_keeps_todays_result_exactly(self, monkeypatch):
        """The hosted path and every external provider: no room measured, nothing capped.
        Without this leg the change would start truncating results it cannot price."""
        _window(monkeypatch, 4096)
        _tokenizer(monkeypatch)
        page = _dense(40_000)
        monkeypatch.setattr(tools, "_web_search", lambda *a, **k: page)
        assert (
            tools.execute_tool("web_search", {"query": "flappy bird"}, result_budget_tokens = None)
            == page
        )

    def test_a_short_result_survives_a_thread_with_no_room(self, monkeypatch):
        """At zero room the notice IS the message, but only while it is the cheaper of the
        two. Replacing "done" with a longer sentence about "done" being gone spends more
        tokens and loses the answer."""
        _window(monkeypatch, 4096)
        _tokenizer(monkeypatch)
        monkeypatch.setattr(tools, "_web_search", lambda *a, **k: "done")
        assert tools.execute_tool("web_search", {"query": "x"}, result_budget_tokens = 0) == "done"


class TestPruningOnlyTouchesStudioSpills:
    def test_files_studio_did_not_write_are_left_alone(self, tmp_path):
        """The sandbox can be a directory the user already had, including one that already
        holds a `.unsloth_tool_output`. Pruning by extension deletes their files."""
        target = tmp_path / tools._SPILL_DIR
        target.mkdir()
        mine = [target / f"{i:012x}.txt" for i in range(tools._SPILL_KEEP + 5)]
        for path in mine:
            path.write_text("spill")
        theirs = [target / "notes.txt", target / "receipts-2026.txt"]
        for path in theirs:
            path.write_text("keep me")

        tools._prune_spills(str(target))

        assert all(path.exists() for path in theirs)
        assert len([p for p in target.iterdir() if p.name.endswith(".txt")]) == (
            tools._SPILL_KEEP + len(theirs)
        )


class TestTheFrontendEnvelopeSurvivesTheCap:
    """Only what the model is shown is measured, and only it is cut.

    An MCP screenshot comes back as a trailing `__MCP_IMAGES__` JSON array that can run to
    megabytes of base64. `strip_result_for_model` removes it before the result is replayed,
    so it costs the window nothing, and every consumer needs the whole valid array: a cut
    inside it does not lose the image quietly, it hands the model the broken fragment.
    """

    @staticmethod
    def _envelope(pixels: int) -> str:
        return '\n__MCP_IMAGES__:[{"data": "%s", "mimeType": "image/png"}]' % ("A" * pixels)

    def _mcp(self, monkeypatch, result):
        monkeypatch.setattr(
            tools.mcp_servers_db,
            "get_server",
            lambda _id: {"url": "https://example.invalid/mcp", "is_enabled": True},
        )
        monkeypatch.setattr(tools, "parse_server_headers", lambda _s: {})
        monkeypatch.setattr(tools, "is_stdio", lambda _u: False)
        monkeypatch.setattr(tools, "call_tool_sync", lambda **k: result)
        return tools.execute_tool(
            f"{tools.MCP_TOOL_PREFIX}srv__screenshot", {}, result_budget_tokens = 120
        )

    def test_an_image_envelope_comes_back_whole(self, monkeypatch):
        from core.inference.tool_loop_controller import strip_result_for_model

        _window(monkeypatch, 4096)
        _tokenizer(monkeypatch)
        envelope = self._envelope(60_000)

        out = self._mcp(monkeypatch, _dense(40_000) + envelope)

        assert out.endswith(envelope), "the image envelope was cut"
        # And the part that is replayed to the model is the capped one.
        _within_room(strip_result_for_model(out, "mcp"), 120)

    def test_the_envelope_is_not_charged_to_the_room(self, monkeypatch):
        """Its size cannot shrink the body: the model never sees those bytes."""
        _window(monkeypatch, 4096)
        _tokenizer(monkeypatch)
        page = _dense(40_000)

        small = self._mcp(monkeypatch, page + self._envelope(100))
        large = self._mcp(monkeypatch, page + self._envelope(500_000))

        assert small.split("\n__MCP_IMAGES__")[0] == large.split("\n__MCP_IMAGES__")[0]

    def test_text_that_merely_mentions_the_marker_is_still_capped(self, monkeypatch):
        """The conservative half, and the same rule the replay path applies: no valid JSON
        array, no envelope, so it is ordinary output and is measured like any other."""
        _window(monkeypatch, 4096)
        _tokenizer(monkeypatch)

        out = self._mcp(monkeypatch, _dense(40_000) + "\n__MCP_IMAGES__: see the docs")

        _within_room(out, 120)
