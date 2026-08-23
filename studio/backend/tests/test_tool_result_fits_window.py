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

import hashlib
import shutil
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


@pytest.fixture(autouse = True)
def _records(tmp_path_factory, monkeypatch):
    """Ownership records live in Studio's own storage, so tests get their own copy of it
    rather than writing into the real one."""
    where = tmp_path_factory.mktemp("tool-output-records")
    monkeypatch.setattr(tools, "_spill_records_dir", lambda: str(where))


def _own(workdir) -> "os.PathLike":
    """The spill directory as Studio itself would leave it: created, and recorded as ours.

    Tests that pre-create it are standing in for a sandbox this process has already
    spilled into. A directory made any other way has no record, which is the case
    `_own_spill_root` deliberately refuses.
    """
    root = workdir / tools._SPILL_DIR
    assert tools._own_spill_root(str(root))
    return root


def _spills(directory) -> list:
    """Everything in a spill directory. The ownership record is not in there."""
    return list(directory.iterdir())


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

    def test_an_unknown_window_with_no_room_keeps_the_caller_cap(self):
        """Nothing measured at all, so there is nothing to lower the cap with."""
        assert tools._dense_char_limit(_dense(40_000), 9_000) == 9_000

    def test_a_known_room_holds_even_with_an_unknown_window(self, monkeypatch):
        """The native case: no resident GGUF, so `_window_context_tokens` sees nothing,
        while the loop that called it knows exactly how full the thread is. Reading the
        window as "no limits" there leaves every result uncapped on the one path with no
        rolling fit to recover."""
        monkeypatch.setattr(tools, "_loaded_token_counter", lambda ctx: None)
        _room(50)

        limit = tools._dense_char_limit(_dense(40_000), 9_000)

        assert 0 < limit <= 50 * 4 * tools._UNMEASURED_ROOM_MARGIN


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
        seen = {}
        real_fdopen = os.fdopen

        def _recording_fdopen(
            fd,
            mode = "r",
            *args,
            **kwargs,
        ):
            # os.fdopen since the spill is opened O_NOFOLLOW by descriptor; the kwarg the
            # newline behaviour rides on is the same one either way.
            if "w" in mode:
                seen.update(kwargs)
            return real_fdopen(fd, mode, *args, **kwargs)

        monkeypatch.setattr(os, "fdopen", _recording_fdopen)
        text = "\n".join(f"line {i}" for i in range(1, 200))
        out = tools._truncate(text, 120, workdir = str(tmp_path))

        assert seen.get("newline") == "", f"spill opened with newline={seen.get('newline')!r}"
        assert (tmp_path / _spill_path(out)).read_bytes() == text.encode("utf-8")

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

        assert len(_spills(tmp_path / tools._SPILL_DIR)) <= tools._SPILL_KEEP

    def test_the_same_output_twice_reuses_one_spill(self, tmp_path):
        """Content addressed, so the notice is identical between the streaming and
        non-streaming runs of one call, and printing the same file twice does not fill
        the sandbox with copies."""
        text = "\n".join(f"line {i}" for i in range(1, 2_000))

        first = tools._truncate(text, 200, workdir = str(tmp_path))
        second = tools._truncate(text, 200, workdir = str(tmp_path))

        assert first == second
        assert len(_spills(tmp_path / tools._SPILL_DIR)) == 1


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

    def test_without_a_tokenizer_the_margin_still_keeps_it_inside(self, monkeypatch):
        """The native path, where nothing can price a string exactly.

        `_loaded_token_counter` answers only for a resident GGUF, so a safetensors model
        converts its room with the four-characters-per-token English estimate, and the
        dense ASCII these tools print runs nearer three. Left alone that overspends by
        about a third, on the one loop with no rolling fit to recover with, so the
        conversion is halved instead (`_UNMEASURED_ROOM_MARGIN`).
        """
        _window(monkeypatch, 4096)  # deliberately NO _tokenizer()
        target = prompt_budget(4096, None)

        spent = _cat_game_html(monkeypatch, _dense(40_000), price_the_room = True)

        assert spent < target


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
        target = _own(tmp_path)
        mine = [target / f"{i:012x}.txt" for i in range(tools._SPILL_KEEP + 5)]
        for path in mine:
            path.write_text("spill")
        tools._write_spill_manifest(
            str(target),
            {p.name: (tools._spill_stamp(str(p)), tools._file_digest(str(p))) for p in mine},
        )
        theirs = [target / "notes.txt", target / "receipts-2026.txt"]
        for path in theirs:
            path.write_text("keep me")

        tools._prune_spills(str(target))

        assert all(path.exists() for path in theirs)
        assert len([p for p in target.iterdir() if p.name.endswith(".txt")]) == (
            tools._SPILL_KEEP + len(theirs)
        )

    def test_a_spill_named_file_this_did_not_write_is_not_pruned(self, tmp_path):
        """The marker records the DIRECTORY, and tool code can create a file with a
        plausible name in it afterwards. A name is not evidence of who wrote it."""
        target = _own(tmp_path)
        theirs = target / "abcdef123456.txt"
        theirs.write_text("mine")
        for n in range(tools._SPILL_KEEP + 5):
            tools._truncate(f"run {n}\n" + _dense(3_000), 200, workdir = str(tmp_path))

        assert theirs.read_text() == "mine"

    def test_and_the_cleanup_reads_it_the_same_way(self, tmp_path):
        target = _own(tmp_path)
        (target / "abcdef123456.txt").write_text("mine")

        assert not tools._holds_no_user_files(str(tmp_path))

    def test_an_unowned_directory_is_not_pruned_at_all(self, tmp_path):
        """A name proves nothing about who wrote it. Without the marker this directory came
        with the sandbox, and every file in it is the user's however it is named."""
        target = tmp_path / tools._SPILL_DIR
        target.mkdir()
        theirs = [target / f"{i:012x}.txt" for i in range(tools._SPILL_KEEP + 5)]
        for path in theirs:
            path.write_text("mine")

        tools._prune_spills(str(target))

        assert all(path.exists() for path in theirs)

    def test_an_unowned_directory_is_never_written_to(self, tmp_path):
        """And nothing is added to it either, so the notice falls back to no paging hint
        rather than putting Studio's files among the user's."""
        (tmp_path / tools._SPILL_DIR).mkdir()
        (tmp_path / tools._SPILL_DIR / "notes.txt").write_text("mine")

        out = tools._truncate("\n".join(str(i) for i in range(5_000)), 200, workdir = str(tmp_path))

        assert "saved to" not in out
        assert [p.name for p in (tmp_path / tools._SPILL_DIR).iterdir()] == ["notes.txt"]

    def test_a_file_named_like_a_spill_in_an_unowned_directory_is_user_content(self, tmp_path):
        """And the cleanup reads it the same way, so deleting the chat does not take it."""
        target = tmp_path / tools._SPILL_DIR
        target.mkdir()
        (target / "abcdef123456.txt").write_text("mine")

        assert not tools._holds_no_user_files(str(tmp_path))


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


class TestTheSpillStaysInsideTheSandbox:
    """The workdir is a directory the model runs commands in, and can be a project the
    user opened. Anything there may be a symlink it made or one that came with the
    project, and following it writes this result outside the sandbox with the backend's
    own permissions."""

    def test_a_symlinked_spill_directory_is_refused(self, tmp_path):
        workdir = tmp_path / "sandbox"
        workdir.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (workdir / tools._SPILL_DIR).symlink_to(outside, target_is_directory = True)

        out = tools._truncate("\n".join(str(i) for i in range(5_000)), 200, workdir = str(workdir))

        assert list(outside.iterdir()) == [], "the spill was written outside the sandbox"
        # Refused, not crashed: the notice is still served, just without a paging hint.
        assert "truncated to" in out
        assert "saved to" not in out

    def test_a_symlinked_spill_file_is_refused(self, tmp_path):
        workdir = tmp_path / "sandbox"
        workdir.mkdir()
        victim = tmp_path / "victim.txt"
        victim.write_text("do not overwrite me")
        text = "\n".join(str(i) for i in range(5_000))
        target = _own(workdir)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        (target / f"{digest}.txt").symlink_to(victim)

        out = tools._truncate(text, 200, workdir = str(workdir))

        assert victim.read_text() == "do not overwrite me"
        # Refused rather than replaced: whatever is at that path is not a spill this
        # recorded, so it is the user's, and the notice does without a paging hint.
        assert "saved to" not in out
        assert (target / f"{digest}.txt").is_symlink()

    def test_pruning_does_not_follow_symlinks_either(self, tmp_path):
        target = _own(tmp_path)
        victim = tmp_path / "victim.txt"
        victim.write_text("keep me")
        for i in range(tools._SPILL_KEEP + 5):
            (target / f"{i:012x}.txt").write_text("spill")
        # Oldest by mtime, so a prune that follows links would unlink it first.
        (target / ("f" * 12 + ".txt")).symlink_to(victim)
        os.utime(target / ("f" * 12 + ".txt"), (0, 0), follow_symlinks = False)

        tools._prune_spills(str(target))

        # os.remove would only unlink the link, so the file it points at is safe either
        # way; what the filter buys is that a name Studio did not write is left alone.
        assert (target / ("f" * 12 + ".txt")).is_symlink()
        assert victim.read_text() == "keep me"


class TestSpillsAreBoundedInBytes:
    def test_one_enormous_result_is_capped_on_disk(self, tmp_path, monkeypatch):
        """The result arrives here whole, and the sandbox file-size limit does not apply to
        output that came through a pipe. Without a byte cap a single `cat` of a huge file
        is retained in full."""
        monkeypatch.setattr(tools, "_SPILL_MAX_BYTES", 4_096)
        text = _dense(50_000)

        out = tools._truncate(text, 200, workdir = str(tmp_path))

        spill = tmp_path / _spill_path(out)
        assert spill.stat().st_size <= 4_096
        # And the notice says so rather than promising the whole thing.
        assert "Full output saved to" not in out
        assert "first 4096 bytes" in out

    def test_the_directory_is_bounded_in_bytes_not_just_in_files(self, tmp_path, monkeypatch):
        """Twenty large-but-legal spills are still tens of gigabytes of the host's disk."""
        monkeypatch.setattr(tools, "_SPILL_MAX_TOTAL_BYTES", 20_000)
        for n in range(10):
            tools._truncate(f"run {n}\n" + _dense(9_000), 200, workdir = str(tmp_path))

        spills = list((tmp_path / tools._SPILL_DIR).iterdir())
        assert sum(p.stat().st_size for p in spills) <= 20_000
        assert spills, "the newest spill has to survive; it is the one just named"

    def test_the_spill_just_named_is_never_pruned(self, tmp_path, monkeypatch):
        """The notice returned with it names that path, so a budget that deletes it on the
        way out leaves the model a hint pointing at nothing."""
        monkeypatch.setattr(tools, "_SPILL_MAX_TOTAL_BYTES", 100)

        out = tools._truncate(_dense(50_000), 200, workdir = str(tmp_path))

        assert (tmp_path / _spill_path(out)).exists()

    def test_a_result_served_whole_leaves_no_spill_behind(self, tmp_path):
        """At zero room a short result is served as it is, so nothing was cut and there is
        nothing to page through. Writing a file (and creating the directory) for it is a
        side effect with nothing on the other side of it."""
        assert tools._truncate("done", 0, workdir = str(tmp_path)) == "done"

        assert not (tmp_path / tools._SPILL_DIR).exists()


class TestTheHintCountsTheLinesItShowed:
    def test_a_head_ending_in_a_newline_does_not_claim_an_extra_line(self, tmp_path):
        """At a limit of 1 on output starting with a blank line the head is "\\n" alone:
        one line shown, so the hint has to resume at line 2. Counting two skips a line the
        reader never saw."""
        text = "\n" + "\n".join(f"line {i}" for i in range(2, 400))

        out = tools._truncate(text, 1, workdir = str(tmp_path))

        assert "showing lines 1-1 of" in out
        assert "sed -n '2," in out


class TestTheSpillCannotBeAimedElsewhere:
    def test_a_hard_link_at_the_spill_path_is_not_written_through(self, tmp_path):
        """The name is a digest of content the model produced, so it can predict it and
        pre-create it. A hard link reports islink() false and shares the inode of a file
        outside the sandbox, so an O_TRUNC open writes through to it."""
        workdir = tmp_path / "sandbox"
        workdir.mkdir()
        victim = tmp_path / "victim.txt"
        victim.write_text("do not overwrite me")
        text = "\n".join(str(i) for i in range(5_000))
        target = _own(workdir)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        os.link(victim, target / f"{digest}.txt")

        out = tools._truncate(text, 200, workdir = str(workdir))

        assert victim.read_text() == "do not overwrite me"
        # And nothing is written at that name either: it is not a spill this recorded, so
        # it is the user's, whatever it is linked to.
        assert "saved to" not in out
        assert (target / f"{digest}.txt").read_text() == "do not overwrite me"

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        tools._truncate("\n".join(str(i) for i in range(5_000)), 200, workdir = str(tmp_path))

        names = [p.name for p in _spills(tmp_path / tools._SPILL_DIR)]
        assert all(tools._SPILL_NAME_RE.fullmatch(n) for n in names), names


class TestAZeroCapStaysZero:
    def test_real_tokenizer_framing_does_not_buy_a_character(self, monkeypatch):
        """A thread at its budget measures a room of zero, and a real tokenizer charges for
        the framing around even an empty string. Handing back one character puts _truncate
        past its stub path and onto the ordinary notice, which is the ~90 tokens the stub
        exists not to spend when the measurement just said there are none."""
        _window(monkeypatch, 4096)
        # Nonzero for an empty probe, which is what a chat template does.
        monkeypatch.setattr(
            tools, "_loaded_token_counter", lambda ctx: (lambda chunk: 4 + len(chunk) // 3)
        )
        _room(0)
        text = _dense(40_000)

        assert tools._dense_char_limit(text, tools._MAX_OUTPUT_CHARS) == 0
        out = tools._truncate(text, tools._tool_result_char_budget())
        assert out.startswith("(output omitted:")
        assert "truncated to" not in out


class TestOneChatsOutputStaysItsOwn:
    """`_get_workdir(None)` is the shared `_default` sandbox, and a project's chats share
    one session by design (`project_session_id`). A spill in either outlives the call under
    a path the next chat can list and read, and can be pruned out from under the chat that
    was told to page through it. Before this change that output existed only in the
    originating response."""

    @staticmethod
    def _spill_args(
        monkeypatch,
        tmp_path,
        session_id,
        thread_id = None,
    ):
        seen = {}
        # A real directory: the command runs with it as cwd, and a path that is not there
        # fails the call long before anything is truncated.
        monkeypatch.setattr(tools, "_get_workdir", lambda _sid: str(tmp_path))
        real = tools._truncate

        def _recording(
            text,
            limit = None,
            workdir = None,
            **kwargs,
        ):
            seen.update(kwargs, workdir = workdir)
            return real(text, limit if limit is not None else 200)

        monkeypatch.setattr(tools, "_truncate", _recording)
        # Builtin printf over a brace expansion: no command substitution, because the
        # sandbox caps processes and a fork fails the call before it ever truncates.
        tools._bash_exec("printf 'x%.0s' {1..5000}", None, 30, session_id, thread_id = thread_id)
        return seen

    def test_a_call_without_a_session_does_not_spill(self, monkeypatch, tmp_path):
        assert self._spill_args(monkeypatch, tmp_path, None)["workdir"] is None

    def test_a_project_chat_without_a_thread_does_not_spill(self, monkeypatch, tmp_path):
        """Nothing identifies the chat, and the session is shared with every other chat in
        the project, so there is nowhere to put it that is only this chat's."""
        session = tools.project_session_id("proj-1")

        assert self._spill_args(monkeypatch, tmp_path, session)["scope"] is None

    def test_a_project_chat_retains_nothing_even_with_a_thread(self, monkeypatch, tmp_path):
        """A sub-directory is not access control, it is a name: every sibling chat in the
        project has a terminal in that same sandbox and can read and prune whatever is in
        it. So a project chat truncates with a notice and no continuation."""
        session = tools.project_session_id("proj-1")

        assert self._spill_args(monkeypatch, tmp_path, session, "chat-a")["scope"] is None
        assert self._spill_args(monkeypatch, tmp_path, session, "chat-b")["scope"] is None

    def test_a_private_session_still_spills(self, monkeypatch, tmp_path):
        """The control: the feature has to keep working where the sandbox is the
        conversation's own."""
        seen = self._spill_args(monkeypatch, tmp_path, "chat-1")

        assert seen["workdir"] == str(tmp_path)
        assert seen["scope"] == ""

    def test_a_scope_puts_the_spill_in_its_own_directory(self, tmp_path):
        """And the notice names that path, so paging still works from the sandbox cwd."""
        text = "\n".join(str(i) for i in range(5_000))

        out = tools._truncate(text, 200, workdir = str(tmp_path), scope = "abc123abc123")

        assert _spill_path(out).startswith(f"{tools._SPILL_DIR}/abc123abc123/")
        assert (tmp_path / _spill_path(out)).read_text().startswith("0\n1\n")

    def test_a_scope_of_none_retains_nothing(self, tmp_path):
        out = tools._truncate(
            "\n".join(str(i) for i in range(5_000)), 200, workdir = str(tmp_path), scope = None
        )

        assert "saved to" not in out
        assert not (tmp_path / tools._SPILL_DIR).exists()


class TestTheRetryHintIsInsideTheCap:
    """`_missing_path_hint` is appended to the result and goes to the model with it, so it
    is part of what has to fit. Added after the cap it is unbudgeted, and a failing
    absolute path is dense text: on a thread with no room left those characters are the
    overflow the cap exists to prevent."""

    _HINT = " " + "x" * 400

    def test_the_hint_is_charged_to_the_limit(self):
        """Paid for out of the body, and at its token cost rather than its length: a
        punctuation-heavy path tokenises far more densely than the prose it displaces."""
        text = "\n".join(str(i) for i in range(5_000))

        capped = tools._truncate(text, 4_000, hint = self._HINT)
        plain = tools._truncate(text, 4_000)

        assert capped.endswith(self._HINT)
        body = capped.split("\n\n... (")[0]
        assert len(body) + len(self._HINT) <= len(plain.split("\n\n... (")[0])

    def test_a_hint_too_large_for_the_room_is_dropped(self):
        """Past half the room the output is worth more than the advice about it."""
        text = "\n".join(str(i) for i in range(5_000))

        capped = tools._truncate(text, 900, hint = self._HINT)

        assert self._HINT not in capped

    def test_a_dense_hint_costs_more_than_its_length(self, monkeypatch):
        """The point of pricing it in tokens. A punctuation-heavy absolute path tokenises
        far more densely than the prose characters that would be dropped to make room for
        it, so subtracting its LENGTH from the character cap buys less than it spends."""
        _window(monkeypatch, 4096)
        # A separator is its own token and a letter is a quarter of one, which is roughly
        # what a tokenizer does to a path next to prose.
        monkeypatch.setattr(
            tools,
            "_loaded_token_counter",
            lambda ctx: (lambda chunk: sum(1.0 if c == "/" else 0.25 for c in chunk)),
        )
        _room(400)
        text = _dense(40_000)
        hint = "/" * 100

        without = tools._truncate(text, tools._MAX_OUTPUT_CHARS).split("\n\n... (")[0]
        with_hint = tools._truncate(text, tools._MAX_OUTPUT_CHARS, hint = hint)

        assert with_hint.endswith(hint)
        body = with_hint.split("\n\n... (")[0]
        # The hint is 100 tokens and the body runs four characters to the token, so the
        # body has to give up about 400 characters to pay for it. Charged as prose it
        # gives up its length, which line rounding can inflate a little: three times over
        # is comfortably past anything that rounding explains.
        assert len(without) - len(body) >= 3 * len(
            hint
        ), "the body gave up about the hint's length, so the hint was charged as prose"

    def test_a_result_that_fits_still_carries_it(self):
        assert tools._truncate("ok", 1_000, hint = self._HINT) == "ok" + self._HINT


class TestTheSafetensorsLoopPricesItToo:
    """The native path runs the same tools against the same window, and has no rolling fit
    behind it: a `cat` of a file the model just wrote lands in the newest exchange with
    nothing downstream able to evict it."""

    @staticmethod
    def _run(
        context_length,
        messages = None,
        calls = 1,
    ):
        """One `terminal` call through the real loop, returning the kwargs it was given."""
        import threading

        from core.inference.safetensors_agentic import run_safetensors_tool_loop

        def _model(messages):
            state["turns"] += 1
            if state["turns"] > 1:
                yield "done"
                return
            call = "".join(
                '<tool_call>{"name":"terminal","arguments":{"command":"cat game%d.html"}}'
                "</tool_call>" % n
                for n in range(calls)
            )
            for n in range(1, len(call) + 1):
                yield call[:n]

        state = {"turns": 0}
        seen = []
        list(
            run_safetensors_tool_loop(
                single_turn = _model,
                messages = list(messages or [{"role": "user", "content": "print game.html"}]),
                tools = [{"type": "function", "function": {"name": "terminal"}}],
                execute_tool = lambda name, args, **kwargs: seen.append(kwargs) or "x" * 40_000,
                cancel_event = threading.Event(),
                max_tool_iterations = 2,
                thread_id = "t-sf",
                context_length = context_length,
                max_tokens = 512,
            )
        )
        return seen[0]

    def test_it_prices_the_room_for_the_tool(self):
        """Without it execute_tool takes the None default and every cap is disabled here."""
        room = self._run(4096).get("result_budget_tokens")

        assert isinstance(room, int)
        assert 0 < room < prompt_budget(4096, 512)

    def test_it_passes_the_window_as_well_as_the_room(self):
        """A room with no window to size against is not a cap: `_dense_char_limit` has no
        share to take and hands back the window-independent constant."""
        assert self._run(4096).get("context_tokens") == 4096

    def test_a_result_already_in_the_thread_is_priced_densely(self):
        """The estimator charges ASCII four characters per token, an English rate, and the
        results these tools return run nearer two. Priced at the English rate the second
        call is handed room the first call already occupies, and this loop has no exact
        count and no rolling fit to catch it.

        Differential, so it cannot pass on the arithmetic alone: the same characters in a
        user turn are ordinary prose and stay at the estimator's rate, while in a tool
        result they are charged twice.
        """
        body = "x" * 4_000
        as_result = self._run(
            4096,
            messages = [
                {"role": "user", "content": "print it"},
                {"role": "tool", "content": body},
            ],
        )["result_budget_tokens"]
        as_prose = self._run(
            4096,
            messages = [
                {"role": "user", "content": "print it"},
                {"role": "user", "content": body},
            ],
        )["result_budget_tokens"]

        assert as_result < as_prose

    def test_a_parallel_batch_splits_the_room(self):
        """One turn can call several tools, and each call is appended only as it runs, so
        the spend knows nothing about the rest of the batch. Sized as if it were alone, the
        first result takes the room the other calls and their results still need, and the
        finished exchange is protected as the newest turn."""
        alone = self._run(4096)["result_budget_tokens"]
        first_of_three = self._run(4096, calls = 3)["result_budget_tokens"]

        assert first_of_three <= alone // 3

    def test_every_result_in_a_batch_gets_its_own_notice_reserve(self):
        """Each truncated result carries its own notice, so an N-call batch spends N of
        them. Reserved once and divided, the batch pays for one and spends N."""
        from core.inference.context_window import _RESULT_NOTICE_RESERVE

        assert (
            tool_result_budget(4096, 512, 500) - tool_result_budget(4096, 512, 500, results = 3)
            == 2 * _RESULT_NOTICE_RESERVE
        )
        # And the loop asks for one per call rather than one per turn. Measured against
        # its own single-call room, since the spend is whatever that thread costs.
        alone = self._run(4096)["result_budget_tokens"]
        first_of_three = self._run(4096, calls = 3)["result_budget_tokens"]

        assert first_of_three <= (alone - 2 * _RESULT_NOTICE_RESERVE) // 3

    def test_an_unknown_window_prices_nothing(self):
        """The same leg the GGUF loop takes: no window, no budget, today's behaviour."""
        assert self._run(None).get("result_budget_tokens") is None


class TestAProjectIsBoundedAsOneWorkspace:
    def test_the_byte_budget_spans_every_scope(self, tmp_path, monkeypatch):
        """A project's chats share one sandbox and get one scope each, so a per-directory
        budget is really a per-chat budget multiplied by however many chats there are."""
        monkeypatch.setattr(tools, "_SPILL_MAX_TOTAL_BYTES", 30_000)
        for chat in range(6):
            tools._truncate(
                f"chat {chat}\n" + _dense(9_000), 200, workdir = str(tmp_path), scope = f"{chat:012x}"
            )

        root = tmp_path / tools._SPILL_DIR
        spilled = [p for p in root.rglob("*.txt")]
        assert sum(p.stat().st_size for p in spilled) <= 30_000
        assert spilled, "the newest spill has to survive; it is the one just named"

    def test_an_emptied_scope_leaves_no_directory_behind(self, tmp_path, monkeypatch):
        """Deleting a chat does not remove its scope, and an empty directory left in the
        sandbox is what makes the cleanup treat it as holding something."""
        monkeypatch.setattr(tools, "_SPILL_MAX_TOTAL_BYTES", 12_000)
        for chat in range(4):
            tools._truncate(
                f"chat {chat}\n" + _dense(9_000), 200, workdir = str(tmp_path), scope = f"{chat:012x}"
            )

        root = tmp_path / tools._SPILL_DIR
        scopes = [p for p in root.iterdir() if p.is_dir()]
        # The budget holds two of the four, so the other two were emptied by the prune and
        # then removed rather than left standing.
        assert len(scopes) < 4
        assert all(any(scope.iterdir()) for scope in scopes)


class TestDeletingAChatIsNotBlockedByItsSpills:
    def test_a_sandbox_holding_only_spills_is_still_removable(self, tmp_path):
        """Spills are Studio's own, written by this process and deliberately kept off the
        file cards. Counted as the user's content they leave an unreachable sandbox behind,
        reported as holding files the user never created."""
        tools._truncate(
            "\n".join(str(i) for i in range(5_000)),
            200,
            workdir = str(tmp_path),
            scope = "abc123abc123",
        )

        assert tools._holds_no_user_files(str(tmp_path))

    def test_a_real_file_beside_them_still_counts(self):
        """The control: this must not turn into "delete any sandbox"."""
        import tempfile
        with tempfile.TemporaryDirectory() as workdir:
            tools._truncate(
                "\n".join(str(i) for i in range(5_000)), 200, workdir = workdir, scope = "abc123abc123"
            )
            open(os.path.join(workdir, "game.html"), "w").close()

            assert not tools._holds_no_user_files(workdir)

    def test_a_user_file_inside_the_spill_directory_still_counts(self, tmp_path):
        """The directory is writable and the tools can create anything in it. Skipping the
        whole tree means deleting a chat deletes a file the user's own code wrote."""
        tools._truncate(
            "\n".join(str(i) for i in range(5_000)),
            200,
            workdir = str(tmp_path),
            scope = "abc123abc123",
        )
        (tmp_path / tools._SPILL_DIR / "notes.txt").write_text("mine")

        assert not tools._holds_no_user_files(str(tmp_path))

    def test_a_file_named_like_a_spill_but_placed_elsewhere_counts(self, tmp_path):
        """The name alone is not the test: twelve hex characters is a plausible name for
        anything, and outside the spill root nothing here wrote it."""
        (tmp_path / "abcdef123456.txt").write_text("mine")

        assert not tools._holds_no_user_files(str(tmp_path))


class TestTheNativePathIsBoundedWithoutATokenizer:
    """`_loaded_token_counter` answers only for a resident GGUF, so on a safetensors model
    the room is converted with the four-characters-per-token English estimate and never
    corrected. Dense ASCII runs nearer two, and that loop has no rolling fit to recover."""

    def test_an_unmeasurable_room_is_halved(self, monkeypatch):
        _window(monkeypatch, 4096)
        monkeypatch.setattr(tools, "_loaded_token_counter", lambda ctx: None)
        _room(400)
        text = _dense(40_000)

        assert tools._dense_char_limit(text, tools._MAX_OUTPUT_CHARS) == pytest.approx(
            400 * 4 * tools._UNMEASURED_ROOM_MARGIN, rel = 0.02
        )

    def test_a_measurable_room_is_not(self, monkeypatch):
        """The control: where the serving model can price the string, the exact fit does
        the work and no blanket margin is applied on top of it."""
        _window(monkeypatch, 4096)
        _tokenizer(monkeypatch)
        _room(400)
        text = _dense(40_000)

        limit = tools._dense_char_limit(text, tools._MAX_OUTPUT_CHARS)
        assert limit > 400 * 4 * tools._UNMEASURED_ROOM_MARGIN
        assert limit // _CHARS_PER_TOKEN <= 400


class TestOwnershipIsNotKeptWhereToolCodeCanWriteIt:
    """The sandbox is a directory the model runs commands in, so nothing kept inside it is
    evidence about it. A marker file there can be replaced with a link, and once it is a
    plain file its contents can be rewritten to name the user's own files as Studio's,
    which turns the cleanup into a delete. The record lives in Studio's own storage."""

    def test_nothing_about_ownership_is_written_into_the_sandbox(self, tmp_path):
        out = tools._truncate("\n".join(str(i) for i in range(5_000)), 200, workdir = str(tmp_path))

        names = [p.name for p in (tmp_path / tools._SPILL_DIR).iterdir()]
        assert names == [os.path.basename(_spill_path(out))]

    def test_a_directory_that_came_with_the_sandbox_is_never_adopted(self, tmp_path):
        (tmp_path / tools._SPILL_DIR).mkdir()
        (tmp_path / tools._SPILL_DIR / "notes.txt").write_text("mine")

        out = tools._truncate("\n".join(str(i) for i in range(5_000)), 200, workdir = str(tmp_path))

        assert "saved to" not in out
        assert not tools._holds_no_user_files(str(tmp_path))

    def test_a_replaced_directory_loses_the_record(self, tmp_path):
        """Tool code can delete `.unsloth_tool_output` and make its own in the same place.
        Same path, different directory, and a record that only knew the path would hand
        the new one's contents to the prune."""
        spilled = tools._truncate(
            "\n".join(str(i) for i in range(5_000)), 200, workdir = str(tmp_path)
        )
        name = os.path.basename(_spill_path(spilled))
        root = tmp_path / tools._SPILL_DIR
        shutil.rmtree(root)
        root.mkdir()
        # The same NAME the record already knows, which is what makes the path alone
        # insufficient: this file is the user's and the record was written about another
        # directory that no longer exists.
        theirs = root / name
        theirs.write_text("mine")
        for n in range(tools._SPILL_KEEP + 5):
            tools._truncate(f"run {n}\n" + _dense(3_000), 200, workdir = str(tmp_path))

        assert theirs.read_text() == "mine"
        assert not tools._is_spill_artifact(str(tmp_path), str(root), name)

    def test_a_spill_written_over_stops_being_ours(self, tmp_path):
        """A recorded path is not the file. Tool code can write its own content over a
        spill, in place, and from then on it is the user's: not something to prune, and
        not something the cleanup may delete the sandbox on top of."""
        spilled = tools._truncate(
            "\n".join(str(i) for i in range(5_000)), 200, workdir = str(tmp_path)
        )
        theirs = tmp_path / _spill_path(spilled)
        theirs.write_text("the user's own data")

        for n in range(tools._SPILL_KEEP + 5):
            tools._truncate(f"run {n}\n" + _dense(3_000), 200, workdir = str(tmp_path))

        assert theirs.read_text() == "the user's own data"
        assert not tools._holds_no_user_files(str(tmp_path))

    def test_an_untouched_spill_is_still_ours(self, tmp_path):
        """The control: reading a spill back, which is the whole point of writing it, must
        not turn it into user content."""
        spilled = tools._truncate(
            "\n".join(str(i) for i in range(5_000)), 200, workdir = str(tmp_path)
        )
        (tmp_path / _spill_path(spilled)).read_text()

        assert tools._holds_no_user_files(str(tmp_path))


class TestConcurrentSpillsKeepTheirRecords:
    def test_a_spill_recorded_during_a_prune_is_not_dropped(self, tmp_path):
        """A project's chats share one sandbox. Appending a spill and rewriting the
        manifest after a prune are a read-modify-write over one file, and a pruner that
        read it before another call appended would discard the newer entry, leaving a
        file nothing counts, prunes, or recognises as Studio's."""
        import threading

        workdir = str(tmp_path)
        started = threading.Barrier(8)
        results = []

        def _spill(n):
            started.wait()
            results.append(
                tools._truncate(
                    f"chat {n}\n" + _dense(3_000), 200, workdir = workdir, scope = f"{n:012x}"
                )
            )

        threads = [threading.Thread(target = _spill, args = (n,)) for n in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        root = tmp_path / tools._SPILL_DIR
        recorded = set(tools._spill_manifest(str(root)))
        on_disk = {str(p.relative_to(root)) for p in root.rglob("*.txt") if p.name != "victim.txt"}
        assert on_disk, results[:1]
        assert on_disk <= recorded, "a spill on disk that the manifest does not record"

    def test_a_spill_restored_to_its_old_mtime_is_still_not_ours(self, tmp_path):
        """mtime alone is forgeable: same-sized content and `os.utime` put it back, and a
        coarse-grained filesystem can leave it unchanged without any help. ctime moves on
        any write and cannot be set from userspace, and the content is checked besides."""
        spilled = tools._truncate(
            "\n".join(str(i) for i in range(5_000)), 200, workdir = str(tmp_path)
        )
        theirs = tmp_path / _spill_path(spilled)
        was = theirs.stat()
        theirs.write_bytes(b"m" * was.st_size)
        os.utime(theirs, ns = (was.st_atime_ns, was.st_mtime_ns))

        assert theirs.stat().st_size == was.st_size
        assert theirs.stat().st_mtime_ns == was.st_mtime_ns
        assert not tools._holds_no_user_files(str(tmp_path))

    @pytest.mark.skipif(
        not tools._DIR_FD_WRITES, reason = "no dir_fd support; the path-based write stands"
    )
    def test_the_spill_write_never_follows_a_swapped_directory(self, tmp_path):
        """A shared project sandbox can lose the race between checking the directory and
        writing into it by name: another call can put a symlink there in between, and a
        path-based create and rename both follow it.

        Checked on the writer itself, since by construction the swap happens after every
        check the caller makes.
        """
        outside = tmp_path / "outside"
        outside.mkdir()
        swapped = tmp_path / "scope"
        swapped.symlink_to(outside, target_is_directory = True)

        assert tools._write_spill_file(str(swapped), "abcdef123456.txt", "x" * 100) is None
        assert list(outside.iterdir()) == [], "the spill was written outside the sandbox"

    def test_the_writer_still_works_on_a_real_directory(self, tmp_path):
        """The control, so the refusal above is not simply "never writes"."""
        assert tools._write_spill_file(str(tmp_path), "abcdef123456.txt", "x" * 100)

        assert (tmp_path / "abcdef123456.txt").read_text() == "x" * 100


class TestARemovedSandboxTakesItsRecordWithIt:
    """Session workdirs are unique, so a record left behind is one small file per deleted
    chat, for the life of the installation."""

    @staticmethod
    def _sandbox(tmp_path, monkeypatch, session):
        monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))
        tools._workdirs.clear()
        workdir = tools.get_sandbox_workdir(session)
        tools._truncate("\n".join(str(i) for i in range(5_000)), 200, workdir = workdir)
        record = tools._spill_record_path(os.path.join(workdir, tools._SPILL_DIR))
        assert os.path.exists(record)
        return workdir, record

    def test_an_empty_sandbox_takes_its_record(self, tmp_path, monkeypatch):
        """Only spills are left in it, so the sandbox goes without opting into files."""
        workdir, record = self._sandbox(tmp_path, monkeypatch, "__LOCALID_spill111")

        assert tools.remove_session_sandbox("__LOCALID_spill111") is True

        assert not os.path.exists(workdir)
        assert not os.path.exists(record)

    def test_a_sandbox_deleted_with_its_files_does_too(self, tmp_path, monkeypatch):
        workdir, record = self._sandbox(tmp_path, monkeypatch, "__LOCALID_spill222")
        open(os.path.join(workdir, "game.html"), "w").close()

        assert tools.remove_session_sandbox("__LOCALID_spill222", delete_files = True) is True

        assert not os.path.exists(record)

    def test_the_fallback_deletion_takes_it_too(self, tmp_path, monkeypatch):
        """The rename can fail (a cross-device sandbox root, a locked tree on Windows), and
        the rmtree that stands in for it deletes just as much, so it has to forget just as
        much."""
        workdir, record = self._sandbox(tmp_path, monkeypatch, "__LOCALID_spill444")

        def _no_rename(*args, **kwargs):
            raise OSError("rename unavailable")

        monkeypatch.setattr(tools.os, "rename", _no_rename)

        assert tools.remove_session_sandbox("__LOCALID_spill444", delete_files = True) is True

        assert not os.path.exists(workdir)
        assert not os.path.exists(record)

    def test_a_sandbox_that_stays_keeps_its_record(self, tmp_path, monkeypatch):
        """The control: the files are the user's, the sandbox stays, and so does the
        record of what in it is Studio's."""
        workdir, record = self._sandbox(tmp_path, monkeypatch, "__LOCALID_spill333")
        open(os.path.join(workdir, "game.html"), "w").close()

        assert tools.remove_session_sandbox("__LOCALID_spill333") is False

        assert os.path.exists(record)

    def test_rerunning_a_command_does_not_overwrite_replaced_content(self, tmp_path):
        """The name comes from the content, so running the same command again lands on the
        same path. If the user's code put its own data there in between, the rename would
        replace it: the manifest already knows it stopped being ours, and the write has to
        ask."""
        text = "\n".join(str(i) for i in range(5_000))
        spilled = tools._truncate(text, 200, workdir = str(tmp_path))
        theirs = tmp_path / _spill_path(spilled)
        theirs.write_text("the user's own data")

        again = tools._truncate(text, 200, workdir = str(tmp_path))

        assert theirs.read_text() == "the user's own data"
        assert "saved to" not in again

    def test_the_same_output_twice_still_reuses_its_own_spill(self, tmp_path):
        """The control: an untouched spill is still ours, so the repeat case this whole
        change is about keeps working instead of refusing on its own file."""
        text = "\n".join(str(i) for i in range(5_000))

        first = tools._truncate(text, 200, workdir = str(tmp_path))
        second = tools._truncate(text, 200, workdir = str(tmp_path))

        assert first == second
        assert len(_spills(tmp_path / tools._SPILL_DIR)) == 1

    def test_a_first_spill_is_not_disowned_by_a_racing_one(self, tmp_path):
        """Two first-time spills in one sandbox can both see the directory absent. The
        slower one must not write an empty record over the winner's: that would leave the
        winner's spill owned by nobody, never pruned and counted as the user's content."""
        import threading

        ready = threading.Barrier(6)
        outs = []

        def _spill(n):
            ready.wait()
            outs.append(tools._truncate(f"chat {n}\n" + _dense(3_000), 200, workdir = str(tmp_path)))

        threads = [threading.Thread(target = _spill, args = (n,)) for n in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        root = tmp_path / tools._SPILL_DIR
        recorded = set(tools._spill_manifest(str(root)))
        on_disk = {p.name for p in root.iterdir()}
        assert on_disk and on_disk <= recorded, "a spill on disk that the record disowned"


class TestTheContinuationChunkDecodes:
    """`head -c` counts bytes. A round number of them lands inside a code point on any
    mixed-width text, and the terminal runner decodes with errors="replace", so the
    continuation the notice advertises would hand back a mangled character every time."""

    @staticmethod
    def _resume(out: str) -> "tuple[int, int]":
        """The byte offset and chunk size out of a `tail -c +N | head -c M` hint."""
        tail, _, head = out.rsplit("tail -c +", 1)[1].partition(" | head -c ")
        return int(tail.split()[0]), int(head.rstrip(")").strip())

    def test_the_chunk_ends_on_a_character_boundary(self, tmp_path):
        # One long line, so the cut is mid-line and the hint is byte-based. 299 ASCII
        # characters and then three-byte ones: the head is 302 bytes, and 302 bytes of
        # what follows is 100 characters plus two bytes of the next.
        text = "a" * 299 + "€" * 4_000

        out = tools._truncate(text, 300, workdir = str(tmp_path))
        offset, span = self._resume(out)

        blob = (tmp_path / _spill_path(out)).read_bytes()
        chunk = blob[offset - 1 : offset - 1 + span]
        # Strict, because errors="replace" is exactly what hides this.
        assert chunk.decode("utf-8")

    def test_the_chunk_resumes_where_the_head_stopped(self, tmp_path):
        text = "a" * 299 + "€" * 4_000

        out = tools._truncate(text, 300, workdir = str(tmp_path))
        offset, span = self._resume(out)

        shown = out.split("\n\n... (")[0]
        blob = (tmp_path / _spill_path(out)).read_bytes()
        assert (
            blob[offset - 1 : offset - 1 + span].decode("utf-8")
            == text[len(shown) : len(shown) + len(shown)]
        )


class TestInstallingASpillNeverReplacesAnything:
    """The caller checks the destination, then the write takes as long as it takes. In
    that window another call sharing the workspace can put a file at the name, and a
    rename would replace it silently: POSIX rename overwrites, link does not."""

    def test_a_name_taken_during_the_write_is_not_overwritten(self, tmp_path):
        theirs = tmp_path / "abcdef123456.txt"
        theirs.write_text("the user's own data")

        assert tools._write_spill_file(str(tmp_path), theirs.name, "x" * 100) is None

        assert theirs.read_text() == "the user's own data"

    def test_no_temporary_is_left_when_the_name_is_taken(self, tmp_path):
        (tmp_path / "abcdef123456.txt").write_text("mine")

        tools._write_spill_file(str(tmp_path), "abcdef123456.txt", "x" * 100)

        assert [p.name for p in tmp_path.iterdir()] == ["abcdef123456.txt"]

    def test_an_unchanged_spill_is_reused_without_writing(self, tmp_path, monkeypatch):
        """And the repeat case does not go near the install at all: the name is the digest
        of the text, so a recorded spill at it already holds exactly this content."""
        text = "\n".join(str(i) for i in range(5_000))
        first = tools._truncate(text, 200, workdir = str(tmp_path))

        def _refuse(*args, **kwargs):
            raise AssertionError("rewrote a spill that was already there")

        monkeypatch.setattr(tools, "_write_spill_file", _refuse)

        assert tools._truncate(text, 200, workdir = str(tmp_path)) == first

    def test_the_record_names_what_was_installed_not_what_is_there_now(self, tmp_path):
        """Between the install and the record, another call sharing the sandbox can replace
        the file. Stating the path then records THAT content as Studio's, and a later prune
        or cleanup deletes the user's data on the strength of it."""
        root = _own(tmp_path)
        name = "abcdef123456.txt"
        stamp = tools._write_spill_file(str(root), name, "the spill")
        # The replacement, in the window the record must not read across.
        (root / name).write_text("the user's own data")

        tools._record_spill(str(root), name, stamp, hashlib.sha256(b"the spill").hexdigest())

        assert not tools._is_recorded_spill(
            str(root), str(root / name), tools._spill_manifest(str(root))
        )
        assert not tools._holds_no_user_files(str(tmp_path))


class TestPruningDeletesOnlyWhatItChecked:
    """The manifest lock orders Studio's own threads; the thing racing here is the sandbox.
    A background process can replace a recorded spill between the check and the unlink, and
    a delete by name then takes the replacement."""

    @staticmethod
    def _swap_after_check(monkeypatch, victim, content):
        """Replace the file after the prune has verified it and before it deletes it.

        Hooked on the sort that runs between the two, since a swap during the verification
        is caught by the verification itself and proves nothing about the window after it.
        """
        real = os.path.getmtime

        def _getmtime(path):
            result = real(path)
            if os.fspath(path) == victim:
                open(victim, "w").write(content)
            return result

        monkeypatch.setattr(os.path, "getmtime", _getmtime)

    def test_a_file_swapped_after_the_check_is_not_deleted(self, tmp_path, monkeypatch):
        # Enough that the oldest are on their way out: a prune with nothing to delete
        # would prove nothing about what it deletes.
        for n in range(tools._SPILL_KEEP + 5):
            tools._truncate(f"run {n}\n" + _dense(3_000), 200, workdir = str(tmp_path))
        root = tmp_path / tools._SPILL_DIR
        victim = sorted(_spills(root), key = lambda p: p.stat().st_mtime)[0]
        # Lowered so this pass has files to delete: the prune after each spill has already
        # brought the directory back to the limit.
        monkeypatch.setattr(tools, "_SPILL_KEEP", 5)
        self._swap_after_check(monkeypatch, str(victim), "the user's own data")

        tools._prune_spills(str(root))

        assert victim.exists(), "the prune deleted a file it had not verified"
        assert victim.read_text() == "the user's own data"

    def test_an_untouched_spill_is_still_pruned(self, tmp_path):
        """The control: the budget still has to bite, or the fix above is just "never
        delete anything"."""
        for n in range(tools._SPILL_KEEP + 5):
            tools._truncate(f"run {n}\n" + _dense(3_000), 200, workdir = str(tmp_path))

        assert len(_spills(tmp_path / tools._SPILL_DIR)) <= tools._SPILL_KEEP

    def test_nothing_is_left_under_a_temporary_name(self, tmp_path):
        for n in range(tools._SPILL_KEEP + 5):
            tools._truncate(f"run {n}\n" + _dense(3_000), 200, workdir = str(tmp_path))

        names = [p.name for p in _spills(tmp_path / tools._SPILL_DIR)]
        assert all(tools._SPILL_NAME_RE.fullmatch(n) for n in names), names


class TestReadingASpillCannotBeRedirected:
    """The stamp is taken a moment before the content is read, and this runs in the
    sandbox's own directory. A plain open by name would follow a symlink installed in
    between and block forever on a FIFO or a device, with no timeout above it."""

    def test_a_symlink_at_the_name_is_not_followed(self, tmp_path):
        elsewhere = tmp_path / "elsewhere.txt"
        elsewhere.write_text("someone else's file")
        link = tmp_path / "abcdef123456.txt"
        link.symlink_to(elsewhere)

        assert tools._file_digest(str(link)) is None

    def test_a_fifo_is_refused_rather_than_read(self, tmp_path):
        fifo = tmp_path / "abcdef123456.txt"
        os.mkfifo(fifo)

        # Would block forever on a plain open, so a hang here is the failure.
        assert tools._file_digest(str(fifo)) is None

    def test_a_file_of_the_wrong_size_is_refused_before_it_is_read(self, tmp_path):
        real = tmp_path / "abcdef123456.txt"
        real.write_text("x" * 100)

        assert tools._file_digest(str(real), 100)
        assert tools._file_digest(str(real), 101) is None

    def test_an_ordinary_spill_still_reads(self, tmp_path):
        """The control: everything above has to leave the normal case working."""
        spilled = tools._truncate(
            "\n".join(str(i) for i in range(5_000)), 200, workdir = str(tmp_path)
        )

        assert tools._holds_no_user_files(str(tmp_path)), spilled
