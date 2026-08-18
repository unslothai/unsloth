# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the edit_file tool in core/inference/tools.py.

edit_file exists so a one-line change costs a one-line tool call, instead of a
whole-file `cat > f <<'EOF'` or open(...).write(...) that re-sends the file and
drops whatever the model failed to retype.

Pinned here: a miss or an ambiguous match fails loudly and writes nothing, and
the file's encoding, line endings and mode survive an edit.
"""

import os
import stat
import sys
import time
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import tools
from core.inference.tools import (
    ALL_TOOLS,
    EDIT_FILE_TOOL,
    EDIT_FILE_TOOL_FULL_ACCESS,
    apply_full_access_tool_descriptions,
    execute_tool,
    is_potentially_unsafe_tool_call,
)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Point the session workdir at a tmp dir, as the executor would."""
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(tmp_path))
    return tmp_path


def _edit(**arguments) -> str:
    return execute_tool("edit_file", arguments, session_id = "t")


class TestReplacement:
    def test_a_unique_match_is_replaced(self, workdir):
        target = workdir / "a.py"
        target.write_text("def a():\n    return 1\n")
        result = _edit(path = "a.py", old_string = "return 1", new_string = "return 42")
        assert target.read_text() == "def a():\n    return 42\n"
        assert "1 replacement" in result

    def test_the_receipt_shows_the_change_not_the_file(self, workdir):
        # The receipt must stay proportional to the edit, not the file.
        target = workdir / "big.py"
        target.write_text("filler = 0\n" * 500 + "TARGET = 1\n")
        result = _edit(path = "big.py", old_string = "TARGET = 1", new_string = "TARGET = 2")
        assert "TARGET = 2" in result
        assert len(result) < 400
        assert result.count("filler = 0") <= 2  # diff context lines only

    def test_a_missing_old_string_writes_nothing(self, workdir):
        target = workdir / "a.py"
        target.write_text("x = 1\n")
        result = _edit(path = "a.py", old_string = "y = 2", new_string = "y = 3")
        assert result.startswith("Error:")
        assert target.read_text() == "x = 1\n"

    def test_an_ambiguous_match_names_the_count_and_writes_nothing(self, workdir):
        target = workdir / "a.py"
        target.write_text("v = 1\nv = 1\nv = 1\n")
        result = _edit(path = "a.py", old_string = "v = 1", new_string = "v = 2")
        assert result.startswith("Error:")
        assert "3" in result  # the model needs the count to decide what to do
        assert target.read_text() == "v = 1\nv = 1\nv = 1\n"

    def test_replace_all_takes_every_occurrence(self, workdir):
        target = workdir / "a.py"
        target.write_text("v = 1\nv = 1\n")
        result = _edit(path = "a.py", old_string = "v = 1", new_string = "v = 2", replace_all = True)
        assert target.read_text() == "v = 2\nv = 2\n"
        assert "2 replacements" in result

    def test_only_the_first_match_changes_without_replace_all(self, workdir):
        # A unique-match rule that silently edited all of them would corrupt
        # files whenever the model's snippet turned out not to be unique.
        target = workdir / "a.py"
        target.write_text("head\nv = 1\nmid\nv = 1\ntail\n")
        _edit(
            path = "a.py",
            old_string = "head\nv = 1",
            new_string = "head\nv = 9",
        )
        assert target.read_text() == "head\nv = 9\nmid\nv = 1\ntail\n"

    def test_an_identical_edit_is_refused(self, workdir):
        target = workdir / "a.py"
        target.write_text("x = 1\n")
        assert _edit(path = "a.py", old_string = "x", new_string = "x").startswith("Error:")

    def test_non_string_arguments_are_refused(self, workdir):
        # str(None) would write the literal "None" into a source file.
        (workdir / "a.py").write_text("x = 1\n")
        assert _edit(path = "a.py", old_string = None, new_string = "y").startswith("Error:")
        assert _edit(path = "a.py", old_string = "x", new_string = 3).startswith("Error:")


class TestCreation:
    def test_an_empty_old_string_creates_the_file(self, workdir):
        result = _edit(path = "new.py", old_string = "", new_string = "x = 1\n")
        assert (workdir / "new.py").read_text() == "x = 1\n"
        assert result.startswith("Created")

    def test_creation_never_clobbers_an_existing_file(self, workdir):
        target = workdir / "a.py"
        target.write_text("keep me\n")
        result = _edit(path = "a.py", old_string = "", new_string = "gone")
        assert result.startswith("Error:")
        assert target.read_text() == "keep me\n"

    def test_editing_a_missing_file_says_how_to_create_it(self, workdir):
        result = _edit(path = "nope.py", old_string = "a", new_string = "b")
        assert result.startswith("Error:")
        assert "old_string" in result


class TestFileShapeSurvives:
    def test_crlf_endings_are_matched_and_preserved(self, workdir):
        # A model copies old_string out of `cat` output with plain newlines, so
        # matching raw bytes would fail every edit of a CRLF file, and writing
        # back LF would rewrite every line of one it did match.
        target = workdir / "a.txt"
        target.write_bytes(b"one\r\ntwo\r\nthree\r\n")
        result = _edit(path = "a.txt", old_string = "two", new_string = "TWO")
        assert not result.startswith("Error:")
        assert target.read_bytes() == b"one\r\nTWO\r\nthree\r\n"

    def test_a_utf8_bom_is_preserved(self, workdir):
        target = workdir / "a.txt"
        target.write_bytes(b"\xef\xbb\xbfhello world\n")
        _edit(path = "a.txt", old_string = "world", new_string = "there")
        assert target.read_bytes() == b"\xef\xbb\xbfhello there\n"

    def test_unicode_content_survives(self, workdir):
        target = workdir / "a.txt"
        target.write_text("こんにちは世界\n", encoding = "utf-8")
        _edit(path = "a.txt", old_string = "世界", new_string = "みなさん")
        assert target.read_text(encoding = "utf-8") == "こんにちはみなさん\n"

    @pytest.mark.skipif(sys.platform == "win32", reason = "POSIX file mode")
    def test_the_executable_bit_is_preserved(self, workdir):
        target = workdir / "run.sh"
        target.write_text("#!/bin/sh\necho hi\n")
        os.chmod(target, 0o755)
        _edit(path = "run.sh", old_string = "echo hi", new_string = "echo bye")
        assert stat.S_IMODE(os.stat(target).st_mode) == 0o755

    def test_a_binary_file_is_refused(self, workdir):
        target = workdir / "blob.bin"
        target.write_bytes(b"\x00\x01\x02binary")
        assert _edit(path = "blob.bin", old_string = "binary", new_string = "x").startswith("Error:")
        assert target.read_bytes() == b"\x00\x01\x02binary"


class TestPathContainment:
    def test_a_traversal_path_is_refused(self, workdir):
        outside = workdir.parent / "outside.txt"
        outside.write_text("secret\n")
        result = _edit(path = "../outside.txt", old_string = "secret", new_string = "pwned")
        assert result.startswith("Error:")
        assert outside.read_text() == "secret\n"

    @pytest.mark.skipif(sys.platform == "win32", reason = "POSIX symlinks")
    def test_a_symlink_out_of_the_workdir_is_refused(self, workdir):
        # Containment is checked on the realpath, so a link planted inside the
        # sandbox cannot be used to reach through it.
        outside = workdir.parent / "outside.txt"
        outside.write_text("secret\n")
        os.symlink(outside, workdir / "link.txt")
        result = _edit(path = "link.txt", old_string = "secret", new_string = "pwned")
        assert result.startswith("Error:")
        assert outside.read_text() == "secret\n"

    def test_a_code_interpreter_habit_path_keeps_its_suffix(self, workdir):
        # Same rewrite the python shim applies, so a path that works in one
        # tool works in the other.
        result = _edit(path = "/mnt/data/out.txt", old_string = "", new_string = "hi\n")
        assert not result.startswith("Error:")
        assert (workdir / "out.txt").read_text() == "hi\n"

    def test_an_empty_path_is_refused(self, workdir):
        assert _edit(path = "   ", old_string = "a", new_string = "b").startswith("Error:")


class TestReviewFindings:
    """Cases raised in review on the PR that added this tool."""

    def test_a_long_line_does_not_blow_up_the_receipt(self, workdir):
        # Minified JS or single-line JSON: capping diff LINES bounds nothing,
        # since one line can be the whole file. Before the cap a 200KB file
        # returned a 400KB receipt, i.e. twice what the tool set out to avoid.
        target = workdir / "min.js"
        target.write_text("var a=" + "x" * 200_000 + ";")
        result = _edit(path = "min.js", old_string = "var a=", new_string = "var b=")
        assert not result.startswith("Error:")
        assert len(result) < 2000

    def test_replace_all_as_the_string_false_does_not_replace_all(self, workdir):
        # bool("false") is True, and models emit the JSON string. Coercing it
        # turned the multi-match guard off and rewrote every occurrence.
        target = workdir / "a.txt"
        target.write_text("a\na\na\n")
        result = _edit(path = "a.txt", old_string = "a", new_string = "b", replace_all = "false")
        assert result.startswith("Error:")
        assert target.read_text() == "a\na\na\n"

    def test_replace_all_as_the_string_true_still_works(self, workdir):
        target = workdir / "a.txt"
        target.write_text("a\na\n")
        _edit(path = "a.txt", old_string = "a", new_string = "b", replace_all = "true")
        assert target.read_text() == "b\nb\n"

    def test_an_unreadable_replace_all_is_refused(self, workdir):
        target = workdir / "a.txt"
        target.write_text("a\n")
        result = _edit(path = "a.txt", old_string = "a", new_string = "b", replace_all = "maybe")
        assert result.startswith("Error:")
        assert target.read_text() == "a\n"

    @pytest.mark.skipif(sys.platform == "win32", reason = "POSIX FIFO")
    def test_a_fifo_is_refused_rather_than_read(self, workdir):
        # read() on a FIFO blocks forever, and this path carries no timeout or
        # cancel event, so the turn could not be recovered.
        os.mkfifo(workdir / "pipe")
        assert _edit(path = "pipe", old_string = "a", new_string = "b").startswith("Error:")

    def test_an_absolute_path_inside_a_workdir_under_a_habit_prefix(self, workdir, monkeypatch):
        # A project rooted at /workspace/repo had its own prefix stripped and
        # rejoined onto itself, resolving to /workspace/repo/repo/a.py.
        monkeypatch.setattr(tools, "_MISSING_PATH_PREFIXES", (str(workdir.parent), "/mnt/data"))
        (workdir / "a.py").write_text("x = 1\n")
        result = _edit(path = str(workdir / "a.py"), old_string = "x = 1", new_string = "x = 2")
        assert not result.startswith("Error:")
        assert (workdir / "a.py").read_text() == "x = 2\n"

    def test_a_habit_path_outside_the_workdir_still_remaps(self, workdir):
        # The fix above must not switch off the remap it narrows.
        result = _edit(path = "/mnt/data/out.txt", old_string = "", new_string = "hi\n")
        assert not result.startswith("Error:")
        assert (workdir / "out.txt").read_text() == "hi\n"

    def test_a_concurrent_write_is_not_silently_reverted(self, workdir):
        # Two chats sharing a project workspace: both read, both write, and the
        # later os.replace used to discard the earlier edit without a word.
        target = workdir / "s.py"
        target.write_text("A = 1\nB = 2\n")
        stale = target.read_bytes()
        _edit(path = "s.py", old_string = "B = 2", new_string = "B = 99")
        error = tools._edit_file_write(
            str(target),
            stale.decode().replace("A = 1", "A = 42"),
            "\n",
            "",
            expect = stale,
        )
        assert error.startswith("Error:")
        assert target.read_text() == "A = 1\nB = 99\n"

    def test_containment_is_rechecked_at_write_time(self, workdir):
        # The path is resolved, then the file is read and diffed before the
        # rename. A parent swapped for a symlink inside that window would
        # otherwise be followed.
        outside = workdir.parent / "escaped.txt"
        error = tools._edit_file_write(str(outside), "pwned", "\n", "", workdir = str(workdir))
        assert error.startswith("Error:")
        assert not outside.exists()

    def test_an_empty_file_stays_writable(self, workdir):
        # Refusing every existing target would strand the model here: an empty
        # old_string would be refused, and no other old_string can match an
        # empty file, so nothing could ever write to it.
        target = workdir / "placeholder.py"
        target.touch()
        result = _edit(path = "placeholder.py", old_string = "", new_string = "x = 1\n")
        assert not result.startswith("Error:")
        assert target.read_text() == "x = 1\n"


class TestSecondReviewFindings:
    """Cases raised in the second review pass."""

    def test_a_huge_replace_all_does_not_build_the_whole_diff(self, workdir):
        # difflib was fed the entire file and its generator drained into a list,
        # so replace_all on a file near the size cap allocated ~500MB and took
        # over a second to return a 200-character receipt.
        target = workdir / "big.txt"
        target.write_text("a\n" * 300_000)
        started = time.monotonic()
        result = _edit(path = "big.txt", old_string = "a", new_string = "b", replace_all = True)
        elapsed = time.monotonic() - started
        assert not result.startswith("Error:")
        assert len(result) < 2000
        # Windowing makes this near-instant; diffing 300k lines does not.
        assert elapsed < 2.0
        assert target.read_text().startswith("b\nb\n")

    def test_the_receipt_keeps_real_file_line_numbers(self, workdir):
        # The window is sliced out of the file, so difflib numbers the hunk from
        # the slice. A receipt pointing at line 3 of a 9000-line file would be
        # worse than none.
        target = workdir / "mid.py"
        target.write_text("".join(f"line{i}\n" for i in range(1, 9001)))
        result = _edit(path = "mid.py", old_string = "line8000\n", new_string = "CHANGED\n")
        assert "@@ -7998" in result
        assert "+CHANGED" in result

    def test_a_change_in_the_first_lines_still_numbers_from_one(self, workdir):
        target = workdir / "top.py"
        target.write_text("".join(f"line{i}\n" for i in range(1, 200)))
        result = _edit(path = "top.py", old_string = "line2\n", new_string = "TOP\n")
        assert "@@ -1" in result

    @pytest.mark.skipif(sys.platform == "win32", reason = "POSIX file mode")
    def test_a_created_file_gets_the_usual_mode(self, workdir):
        # mkstemp makes the temp file 0600 and copymode had nothing to copy from,
        # so new files landed 0600 and locked out anyone reading generated files.
        _edit(path = "fresh.py", old_string = "", new_string = "x = 1\n")
        umask = os.umask(0)
        os.umask(umask)
        mode = stat.S_IMODE(os.stat(workdir / "fresh.py").st_mode)
        assert mode == 0o666 & ~umask

    def test_creating_a_file_that_appeared_meanwhile_is_refused(self, workdir):
        # O_EXCL rather than lexists-then-write: two chats sharing a workspace
        # could both pass the check and the later write drop the earlier file.
        target = workdir / "race.py"
        assert not _edit(path = "race.py", old_string = "", new_string = "first\n").startswith("Error:")
        result = _edit(path = "race.py", old_string = "", new_string = "second\n")
        assert result.startswith("Error:")
        assert target.read_text() == "first\n"

    def test_filling_an_empty_file_is_guarded_against_a_racer(self, workdir):
        # The zero-byte path goes through the normal write, so it carries the
        # same expect check as an edit rather than clobbering blindly.
        target = workdir / "z.py"
        target.touch()
        target.write_text("someone got here first\n")
        error = tools._edit_file_write(str(target), "mine\n", "\n", "", expect = b"")
        assert error.startswith("Error:")
        assert target.read_text() == "someone got here first\n"


class TestThirdReviewFindings:
    """Cases raised in the cross-platform / filesystem review pass."""

    def test_the_receipt_does_not_invent_deletions_at_the_window_edge(self, workdir):
        # The diff window was cut out of each text by LINE COUNT. An edit that
        # adds or removes lines shifts everything after it, so the two windows
        # ended on different lines and difflib called that a second hunk: the
        # receipt reported "-line319" for a line still in the file, 119 lines
        # from anything the edit touched. A model that trusts the receipt --
        # the only reason to send one -- then "restores" a line that never left.
        target = workdir / "shift.py"
        target.write_text("".join(f"line{i}\n" for i in range(1, 401)))
        result = _edit(path = "shift.py", old_string = "line200\n", new_string = "A\nB\n")
        after = target.read_text()
        removed = [line[1:] for line in result.splitlines() if line.startswith("-")]
        assert removed == ["line200"]
        assert all(removed_line + "\n" not in after for removed_line in removed)

    def test_the_receipt_numbers_a_line_count_change_from_the_real_line(self, workdir):
        target = workdir / "grow.py"
        target.write_text("".join(f"line{i}\n" for i in range(1, 401)))
        result = _edit(path = "grow.py", old_string = "line200\n", new_string = "A\nB\n")
        assert "@@ -198,5 +198,6 @@" in result

    def test_a_deletion_does_not_invent_additions_at_the_window_edge(self, workdir):
        target = workdir / "shrink.py"
        target.write_text("".join(f"line{i}\n" for i in range(1, 401)))
        result = _edit(
            path = "shrink.py",
            old_string = "line200\nline201\nline202\n",
            new_string = "M\n",
        )
        after = target.read_text()
        added = [line[1:] for line in result.splitlines() if line.startswith("+")]
        assert added == ["M"]
        assert all(added_line + "\n" in after for added_line in added)

    @pytest.mark.skipif(sys.platform == "win32", reason = "POSIX FIFO")
    def test_creating_over_a_fifo_is_refused_rather_than_reopened(self, workdir):
        # The edit path refuses anything that is not a regular file, but a FIFO
        # reports st_size 0, so an empty old_string fell into the zero-byte
        # branch instead -- whose write re-reads the target to check nothing
        # changed underneath, and open() on a pipe with no writer never returns.
        # No timeout or cancel event reaches here, so the turn was unrecoverable.
        import threading

        os.mkfifo(workdir / "pipe")
        done = []
        worker = threading.Thread(
            target = lambda: done.append(_edit(path = "pipe", old_string = "", new_string = "x\n")),
            daemon = True,
        )
        worker.start()
        worker.join(10)
        assert done, "edit_file blocked forever on a FIFO"
        assert done[0].startswith("Error:")
        assert stat.S_ISFIFO(os.stat(workdir / "pipe").st_mode)

    def test_the_receipt_never_reports_a_change_the_file_does_not_show(self, workdir):
        # The property behind the two window cases above, over the shapes that
        # produced them: every '-' line must really be gone from the file and
        # every '+' line must really be in it. The receipt is the only thing the
        # model learns about the edit, so an untruthful one is a wrong answer
        # even when the bytes on disk are right.
        import random

        random.seed(7)
        target = workdir / "prop.py"
        for _ in range(60):
            total = random.choice([50, 130, 260, 500])
            at = random.randrange(1, total)
            grow = random.randrange(0, 6)
            target.write_text("".join(f"line{i}\n" for i in range(1, total + 1)))
            result = _edit(
                path = "prop.py",
                old_string = f"line{at}\n",
                new_string = "".join(f"N{j}\n" for j in range(grow)) or "Z\n",
            )
            after = target.read_text()
            for line in result.splitlines():
                if line.startswith("-") and not line.startswith("---"):
                    assert line[1:] + "\n" not in after, (total, at, grow, line)
                if line.startswith("+") and not line.startswith("+++"):
                    assert line[1:] + "\n" in after, (total, at, grow, line)

    @pytest.mark.skipif(sys.platform == "win32", reason = "POSIX device node")
    def test_full_access_does_not_replace_a_device_node(self, workdir):
        # Full access resolves an absolute path as written, and /dev/null stats
        # as zero bytes. Measuring size alone sent it down the create-an-empty-
        # file branch, whose atomic rename would have swapped the character
        # device for a regular file and broken every process on the box.
        result = execute_tool(
            "edit_file",
            {"path": "/dev/null", "old_string": "", "new_string": "x\n"},
            session_id = "t",
            disable_sandbox = True,
        )
        assert result.startswith("Error:")
        assert stat.S_ISCHR(os.stat("/dev/null").st_mode)

    @pytest.mark.parametrize("path,old", [("app.py", "TODO"), ("fresh.py", "")])
    def test_an_unencodable_new_string_is_refused_not_dropped(self, workdir, path, old):
        # '"\ud83d"' is what a truncated emoji looks like after json.loads: a
        # lone surrogate, a str that cannot be encoded. The write encoded before
        # its first try block, so the UnicodeEncodeError escaped, and the
        # in-flight contextmanager upstream swallowed it into "Unknown tool:
        # edit_file". Telling the model the tool does not exist is the worst
        # possible answer -- it goes back to rewriting the whole file, which is
        # the cost this tool exists to remove. Both the edit and the create path
        # reach the same encode, so both are pinned.
        import json

        arguments = json.loads(
            '{"path": "%s", "old_string": "%s", "new_string": "\\ud83d launch"}' % (path, old)
        )
        target = workdir / "app.py"
        target.write_text("x = 1\n# TODO\ny = 2\n")

        result = _edit(**arguments)

        assert result.startswith("Error:"), result
        assert "surrogate" in result
        assert target.read_text() == "x = 1\n# TODO\ny = 2\n"
        assert not (workdir / "fresh.py").exists()

    def test_a_paired_surrogate_emoji_still_writes_normally(self, workdir):
        # The guard must reject only what cannot be encoded; a real emoji
        # arrives as a matched pair and is ordinary text.
        target = workdir / "app.py"
        target.write_text("# TODO\n")
        result = _edit(path = "app.py", old_string = "TODO", new_string = "done \U0001f680")
        assert not result.startswith("Error:"), result
        assert target.read_text() == "# done \U0001f680\n"


class TestPublicSchema:
    def test_the_request_schema_lists_edit_file(self):
        # Programmatic clients pick tools off the generated OpenAPI schema; a
        # built-in missing from it is undiscoverable.
        from models.inference import ChatCompletionRequest
        description = ChatCompletionRequest.model_fields["enabled_tools"].description
        assert "edit_file" in description

    def test_bypass_permissions_says_edit_file_is_unconfined(self):
        from models.inference import ChatCompletionRequest
        description = ChatCompletionRequest.model_fields["bypass_permissions"].description
        assert "edit_file" in description


class TestRegistration:
    def test_the_tool_is_offered(self):
        assert EDIT_FILE_TOOL in ALL_TOOLS

    def test_the_description_steers_away_from_whole_file_rewrites(self):
        # A model handed this tool with no such steer keeps writing heredocs,
        # because that is what its training data is full of.
        description = EDIT_FILE_TOOL["function"]["description"].lower()
        assert "prefer this" in description
        assert "rewriting" in description

    def test_an_edit_still_asks_in_auto_mode(self):
        # python's open(..., "w") already prompts; the cheaper tool must not
        # become the quiet way around that.
        assert is_potentially_unsafe_tool_call("edit_file", {"path": "a.py"}) is True

    def test_full_access_says_absolute_paths_resolve(self):
        # Otherwise the model assumes it cannot reach a real checkout and falls
        # back to the whole-file rewrite exactly where files are largest.
        swapped = apply_full_access_tool_descriptions([EDIT_FILE_TOOL])
        assert swapped == [EDIT_FILE_TOOL_FULL_ACCESS]
        assert "absolute path" in swapped[0]["function"]["description"]

    def test_full_access_leaves_the_schema_alone(self):
        sandboxed = EDIT_FILE_TOOL["function"]["parameters"]
        assert EDIT_FILE_TOOL_FULL_ACCESS["function"]["parameters"] == sandboxed


class TestFullAccessEscapesTheWorkdir:
    def test_an_absolute_path_resolves_when_the_sandbox_is_off(self, workdir, tmp_path):
        # Full access runs python/terminal unsandboxed already; holding this one
        # tool to the workdir there would just push the model back to cat.
        outside = tmp_path.parent / "real_project.py"
        outside.write_text("x = 1\n")
        result = execute_tool(
            "edit_file",
            {"path": str(outside), "old_string": "x = 1", "new_string": "x = 2"},
            session_id = "t",
            disable_sandbox = True,
        )
        assert not result.startswith("Error:")
        assert outside.read_text() == "x = 2\n"


@pytest.mark.skipif(os.name != "posix", reason = "RLIMIT_FSIZE is POSIX-only")
class TestCreationLeavesNothingBehindWhenTheWriteFails:
    """A create that runs out of space must not strand a truncated file.

    RLIMIT_FSIZE is a real kernel write failure with the same shape as ENOSPC or
    a quota: the bytes that fit are on disk and the rest fail. Mocked here only
    to the extent of choosing when it happens.
    """

    @staticmethod
    def _capped(limit):
        import resource
        import signal

        soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
        previous = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, hard))
        return (soft, hard, previous)

    @staticmethod
    def _restore(saved):
        import resource
        import signal

        soft, hard, previous = saved
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
        signal.signal(signal.SIGXFSZ, previous)

    def test_a_half_written_file_is_removed_not_left_truncated(self, workdir):
        # The failure lands mid-payload, so without the cleanup the file is a
        # source file cut off in the middle of a token.
        body = "".join(f"def f{i}():\n    return {i}\n\n" for i in range(4000))
        saved = self._capped(4096)
        try:
            result = execute_tool(
                "edit_file",
                {"path": "report.py", "old_string": "", "new_string": body},
                session_id = "t",
            )
        finally:
            self._restore(saved)
        assert result.startswith("Error:")
        assert not (workdir / "report.py").exists()

    def test_the_retry_the_error_asks_for_then_succeeds(self, workdir):
        # An empty old_string refuses a non-empty target, so a leftover partial
        # file would make the failure permanent: nothing could ever write it.
        body = "".join(f"def f{i}():\n    return {i}\n\n" for i in range(4000))
        saved = self._capped(4096)
        try:
            execute_tool(
                "edit_file",
                {"path": "report.py", "old_string": "", "new_string": body},
                session_id = "t",
            )
        finally:
            self._restore(saved)
        retry = execute_tool(
            "edit_file",
            {"path": "report.py", "old_string": "", "new_string": body},
            session_id = "t",
        )
        assert not retry.startswith("Error:")
        assert (workdir / "report.py").read_text() == body

    def test_a_failure_at_close_leaves_nothing_either(self, workdir, monkeypatch):
        # A payload smaller than the io buffer reaches the disk only at close,
        # and close() is exactly where a full disk reports the failure of data
        # written earlier. Injected rather than rlimit'd so it lands at close
        # whatever the filesystem's block size makes the buffer.
        real = os.fdopen

        def failing(fd, *args, **kwargs):
            handle = real(fd, *args, **kwargs)
            closed = handle.close

            def close():
                # CPython releases the descriptor even when the closing flush
                # fails, so the real failure closes before it raises.
                closed()
                raise OSError(28, "No space left on device")

            handle.close = close
            return handle

        monkeypatch.setattr(os, "fdopen", failing)
        result = execute_tool(
            "edit_file",
            {"path": "notes.py", "old_string": "", "new_string": "print('hi')\n"},
            session_id = "t",
        )
        monkeypatch.undo()
        assert result.startswith("Error:")
        assert not (workdir / "notes.py").exists()

    def test_a_failed_create_does_not_remove_someone_elses_file(self, workdir):
        # The cleanup must only ever reach the inode this call created. An
        # existing non-empty target is refused before the O_EXCL open, so it
        # can never reach the failure path at all.
        target = workdir / "keep.py"
        target.write_text("x = 1\n")
        result = execute_tool(
            "edit_file",
            {"path": "keep.py", "old_string": "", "new_string": "y = 2\n"},
            session_id = "t",
        )
        assert result.startswith("Error:")
        assert "already exists" in result
        assert target.read_text() == "x = 1\n"
