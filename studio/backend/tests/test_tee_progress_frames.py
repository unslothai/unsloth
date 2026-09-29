# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The session log's copy of carriage-return progress bars.

A terminal overwrites a redraw in place; a file keeps every frame, so one tqdm bar landed as
kilobytes of near-identical text. The tee keeps the last frame only, and withholds nothing
except frames -- anything without a "\\r" is written the moment it arrives, so a hang cannot
swallow a partial traceback or a prompt.
"""

import io
import json

from run import _TeeStream


class _Sink:
    def __init__(self):
        self.buf = io.StringIO()

    def write(self, data):
        self.buf.write(data)
        return len(data)

    def flush(self):
        pass

    @property
    def text(self):
        return self.buf.getvalue()


def _tee(chunks):
    log, console = _Sink(), _Sink()
    stream = _TeeStream(console, log)
    for chunk in chunks:
        stream.write(chunk)
    return log.text, console.text


def test_plain_output_is_unchanged_on_both_sides():
    log, console = _tee(["plain line\n", "another\n"])
    assert log == "plain line\nanother\n"
    assert console == "plain line\nanother\n"


def test_console_always_sees_every_frame():
    _log, console = _tee(["\rbar 1%", "\rbar 50%", "\rbar 100%", "\n"])
    # The animation is the console's whole point; only the file copy collapses.
    assert console == "\rbar 1%\rbar 50%\rbar 100%\n"


def test_progress_bar_collapses_to_its_final_frame():
    log, _console = _tee(["\rbar 1%", "\rbar 50%", "\rbar 100%", "\n"])
    assert log == "bar 100%\n"


def test_bar_between_real_lines_keeps_both():
    log, _console = _tee(["Loading\n", "\ra 10%", "\ra 99%", "\n", "done\n"])
    assert log == "Loading\na 99%\ndone\n"


def test_several_bars_in_one_chunk_collapse_per_line():
    log, _console = _tee(["a\rb\rc\nd\re\n"])
    assert log == "c\ne\n"


def test_unterminated_prompt_after_a_bar_is_not_withheld():
    # "Start Unsloth Studio now? [Y/n]: " never gets a newline; it must still reach the file,
    # and on its own line rather than glued to the frame that was being held.
    log, _console = _tee(["\rbar 40%", "Start Unsloth Studio now? [Y/n]: "])
    assert log == "bar 40%\nStart Unsloth Studio now? [Y/n]: "


def test_record_after_a_held_frame_stays_parseable():
    # The reason the frame is closed off rather than prefixed: a structlog record arriving
    # while a bar is mid-redraw must still be one JSON object on one line.
    log, _console = _tee(["\rLoading weights:  47%", '{"event": "model_loaded"}\n'])
    lines = log.splitlines()
    assert lines == ["Loading weights:  47%", '{"event": "model_loaded"}']
    json.loads(lines[-1])


def test_close_lands_a_frame_nothing_came_back_to_supersede():
    log, console = _Sink(), _Sink()
    stream = _TeeStream(console, log)
    stream.write("\rbar 90%")
    stream.close()
    assert log.text == "bar 90%\n"


def test_hang_mid_bar_keeps_the_real_partial_line():
    # The case that decides whether this is safe: a torn line is written, a frame is not.
    log, _console = _tee(["Traceback (most recent call last):", "\rbar 5%"])
    assert log == "Traceback (most recent call last):"


def test_file_failure_never_reaches_the_console():
    class Exploding(_Sink):
        def write(self, data):
            raise OSError("disk full")

    console = _Sink()
    stream = _TeeStream(console, Exploding())
    stream.write("still printed\n")
    assert console.text == "still printed\n"


# ---------------------------------------------------------------------------------------
# A "\r" is only a redraw when something follows it on the same line.
# ---------------------------------------------------------------------------------------


def test_a_crlf_line_keeps_its_payload():
    # "\r\n" is one terminator. Reading its "\r" as a redraw keeps the empty text after it
    # and drops the line -- and on Windows every relayed child line arrives in this shape,
    # so the session log goes blank exactly where the evidence should be.
    log, _console = _tee(["Hardware detected: NVIDIA GeForce RTX 4090\r\n"])
    assert log == "Hardware detected: NVIDIA GeForce RTX 4090\n"


def test_a_crlf_traceback_is_not_reduced_to_blank_lines():
    log, _console = _tee(
        ['Traceback (most recent call last):\r\n  File "run.py", line 3\r\nRuntimeError: boom\r\n']
    )
    assert log.splitlines() == [
        "Traceback (most recent call last):",
        '  File "run.py", line 3',
        "RuntimeError: boom",
    ]


def test_a_crlf_record_stays_one_json_object():
    log, _console = _tee(['{"event": "model_loaded"}\r\n'])
    assert log == '{"event": "model_loaded"}\n'
    json.loads(log.strip())


def test_a_bar_signing_off_with_a_bare_cr_keeps_its_last_frame():
    # tqdm's close() can leave the terminator on the same write as the final frame.
    log, _console = _tee(["Map:  50%\rMap: 100%\r\n"])
    assert log == "Map: 100%\n"


def test_an_all_blank_line_never_writes_a_carriage_return():
    # The handle appends the platform terminator itself, so a surviving "\r" lands as
    # "\r\r\n" on Windows.
    for chunk in ("\r\n", "\r\r\r\n", "   \r   \n"):
        log, _console = _tee([chunk])
        assert "\r" not in log, repr(chunk)


def test_a_zero_length_write_does_not_glue_a_frame_onto_the_next_record():
    # print("", end = "") is enough: an empty write used to read as a continuation of the
    # held frame, which then fell through and was written with no newline.
    log, _console = _tee(["\rLoading weights:  47%", "", '{"event": "model_loaded"}\n'])
    lines = log.splitlines()
    assert lines == ["Loading weights:  47%", '{"event": "model_loaded"}']
    json.loads(lines[-1])


def test_the_collapse_matches_the_desktop_reader():
    """Same rule as collapse_progress_frames in src-tauri/src/process.rs.

    Settings > Logs offers both sinks side by side, so a line must look the same in either.
    """
    cases = {
        "plain line": "plain line",
        "a\rb\rc": "c",
        "bar 100%\r": "bar 100%",
        "Map:  50%\rMap: 100%\r   ": "Map: 100%",
        "Hardware detected: ROCm": "Hardware detected: ROCm",
        "TAURI_PORT=8888\r": "TAURI_PORT=8888",
    }
    for line, expected in cases.items():
        log, _console = _tee([line + "\n"])
        assert log == expected + "\n", f"{line!r} -> {log!r}, expected {expected!r}"
