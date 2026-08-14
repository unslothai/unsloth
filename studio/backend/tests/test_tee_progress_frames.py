# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The session log's copy of carriage-return progress bars.

A terminal overwrites a redraw in place; a file keeps every frame, so one tqdm bar landed as
kilobytes of near-identical text. The tee keeps the last frame only, and withholds nothing
except frames -- anything without a "\\r" is written the moment it arrives, so a hang cannot
swallow a partial traceback or a prompt.
"""

import io

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
    # "Start Unsloth Studio now? [Y/n]: " never gets a newline; it must still reach the file.
    log, _console = _tee(["\rbar 40%", "Start Unsloth Studio now? [Y/n]: "])
    assert log.endswith("Start Unsloth Studio now? [Y/n]: ")


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
