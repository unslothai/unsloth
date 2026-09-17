# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth#7843: a worker that exits without answering must not lose its own error output.

The reported failure surfaced as "The inference worker stopped unexpectedly while generating
a response. Details: pid=6145, exitcode=1." and nothing else. The worker had printed a
traceback, but onto the stderr it inherited from the server rather than through the response
queue, so the orchestrator had an exit status and no cause. The issue's "Expected" is exactly
that the underlying exception be preserved.

Covered here: the decoder (Windows CRLF and cp1252 bytes, macOS and Linux UTF-8, undecodable
bytes), the tail bounds, the mirror file's size bound, the message the orchestrator builds,
and a real spawned worker that writes to stderr and exits 1.
"""

import subprocess
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import multiprocessing as mp
import os
import signal
import sys

import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils.worker_stderr import (  # noqa: E402
    MIRROR_FILE_CAP_BYTES,
    STDERR_MIRROR_KWARG,
    WorkerStderrCapture,
    decode_worker_stderr,
    stderr_tail_from_bytes,
)


def _load_orchestrator_module():
    """Load orchestrator.py by path, the way test_inference_orchestrator_crash_message does."""
    spec = importlib.util.spec_from_file_location(
        "inference_orchestrator_stderr_tail_under_test",
        Path(_BACKEND_DIR) / "core/inference/orchestrator.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _orchestrator_with(
    exitcode,
    capture = None,
    pid = 6145,
):
    module = _load_orchestrator_module()
    orchestrator = module.InferenceOrchestrator.__new__(module.InferenceOrchestrator)
    orchestrator._proc = SimpleNamespace(pid = pid, exitcode = exitcode)
    orchestrator._stderr_capture = capture
    return orchestrator


# --- the decoder -----------------------------------------------------------------------


def test_a_windows_worker_arrives_as_crlf_cp1252_and_decodes_cleanly():
    # sys.stderr on Windows uses the ANSI code page unless UTF-8 mode is on, and native
    # libraries write the console page. cp1252 with CRLF is the common shape.
    raw = "Traceback (most recent call last):\r\n  File “worker.py”, line 1\r\n".encode(
        "cp1252",
    )
    assert b"\r\n" in raw
    text = decode_worker_stderr(raw)
    assert "\r" not in text
    assert text.splitlines()[0] == "Traceback (most recent call last):"
    assert "“worker.py”" in text


def test_a_utf8_worker_is_never_mis_read_as_cp1252():
    # Linux and macOS write UTF-8. Probed in that order precisely so a UTF-8 em dash does
    # not come back as the two cp1252 characters its bytes spell.
    raw = "RuntimeError: — out of memory é\n".encode("utf-8")
    assert decode_worker_stderr(raw) == "RuntimeError: — out of memory é\n"


def test_cp1252_bytes_that_are_not_valid_utf8_still_decode():
    raw = b"warning: caf\xe9 layer skipped\r\n"
    assert decode_worker_stderr(raw) == "warning: café layer skipped\n"


def test_undecodable_bytes_are_replaced_rather_than_dropped():
    # 0x81 and 0x90 are undefined in cp1252 and invalid as UTF-8. A mangled traceback still
    # has to name the exception, so the text is kept with replacement characters.
    raw = b"RuntimeError: \x81\x90 bad bytes\n"
    text = decode_worker_stderr(raw)
    assert "RuntimeError:" in text
    assert "bad bytes" in text


def test_a_progress_bar_redrawn_with_carriage_returns_does_not_hide_the_traceback():
    raw = (
        b"Loading checkpoint:  10%\rLoading checkpoint:  55%\rLoading checkpoint: 100%\r\n"
        b"RuntimeError: the real failure\n"
    )
    tail = stderr_tail_from_bytes(raw, max_lines = 2)
    assert tail.splitlines()[-1] == "RuntimeError: the real failure"


# --- the tail bounds -------------------------------------------------------------------


def test_the_tail_keeps_the_last_lines_and_drops_blank_ones():
    raw = ("\n".join(f"line {index}" for index in range(100)) + "\n\n\n").encode("utf-8")
    tail = stderr_tail_from_bytes(raw, max_lines = 3)
    assert tail == "line 97\nline 98\nline 99"


def test_the_character_cap_trims_from_the_front_and_never_shows_a_half_line():
    raw = ("first line that is quite long\n" + "b" * 200 + "\nlast line\n").encode("utf-8")
    tail = stderr_tail_from_bytes(raw, max_lines = 10, max_chars = 60)
    assert tail.endswith("last line")
    assert len(tail) <= 60
    # A partial leading line is dropped rather than shown cut in half.
    assert not tail.startswith("b")


def test_an_empty_sink_produces_no_tail():
    assert stderr_tail_from_bytes(b"") == ""
    assert stderr_tail_from_bytes(b"\n  \n\r\n") == ""


def test_a_capture_whose_file_is_gone_reports_no_tail_instead_of_raising():
    capture = WorkerStderrCapture(prefix = "unsloth-test-")
    capture.close()
    assert capture.tail() == ""
    # Idempotent: a second close is what a Windows host takes when the child still holds it.
    capture.close()


# --- the message the orchestrator builds ------------------------------------------------


def test_a_positive_exit_carries_the_worker_stderr_tail(tmp_path):
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    Path(capture.path).write_bytes(
        b"Starting text generation\r\n"
        b"Traceback (most recent call last):\r\n"
        b"RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB\r\n",
    )

    message = _orchestrator_with(1, capture)._subprocess_crash_message(
        "generation", with_worker_output = True
    )

    assert message.startswith(
        "The inference worker stopped unexpectedly while generating a response.",
    )
    assert "Details: pid=6145, exitcode=1." in message
    assert "Worker error output:" in message
    assert "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB" in message
    # Normalised on the way in, so the surfaced error is not littered with stray returns.
    assert "\r" not in message


def test_a_signalled_exit_keeps_its_hint_and_gains_the_tail(tmp_path):
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    Path(capture.path).write_bytes(b"terminate called after throwing an instance of 'c10::Error'\n")

    message = _orchestrator_with(-9, capture)._subprocess_crash_message(
        "wait", with_worker_output = True
    )

    # The part this change is responsible for, on every platform: the negative exit code is
    # still reported as a signal and the memory-pressure hint is still there, and the worker's
    # own output is now carried alongside them rather than instead of nothing.
    assert "exitcode=-9" in message
    assert "c10::Error" in message

    # The hint itself is POSIX. A Windows process exits with a DWORD and never reports a
    # negative code, and signal.Signals(9) raises there, so the existing branch names the
    # signal SIG9 and offers no hint. Asserting the Linux and macOS wording unconditionally
    # is what made this test the first thing in the file to fail on a Windows runner.
    if hasattr(signal, "SIGKILL"):
        assert "memory pressure" in message
        assert "signal=SIGKILL" in message
    else:
        assert "signal=SIG9" in message


def test_nothing_captured_leaves_the_message_exactly_as_it_was(tmp_path):
    empty = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    for capture in (None, empty):
        message = _orchestrator_with(1, capture)._subprocess_crash_message(
            "generation", with_worker_output = True
        )
        assert message == (
            "The inference worker stopped unexpectedly while generating a response. "
            "Details: pid=6145, exitcode=1."
        )


def test_a_worker_still_running_is_not_given_a_tail(tmp_path):
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    Path(capture.path).write_bytes(b"still going\n")
    message = _orchestrator_with(None, capture)._subprocess_crash_message(
        "generation", with_worker_output = True
    )
    assert message.endswith("Details: pid=6145.")


def test_the_orchestrator_hands_the_sink_to_the_child_it_spawns():
    # The wiring, read from the source: exercising it for real needs a model load. Paired
    # with the spawned-worker tests below, which drive the same shared child entrypoint.
    source = (
        (Path(_BACKEND_DIR) / "core/inference/orchestrator.py")
        .read_text(
            encoding = "utf-8",
        )
        .replace("\r\n", "\n")
    )
    start = source.index("def _spawn_subprocess(")
    body = source[start : source.index("\n    def ", start + 10)]
    assert "WorkerStderrCapture" in body, "the spawn path no longer opens a stderr sink"
    assert "STDERR_MIRROR_KWARG" in body, "the sink is no longer passed to the child"
    assert "_retire_stderr_capture()" in body, "a stale sink now survives into the next worker"


def test_the_reserved_kwarg_name_is_the_same_on_both_sides():
    # utils/native_path_leases.py spells the name out to keep its import graph stdlib only.
    from utils.native_path_leases import STDERR_MIRROR_KWARG as entrypoint_name
    assert entrypoint_name == STDERR_MIRROR_KWARG


# --- a real spawned worker --------------------------------------------------------------


def _spawn(
    function_name,
    mirror_path,
    timeout = 180,
):
    from utils.native_path_leases import run_without_native_path_secret

    # Imported by bare name, and this directory put on sys.path for it: "tests" resolves to
    # the repository's own top-level tests package when pytest is run from the repo root, so
    # a dotted name here finds the wrong one. The spawn start method hands the child the
    # parent's sys.path, which is what makes this enough.
    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)

    context = mp.get_context("spawn")
    kwargs = {} if mirror_path is None else {STDERR_MIRROR_KWARG: mirror_path}
    process = context.Process(
        target = run_without_native_path_secret,
        args = ("_worker_stderr_boom", function_name, {}),
        kwargs = kwargs,
    )
    process.start()
    process.join(timeout)
    assert not process.is_alive(), "the test worker did not exit"
    return process


def test_a_worker_that_exits_one_after_writing_to_stderr_keeps_its_traceback(tmp_path):
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    process = _spawn("exit_one_after_writing_to_stderr", capture.path)
    assert process.exitcode == 1

    orchestrator = _orchestrator_with(process.exitcode, capture, pid = process.pid)
    message = orchestrator._subprocess_crash_message("generation", with_worker_output = True)

    assert "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB" in message
    assert "Traceback (most recent call last)" in message
    # The log lines that preceded it are the context the issue quotes from the server log.
    assert "Starting text generation" in capture.tail(max_lines = 100)


def test_the_reserved_kwarg_is_consumed_and_never_reaches_the_entrypoint(tmp_path):
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    process = _spawn("record_received_kwargs", capture.path)
    assert process.exitcode == 1
    tail = capture.tail(max_lines = 100)
    assert "entrypoint kwargs: []" in tail, tail


def test_the_sink_is_bounded_even_when_the_worker_floods_stderr(tmp_path):
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    process = _spawn("write_far_more_than_the_cap", capture.path)
    assert process.exitcode == 1

    size = os.path.getsize(capture.path)
    assert size <= 2 * MIRROR_FILE_CAP_BYTES + 65536, size
    # Compaction keeps the end, which is the only part worth keeping.
    assert "the last line before the worker died" in capture.tail(max_lines = 200)


# --- installs that are already out there --------------------------------------------------


def test_a_caller_that_passes_no_mirror_is_unchanged(tmp_path):
    """Every other worker in the app, and every Studio that has not been updated yet.

    ``run_without_native_path_secret`` is the shared entrypoint for the inference, export,
    training, data-recipe, speech and diffusion-quant workers, and only the inference
    orchestrator opens a sink. The rest pass no such keyword argument and must be affected in
    no way at all: same target, same arguments, same exit status, no sink and no tail.
    """
    process = _spawn("exit_one_after_writing_to_stderr", None)
    assert process.exitcode == 1
    assert _orchestrator_with(process.exitcode, None, pid = process.pid)._subprocess_crash_message(
        "generation",
    ) == (
        "The inference worker stopped unexpectedly while generating a response. "
        f"Details: pid={process.pid}, exitcode=1."
    )


def test_a_server_that_exits_leaves_no_sink_behind(tmp_path):
    """Residue bound. A sink is retired when the next worker is spawned, so the case with no
    next worker is the server exiting, and a desktop that is started and stopped every day
    would otherwise drop one file per session into the temporary directory for ever."""
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from utils.worker_stderr import WorkerStderrCapture\n"
        "print(WorkerStderrCapture(directory = %r, prefix = 'unsloth-test-').path)\n"
    ) % (_BACKEND_DIR, str(tmp_path))
    import subprocess

    path = subprocess.run(
        [sys.executable, "-c", script],
        capture_output = True,
        text = True,
        check = True,
    ).stdout.strip()
    assert path.startswith(str(tmp_path))
    assert not os.path.exists(path), "the sink outlived the process that opened it"


def test_a_severed_multibyte_character_does_not_mojibake_the_whole_tail():
    """The byte window the parent reads opens at an arbitrary offset.

    ``tail`` reads the last ``TAIL_READ_BYTES`` of the sink, and the pump compacts the sink
    to its last ``cap_bytes``; neither can land on a character boundary on purpose. cp1252
    has a meaning for almost every byte, so a single severed character used to make strict
    UTF-8 fail for the entire window and hand the whole tail to cp1252: "modellädt nicht —"
    came back as "modellÃ¤dt nicht â€”". The severed character is dropped instead.
    """
    from utils.worker_stderr import TAIL_READ_BYTES

    line = "RuntimeError: modellädt nicht — CUDA out of memory\n".encode("utf-8")
    body = line * 30
    # Exactly one continuation byte at the front, so the window opens mid-character.
    window = b"\xa9" + b"." * (TAIL_READ_BYTES - 1 - len(body)) + body
    assert len(window) == TAIL_READ_BYTES
    with pytest.raises(UnicodeDecodeError):
        window.decode("utf-8")

    tail = stderr_tail_from_bytes(window)
    assert "modellädt nicht — CUDA out of memory" in tail, tail
    assert "Ã¤" not in tail and "â€”" not in tail, tail


def test_a_severed_character_at_the_end_of_the_window_is_dropped_too():
    """Compaction cuts the far end as well, so the same trim applies there."""
    raw = "RuntimeError: café blew up — ".encode("utf-8") + "é".encode("utf-8")[:1]
    text = decode_worker_stderr(raw)
    assert "café blew up —" in text, text
    assert "Ã©" not in text, text


def test_a_real_worker_whose_output_outgrows_the_read_window_still_reads_cleanly(tmp_path):
    """End to end, through a spawned worker, not through the decoder alone."""
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    process = _spawn("write_non_ascii_far_past_the_read_window", capture.path, timeout = 60)
    assert process.exitcode == 1
    tail = capture.tail(max_lines = 5)
    assert "modellädt nicht — CUDA out of memory" in tail, tail
    assert "Ã¤" not in tail, tail
    capture.close()


def test_a_missing_sink_is_never_re_created_by_the_child(tmp_path):
    """The only way the child finds the sink gone is that the parent retired it.

    Re-creating it would create in a shared temporary directory with ``0666 & ~umask``,
    measured at 0664 under the default umask, so another account on the machine could read a
    worker's stderr. No sink is the old behaviour and is the right answer here.
    """
    from utils.worker_stderr import install_worker_stderr_mirror

    missing = tmp_path / "already-retired.stderr"
    assert install_worker_stderr_mirror(str(missing)) is False
    assert not missing.exists(), "the child created a sink the parent had retired"


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason = "O_NOFOLLOW is POSIX only")
def test_a_symlink_at_the_sink_path_is_not_followed(tmp_path):
    """The sink path is a name in a world-writable directory once the parent unlinks it.

    Following a symlink there would open, and with the old "wb" fallback truncate, any file
    the Studio user can write, and then fill it with the worker's stderr.
    """
    from utils.worker_stderr import install_worker_stderr_mirror

    victim = tmp_path / "victim.txt"
    victim.write_text("ORIGINAL\n", encoding = "utf-8")
    link = tmp_path / "sink.stderr"
    link.symlink_to(victim)

    assert install_worker_stderr_mirror(str(link)) is False
    assert victim.read_text(encoding = "utf-8") == "ORIGINAL\n"


def test_a_worker_holding_a_second_handle_on_stderr_still_exits_promptly(tmp_path):
    """A logging handler or a native library that dup'd fd 2 keeps the mirror's pipe open.

    The teardown then cannot see EOF, so it must not close the inherited descriptor while
    the pump thread may still be writing to it: the number would be free for the next
    ``open()`` in any thread and the worker's stderr would land in an unrelated file.
    """
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    process = _spawn("hold_a_second_handle_on_stderr_then_exit", capture.path, timeout = 60)
    assert process.exitcode == 1
    assert "held a second handle on stderr" in capture.tail(), capture.tail()
    capture.close()

    # The race itself cannot be asserted from the outside: it is a descriptor number being
    # reused between two threads, and a run that happens not to hit it proves nothing. So the
    # guard is read out of the source, the same way the sweep guard below is.
    source = (Path(_BACKEND_DIR) / "utils/worker_stderr.py").read_text(encoding = "utf-8")
    teardown = source.split("def _stop_mirror(", 1)[1].split("\ndef ", 1)[0]
    close_at = teardown.index("os.close(inherited_fd)")
    assert "if pump.is_alive():" in teardown[:close_at], (
        "the teardown closes the inherited stderr without first checking that the pump "
        "thread has finished with it"
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason = "fork is POSIX only")
def test_a_forked_child_does_not_retire_its_parents_live_sink(tmp_path):
    """A fork inherits both the open-sink set and the atexit registration.

    A forked child that exits normally then runs the inherited handler and unlinks a sink its
    parent is still filling, which is the same class of mistake as sweeping the directory:
    the paths are named exactly, but they are somebody else's. Spawn children re-import this
    module clean, so this is about the backend's own fork sites (dataset preprocessing pools)
    rather than about the workers.
    """
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    Path(capture.path).write_bytes(b"the parent is still writing here\n")

    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        # What a normal interpreter exit does, without unwinding into pytest's own teardown.
        import atexit as _atexit
        try:
            _atexit._run_exitfuncs()
        finally:
            os._exit(0)
    os.waitpid(pid, 0)

    assert os.path.exists(capture.path), "a forked child retired its parent's live sink"
    assert "the parent is still writing here" in capture.tail()
    capture.close()


def test_the_sink_directory_is_never_swept_by_pattern():
    """Another Studio's live sink is not ours to delete. Several installs share one temporary
    directory (a second UNSLOTH_STUDIO_HOME, a second account), so cleanup must name the exact
    paths this process opened and must never match on the prefix or an age."""
    source = (Path(_BACKEND_DIR) / "utils/worker_stderr.py").read_text(encoding = "utf-8")
    # Spelled as calls, not as bare words: "glob" is a substring of "global".
    for forbidden in (
        "glob(",
        "iglob(",
        "iterdir(",
        "listdir(",
        "scandir(",
        "walk(",
        "st_mtime",
        "rmtree(",
    ):
        assert forbidden not in source, f"cleanup grew a directory sweep ({forbidden})"


# The directory layouts a real install's temporary directory sits in. An Unsloth that was
# installed months ago keeps whatever TMPDIR, TEMP or per-user temporary directory it had,
# and the sink is created there. Spaces and non-ASCII are the two that actually break code
# which builds paths by string: a Windows profile is routinely "C:\Users\Jose Munoz\AppData
# \Local\Temp" with an accent in it, WSL reaches the same directory through /mnt/c, and macOS
# hands every process a generated /var/folders path.
_INSTALL_TEMP_SHAPES = {
    "linux": "tmp",
    "windows-profile": "C_/Users/Jos\u00e9 Mu\u00f1oz/AppData/Local/Temp",
    "wsl-windows-drive": "mnt/c/Users/Jos\u00e9 Mu\u00f1oz/AppData/Local/Temp",
    "macos-per-user": "var/folders/9k/9k1_2p_s0qz3h/T",
}


@pytest.mark.parametrize("shape", sorted(_INSTALL_TEMP_SHAPES))
def test_the_sink_works_on_every_install_path_shape(tmp_path, shape):
    directory = tmp_path.joinpath(*_INSTALL_TEMP_SHAPES[shape].split("/"))
    try:
        directory.mkdir(parents = True)
    except (OSError, UnicodeError) as exc:
        pytest.skip(f"this filesystem cannot hold a {shape} path: {exc}")

    capture = WorkerStderrCapture(directory = str(directory), prefix = "unsloth-test-")
    assert Path(capture.path).parent == directory
    process = _spawn("exit_one_after_writing_to_stderr", capture.path)
    assert process.exitcode == 1

    message = _orchestrator_with(
        process.exitcode,
        capture,
        pid = process.pid,
    )._subprocess_crash_message("generation", with_worker_output = True)
    assert "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB" in message

    capture.close()


@pytest.mark.skipif(os.name == "nt", reason = "the parent's stderr is redirected per platform")
def test_the_mirror_writes_through_to_the_inherited_stderr(tmp_path, capfd):
    # The server log must keep everything it had: the mirror tees, it does not divert.
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    _spawn("exit_one_after_writing_to_stderr", capture.path)
    captured = capfd.readouterr()
    assert "Starting text generation" in captured.err
    assert "CUDA out of memory" in captured.err


# ---------------------------------------------------------------------------
# What leaves the host is not what goes in the log
# ---------------------------------------------------------------------------


class _FixedCapture:
    def __init__(self, text):
        self._text = text

    def tail(self):
        return self._text


def _orchestrator_with_capture(text):
    from core.inference.orchestrator import InferenceOrchestrator

    instance = InferenceOrchestrator.__new__(InferenceOrchestrator)
    instance._stderr_capture = _FixedCapture(text)
    return instance


CROSS_ACCOUNT = (
    "2026-09-16 10:00:01 audio_codecs.decode_bicodec: generated text: "
    "another account's private prompt\n"
    "2026-09-16 10:00:02 worker: request 41 finished\n"
)
TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/home/alice/.unsloth/studio/worker.py", line 42, in run\n'
    '    raise RuntimeError("boom")\n'
    "RuntimeError: boom\n"
)


def test_the_public_tail_drops_everything_before_the_traceback():
    """The capture is the worker's whole lifetime, not this request.

    On a managed multi-user install this string goes out through
    `GenStreamError(public = True)`, which returns it verbatim rather than reducing it
    through `safe_error_detail`. Anything another account's request logged in those lines
    went out with it, and `audio_codecs.decode_bicodec` logs the first 500 characters of
    generated text.
    """
    public = _orchestrator_with_capture(CROSS_ACCOUNT + TRACEBACK)._public_worker_stderr_tail()
    assert "another account" not in public, public
    assert "request 41 finished" not in public, public
    # And the crash itself is still reported, or the redaction has cost the user #7843.
    assert public.startswith("Traceback (most recent call last):"), public
    assert "RuntimeError: boom" in public


def test_the_public_tail_keeps_only_the_last_traceback():
    """An earlier, recovered-from traceback belonged to someone else's request too."""
    earlier = TRACEBACK.replace("boom", "an earlier unrelated failure")
    public = _orchestrator_with_capture(
        earlier + CROSS_ACCOUNT + TRACEBACK
    )._public_worker_stderr_tail()
    assert "an earlier unrelated failure" not in public, public
    assert "RuntimeError: boom" in public


def test_a_capture_of_nothing_but_logging_is_dropped_entirely():
    """With no traceback, what is left is ordinary logging, which is exactly the
    cross-account content. The message degrades to the exit status, which is what it was
    before this capture existed."""
    assert _orchestrator_with_capture(CROSS_ACCOUNT)._public_worker_stderr_tail() == ""
    assert _orchestrator_with_capture("")._public_worker_stderr_tail() == ""


def test_a_native_abort_with_no_traceback_is_still_reported():
    """`terminate called after throwing an instance of 'c10::Error'` IS the diagnosis, and
    it arrives with no traceback at all. Dropping every capture that lacks one would cost
    the most common native crash its only explanation, so the filter is on the shape of a
    log record rather than on the absence of a traceback.

    The distinction is real rather than convenient: application code prints content through
    the logger, and a native abort or a fatal signal writes a bare line.
    """
    abort = "terminate called after throwing an instance of 'c10::Error'\n"
    public = _orchestrator_with_capture(CROSS_ACCOUNT + abort)._public_worker_stderr_tail()
    assert "c10::Error" in public, public
    assert "another account" not in public, public


def test_the_log_record_shapes_are_the_ones_content_arrives_in():
    """Wrong in one direction this drops a diagnostic line; in the other it forwards
    someone else's logged content to a client."""
    from core.inference.orchestrator import _looks_like_a_log_record

    for record in (
        "2026-09-16 10:00:01 audio_codecs.decode_bicodec: generated text: hello",
        "2026-09-16T10:00:01 worker: started",
        "INFO: loaded the model",
        "[WARNING] falling back to CPU",
        "audio_codecs.decode_bicodec: generated text: hello",
    ):
        assert _looks_like_a_log_record(record) is True, record

    for kept in (
        "terminate called after throwing an instance of 'c10::Error'",
        "Traceback (most recent call last):",
        "RuntimeError: boom",
        '  File "/x/worker.py", line 42, in run',
        "Segmentation fault (core dumped)",
    ):
        assert _looks_like_a_log_record(kept) is False, kept


def test_the_public_tail_redacts_paths_and_credentials():
    """Ordinary tracebacks carry the operator's filesystem layout, and an exception message
    can carry a token. The file NAME survives, because that is what makes the report
    useful and it is Unsloth's own module rather than the user's data."""
    from core.inference.orchestrator import _redact_worker_output

    text = (
        "Traceback (most recent call last):\n"
        '  File "/home/alice/.unsloth/studio/worker.py", line 42, in run\n'
        '    login(token="hf_abcdefghijklmnopqrstuvwxyz012345")\n'
        "RuntimeError: refused, Authorization: Bearer sk-secret-value\n"
    )
    public = _redact_worker_output(text)
    assert "/home/alice" not in public, public
    assert "worker.py" in public, public
    assert "hf_abcdefghijklmnopqrstuvwxyz012345" not in public, public
    assert "sk-secret-value" not in public, public
    assert "RuntimeError: refused" in public, public


def test_the_crash_message_uses_the_public_tail():
    """The narrowing is only worth anything if the message-building call site uses it. A
    call site still reading the raw capture would leave every case above passing."""
    import inspect

    from core.inference.orchestrator import InferenceOrchestrator

    source = inspect.getsource(InferenceOrchestrator._subprocess_crash_message)
    assert "_public_worker_stderr_tail()" in source
    assert "= self._worker_stderr_tail()" not in source


# ---------------------------------------------------------------------------
# A fatal signal must not take the diagnostic with it
# ---------------------------------------------------------------------------

_FATAL_CHILD = r"""
import os, signal, sys
sys.path.insert(0, %(backend)r)
from utils.worker_stderr import install_worker_stderr_mirror

assert install_worker_stderr_mirror(%(path)r) is True
os.write(2, b"terminate called after throwing an instance of 'c10::Error'\n")
# No flush, no atexit, no interpreter shutdown: the process is gone between one
# instruction and the next, which is what SIGSEGV and SIGABRT do.
# SIGKILL on POSIX; Windows has no such signal, and os.kill there calls
# TerminateProcess, which is the same thing for this purpose: no handler, no cleanup.
os.kill(os.getpid(), getattr(signal, "SIGKILL", signal.SIGTERM))
"""


def test_a_worker_killed_outright_still_leaves_its_last_words(tmp_path):
    """The case a pipe on fd 2 cannot serve.

    A pipe makes this process both the writer and the only reader, so bytes still in the
    buffer when a fatal signal arrives die with it -- and they never reach the inherited
    stderr either, which means the capture made the crash LESS visible than it was before
    the capture existed. Writing fd 2 straight into the sink puts every byte in the file
    the moment the kernel returns from the write.

    SIGKILL rather than SIGSEGV because it is the one signal nothing can intercept, so a
    pass here cannot be an artefact of a handler running. Windows has no SIGKILL at all --
    `signal.SIGKILL` raises AttributeError there, which is what the Windows CI leg caught --
    and `os.kill` on Windows calls TerminateProcess, which ends the process just as abruptly.
    The exit STATUS differs (a negative signal number against a Windows exit code), so only
    the part that means the same thing on both is asserted: the process did not exit
    normally, and the bytes are in the file anyway.
    """
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    backend = str(Path(__file__).resolve().parent.parent)
    script = _FATAL_CHILD % {"backend": backend, "path": capture.path}

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output = True,
        timeout = 120,
    )
    if os.name == "nt":
        assert completed.returncode != 0, completed.stderr[-400:]
    else:
        assert completed.returncode == -signal.SIGKILL, completed.stderr[-400:]
    assert "c10::Error" in capture.tail(), capture.tail()


def test_the_mirror_puts_the_sink_on_fd_two_rather_than_a_pipe():
    """Structural, because the mechanism is the fix.

    A tail thread that still owned a pipe would pass the case above whenever the buffer
    happened to be drained in time, and fail it on a loaded machine.
    """
    from utils import worker_stderr

    source = Path(worker_stderr.__file__).read_text(encoding = "utf-8")
    install = source.split("def install_worker_stderr_mirror(", 1)[1]
    assert "os.pipe()" not in install, "fd 2 is a pipe again"
    assert "_open_sink_for_append" in install
    assert "os.dup2(writer_fd, 2)" in install
    # O_APPEND is not decoration: the compaction rewrites the file from the front while
    # fd 2 still points at it, and a fixed offset would overwrite or leave a hole.
    append = source.split("def _open_sink_for_append(", 1)[1].split("\ndef ", 1)[0]
    assert "os.O_APPEND" in append


MULTILINE_CROSS_ACCOUNT = (
    "2026-09-16 10:00:01 audio_codecs.decode_bicodec: generated text: the first line\n"
    "another account's second line\n"
    "and a third line of the same prompt\n"
)


def test_a_logged_message_cannot_leak_through_its_own_continuation_lines():
    """A log record is not always one line.

    `logger.info("generated text: %s", text)` puts the timestamp and the logger name on the
    FIRST line only, so a filter that rejects lines one at a time drops the prefix and keeps
    everything the message actually contained. The continuations have to go with the record
    they belong to.
    """
    abort = (
        "terminate called after throwing an instance of 'c10::Error'\n"
        "  what():  CUDA error: device-side assert triggered\n"
    )
    public = _orchestrator_with_capture(
        MULTILINE_CROSS_ACCOUNT + abort
    )._public_worker_stderr_tail()
    assert "c10::Error" in public, public
    assert "device-side assert" in public, public
    assert "another account" not in public, public
    assert "third line" not in public, public


def test_compaction_never_deletes_what_a_racing_writer_appended(tmp_path):
    """fd 2 stays O_APPEND on the file the compactor rewrites.

    Anything the worker appends after the compactor has read the tail lies beyond the offset
    it is about to truncate to, and the pump has not forwarded it either, so a plain
    read-rewrite-truncate loses a fatal diagnostic from the capture and from the console
    both. The write here goes through a real O_APPEND descriptor, from inside the
    compactor's own read, which is the window.
    """
    from utils.worker_stderr import _compact_sink

    path = tmp_path / "sink"
    path.write_bytes(b"x" * 4096)
    appender = os.open(str(path), os.O_WRONLY | os.O_APPEND)
    handle = open(path, "r+b", buffering = 0)

    class _WriterRacesTheRead:
        def __init__(self, inner):
            self._inner = inner
            self._raced = False
            self._raced_the_write = False

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def read(self, *args):
            data = self._inner.read(*args)
            if not self._raced:
                self._raced = True
                os.write(appender, b"terminate called after throwing an instance of 'c10::Error'\n")
            return data

        def write(self, payload):
            written = self._inner.write(payload)
            if not self._raced_the_write:
                self._raced_the_write = True
                os.write(appender, b"what():  CUDA error: device-side assert triggered\n")
            return written

    try:
        _compact_sink(_WriterRacesTheRead(handle), 1024)
    finally:
        handle.close()
        os.close(appender)

    kept = path.read_bytes()
    # Both windows: the append that landed during the tail READ, and the one that landed
    # during the REWRITE, which is the other half of a quarter-megabyte round trip.
    assert b"terminate called" in kept, kept[-300:]
    assert b"device-side assert" in kept, kept[-300:]
    assert len(kept) <= 1024 + 256, len(kept)


def test_the_operator_still_gets_the_last_words_in_the_server_log(monkeypatch):
    """fd 2 in the worker is the sink now, and the thread that forwards it onward to the
    inherited stderr is a daemon a fatal signal can end before it runs.

    Those lines used to reach the server log synchronously, because fd 2 WAS the server's
    stderr. The parent replays them from the capture after the worker is gone, unredacted,
    which is right for the operator's own log on the operator's own machine.
    """
    from core.inference import orchestrator as orchestrator_module

    written = []
    monkeypatch.setattr(
        orchestrator_module.logger,
        "error",
        lambda message, *args, **kwargs: written.append(message % args if args else message),
    )

    abort = (
        "2026-09-16 10:00:01 audio_codecs.decode_bicodec: generated text: private\n"
        "terminate called after throwing an instance of 'c10::Error'\n"
        "  what():  CUDA error at /home/alice/.unsloth/studio/worker.py\n"
    )
    instance = _orchestrator_with_capture(abort)
    instance._proc = SimpleNamespace(exitcode = -6, pid = 4242, is_alive = lambda: False)
    message = instance._subprocess_crash_message("generation", with_worker_output = True)

    logged = "\n".join(written)
    assert "terminate called" in logged, logged
    # The operator's log is not the client's message: it keeps the path and the logging.
    assert "/home/alice" in logged, logged
    assert "generated text: private" in logged, logged
    # And the client's message still does not.
    assert "/home/alice" not in message, message
    assert "generated text" not in message, message

    # Once per worker, however many paths ask for the message.
    written.clear()
    instance._subprocess_crash_message("wait")
    assert written == [], written


def test_a_teardown_that_clears_the_handle_first_still_logs_the_tail(monkeypatch):
    """A concurrent teardown can clear `_proc` between the worker dying and the blocked
    generation noticing.

    The message then degrades to "process missing", which is fine, but the capture is retired
    when the next worker spawns, so returning without reading it threw the diagnostic away
    entirely -- the thing the capture was added to keep.
    """
    from core.inference import orchestrator as orchestrator_module

    written = []
    monkeypatch.setattr(
        orchestrator_module.logger,
        "error",
        lambda message, *args, **kwargs: written.append(message % args if args else message),
    )
    instance = _orchestrator_with_capture(
        "terminate called after throwing an instance of 'c10::Error'\n"
    )
    instance._proc = None
    message = instance._subprocess_crash_message("generation", with_worker_output = True)
    assert "process missing" in message
    assert any("terminate called" in line for line in written), written

    # And the exit-status-not-yet-available path, which is the same race one step later.
    written.clear()
    other = _orchestrator_with_capture("Fatal Python error: Segmentation fault\n")
    other._proc = SimpleNamespace(exitcode = None, pid = 99, is_alive = lambda: True)
    other._subprocess_crash_message("wait")
    assert any("Segmentation fault" in line for line in written), written


def test_a_logged_traceback_is_not_mistaken_for_the_crash():
    """`exc_info = True` writes the record and then the traceback, at column 0.

    `core/inference/worker.py` logs a recovered request failure that way, so the capture
    holds a `Traceback (most recent call last):` that no crash wrote. A later native abort
    or fatal signal writes no traceback of its own, and selecting the LAST header in the
    raw capture then returned the logged one as this crash: another account's exception,
    with its own frames and its own message.
    """
    logged = (
        "2026-09-16 10:00:03 worker: request 41 failed, retrying\n"
        "Traceback (most recent call last):\n"
        '  File "/home/alice/.unsloth/studio/worker.py", line 9, in handle\n'
        '    raise ValueError("another account\'s prompt was rejected")\n'
        "ValueError: another account's prompt was rejected\n"
    )
    abort = "Fatal Python error: Aborted\n\nCurrent thread 0x00007f1a (most recent call first):\n"
    public = _orchestrator_with_capture(logged + abort)._public_worker_stderr_tail()
    assert "another account" not in public, public
    assert "ValueError" not in public, public
    # And the crash that did happen is still the message.
    assert "Fatal Python error: Aborted" in public, public


def test_a_runtimes_own_traceback_after_a_log_line_is_still_reported():
    """The other direction. Only a header DIRECTLY under a log record is that record's;
    one the runtime wrote after the logging stack had finished a line is the crash, and
    dropping it would cost the report its only explanation."""
    text = "2026-09-16 10:00:02 worker: request 41 finished\n\n" + TRACEBACK
    public = _orchestrator_with_capture(text)._public_worker_stderr_tail()
    assert "RuntimeError: boom" in public, public
    assert "request 41 finished" not in public, public


def test_a_path_component_with_punctuation_in_it_is_still_redacted():
    """The component list was a whitelist of the characters a filename usually has, so an
    apostrophe, a colon or a bracket stopped the match partway and left the rest of the
    path -- the account name with it -- in a message that leaves the host."""
    from core.inference.orchestrator import _redact_worker_output

    for path in (
        "/home/o'connor/private/model.py",
        "/home/a(1)/private/model.py",
        "/srv/models/llama:8b/weights.gguf",
        "C:\\Users\\O'Brien\\models\\weights.gguf",
        "\\\\share\\team\\o'brien\\model.gguf",
    ):
        public = _redact_worker_output(f'  File "{path}", line 1, in run\n')
        assert "private" not in public, public
        assert "o'connor" not in public.lower(), public
        assert "o'brien" not in public.lower(), public
        assert "llama:8b" not in public, public
        assert public.strip().startswith('File ".../'), public

    # A path with a space in it keeps working, and a path mid-sentence does not swallow
    # the words after it or the second path on the same line.
    spaced = _redact_worker_output("C:\\Program Files\\unsloth\\weights.gguf failed\n")
    assert spaced.strip() == ".../weights.gguf failed", spaced
    two = _redact_worker_output('  File "/a/b.py", line 1, then /etc/passwd here\n')
    assert "/etc/passwd" not in two, two
    assert "line 1, then" in two, two
    # And what was never a path is left as it was written.
    assert _redact_worker_output("a ratio of 3/4 at https://host/path/x\n").strip() == (
        "a ratio of 3/4 at https://host/path/x"
    )


MARK = "    | "


def test_the_worker_marks_every_continuation_line_of_a_record():
    """A record is prefixed on its first line only, so everything after it is bare content
    at column 0 -- including the traceback `exc_info = True` appends. The writer is the only
    place that knows which it is, so the writer says so."""
    import logging
    from utils.worker_stderr import LOG_RECORD_CONTINUATION_PREFIX, mark_log_record_continuations

    assert LOG_RECORD_CONTINUATION_PREFIX == MARK

    logger_object = logging.getLogger("unsloth-test-marking")
    logger_object.handlers = []
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger_object.addHandler(handler)

    # cover_later_handlers off: the process-wide half is exercised on its own below, and
    # leaving it off here keeps this case from editing the interpreter's logging for the
    # rest of the session.
    assert mark_log_record_continuations(logger_object, cover_later_handlers = False) == 1
    # Idempotent: a second call after a reconfiguration must not double the prefix.
    assert mark_log_record_continuations(logger_object, cover_later_handlers = False) == 0

    record = logger_object.makeRecord(
        "unsloth-test-marking",
        logging.ERROR,
        __file__,
        1,
        "generated text: line one\nRuntimeError: not really\nTraceback (most recent call last):",
        (),
        None,
    )
    formatted = handler.formatter.format(record)
    lines = formatted.split("\n")
    assert lines[0].startswith("ERROR unsloth-test-marking:"), lines
    assert all(line.startswith(MARK) for line in lines[1:]), lines

    single = logger_object.makeRecord(
        "unsloth-test-marking",
        logging.ERROR,
        __file__,
        1,
        "one line",
        (),
        None,
    )
    assert "\n" not in handler.formatter.format(single)


def test_a_logged_traceback_is_not_returned_when_the_worker_dies_silently():
    """The case a later diagnostic cannot rescue. One request raises a handled error, its
    traceback is logged, the worker keeps running, and a LATER request ends it by SIGKILL
    or the OOM killer, which write nothing at all. The last traceback in the capture is
    then the earlier request's, and returning it hands another account's exception text and
    frames to whoever asked second."""
    logged = (
        "2026-09-16 10:00:03 worker: Generation error: another account's prompt was rejected\n"
        + MARK
        + "Traceback (most recent call last):\n"
        + MARK
        + '  File "/home/alice/.unsloth/studio/worker.py", line 9, in handle\n'
        + MARK
        + "ValueError: another account's prompt was rejected\n"
    )
    public = _orchestrator_with_capture(logged)._public_worker_stderr_tail()
    assert public == "", public


def test_a_marked_continuation_that_reads_like_a_diagnostic_is_still_content():
    """`decode_bicodec` logs the first 500 characters of generated text, and generated text
    can contain anything -- `RuntimeError:`, `Fatal Python error:`, `Killed`. Reclassifying
    a continuation by what it says is what let that content out."""
    content = (
        "2026-09-16 10:00:01 audio_codecs.decode_bicodec: generated text:\n"
        + MARK
        + "RuntimeError: another account's private prompt\n"
        + MARK
        + "Fatal Python error: also theirs\n"
        + MARK
        + "Killed\n"
    )
    public = _orchestrator_with_capture(content)._public_worker_stderr_tail()
    assert public == "", public
    assert "another account" not in public


def test_a_crash_after_a_marked_record_is_still_the_crash():
    """The marking is what makes the record self-delimiting: its continuations are marked,
    so the first UNMARKED line after it is not a continuation and is classified on its own.
    Without that the strict reading would have to drop a real traceback written straight
    after a log line, which is most of them."""
    text = (
        "2026-09-16 10:00:01 audio_codecs.decode_bicodec: generated text:\n"
        + MARK
        + "another account's private prompt\n"
        + TRACEBACK
    )
    public = _orchestrator_with_capture(text)._public_worker_stderr_tail()
    assert "another account" not in public, public
    assert "RuntimeError: boom" in public, public


def test_the_worker_installs_the_marking_where_its_logging_is_configured():
    """The filter above is only sound because the writer marks. A worker that configured
    logging and never called this would leave every case here passing and the product
    unchanged."""
    import inspect
    from core.inference import worker as worker_module

    source = inspect.getsource(worker_module)
    assert "mark_log_record_continuations()" in source
    setup = source.index("LogConfig.setup_logging(")
    assert 0 < setup < source.index("mark_log_record_continuations()")


def test_a_marked_record_under_a_diagnostic_is_not_adopted_by_it():
    """Indentation alone cannot carry this. An indented line CONTINUES whatever is open, and
    `  what():  CUDA error: ...` under an abort genuinely does, so a logged line that merely
    happens to be indented was being adopted by the abort above it and sent out with it. The
    marker says what the indentation only suggests."""
    text = (
        "terminate called after throwing an instance of 'c10::Error'\n"
        + MARK
        + "another account's private prompt\n"
        "  what():  CUDA error: device-side assert triggered\n"
    )
    public = _orchestrator_with_capture(text)._public_worker_stderr_tail()
    assert "another account" not in public, public
    # And the abort itself, with its own genuine continuation, is still the message.
    assert "terminate called" in public, public
    assert "CUDA error: device-side assert triggered" in public, public


def test_a_live_replacements_own_crash_is_still_written_to_the_log(monkeypatch):
    """A stream whose worker was REPLACED reaches the crash message against the replacement.

    That replacement is healthy, so its exitcode is None, and marking its capture as replayed
    there spent the one replay on a live worker: when the replacement itself later died, the
    operator's log skipped its tail as already written -- and that tail is the fatal
    diagnostic the daemon forwarding thread may never have reached the log with, which is the
    whole reason the capture exists.
    """
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(
        "inference_orchestrator_stderr_log_once_under_test",
        Path(_BACKEND_DIR) / "core/inference/orchestrator.py",
    )
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)

    written: "list[tuple]" = []
    monkeypatch.setattr(
        module.logger,
        "error",
        lambda *args, **kwargs: written.append(args),
        raising = False,
    )

    capture = _FixedCapture("Fatal Python error: Aborted\n")
    orchestrator = module.InferenceOrchestrator.__new__(module.InferenceOrchestrator)
    orchestrator._stderr_capture = capture

    # The old stream, against a replacement that is running.
    orchestrator._proc = SimpleNamespace(pid = 4242, exitcode = None)
    orchestrator._subprocess_crash_message("generation")
    assert len(written) == 1, written
    # A second such call does not repeat it: this is still once per worker.
    orchestrator._subprocess_crash_message("generation")
    assert len(written) == 1, written

    # Now the replacement itself dies, with more in its capture than before.
    orchestrator._proc = SimpleNamespace(pid = 4242, exitcode = -6)
    orchestrator._subprocess_crash_message("generation", with_worker_output = True)
    assert len(written) == 2, "the replacement's own fatal output was skipped as already logged"
    # And THAT one is final: the exit has been reported, so nothing replays it again.
    orchestrator._subprocess_crash_message("generation", with_worker_output = True)
    assert len(written) == 2, written


def test_a_worker_that_died_between_requests_is_replayed_before_its_sink_closes(monkeypatch):
    """A native fault between requests has no waiter.

    Nothing reaches `_subprocess_crash_message`, so the replay that exists for a forwarding
    thread the signal ended never runs, and the next load retired the capture -- closing the
    only remaining copy of the cause. What the user saw was the load's own liveness check:
    "Inference subprocess is not running", and nothing about why.
    """
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(
        "inference_orchestrator_idle_retire_under_test",
        Path(_BACKEND_DIR) / "core/inference/orchestrator.py",
    )
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)

    written: "list[tuple]" = []
    monkeypatch.setattr(
        module.logger, "error", lambda *args, **kwargs: written.append(args), raising = False,
    )

    closed: "list[bool]" = []

    class _ClosableCapture(_FixedCapture):
        def close(self):
            closed.append(True)

    orchestrator = module.InferenceOrchestrator.__new__(module.InferenceOrchestrator)
    orchestrator._stderr_capture = _ClosableCapture(
        "Fatal Python error: Segmentation fault\n"
    )
    orchestrator._proc = SimpleNamespace(
        pid = 7331, exitcode = -11, is_alive = lambda: False,
    )

    orchestrator._retire_stderr_capture()
    assert closed == [True]
    assert written, "the capture of a worker that died unattended was closed unread"
    assert "Fatal Python error: Segmentation fault" in str(written[0])
    assert orchestrator._stderr_capture is None

    # A worker that is still running has said nothing final, and its sink is retired without
    # being treated as a crash.
    written.clear()
    orchestrator._stderr_capture = _ClosableCapture("loading shards: 40%\n")
    orchestrator._proc = SimpleNamespace(pid = 7332, exitcode = None, is_alive = lambda: True)
    orchestrator._retire_stderr_capture()
    assert written == [], written


def test_a_request_queued_behind_the_crash_is_not_given_its_last_words(monkeypatch):
    """Compare mode keeps several mailboxes in flight while the subprocess runs the commands
    one at a time.

    When it dies, every waiting stream reaches the crash message, and the tail belongs to
    whichever request was EXECUTING -- its traceback, its exception text, on a shared install
    another account's. A request that never started gets the exit status and nothing else.
    """
    capture = _FixedCapture(
        "Traceback (most recent call last):\n"
        '  File "/home/alice/.unsloth/studio/worker.py", line 42, in run\n'
        "RuntimeError: another account's prompt overflowed the context\n"
    )
    orchestrator = _orchestrator_with(-9, capture)
    logged: "list[str]" = []
    monkeypatch.setattr(
        type(orchestrator),
        "_log_worker_stderr_once",
        lambda self, pid, exitcode, **_kwargs: logged.append(str(pid)),
        raising = False,
    )

    executing = orchestrator._subprocess_crash_message("generation", with_worker_output = True)
    assert "Worker error output:" in executing
    assert "RuntimeError: another account" in executing

    orchestrator._stderr_tail_logged = None
    queued = orchestrator._subprocess_crash_message("generation", with_worker_output = False)
    assert "Worker error output:" not in queued, queued
    assert "another account" not in queued, queued
    # Still a real report: the context and the exit status are what it always had.
    assert "generating a response" in queued
    # The exit status, in whichever spelling this platform gives it: `signal.Signals(9)`
    # raises on Windows, so the message says SIG9 there and SIGKILL everywhere else. The
    # assertion is about the status still being reported, not about the wording, and naming
    # only the POSIX one is how this file has already failed a Windows leg once.
    assert "signal=SIGKILL" in queued or "signal=SIG9" in queued, queued
    assert "exitcode=-9" in queued, queued
    # And the operator's copy is written either way -- the narrowing is about the wire.
    assert logged, "the server log lost the tail for the queued request's path"


def test_the_stream_asks_who_owned_the_worker_before_handing_over_the_tail():
    """The narrowing is only worth anything if the call site applies it, and `_owns_worker`
    is the same test a Stop goes through, for the same reason: claimed but queued is not
    executing. It answers True when nothing is in flight, so an ordinary single-request
    crash is unchanged."""
    import inspect
    from core.inference import orchestrator as orchestrator_module

    body = inspect.getsource(orchestrator_module.InferenceOrchestrator._consume_token_stream)
    # One of the two crash exits. The other, the branch that fires when the worker was
    # SWAPPED under this stream, passes no tail at all: see the test below.
    assert body.count("with_worker_output = self._owns_worker(cancel_event)") == 1, body


def test_a_queued_compare_request_does_not_own_the_worker():
    """The property the narrowing rests on, pinned here so a change to the claim
    bookkeeping cannot quietly widen it again."""
    module = _load_orchestrator_module()
    orchestrator = module.InferenceOrchestrator.__new__(module.InferenceOrchestrator)
    import threading

    orchestrator._active_cancel_lock = threading.Lock()
    first, second = threading.Event(), threading.Event()
    orchestrator._active_cancel_events = []
    orchestrator._executing_cancel_events = []
    # Nothing in flight: an ordinary single-request crash still gets its tail.
    assert orchestrator._owns_worker(first) is True

    orchestrator._active_cancel_events = [first, second]
    orchestrator._executing_cancel_events = [first]
    assert orchestrator._owns_worker(first) is True
    assert orchestrator._owns_worker(second) is False


def test_the_public_tail_redacts_the_credentials_a_crash_actually_carries():
    """`scrub_secrets` knows an HF token and a bearer value, which is what a DOWNLOAD carries.

    A crash diagnostic carries whatever the process had in scope, and this string is returned
    to the client verbatim on a managed install, so the gap between what a log may hold and
    what a client may see had to be closed with the reader this repository already has for
    the rest of them.
    """
    from core.inference.orchestrator import _redact_worker_output

    text = (
        "Traceback (most recent call last):\n"
        '  File "/home/alice/.unsloth/studio/worker.py", line 42, in run\n'
        "    client = OpenAI()  # OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789\n"
        "    aws = 'AKIAIOSFODNN7EXAMPLE'\n"
        "    gh = 'ghp_abcdefghijklmnopqrstuvwxyz0123456789'\n"
        "    password=hunter2seventeen\n"
        "RuntimeError: refused\n"
    )
    public = _redact_worker_output(text)
    for secret in (
        "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "hunter2seventeen",
    ):
        assert secret not in public, (secret, public)
    # And the diagnosis survives it, which is the whole reason the tail is returned.
    assert "RuntimeError: refused" in public, public
    assert "worker.py" in public, public


def test_the_worker_output_is_opt_in_at_every_call_site():
    """A default of True gave the tail to call sites that take no part in the claim
    bookkeeping -- `count_chat_tokens` uses an addressed mailbox alongside compare-mode
    generations -- so a count queued behind another account's generation was handed that
    generation's traceback. Opting in is the only safe direction: forgetting it costs a
    diagnostic, forgetting the other one discloses somebody else's."""
    import inspect
    from core.inference import orchestrator as orchestrator_module

    source = inspect.getsource(orchestrator_module)
    assert "with_worker_output: bool = False" in source
    # Every call site decides from ownership rather than from a constant.
    calls = source.count("self._subprocess_crash_message(")
    owned = source.count("with_worker_output = self._owns_worker(")
    # Every call site but one asks ownership; the exception is the swapped-worker branch,
    # which asks for no tail at all because the worker it would read is not the one this
    # stream was latched to.
    assert calls == owned + 1, (calls, owned)


def test_a_stream_whose_worker_was_swapped_gets_no_tail_at_all():
    """`initial_proc` is latched for a reason: this branch fires when the worker underneath
    the stream has been replaced.

    `_subprocess_crash_message` reads `self._proc` and `self._stderr_capture`, which are the
    REPLACEMENT's, and a shutdown clears the ownership lists, so `_owns_worker` answers True
    for a request that owns nothing. A stale reader would be handed the traceback of a
    generation that started after it -- on a shared install, somebody else's.
    """
    import inspect
    from core.inference import orchestrator as orchestrator_module

    body = inspect.getsource(orchestrator_module.InferenceOrchestrator._consume_token_stream)
    swap = body.index("initial_proc or self._resp_queue is not initial_resp_queue")
    following = body[swap : body.index("resp = read_one(read_timeout)", swap)]
    assert "_subprocess_crash_message(crash_context)" in following, following
    assert "with_worker_output" not in following, following


@pytest.fixture
def _logging_restored():
    """Undo the process-wide half of the marking, whatever the case below does with it.

    It patches `logging.Handler.setFormatter`, `logging.Logger.addHandler` and the formatter
    on the interpreter's shared `lastResort` handler, none of which belong to one test.
    """
    import logging

    from utils import worker_stderr

    # Read with getattr so a build without the hook restores what it can and the CASE is
    # what reports the gap, rather than this fixture erroring before the assertions run.
    saved = (
        logging.Handler.setFormatter,
        logging.Logger.addHandler,
        getattr(logging.lastResort, "formatter", None),
        getattr(worker_stderr, "_UNHOOKED_SET_FORMATTER", None),
        getattr(worker_stderr, "_UNHOOKED_ADD_HANDLER", None),
    )
    try:
        yield
    finally:
        logging.Handler.setFormatter = saved[0]
        logging.Logger.addHandler = saved[1]
        logging.lastResort.formatter = saved[2]
        if hasattr(worker_stderr, "_UNHOOKED_SET_FORMATTER"):
            worker_stderr._UNHOOKED_SET_FORMATTER = saved[3]
            worker_stderr._UNHOOKED_ADD_HANDLER = saved[4]


def test_the_marking_covers_the_last_resort_handler(_logging_restored):
    """The worker's structlog setup adds no root handler, so ordinary stdlib logging lands on
    `logging.lastResort` -- which is not in anyone's `handlers` list and was therefore never
    marked. A library's recovered traceback went out bare at column 0 through it, which is
    exactly what the tail filter reads as a crash."""
    import logging

    from utils.worker_stderr import mark_log_record_continuations

    logging.lastResort.setFormatter(logging.Formatter("%(message)s"))
    mark_log_record_continuations(logging.getLogger("unsloth-test-lastresort"))

    record = logging.LogRecord(
        "unsloth-test-lastresort", logging.ERROR, __file__, 1, "one\ntwo\nthree", (), None
    )
    lines = logging.lastResort.format(record).split("\n")
    assert lines[0] == "one", lines
    assert all(line.startswith(MARK) for line in lines[1:]), lines


def test_a_handler_installed_after_startup_is_marked_too(_logging_restored):
    """The worker configures logging and then imports the ML stack, which adds handlers of
    its own on import. Marking only what existed at that instant left those unmarked."""
    import logging

    from utils.worker_stderr import mark_log_record_continuations

    mark_log_record_continuations(logging.getLogger("unsloth-test-late"))

    late = logging.getLogger("unsloth-test-late-library")
    late.handlers = []
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    late.addHandler(handler)

    record = logging.LogRecord(
        "unsloth-test-late-library", logging.ERROR, __file__, 1, "one\ntwo", (), None
    )
    assert handler.format(record).split("\n")[1].startswith(MARK), handler.format(record)

    # And a library that sets its formatter AFTER the handler is installed does not undo it.
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    formatted = handler.format(record)
    assert formatted.split("\n")[0] == "ERROR one", formatted
    assert formatted.split("\n")[1].startswith(MARK), formatted

    # Clearing the formatter falls back to logging's default, which marks nothing, so that
    # is the one case the hook cannot wrap -- and it must not crash on it either.
    handler.setFormatter(None)
    assert handler.formatter is None


def test_a_root_level_path_is_redacted_too():
    """One component after the root is still an absolute path.

    The expression required a separator AFTER a component, so `/model.gguf`, `C:\\model.gguf`
    and `\\\\server\\share` matched nothing at all and went to the client verbatim. Nothing else
    covers them: the earlier redactors handle registered native paths and recognised secrets,
    and a crash diagnostic naming a file at the root is neither.
    """
    from core.inference.orchestrator import _redact_worker_output

    for path, tail in (
        ("/model.gguf", "model.gguf"),
        ("C:\\model.gguf", "model.gguf"),
        ("\\\\fileserver\\share", "share"),
        ("/opt", "opt"),
    ):
        public = _redact_worker_output(f"could not open {path}\n")
        # The tail survives by design -- a diagnostic still has to name the file -- so what
        # must be gone is everything that locates it on this host.
        assert public.strip() == f"could not open .../{tail}", public
        assert not public.count("C:"), public

    # And what is not a path is still not one: a lone separator, a ratio, a URL and a word
    # with a slash in it are returned as they were written.
    for text in (
        'the "/" separator is not a path',
        "a ratio of 3/4 at https://host/path/x",
        "use / to split and/or join",
    ):
        assert _redact_worker_output(text + "\n").strip() == text, text


def test_two_paths_on_one_line_are_two_matches():
    """A component may hold a space, and it must not spend that on the next path's root.

    `copy C:\\x\\old to C:\\y\\new` let one component be `old to C:` -- a space and a colon are
    both legal in a filename -- so the pair matched as ONE path and the message came out as
    `copy .../new`, with the source file and the operation deleted from a crash diagnostic
    that had just been made visible. Copy, move and cache-resolution failures name two paths
    like this routinely.
    """
    from core.inference.orchestrator import _redact_worker_output

    windows = _redact_worker_output("copy C:\\Users\\ann\\old.gguf to C:\\tmp\\new.gguf\n")
    assert windows.strip() == "copy .../old.gguf to .../new.gguf", windows
    posix = _redact_worker_output("cannot copy /home/ann/old.gguf to /tmp/new.gguf\n")
    assert posix.strip() == "cannot copy .../old.gguf to .../new.gguf", posix
    unc = _redact_worker_output("copy \\\\share\\team\\old.gguf to \\\\other\\team\\new.gguf\n")
    assert unc.strip() == "copy .../old.gguf to .../new.gguf", unc

    # And a single path with spaces in it is still ONE match, which is what the space in a
    # component is for.
    spaced = _redact_worker_output("C:\\Program Files\\unsloth\\weights.gguf failed\n")
    assert spaced.strip() == ".../weights.gguf failed", spaced
