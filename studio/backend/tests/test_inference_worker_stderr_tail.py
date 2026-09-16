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

    message = _orchestrator_with(1, capture)._subprocess_crash_message("generation")

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

    message = _orchestrator_with(-9, capture)._subprocess_crash_message("wait")

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
        message = _orchestrator_with(1, capture)._subprocess_crash_message("generation")
        assert message == (
            "The inference worker stopped unexpectedly while generating a response. "
            "Details: pid=6145, exitcode=1."
        )


def test_a_worker_still_running_is_not_given_a_tail(tmp_path):
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    Path(capture.path).write_bytes(b"still going\n")
    message = _orchestrator_with(None, capture)._subprocess_crash_message("generation")
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
    message = orchestrator._subprocess_crash_message("generation")

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
    )._subprocess_crash_message("generation")
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
