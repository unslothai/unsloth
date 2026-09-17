# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth#7843: a worker that exits without answering must not lose its own error output."""

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


def test_a_windows_worker_arrives_as_crlf_cp1252_and_decodes_cleanly():
    raw = "Traceback (most recent call last):\r\n  File “worker.py”, line 1\r\n".encode(
        "cp1252",
    )
    assert b"\r\n" in raw
    text = decode_worker_stderr(raw)
    assert "\r" not in text
    assert text.splitlines()[0] == "Traceback (most recent call last):"
    assert "“worker.py”" in text


def test_a_utf8_worker_is_never_mis_read_as_cp1252():
    raw = "RuntimeError: — out of memory é\n".encode("utf-8")
    assert decode_worker_stderr(raw) == "RuntimeError: — out of memory é\n"


def test_cp1252_bytes_that_are_not_valid_utf8_still_decode():
    raw = b"warning: caf\xe9 layer skipped\r\n"
    assert decode_worker_stderr(raw) == "warning: café layer skipped\n"


def test_a_cp1252_character_at_the_very_end_is_not_trimmed_away():
    assert decode_worker_stderr(b"RuntimeError: caf\xe9") == "RuntimeError: café"
    assert decode_worker_stderr("é at the start".encode("utf-8")[1:]).endswith(" at the start")
    truncated = "RuntimeError: é".encode("utf-8")[:-1]
    assert decode_worker_stderr(truncated, ends_at_eof = False) == "RuntimeError: "
    assert decode_worker_stderr(truncated) == "RuntimeError: \u00c3"
    assert decode_worker_stderr("done é".encode("utf-8")) == "done é"


def test_undecodable_bytes_are_replaced_rather_than_dropped():
    # 0x81 and 0x90 are undefined in cp1252 and invalid as UTF-8.
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


def test_the_tail_keeps_the_last_lines_and_drops_blank_ones():
    raw = ("\n".join(f"line {index}" for index in range(100)) + "\n\n\n").encode("utf-8")
    tail = stderr_tail_from_bytes(raw, max_lines = 3)
    assert tail == "line 97\nline 98\nline 99"


def test_the_character_cap_trims_from_the_front_and_never_shows_a_half_line():
    raw = ("first line that is quite long\n" + "b" * 200 + "\nlast line\n").encode("utf-8")
    tail = stderr_tail_from_bytes(raw, max_lines = 10, max_chars = 60)
    assert tail.endswith("last line")
    assert len(tail) <= 60
    assert not tail.startswith("b")


def test_an_empty_sink_produces_no_tail():
    assert stderr_tail_from_bytes(b"") == ""
    assert stderr_tail_from_bytes(b"\n  \n\r\n") == ""


def test_a_capture_whose_file_is_gone_reports_no_tail_instead_of_raising():
    capture = WorkerStderrCapture(prefix = "unsloth-test-")
    capture.close()
    assert capture.tail() == ""
    # Idempotent: a Windows host takes a second close when the child still holds it.
    capture.close()


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
    assert "\r" not in message


def test_a_signalled_exit_keeps_its_hint_and_gains_the_tail(tmp_path):
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    Path(capture.path).write_bytes(b"terminate called after throwing an instance of 'c10::Error'\n")

    message = _orchestrator_with(-9, capture)._subprocess_crash_message(
        "wait", with_worker_output = True
    )

    assert "exitcode=-9" in message
    assert "c10::Error" in message

    # POSIX only: a Windows process never reports a negative code and signal.Signals(9)
    # raises there, so that branch names the signal SIG9 and offers no hint.
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
    # Read from the source: exercising it for real needs a model load.
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


def _spawn(
    function_name,
    mirror_path,
    timeout = 180,
):
    from utils.native_path_leases import run_without_native_path_secret

    # Bare name, with this directory on sys.path: a dotted name finds the repository's own
    # top-level "tests" package instead. Spawn hands the child the parent's sys.path.
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
    assert "the last line before the worker died" in capture.tail(max_lines = 200)


_STALLED_READER_CHILD = r"""
import os, sys
sys.path.insert(0, %(backend)r)
from utils.worker_stderr import MIRROR_FILE_CAP_BYTES, install_worker_stderr_mirror

assert install_worker_stderr_mirror(%(path)r) is True
line = b"x" * 1023 + b"\n"
for _ in range((MIRROR_FILE_CAP_BYTES * 32) // len(line)):
    os.write(2, line)
os.write(2, b"the last line before the worker died\n")
"""


def test_the_sink_is_bounded_even_when_nobody_drains_the_operators_stderr(tmp_path):
    """The mirrored copy is a courtesy; the cap is not. 68 MB landed here when a full pipe
    parked the pump inside os.write and the cap counted bytes relayed rather than bytes written."""
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    script = _STALLED_READER_CHILD % {"backend": _BACKEND_DIR, "path": capture.path}
    read_end, write_end = os.pipe()
    try:
        process = subprocess.Popen([sys.executable, "-c", script], stderr = write_end)
        os.close(write_end)
        write_end = None
        # Nothing ever reads `read_end`, which is what a stalled log consumer looks like.
        assert process.wait(timeout = 300) == 0
    finally:
        if write_end is not None:
            os.close(write_end)
        os.close(read_end)

    size = os.path.getsize(capture.path)
    assert size <= 2 * MIRROR_FILE_CAP_BYTES + 65536, size
    assert "the last line before the worker died" in capture.tail(max_lines = 200)
    capture.close()


def test_a_caller_that_passes_no_mirror_is_unchanged(tmp_path):
    process = _spawn("exit_one_after_writing_to_stderr", None)
    assert process.exitcode == 1
    assert _orchestrator_with(process.exitcode, None, pid = process.pid)._subprocess_crash_message(
        "generation",
    ) == (
        "The inference worker stopped unexpectedly while generating a response. "
        f"Details: pid={process.pid}, exitcode=1."
    )


def test_a_server_that_exits_leaves_no_sink_behind(tmp_path):
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
    from utils.worker_stderr import TAIL_READ_BYTES

    line = "RuntimeError: modellädt nicht — CUDA out of memory\n".encode("utf-8")
    body = line * 30
    window = b"\xa9" + b"." * (TAIL_READ_BYTES - 1 - len(body)) + body
    assert len(window) == TAIL_READ_BYTES
    with pytest.raises(UnicodeDecodeError):
        window.decode("utf-8")

    tail = stderr_tail_from_bytes(window)
    assert "modellädt nicht — CUDA out of memory" in tail, tail
    assert "Ã¤" not in tail and "â€”" not in tail, tail


def test_a_severed_character_at_the_end_of_the_window_is_dropped_too():
    raw = "RuntimeError: café blew up — ".encode("utf-8") + "é".encode("utf-8")[:1]
    text = decode_worker_stderr(raw)
    assert "café blew up —" in text, text
    assert "Ã©" not in text, text


def test_a_real_worker_whose_output_outgrows_the_read_window_still_reads_cleanly(tmp_path):
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    process = _spawn("write_non_ascii_far_past_the_read_window", capture.path, timeout = 60)
    assert process.exitcode == 1
    tail = capture.tail(max_lines = 5)
    assert "modellädt nicht — CUDA out of memory" in tail, tail
    assert "Ã¤" not in tail, tail
    capture.close()


def test_a_missing_sink_is_never_re_created_by_the_child(tmp_path):
    from utils.worker_stderr import install_worker_stderr_mirror

    missing = tmp_path / "already-retired.stderr"
    assert install_worker_stderr_mirror(str(missing)) is False
    assert not missing.exists(), "the child created a sink the parent had retired"


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason = "O_NOFOLLOW is POSIX only")
def test_a_symlink_at_the_sink_path_is_not_followed(tmp_path):
    from utils.worker_stderr import install_worker_stderr_mirror

    victim = tmp_path / "victim.txt"
    victim.write_text("ORIGINAL\n", encoding = "utf-8")
    link = tmp_path / "sink.stderr"
    link.symlink_to(victim)

    assert install_worker_stderr_mirror(str(link)) is False
    assert victim.read_text(encoding = "utf-8") == "ORIGINAL\n"


def test_a_worker_holding_a_second_handle_on_stderr_still_exits_promptly(tmp_path):
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    process = _spawn("hold_a_second_handle_on_stderr_then_exit", capture.path, timeout = 60)
    assert process.exitcode == 1
    assert "held a second handle on stderr" in capture.tail(), capture.tail()
    capture.close()

    # The race cannot be asserted from outside: a run that does not hit it proves nothing.
    source = (Path(_BACKEND_DIR) / "utils/worker_stderr.py").read_text(encoding = "utf-8")
    teardown = source.split("def _stop_mirror(", 1)[1].split("\ndef ", 1)[0]
    close_at = teardown.index("os.close(inherited_fd)")
    # Both threads write to the inherited descriptor, so both must be done with it.
    for guard in ("pump.is_alive()", "relay.is_alive()"):
        assert guard in teardown[:close_at], (
            "the teardown closes the inherited stderr without first checking that the "
            f"thread behind {guard} has finished with it"
        )


@pytest.mark.skipif(not hasattr(os, "fork"), reason = "fork is POSIX only")
def test_a_forked_child_does_not_retire_its_parents_live_sink(tmp_path):
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    Path(capture.path).write_bytes(b"the parent is still writing here\n")

    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
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
    source = (Path(_BACKEND_DIR) / "utils/worker_stderr.py").read_text(encoding = "utf-8")
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


# Spaces and non-ASCII are what break code that builds paths by string, and a Windows
# profile routinely has both.
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
    capture = WorkerStderrCapture(directory = str(tmp_path), prefix = "unsloth-test-")
    _spawn("exit_one_after_writing_to_stderr", capture.path)
    captured = capfd.readouterr()
    assert "Starting text generation" in captured.err
    assert "CUDA out of memory" in captured.err


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
    public = _orchestrator_with_capture(CROSS_ACCOUNT + TRACEBACK)._public_worker_stderr_tail()
    assert "another account" not in public, public
    assert "request 41 finished" not in public, public
    assert public.startswith("Traceback (most recent call last):"), public
    assert "RuntimeError: boom" in public


def test_the_public_tail_keeps_only_the_last_traceback():
    earlier = TRACEBACK.replace("boom", "an earlier unrelated failure")
    public = _orchestrator_with_capture(
        earlier + CROSS_ACCOUNT + TRACEBACK
    )._public_worker_stderr_tail()
    assert "an earlier unrelated failure" not in public, public
    assert "RuntimeError: boom" in public


def test_a_capture_of_nothing_but_logging_is_dropped_entirely():
    assert _orchestrator_with_capture(CROSS_ACCOUNT)._public_worker_stderr_tail() == ""
    assert _orchestrator_with_capture("")._public_worker_stderr_tail() == ""


def test_a_native_abort_with_no_traceback_is_still_reported():
    abort = "terminate called after throwing an instance of 'c10::Error'\n"
    public = _orchestrator_with_capture(CROSS_ACCOUNT + abort)._public_worker_stderr_tail()
    assert "c10::Error" in public, public
    assert "another account" not in public, public


def test_the_log_record_shapes_are_the_ones_content_arrives_in():
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
    import inspect

    from core.inference.orchestrator import InferenceOrchestrator

    source = inspect.getsource(InferenceOrchestrator._subprocess_crash_message)
    assert "_public_worker_stderr_tail()" in source
    assert "= self._worker_stderr_tail()" not in source


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
    from utils import worker_stderr

    source = Path(worker_stderr.__file__).read_text(encoding = "utf-8")
    install = source.split("def install_worker_stderr_mirror(", 1)[1]
    assert "os.pipe()" not in install, "fd 2 is a pipe again"
    assert "_open_sink_for_append" in install
    assert "os.dup2(writer_fd, 2)" in install
    # Compaction rewrites the file from the front while fd 2 still points at it.
    append = source.split("def _open_sink_for_append(", 1)[1].split("\ndef ", 1)[0]
    assert "os.O_APPEND" in append


MULTILINE_CROSS_ACCOUNT = (
    "2026-09-16 10:00:01 audio_codecs.decode_bicodec: generated text: the first line\n"
    "another account's second line\n"
    "and a third line of the same prompt\n"
)


def test_a_logged_message_cannot_leak_through_its_own_continuation_lines():
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
    # Both windows: the append during the tail READ and the one during the REWRITE.
    assert b"terminate called" in kept, kept[-300:]
    assert b"device-side assert" in kept, kept[-300:]
    assert len(kept) <= 1024 + 256, len(kept)


def test_the_operator_still_gets_the_last_words_in_the_server_log(monkeypatch):
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
    assert "/home/alice" in logged, logged
    assert "generated text: private" in logged, logged
    assert "/home/alice" not in message, message
    assert "generated text" not in message, message

    written.clear()
    instance._subprocess_crash_message("wait")
    assert written == [], written


def test_a_teardown_that_clears_the_handle_first_still_logs_the_tail(monkeypatch):
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

    written.clear()
    other = _orchestrator_with_capture("Fatal Python error: Segmentation fault\n")
    other._proc = SimpleNamespace(exitcode = None, pid = 99, is_alive = lambda: True)
    other._subprocess_crash_message("wait")
    assert any("Segmentation fault" in line for line in written), written


def test_a_logged_traceback_is_not_mistaken_for_the_crash():
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
    assert "Fatal Python error: Aborted" in public, public


def test_a_runtimes_own_traceback_after_a_log_line_is_still_reported():
    text = "2026-09-16 10:00:02 worker: request 41 finished\n\n" + TRACEBACK
    public = _orchestrator_with_capture(text)._public_worker_stderr_tail()
    assert "RuntimeError: boom" in public, public
    assert "request 41 finished" not in public, public


def test_a_path_component_with_punctuation_in_it_is_still_redacted():
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

    spaced = _redact_worker_output("C:\\Program Files\\unsloth\\weights.gguf failed\n")
    assert spaced.strip() == ".../weights.gguf failed", spaced
    two = _redact_worker_output('  File "/a/b.py", line 1, then /etc/passwd here\n')
    assert "/etc/passwd" not in two, two
    assert "line 1, then" in two, two
    assert _redact_worker_output("a ratio of 3/4 at https://host/path/x\n").strip() == (
        "a ratio of 3/4 at https://host/path/x"
    )


MARK = "    | "
START_MARK = "\x1f"


def test_the_worker_marks_every_continuation_line_of_a_record():
    import logging
    from utils.worker_stderr import LOG_RECORD_CONTINUATION_PREFIX, mark_log_record_continuations

    assert LOG_RECORD_CONTINUATION_PREFIX == MARK

    logger_object = logging.getLogger("unsloth-test-marking")
    logger_object.handlers = []
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger_object.addHandler(handler)

    # cover_later_handlers off so this case does not edit the interpreter's logging for
    # the rest of the session; the process-wide half is exercised on its own below.
    assert mark_log_record_continuations(logger_object, cover_later_handlers = False) == 1
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
    assert lines[0].startswith(START_MARK + "ERROR unsloth-test-marking:"), lines
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
    one_line = handler.formatter.format(single)
    assert "\n" not in one_line
    # A single-line record through a default formatter is otherwise the same bytes a dying
    # runtime writes.
    assert one_line.startswith(START_MARK), repr(one_line)


def test_a_default_formatter_record_is_not_mistaken_for_a_crash():
    logged = START_MARK + "RuntimeError: another account's prompt was rejected\n"
    assert _orchestrator_with_capture(logged)._public_worker_stderr_tail() == ""

    public = _orchestrator_with_capture(
        logged
        + "Fatal Python error: Aborted\n"
        + '  File "/home/alice/.unsloth/studio/worker.py", line 9, in handle\n'
    )._public_worker_stderr_tail()
    assert "another account" not in public, public
    assert "Fatal Python error: Aborted" in public, public


def test_a_logged_traceback_is_not_returned_when_the_worker_dies_silently():
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
    import inspect
    from core.inference import worker as worker_module

    source = inspect.getsource(worker_module)
    assert "mark_log_record_continuations()" in source
    setup = source.index("LogConfig.setup_logging(")
    assert 0 < setup < source.index("mark_log_record_continuations()")


def test_a_marked_record_under_a_diagnostic_is_not_adopted_by_it():
    text = (
        "terminate called after throwing an instance of 'c10::Error'\n"
        + MARK
        + "another account's private prompt\n"
        "  what():  CUDA error: device-side assert triggered\n"
    )
    public = _orchestrator_with_capture(text)._public_worker_stderr_tail()
    assert "another account" not in public, public
    assert "terminate called" in public, public
    assert "CUDA error: device-side assert triggered" in public, public


def test_a_live_replacements_own_crash_is_still_written_to_the_log(monkeypatch):
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

    orchestrator._proc = SimpleNamespace(pid = 4242, exitcode = None)
    orchestrator._subprocess_crash_message("generation")
    assert len(written) == 1, written
    orchestrator._subprocess_crash_message("generation")
    assert len(written) == 1, written

    orchestrator._proc = SimpleNamespace(pid = 4242, exitcode = -6)
    orchestrator._subprocess_crash_message("generation", with_worker_output = True)
    assert len(written) == 2, "the replacement's own fatal output was skipped as already logged"
    orchestrator._subprocess_crash_message("generation", with_worker_output = True)
    assert len(written) == 2, written


def test_a_worker_we_stopped_on_purpose_is_not_replayed_as_a_crash(monkeypatch):
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(
        "inference_orchestrator_deliberate_stop_under_test",
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

    orchestrator = module.InferenceOrchestrator.__new__(module.InferenceOrchestrator)
    orchestrator._stderr_capture = _FixedCapture("loading shards: 100%\n")
    orchestrator._proc = None
    orchestrator._worker_stopped_deliberately = True
    orchestrator._retire_stderr_capture()
    assert written == [], written

    orchestrator._stderr_capture = _FixedCapture("Fatal Python error: Aborted\n")
    orchestrator._worker_stopped_deliberately = False
    orchestrator._retire_stderr_capture()
    assert written, "an unattended crash was skipped because a previous stop was deliberate"


def test_a_worker_that_died_between_requests_is_replayed_before_its_sink_closes(monkeypatch):
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(
        "inference_orchestrator_idle_retire_under_test",
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

    closed: "list[bool]" = []

    class _ClosableCapture(_FixedCapture):
        def close(self):
            closed.append(True)

    orchestrator = module.InferenceOrchestrator.__new__(module.InferenceOrchestrator)
    orchestrator._stderr_capture = _ClosableCapture("Fatal Python error: Segmentation fault\n")
    orchestrator._proc = SimpleNamespace(
        pid = 7331,
        exitcode = -11,
        is_alive = lambda: False,
    )

    orchestrator._retire_stderr_capture()
    assert closed == [True]
    assert written, "the capture of a worker that died unattended was closed unread"
    assert "Fatal Python error: Segmentation fault" in str(written[0])
    assert orchestrator._stderr_capture is None

    written.clear()
    orchestrator._stderr_capture = _ClosableCapture("loading shards: 40%\n")
    orchestrator._proc = SimpleNamespace(pid = 7332, exitcode = None, is_alive = lambda: True)
    orchestrator._retire_stderr_capture()
    assert written == [], written


def test_a_request_queued_behind_the_crash_is_not_given_its_last_words(monkeypatch):
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
    assert "generating a response" in queued
    # `signal.Signals(9)` raises on Windows, so the message says SIG9 there.
    assert "signal=SIGKILL" in queued or "signal=SIG9" in queued, queued
    assert "exitcode=-9" in queued, queued
    assert logged, "the server log lost the tail for the queued request's path"


def test_the_stream_asks_who_owned_the_worker_before_handing_over_the_tail():
    import inspect
    from core.inference import orchestrator as orchestrator_module

    body = inspect.getsource(orchestrator_module.InferenceOrchestrator._consume_token_stream)
    # The other crash exit, the swapped-worker branch, passes no tail at all.
    assert body.count("with_worker_output = self._owns_worker(cancel_event)") == 1, body


def test_a_queued_compare_request_does_not_own_the_worker():
    module = _load_orchestrator_module()
    orchestrator = module.InferenceOrchestrator.__new__(module.InferenceOrchestrator)
    import threading

    orchestrator._active_cancel_lock = threading.Lock()
    first, second = threading.Event(), threading.Event()
    orchestrator._active_cancel_events = []
    orchestrator._executing_cancel_events = []
    assert orchestrator._owns_worker(first) is True

    orchestrator._active_cancel_events = [first, second]
    orchestrator._executing_cancel_events = [first]
    assert orchestrator._owns_worker(first) is True
    assert orchestrator._owns_worker(second) is False


def test_the_public_tail_redacts_the_credentials_a_crash_actually_carries():
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
    assert "RuntimeError: refused" in public, public
    assert "worker.py" in public, public


def test_the_worker_output_is_opt_in_at_every_call_site():
    import inspect
    from core.inference import orchestrator as orchestrator_module

    source = inspect.getsource(orchestrator_module)
    assert "with_worker_output: bool = False" in source
    calls = source.count("self._subprocess_crash_message(")
    owned = source.count("with_worker_output = self._owns_worker(")
    # Every call site but one asks ownership; the exception is the swapped-worker branch.
    assert calls == owned + 1, (calls, owned)


def test_a_stream_whose_worker_was_swapped_gets_no_tail_at_all():
    import inspect
    from core.inference import orchestrator as orchestrator_module

    body = inspect.getsource(orchestrator_module.InferenceOrchestrator._consume_token_stream)
    swap = body.index("initial_proc or self._resp_queue is not initial_resp_queue")
    following = body[swap : body.index("resp = read_one(read_timeout)", swap)]
    assert "_subprocess_crash_message(crash_context)" in following, following
    assert "with_worker_output" not in following, following


@pytest.fixture
def _logging_restored():
    import logging

    from utils import worker_stderr

    # getattr so a build without the hook lets the CASE report the gap, not this fixture.
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
    import logging

    from utils.worker_stderr import mark_log_record_continuations

    logging.lastResort.setFormatter(logging.Formatter("%(message)s"))
    mark_log_record_continuations(logging.getLogger("unsloth-test-lastresort"))

    record = logging.LogRecord(
        "unsloth-test-lastresort", logging.ERROR, __file__, 1, "one\ntwo\nthree", (), None
    )
    lines = logging.lastResort.format(record).split("\n")
    assert lines[0] == START_MARK + "one", lines
    assert all(line.startswith(MARK) for line in lines[1:]), lines


def test_a_handler_installed_after_startup_is_marked_too(_logging_restored):
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

    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    formatted = handler.format(record)
    assert formatted.split("\n")[0] == START_MARK + "ERROR one", formatted
    assert formatted.split("\n")[1].startswith(MARK), formatted

    # Clearing the formatter is the one case the hook cannot wrap; it must not crash.
    handler.setFormatter(None)
    assert handler.formatter is None


def test_a_root_level_path_is_redacted_too():
    from core.inference.orchestrator import _redact_worker_output
    for path, tail in (
        ("/model.gguf", "model.gguf"),
        ("C:\\model.gguf", "model.gguf"),
        ("\\\\fileserver\\share", "share"),
        ("/opt", "opt"),
    ):
        public = _redact_worker_output(f"could not open {path}\n")
        assert public.strip() == f"could not open .../{tail}", public
        assert not public.count("C:"), public

    for text in (
        'the "/" separator is not a path',
        "a ratio of 3/4 at https://host/path/x",
        "use / to split and/or join",
    ):
        assert _redact_worker_output(text + "\n").strip() == text, text


def test_two_paths_on_one_line_are_two_matches():
    from core.inference.orchestrator import _redact_worker_output

    windows = _redact_worker_output("copy C:\\Users\\ann\\old.gguf to C:\\tmp\\new.gguf\n")
    assert windows.strip() == "copy .../old.gguf to .../new.gguf", windows
    posix = _redact_worker_output("cannot copy /home/ann/old.gguf to /tmp/new.gguf\n")
    assert posix.strip() == "cannot copy .../old.gguf to .../new.gguf", posix
    unc = _redact_worker_output("copy \\\\share\\team\\old.gguf to \\\\other\\team\\new.gguf\n")
    assert unc.strip() == "copy .../old.gguf to .../new.gguf", unc

    spaced = _redact_worker_output("C:\\Program Files\\unsloth\\weights.gguf failed\n")
    assert spaced.strip() == ".../weights.gguf failed", spaced
