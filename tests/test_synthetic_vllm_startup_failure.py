# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A vLLM server that never starts must stop SyntheticDataKit, not be ignored.

`Meta_Synthetic_Data_Llama3_2_(3B).ipynb` on a Colab T4 lost its vLLM server at
import time:

    ImportError: cannot import name 'ProcessorMixin' from 'transformers'

`SyntheticDataKit.__init__` printed the tails and `return`ed, so the notebook
received a fully constructed object with no server behind it. `ingest`,
`create` and `save-as` then each printed "VLLM server not available", wrote
nothing, and still exited 0, because a `!` shell line does not raise on a
non-zero status. The first thing that actually stopped the notebook was, five
cells and fourteen error messages later:

    FileNotFoundError: File data/final/arxiv_org_0_qa_pairs_ft.json does not
    exist

which names a file no step had ever been in a position to write, and points at
`pd.read_json` rather than at the server.

These drive the readiness path directly with fakes: no GPU, no vLLM, no
network. `chunk_data` is covered separately in test_synthetic_chunk_data.py.
"""

import re
import threading
import time

import pytest

from unsloth.dataprep.synthetic import SyntheticDataKit


class _FakeCapture:
    """Stands in for PipeCapture without a pipe or a reader thread."""

    def __init__(
        self,
        ready = False,
        closed = False,
        lines = "",
        ready_after = None,
    ):
        self._ready = threading.Event()
        if ready:
            self._ready.set()
        self._closed = closed
        self._lines = lines
        self._ready_after = ready_after
        self._first_wait = None

    def wait_for_ready(self, timeout = None):
        if self._ready_after is not None:
            if self._first_wait is None:
                self._first_wait = time.monotonic()
            if time.monotonic() - self._first_wait >= self._ready_after:
                self._ready.set()
        return self._ready.wait(timeout)

    def has_closed(self):
        return self._closed

    def tail(self, n = 200):
        return self._lines


class _FakeProcess:
    def __init__(self, returncode = None):
        self._returncode = returncode
        self.terminated = False
        self.killed = False
        self.pid = -1

    def poll(self):
        return self._returncode

    def terminate(self):
        self.terminated = True
        self._returncode = -15

    def kill(self):
        self.killed = True
        self._returncode = -9

    def wait(self, timeout = None):
        return self._returncode


def _kit(stdout_capture, stderr_capture, process):
    kit = SyntheticDataKit.__new__(SyntheticDataKit)
    kit.stdout_capture = stdout_capture
    kit.stderr_capture = stderr_capture
    kit.vllm_process = process
    return kit


REAL_STDERR_TAIL = (
    '  File "/usr/local/lib/python3.12/dist-packages/vllm/inputs/registry.py", '
    "line 9, in <module>\n"
    "    from transformers import BatchFeature, PretrainedConfig, ProcessorMixin\n"
    "ImportError: cannot import name 'ProcessorMixin' from 'transformers'"
)


def test_a_server_that_exited_raises_instead_of_returning():
    kit = _kit(
        _FakeCapture(ready = False, closed = True, lines = "vLLM STDOUT: booting"),
        _FakeCapture(lines = REAL_STDERR_TAIL),
        _FakeProcess(returncode = 1),
    )
    with pytest.raises(RuntimeError) as excinfo:
        kit._await_vllm_server(timeout = 30, poll_interval = 0.01)
    assert "exited with code 1" in str(excinfo.value)


def test_the_servers_own_traceback_is_in_the_message():
    """The cause is in the child's stderr; printing it and returning lost it."""
    kit = _kit(
        _FakeCapture(ready = False, closed = True, lines = "vLLM STDOUT: booting"),
        _FakeCapture(lines = REAL_STDERR_TAIL),
        _FakeProcess(returncode = 1),
    )
    with pytest.raises(RuntimeError) as excinfo:
        kit._await_vllm_server(timeout = 30, poll_interval = 0.01)
    message = str(excinfo.value)
    assert "ImportError: cannot import name 'ProcessorMixin'" in message
    assert "vLLM STDOUT: booting" in message
    assert "vLLM stdout (last 50 lines)" in message
    assert "vLLM stderr (last 50 lines)" in message


def test_a_closed_stdout_on_a_live_process_also_raises():
    kit = _kit(
        _FakeCapture(ready = False, closed = True),
        _FakeCapture(),
        _FakeProcess(returncode = None),
    )
    with pytest.raises(RuntimeError, match = "closed its stdout"):
        kit._await_vllm_server(timeout = 30, poll_interval = 0.01)


def test_a_silent_server_raises_on_the_timeout():
    kit = _kit(
        _FakeCapture(ready = False, closed = False),
        _FakeCapture(),
        _FakeProcess(returncode = None),
    )
    with pytest.raises(RuntimeError, match = "was not ready within"):
        kit._await_vllm_server(timeout = 0.05, poll_interval = 0.01)


def test_a_dead_server_is_noticed_long_before_the_timeout():
    """The old code waited on the readiness event alone, so a server that died
    in twenty seconds still burned the full twenty-minute default."""
    kit = _kit(
        _FakeCapture(ready = False, closed = True),
        _FakeCapture(),
        _FakeProcess(returncode = 1),
    )
    started = time.monotonic()
    with pytest.raises(RuntimeError):
        kit._await_vllm_server(timeout = 1200, poll_interval = 0.01)
    assert time.monotonic() - started < 10


def test_the_process_is_terminated_before_raising():
    """Otherwise the failed server keeps the GPU and port 8000."""
    process = _FakeProcess(returncode = None)
    kit = _kit(_FakeCapture(ready = False, closed = True), _FakeCapture(), process)
    with pytest.raises(RuntimeError):
        kit._await_vllm_server(timeout = 30, poll_interval = 0.01)
    assert process.terminated or process.killed


def test_a_ready_server_returns_quietly():
    kit = _kit(_FakeCapture(ready = True), _FakeCapture(), _FakeProcess())
    assert kit._await_vllm_server(timeout = 30, poll_interval = 0.01) is None


def test_readiness_arriving_late_is_still_success():
    """A slow but healthy start must not be reported as a failure."""
    kit = _kit(
        _FakeCapture(ready = False, ready_after = 0.05),
        _FakeCapture(),
        _FakeProcess(returncode = None),
    )
    assert kit._await_vllm_server(timeout = 30, poll_interval = 0.01) is None


def test_readiness_wins_over_a_process_that_then_exits():
    """The ready line is checked first, so a server that prints it and exits a
    moment later is not misreported as never having started."""
    kit = _kit(_FakeCapture(ready = True), _FakeCapture(), _FakeProcess(returncode = 1))
    assert kit._await_vllm_server(timeout = 30, poll_interval = 0.01) is None


def test_the_message_says_what_the_user_should_do_next():
    kit = _kit(_FakeCapture(ready = False, closed = True), _FakeCapture(), _FakeProcess(1))
    with pytest.raises(RuntimeError) as excinfo:
        kit._await_vllm_server(timeout = 30, poll_interval = 0.01)
    message = str(excinfo.value)
    assert "synthetic-data-kit" in message
    assert "FileNotFoundError" in message


# --- the readiness probe --------------------------------------------------


def test_check_vllm_status_is_false_when_nothing_is_listening(monkeypatch):
    import requests

    def _refused(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", _refused)
    assert SyntheticDataKit.check_vllm_status() is False


def test_check_vllm_status_swallows_a_read_timeout(monkeypatch):
    """Only ConnectionError was caught, so a stalled server left the readiness
    loop as an unrelated traceback."""
    import requests

    def _timeout(*args, **kwargs):
        raise requests.exceptions.ReadTimeout("stalled")

    monkeypatch.setattr(requests, "get", _timeout)
    assert SyntheticDataKit.check_vllm_status() is False


def test_check_vllm_status_passes_a_timeout(monkeypatch):
    """Without one, requests waits forever on a server that never answers."""
    import requests

    seen = {}

    class _Response:
        status_code = 200

    def _get(url, **kwargs):
        seen.update(kwargs)
        return _Response()

    monkeypatch.setattr(requests, "get", _get)
    assert SyntheticDataKit.check_vllm_status() is True
    assert seen.get("timeout")


def test_check_vllm_status_is_false_on_a_non_200(monkeypatch):
    import requests

    class _Response:
        status_code = 503

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response())
    assert SyntheticDataKit.check_vllm_status() is False


def test_the_failure_path_is_not_a_bare_return_any_more():
    """Guards the shape of the bug: printing and returning handed the caller a
    kit with no server, which is what let the notebook run on for five cells."""
    import inspect

    source = inspect.getsource(SyntheticDataKit.__init__)
    assert "_await_vllm_server" in source
    assert not re.search(r"terminate_tree\(self\.vllm_process\)\s*\n\s*return", source)
