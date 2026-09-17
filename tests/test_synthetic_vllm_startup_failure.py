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


# Every kit built here, kept alive past the end of the test that made it so the fixture below, and not the dying test
# frame, decides when it is collected.
_LIVE_KITS = []


def _kit(stdout_capture, stderr_capture, process):
    kit = SyntheticDataKit.__new__(SyntheticDataKit)
    kit.stdout_capture = stdout_capture
    kit.stderr_capture = stderr_capture
    kit.vllm_process = process
    _LIVE_KITS.append(kit)
    return kit


@pytest.fixture(autouse = True)
def _retire_kits_without_reaping_a_server():
    """Stop `__del__` running a real server teardown against a fake process.

    `SyntheticDataKit.__del__` calls `cleanup()`, which ends in
    `for _ in range(10): torch.cuda.empty_cache(); gc.collect()`. Ten full
    collections cost about 3s per test when this file runs alone, but scale with
    the live heap: inside the whole repo suite they cost about 40s per test, which
    is why 15 of the 16 slowest tests in the suite are in this file.

    None of these tests have a server to reap, and none of them assert on
    `cleanup()`. `cleanup()` returns immediately when `vllm_process` is absent, so
    dropping the attribute retires the kit without the collections. The teardown
    the tests do care about, `terminate_tree` on the failure path, is asserted
    inside the tests themselves and is untouched.
    """
    yield
    while _LIVE_KITS:
        kit = _LIVE_KITS.pop()
        for attribute in ("vllm_process", "_delete_vllm"):
            if hasattr(kit, attribute):
                delattr(kit, attribute)


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


# Both waits bound work by ELAPSED time.


# --- the timeout is a deadline, not a number of laps ----------------------
# Both waits bound work by ELAPSED time. Attempt counts and flat poll intervals
# only agree with that when each attempt is instant, which is exactly what the
# failing cases below are not.
class _RecordingCapture(_FakeCapture):
    """Remembers every timeout it was asked to wait for."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.waits = []

    def wait_for_ready(self, timeout = None):
        self.waits.append(timeout)
        return super().wait_for_ready(timeout)


def test_no_single_wait_outruns_the_callers_timeout():
    """`timeout = 0.05` used to wait a full second on the first lap, so the
    method could return success from a deadline that had already passed."""
    capture = _RecordingCapture(ready = False)
    kit = _kit(capture, _FakeCapture(), _FakeProcess())
    with pytest.raises(RuntimeError):
        kit._await_vllm_server(timeout = 0.05, poll_interval = 1.0)
    assert capture.waits, "the readiness event was never waited on"
    assert max(capture.waits) <= 0.05, (
        f"waited {max(capture.waits)}s against a 0.05s timeout; a lap must be "
        f"capped to the time left"
    )


def test_a_short_timeout_returns_promptly():
    """The wall clock, not just the arithmetic."""
    kit = _kit(_FakeCapture(ready = False), _FakeCapture(), _FakeProcess())
    started = time.monotonic()
    with pytest.raises(RuntimeError):
        kit._await_vllm_server(timeout = 0.05, poll_interval = 1.0)
    elapsed = time.monotonic() - started
    assert elapsed < 0.5, f"took {elapsed:.2f}s to honour a 0.05s timeout"


def test_the_whole_timeout_is_still_used_when_it_is_long():
    """The cap must not cut a long wait short: readiness at 0.2s must be seen."""
    kit = _kit(
        _FakeCapture(ready = False, ready_after = 0.2),
        _FakeCapture(),
        _FakeProcess(),
    )
    kit._await_vllm_server(timeout = 5.0, poll_interval = 0.05)


def test_the_metrics_wait_is_bounded_by_time_not_by_attempts(monkeypatch):
    """Each request gets 5 seconds, so counting to 100 with a 1 second sleep
    spent up to ten minutes against a message promising 100 seconds."""
    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    # each health check burns its full 5s request timeout, then the 1s sleep
    monkeypatch.setattr(time, "sleep", lambda s: clock.__setitem__("now", clock["now"] + s))

    calls = {"n": 0}

    def stalled_check():
        calls["n"] += 1
        clock["now"] += 5.0
        return False

    kit = _kit(_FakeCapture(ready = True), _FakeCapture(), _FakeProcess())
    kit.check_vllm_status = stalled_check

    started = clock["now"]
    with pytest.raises(RuntimeError):
        kit._await_metrics_endpoint()
    spent = clock["now"] - started
    assert (
        spent <= 110
    ), f"spent {spent:.0f}s of simulated time on a wait the message calls 100 seconds"


def test_an_unbounded_timeout_is_a_wait_not_a_type_error():
    """`timeout = None` is a legal call meaning wait as long as it takes.

    Deadline arithmetic on `None` raised `TypeError` seconds after vLLM was
    spawned, losing both the wait asked for and the child started.
    """
    kit = _kit(
        _FakeCapture(ready = False, ready_after = 0.05),
        _FakeCapture(),
        _FakeProcess(),
    )
    kit._await_vllm_server(timeout = None, poll_interval = 0.01)


def test_an_unbounded_wait_still_notices_a_dead_child():
    """`None` removes the deadline, not the other exit conditions. The bare
    `Event.wait(None)` this replaces blocked forever on a dead server."""
    kit = _kit(_FakeCapture(ready = False), _FakeCapture(), _FakeProcess(returncode = 1))
    started = time.monotonic()
    with pytest.raises(RuntimeError, match = "exited with code 1"):
        kit._await_vllm_server(timeout = None, poll_interval = 0.01)
    elapsed = time.monotonic() - started
    assert elapsed < 0.5, f"took {elapsed:.2f}s to notice a process that had already exited"


def test_an_unbounded_wait_never_expires(monkeypatch):
    """No deadline means no deadline, however long the clock runs."""
    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    capture = _RecordingCapture(ready = False)
    laps = {"n": 0}

    def slow_wait(timeout = None):
        laps["n"] += 1
        clock["now"] += 10_000.0
        if laps["n"] >= 3:
            capture._ready.set()
        return capture._ready.is_set()

    monkeypatch.setattr(capture, "wait_for_ready", slow_wait)
    kit = _kit(capture, _FakeCapture(), _FakeProcess())
    kit._await_vllm_server(timeout = None, poll_interval = 1.0)
    assert laps["n"] == 3


def test_the_metrics_wait_accepts_an_unbounded_timeout():
    """Same arithmetic, same fix, pinned so the helper is not left half done."""
    calls = {"n": 0}

    def check():
        calls["n"] += 1
        return calls["n"] >= 3

    kit = _kit(_FakeCapture(ready = True), _FakeCapture(), _FakeProcess())
    kit.check_vllm_status = check
    kit._await_metrics_endpoint(timeout = None, poll_interval = 0.0)
    assert calls["n"] == 3
