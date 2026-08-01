# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A second lifespan in one process must re-detect and re-warm, not inherit.

Shutdown clears hardware.DEVICE, so after it the process holds no measured device. Two
pieces of bookkeeping have to be cleared with it or the next lifespan disagrees with
reality:

  * DETECTION_COMPLETE. /api/health takes a set event as "DEVICE is authoritative", so
    left set over a cleared DEVICE it publishes a device that is gone instead of
    kicking a new detection.
  * torch_warmup's one-warm-per-process latch. A finished thread left in place makes
    the second lifespan's start_background_warm() a no-op, so the stack stays cold and
    the first request pays for the import again.

Reachable through repeated ASGI lifespan contexts and an embedded restart, both of
which reuse the app object in one interpreter.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.torch_warmup as warmup  # noqa: E402
from utils.lifespan_shutdown import run_lifespan_shutdown  # noqa: E402


def _stub_hardware(*, with_event: bool = True) -> types.SimpleNamespace:
    stub = types.SimpleNamespace(DEVICE = "cuda:0")
    if with_event:
        stub.DETECTION_COMPLETE = threading.Event()
        stub.DETECTION_COMPLETE.set()
    return stub


def test_shutdown_clears_the_detection_completion_signal():
    hw = _stub_hardware()
    asyncio.run(run_lifespan_shutdown(lambda: None, lambda: None, hw))
    assert hw.DEVICE is None
    assert not hw.DETECTION_COMPLETE.is_set(), (
        "shutdown cleared DEVICE but left DETECTION_COMPLETE set; /api/health "
        "would report the torn-down device as measured instead of re-detecting"
    )


def test_shutdown_still_clears_device_without_the_event():
    """Negative control: a hardware module with no event must not break shutdown."""
    hw = _stub_hardware(with_event = False)
    asyncio.run(run_lifespan_shutdown(lambda: None, lambda: None, hw))
    assert hw.DEVICE is None


def test_shutdown_runs_the_later_steps_even_if_clearing_raises():
    """A hardware module that rejects the write must not skip clear_compiled_cache,
    which is what frees the on-disk cache."""

    class Hostile:
        DEVICE = "cuda:0"

        def __setattr__(self, name, value):
            raise RuntimeError("read-only hardware module")

    cleared: list[str] = []
    asyncio.run(
        run_lifespan_shutdown(
            lambda: cleared.append("downloads"),
            lambda: cleared.append("compiled_cache"),
            Hostile(),
        )
    )
    assert cleared == ["downloads", "compiled_cache"]


def _restore(monkeypatch) -> None:
    """Put the module-level warm bookkeeping back after a test mutates it."""
    monkeypatch.setattr(warmup, "_thread", None, raising = False)
    monkeypatch.setattr(
        warmup,
        "_status",
        {"started": False, "finished": False, "stages": {}},
        raising = False,
    )


def test_a_second_lifespan_warms_again_however_the_first_ended(monkeypatch):
    """Both restart paths must re-warm, not just the one that got a clean reset.

    reset_background_warm() declines while a warm is alive, the normal case for a
    shutdown landing mid-warm. If that warm then finishes, its thread object is left
    behind, and treating it as "already started" would skip the second lifespan's warm
    entirely, over hardware state the same shutdown just cleared.
    """
    _restore(monkeypatch)
    runs: list[int] = []
    monkeypatch.setattr(warmup, "_warm", lambda *_: runs.append(1))

    assert warmup.start_background_warm() is True
    assert warmup.join_background_warm(30) is True

    # Path one: shutdown reset cleanly, the next lifespan warms.
    assert warmup.reset_background_warm() is True
    assert warmup.start_background_warm() is True
    assert warmup.join_background_warm(30) is True
    assert runs == [1, 1], "the second lifespan did not run the warm again"

    # Path two: no reset ran, because the warm outlived the shutdown that tried.
    # The finished thread must not still hold the latch.
    assert warmup.start_background_warm() is True
    assert warmup.join_background_warm(30) is True
    assert runs == [1, 1, 1], (
        "a finished warm kept the latch, so the restart served with the ML stack "
        "cold until some request kicked detection"
    )


def test_a_live_warm_still_holds_the_latch(monkeypatch):
    """Negative control: clearing the dead thread must not allow two live warms."""
    _restore(monkeypatch)
    release = threading.Event()
    entered = threading.Event()

    def _slow_warm(*_) -> None:
        entered.set()
        release.wait(30)

    monkeypatch.setattr(warmup, "_warm", _slow_warm)
    try:
        assert warmup.start_background_warm() is True
        assert entered.wait(30)
        assert warmup.start_background_warm() is False, "a second warm ran beside a live one"
        assert warmup.reset_background_warm() is False
    finally:
        release.set()
        warmup.join_background_warm(30)


def test_reset_clears_the_reported_warm_status(monkeypatch):
    _restore(monkeypatch)
    monkeypatch.setattr(warmup, "_warm", lambda *_: warmup._run_stage("hardware", lambda: None))

    warmup.start_background_warm()
    warmup.join_background_warm(30)
    assert warmup.warm_status()["stages"]["hardware"]["ok"] is True

    warmup.reset_background_warm()
    status = warmup.warm_status()
    assert status["started"] is False
    assert status["finished"] is False
    assert status["stages"] == {}, "a stale stage table outlived the warm it described"
    assert status["seconds"] is None


def test_reset_declines_while_the_warm_is_still_running(monkeypatch):
    """Two warms on the same imports is worse than a cold second lifespan."""
    _restore(monkeypatch)
    release = threading.Event()
    entered = threading.Event()

    def _slow(*_) -> None:
        entered.set()
        release.wait(30)

    monkeypatch.setattr(warmup, "_warm", _slow)
    assert warmup.start_background_warm() is True
    assert entered.wait(30) is True
    running = warmup._thread

    try:
        assert warmup.reset_background_warm() is False
        assert warmup._thread is running, "the reset dropped a live warm thread"
    finally:
        release.set()
        warmup.join_background_warm(30)


def test_the_lifespan_resets_the_warm_after_shutdown():
    """Guard the wiring, not just the helper: the reset has to be reachable."""
    import ast

    tree = ast.parse((_BACKEND / "main.py").read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan":
            break
    else:
        raise AssertionError("lifespan not found in main.py")

    called = {
        sub.func.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
    }
    assert "reset_background_warm" in called, (
        "the lifespan never releases the one-warm-per-process latch, so a "
        "second lifespan in this process leaves the ML stack cold"
    )
