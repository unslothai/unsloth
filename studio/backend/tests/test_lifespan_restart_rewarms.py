# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A second lifespan in one process must re-detect and re-warm, not inherit.

Shutdown clears hardware.DEVICE, so after it the process holds no measured
device. Two pieces of bookkeeping have to be cleared with it or the next
lifespan disagrees with reality:

  * DETECTION_COMPLETE. /api/health takes a set event as "detection finished,
    DEVICE is authoritative" (that is exactly why it stopped polling DEVICE --
    the detection branches assign DEVICE and keep probing). Left set over a
    cleared DEVICE, health publishes a device that is gone instead of kicking a
    new detection.
  * the one-warm-per-process latch in torch_warmup. A finished thread left in
    place makes the second lifespan's start_background_warm() a no-op, so the
    stack stays cold and the first request pays for the import again -- the
    stall this module exists to remove.

Reachable through repeated ASGI lifespan contexts and an embedded restart, both
of which reuse the app object in one interpreter.
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
    """The clear is guarded: a hardware module that rejects the write must not
    skip clear_compiled_cache, which is what frees the on-disk cache."""

    class Hostile:
        DEVICE = "cuda:0"

        def __setattr__(self, name, value):
            raise RuntimeError("read-only hardware module")

    cleared: list[str] = []
    asyncio.run(run_lifespan_shutdown(
        lambda: cleared.append("downloads"),
        lambda: cleared.append("compiled_cache"),
        Hostile(),
    ))
    assert cleared == ["downloads", "compiled_cache"]


def _restore(monkeypatch) -> None:
    """Put the module-level warm bookkeeping back after a test mutates it."""
    monkeypatch.setattr(warmup, "_thread", None, raising = False)
    monkeypatch.setattr(
        warmup, "_status", {"started": False, "finished": False, "stages": {}}, raising = False,
    )


def test_reset_lets_a_second_lifespan_start_a_fresh_warm(monkeypatch):
    _restore(monkeypatch)
    runs: list[int] = []
    monkeypatch.setattr(warmup, "_warm", lambda: runs.append(1))

    assert warmup.start_background_warm() is True
    assert warmup.join_background_warm(30) is True
    # Without a reset this is the second lifespan's outcome: the latch is still
    # held by the finished thread, so nothing warms.
    assert warmup.start_background_warm() is False

    assert warmup.reset_background_warm() is True
    assert warmup.start_background_warm() is True
    assert warmup.join_background_warm(30) is True
    assert runs == [1, 1], "the second lifespan did not run the warm again"


def test_reset_clears_the_reported_warm_status(monkeypatch):
    _restore(monkeypatch)
    monkeypatch.setattr(warmup, "_warm", lambda: warmup._run_stage("hardware", lambda: None))

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

    def _slow() -> None:
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
        sub.func.id for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
    }
    assert "reset_background_warm" in called, (
        "the lifespan never releases the one-warm-per-process latch, so a "
        "second lifespan in this process leaves the ML stack cold"
    )
