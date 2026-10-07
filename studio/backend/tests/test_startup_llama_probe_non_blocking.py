# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The llama.cpp startup probes must run OFF the FastAPI lifespan critical path.

Regression guard for the macOS slow-startup bug: the capability + freshness probes
(added in #5528/#5529) used to run inline in `lifespan`, so a cold/slow GitHub
freshness check blocked `Application startup complete` for tens of seconds. They now
run on a daemon thread, and are skipped entirely when update checks are disabled.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import main  # noqa: E402
import utils.llama_cpp_freshness as freshness  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

# Deadlock backstops, not pacing: a regression fails after 30s instead of hanging CI.
_BACKSTOP_S = 30.0


class _FakeApp:
    class _State:
        pass

    def __init__(self) -> None:
        self.state = _FakeApp._State()
        self.state.llama_cpp_capabilities = None
        self.state.llama_cpp_freshness = None


@pytest.fixture(autouse = True)
def _fast_capability_probe(monkeypatch):
    monkeypatch.setattr(
        LlamaCppBackend,
        "_find_llama_server_binary",
        staticmethod(lambda: "/no/such/llama-server"),
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        staticmethod(lambda _b: {"found": False}),
    )
    monkeypatch.delenv("UNSLOTH_DISABLE_UPDATE_CHECK", raising = False)


def test_probe_does_not_block_startup(monkeypatch):
    """`_start_llama_cpp_probes_if_enabled` returns immediately even though the
    freshness check is still stalled, then populates app.state later."""

    entered = threading.Event()
    release = threading.Event()

    def _slow_freshness(_bin, **_kw):
        # Blocks until released, so an inline probe hangs and fails rather than merely running slow.
        entered.set()
        assert release.wait(_BACKSTOP_S), "the test never released the freshness check"
        return {"stale": False, "behind": False}

    monkeypatch.setattr(freshness, "check_prebuilt_freshness", _slow_freshness)

    app = _FakeApp()
    t0 = time.monotonic()
    main._start_llama_cpp_probes_if_enabled(app)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.5, f"startup probe blocked the caller for {elapsed:.2f}s"

    assert entered.wait(_BACKSTOP_S), "the probe thread never reached the freshness check"
    release.set()

    deadline = time.monotonic() + _BACKSTOP_S
    while app.state.llama_cpp_freshness is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert app.state.llama_cpp_freshness == {"stale": False, "behind": False}


def test_disable_env_skips_probe_entirely(monkeypatch):
    """UNSLOTH_DISABLE_UPDATE_CHECK=1 starts no probe thread and makes no call."""
    calls: list[int] = []

    def _freshness(_bin, **_kw):
        calls.append(1)
        return {"stale": False}

    monkeypatch.setattr(freshness, "check_prebuilt_freshness", _freshness)
    monkeypatch.setenv("UNSLOTH_DISABLE_UPDATE_CHECK", "1")

    app = _FakeApp()
    before = {t for t in threading.enumerate()}
    main._start_llama_cpp_probes_if_enabled(app)

    # Enumerate threads rather than sleep: proves no probe thread was ever created.
    started = [
        t for t in threading.enumerate() if t not in before and t.name == "llama-cpp-startup-probe"
    ]
    assert started == [], "a probe thread was started despite UNSLOTH_DISABLE_UPDATE_CHECK=1"
    assert calls == [], "freshness check ran despite UNSLOTH_DISABLE_UPDATE_CHECK=1"
    assert app.state.llama_cpp_freshness is None
