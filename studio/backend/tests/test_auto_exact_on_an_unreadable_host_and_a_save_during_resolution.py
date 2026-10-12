# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unreadable host certifies no exact parks, and a save during resolution is not missed.

Before launch, auto skipped its fallback when the host could not be read, so the child ran
with the whole budget sized for exact parks, where a plain launch on the same host parks
nothing. And a save landing between a load reading its setting and publishing it saw neither
a child nor a pending load, answering that no reload was needed.
"""

from __future__ import annotations

import inspect
import textwrap
import weakref

from core.inference import llama_cpp as llama_mod
from core.inference import llama_exact as exact
from core.inference.llama_cpp import LlamaCppBackend
import utils.exact_concurrency_settings as stored
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


def _lift(start: str, end: str) -> str:
    source = inspect.getsource(LlamaCppBackend.load_model)
    begin = source.index(start)
    begin = source.rindex("\n", 0, begin) + 1
    stop = source.index(end, begin)
    stop = source.rindex("\n", 0, stop) + 1
    return textwrap.dedent(source[begin:stop])


class _Backend:
    def __init__(self, writes = 40_000):
        self._exact_parking_writes = writes
        self._exact_host_short = None
        self._parking_reserving_mib = None
        self.warnings: list[str] = []

    def _record_load_warning(self, text):
        self.warnings.append(text)


@pytest.fixture
def unreadable(monkeypatch):
    monkeypatch.setattr(llama_mod, "_available_host_memory_mib", lambda: None)
    monkeypatch.setattr(llama_mod, "_total_host_memory_mib", lambda: None)
    monkeypatch.setattr(llama_mod, "_serving_backends_snapshot", lambda: [])
    monkeypatch.setattr(llama_mod, "_parking_backends", weakref.WeakSet())


def _run(block: str, backend, *, setting, **extra):
    cmd = ["llama-server", "--preempt-ram", "40000"]
    namespace = dict(vars(llama_mod))
    namespace.update(
        self = backend,
        cmd = cmd,
        _exact = exact,
        _exact_wanted = True,
        _exact_setting = setting,
        _exact_budget = 40_000,
        _exact_writes = 40_000,
        **extra,
    )
    exec(block, namespace)
    return namespace, cmd


_BEFORE_LAUNCH = (
    "_host_free_mib = _exact_host_free_mib(self)",
    "# An auto-fit context leaves the pool unknown",
)
_AFTER_PLACEMENT = (
    '_exact_writes_now = getattr(self, "_exact_parking_writes", None)',
    "_host_ram_msg = _offload_msg",
)


class TestAnUnreadableHostIsAShortfall:
    def test_auto_falls_back_before_launch_and_parks_nothing(self, unreadable):
        backend = _Backend()
        namespace, cmd = _run(_lift(*_BEFORE_LAUNCH), backend, setting = exact.EXACT_AUTO)
        assert namespace["_exact_wanted"] is False, "auto kept exact mode on an unreadable host"
        assert backend._exact_host_short == (40_000, -1)
        assert "--preempt-ram" not in cmd, cmd
        assert any("could not be read" in w for w in backend.warnings)

    def test_on_keeps_the_mode_and_is_failed_after_the_load(self, unreadable):
        backend = _Backend()
        namespace, cmd = _run(_lift(*_BEFORE_LAUNCH), backend, setting = exact.EXACT_ON)
        assert namespace["_exact_wanted"] is True
        assert backend._exact_host_short == (40_000, -1)
        assert cmd[-1] == "40000"

    def test_the_check_after_placement_reads_it_the_same_way(self, unreadable):
        backend = _Backend()
        namespace, cmd = _run(
            _lift(*_AFTER_PLACEMENT), backend, setting = exact.EXACT_AUTO, _host_weights_mib = 8000
        )
        assert namespace["_exact_wanted"] is False
        assert backend._exact_host_short == (40_000, -1)
        assert "--preempt-ram" not in cmd, cmd


class TestASaveDuringResolutionIsNotMissed:
    def test_a_write_between_the_read_and_the_publication_republishes(self):
        published: list[str] = []
        reads = iter(["off", "on"])

        def resolve():
            value = next(reads)
            if value == "off":
                # The owner saves while this read is in flight.
                stored._invalidate(stored.EXACT_CONCURRENCY_SETTING_KEY)
            return value

        assert stored.capture_exact_concurrency(resolve, published.append) == "on"
        assert published == ["off", "on"], "the load kept the setting the save replaced"

    def test_a_quiet_read_publishes_once(self):
        published: list[str] = []
        assert stored.capture_exact_concurrency(lambda: "auto", published.append) == "auto"
        assert published == ["auto"]

    def test_the_load_resolves_through_the_capture(self):
        source = inspect.getsource(LlamaCppBackend.load_model)
        assert "capture_exact_concurrency(" in source
        assert "_exact_setting = _exact.resolve_exact_setting(" not in source
