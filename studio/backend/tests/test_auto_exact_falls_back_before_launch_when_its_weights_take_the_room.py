# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto exact judges its parks again once the placement is known, before the child starts.

The parks were measured against host RAM before the weights' placement was known. Partially
offloaded weights then took that room during startup, and the check after the load could only
record the shortfall: the child kept exact mode and the whole budget sized for exact parks.
"""

from __future__ import annotations

import inspect
import textwrap
import weakref

from core.inference import llama_cpp as llama_mod
from core.inference import llama_exact as exact
from core.inference.llama_cpp import LlamaCppBackend
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")

_START = '_exact_writes_now = getattr(self, "_exact_parking_writes", None)'
_END = "_host_ram_msg = _offload_msg"


def _block() -> str:
    source = inspect.getsource(LlamaCppBackend.load_model)
    start = source.index(_START)
    start = source.rindex("\n", 0, start) + 1
    end = source.index(_END, start)
    end = source.rindex("\n", 0, end) + 1
    return textwrap.dedent(source[start:end])


class _Backend:
    def __init__(self, writes):
        self._exact_parking_writes = writes
        self._exact_host_short = None
        self._parking_reserving_mib = None
        self.warnings: list[str] = []

    def _record_load_warning(self, text):
        self.warnings.append(text)


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(llama_mod, "_available_host_memory_mib", lambda: 100_000)
    monkeypatch.setattr(llama_mod, "_total_host_memory_mib", lambda: 128_000)
    monkeypatch.setattr(llama_mod, "_serving_backends_snapshot", lambda: [])
    monkeypatch.setattr(llama_mod, "_parking_backends", weakref.WeakSet())


def _run(
    backend,
    *,
    setting,
    weights,
    budget = 40_000,
):
    cmd = ["llama-server", "--preempt-ram", str(budget)]
    namespace = dict(vars(llama_mod))
    namespace.update(
        self = backend,
        cmd = cmd,
        _exact = exact,
        _exact_wanted = True,
        _exact_setting = setting,
        _exact_budget = budget,
        _host_weights_mib = weights,
    )
    exec(_block(), namespace)
    return namespace, cmd


def test_auto_falls_back_with_a_bounded_budget(host):
    backend = _Backend(writes = 40_000)
    # What a plain launch with these weights is allowed: 100 GB free less 80 GB of weights.
    plain = llama_mod._host_bounded_parking_mib(40_000, backend, 80_000)
    assert 0 < plain < 40_000
    namespace, cmd = _run(backend, setting = exact.EXACT_AUTO, weights = 80_000)
    assert namespace["_exact_wanted"] is False, "auto kept exact mode into a short host"
    assert backend._exact_host_short == (40_000, 20_000)
    assert any("80000" in w for w in backend.warnings)
    assert cmd[-2:] == ["--preempt-ram", str(plain)], cmd
    assert backend._parking_reserving_mib == plain + 80_000


def test_on_keeps_the_mode_and_is_failed_after_the_load(host):
    backend = _Backend(writes = 40_000)
    namespace, cmd = _run(backend, setting = exact.EXACT_ON, weights = 80_000)
    assert namespace["_exact_wanted"] is True
    assert backend._exact_host_short == (40_000, 20_000)
    assert cmd[-1] == "40000"


def test_room_enough_keeps_exact(host):
    backend = _Backend(writes = 4096)
    namespace, cmd = _run(backend, setting = exact.EXACT_AUTO, weights = 20_000)
    assert namespace["_exact_wanted"] is True
    assert backend._exact_host_short is None and not backend.warnings
    assert cmd[-1] == "40000"


def test_an_unreadable_host_certifies_nothing(host, monkeypatch):
    # Read as the check after the load reads it: an unknown reading is no room.
    monkeypatch.setattr(llama_mod, "_exact_host_free_mib", lambda backend: None)
    backend = _Backend(writes = 40_000)
    namespace, _cmd = _run(backend, setting = exact.EXACT_AUTO, weights = 80_000)
    assert namespace["_exact_wanted"] is False
    assert backend._exact_host_short == (40_000, -1)


def test_it_runs_before_the_child_env_is_set():
    source = inspect.getsource(LlamaCppBackend.load_model)
    assert source.index(_START) < source.index("_exact.apply_child_env(env, on = _exact_wanted)")
