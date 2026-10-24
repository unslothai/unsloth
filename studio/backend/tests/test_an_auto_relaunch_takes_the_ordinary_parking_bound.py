# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An Auto load relaunched as an ordinary server because the child still ran exact kernels takes
the ordinary launch's host-bounded parking budget, as the retries dropping exact mode do, even
when the host could hold the one sized whole for exact parks. A child the host holds and that
runs no exact kernels is not relaunched for the budget alone."""

from __future__ import annotations

import weakref

from core.inference import llama_cpp as llama_mod
import pytest

from .test_an_auto_load_with_an_unsized_pool_does_not_keep_an_oversized_parking_budget import (
    _Backend,
    _Parking,
    _lift,
)

pytestmark = pytest.mark.usefixtures("preemption_opted_in")

_EXACT_BUDGET = 40_000


@pytest.fixture
def roomy_host(monkeypatch):
    monkeypatch.setattr(llama_mod, "_available_host_memory_mib", lambda: 100_000)
    monkeypatch.setattr(llama_mod, "_total_host_memory_mib", lambda: 128_000)
    monkeypatch.setattr(llama_mod, "_serving_backends_snapshot", lambda: [])
    monkeypatch.setattr(llama_mod, "_parking_backends", weakref.WeakSet())


def _relaunch(backend, *, named = _EXACT_BUDGET, exact_env = False):
    launched = []
    namespace = dict(vars(llama_mod))
    argv = ["llama-server", "-c", "0", "--kv-unified", "--preempt-ram", str(named)]

    def spawn(run_cmd, *, label = ""):
        launched.append(list(run_cmd))
        namespace["_last_spawn_cmd"] = list(run_cmd)
        return True

    namespace.update(
        self = backend,
        cmd = argv,
        env = {llama_mod._exact.CHILD_ENV: "1"} if exact_env else {},
        _last_spawn_cmd = list(argv),
        _exact_budget = _EXACT_BUDGET,
        _exact_wanted = True,
        kv_cache_unified = True,
        n_parallel = 4,
        _spawn_and_wait = spawn,
        _finish_cancelled_health_wait = lambda message: False,
        _preemption = _Parking(),
    )
    return _lift("_relaunch_as_ordinary", namespace), launched


def test_a_relaunch_dropping_exact_kernels_takes_the_ordinary_budget(roomy_host):
    backend = _Backend()
    backend._launched_parking_mib = _EXACT_BUDGET
    ordinary = llama_mod._host_bounded_parking_mib(_EXACT_BUDGET, backend)
    assert ordinary == 25_000
    assert llama_mod._exact_host_shortfall_after_load(_EXACT_BUDGET, 100_000) is None
    relaunch, launched = _relaunch(backend, exact_env = True)
    assert relaunch(None, True) is True
    assert len(launched) == 1
    assert launched[0][-2:] == ["--preempt-ram", str(ordinary)]
    assert backend._launched_parking_mib == ordinary


def test_a_child_the_host_holds_is_not_relaunched_for_the_budget_alone(roomy_host):
    backend = _Backend()
    relaunch, launched = _relaunch(backend)
    assert relaunch(None, False) is True
    assert launched == []


def test_a_budget_the_user_named_is_still_kept_on_the_relaunch(roomy_host):
    backend = _Backend()
    relaunch, launched = _relaunch(backend, named = 60_000, exact_env = True)
    assert relaunch(None, True) is True
    assert len(launched) == 1
    assert launched[0][-2:] == ["--preempt-ram", "60000"]
