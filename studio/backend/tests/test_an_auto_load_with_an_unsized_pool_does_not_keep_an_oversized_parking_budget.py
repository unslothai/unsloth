# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto does not keep a child on a parking budget the host was just found unable to hold.

An auto-fit context leaves the pool unsized, so the child launches on the whole budget exact
parks would need and the host is judged only once it is up. Auto then publishes that child
anyway, still free to park the whole budget into RAM the check found missing, and no running
llama-server lowers its own `--preempt-ram`: it is launched again on the bounded one.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import weakref

from core.inference import llama_cpp as llama_mod
from core.inference.llama_cpp import LlamaCppBackend
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


def _lift(name: str, namespace: dict):
    tree = ast.parse(textwrap.dedent(inspect.getsource(LlamaCppBackend.load_model)))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(
                compile(ast.Module(body = [node], type_ignores = []), "<load_model>", "exec"), namespace
            )
            return namespace[name]
    raise AssertionError(f"{name} is gone from load_model")


class _Backend:
    def __init__(self):
        self._server_launched_parking = True
        self._server_preempts_kv = True
        self._parking_reserving_mib = None
        self._launched_parking_mib = 8192
        self.killed = 0

    def _kill_process(self):
        self.killed += 1

    def _planned_host_weights_mib(self, *args, **kwargs):
        return 0


@pytest.fixture
def host(monkeypatch):
    # 10 GB free of 16: a quarter of it, 2500 MiB, is what an ordinary launch may pin.
    monkeypatch.setattr(llama_mod, "_available_host_memory_mib", lambda: 10_000)
    monkeypatch.setattr(llama_mod, "_total_host_memory_mib", lambda: 16_000)
    monkeypatch.setattr(llama_mod, "_serving_backends_snapshot", lambda: [])
    monkeypatch.setattr(llama_mod, "_parking_backends", weakref.WeakSet())


class _Parking:
    def __init__(self):
        self.noted = []

    def note_server_parking(self, parks, backend):
        self.noted.append(parks)


def _relaunch(
    backend,
    *,
    wanted = True,
    named = 8192,
    healthy = True,
):
    launched = []
    namespace = dict(vars(llama_mod))
    argv = ["llama-server", "-c", "0", "--kv-unified", "--preempt-ram", str(named)]

    def spawn(run_cmd, *, label = ""):
        launched.append(list(run_cmd))
        namespace["_last_spawn_cmd"] = list(run_cmd)
        return healthy

    namespace.update(
        self = backend,
        cmd = argv,
        env = {llama_mod._exact.CHILD_ENV: "1"},
        _last_spawn_cmd = list(argv),
        _exact_budget = 8192,
        _exact_wanted = wanted,
        kv_cache_unified = True,
        _spawn_and_wait = spawn,
        _finish_cancelled_health_wait = lambda message: False,
        _preemption = _Parking(),
    )
    relaunch = _lift("_relaunch_as_ordinary", namespace)
    return relaunch, launched, namespace


def test_the_child_is_launched_again_on_the_bounded_budget(host):
    backend = _Backend()
    relaunch, launched, namespace = _relaunch(backend)
    assert relaunch((8192, 3000), True) is True
    bounded = llama_mod._host_bounded_parking_mib(8192, backend)
    assert 0 < bounded < 8192
    assert backend.killed == 1 and len(launched) == 1, launched
    assert launched[0][-2:] == ["--preempt-ram", str(bounded)]
    assert backend._launched_parking_mib == bounded
    assert backend._parking_reserving_mib is None
    # Published as unavailable, so the replacement runs as an ordinary server.
    assert llama_mod._exact.CHILD_ENV not in namespace["env"], namespace["env"]


def test_nothing_is_relaunched_without_a_shortfall_or_for_a_bounded_launch(host):
    # The child runs no exact kernels here, so nothing else would call for a relaunch either.
    for args, kwargs in (((None, False), {}), (((8192, 3000), False), {"wanted": False})):
        backend = _Backend()
        relaunch, launched, _ = _relaunch(backend, **kwargs)
        assert relaunch(*args) is True
        assert backend.killed == 0 and launched == []


def test_a_child_still_running_exact_kernels_is_relaunched_without_them(host):
    # Unavailable for another reason (parking it cannot confirm): no shortfall and no budget to
    # bound, but the child still runs exact mode, which the published state says it does not.
    backend = _Backend()
    relaunch, launched, namespace = _relaunch(backend)
    assert relaunch(None, True) is True
    assert backend.killed == 1 and len(launched) == 1, launched
    assert launched[0][-2:] == ["--preempt-ram", "8192"]
    assert llama_mod._exact.CHILD_ENV not in namespace["env"], namespace["env"]


def test_a_budget_the_user_named_is_kept(host):
    backend = _Backend()
    relaunch, launched, _ = _relaunch(backend, named = 12000)
    assert relaunch((12000, 3000), False) is True
    assert launched == []


def test_auto_relaunches_before_it_publishes():
    source = inspect.getsource(LlamaCppBackend.load_model)
    warning = source.index("\" Set exact concurrency to 'on' to fail the load instead.\"")
    call = source.index("_relaunch_as_ordinary(_exact_host_short, _exact_running)", warning)
    assert call < source.index("if not self._publish_healthy():", warning)
