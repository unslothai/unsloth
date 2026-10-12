# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A retry that drops exact mode prices its parking as a plain launch does.

An exact launch names a budget sized whole for exact parks and skips the host-weight deduction an
ordinary launch takes. When the server refuses the mode, or a flash-attention crash forces it off,
the plain server left behind must get the host-bounded budget less its own weights, or weights
plus parks overrun the host's reserve.
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


def _load_source() -> str:
    return inspect.getsource(LlamaCppBackend.load_model)


def _lift_closure(name: str, namespace: dict):
    tree = ast.parse(textwrap.dedent(_load_source()))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body = [node], type_ignores = [])
            exec(compile(module, "<load_model>", "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"{name} is gone from load_model")


class _Backend:
    def __init__(self, planned_mib: int):
        self.planned_mib = planned_mib
        self._parking_reserving_mib = None

    def _planned_host_weights_mib(self, cmd, detected_gpus, env = None, **kwargs) -> int:
        return self.planned_mib


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(llama_mod, "_available_host_memory_mib", lambda: 100_000)
    monkeypatch.setattr(llama_mod, "_total_host_memory_mib", lambda: 128_000)
    monkeypatch.setattr(llama_mod, "_serving_backends_snapshot", lambda: [])
    monkeypatch.setattr(llama_mod, "_parking_backends", weakref.WeakSet())


def _bound(backend, *, exact_wanted = True, budget = 40_000, done = None):
    namespace = dict(vars(llama_mod))
    namespace.update(
        self = backend,
        _exact_budget = budget,
        _exact_wanted = exact_wanted,
        _parking_off_exact = [] if done is None else done,
        _reachable_gpus = [],
        env = {},
        _child_has_no_gpu = False,
        _shared_gpu_ids = None,
    )
    return _lift_closure("_bound_parking_off_exact", namespace)


class TestTheRetryPricesItsOwnWeights:
    def test_weights_that_fill_the_host_leave_no_parking(self, host):
        backend = _Backend(planned_mib = 90_000)
        cmd = ["llama-server", "--kv-unified", "--preempt-ram", "40000"]
        _bound(backend)(cmd)
        assert "--preempt-ram" not in cmd, (
            f"the plain retry still pins parks past its own weights: {cmd}"
        )

    def test_the_weights_come_off_what_it_may_pin(self, host):
        # What the retry pinned before: the host-bounded budget with no weights taken off.
        backend = _Backend(planned_mib = 80_000)
        without = ["llama-server", "--preempt-ram", "40000"]
        llama_mod._rebound_named_parking_budget(without, 40_000, backend)
        with_weights = ["llama-server", "--preempt-ram", "40000"]
        _bound(backend)(with_weights)
        assert int(with_weights[-1]) < int(without[-1]), (with_weights, without)

    def test_the_weights_stay_reserved_beside_the_budget(self, host):
        backend = _Backend(planned_mib = 20_000)
        cmd = ["llama-server", "--preempt-ram", "40000"]
        _bound(backend)(cmd)
        assert backend._parking_reserving_mib == int(cmd[-1]) + 20_000

    def test_once_per_launch(self, host):
        backend = _Backend(planned_mib = 20_000)
        done: list = []
        cmd = ["llama-server", "--preempt-ram", "40000"]
        _bound(backend, done = done)(cmd)
        first = (list(cmd), backend._parking_reserving_mib)
        _bound(backend, done = done)(cmd)
        assert (cmd, backend._parking_reserving_mib) == first

    def test_a_plain_launch_or_a_named_budget_is_left_alone(self, host):
        cmd = ["llama-server", "--preempt-ram", "40000"]
        _bound(_Backend(90_000), exact_wanted = False)(cmd)
        _bound(_Backend(90_000), budget = None)(cmd)
        assert cmd[-1] == "40000"

    def test_every_retry_that_drops_the_mode_takes_it(self):
        source = _load_source()
        refusal = source.index("_did_exact_retry = True")
        assert "_bound_parking_off_exact(run_cmd)" in source[refusal : refusal + 200]
        for label in ('label = "-noflash")', 'label = "-noflash-mtp")'):
            spawn = source.index(label)
            window = source[spawn - 900 : spawn]
            assert "_drop_exact_for_no_flash(env, _exact_setting)" in window
            assert "_bound_parking_off_exact(_fa_cmd)" in window, label
