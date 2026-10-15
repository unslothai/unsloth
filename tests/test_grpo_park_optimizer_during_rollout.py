# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_PARK_OPTIMIZER_DURING_ROLLOUT=1 moves optimizer state and grads to host around GRPO
generation and puts them back unchanged. The helpers are lifted from rl.py with ast and run on
CPU: a torch shim stands in for pinned memory and CUDA sync, and every tensor counts as parkable.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from _rl_source import load_rl_wrapper  # noqa: E402


NAMES = (
    "_park_optimizer_enabled",
    "_is_parkable_tensor",
    "_park_optimizer_to_host",
    "_restore_optimizer_from_host",
    "_wrap_grpo_generate_and_score",
)


def _load():
    fns = load_rl_wrapper(NAMES)
    namespace = fns["_wrap_grpo_generate_and_score"].__globals__
    cuda = types.SimpleNamespace(
        calls = [],
        is_available = lambda: True,
    )
    cuda.synchronize = lambda: cuda.calls.append("synchronize")
    cuda.empty_cache = lambda: cuda.calls.append("empty_cache")

    def empty(*args, pin_memory = False, **kwargs):
        return torch.empty(*args, **kwargs)

    namespace["torch"] = types.SimpleNamespace(Tensor = torch.Tensor, empty = empty, cuda = cuda)
    namespace["_is_parkable_tensor"] = lambda value: isinstance(value, torch.Tensor)
    return fns, cuda


class _Model:
    training = True


def _trainer_after_one_step():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-2)
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    # Mid-accumulation: a fresh grad is pending while the next rollout happens.
    model(torch.randn(2, 4)).sum().backward()
    trainer = types.SimpleNamespace(model = _Model(), optimizer = optimizer)
    return trainer, model, optimizer


def _snapshot(optimizer, model):
    state = {
        (id(p), k): v.clone()
        for p, s in optimizer.state.items()
        for k, v in s.items()
        if isinstance(v, torch.Tensor)
    }
    grads = {id(p): p.grad.clone() for p in model.parameters()}
    return state, grads


def _run(monkeypatch, enabled):
    fns, cuda = _load()
    trainer, model, optimizer = _trainer_after_one_step()
    before = _snapshot(optimizer, model)
    original_state = {
        (id(p), k): v for p, s in optimizer.state.items() for k, v in s.items()
    }
    seen = {}

    class Trainer:
        def _generate_and_score_completions(self, inputs):
            seen["grads"] = [p.grad for p in model.parameters()]
            seen["state_objects_kept"] = all(
                v is original_state[(id(p), k)]
                for p, s in optimizer.state.items()
                for k, v in s.items()
            )
            return inputs

    fns["_wrap_grpo_generate_and_score"](Trainer)
    instance = Trainer()
    instance.__dict__.update(vars(trainer))
    if enabled:
        monkeypatch.setenv("UNSLOTH_PARK_OPTIMIZER_DURING_ROLLOUT", "1")
    else:
        monkeypatch.delenv("UNSLOTH_PARK_OPTIMIZER_DURING_ROLLOUT", raising = False)
    assert instance._generate_and_score_completions("x") == "x"
    return instance, model, optimizer, before, seen, cuda


def test_park_moves_state_and_grads_then_restores_exact_values(monkeypatch):
    instance, model, optimizer, (state_before, grads_before), seen, cuda = _run(monkeypatch, True)
    assert getattr(instance, "_unsloth_park_count", 0) == 1
    assert seen["grads"] == [None, None], "grads must be released during generation"
    assert seen["state_objects_kept"] is False, "state tensors must be swapped for host buffers"
    assert cuda.calls == ["synchronize", "empty_cache"]
    for p, s in optimizer.state.items():
        for k, v in s.items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, state_before[(id(p), k)])
    for p in model.parameters():
        assert torch.equal(p.grad, grads_before[id(p)])

    # The optimizer still steps after the round trip.
    optimizer.step()


def test_park_off_by_default_leaves_everything_alone(monkeypatch):
    instance, model, optimizer, _, seen, cuda = _run(monkeypatch, False)
    assert getattr(instance, "_unsloth_park_count", 0) == 0
    assert all(g is not None for g in seen["grads"])
    assert seen["state_objects_kept"] is True
    assert cuda.calls == []


def test_shared_state_tensor_stays_shared(monkeypatch):
    fns, _ = _load()
    params = [torch.nn.Parameter(torch.zeros(3)) for _ in range(2)]
    optimizer = torch.optim.SGD(params, lr = 0.1)
    qmap = torch.arange(4.0)
    for p in params:
        optimizer.state[p] = {"qmap1": qmap, "state1": torch.ones(3)}
    trainer = types.SimpleNamespace(optimizer = optimizer)
    record = fns["_park_optimizer_to_host"](trainer)
    fns["_restore_optimizer_from_host"](record)
    a, b = (optimizer.state[p]["qmap1"] for p in params)
    assert a is b and torch.equal(a, qmap)


def test_first_step_without_state_or_grads_is_a_noop(monkeypatch):
    fns, cuda = _load()
    model = torch.nn.Linear(2, 2)
    trainer = types.SimpleNamespace(optimizer = torch.optim.AdamW(model.parameters()))
    assert fns["_park_optimizer_to_host"](trainer) is None
    assert cuda.calls == []


def test_gate_skips_without_optimizer_or_under_deepspeed():
    fns, _ = _load()
    enabled = fns["_park_optimizer_enabled"]
    assert enabled(types.SimpleNamespace(optimizer = None)) is False
    assert enabled(types.SimpleNamespace(optimizer = object(), is_deepspeed_enabled = True)) is False
    assert enabled(types.SimpleNamespace(optimizer = object())) is True
    multi = types.SimpleNamespace(world_size = 2)
    assert enabled(types.SimpleNamespace(optimizer = object(), args = multi)) is False
