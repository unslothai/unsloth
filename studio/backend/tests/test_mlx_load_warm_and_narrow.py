# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A load serves through the batch before it reports loaded, and a smaller context reuses the held weights."""

from __future__ import annotations

import inspect
import threading
from types import SimpleNamespace

import pytest

from core.inference import mlx_inference, orchestrator
from core.inference.mlx_inference import MLXInferenceBackend
from core.inference.runtime_context import runtime_context_length


class _Session:
    def __init__(self, backend, fail):
        self.backend, self.fail, self.rows_in_flight = backend, fail, 0

    def admit(self, request, _handle):
        self.backend.temperatures.append(request["temperature"])
        if self.backend._speculative_draft is not None:
            self.backend._speculative_draft.controller.replies += 1
        self.rows_in_flight = 1

    def step(self):
        self.rows_in_flight = 0
        yield "warm", self.fail

    def close(self):
        self.backend.closed += 1


def _warmed(
    fail = None,
    room = None,
    draft = None,
):
    backend = MLXInferenceBackend.__new__(MLXInferenceBackend)
    vars(backend).update(
        temperatures = [], closed = 0, cleared = 0, _prompt_cache_room = room, _speculative_draft = draft
    )

    def clear():
        backend.cleared += 1
        backend._prompt_cache_room = None

    backend.resident_unavailable_reason = lambda _request: None
    backend.open_resident_batch = lambda *, width: _Session(backend, fail)
    backend._clear_prompt_cache = clear
    backend._warm_resident_batch()
    return backend


def test_warm_serves_greedy_twice_then_sampled_and_clears_the_cache():
    backend = _warmed(room = 123)
    assert backend.temperatures == [0.0, 0.0, 0.7] and backend.closed == 3
    assert backend.cleared == 1 and backend._prompt_cache_room == 123


def test_warm_replies_leave_the_drafter_controller_untaught():
    controller = SimpleNamespace(replies = 0)
    draft = SimpleNamespace(controller = controller)
    _warmed(draft = draft)
    assert draft.controller is controller and controller.replies == 0


def test_the_load_warms_the_batch_with_the_rng_preserved():
    source = inspect.getsource(MLXInferenceBackend.load_model)
    assert "with mlx_rng_preserved():\n                self._warm_resident_batch()" in source


def test_a_failed_warm_leaves_the_load_standing_and_the_cache_clear():
    backend = _warmed(fail = ValueError("row"))
    assert backend.temperatures == [0.0] and backend.closed == 1 and backend.cleared == 1


def test_a_dead_metal_queue_still_fails_the_load(monkeypatch):
    monkeypatch.setattr(mlx_inference, "is_metal_queue_dead", lambda _error: True)
    with pytest.raises(ValueError):
        _warmed(fail = ValueError("queue"))


def _narrowing(**entry):
    backend = MLXInferenceBackend.__new__(MLXInferenceBackend)
    vars(backend).update(
        active_model_name = "org/m",
        models = {"org/m": {"requested_context_length": 8192, **entry}},
        _model = SimpleNamespace(max_seq_length = 8192),
        _is_vlm = False,
        _served_context = 8192,
        _kv_quant = {"requested_kv_bits": None},
        _turboquant = False,
        _prompt_cache_room = None,
        _speculative_draft = None,
        _kv_cache_window = 8192,
        _kv_context_budget = None,
        _resume_files = (("w.safetensors", 1, 1),),
        _resume_weight_files = lambda: (("w.safetensors", 1, 1),),
        cleared = 0,
    )
    backend._clear_prompt_cache = lambda: setattr(backend, "cleared", backend.cleared + 1)
    backend._resolve_context_lengths = lambda model, pinned: (
        runtime_context_length(model, pinned) or 4096,
    )
    backend._resolve_kv_policy = lambda _vlm, _bits, _pinned, served: ({}, served, True, None)
    return backend


def test_a_smaller_context_is_applied_to_the_held_model():
    backend = _narrowing(context_length_fitted = 8192)
    fields = backend.narrow_context(4096)
    assert fields["requested_context_length"] == fields["context_length"] == 4096
    assert backend.models["org/m"]["context_length_fitted"] is None and backend.cleared == 1
    assert backend._served_context == backend._kv_cache_window == 4096


@pytest.mark.parametrize(
    "size, entry, attrs",
    [
        (8193, {}, {}),
        (0, {}, {}),
        (4096, {"mlx_offload": {"ple": {"applied": True}}}, {}),
        (4096, {}, {"_kv_quant": {"requested_kv_bits": 8}}),
        (4096, {}, {"_turboquant": True}),
        (4096, {}, {"_prompt_cache_room": 1}),
        (4096, {"spec_fallback_reason": "does not fit"}, {}),
        (4096, {"spec_fallback_reason": 1}, {"_speculative_draft": SimpleNamespace(drafter = None)}),
        (
            4096,
            {"spec_fallback_reason": 1},
            {"_speculative_draft": SimpleNamespace(drafter = object())},
        ),
        (4096, {}, {"_resume_weight_files": lambda: (("w.safetensors", 2, 2),)}),
        (4096, {}, {"_resume_files": False}),
        (4096, {}, {"_resume_files": None}),
    ],
)
def test_a_load_a_fresh_one_could_plan_differently_is_not_narrowed(size, entry, attrs):
    backend = _narrowing(**entry)
    vars(backend).update(attrs)
    assert backend.narrow_context(size) is None
    assert backend._served_context == backend._model.max_seq_length == 8192 and backend.cleared == 0


def _held(fields = {"context_length": 4096}, **settings):
    proc, sent = SimpleNamespace(is_alive = lambda: True), []
    config = dict(
        device_backend = "mlx", mlx_distributed = False, max_seq_length = 4096, gpu_selection = {"gb": 3}
    )
    config.update(settings)
    backend = orchestrator.InferenceOrchestrator.__new__(orchestrator.InferenceOrchestrator)
    vars(backend).update(
        _proc = proc,
        active_model_name = "org/m",
        models = {"org/m": {"parallel_slots": 1, "context_length": 8192}},
        load_generation = 0,
        _loaded_by_user_action = True,
        _worker_reserved_for = None,
        _active_cancel_events = set(),
        loading_models = {"org/m"},
        _replies_in_flight = lambda: 0,
        _wait_worker_idle = lambda: sent.append("idle") or True,
        _send_cmd = sent.append,
        _wait_response = lambda kind, timeout: {"type": kind, "fields": fields},
        _worker_load_settings = (proc, orchestrator._load_settings({**config, "gpu_selection": {}})),
    )
    for lock in (
        "_dispatcher_lifecycle",
        "_gen",
        "_active_cancel",
        "_send_order",
        "_subprocess_shutdown",
    ):
        setattr(backend, f"{lock}_lock", threading.Lock())
    return backend, config, sent


def test_the_held_worker_takes_a_smaller_context_and_the_parent_mirrors_it():
    backend, config, sent = _held()
    assert backend._narrow_context_in_place(config, "org/m", 1) is True
    assert sent == ["idle", {"type": "narrow_context", "max_seq_length": 4096}]
    assert backend.models["org/m"]["context_length"] == 4096 and backend.load_generation == 1
    assert backend._worker_reserved_for is None and not backend._gen_lock.locked()


@pytest.mark.parametrize("outcome", ["declined", "event", "withdrawn"])
def test_a_load_declined_or_cancelled_meanwhile_is_not_published(outcome):
    backend, config, _sent = _held(fields = None if outcome == "declined" else {"context_length": 1})
    event, answer = threading.Event(), backend._wait_response

    def answered(kind, timeout):
        event.set() if outcome == "event" else None
        backend.loading_models.clear() if outcome == "withdrawn" else None
        return answer(kind, timeout)

    backend._wait_response = answered
    assert backend._narrow_context_in_place(config, "org/m", 1, event) is False
    assert backend.models["org/m"]["context_length"] == 8192 and backend.load_generation == 0


@pytest.mark.parametrize(
    "change",
    [
        lambda b, c: c.update(mlx_kv_quant = "8"),
        lambda b, c: c.update(mlx_distributed = True),
        lambda b, c: c.update(max_seq_length = 8192),
        lambda b, c: setattr(b, "_wait_worker_idle", lambda: False),
        lambda b, c: setattr(b, "_replies_in_flight", lambda: 1),
        lambda b, c: b.models["org/m"].update(parallel_slots = 4),
    ],
)
def test_any_other_change_spawns_fresh(change):
    backend, config, sent = _held()
    change(backend, config)
    assert backend._narrow_context_in_place(config, "org/m", 1) is False
    assert not [cmd for cmd in sent if cmd != "idle"]


@pytest.mark.parametrize("settings", [{"device_backend": "cuda"}, {"mlx_distributed": True}])
def test_only_a_single_mlx_worker_narrows(settings):
    backend, config, sent = _held(**settings)
    assert backend._narrow_context_in_place(config, "org/m", 1) is False and sent == []
