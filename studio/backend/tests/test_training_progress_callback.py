# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training progress callbacks must report an active status once training starts.

Both training paths used to leave the parent on the pre-train "Starting ..." status
for the whole run, so /api/train/status and the progress card read "Starting
training..." while the loss was already moving: the callbacks report an empty status
on every log and the parent only overwrites a non-empty one. These tests drive the
real callbacks, worker emit rule and parent handler. Fakes only; no GPU, no model.
"""

from __future__ import annotations

import importlib
import queue as _queue
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# core/training/trainer.py imports unsloth and trl at module level (heavy, GPU init).
# Stub whichever are missing just long enough to import it, then restore so this file
# never pollutes the shared session.
_STUBS = {
    "unsloth": ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"),
    "unsloth.chat_templates": ("get_chat_template",),
    "trl": ("SFTTrainer", "SFTConfig"),
}
_STUBBED: list[str] = []
_TRAINER_PRE_IMPORTED = "core.training.trainer" in sys.modules


def _stub_if_missing(name, attrs):
    """Stub ``name`` unless the real package is installed."""
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:
        pass
    _STUBBED.append(name)
    module = types.ModuleType(name)
    # A spec-less module reads as "no namespace shadow" to ensure_real_packages.
    module.__spec__ = None
    for attr in attrs:
        setattr(module, attr, MagicMock())
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, module)


if not _TRAINER_PRE_IMPORTED:
    for _name, _attrs in _STUBS.items():
        _stub_if_missing(_name, _attrs)

from core.training.trainer import UnslothTrainer  # noqa: E402
from core.training.training import TrainingBackend, _MLXTrainerAdapter  # noqa: E402
from core.training.worker import (  # noqa: E402
    _create_embedding_progress_callback,
    _create_trainer_progress_callback,
)

if not _TRAINER_PRE_IMPORTED:
    for _name in _STUBBED:
        sys.modules.pop(_name, None)
    # Drop the stub-bound module and its parent package (which still holds it as an
    # attribute) so a later test re-imports it against the real packages; the
    # UnslothTrainer class held above stays usable.
    sys.modules.pop("core.training.trainer", None)
    sys.modules.pop("core.training", None)

ACTIVE = "Training in progress..."


class _FakeQueue:
    """Stands in for the mp.Queue the worker sends events on."""

    def __init__(self):
        self.events: list[dict] = []

    def put(self, event, *args, **kwargs):
        self.events.append(event)


def _state():
    return SimpleNamespace(global_step = 0, epoch = 0.0, num_input_tokens_seen = 0)


def _drive(
    callback,
    steps = 3,
    control = None,
    on_step = None,
):
    """Run the HuggingFace callback lifecycle the way Trainer.train() does."""
    state = _state()
    control = control if control is not None else SimpleNamespace(should_training_stop = False)
    callback.on_train_begin(None, state, control)
    for step in range(1, steps + 1):
        state.global_step = step
        state.epoch = round(0.5 * step, 2)
        state.num_input_tokens_seen = 128 * step
        callback.on_log(None, state, control, logs = {"loss": 1.0 / step, "learning_rate": 1e-4})
        callback.on_step_end(None, state, control)
        if on_step is not None:
            on_step(step)
    # Once at the end: HuggingFace calls on_epoch_end per epoch, not per step.
    callback.on_epoch_end(None, state, control)
    return state, control


# ---------------------------------------------------------------------------
# LLM/VLM/audio path: UnslothTrainer._create_progress_callback ->
# worker._create_trainer_progress_callback
# ---------------------------------------------------------------------------


def _make_owner():
    # __new__ dispatches to the MLX adapter on Apple hardware, which has no
    # _create_progress_callback; go straight to the class under test.
    owner = object.__new__(UnslothTrainer)
    UnslothTrainer.__init__(owner)
    owner._update_progress(is_training = True, total_steps = 4, status_message = "Starting training...")
    return owner


def test_train_begin_reports_active_status():
    owner = _make_owner()
    callback = owner._create_progress_callback()

    callback.on_train_begin(None, _state(), SimpleNamespace())

    assert owner.training_progress.status_message == ACTIVE


def test_logging_reports_an_empty_status_so_the_active_one_is_sent_once():
    # The parent keeps the last non-empty status, so a run costs one status event.
    owner = _make_owner()
    reported: list[str] = []
    owner.add_progress_callback(lambda progress: reported.append(progress.status_message))

    _drive(owner._create_progress_callback(), steps = 3)

    assert owner.training_progress.status_message == ""
    assert [status for status in reported if status] == [ACTIVE]
    assert owner.training_progress.step == 3
    assert owner.training_progress.loss == pytest.approx(1 / 3)
    assert owner.training_progress.num_tokens == 384


def test_parent_status_advances_over_the_whole_chain():
    owner = _make_owner()
    backend = TrainingBackend()
    event_queue = _FakeQueue()
    owner.add_progress_callback(_create_trainer_progress_callback(event_queue))
    # The worker sends this right before trainer.train().
    event_queue.put({"type": "status", "message": "Starting training...", "ts": 0.0})

    _drive(owner._create_progress_callback(), steps = 3)
    for event in event_queue.events:
        backend._handle_event(event)

    assert backend._progress.status_message == ACTIVE
    assert backend._progress.step == 3
    assert backend._progress.is_training is True


def test_training_warning_is_emitted_once_and_survives_later_status_updates():
    owner = _make_owner()
    backend = TrainingBackend()
    event_queue = _FakeQueue()
    owner.add_progress_callback(_create_trainer_progress_callback(event_queue))

    owner._record_warning("Evaluation fell back to a held-out training split.")
    owner._record_warning("Evaluation fell back to a held-out training split.")
    owner._update_progress(status_message = ACTIVE)
    for event in event_queue.events:
        backend._handle_event(event)

    warning_events = [event for event in event_queue.events if event["type"] == "warning"]
    assert [event["message"] for event in warning_events] == [
        "Evaluation fell back to a held-out training split."
    ]
    assert backend._progress.warnings == [
        "Evaluation fell back to a held-out training split."
    ]
    assert backend._progress.status_message == ACTIVE


def test_mlx_adapter_deduplicates_warning_events():
    adapter = _MLXTrainerAdapter()

    adapter._handle_event({"type": "warning", "message": "Evaluation was disabled."})
    adapter._handle_event({"type": "warning", "message": "Evaluation was disabled."})
    adapter._handle_event({"type": "warning", "message": "  "})

    assert adapter.training_progress.warnings == ["Evaluation was disabled."]


@pytest.mark.parametrize(
    "stop_status",
    [
        "Stopping training and saving checkpoint...",
        "Cancelling training...",
    ],
)
def test_stop_status_is_never_replaced_by_the_active_one(stop_status):
    owner = _make_owner()
    backend = TrainingBackend()
    event_queue = _FakeQueue()
    owner.add_progress_callback(_create_trainer_progress_callback(event_queue))
    callback = owner._create_progress_callback()

    def _stop_after_first_step(step):
        if step == 1:
            owner.should_stop = True
            owner._update_progress(status_message = stop_status)

    _, control = _drive(callback, steps = 2, on_step = _stop_after_first_step)
    # A resumed run re-enters on_train_begin; an already requested stop must survive.
    callback.on_train_begin(None, _state(), SimpleNamespace())
    for event in event_queue.events:
        backend._handle_event(event)

    assert [e["message"] for e in event_queue.events if e["type"] == "status"] == [
        ACTIVE,
        stop_status,
    ]
    assert backend._progress.status_message == stop_status
    assert control.should_training_stop is True


# ---------------------------------------------------------------------------
# Embedding path: worker._create_embedding_progress_callback
# ---------------------------------------------------------------------------


def _make_embedding_callback(event_queue, should_stop = lambda: False):
    return _create_embedding_progress_callback(
        event_queue,
        total_steps = 4,
        training_start_time = 0.0,
        should_stop = should_stop,
    )


def test_embedding_parent_status_advances_over_the_whole_chain():
    event_queue = _FakeQueue()
    backend = TrainingBackend()
    # The worker sends this right before trainer.train().
    event_queue.put({"type": "status", "message": "Starting embedding training...", "ts": 0.0})

    _drive(_make_embedding_callback(event_queue), steps = 3)
    for event in event_queue.events:
        backend._handle_event(event)

    assert [e["message"] for e in event_queue.events if e["type"] == "status"] == [
        "Starting embedding training...",
        ACTIVE,
    ]
    assert backend._progress.status_message == ACTIVE
    assert backend._progress.step == 3
    assert backend._progress.loss == pytest.approx(1 / 3)
    assert backend._progress.total_steps == 4


def test_embedding_train_begin_reports_nothing_once_a_stop_was_requested():
    event_queue = _FakeQueue()
    control = SimpleNamespace(should_training_stop = False)

    _drive(
        _make_embedding_callback(event_queue, should_stop = lambda: True), steps = 1, control = control
    )

    assert [e for e in event_queue.events if e["type"] == "status"] == []
    assert control.should_training_stop is True


def test_embedding_callback_survives_a_real_queue():
    # The worker's queue is an mp.Queue; nothing put on it may be unpicklable.
    import pickle

    event_queue = _queue.Queue()
    _drive(_make_embedding_callback(event_queue), steps = 1)

    events = [event_queue.get_nowait() for _ in range(event_queue.qsize())]
    assert [e["type"] for e in events] == ["status", "progress"]
    assert pickle.loads(pickle.dumps(events)) == events
