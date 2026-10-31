# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Early stopping on eval loss, driven through a real transformers Trainer on CPU."""

import math

import pytest
import torch
from torch import nn
from transformers import Trainer, TrainingArguments

from core.training import early_stopping as es
from core.training.early_stopping import attach_early_stopping, early_stopping_patience


class _Line(nn.Module):
    """y = w, so loss is (w - target)^2 and the step direction is fully determined by the data."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x,
        labels = None,
    ):
        pred = self.w.expand_as(x)
        return {"loss": ((pred - labels) ** 2).mean(), "logits": pred}


class _Rows(torch.utils.data.Dataset):
    def __init__(
        self,
        target,
        n = 8,
    ):
        self.target, self.n = target, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"x": torch.ones(1), "labels": torch.full((1,), float(self.target))}


def _trainer(
    tmp_path,
    *,
    eval_target = -1.0,
    max_steps = 20,
    eval_dataset = True,
    **args,
):
    # Training pulls w toward +1 and evaluation scores it against eval_target, so with -1 every step makes
    # eval loss worse and with +1 every step makes it better.
    config = dict(
        output_dir = str(tmp_path),
        per_device_train_batch_size = 2,
        learning_rate = 0.05,
        max_steps = max_steps,
        eval_strategy = "steps",
        eval_steps = 1,
        save_strategy = "no",
        logging_steps = 1,
        report_to = "none",
        use_cpu = True,
        lr_scheduler_type = "constant",
        optim = "sgd",
        disable_tqdm = True,
    )
    config.update(args)
    return Trainer(
        model = _Line(),
        args = TrainingArguments(**config),
        train_dataset = _Rows(1.0),
        eval_dataset = _Rows(eval_target) if eval_dataset else None,
    )


def _w_after_steps(tmp_path, steps):
    trainer = _trainer(tmp_path / f"ref{steps}", max_steps = steps, eval_strategy = "no")
    trainer.train()
    return trainer.model.w.item()


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, 0),
        (0, 0),
        (3, 3),
        ("4", 4),
        (2.9, 2),
        (True, 0),
        (-1, 0),
        (float("nan"), 0),
        (float("inf"), 0),
        ("abc", 0),
        (10**9, es.MAX_EARLY_STOPPING_PATIENCE),
    ],
)
def test_patience_coercion(value, expected):
    assert early_stopping_patience(value) == expected


def test_stops_after_patience_and_keeps_the_best_weights(tmp_path):
    trainer = _trainer(tmp_path)
    callback = attach_early_stopping(trainer, 2)
    trainer.train()

    # Step 1 is the first evaluation and so the best; steps 2 and 3 are worse, which is patience 2.
    assert trainer.state.global_step == 3
    assert callback.stopped_at == 3 and callback.best_step == 1 and callback.restored
    assert trainer.model.w.item() == pytest.approx(_w_after_steps(tmp_path, 1))
    assert trainer.model.w.item() != pytest.approx(_w_after_steps(tmp_path, 3))


def test_improving_run_is_untouched(tmp_path):
    trainer = _trainer(tmp_path, eval_target = 1.0, max_steps = 6)
    callback = attach_early_stopping(trainer, 2)
    trainer.train()

    assert trainer.state.global_step == 6
    assert callback.stopped_at is None and not callback.restored and callback.best_step == 6
    assert trainer.model.w.item() == pytest.approx(_w_after_steps(tmp_path, 6))


def test_run_ending_on_worse_evals_still_saves_the_best(tmp_path):
    trainer = _trainer(tmp_path, max_steps = 3)
    callback = attach_early_stopping(trainer, 5)
    trainer.train()

    assert trainer.state.global_step == 3 and callback.stopped_at is None
    assert callback.restored and callback.best_step == 1
    assert trainer.model.w.item() == pytest.approx(_w_after_steps(tmp_path, 1))


def test_off_or_without_evaluation_adds_nothing(tmp_path):
    for trainer, value in [
        (_trainer(tmp_path / "a"), 0),
        (_trainer(tmp_path / "b"), None),
        (_trainer(tmp_path / "c", eval_dataset = False, eval_strategy = "no"), 3),
        (_trainer(tmp_path / "d", eval_strategy = "no"), 3),
    ]:
        before = list(trainer.callback_handler.callbacks)
        assert attach_early_stopping(trainer, value) is None
        assert trainer.callback_handler.callbacks == before


def _feed(callback, losses):
    class _State:
        global_step = 0

    class _Control:
        should_training_stop = False

    state, control = _State(), _Control()
    for step, loss in enumerate(losses, start = 1):
        state.global_step = step
        callback.on_evaluate(None, state, control, metrics = {"eval_loss": loss})
    return control


def test_nan_eval_loss_counts_against_patience():
    callback = es.create_early_stopping_callback(2)
    control = _feed(callback, [1.0, float("nan"), float("nan")])
    assert control.should_training_stop and callback.best_loss == 1.0 and callback.stopped_at == 3


def test_a_nan_first_eval_never_becomes_the_best():
    callback = es.create_early_stopping_callback(2)
    control = _feed(callback, [float("nan"), 0.9, 0.8])
    assert not control.should_training_stop
    assert callback.best_loss == 0.8 and callback.best_step == 3 and callback.bad_evals == 0
    assert not math.isnan(callback.best_loss)


def test_a_user_stop_keeps_the_current_weights(tmp_path):
    # Stop-and-save writes a resume checkpoint with this step's optimizer state, so the weights must match it.
    from transformers import TrainerCallback

    class _UserStop(TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            if state.global_step == 2:
                control.should_training_stop = True

    trainer = _trainer(tmp_path)
    callback = attach_early_stopping(trainer, 5)
    trainer.add_callback(_UserStop())
    trainer.train()

    assert (
        trainer.state.global_step == 2 and callback.bad_evals == 1 and callback.stopped_at is None
    )
    assert not callback.restored
    assert trainer.model.w.item() == pytest.approx(_w_after_steps(tmp_path, 2))


def test_too_large_to_copy_still_stops_with_the_final_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(es, "MAX_BEST_WEIGHTS_BYTES", 0)
    trainer = _trainer(tmp_path)
    callback = attach_early_stopping(trainer, 2)
    trainer.train()

    assert trainer.state.global_step == 3 and callback.stopped_at == 3 and not callback.restored
    assert trainer.model.w.item() == pytest.approx(_w_after_steps(tmp_path, 3))


def test_request_model_accepts_and_validates_patience():
    from pydantic import ValidationError

    from models.training import TrainingStartRequest

    base = dict(model_name = "x", format_type = "auto", hf_dataset = "a/b", training_type = "LoRA/QLoRA")
    assert TrainingStartRequest(**base).early_stopping_patience == 0
    assert TrainingStartRequest(**base, early_stopping_patience = 3).early_stopping_patience == 3
    for bad in (True, -1, 2.5, 1001):
        with pytest.raises(ValidationError):
            TrainingStartRequest(**base, early_stopping_patience = bad)


def test_worker_config_carries_patience():
    from core.training.training import _build_training_worker_config

    assert _build_training_worker_config({"model_name": "m"})["early_stopping_patience"] == 0
    config = _build_training_worker_config({"model_name": "m", "early_stopping_patience": 4})
    assert config["early_stopping_patience"] == 4


def test_mlx_worker_hands_patience_to_zoo_only_when_it_can_take_it():
    # MLXTrainingConfig implements patience and best-adapter restore itself; the worker only forwards the
    # setting, gated on the fields so an older unsloth-zoo still constructs its config.
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py").read_text(
        encoding = "utf-8"
    )
    gate = source.index('"early_stopping_patience" in _supported_fields')
    block = source[gate : source.index('if "append_eos" in _supported_fields:', gate)]
    assert '"load_best_model_at_end" in _supported_fields' in block
    assert "if patience and eval_dataset is not None and eval_steps_value > 0:" in block
    assert 'mlx_config_kwargs["early_stopping_patience"] = patience' in block
    assert 'mlx_config_kwargs["load_best_model_at_end"] = True' in block
    assert source.index("mlx_config_kwargs = dict(") < gate < source.index("trainer = MLXTrainer(")


def test_mlx_manual_stop_turns_off_the_best_adapter_restore():
    # Zoo restores the best adapters before the stop checkpoint is written; a manual stop must keep the current ones.
    import queue
    import time
    import types

    from core.training import worker

    stop_queue = queue.Queue()
    _, _, trainer_ref, _, stop_thread = worker._start_mlx_stop_poller(stop_queue)
    trainer = types.SimpleNamespace(
        stop_requested = False, args = types.SimpleNamespace(load_best_model_at_end = True)
    )
    trainer_ref[0] = trainer
    stop_queue.put({"type": "stop", "save": True})
    deadline = time.time() + 5
    while not trainer.stop_requested and time.time() < deadline:
        time.sleep(0.05)
    assert trainer.stop_requested and trainer.args.load_best_model_at_end is False
    stop_queue.put({"type": "stop", "save": False})
    stop_thread.join(timeout = 5)


def test_status_reports_an_early_stop_even_without_a_restore(tmp_path, monkeypatch):
    import types

    from core.training.trainer import UnslothTrainer

    monkeypatch.setattr(es, "MAX_BEST_WEIGHTS_BYTES", 0)
    trainer = _trainer(tmp_path)
    callback = attach_early_stopping(trainer, 2)
    trainer.train()
    summary = UnslothTrainer._early_stopping_summary(
        types.SimpleNamespace(_early_stopping = callback)
    )
    assert summary == "Stopped early at step 3. "
    assert callback._best_weights is None


def test_the_best_weight_copy_is_released_when_not_needed(tmp_path):
    trainer = _trainer(tmp_path, eval_target = 1.0, max_steps = 4)
    callback = attach_early_stopping(trainer, 2)
    trainer.train()
    assert not callback.restored and callback._best_weights is None


def test_a_new_best_refills_the_copy_in_place():
    # Building a fresh dict per improvement would hold two full copies of the trainable weights at once.
    callback = es.create_early_stopping_callback(3)
    model = _Line()

    class _State:
        global_step = 1

    callback.on_evaluate(None, _State(), None, metrics = {"eval_loss": 2.0}, model = model)
    first = callback._best_weights["w"]
    with torch.no_grad():
        model.w.fill_(5.0)
    _State.global_step = 2
    callback.on_evaluate(None, _State(), None, metrics = {"eval_loss": 1.0}, model = model)
    assert callback._best_weights["w"] is first and first.item() == 5.0


def test_mlx_early_stop_is_named_in_the_final_status():
    import types

    from core.training import worker

    def finish(early_stopped):
        events = []
        trainer = types.SimpleNamespace(
            distributed_world_size = 1,
            is_main_process = True,
            stop_requested = False,
            _early_stopped = early_stopped,
            save_model = lambda path: None,
        )
        worker._finalize_mlx_training(
            trainer,
            lambda: (False, True),
            "/out",
            lambda: None,
            lambda kind, **payload: events.append((kind, payload)),
            lambda: True,
        )
        return events[-1]

    assert finish(False) == (
        "complete",
        {"output_dir": "/out", "status_message": "Training completed"},
    )
    assert finish(True)[1]["status_message"] == (
        "Training completed (stopped early: eval loss stopped improving)"
    )


def test_steps_after_the_best_evaluation_roll_back_like_load_best_model_at_end(tmp_path):
    # Zoo's MLX trainer restores the best adapters whenever the run finishes; the CUDA path keeps the same contract.
    trainer = _trainer(tmp_path, eval_target = 1.0, max_steps = 5, eval_steps = 2)
    callback = attach_early_stopping(trainer, 3)
    trainer.train()

    # transformers 5 also evaluates the final step, leaving no tail; older releases stop evaluating at step 4.
    assert trainer.state.global_step == 5 and callback.best_step in (4, 5)
    assert callback.restored == (callback.best_step == 4)
    assert trainer.model.w.item() == pytest.approx(_w_after_steps(tmp_path, callback.best_step))


def test_a_user_stop_on_the_step_patience_runs_out_still_keeps_the_current_weights(tmp_path):
    trainer = _trainer(tmp_path)
    callback = attach_early_stopping(trainer, 2, user_stopped = lambda: True)
    trainer.train()

    assert trainer.state.global_step == 3 and callback.stopped_at == 3 and not callback.restored
    assert trainer.model.w.item() == pytest.approx(_w_after_steps(tmp_path, 3))


def test_a_finished_run_rolls_back_unevaluated_steps():
    callback = es.create_early_stopping_callback(3)
    model = _Line()

    class _State:
        global_step = 4
        max_steps = 5

    callback.on_evaluate(None, _State(), None, metrics = {"eval_loss": 1.0}, model = model)
    with torch.no_grad():
        model.w.fill_(7.0)
    _State.global_step = 5
    callback.on_train_end(None, _State(), None, model = model)
    assert callback.restored and model.w.item() == 0.0 and callback._best_weights is None


def test_running_out_of_patience_on_the_last_step_is_not_an_early_stop(tmp_path):
    trainer = _trainer(tmp_path, max_steps = 3)
    callback = attach_early_stopping(trainer, 2)
    trainer.train()

    assert (
        trainer.state.global_step == 3 and callback.bad_evals == 2 and callback.stopped_at is None
    )
    assert callback.restored and trainer.model.w.item() == pytest.approx(
        _w_after_steps(tmp_path, 1)
    )
