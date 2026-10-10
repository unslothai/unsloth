# SPDX-License-Identifier: AGPL-3.0-only

from types import SimpleNamespace

import pytest

from core.training.eval_loss import use_local_eval_loss


@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("previous", [False, True])
def test_eval_mean_not_multiplied_by_world_size(world_size, previous):
    trainer = SimpleNamespace(args = SimpleNamespace(average_tokens_across_devices = previous))
    calls = []

    def prediction_step(
        model,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
    ):
        calls.append((model, inputs, prediction_loss_only, ignore_keys))
        loss = 0.08
        if trainer.args.average_tokens_across_devices:
            loss *= world_size
        return loss, None, None

    trainer.prediction_step = prediction_step
    use_local_eval_loss(trainer)
    assert trainer.prediction_step("model", {"labels": [1]}, True, ignore_keys = ["logits"]) == (
        0.08,
        None,
        None,
    )
    assert calls == [("model", {"labels": [1]}, True, ["logits"])]
    assert trainer.args.average_tokens_across_devices is previous


def test_eval_failure_restores_training_normalization():
    trainer = SimpleNamespace(args = SimpleNamespace(average_tokens_across_devices = True))

    def prediction_step(*args, **kwargs):
        assert trainer.args.average_tokens_across_devices is False
        raise RuntimeError("evaluation failed")

    trainer.prediction_step = prediction_step
    use_local_eval_loss(trainer)
    with pytest.raises(RuntimeError, match = "evaluation failed"):
        trainer.prediction_step(None, {}, True)
    assert trainer.args.average_tokens_across_devices is True


def test_installing_local_eval_loss_wrapper_twice_is_idempotent():
    trainer = SimpleNamespace(args = SimpleNamespace(average_tokens_across_devices = True))
    calls = []

    def prediction_step(*args, **kwargs):
        calls.append((args, kwargs))
        return "result"

    trainer.prediction_step = prediction_step
    use_local_eval_loss(trainer)
    wrapped = trainer.prediction_step
    use_local_eval_loss(trainer)

    assert trainer.prediction_step is wrapped
    assert trainer.prediction_step("model", {}, True) == "result"
    assert calls == [(("model", {}, True), {})]
    assert trainer.args.average_tokens_across_devices is True
