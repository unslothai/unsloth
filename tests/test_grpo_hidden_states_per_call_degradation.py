# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Degradation is a property of the CALL, not of the model.

`_install_grpo_hidden_states_forward_wrapper` asks the model for hidden states
and falls back to real logits when it cannot get them. The dispatch helper reads
that outcome right after the forward returns, so the flag it reads has to
describe the call that just finished.

`_warn_grpo_hidden_states_fallback_once` is warn-once bookkeeping: it only ever
sets its flag. Reading it as the per-call outcome makes it mean "ever degraded",
and a forward can degrade on one batch and succeed on the next -- a
*ForConditionalGeneration that splats **kwargs into a vision tower rejects the
extra flags only on the batches carrying pixel_values. With
`vocab_size == hidden_size` the width test cannot correct that, so every later
hidden-state tensor gets routed to the raw-logits helper: the lm_head matmul is
skipped, log probabilities are wrong and the head gets no gradient.

CPU-only: the model here is a few `torch.randn` calls behind a square head.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _grpo_dispatch_source import load_dispatch_helpers  # noqa: E402
from _rl_source import load_rl_wrapper  # noqa: E402


returns_hidden_states = load_dispatch_helpers()["_unsloth_grpo_returns_hidden_states"]
_RL = load_rl_wrapper()
install_wrapper = _RL["_install_grpo_hidden_states_forward_wrapper"]
drop_positional_kwargs = _RL["_drop_forward_kwargs_consumed_positionally"]

DEGRADED = "_unsloth_grpo_hidden_states_degraded"
WARNED = "_unsloth_grpo_hidden_states_warning_issued"
WIDTH = 8  # square lm_head: vocab_size == hidden_size, the ambiguous case


class _Output:
    def __init__(self, logits, hidden_states = None):
        self.logits = logits
        self.hidden_states = hidden_states


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(WIDTH, WIDTH))


class _SquareModel(torch.nn.Module):
    """Degrades only on the batches that carry `pixel_values`."""

    def __init__(self, mode):
        super().__init__()
        self.lm_head = _Head()
        self.mode = mode

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids = None, pixel_values = None, **kwargs):
        batch, length = input_ids.shape
        hidden = torch.full((batch, length, WIDTH), 0.5)
        logits = torch.full((batch, length, WIDTH), -0.5)
        if pixel_values is not None:
            if self.mode == "typeerror" and (
                "return_dict" in kwargs or "output_hidden_states" in kwargs
            ):
                raise TypeError(
                    "VisionTower.forward() got an unexpected keyword argument 'return_dict'"
                )
            if self.mode == "no_hidden_states":
                return _Output(logits = logits, hidden_states = None)
        return _Output(logits = logits, hidden_states = (hidden,))


@pytest.fixture
def hidden_states_env():
    previous = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = previous


@pytest.mark.parametrize("mode", ["no_hidden_states", "typeerror"])
def test_a_degraded_call_does_not_poison_the_next_call(hidden_states_env, mode):
    model = _SquareModel(mode)
    assert install_wrapper(model) is True
    head = model.lm_head.weight
    ids = torch.zeros(1, 3, dtype = torch.long)

    degraded = model.forward(input_ids = ids, pixel_values = torch.zeros(1, 3, 4, 4))
    assert returns_hidden_states(model, degraded.logits, head) is False

    honoured = model.forward(input_ids = ids)
    assert torch.equal(honoured.logits, torch.full((1, 3, WIDTH), 0.5)), (
        "the second call really did hand back hidden states"
    )
    assert returns_hidden_states(model, honoured.logits, head) is True


@pytest.mark.parametrize("mode", ["no_hidden_states", "typeerror"])
def test_the_warn_once_flag_stays_sticky_for_logging(hidden_states_env, mode):
    model = _SquareModel(mode)
    install_wrapper(model)
    ids = torch.zeros(1, 3, dtype = torch.long)
    model.forward(input_ids = ids, pixel_values = torch.zeros(1, 3, 4, 4))
    model.forward(input_ids = ids)
    assert getattr(model, WARNED) is True, "the warning must not be re-emitted per call"
    assert getattr(model, DEGRADED) is False, "the per-call flag must track the last call"


def test_a_forward_run_with_the_flag_off_reports_real_logits():
    model = _SquareModel("no_hidden_states")
    install_wrapper(model)
    previous = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
    try:
        out = model.forward(input_ids = torch.zeros(1, 3, dtype = torch.long))
    finally:
        if previous is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = previous
    assert torch.equal(out.logits, torch.full((1, 3, WIDTH), -0.5))
    assert returns_hidden_states(model, out.logits, model.lm_head.weight) is False


def test_the_fallback_retry_does_not_reuse_the_rejected_kwargs(hidden_states_env):
    """The retry has to send the caller's original kwargs.

    `_drop_forward_kwargs_consumed_positionally` returns the caller's dict itself
    when there is nothing to drop, which every GRPO call site hits: they pass
    everything by keyword. Mutating it to add `output_hidden_states`/`return_dict`
    would make the fallback re-send exactly what the model just rejected, so the
    TypeError branch would re-raise instead of degrading.
    """
    model = _SquareModel("typeerror")
    install_wrapper(model)
    kwargs = {
        "input_ids": torch.zeros(1, 3, dtype = torch.long),
        "pixel_values": torch.zeros(1, 3, 4, 4),
    }
    out = model.forward(**kwargs)  # must not raise
    assert torch.equal(out.logits, torch.full((1, 3, WIDTH), -0.5))
    assert "return_dict" not in kwargs and "output_hidden_states" not in kwargs


def test_dropping_positional_kwargs_never_hands_back_the_callers_dict():
    import inspect

    def forward(input_ids = None, pixel_values = None, **kwargs):
        pass

    signature = inspect.signature(forward)
    for args in ((), (1,)):
        kwargs = {"input_ids": 1, "pixel_values": 2}
        assert drop_positional_kwargs(signature, args, kwargs) is not None
    # the wrapper copies before mutating, so the caller's dict survives a forward
    kwargs = {"input_ids": 1}
    result = drop_positional_kwargs(signature, (), kwargs)
    dict(result)["output_hidden_states"] = True
    assert "output_hidden_states" not in kwargs
