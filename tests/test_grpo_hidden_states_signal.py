# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A width comparison cannot answer `vocab_size == hidden_size`.

`_get_per_token_logps_and_entropies` sets UNSLOTH_RETURN_HIDDEN_STATES=1 and
then has to decide, per call site, whether `.logits` came back as hidden states
(apply the lm_head) or as real logits (do not). Comparing the last dim against
`lm_head.shape[1]` answers that for every model whose vocab is wider than its
hidden size, which is nearly all of them -- but a model with
`vocab_size == hidden_size` returns real logits that are exactly as wide as its
hidden states. Those logits pass the width test, go through
`chunked_hidden_states_selective_log_softmax`, and get the lm_head applied a
second time. Nothing raises: the matmul is square, so the run keeps going with
silently wrong GRPO log probabilities. In the packed path the per-row verifier
misreads the width the same way, agrees with the corrupted packed result, and
marks the shape trusted.

`_unsloth_grpo_returns_hidden_states` therefore prefers an explicit signal that
the forward honoured the flag, and only falls back to the width comparison when
there is no signal to read. The signal is not invented here: it is the
`__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__` marker `unsloth_zoo.compiler` writes
onto a generated class, and the
`_unsloth_grpo_hidden_states_forward_wrapped` / `..._warning_issued` pair that
`_install_grpo_hidden_states_forward_wrapper` in `unsloth/models/rl.py` keeps on
models the compiler did not rewrite.

Both halves are covered below: the signal decides the ambiguous case, and an
absent signal leaves today's behaviour untouched. CPU-only, tiny shapes, never
skips beyond torch.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _grpo_dispatch_source import (  # noqa: E402
    load_dispatch_helpers,
    load_padded_loop_source,
)

_HELPERS = load_dispatch_helpers()
returns_hidden_states = _HELPERS["_unsloth_grpo_returns_hidden_states"]
hidden_states_signal = _HELPERS["_unsloth_grpo_hidden_states_signal"]

MARKER = "__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__"
WRAPPED = "_unsloth_grpo_hidden_states_forward_wrapped"
DEGRADED = "_unsloth_grpo_hidden_states_warning_issued"




class _Plain:
    """A model Unsloth never touched: no marker, no wrapper.

    `forward` is here because the signal only descends into children that have
    one, which is how it avoids walking into configs and buffers; every real
    nn.Module qualifies.
    """

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _Compiled(_Plain):
    """What unsloth_zoo.compiler emits: the marker lives on the class."""


setattr(_Compiled, MARKER, True)


def _wrapped(degraded = False):
    """What _install_grpo_hidden_states_forward_wrapper leaves on the model."""
    model = _Plain()
    setattr(model, WRAPPED, True)
    if degraded:
        setattr(model, DEGRADED, True)
    return model


def test_an_untouched_model_gives_no_signal():
    assert hidden_states_signal(_Plain()) is None


def test_the_compiler_marker_is_a_positive_signal():
    assert hidden_states_signal(_Compiled()) is True


def test_the_marker_is_found_through_the_peft_base_model():
    """PEFT: the marker sits on the class the compiler rewrote, not the wrapper."""
    peft = SimpleNamespace(get_base_model = lambda: _Compiled())
    assert hidden_states_signal(peft) is True


def test_the_marker_is_found_through_the_ddp_module():
    assert hidden_states_signal(SimpleNamespace(module = _Compiled())) is True


def test_the_trainer_wrapper_is_a_positive_signal():
    assert hidden_states_signal(_wrapped()) is True


def test_a_degraded_trainer_wrapper_is_a_negative_signal():
    """The wrapper records that it could not get hidden states, before returning.

    That is the case the width comparison silently gets wrong.
    """
    assert hidden_states_signal(_wrapped(degraded = True)) is False


def test_the_degradation_is_found_through_the_base_model():
    """The wrapper sets WRAPPED on both objects but DEGRADED only on its target."""
    base = _Plain()
    setattr(base, WRAPPED, True)
    setattr(base, DEGRADED, True)
    outer = SimpleNamespace(base_model = base, forward = lambda: None)
    setattr(outer, WRAPPED, True)
    assert hidden_states_signal(outer) is False


def test_a_broken_get_base_model_does_not_escape():
    def _raise():
        raise RuntimeError("no base model here")

    model = SimpleNamespace(get_base_model = _raise)
    assert hidden_states_signal(model) is None


def test_a_self_referencing_wrapper_chain_terminates():
    model = SimpleNamespace(forward = lambda: None)
    model.model = model
    assert hidden_states_signal(model) is None




def _lm_head(vocab, hidden):
    return torch.zeros(vocab, hidden)


def _tensor(width):
    return torch.zeros(1, 2, width)


class _Poison:
    """Any attribute read raises, so a test can prove the signal was not consulted."""

    def __getattr__(self, name):
        raise RuntimeError(f"the signal should not have been consulted ({name})")


def test_a_vocab_wide_tensor_is_never_hidden_states():
    head = _lm_head(17, 8)
    assert returns_hidden_states(_Compiled(), _tensor(17), head) is False


def test_a_hidden_wide_tensor_is_hidden_states_when_the_width_decides():
    head = _lm_head(17, 8)
    assert returns_hidden_states(_Plain(), _tensor(8), head) is True


@pytest.mark.parametrize("width", [8, 17])
def test_the_signal_is_not_consulted_when_the_width_decides(width):
    """vocab_size != hidden_size: the shape is exact, so nothing may override it."""
    head = _lm_head(17, 8)
    assert returns_hidden_states(_Poison(), _tensor(width), head) == (width == 8)


def test_a_square_lm_head_without_a_signal_keeps_todays_behaviour():
    """The documented fallback: an unsloth_zoo too old to write the marker."""
    head = _lm_head(12, 12)
    assert returns_hidden_states(_Plain(), _tensor(12), head) is True


def test_a_square_lm_head_with_a_positive_signal_stays_on_the_hidden_path():
    head = _lm_head(12, 12)
    assert returns_hidden_states(_Compiled(), _tensor(12), head) is True


def test_a_square_lm_head_with_a_negative_signal_takes_the_raw_logits_path():
    """The bug: real logits that are hidden-width because vocab_size == hidden_size."""
    head = _lm_head(12, 12)
    assert returns_hidden_states(_wrapped(degraded = True), _tensor(12), head) is False


def test_a_negative_signal_cannot_overrule_a_decisive_width_test():
    """vocab_size != hidden_size and the tensor is hidden-wide: it is hidden states.

    Trusting the signal here would send hidden states into the plain
    log-softmax, whose gather indexes with token ids far past the hidden dim.
    """
    head = _lm_head(17, 8)
    assert returns_hidden_states(_wrapped(degraded = True), _tensor(8), head) is True



SQUARE = 12  # vocab_size == hidden_size: the width comparison cannot decide
BATCH, SEQ = 2, 7
KEEP, MAX_LEFT_PAD, MULTIPLIER = 3, 1, 1

_BLOCK = compile(load_padded_loop_source(), "<rl_replacements padded loop>", "exec")


def _eager_hidden_states_log_softmax(
    hidden_states,
    lm_head,
    index,
    chunks = 4,
    logit_scale_multiply = 0.0,
    logit_scale_divide = 0.0,
    logit_softcapping = 0.0,
    temperature = 1.0,
):
    logits = hidden_states.to(lm_head.dtype) @ lm_head.t()
    if logit_scale_multiply != 0.0:
        logits = logits * logit_scale_multiply
    if logit_scale_divide != 0.0:
        logits = logits / logit_scale_divide
    if logit_softcapping != 0.0:
        logits = logit_softcapping * torch.tanh(logits / logit_softcapping)
    return _eager_log_softmax(logits, index, temperature)


def _eager_log_softmax(
    logits,
    index,
    temperature = 1.0,
    chunks = 4,
):
    logits = logits.to(torch.float32)
    if temperature != 1.0:
        logits = logits / temperature
    return torch.gather(
        torch.log_softmax(logits, dim = -1), dim = -1, index = index.unsqueeze(-1)
    ).squeeze(-1)


def _load_zoo_helpers():
    """Prefer the shipped helpers; the eager mirrors above keep this CPU-portable."""
    try:
        from unsloth_zoo.rl_replacements import (
            chunked_hidden_states_selective_log_softmax as zoo_hidden,
            chunked_selective_log_softmax as zoo_raw,
        )
        zoo_hidden(
            torch.zeros(1, 2, SQUARE),
            torch.zeros(SQUARE, SQUARE),
            torch.zeros(1, 2, dtype = torch.long),
            1,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        zoo_raw(torch.zeros(1, 2, SQUARE), torch.zeros(1, 2, dtype = torch.long), 1.0, 1)
    except Exception:
        return _eager_hidden_states_log_softmax, _eager_log_softmax
    return zoo_hidden, zoo_raw


_ZOO_HIDDEN, _ZOO_RAW = _load_zoo_helpers()


class _SquareModel:
    """vocab_size == hidden_size, so both outputs have the same shape."""

    def __init__(self, embedding, lm_head, returns_hidden_states, signal):
        self.embedding = embedding
        self.lm_head = lm_head
        self.returns_hidden_states = returns_hidden_states
        if signal == "compiled":
            # The compiler writes the marker onto the class it generated, so give this instance its own class rather
            self.__class__ = type("_CompiledSquareModel", (_SquareModel,), {MARKER: True})
        elif signal in ("wrapped", "degraded"):
            setattr(self, WRAPPED, True)
            if signal == "degraded":
                setattr(self, DEGRADED, True)

    def __call__(
        self,
        input_ids = None,
        logits_to_keep = None,
        **kwargs,
    ):
        hidden = self.embedding[input_ids]
        out = hidden if self.returns_hidden_states else hidden @ self.lm_head.t()
        assert out.shape[-1] == SQUARE  # the whole point: indistinguishable by width
        return SimpleNamespace(logits = out)


def _run_padded_loop(*, returns_hidden_states, signal):
    generator = torch.Generator().manual_seed(20260803)
    embedding = torch.randn(SQUARE, SQUARE, generator = generator)
    lm_head = torch.randn(SQUARE, SQUARE, generator = generator)
    input_ids = torch.randint(0, SQUARE, (BATCH, SEQ), generator = generator)
    model = _SquareModel(embedding, lm_head, returns_hidden_states, signal)

    namespace = {
        "torch": torch,
        "os": os,
        **_HELPERS,
        "chunked_hidden_states_selective_log_softmax": _ZOO_HIDDEN,
        "chunked_selective_log_softmax": _ZOO_RAW,
        "device_synchronize": lambda *a, **k: None,
        "_get_inference_mode_context_manager": lambda _model: contextlib.nullcontext(),
        "model": model,
        "unwrapped_model": model,
        "self": SimpleNamespace(_autocast_dtype = torch.float32),
        "pixel_values": None,
        "lm_head": lm_head,
        "zipped_inputs": [
            (input_ids[i : i + 1], torch.ones(1, SEQ, dtype = torch.long)) + (None,) * 6
            for i in range(BATCH)
        ],
        "logits_to_keep": KEEP,
        "max_left_pad": MAX_LEFT_PAD,
        "multiplier": MULTIPLIER,
        "logit_scale_multiply": 0,
        "logit_scale_divide": 0,
        "logit_softcapping": 0,
        "temperature": 1.0,
        "all_logprobs_list": [],
        "logprobs": None,
    }
    exec(_BLOCK, namespace)
    return namespace["logprobs"], embedding, lm_head, input_ids


def _reference(embedding, lm_head, input_ids):
    """The only correct answer: one lm_head application, then log-softmax."""
    logits = (embedding[input_ids] @ lm_head.t()).to(torch.float32)
    width = KEEP + MAX_LEFT_PAD
    predictions = torch.log_softmax(logits, dim = -1)[:, -(width + 1) : -1, :]
    targets = input_ids[:, -width:]
    return torch.gather(predictions, dim = -1, index = targets.unsqueeze(-1)).squeeze(-1)


def test_square_lm_head_raw_logits_are_not_run_through_the_lm_head_twice():
    """The regression. Without the signal this returns log_softmax(logits @ W.T).

    Nothing raises, because a square lm_head makes the second matmul legal; the
    run just carries on with wrong log probabilities.
    """
    logprobs, embedding, lm_head, input_ids = _run_padded_loop(
        returns_hidden_states = False, signal = "degraded"
    )
    expected = _reference(embedding, lm_head, input_ids)
    assert logprobs.shape == (BATCH, KEEP + MAX_LEFT_PAD)
    torch.testing.assert_close(logprobs, expected, rtol = 1e-5, atol = 1e-5)


def test_square_lm_head_double_application_is_actually_detectable():
    """Guard against a vacuous assertion above: the wrong answer must differ."""
    generator = torch.Generator().manual_seed(20260803)
    embedding = torch.randn(SQUARE, SQUARE, generator = generator)
    lm_head = torch.randn(SQUARE, SQUARE, generator = generator)
    input_ids = torch.randint(0, SQUARE, (BATCH, SEQ), generator = generator)
    correct = _reference(embedding, lm_head, input_ids)
    doubled = _reference(embedding @ lm_head.t(), lm_head, input_ids)
    assert not torch.allclose(correct, doubled, rtol = 1e-2, atol = 1e-2)


@pytest.mark.parametrize("signal", ["compiled", "wrapped"])
def test_square_lm_head_hidden_states_still_take_the_fused_path(signal):
    logprobs, embedding, lm_head, input_ids = _run_padded_loop(
        returns_hidden_states = True, signal = signal
    )
    expected = _reference(embedding, lm_head, input_ids)
    torch.testing.assert_close(logprobs, expected, rtol = 1e-5, atol = 1e-5)


def test_square_lm_head_without_a_signal_falls_back_to_the_width_test():
    """No marker and no wrapper: unchanged from before, hidden states still work."""
    logprobs, embedding, lm_head, input_ids = _run_padded_loop(
        returns_hidden_states = True, signal = "none"
    )
    expected = _reference(embedding, lm_head, input_ids)
    torch.testing.assert_close(logprobs, expected, rtol = 1e-5, atol = 1e-5)
