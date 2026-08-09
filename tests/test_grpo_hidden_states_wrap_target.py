# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The GRPO hidden-states fallback must wrap the module that owns the head.

TRL builds GRPO's `ref_model` as a bare `*ForCausalLM`, and that also has a
`.model`, so walking `("base_model", "model")` landed the wrapper on the decoder
body. Nothing raised: the head above ran untouched and the caller silently got
[B, T, vocab] where it expects [B, T, hidden], which blows up later as a reduction
dim mismatch in `chunked_hidden_states_selective_log_softmax`.
"""

import contextlib
import os
from types import MethodType

import pytest

torch = pytest.importorskip("torch")

import unsloth  # noqa: F401,E402  (must be imported before transformers)
from transformers import Qwen2Config  # noqa: E402
from unsloth.models.rl import (  # noqa: E402
    _grpo_hidden_states_wrap_target,
    _install_grpo_hidden_states_forward_wrapper,
    _module_returns_logits,
)


def _tiny_causal_lm():
    """A real transformers `*ForCausalLM`, shaped like TRL's `ref_model`."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    config = Qwen2Config(
        num_hidden_layers = 2,
        hidden_size = 64,
        intermediate_size = 128,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 128,
        max_position_embeddings = 64,
        pad_token_id = None,
        tie_word_embeddings = False,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(config).eval(), config


@contextlib.contextmanager
def _return_hidden_states(value):
    """Pin the switch, then restore the caller's environment exactly, unset included."""
    previous = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = previous


class _Wrapper(torch.nn.Module):
    """An adapter-shaped wrapper: `.model` is itself a head-owning model."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def test_decoder_body_is_not_a_wrap_target():
    model, _ = _tiny_causal_lm()
    assert _module_returns_logits(model)
    assert not _module_returns_logits(model.model)
    assert _grpo_hidden_states_wrap_target(model) is model


def test_adapter_style_wrapper_is_still_unwrapped():
    model, _ = _tiny_causal_lm()
    wrapper = _Wrapper(model)
    assert _grpo_hidden_states_wrap_target(wrapper) is model


def test_plain_causal_lm_returns_hidden_states_after_the_wrapper():
    model, config = _tiny_causal_lm()
    assert _install_grpo_hidden_states_forward_wrapper(model) is True

    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    with _return_hidden_states("1"), torch.no_grad():
        wrapped = model(input_ids = input_ids).logits

    assert wrapped.shape == (
        2,
        6,
        config.hidden_size,
    ), f"expected hidden states of width {config.hidden_size}, got {tuple(wrapped.shape)}"

    # Must be the hidden states the head consumes, or the logprobs are wrong rather
    # than merely mis-shaped.
    with _return_hidden_states("0"), torch.no_grad():
        reference = model(input_ids = input_ids).logits
    lm_head = model.get_output_embeddings().weight
    assert reference.shape == (2, 6, config.vocab_size)
    assert torch.allclose(wrapped @ lm_head.t(), reference, atol = 1e-4)


def test_the_switch_is_still_honoured():
    """Off means off: the wrapper must not change the default output."""
    model, config = _tiny_causal_lm()
    _install_grpo_hidden_states_forward_wrapper(model)

    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    with _return_hidden_states("0"), torch.no_grad():
        out = model(input_ids = input_ids).logits
    assert out.shape == (1, 4, config.vocab_size)


def test_survives_the_accelerate_forward_rebind():
    """accelerate's `extract_model_from_parallel(keep_fp32_wrapper = False)`, which the
    GRPO loop calls every step, rebinds an instance forward as `MethodType(forward,
    model)`, so the module arrives as a leading positional argument."""
    model, config = _tiny_causal_lm()
    assert _install_grpo_hidden_states_forward_wrapper(model) is True
    model.forward = MethodType(model.forward, model)

    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    with _return_hidden_states("1"), torch.no_grad():
        wrapped = model(input_ids = input_ids).logits
    assert wrapped.shape == (2, 6, config.hidden_size)

    with _return_hidden_states("0"), torch.no_grad():
        reference = model(input_ids = input_ids).logits
    assert reference.shape == (2, 6, config.vocab_size)
