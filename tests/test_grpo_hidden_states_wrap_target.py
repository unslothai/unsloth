# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The GRPO hidden-states fallback must wrap the module that owns the head.

`_install_grpo_hidden_states_forward_wrapper` is the safety net for a model
whose forward does not honour UNSLOTH_RETURN_HIDDEN_STATES. TRL builds GRPO's
`ref_model` with a plain `architecture.from_pretrained(model_id)`, so on every
non-PEFT (full finetuning) GRPO run the net has to catch a bare
`*ForCausalLM`.

`_grpo_hidden_states_wrap_target` walked `("base_model", "model")` to find the
adapter's wrapped model. A bare `*ForCausalLM` also has `.model` -- its decoder
body -- so the wrapper landed on the decoder. The decoder returns no `.logits`
at all, the head above it then ran untouched, and the caller got
[B, T, vocab] where it expects [B, T, hidden]. No warning was emitted, because
from the wrapper's point of view nothing had failed.

Downstream that is the lm_head matmul in
`chunked_hidden_states_selective_log_softmax`:

    a and b must have same reduction dim, but got
    [((s47*s87 + 255)//256), s33] X [1536, 151936]
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():                       # unsloth needs an accelerator to import
    pytest.skip("needs an accelerator to import unsloth", allow_module_level = True)

import os

import unsloth  # noqa: F401  (must be imported before transformers)
from transformers import AutoConfig
from unsloth.models.rl import (
    _grpo_hidden_states_wrap_target,
    _install_grpo_hidden_states_forward_wrapper,
    _module_returns_logits,
)


def _tiny_causal_lm():
    """A real transformers `*ForCausalLM`, shaped like TRL's `ref_model`."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    config = AutoConfig.from_pretrained("unsloth/Qwen2.5-0.5B-Instruct")
    config.num_hidden_layers   = 2
    config.hidden_size         = 64
    config.intermediate_size   = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.vocab_size          = 128
    config.pad_token_id        = None
    config.tie_word_embeddings = False
    torch.manual_seed(0)
    return Qwen2ForCausalLM(config).eval(), config


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

    previous = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        with torch.no_grad():
            wrapped = model(input_ids = input_ids).logits
    finally:
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = previous

    assert wrapped.shape == (2, 6, config.hidden_size), (
        f"expected hidden states of width {config.hidden_size}, got {tuple(wrapped.shape)}"
    )

    # The hidden states must be the ones the head consumes, or every logprob
    # computed from them is wrong rather than merely differently shaped.
    with torch.no_grad():
        reference = model(input_ids = input_ids).logits
    lm_head = model.get_output_embeddings().weight
    assert reference.shape == (2, 6, config.vocab_size)
    assert torch.allclose(wrapped @ lm_head.t(), reference, atol = 1e-4)


def test_the_switch_is_still_honoured():
    """Off means off: the wrapper must not change the default output."""
    model, config = _tiny_causal_lm()
    _install_grpo_hidden_states_forward_wrapper(model)

    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    previous = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
    try:
        with torch.no_grad():
            out = model(input_ids = input_ids).logits
    finally:
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = previous
    assert out.shape == (1, 4, config.vocab_size)
