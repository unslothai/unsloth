# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The fast CausalLM forwards honour UNSLOTH_RETURN_HIDDEN_STATES, so GRPO's fallback wrapper must not wrap them.

The wrapper forces ``output_hidden_states=True`` and keeps every layer's state alive until the forward returns. The
fast forwards (``CausalLM_fast_forward`` in llama.py, ``MistralForCausalLM_fast_forward``) return the head input
directly. On CPU the fast forward runs over the stock transformers decoder body, so the head logic under test is
the real one, and the wrapped stock forward and the native fast forward are compared on the same weights.
"""

import contextlib
import os

import pytest

torch = pytest.importorskip("torch")

import unsloth  # noqa: F401,E402  (must be imported before transformers)
from transformers import MistralConfig, Qwen2Config  # noqa: E402
from transformers.models.mistral.modeling_mistral import MistralForCausalLM  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM  # noqa: E402

from unsloth.models import rl  # noqa: E402
from unsloth.models.llama import (  # noqa: E402
    CausalLM_fast_forward,
    LlamaModel_fast_forward_inference,
)
from unsloth.models.mistral import MistralForCausalLM_fast_forward  # noqa: E402
from unsloth.models.rl_replacements import _unsloth_grpo_hidden_states_signal  # noqa: E402

_WRAPPED = "_unsloth_grpo_hidden_states_forward_wrapped"
_CONFIG = dict(
    num_hidden_layers = 3,
    hidden_size = 64,
    intermediate_size = 128,
    num_attention_heads = 4,
    num_key_value_heads = 2,
    vocab_size = 160,
    max_position_embeddings = 128,
    pad_token_id = None,
    tie_word_embeddings = False,
)


def _qwen2_fast_cls():
    return type(
        "FastQwen2ForCausalLM",
        (Qwen2ForCausalLM,),
        {"forward": CausalLM_fast_forward(LlamaModel_fast_forward_inference)},
    )


def _mistral_fast_cls():
    return type(
        "FastMistralForCausalLM",
        (MistralForCausalLM,),
        {"forward": MistralForCausalLM_fast_forward},
    )


FAMILIES = {
    "qwen2": (Qwen2ForCausalLM, _qwen2_fast_cls, Qwen2Config),
    "mistral": (MistralForCausalLM, _mistral_fast_cls, MistralConfig),
}


def _pair(family):
    """The stock class and its fast-forward subclass, same weights."""
    stock_cls, fast_factory, config_cls = FAMILIES[family]
    config = config_cls(**_CONFIG)
    if family == "mistral":
        config.sliding_window = None
    torch.manual_seed(0)
    stock = stock_cls(config).eval()
    fast = fast_factory()(config).eval()
    fast.load_state_dict(stock.state_dict())
    return stock, fast


@contextlib.contextmanager
def _return_hidden_states(value):
    previous = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = previous


def _logps(hidden, head, ids, logits_to_keep):
    """TRL's per-token logps: drop the last position, score the completion window."""
    logits = head(hidden)[:, :-1, :].float()
    targets = ids[:, -logits_to_keep:]
    return torch.log_softmax(logits, dim = -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def test_fast_forward_functions_carry_the_marker():
    assert getattr(
        CausalLM_fast_forward(LlamaModel_fast_forward_inference),
        "__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__",
        False,
    )
    assert getattr(
        MistralForCausalLM_fast_forward, "__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__", False
    )


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_wrapper_installed_on_stock_forward_skipped_on_fast_forward(family):
    stock, fast = _pair(family)
    assert rl._model_supports_unsloth_return_hidden_states(stock) is False
    assert rl._install_grpo_hidden_states_forward_wrapper(stock) is True
    assert getattr(stock, _WRAPPED, False) is True

    assert rl._model_supports_unsloth_return_hidden_states(fast) is True
    assert rl._install_grpo_hidden_states_forward_wrapper(fast) is False
    assert getattr(fast, _WRAPPED, False) is False
    assert "forward" not in vars(fast)
    assert _unsloth_grpo_hidden_states_signal(fast) is True


def test_fast_forward_under_an_adapter_is_skipped():
    """A PEFT-shaped holder: the wrap target is the head owner, whose forward is the fast one."""

    class Holder(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner

        def get_output_embeddings(self):
            return self.model.get_output_embeddings()

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)

    _, fast = _pair("qwen2")
    holder = Holder(fast)
    assert rl._grpo_hidden_states_wrap_target(holder) is fast
    assert rl._install_grpo_hidden_states_forward_wrapper(holder) is False
    assert getattr(fast, _WRAPPED, False) is False
    assert _unsloth_grpo_hidden_states_signal(holder) is True


def test_replaced_forward_loses_the_marker():
    """The marker lives on the function, so a model whose forward is swapped out gets the wrapper again."""
    _, fast = _pair("qwen2")
    original = fast.forward

    def other_forward(*args, **kwargs):
        return original(*args, **kwargs)

    fast.forward = other_forward
    assert rl._model_supports_unsloth_return_hidden_states(fast) is False
    assert rl._install_grpo_hidden_states_forward_wrapper(fast) is True


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("grad", [False, True])
def test_native_hidden_states_and_logps_match_the_wrapper(family, grad):
    stock, fast = _pair(family)
    assert rl._install_grpo_hidden_states_forward_wrapper(stock) is True
    assert rl._install_grpo_hidden_states_forward_wrapper(fast) is False
    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(0, _CONFIG["vocab_size"], (3, 24), generator = generator)
    mask = torch.ones_like(ids)
    mask[0, :5] = 0  # left padding, as GRPO batches have
    logits_to_keep = 9
    context = contextlib.nullcontext() if grad else torch.no_grad()
    with context, _return_hidden_states("1"):
        wrapped = stock(input_ids = ids, attention_mask = mask, logits_to_keep = logits_to_keep + 1)
        native = fast(input_ids = ids, attention_mask = mask, logits_to_keep = logits_to_keep + 1)
    assert wrapped.hidden_states is None and native.hidden_states is None
    assert (
        wrapped.logits.shape
        == native.logits.shape
        == (3, logits_to_keep + 1, _CONFIG["hidden_size"])
    )
    assert torch.equal(wrapped.logits, native.logits)

    head = fast.get_output_embeddings()
    logps_native = _logps(native.logits, head, ids, logits_to_keep)
    logps_wrapped = _logps(wrapped.logits, stock.get_output_embeddings(), ids, logits_to_keep)
    assert torch.equal(logps_native, logps_wrapped)

    with torch.no_grad(), _return_hidden_states("0"):
        real = fast(input_ids = ids, attention_mask = mask, logits_to_keep = logits_to_keep + 1).logits
    reference = torch.log_softmax(real[:, :-1, :].float(), dim = -1)
    reference = reference.gather(-1, ids[:, -logits_to_keep:].unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(logps_native.detach(), reference, rtol = 0, atol = 1e-6)

    if grad:
        logps_native.sum().backward()
        logps_wrapped.sum().backward()
        for (name, a), (_, b) in zip(fast.named_parameters(), stock.named_parameters()):
            assert torch.equal(a.grad, b.grad), name


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_return_dict_false_still_returns_hidden_states(family):
    # The wrapper used to force return_dict = True; without it, a model configured with return_dict = False must
    # still hand GRPO its hidden states in .logits, the same ones as with return_dict = True.
    _, fast = _pair(family)
    ids = torch.randint(
        0, _CONFIG["vocab_size"], (2, 12), generator = torch.Generator().manual_seed(2)
    )
    mask = torch.ones_like(ids)
    with torch.no_grad(), _return_hidden_states("1"):
        as_dict = fast(input_ids = ids, attention_mask = mask, logits_to_keep = 5).logits
        fast.config.return_dict = False
        as_tuple = fast(input_ids = ids, attention_mask = mask, logits_to_keep = 5).logits
    assert as_tuple.shape == (2, 5, _CONFIG["hidden_size"])
    assert torch.equal(as_tuple, as_dict)
