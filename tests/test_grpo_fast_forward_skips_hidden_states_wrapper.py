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
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

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
from unsloth.models.rl_replacements import (  # noqa: E402
    _unsloth_grpo_hidden_states_signal,
    _unsloth_grpo_returns_hidden_states,
    _unsloth_pref_returns_hidden_states,
)

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
    # The marker is the function itself, so functools.wraps copies of it are told apart by identity.
    fast_forward = CausalLM_fast_forward(LlamaModel_fast_forward_inference)
    assert getattr(fast_forward, "__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__", None) is fast_forward
    assert (
        getattr(MistralForCausalLM_fast_forward, "__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__", None)
        is MistralForCausalLM_fast_forward
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


def _square_pair(family = "qwen2"):
    """A square head (vocab_size == hidden_size): only the capability signal can tell hidden states from logits."""
    stock_cls, fast_factory, config_cls = FAMILIES[family]
    config = config_cls(**{**_CONFIG, "vocab_size": _CONFIG["hidden_size"]})
    if family == "mistral":
        config.sliding_window = None
    torch.manual_seed(0)
    stock = stock_cls(config).eval()
    fast = fast_factory()(config).eval()
    fast.load_state_dict(stock.state_dict())
    return stock, fast


def _gated_cls(family = "qwen2"):
    """The class shape pre_patch leaves behind: the fast forward behind the _unsloth_fast_path gate."""
    from unsloth.models.llama import _gated_method

    stock_cls, fast_factory, _ = FAMILIES[family]
    cls = type(f"Gated{stock_cls.__name__}", (stock_cls,), {})
    cls.forward = _gated_method(cls, "forward", vars(fast_factory())["forward"], stock_cls.forward)
    return cls


def _amp_prepare(model):
    """What Accelerator.prepare_model does for native AMP (accelerate/accelerator.py): autocast, then fp32 outputs."""
    from types import MethodType

    from accelerate.utils.operations import convert_outputs_to_fp32

    model._original_forward = model.forward
    autocast_context = torch.autocast(device_type = "cpu", dtype = torch.bfloat16)
    model.forward = MethodType(autocast_context(model.forward.__func__), model)
    model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)
    return model


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_decorated_logits_forward_does_not_inherit_capability(family):
    """functools.wraps copies the marker onto a replacement that returns real logits: it must still be wrapped."""
    from functools import wraps

    stock, fast = _square_pair(family)

    @wraps(type(fast).forward)
    def logits_forward(*args, **kwargs):
        kwargs["output_hidden_states"] = False
        return stock(*args, **kwargs)

    fast.forward = logits_forward
    assert rl._model_supports_unsloth_return_hidden_states(fast) is False
    assert rl._install_grpo_hidden_states_forward_wrapper(fast) is True
    ids = torch.arange(8).unsqueeze(0)
    with torch.no_grad(), _return_hidden_states("1"):
        actual = fast(input_ids = ids).logits
    # The fallback could not get hidden states (output_hidden_states forced off), so it says this call is logits.
    assert _unsloth_grpo_hidden_states_signal(fast) is False
    with torch.no_grad(), _return_hidden_states("0"):
        expected = stock(input_ids = ids).logits
    torch.testing.assert_close(actual, expected, rtol = 0, atol = 0)


def test_marked_descendant_does_not_override_the_wrappers_call_status():
    """A fast forward on an auxiliary child must not outvote the installed fallback's per-call degradation."""
    stock, fast = _square_pair()
    assert rl._install_grpo_hidden_states_forward_wrapper(stock) is True
    stock.module = fast  # an auxiliary .module that the signal's chain walk visits
    stock._unsloth_grpo_hidden_states_degraded = True
    assert _unsloth_grpo_hidden_states_signal(stock) is False
    stock._unsloth_grpo_hidden_states_degraded = False
    assert _unsloth_grpo_hidden_states_signal(stock) is True


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_loader_gate_counts_only_on_fast_path_instances(family):
    """pre_patch's gate runs the stock forward on an instance not marked _unsloth_fast_path."""
    from unsloth.models.llama import mark_unsloth_fast_path

    _, _, config_cls = FAMILIES[family]
    config = config_cls(**{**_CONFIG, "vocab_size": _CONFIG["hidden_size"]})
    if family == "mistral":
        config.sliding_window = None
    cls = _gated_cls(family)
    torch.manual_seed(0)
    plain = cls(config).eval()
    marked = cls(config).eval()
    marked.load_state_dict(plain.state_dict())
    mark_unsloth_fast_path(marked)

    assert rl._model_supports_unsloth_return_hidden_states(plain) is False
    assert _unsloth_grpo_hidden_states_signal(plain) is None
    assert rl._install_grpo_hidden_states_forward_wrapper(plain) is True

    assert rl._model_supports_unsloth_return_hidden_states(marked) is True
    assert rl._install_grpo_hidden_states_forward_wrapper(marked) is False
    assert _unsloth_grpo_hidden_states_signal(marked) is True

    ids = torch.arange(10).unsqueeze(0)
    with torch.no_grad(), _return_hidden_states("1"):
        from_plain = plain(input_ids = ids, logits_to_keep = 5).logits
        from_marked = marked(input_ids = ids, logits_to_keep = 5).logits
    assert _unsloth_grpo_hidden_states_signal(plain) is True
    assert torch.equal(from_plain, from_marked)


def test_amp_prepared_fast_forward_keeps_the_bypass():
    """Accelerate's AMP prepare hides the marker behind convert_outputs_to_fp32 and autocast; both pass through."""
    pytest.importorskip("accelerate")
    _, fast = _square_pair()
    _amp_prepare(fast)
    assert rl._model_supports_unsloth_return_hidden_states(fast) is True
    assert rl._install_grpo_hidden_states_forward_wrapper(fast) is False
    assert _unsloth_grpo_hidden_states_signal(fast) is True
    ids = torch.arange(8).unsqueeze(0)
    with torch.no_grad(), _return_hidden_states("1"):
        hidden = fast(input_ids = ids, logits_to_keep = 5).logits
        head = fast.get_output_embeddings().weight
        assert _unsloth_pref_returns_hidden_states(fast, hidden, head) is True
    assert hidden.shape == (1, 5, _CONFIG["hidden_size"])


def test_amp_prepared_logits_forward_is_still_wrapped():
    """The AMP wrappers are followed only to what they really call: a functools.wraps copy under them, or a stock
    instance behind the gate, still gets the fallback."""
    from functools import wraps

    pytest.importorskip("accelerate")
    stock, fast = _square_pair()

    @wraps(type(fast).forward)
    def logits_forward(self, *args, **kwargs):
        return stock(*args, **kwargs)

    fast.forward = logits_forward.__get__(fast)
    _amp_prepare(fast)
    assert rl._model_supports_unsloth_return_hidden_states(fast) is False

    config = Qwen2Config(**{**_CONFIG, "vocab_size": _CONFIG["hidden_size"]})
    plain = _amp_prepare(_gated_cls()(config).eval())
    assert rl._model_supports_unsloth_return_hidden_states(plain) is False


def test_device_hook_follows_the_live_delegate():
    """Accelerate's device hook copies the marker onto its partial; only its current _old_forward counts."""
    hooks = pytest.importorskip("accelerate.hooks")
    stock, fast = _square_pair()
    hooks.add_hook_to_module(
        fast, hooks.AlignDevicesHook(execution_device = "cpu", io_same_device = True)
    )
    assert rl._model_supports_unsloth_return_hidden_states(fast) is True
    assert _unsloth_grpo_hidden_states_signal(fast) is True

    _, stale = _square_pair()
    hooks.add_hook_to_module(
        stale, hooks.AlignDevicesHook(execution_device = "cpu", io_same_device = True)
    )
    stale._old_forward = stock.forward  # the hook now delegates to a forward that returns logits
    assert rl._model_supports_unsloth_return_hidden_states(stale) is False
    assert _unsloth_grpo_hidden_states_signal(stale) is None
    assert rl._install_grpo_hidden_states_forward_wrapper(stale) is True


def test_data_parallel_wrapper_reaches_the_fast_forward():
    """DataParallel runs its .module: the installer and the scorer both look through it."""
    _, fast = _square_pair()
    parallel = torch.nn.DataParallel(fast)
    assert rl._model_supports_unsloth_return_hidden_states(parallel) is True
    assert rl._install_grpo_hidden_states_forward_wrapper(parallel) is False
    assert _unsloth_grpo_hidden_states_signal(parallel) is True


@pytest.mark.parametrize("device_hook", [False, True])
def test_square_head_scoring_dispatch(device_hook):
    """Square head end to end: the signal routes the native hidden states into the head exactly once."""
    _, fast = _square_pair()
    if device_hook:
        hooks = pytest.importorskip("accelerate.hooks")
        hooks.add_hook_to_module(
            fast, hooks.AlignDevicesHook(execution_device = "cpu", io_same_device = True)
        )
    assert rl._install_grpo_hidden_states_forward_wrapper(fast) is False
    ids = torch.arange(8).unsqueeze(0)
    with torch.no_grad(), _return_hidden_states("1"):
        hidden = fast(input_ids = ids, logits_to_keep = 5).logits
        head = fast.get_output_embeddings()
        assert _unsloth_grpo_returns_hidden_states(fast, hidden, head.weight) is True
        assert _unsloth_pref_returns_hidden_states(fast, hidden, head.weight) is True
        actual = _logps(hidden, head, ids, 4)
    with torch.no_grad(), _return_hidden_states("0"):
        logits = fast(input_ids = ids, logits_to_keep = 5).logits
    expected = logits[:, :-1].float().log_softmax(-1)
    expected = expected.gather(-1, ids[:, -4:, None]).squeeze(-1)
    torch.testing.assert_close(actual, expected, rtol = 0, atol = 1e-6)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_scoring_does_not_collect_the_layer_history(family):
    """config.output_hidden_states = True must not make the decoder keep every layer's state while scoring."""
    _, fast = _pair(family)
    fast.config.output_hidden_states = True
    requested = []
    fast.model.register_forward_pre_hook(
        lambda module, args, kwargs: requested.append(kwargs.get("output_hidden_states")),
        with_kwargs = True,
    )
    ids = torch.arange(10).unsqueeze(0)
    with torch.no_grad(), _return_hidden_states("1"):
        out = fast(input_ids = ids, logits_to_keep = 5)
    assert requested == [False]
    assert out.hidden_states is None
    with torch.no_grad(), _return_hidden_states("0"):
        assert len(fast(input_ids = ids).hidden_states) == _CONFIG["num_hidden_layers"] + 1


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("name", ["logits_to_keep", "num_logits_to_keep"])
def test_none_logits_to_keep_means_every_position(family, name):
    _, fast = _pair(family)
    ids = torch.arange(6).unsqueeze(0)
    with torch.no_grad(), _return_hidden_states("1"):
        hidden = fast(input_ids = ids, **{name: None}).logits
    assert hidden.shape == (1, 6, _CONFIG["hidden_size"])
    with torch.no_grad(), _return_hidden_states("0"):
        assert fast(input_ids = ids, **{name: None}).logits.shape == (1, 6, _CONFIG["vocab_size"])


def test_unsloth_delegate_wrapper_counts_only_for_itself():
    """A wrapper naming its delegate in _unsloth_forward_delegate passes through; a functools.wraps copy does not."""
    from functools import wraps

    stock, fast = _square_pair()
    delegate = fast.forward

    def forward(*args, **kwargs):
        return delegate(*args, **kwargs)

    forward._unsloth_forward_delegate = (forward, delegate)
    fast.forward = forward
    assert rl._model_supports_unsloth_return_hidden_states(fast) is True

    @wraps(forward)
    def copied(*args, **kwargs):
        return stock(*args, **kwargs)

    fast.forward = copied
    assert rl._model_supports_unsloth_return_hidden_states(fast) is False


def test_trl_fused_lm_head_passes_through_to_the_fast_forward():
    """TRL >= 1.15 GRPO puts add_fused_lm_head's forward on every model, copying type(model).forward's metadata."""
    trl_utils = pytest.importorskip("trl.trainer.utils")
    if not hasattr(trl_utils, "add_fused_lm_head"):
        pytest.skip("TRL < 1.15 has no add_fused_lm_head")
    from unsloth.models.llama import mark_unsloth_fast_path

    config = Qwen2Config(**{**_CONFIG, "vocab_size": _CONFIG["hidden_size"]})
    cls = _gated_cls()
    marked, plain = cls(config).eval(), cls(config).eval()
    mark_unsloth_fast_path(marked)
    for model in (marked, plain):
        trl_utils.add_fused_lm_head(model)
        assert "forward" in vars(model)
    assert rl._model_supports_unsloth_return_hidden_states(marked) is True
    assert rl._install_grpo_hidden_states_forward_wrapper(marked) is False
    assert _unsloth_grpo_hidden_states_signal(marked) is True
    assert rl._model_supports_unsloth_return_hidden_states(plain) is False


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_fallback_over_an_unrecognised_wrapper_keeps_its_hidden_states(family):
    """A fast forward behind a wrapper the predicate does not know gets the fallback, which asks for
    output_hidden_states explicitly: scoring mode must honour that request, or every call degrades."""
    _, fast = _square_pair(family)
    original = fast.forward

    def other_forward(*args, **kwargs):
        return original(*args, **kwargs)

    fast.forward = other_forward
    assert rl._install_grpo_hidden_states_forward_wrapper(fast) is True
    ids = torch.arange(10).unsqueeze(0)
    with torch.no_grad(), _return_hidden_states("1"):
        scored = fast(input_ids = ids, logits_to_keep = 5).logits
        assert _unsloth_grpo_hidden_states_signal(fast) is True
        native = original(input_ids = ids, logits_to_keep = 5).logits
    assert getattr(fast, "_unsloth_grpo_hidden_states_warning_issued", False) is False
    assert scored.shape == (1, 5, _CONFIG["hidden_size"])
    assert torch.equal(scored, native)


class _CountEqual(TorchDispatchMode):
    """Counts aten::equal: on CUDA each one reads its result back to the host."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        if func is torch.ops.aten.equal.default:
            self.calls += 1
        return func(*args, **(kwargs or {}))


def _equal_calls_over_three_scoring_calls(model):
    ids = torch.arange(10).unsqueeze(0)
    counter = _CountEqual()
    with torch.no_grad(), _return_hidden_states("1"):
        for _ in range(3):
            with counter:
                scored = model(input_ids = ids, logits_to_keep = 5).logits
            assert scored.shape == (1, 5, _CONFIG["hidden_size"])
    return counter.calls


@pytest.mark.parametrize("wrapper", ["verified_stock", "unrecognised_fast"])
def test_fallback_compares_head_input_once_per_model(wrapper):
    """The fallback's head-input comparison is a host read per call; once the model has a verdict
    (verified head, or a forward that returns the head input itself) later calls must skip it."""
    stock, fast = _square_pair()
    if wrapper == "verified_stock":
        model = stock
    else:
        model = fast
        original = fast.forward

        def other_forward(*args, **kwargs):
            return original(*args, **kwargs)

        fast.forward = other_forward
    assert rl._install_grpo_hidden_states_forward_wrapper(model) is True
    assert _equal_calls_over_three_scoring_calls(model) == 1
    assert _unsloth_grpo_hidden_states_signal(model) is True
    assert getattr(model, "_unsloth_grpo_hidden_states_unsafe", False) is False
