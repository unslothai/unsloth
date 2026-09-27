# SPDX-License-Identifier: AGPL-3.0-only
"""Remote code that embeds in `get_input_embeddings` or returns no loss (Step-3.7-Flash)."""

import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class TinyConfig(PretrainedConfig):
    model_type = "tiny_remote"

    def __init__(
        self,
        vocab_size = 32,
        hidden_size = 8,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        super().__init__(**kwargs)


class Inner(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)

    def get_input_embeddings(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids = None,
        inputs_embeds = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)
        return inputs_embeds


class Outer(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Inner(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias = False)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()  # the inner model has no head: None

    def forward(
        self,
        input_ids = None,
        labels = None,
        **kwargs,
    ):
        logits = self.lm_head(self.model(input_ids))
        if labels is not None:
            self.config.text_config.vocab_size  # the port's own loss code trips here
        return CausalLMOutputWithPast(logits = logits)


# Remote code lives under transformers_modules; the loss shim keys on that.
Outer.__module__ = "transformers_modules.tiny_remote.modeling_tiny"
# Newer transformers reads sys.modules[cls.__module__] while building a model
# (the experts implementation probe), so the name must resolve to a real module.
sys.modules.setdefault(Outer.__module__, sys.modules[__name__])


@pytest.fixture
def model():
    torch.manual_seed(0)
    return Outer(TinyConfig())


def test_the_defect_before_the_shim(model):
    """The arm that fails on main."""
    with pytest.raises(TypeError, match = "missing 1 required positional argument"):
        model.get_input_embeddings()


def test_accessor_serves_both_contracts_after_the_shim(model):
    from unsloth.models.remote_code_shims import apply_remote_code_shims

    apply_remote_code_shims(
        model
    )  # class-level, so a second call in the same process repairs nothing new
    assert Inner._unsloth_original_get_input_embeddings is not None
    assert model.get_input_embeddings() is model.model.embed_tokens
    assert model.model.get_input_embeddings() is model.model.embed_tokens
    ids = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(model.model.get_input_embeddings(ids), model.model.embed_tokens(ids))
    model.enable_input_require_grads()


def test_output_accessor_finds_the_head_the_port_forgot(model):
    from unsloth.models.remote_code_shims import apply_remote_code_shims

    assert (
        model.get_output_embeddings() is None
        or Outer._unsloth_original_get_output_embeddings is not None
    )
    apply_remote_code_shims(model)
    assert model.get_output_embeddings() is model.lm_head


def test_forward_gains_a_loss_when_the_original_has_none(model):
    from unsloth.models.remote_code_shims import apply_remote_code_shims

    apply_remote_code_shims(model)
    assert Outer._unsloth_original_forward is not None
    ids = torch.randint(0, 32, (2, 6))
    out = model(input_ids = ids, labels = ids)
    assert out.loss is not None and torch.isfinite(out.loss)
    from transformers.loss.loss_utils import ForCausalLMLoss

    torch.testing.assert_close(out.loss, ForCausalLMLoss(out.logits, ids, 32))
    out.loss.backward()
    assert model.lm_head.weight.grad is not None
    assert model(input_ids = ids).loss is None


def test_a_forward_that_returns_its_own_loss_is_left_alone():
    class Good(Outer):
        def forward(
            self,
            input_ids = None,
            labels = None,
            **kwargs,
        ):
            logits = self.lm_head(self.model(input_ids))
            return CausalLMOutputWithPast(loss = torch.tensor(42.0), logits = logits)

    Good.__module__ = "transformers_modules.tiny_remote.modeling_tiny"
    from unsloth.models.remote_code_shims import apply_remote_code_shims

    model = Good(TinyConfig())
    apply_remote_code_shims(model)
    ids = torch.randint(0, 32, (1, 4))
    assert float(model(input_ids = ids, labels = ids).loss) == 42.0


def test_a_tuple_output_with_its_own_loss_is_left_alone():
    """return_dict = False gives (loss, logits); the model's own loss must survive the probe."""

    class TupleGood(Outer):
        def forward(
            self,
            input_ids = None,
            labels = None,
            **kwargs,
        ):
            logits = self.lm_head(self.model(input_ids))
            return (torch.tensor(42.0), logits) if labels is not None else (logits,)

    TupleGood.__module__ = "transformers_modules.tiny_remote.modeling_tiny"
    from unsloth.models.remote_code_shims import apply_remote_code_shims

    model = TupleGood(TinyConfig())
    apply_remote_code_shims(model)
    ids = torch.randint(0, 32, (1, 4))
    for _ in range(2):
        out = model(input_ids = ids, labels = ids)
        assert float(out[0]) == 42.0 and out[1].shape == (1, 4, 32)


@pytest.mark.parametrize("healthy_first", [True, False])
def test_loss_support_is_decided_per_instance(healthy_first):
    """One remote class, two configs: only the broken one gets the synthesized loss."""

    class Mixed(Outer):
        def forward(
            self,
            input_ids = None,
            labels = None,
            **kwargs,
        ):
            logits = self.lm_head(self.model(input_ids))
            if labels is not None and self.config.own_loss:
                return CausalLMOutputWithPast(loss = torch.tensor(42.0), logits = logits)
            return CausalLMOutputWithPast(logits = logits)

    Mixed.__module__ = "transformers_modules.tiny_remote.modeling_tiny"
    from unsloth.models.remote_code_shims import apply_remote_code_shims

    healthy, broken = Mixed(TinyConfig(own_loss = True)), Mixed(TinyConfig(own_loss = False))
    apply_remote_code_shims(healthy)
    apply_remote_code_shims(broken)
    ids = torch.randint(0, 32, (1, 4))
    order = [healthy, broken] if healthy_first else [broken, healthy]
    for model in order + order:
        loss = float(model(input_ids = ids, labels = ids).loss)
        assert (loss == 42.0) is (model is healthy), (model.config.own_loss, loss)


def test_a_subclass_override_of_a_repaired_accessor_is_repaired_too(model):
    from unsloth.models.remote_code_shims import apply_remote_code_shims

    apply_remote_code_shims(model)  # repairs Inner.get_input_embeddings

    class SubInner(Inner):
        def get_input_embeddings(self, input_ids):
            return self.embed_tokens(input_ids) * 2

    class SubOuter(Outer):
        def __init__(self, config):
            super().__init__(config)
            self.model = SubInner(config)

    SubOuter.__module__ = SubInner.__module__ = "transformers_modules.tiny_remote.modeling_tiny"
    sub = SubOuter(TinyConfig())
    apply_remote_code_shims(sub)
    assert sub.model.get_input_embeddings() is sub.model.embed_tokens
    ids = torch.randint(0, 32, (1, 4))
    assert torch.equal(sub.model.get_input_embeddings(ids), sub.model.embed_tokens(ids) * 2)


def test_transformers_own_classes_are_not_touched():
    from unsloth.models.remote_code_shims import (
        apply_remote_code_shims,
        accessor_requires_arguments,
    )
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size = 16,
            hidden_size = 8,
            intermediate_size = 16,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            num_key_value_heads = 2,
        )
    )
    assert apply_remote_code_shims(model) == []
    assert not hasattr(LlamaForCausalLM, "_unsloth_original_forward")
    assert not accessor_requires_arguments(LlamaForCausalLM.get_input_embeddings)


def test_the_repaired_forward_is_reached_through_an_accelerate_hook():
    """device_map loading hooks forward before the shims run (Step-3.7 in 16-bit)."""
    accelerate = pytest.importorskip("accelerate")
    from accelerate.hooks import add_hook_to_module, ModelHook
    from unsloth.models.remote_code_shims import apply_remote_code_shims

    torch.manual_seed(0)
    model = Outer(TinyConfig())
    add_hook_to_module(model, ModelHook())
    apply_remote_code_shims(model)
    ids = torch.randint(0, 32, (2, 6))
    out = model(input_ids = ids, labels = ids)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_the_synthesized_loss_is_the_first_ordered_entry(model):
    """Positional readers take output[0] and to_tuple()[0] as the loss when labels were given."""
    from unsloth.models.remote_code_shims import apply_remote_code_shims

    apply_remote_code_shims(model)
    ids = torch.randint(0, 32, (2, 6))
    out = model(input_ids = ids, labels = ids)
    assert list(out.keys())[0] == "loss"
    assert out[0] is out.loss and out.to_tuple()[0] is out.loss


def test_positional_labels_reach_the_synthesized_loss(model):
    """Positional labels get the same loss as keyword labels."""
    from unsloth.models.remote_code_shims import apply_remote_code_shims

    apply_remote_code_shims(model)
    ids = torch.randint(0, 32, (2, 6))
    by_keyword = model(input_ids = ids, labels = ids)
    by_position = model(ids, ids)
    assert by_position.loss is not None
    torch.testing.assert_close(by_position.loss, by_keyword.loss)


def test_a_forward_without_a_loss_runs_once_on_the_probing_call():
    """A second forward would hold both autograd graphs at once."""
    from unsloth.models.remote_code_shims import apply_remote_code_shims

    calls = []

    class NoLoss(Outer):
        def forward(
            self,
            input_ids = None,
            labels = None,
            **kwargs,
        ):
            calls.append(labels is not None)
            return CausalLMOutputWithPast(logits = self.lm_head(self.model(input_ids)))

    NoLoss.__module__ = "transformers_modules.tiny_remote.modeling_tiny"
    torch.manual_seed(0)
    model = NoLoss(TinyConfig())
    apply_remote_code_shims(model)
    ids = torch.randint(0, 32, (2, 6))
    out = model(input_ids = ids, labels = ids)
    assert calls == [True]
    from transformers.loss.loss_utils import ForCausalLMLoss

    torch.testing.assert_close(out.loss, ForCausalLMLoss(out.logits, ids, 32))
    model(input_ids = ids, labels = ids)
    assert calls == [True, False]


def test_non_token_logits_never_get_a_causal_loss():
    """A classification head's (batch, n_labels) logits must not be shifted across examples."""
    from unsloth.models.remote_code_shims import apply_remote_code_shims

    class Classifier(Outer):
        def forward(
            self,
            input_ids = None,
            labels = None,
            **kwargs,
        ):
            return CausalLMOutputWithPast(logits = self.lm_head(self.model(input_ids)).mean(1))

    Classifier.__module__ = "transformers_modules.tiny_remote.modeling_tiny"
    model = Classifier(TinyConfig())
    apply_remote_code_shims(model)
    ids = torch.randint(0, 32, (4, 6))
    with pytest.raises(RuntimeError, match = "token-level"):
        model(input_ids = ids, labels = torch.tensor([0, 1, 2, 3]))


def test_an_unrelated_first_call_error_does_not_switch_the_objective():
    """A failing first call must not pin the fallback: the model's own loss wins once calls work."""
    from unsloth.models.remote_code_shims import apply_remote_code_shims

    class Picky(Outer):
        def forward(
            self,
            input_ids = None,
            labels = None,
            bad = False,
            **kwargs,
        ):
            if bad:
                raise TypeError("bad batch")
            logits = self.lm_head(self.model(input_ids))
            return CausalLMOutputWithPast(loss = torch.tensor(7.0), logits = logits)

    Picky.__module__ = "transformers_modules.tiny_remote.modeling_tiny"
    model = Picky(TinyConfig())
    apply_remote_code_shims(model)
    ids = torch.randint(0, 32, (1, 4))
    with pytest.raises(TypeError, match = "bad batch"):
        model(input_ids = ids, labels = ids, bad = True)
    assert float(model(input_ids = ids, labels = ids).loss) == 7.0
