"""Remote modeling code that embeds in `get_input_embeddings` or returns no loss.

Step-3.7-Flash's `modeling_step3p7.py` (a vLLM port) defines
`Step3p7TextModel.get_input_embeddings(self, input_ids)` and a top-level
`forward` that accepts `labels`, computes a loss and never returns it. Every
no-argument call on the training side then fails with
`get_input_embeddings() missing 1 required positional argument: 'input_ids'`.
"""

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
    # The training-side hook that failed on Step-3.7 now attaches.
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
    # No labels: the original runs untouched.
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
