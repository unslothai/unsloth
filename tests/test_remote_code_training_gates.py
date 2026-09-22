"""Two gates that let a remote-code multimodal model reach its first training step.

Padding-free batching adds `packed_seq_lengths` to every batch. A remote-code
forward with a fixed signature (microsoft/Phi-4-reasoning-vision-15B) raised
"got an unexpected keyword argument 'packed_seq_lengths'" on the first step, so
auto padding-free now stays off for a forward that can take neither that key
nor `**kwargs`.

A wrapper model that keeps transformers' default
`supports_gradient_checkpointing = False` around a CausalLM that supports it
(nvidia/Nemotron-3-Nano-Omni-30B-A3B) made Trainer's
`gradient_checkpointing_enable` raise. The wrapper now inherits the answer of
the model it wraps.

Small models, no downloads; each test states which arm it measures.
"""
import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from unsloth.trainer import _forward_accepts_packed_seq_lengths
from unsloth.models.vision import _inherit_gradient_checkpointing_support


class _Fixed(nn.Module):
    """The Phi-4 shape: named arguments only."""
    def forward(self, input_ids = None, attention_mask = None, labels = None):
        return input_ids


class _Kwargs(nn.Module):
    def forward(self, input_ids = None, **kwargs):
        return input_ids


class _Explicit(nn.Module):
    def forward(self, input_ids = None, packed_seq_lengths = None):
        return input_ids


class _PeftLike(nn.Module):
    """PEFT forwards every keyword to the wrapped model."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.peft_config = {"default": None}
    def get_base_model(self):
        return self.inner
    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


def test_fixed_signature_cannot_take_packed_seq_lengths():
    """The arm that fails on a tree without the gate: this model was padded-free."""
    assert _forward_accepts_packed_seq_lengths(_Fixed()) is False


def test_kwargs_and_explicit_parameter_can():
    assert _forward_accepts_packed_seq_lengths(_Kwargs()) is True
    assert _forward_accepts_packed_seq_lengths(_Explicit()) is True


def test_peft_wrapper_is_looked_through():
    assert _forward_accepts_packed_seq_lengths(_PeftLike(_Fixed())) is False
    assert _forward_accepts_packed_seq_lengths(_PeftLike(_Kwargs())) is True


def test_unknown_shapes_leave_the_decision_alone():
    assert _forward_accepts_packed_seq_lengths(None) is True
    assert _forward_accepts_packed_seq_lengths("unsloth/Qwen3-0.6B") is True
    assert _forward_accepts_packed_seq_lengths(object()) is True


class _Cfg(PretrainedConfig):
    model_type = "unsloth-test-gc-wrapper"


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.gradient_checkpointing = False
    def forward(self, x):
        return self.linear(x)


class _CausalLM(PreTrainedModel):
    config_class = _Cfg
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([_Layer(), _Layer()])
        self.lm_head = nn.Linear(4, 8, bias = False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids = None, **kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.layers: x = layer(x)
        return self.lm_head(x)


class _Wrapper(PreTrainedModel):
    """The Nemotron-Omni shape: transformers' default False around a model that supports it."""
    config_class = _Cfg

    def __init__(self, config):
        super().__init__(config)
        self.language_model = _CausalLM(config)
        self.vision_model = nn.Linear(4, 4)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def forward(self, input_ids = None, **kwargs):
        return self.language_model(input_ids = input_ids)


def test_wrapper_refuses_gradient_checkpointing_on_its_own():
    """The precondition, on this transformers version."""
    model = _Wrapper(_Cfg())
    assert model.supports_gradient_checkpointing is False
    with pytest.raises(ValueError, match = "does not support gradient checkpointing"):
        model.gradient_checkpointing_enable()


def test_wrapper_inherits_support_and_every_layer_is_switched_on():
    model = _Wrapper(_Cfg())
    assert _inherit_gradient_checkpointing_support(model) is True
    assert model.supports_gradient_checkpointing is True
    model.gradient_checkpointing_enable()
    assert all(layer.gradient_checkpointing for layer in model.language_model.layers)
    # the class default is untouched: a fresh wrapper starts from False again
    assert _Wrapper.supports_gradient_checkpointing is False


def test_models_that_already_answer_are_left_alone():
    model = _CausalLM(_Cfg())
    assert _inherit_gradient_checkpointing_support(model) is False
    assert model.supports_gradient_checkpointing is True


def test_wrapper_without_a_supporting_submodel_stays_false():
    class _Plain(PreTrainedModel):
        config_class = _Cfg
        def __init__(self, config):
            super().__init__(config)
            self.encoder = nn.Linear(4, 4)
    model = _Plain(_Cfg())
    assert _inherit_gradient_checkpointing_support(model) is False
    assert model.supports_gradient_checkpointing is False
