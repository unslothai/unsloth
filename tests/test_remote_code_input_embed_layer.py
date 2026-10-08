# SPDX-License-Identifier: AGPL-3.0-only
"""Remote code keeping its token embedding at `wte` with no accessor of its own (EXAONE 3.5, unsloth#1406)."""

import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import PreTrainedModel, PretrainedConfig


class WteConfig(PretrainedConfig):
    model_type = "tiny_wte_remote"

    def __init__(
        self,
        vocab_size = 32,
        hidden_size = 8,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        super().__init__(**kwargs)


class WteModel(PreTrainedModel):
    config_class = WteConfig
    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)
        self.wte = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias = False)

    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
        return self.proj(self.wte(input_ids))


class WteForCausalLM(PreTrainedModel):
    config_class = WteConfig
    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)
        self.transformer = WteModel(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias = False)

    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
        return self.lm_head(self.transformer(input_ids))


WteModel.__module__ = WteForCausalLM.__module__ = "transformers_modules.tiny_wte.modeling_tiny"
sys.modules.setdefault(WteModel.__module__, sys.modules[__name__])


def _fresh_classes():
    # The shim sets a class attribute: give each test its own classes.
    inner = type("WteModel", (WteModel,), {"__module__": WteModel.__module__})
    outer = type("WteForCausalLM", (WteForCausalLM,), {"__module__": WteForCausalLM.__module__})

    def init(self, config):
        PreTrainedModel.__init__(self, config)
        self.transformer = inner(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias = False)

    outer.__init__ = init
    return inner, outer


def _needs_default_accessor_to_raise(model):
    try:
        model.get_input_embeddings()
    except NotImplementedError:
        return
    pytest.skip("this transformers resolves `wte` by itself")


def test_wte_accessor_is_named_and_peft_attaches():
    from unsloth.models.remote_code_shims import apply_remote_code_shims

    _, outer = _fresh_classes()
    model = outer(WteConfig())
    _needs_default_accessor_to_raise(model)

    repaired = apply_remote_code_shims(model)

    assert "WteModel.get_input_embeddings" in repaired
    assert model.get_input_embeddings() is model.transformer.wte
    assert model.transformer.get_input_embeddings() is model.transformer.wte
    new = torch.nn.Embedding(32, 8)
    model.set_input_embeddings(new)
    assert model.transformer.wte is new

    peft = pytest.importorskip("peft")
    lora = peft.get_peft_model(model, peft.LoraConfig(r = 2, target_modules = ["proj"]))
    assert any("lora_A" in name for name, _ in lora.named_parameters())


def test_second_pass_is_a_no_op():
    from unsloth.models.remote_code_shims import apply_remote_code_shims

    _, outer = _fresh_classes()
    model = outer(WteConfig())
    _needs_default_accessor_to_raise(model)
    apply_remote_code_shims(model)
    assert apply_remote_code_shims(outer(WteConfig())) == []


def test_native_and_own_accessor_classes_are_left_alone():
    from unsloth.models.remote_code_shims import apply_remote_code_shims

    native = type(
        "NativeWte", (WteModel,), {"__module__": "transformers.models.fake.modeling_fake"}
    )
    model = native(WteConfig())
    _needs_default_accessor_to_raise(model)
    apply_remote_code_shims(model)
    assert "_input_embed_layer" not in native.__dict__

    own = type(
        "OwnAccessor",
        (WteModel,),
        {"__module__": WteModel.__module__, "get_input_embeddings": lambda self: self.wte},
    )
    assert apply_remote_code_shims(own(WteConfig())) == []
    assert "_input_embed_layer" not in own.__dict__
