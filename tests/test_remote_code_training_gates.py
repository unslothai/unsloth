# SPDX-License-Identifier: AGPL-3.0-only
"""Gates that let a remote-code multimodal model reach its first training step:
padding-free off for a fixed-signature forward, gradient checkpointing inherited by
a wrapper, and text training through the wrapped language model. No downloads.
"""

import os

import pytest
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from unsloth.trainer import _forward_accepts_packing_kwargs
from unsloth.models.vision import _inherit_gradient_checkpointing_support


class _Fixed(nn.Module):
    """The Phi-4 shape: named arguments only."""

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        labels = None,
    ):
        return input_ids


class _Kwargs(nn.Module):
    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
        return input_ids


class _Explicit(nn.Module):
    def forward(
        self,
        input_ids = None,
        packed_seq_lengths = None,
    ):
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
    """Without the gate this model was made padding-free."""
    assert _forward_accepts_packing_kwargs(_Fixed()) is False


def test_kwargs_and_explicit_parameter_can():
    assert _forward_accepts_packing_kwargs(_Kwargs()) is True
    assert _forward_accepts_packing_kwargs(_Explicit()) is True


def test_peft_wrapper_is_looked_through():
    assert _forward_accepts_packing_kwargs(_PeftLike(_Fixed())) is False
    assert _forward_accepts_packing_kwargs(_PeftLike(_Kwargs())) is True


def test_unknown_shapes_leave_the_decision_alone():
    assert _forward_accepts_packing_kwargs(None) is True
    assert _forward_accepts_packing_kwargs("unsloth/Qwen3-0.6B") is True
    assert _forward_accepts_packing_kwargs(object()) is True


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

    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
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

    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
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


from unsloth.models.vision import _text_trainable_core, _required_non_text_inputs


class _OmniWrapper(PreTrainedModel):
    """The Nemotron-Omni shape: pixel_values has no default."""

    config_class = _Cfg

    def __init__(self, config):
        super().__init__(config)
        self.language_model = _CausalLM(config)
        self.vision_model = nn.Linear(4, 4)
        self.mlp1 = nn.Linear(4, 4)

    def forward(
        self,
        pixel_values,
        input_ids = None,
        attention_mask = None,
        image_flags = None,
        labels = None,
    ):
        return self.language_model(input_ids = input_ids)


class _VlmWrapper(_OmniWrapper):
    """Every transformers VLM: image inputs default to None, so a text batch is fine."""

    def forward(
        self,
        input_ids = None,
        pixel_values = None,
        attention_mask = None,
        labels = None,
    ):
        return self.language_model(input_ids = input_ids)


def test_required_non_text_inputs_are_listed():
    assert _required_non_text_inputs(_OmniWrapper.forward) == ["pixel_values"]
    assert _required_non_text_inputs(_VlmWrapper.forward) == []
    assert _required_non_text_inputs(_CausalLM.forward) == []


def test_omni_wrapper_trains_its_language_model():
    """Without the unwrap the wrapper was trained as is."""
    model = _OmniWrapper(_Cfg())
    core = _text_trainable_core(model)
    assert isinstance(core, _CausalLM)
    assert core._unsloth_composed_parent == "_OmniWrapper"
    assert not hasattr(model, "vision_model") and not hasattr(model, "mlp1")
    core(input_ids = torch.tensor([[1, 2, 3]]))


def test_vlm_wrapper_and_plain_causal_lm_are_left_alone():
    vlm = _VlmWrapper(_Cfg())
    assert _text_trainable_core(vlm) is vlm
    assert hasattr(vlm, "vision_model")
    lm = _CausalLM(_Cfg())
    assert _text_trainable_core(lm) is lm


def test_ambiguous_wrapper_is_left_alone():
    class _Two(_OmniWrapper):
        def __init__(self, config):
            super().__init__(config)
            self.talker = _CausalLM(config)
            self.listener = _CausalLM(config)
            del self.language_model

    model = _Two(_Cfg())
    assert _text_trainable_core(model) is model


def test_multimodal_intent_keeps_the_wrapper(capsys):
    """Without text_only the wrapper is kept and the hint names text_only = True."""
    model = _OmniWrapper(_Cfg())
    assert _text_trainable_core(model, text_intent = False) is model
    assert hasattr(model, "vision_model")
    out = capsys.readouterr().out
    assert "pixel_values" in out and "text_only = True" in out
    # A wrapper that can take a text batch prints nothing either way.
    vlm = _VlmWrapper(_Cfg())
    assert _text_trainable_core(vlm, text_intent = False) is vlm
    assert capsys.readouterr().out == ""


def test_opt_out_env_keeps_the_wrapper(monkeypatch):
    monkeypatch.setenv("UNSLOTH_KEEP_COMPOSED_WRAPPER", "1")
    model = _OmniWrapper(_Cfg())
    assert _text_trainable_core(model) is model
    assert hasattr(model, "vision_model")


def test_core_carries_the_wrapper_loader_state():
    """The core must carry the wrapper's bitsandbytes flags and device map."""
    model = _OmniWrapper(_Cfg())
    model.is_loaded_in_4bit = True
    model.is_quantized = True
    model.quantization_method = "bitsandbytes"
    model.hf_quantizer = object()
    model.hf_device_map = {
        "language_model.embed_tokens": 0,
        "language_model.layers.0": 0,
        "language_model.layers.1": 1,
        "language_model.lm_head": 1,
        "vision_model": 0,
        "mlp1": 0,
    }
    model.config.quantization_config = {"quant_method": "bitsandbytes", "load_in_4bit": True}
    core = _text_trainable_core(model)
    assert isinstance(core, _CausalLM)
    assert core.is_loaded_in_4bit is True
    assert core.is_quantized is True
    assert core.quantization_method == "bitsandbytes"
    assert core.hf_quantizer is model.hf_quantizer
    assert core.hf_device_map == {"embed_tokens": 0, "layers.0": 0, "layers.1": 1, "lm_head": 1}
    assert core.config.quantization_config == model.config.quantization_config


def test_core_inherits_the_wrapper_dtype_and_repo_path():
    """A sub-config with no dtype left bnb's compute dtype None (backward crashed), and no path broke adapter reloads."""
    model = _OmniWrapper(_Cfg())
    model.language_model.config = _Cfg()
    model.language_model.config.dtype = None
    model.config.dtype = torch.bfloat16
    model.config._name_or_path = "org/omni-repo"
    core = _text_trainable_core(model)
    assert core.config.dtype == torch.bfloat16
    assert core.config._name_or_path == "org/omni-repo"
    assert core.name_or_path == "org/omni-repo"


def test_merged_save_of_a_text_core_is_refused(tmp_path):
    """The merge re-reads the wrapper-layout shards: it wrote an unmerged or unloadable checkpoint."""
    from unsloth.save import unsloth_generic_save

    from peft import LoraConfig, get_peft_model

    core = _text_trainable_core(_OmniWrapper(_Cfg()))
    peft_model = get_peft_model(core, LoraConfig(r = 2, target_modules = ["lm_head"]))
    with pytest.raises(NotImplementedError, match = "text_only = True"):
        unsloth_generic_save(peft_model, None, str(tmp_path), save_method = "merged_16bit")


def test_full_finetuned_text_core_saves_its_own_weights(tmp_path):
    """No adapter to merge: the core writes its resident state_dict, which the refusal used to block."""
    from unsloth.save import unsloth_generic_save

    core = _text_trainable_core(_OmniWrapper(_Cfg()))
    unsloth_generic_save(core, None, str(tmp_path), save_method = "merged_16bit")
    assert any(name.endswith(".safetensors") for name in os.listdir(tmp_path))


def test_core_without_loader_state_gets_none_invented():
    model = _OmniWrapper(_Cfg())
    core = _text_trainable_core(model)
    assert "is_loaded_in_4bit" not in vars(core)
    assert getattr(core, "hf_device_map", None) is None


@pytest.mark.skipif(not has_real_cuda(), reason = "bitsandbytes 4-bit needs a GPU")
def test_peft_dispatches_the_4bit_lora_layer_on_the_core():
    """Without the flags PEFT wraps a Linear4bit in the plain lora.Linear."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    import peft.tuners.lora.bnb as lora_bnb

    inner = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        device_map = {"": 0},
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True, bnb_4bit_compute_dtype = torch.bfloat16
        ),
    )

    class _Composed(_OmniWrapper):
        def __init__(self, config):
            PreTrainedModel.__init__(self, config)
            self.language_model = inner
            self.vision_model = nn.Linear(4, 4)

    wrapper = _Composed(_Cfg())
    for attribute in ("is_loaded_in_4bit", "is_quantized", "quantization_method", "hf_quantizer"):
        setattr(wrapper, attribute, vars(inner).pop(attribute))
    core = _text_trainable_core(wrapper)
    assert core is inner
    peft_model = get_peft_model(core, LoraConfig(r = 8, target_modules = ["q_proj"]))
    layer = peft_model.base_model.model.model.layers[0].self_attn.q_proj
    assert isinstance(layer, lora_bnb.Linear4bit), type(layer)


def test_model_built_on_gradient_checkpointing_layer_is_recognised():
    from transformers.modeling_layers import GradientCheckpointingLayer

    class _Block(GradientCheckpointingLayer):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    class _RemoteCausalLM(PreTrainedModel):
        """The NemotronHForCausalLM shape: blocks checkpoint, the class says False."""

        config_class = _Cfg

        def __init__(self, config):
            super().__init__(config)
            self.embed_tokens = nn.Embedding(8, 4)
            self.layers = nn.ModuleList([_Block(), _Block()])

        def get_input_embeddings(self):
            return self.embed_tokens

    model = _RemoteCausalLM(_Cfg())
    assert model.supports_gradient_checkpointing is False
    assert _inherit_gradient_checkpointing_support(model) is True
    model.gradient_checkpointing_enable()
    assert all(layer.gradient_checkpointing for layer in model.layers)


def test_the_loader_carries_the_callers_text_intent_past_its_own_normalisation():
    """loader.py passes the caller's own text_only as text_intent, not its normalised value."""
    import inspect
    from unsloth.models import loader, vision

    assert "text_intent" in inspect.signature(vision.FastBaseModel.from_pretrained).parameters
    source = inspect.getsource(loader.FastModel.from_pretrained)
    assert "text_intent = bool(text_only)" in source
    assert "text_only = load_text_only" in source


def test_standard_tokenizer_fields_count_as_text_inputs():
    """token_type_ids is supplied by the Trainer, so it is no reason to unwrap."""
    from unsloth.models.vision import _TEXT_BATCH_KEYS
    assert "token_type_ids" in _TEXT_BATCH_KEYS


def test_a_required_cache_control_is_a_missing_text_input():
    """A required cache_position is not supplied by the collator; input_ids is."""
    from unsloth.models.vision import _required_non_text_inputs

    def needs_cache(
        self,
        input_ids,
        attention_mask,
        cache_position,
        labels = None,
    ):
        pass

    def plain(
        self,
        input_ids,
        attention_mask,
        labels = None,
        cache_position = None,
    ):
        pass

    assert _required_non_text_inputs(needs_cache) == ["cache_position"]
    assert _required_non_text_inputs(plain) == []


def test_non_bitsandbytes_quantizers_keep_their_own_dtype_handling(monkeypatch):
    """GPTQ / HQQ / Quark refuse or special-case dtype casts; the shim must not take them over."""
    import unsloth.models.vision as vision

    called = []
    monkeypatch.setattr(vision, "_cast_unquantized_floats", lambda m, d: called.append(d))
    model = _CausalLM(_Cfg())
    model.quantization_method = "gptq"
    with vision._tolerate_dtype_cast_on_quantized_model(True):
        try:
            model.to(torch.bfloat16)
        except Exception:
            pass
    assert called == []
    model.quantization_method = "bitsandbytes"
    with vision._tolerate_dtype_cast_on_quantized_model(True):
        model.to(torch.bfloat16)
    assert called == [torch.bfloat16]


def test_core_keeps_the_wrapper_generation_config():
    """generation_config.json lands on the wrapper; the child only had config defaults."""
    model = _OmniWrapper(_Cfg())
    from transformers import GenerationConfig

    model.generation_config = GenerationConfig(eos_token_id = [7, 8])
    core = _text_trainable_core(model)
    assert core.generation_config.eos_token_id == [7, 8]


@pytest.mark.parametrize("spelling", ["LoRA", "lora ", "LORA"])
def test_adapter_save_spellings_of_a_text_core_are_not_refused(spelling, tmp_path):
    from unsloth.save import unsloth_generic_save

    core = _text_trainable_core(_OmniWrapper(_Cfg()))
    try:
        unsloth_generic_save(core, None, str(tmp_path), save_method = spelling)
    except NotImplementedError as error:
        pytest.fail(f"adapter save refused: {error}")
    except Exception:
        pass
