# SPDX-License-Identifier: AGPL-3.0-only
"""A composition with no forward of its own (Qwen3-Omni) trains through its thinker.

`Qwen3OmniMoeForConditionalGeneration` composes a thinker, a talker and a
speech decoder and defines no `forward`, so a trainer's `model(input_ids=...)`
reached `nn.Module.forward` and failed with
`_forward_unimplemented() got an unexpected keyword argument 'input_ids'`.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import PreTrainedModel, PretrainedConfig


class TinyConfig(PretrainedConfig):
    model_type = "tiny_composed"

    def __init__(
        self,
        vocab_size = 16,
        hidden_size = 8,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        super().__init__(**kwargs)


class Thinker(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias = False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
        return self.lm_head(self.embed_tokens(input_ids))


class Talker(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, config):
        super().__init__(config)
        self.proj = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states = None,
        **kwargs,
    ):
        return self.proj(hidden_states)


class Composed(PreTrainedModel):
    """No forward of its own, like Qwen3OmniMoeForConditionalGeneration."""

    config_class = TinyConfig

    def __init__(self, config):
        super().__init__(config)
        self.thinker = Thinker(config)
        self.talker = Talker(config)
        self.code2wav = torch.nn.Linear(config.hidden_size, 1)


def test_the_defect_before_the_fix():
    """The arm that fails on main."""
    model = Composed(TinyConfig())
    with pytest.raises(TypeError, match = "unexpected keyword argument 'input_ids'"):
        model(input_ids = torch.tensor([[1, 2]]))


def test_composition_without_forward_hands_off_to_the_thinker():
    from unsloth.models.vision import _text_trainable_core

    model = Composed(TinyConfig())
    core = _text_trainable_core(model)
    assert isinstance(core, Thinker)
    assert core._unsloth_composed_parent == "Composed"
    assert not hasattr(model, "talker") and not hasattr(model, "code2wav")
    logits = core(input_ids = torch.tensor([[1, 2, 3]]))
    assert logits.shape == (1, 3, 16)


def test_a_model_with_a_text_forward_is_returned_unchanged():
    from unsloth.models.vision import _text_trainable_core
    model = Thinker(TinyConfig())
    assert _text_trainable_core(model) is model


def test_ambiguous_compositions_are_left_alone():
    from unsloth.models.vision import _text_trainable_core

    class TwoCores(PreTrainedModel):
        config_class = TinyConfig

        def __init__(self, config):
            super().__init__(config)
            self.encoder = Thinker(config)
            self.decoder = Thinker(config)

    model = TwoCores(TinyConfig())
    assert _text_trainable_core(model) is model
    assert hasattr(model, "encoder") and hasattr(model, "decoder")


def test_the_off_switch_keeps_the_wrapper(monkeypatch):
    from unsloth.models.vision import _text_trainable_core

    monkeypatch.setenv("UNSLOTH_KEEP_COMPOSED_WRAPPER", "1")
    model = Composed(TinyConfig())
    assert _text_trainable_core(model) is model


def test_a_multimodal_load_keeps_the_composition_for_generation(capsys):
    """text_intent = False is an inference or multimodal load: the talker and the speech
    decoder must survive, so the composition is returned whole with the text_only hint."""
    from unsloth.models.vision import _text_trainable_core

    model = Composed(TinyConfig())
    assert _text_trainable_core(model, text_intent = False) is model
    assert hasattr(model, "talker")
    assert "text_only = True" in capsys.readouterr().out
