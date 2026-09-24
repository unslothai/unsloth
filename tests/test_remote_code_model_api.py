# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Remote modeling code written for transformers 4.x (moonshotai/Kimi-K3's
modeling_kimi_linear.py) against the transformers 5 model API: the OutputRecorder import,
a `tie_weights(self)` override and a list `_tied_weights_keys`. Runs offline on CPU with a
synthetic model defined under a `transformers_modules` module name."""

import inspect
import linecache
import sys
import types

import torch

# Import unsloth first so its import-time fixes are installed.
import unsloth  # noqa: F401
from transformers import PreTrainedModel

_SRC = """
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils.generic import OutputRecorder

class TinyRemoteConfig(PretrainedConfig):
    model_type = "tiny_remote_k3"
    def __init__(self, vocab_size=32, hidden_size=8, tie_word_embeddings=True, **kwargs):
        self.vocab_size = vocab_size; self.hidden_size = hidden_size
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

class TinyRemoteModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

class TinyRemoteForCausalLM(PreTrainedModel):
    config_class = TinyRemoteConfig
    _tied_weights_keys = ["lm_head.weight"]
    tie_calls = 0
    def __init__(self, config):
        super().__init__(config)
        self.model = TinyRemoteModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    def get_input_embeddings(self):
        return self.model.embed_tokens
    def get_output_embeddings(self):
        return self.lm_head
    def tie_weights(self):
        type(self).tie_calls += 1
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    def forward(self, input_ids):
        return self.lm_head(self.model.embed_tokens(input_ids))
"""


def _remote_module(name):
    filename = f"<{name}>"
    linecache.cache[filename] = (len(_SRC), None, _SRC.splitlines(True), filename)
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    exec(compile(_SRC, filename, "exec"), mod.__dict__)
    for cls in (mod.TinyRemoteConfig, mod.TinyRemoteModel, mod.TinyRemoteForCausalLM):
        cls.__module__ = name
    return mod


def _is_transformers5():
    return "recompute_mapping" in inspect.signature(PreTrainedModel.tie_weights).parameters


def test_output_recorder_imports_from_the_4x_location():
    from transformers.utils.generic import OutputRecorder  # noqa: F401
    mod = _remote_module("transformers_modules.tiny_remote_a.modeling_tiny")
    assert mod.OutputRecorder is not None


def test_a_4x_remote_model_builds_and_ties_on_this_transformers():
    mod = _remote_module("transformers_modules.tiny_remote_b.modeling_tiny")
    model = mod.TinyRemoteForCausalLM(mod.TinyRemoteConfig())
    assert model.lm_head.weight is model.model.embed_tokens.weight
    # transformers 5 calls tie_weights with keywords the 4.x override does not take.
    kwargs = {"recompute_mapping": False} if _is_transformers5() else {}
    model.tie_weights(**kwargs)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert mod.TinyRemoteForCausalLM.tie_calls >= 1
    if _is_transformers5():
        assert model._tied_weights_keys == {"lm_head.weight": "model.embed_tokens.weight"}
    else:
        # 4.x reads the list itself and must keep it.
        assert model._tied_weights_keys == ["lm_head.weight"]
    ids = torch.tensor([[1, 2, 3]])
    assert model(ids).shape == (1, 3, 32)


def test_an_untied_remote_model_keeps_its_own_head():
    mod = _remote_module("transformers_modules.tiny_remote_c.modeling_tiny")
    model = mod.TinyRemoteForCausalLM(mod.TinyRemoteConfig(tie_word_embeddings = False))
    assert model.lm_head.weight is not model.model.embed_tokens.weight


def test_native_classes_are_left_alone():
    """Only `transformers_modules` classes get the tie_weights wrapper or the mapping."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        vocab_size = 32,
    )
    before = LlamaForCausalLM.__dict__.get("tie_weights")
    keys = LlamaForCausalLM._tied_weights_keys
    LlamaForCausalLM(cfg)
    assert LlamaForCausalLM.__dict__.get("tie_weights") is before
    assert LlamaForCausalLM._tied_weights_keys == keys


def test_only_the_output_embedding_is_mapped_onto_the_input_embedding():
    """A 4.x list may also name keys the remote code shares itself (a projection reused across
    layers); mapping those onto the input embedding would tie them to the wrong tensor."""
    if not _is_transformers5():
        return
    mod = _remote_module("transformers_modules.tiny_remote_d.modeling_tiny")

    class SharedProjection(mod.TinyRemoteForCausalLM):
        _tied_weights_keys = ["lm_head.weight", "proj.weight"]

        def __init__(self, config):
            PreTrainedModel.__init__(self, config)
            self.model = mod.TinyRemoteModel(config)
            self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias = False)
            self.proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias = False)
            self.post_init()

    SharedProjection.__module__ = mod.__name__
    model = SharedProjection(mod.TinyRemoteConfig())
    assert model._tied_weights_keys == {"lm_head.weight": "model.embed_tokens.weight"}
    assert model.proj.weight is not model.model.embed_tokens.weight
    assert model.lm_head.weight is model.model.embed_tokens.weight
