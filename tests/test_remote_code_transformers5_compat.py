# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Tiny CPU repros of 4.x remote code (Trinity-Large), a remote class shadowing a native name, and config-only remote code (MiniMax-M3) on transformers 5."""

import json
import textwrap
from types import SimpleNamespace

import pytest


@pytest.fixture(scope = "module")
def unsloth_loaded():
    try:
        import unsloth  # noqa: F401  installs the import fixes
    except Exception as e:  # pragma: no cover - environment without a usable accelerator
        pytest.skip(f"unsloth does not import here: {e}")
    import transformers
    return transformers


def _write(path, name, source):
    (path / name).write_text(textwrap.dedent(source))


_LEGACY_CONFIG = """
from transformers import PretrainedConfig


class LegacyToyConfig(PretrainedConfig):
    model_type = "legacy_toy"

    def __init__(self, vocab_size = 64, hidden_size = 32, num_attention_heads = 2,
                 num_hidden_layers = 1, rope_theta = 10000.0, rope_scaling = None, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        super().__init__(**kwargs)
"""

_LEGACY_MODELING = """
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

try:
    from .configuration_legacy_toy import LegacyToyConfig
except ImportError:
    from configuration_legacy_toy import LegacyToyConfig


class LegacyToyRotaryEmbedding(nn.Module):
    def __init__(self, config, device = None):
        super().__init__()
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent = False)
        self.original_inv_freq = self.inv_freq


class LegacyToyPreTrainedModel(PreTrainedModel):
    config_class = LegacyToyConfig
    base_model_prefix = "model"
    _supports_sdpa = True


class LegacyToyForCausalLM(LegacyToyPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = LegacyToyRotaryEmbedding(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)
        self.post_init()

    def forward(self, input_ids = None, attention_mask = None, labels = None, **kwargs):
        inputs_embeds = self.embed_tokens(input_ids)
        cache_position = torch.arange(input_ids.shape[1], device = input_ids.device)
        mask = create_causal_mask(
            config = self.config,
            input_embeds = inputs_embeds,
            attention_mask = attention_mask,
            cache_position = cache_position,
            past_key_values = None,
            position_ids = cache_position.unsqueeze(0),
        )
        logits = self.lm_head(inputs_embeds * self.rotary_emb.inv_freq.mean())
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten())
        return CausalLMOutput(loss = loss, logits = logits)
"""


@pytest.fixture()
def legacy_repo(tmp_path):
    _write(tmp_path, "configuration_legacy_toy.py", _LEGACY_CONFIG)
    _write(tmp_path, "modeling_legacy_toy.py", _LEGACY_MODELING)
    config = {
        "model_type": "legacy_toy",
        "architectures": ["LegacyToyForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_legacy_toy.LegacyToyConfig",
            "AutoModelForCausalLM": "modeling_legacy_toy.LegacyToyForCausalLM",
        },
        "vocab_size": 64,
        "hidden_size": 32,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_transformers4_remote_model_builds_and_runs(unsloth_loaded, legacy_repo):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(legacy_repo, trust_remote_code = True)
    assert config.pad_token_id is None
    model = AutoModelForCausalLM.from_config(config, trust_remote_code = True)
    input_ids = torch.randint(0, 64, (1, 8))
    out = model(input_ids = input_ids, attention_mask = torch.ones_like(input_ids), labels = input_ids)
    assert torch.isfinite(out.loss)
    dim = 32 // 2
    expected = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
    assert torch.allclose(model.rotary_emb.inv_freq.float(), expected)


def test_a_remote_file_edited_between_loads_is_patched_again(unsloth_loaded, legacy_repo):
    """transformers 5.4 re-executes a changed remote file in the same module object; new classes need the defaults too."""
    from transformers import AutoConfig

    assert AutoConfig.from_pretrained(legacy_repo, trust_remote_code = True).pad_token_id is None
    source = legacy_repo / "configuration_legacy_toy.py"
    source.write_text(source.read_text() + "\n# edited between loads\n")
    config = AutoConfig.from_pretrained(legacy_repo, trust_remote_code = True)
    assert config.pad_token_id is None


def test_checkpoint_token_ids_still_win(unsloth_loaded, legacy_repo):
    from transformers import AutoConfig

    config_json = json.loads((legacy_repo / "config.json").read_text())
    config_json["pad_token_id"] = 3
    (legacy_repo / "config.json").write_text(json.dumps(config_json))
    config = AutoConfig.from_pretrained(legacy_repo, trust_remote_code = True)
    assert config.pad_token_id == 3
    assert vars(config).get("bos_token_id") is None
    assert config.to_dict().get("bos_token_id") is None


def test_native_rope_registry_is_left_alone(unsloth_loaded, legacy_repo):
    """A "default" key in ROPE_INIT_FUNCTIONS would override every native model's own in `_init_weights`."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    had_default = "default" in ROPE_INIT_FUNCTIONS
    config = AutoConfig.from_pretrained(legacy_repo, trust_remote_code = True)
    AutoModelForCausalLM.from_config(config, trust_remote_code = True)
    assert ("default" in ROPE_INIT_FUNCTIONS) == had_default


def test_legacy_defaults_touch_only_remote_classes(unsloth_loaded):
    from transformers import PretrainedConfig
    for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        assert name not in PretrainedConfig.__dict__


_SHADOW_CONFIG = """
from transformers import LlamaConfig as _NativeLlamaConfig


class LlamaConfig(_NativeLlamaConfig):
    pass
"""

_SHADOW_MODELING = """
from transformers import LlamaForCausalLM as _NativeLlamaForCausalLM

try:
    from .configuration_shadow import LlamaConfig
except ImportError:
    from configuration_shadow import LlamaConfig


class LlamaForCausalLM(_NativeLlamaForCausalLM):
    config_class = LlamaConfig
    _supports_flash_attn = False
    _supports_flash_attn_2 = False
    _supports_sdpa = True
"""


@pytest.fixture()
def shadow_repo(tmp_path):
    _write(tmp_path, "configuration_shadow.py", _SHADOW_CONFIG)
    _write(tmp_path, "modeling_shadow.py", _SHADOW_MODELING)
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_shadow.LlamaConfig",
            "AutoModelForCausalLM": "modeling_shadow.LlamaForCausalLM",
        },
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_remote_class_is_what_attention_is_resolved_for(unsloth_loaded, shadow_repo):
    from transformers import AutoConfig, AutoModelForCausalLM
    from unsloth.models._utils import resolve_model_class, resolve_remote_code_model_class

    config = AutoConfig.from_pretrained(shadow_repo, trust_remote_code = True)
    assert type(config).__module__.startswith("transformers_modules")
    native = resolve_model_class(AutoModelForCausalLM, config)
    assert native is not None and native.__module__.startswith("transformers.")

    builds_remote, remote = resolve_remote_code_model_class(
        AutoModelForCausalLM, config, str(shadow_repo), trust_remote_code = True
    )
    assert builds_remote is True
    assert remote is not None and remote.__module__.startswith("transformers_modules")
    assert remote._supports_flash_attn is False


def test_remote_class_only_replaces_a_native_class_it_shadows(unsloth_loaded):
    from unsloth.models._utils import attention_class_for_load

    class Native:
        _supports_sdpa = True

    class Remote:
        _supports_flash_attn_2 = True
        _supports_sdpa = False

    assert attention_class_for_load(Native, True, Remote, True) == (Remote, False)
    # No native class shadowed: remote flags must not route a working load onto flash attention.
    assert attention_class_for_load(None, True, Remote, True) == (None, False)
    Remote._supports_sdpa = True
    assert attention_class_for_load(None, True, Remote, True) == (None, True)
    assert attention_class_for_load(None, True, None, True) == (None, True)
    assert attention_class_for_load(Native, True, None, True) == (None, False)
    assert attention_class_for_load(Native, False, None, True) == (Native, True)


def test_remote_class_not_used_without_trust(unsloth_loaded, shadow_repo):
    from transformers import AutoModelForCausalLM, LlamaConfig
    from unsloth.models._utils import resolve_remote_code_model_class

    config = LlamaConfig.from_pretrained(shadow_repo)
    assert resolve_remote_code_model_class(
        AutoModelForCausalLM, config, str(shadow_repo), trust_remote_code = False
    ) == (False, None)
    from transformers import AutoModelForSequenceClassification

    assert resolve_remote_code_model_class(
        AutoModelForSequenceClassification, config, str(shadow_repo), trust_remote_code = True
    ) == (False, None)


def test_only_an_exact_registration_overrides_remote_code(unsloth_loaded):
    """Only an exact registration of this config class overrides auto_map, not a registered parent."""
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
    from unsloth.models._utils import resolve_remote_code_model_class

    class ParentConfig(PretrainedConfig):
        model_type = "unsloth_exact_registration_parent"

    class ParentModel(nn.Module):
        config_class = ParentConfig

    class RemoteChildConfig(ParentConfig):
        pass

    AutoConfig.register(ParentConfig.model_type, ParentConfig, exist_ok = True)
    AutoModelForCausalLM.register(ParentConfig, ParentModel, exist_ok = True)
    try:
        child = RemoteChildConfig(auto_map = {"AutoModelForCausalLM": "modeling_missing.Missing"})
        assert resolve_remote_code_model_class(
            AutoModelForCausalLM, child, "/nonexistent/unsloth/repo", trust_remote_code = True
        ) == (True, None)
    finally:
        AutoModelForCausalLM._model_mapping._extra_content.pop(ParentConfig, None)
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        CONFIG_MAPPING._extra_content.pop(ParentConfig.model_type, None)


def test_the_remote_class_lookup_uses_the_loads_code_revision(unsloth_loaded, monkeypatch):
    import transformers.dynamic_module_utils as dynamic_module_utils
    from transformers import AutoModelForCausalLM, LlamaConfig
    from unsloth.models import vision
    from unsloth.models._utils import resolve_remote_code_model_class

    seen = {}

    def fetch(class_ref, repo, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(dynamic_module_utils, "get_class_from_dynamic_module", fetch)
    config = LlamaConfig(auto_map = {"AutoModelForCausalLM": "modeling_x.X"})
    options = {
        "code_revision": "abc",
        "cache_dir": "/cache",
        "proxies": {"https": "http://proxy"},
        "force_download": True,
    }
    resolve_remote_code_model_class(
        AutoModelForCausalLM, config, "some/repo", trust_remote_code = True, **options
    )
    assert {k: seen.get(k) for k in options} == options
    import inspect

    assert set(options) <= set(vision._REMOTE_CLASS_HUB_OPTIONS)
    assert "for k in _REMOTE_CLASS_HUB_OPTIONS" in inspect.getsource(vision)


def test_unfetchable_remote_class_is_unknown_not_native(unsloth_loaded):
    from transformers import AutoModelForCausalLM, LlamaConfig
    from unsloth.models._utils import resolve_remote_code_model_class

    config = LlamaConfig(auto_map = {"AutoModelForCausalLM": "modeling_missing.Missing"})
    assert resolve_remote_code_model_class(
        AutoModelForCausalLM, config, "/nonexistent/unsloth/repo", trust_remote_code = True
    ) == (True, None)


_CONFIG_ONLY = """
from transformers import PretrainedConfig


class LlavaConfig(PretrainedConfig):
    model_type = "llava"

    def __init__(self, vision_config = None, text_config = None, **kwargs):
        # What a converter-facing shim does: keep the sub-configs generic.
        self.vision_config = PretrainedConfig(**(vision_config or {}))
        self.text_config = PretrainedConfig(**(text_config or {}))
        super().__init__(**kwargs)
"""


def _config_only_repo(path, auto_map):
    _write(path, "configuration_shim.py", _CONFIG_ONLY)
    config = {
        "model_type": "llava",
        "architectures": ["LlavaForConditionalGeneration"],
        "auto_map": auto_map,
        "vision_config": {
            "model_type": "clip_vision_model",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
        },
        "text_config": {
            "model_type": "llama",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    return path


def test_config_only_remote_code_loads_the_native_config(unsloth_loaded, tmp_path):
    from transformers import AutoConfig, LlavaConfig

    repo = _config_only_repo(tmp_path, {"AutoConfig": "configuration_shim.LlavaConfig"})
    config = AutoConfig.from_pretrained(repo, trust_remote_code = True)
    assert type(config) is LlavaConfig
    assert type(config.vision_config).__name__ == "CLIPVisionConfig"
    assert type(AutoConfig.from_pretrained(repo)) is LlavaConfig


def test_config_only_swap_keeps_return_unused_kwargs(unsloth_loaded, tmp_path):
    from transformers import AutoConfig, LlavaConfig

    repo = _config_only_repo(tmp_path, {"AutoConfig": "configuration_shim.LlavaConfig"})
    config, unused = AutoConfig.from_pretrained(
        repo, trust_remote_code = True, return_unused_kwargs = True, foo = 1
    )
    assert type(config) is LlavaConfig and unused == {"foo": 1}


def test_repo_with_its_own_model_keeps_its_config(unsloth_loaded, tmp_path):
    from transformers import AutoConfig

    repo = _config_only_repo(
        tmp_path,
        {
            "AutoConfig": "configuration_shim.LlavaConfig",
            "AutoModelForImageTextToText": "modeling_shim.Model",
        },
    )
    config = AutoConfig.from_pretrained(repo, trust_remote_code = True)
    assert type(config).__module__.startswith("transformers_modules")


def test_native_config_with_config_only_auto_map_keeps_the_compiler(unsloth_loaded):
    from transformers import LlamaConfig
    from unsloth.models.loader import _config_uses_remote_code

    native = LlamaConfig(auto_map = {"AutoConfig": "configuration_x.XConfig"})
    assert _config_uses_remote_code(native) is False
    native.auto_map = {"AutoConfig": "c.X", "AutoModelForCausalLM": "m.X"}
    assert _config_uses_remote_code(native) is True
    assert _config_uses_remote_code(SimpleNamespace(auto_map = {"AutoConfig": "c.X"})) is True
