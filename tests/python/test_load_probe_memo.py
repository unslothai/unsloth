# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import json

import pytest

import unsloth
import peft
import transformers

from unsloth.models import loader_utils


def _save_tiny(tmp_path, name, config):
    path = tmp_path / name
    config.save_pretrained(str(path))
    return str(path)


class _ReachedWeights(Exception):
    pass


def _count_probes(monkeypatch):
    calls = {"autoconfig": [], "peftconfig": []}
    autoconfig = transformers.AutoConfig.from_pretrained.__func__
    peftconfig = peft.config.PeftConfigMixin.from_pretrained.__func__

    def _autoconfig(cls, name, *args, **kwargs):
        calls["autoconfig"].append(str(name))
        return autoconfig(cls, name, *args, **kwargs)

    def _peftconfig(cls, name, *args, **kwargs):
        calls["peftconfig"].append(str(name))
        return peftconfig(cls, name, *args, **kwargs)

    def _stop(*args, **kwargs):
        raise _ReachedWeights()

    from unsloth.models.loader import FastBaseModel

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", classmethod(_autoconfig))
    monkeypatch.setattr(peft.config.PeftConfigMixin, "from_pretrained", classmethod(_peftconfig))
    monkeypatch.setattr(FastBaseModel, "from_pretrained", staticmethod(_stop))
    return calls


@pytest.mark.skipif(
    getattr(unsloth, "_IS_MLX", False), reason = "the MLX loader needs real safetensors"
)
def test_delegated_load_probes_config_and_adapter_once(tmp_path, monkeypatch):
    from unsloth import FastLanguageModel

    config = transformers.GraniteConfig(
        vocab_size = 128,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
    )
    path = _save_tiny(tmp_path, "granite", config)
    calls = _count_probes(monkeypatch)
    resolved = []
    with pytest.raises(_ReachedWeights):
        FastLanguageModel.from_pretrained(
            path, max_seq_length = 64, load_in_4bit = False, on_model_resolved = resolved.append
        )
    assert calls["autoconfig"].count(path) == 1, calls
    assert calls["peftconfig"].count(path) == 1, calls
    assert resolved == [path, path]


def test_mapper_fetched_once_per_load(monkeypatch):
    fetches = []

    def _fetch():
        fetches.append(1)
        return ({"org/old": "org/old"}, {}, {}, {}, {})

    monkeypatch.setattr(loader_utils, "_get_new_mapper", _fetch)
    monkeypatch.setattr(loader_utils, "_env_says_offline", lambda: False)

    def _two_lookups():
        return [
            loader_utils.get_model_name("org/unmapped-model", load_in_4bit = False) for _ in range(2)
        ]

    assert _two_lookups() == ["org/unmapped-model"] * 2
    assert len(fetches) == 2

    fetches.clear()
    assert loader_utils._load_probe_memo_scope(_two_lookups)() == ["org/unmapped-model"] * 2
    assert len(fetches) == 1

    fetches.clear()
    monkeypatch.setattr(
        loader_utils, "_get_new_mapper", lambda: fetches.append(1) or ({}, {}, {}, {}, {})
    )
    loader_utils._load_probe_memo_scope(_two_lookups)()
    assert len(fetches) == 2


def test_probe_memo_copies_reraises_and_skips_network_errors():
    calls = []

    class Config:
        def __init__(self):
            self.values = {"a": 1}

    def load(name, **kwargs):
        calls.append((name, kwargs))
        if name == "missing":
            raise ValueError("Can't find 'adapter_config.json' at 'missing'")
        if name == "flaky":
            raise ConnectionError("network down")
        return Config()

    probe = loader_utils._memo_config_probe

    probe(load, "repo", token = None)
    probe(load, "repo", token = None)
    assert len(calls) == 2

    def body():
        calls.clear()
        first = probe(load, "repo", token = None, revision = None, cache_dir = None)
        first.values["a"] = 2
        second = probe(load, "repo", token = None, revision = None)
        assert second is not first and second.values == {"a": 1}
        assert len(calls) == 1
        probe(load, "repo", token = "hf_x", revision = None)
        probe(load, "repo", token = None, revision = "v2")
        probe(load, "repo", token = None, revision = None, trust_remote_code = True)
        probe(load, "repo", token = None, revision = None, trust_remote_code = True)
        assert len(calls) == 5
        raised = []
        for _ in range(3):
            with pytest.raises(ValueError, match = "adapter_config.json") as excinfo:
                probe(load, "missing", token = None)
            raised.append(excinfo.value)
        assert len(calls) == 6
        assert len({id(e) for e in raised}) == 3
        assert all(type(e) is ValueError and str(e) == str(raised[0]) for e in raised)
        for _ in range(2):
            with pytest.raises(ConnectionError):
                probe(load, "flaky", token = None)
        assert len(calls) == 8

    loader_utils._load_probe_memo_scope(body)()


def test_reraised_probe_error_notes_stay_per_raise():
    def load(name, **kwargs):
        error = ValueError("Can't find 'adapter_config.json'")
        error.__notes__ = ["from the loader"]
        raise error

    seen = []

    def body():
        for i in range(3):
            try:
                loader_utils._memo_config_probe(load, "repo")
            except ValueError as error:
                seen.append(list(error.__notes__))
                error.__notes__.append(f"caller {i}")

    loader_utils._load_probe_memo_scope(body)()
    assert seen == [["from the loader"]] * 3, seen


def _tiny_configs():
    yield (
        "llama",
        transformers.LlamaConfig(
            vocab_size = 128,
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            num_key_value_heads = 1,
            rope_scaling = {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 64,
            },
        ),
    )
    yield (
        "granite",
        transformers.GraniteConfig(
            vocab_size = 128,
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            num_key_value_heads = 1,
        ),
    )
    yield (
        "gemma3",
        transformers.Gemma3Config(
            text_config = {
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            },
            vision_config = {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "image_size": 28,
                "patch_size": 14,
            },
        ),
    )
    yield (
        "qwen2_vl",
        transformers.Qwen2VLConfig(
            text_config = {
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            },
            vision_config = {"depth": 1, "embed_dim": 32, "hidden_size": 32, "num_heads": 2},
        ),
    )
    yield (
        "bnb_prequant",
        transformers.LlamaConfig(
            vocab_size = 128,
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            num_key_value_heads = 1,
            quantization_config = {
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
            },
        ),
    )


@pytest.mark.parametrize(
    "name,config", list(_tiny_configs()), ids = lambda v: v if isinstance(v, str) else ""
)
def test_memoized_config_equals_a_fresh_load(tmp_path, name, config):
    path = _save_tiny(tmp_path, name, config)

    def body():
        first = loader_utils._memo_config_probe(transformers.AutoConfig.from_pretrained, path)
        hit = loader_utils._memo_config_probe(transformers.AutoConfig.from_pretrained, path)
        return first, hit

    first, hit = loader_utils._load_probe_memo_scope(body)()
    fresh = transformers.AutoConfig.from_pretrained(path)
    for got in (first, hit):
        assert type(got) is type(fresh)
        assert json.loads(got.to_json_string()) == json.loads(fresh.to_json_string())
        assert sorted(vars(got)) == sorted(vars(fresh))
