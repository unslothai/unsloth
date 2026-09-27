"""`exclude_no_placement_params`: a model's `_no_placement_params` stay off the device map.

Qwen4Exp (Qwen3.8-Flash-Next) declares its ~102 GB hashed n-gram table unplaceable. On one
GPU transformers could not escape it (it only does so when the next device is a GPU), so
FastModel sent the whole model to CPU and the first CUDA forward failed. The helper must
place every other tensor where the map put it and leave the table out, and must return the
map untouched for any model without the attribute."""
import pytest
import torch
from torch import nn

import unsloth  # noqa: F401
from unsloth.models.loader_utils import exclude_no_placement_params


class ScaledEmbedding(nn.Embedding):
    """Like transformers' FP8Embedding: the lookup multiplies by a sibling weight_scale."""

    def __init__(self, n, d):
        super().__init__(n, d)
        self.weight_scale = nn.Parameter(torch.ones(1))


class Table(nn.Module):
    def __init__(self):
        super().__init__()
        self.ngram_embedding = ScaledEmbedding(1000, 8)
        self.register_buffer("offsets", torch.zeros(4, dtype=torch.long))


class Ple(nn.Module):
    def __init__(self):
        super().__init__()
        self.ple_embedding = Table()
        self.key_proj = nn.Linear(8, 8)


class Layer(nn.Module):
    def __init__(self, with_ple):
        super().__init__()
        self.mlp = nn.Linear(8, 8)
        self.ple = Ple() if with_ple else None
        self.norm_weight = nn.Parameter(torch.ones(8))


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 8)
        self.layers = nn.ModuleList([Layer(False), Layer(True), Layer(False)])


class Model(nn.Module):
    _no_placement_params = ["ple.ple_embedding.ngram_embedding.weight"]

    def __init__(self, config=None):
        super().__init__()
        self.model = Inner()
        self.lm_head = nn.Linear(8, 10)

    @classmethod
    def _from_config(cls, config):
        return cls(config)


class Plain(Model):
    _no_placement_params = None


def covered(device_map, name):
    return any(k == "" or name == k or name.startswith(k + ".") for k in device_map)


def check(device_map, device):
    names = [n for n, _ in list(Model().named_parameters()) + list(Model().named_buffers())]
    table = "model.layers.1.ple.ple_embedding.ngram_embedding."
    # The whole owning module stays off the map, its weight_scale included.
    assert not covered(device_map, table + "weight") and not covered(device_map, table + "weight_scale")
    for n in names:
        if not n.startswith(table):
            assert covered(device_map, n), n
    assert set(device_map.values()) == {device}
    # No entry is a prefix of another (accelerate needs a disjoint map).
    keys = list(device_map)
    assert not any(a != b and b.startswith(a + ".") for a in keys for b in keys)


def test_dict_map_splits_only_the_owner():
    out = exclude_no_placement_params({"": 0}, Model, None)
    check(out, 0)
    assert "model.layers.0" in out and "model.layers.2" in out and "lm_head" in out
    assert "model.layers.1.ple.ple_embedding.offsets" in out


def test_single_gpu_string_map(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    check(exclude_no_placement_params("sequential", Model, None), 0)
    check(exclude_no_placement_params("cuda:0", Model, None), 0)


def test_untouched_without_attribute_or_with_kill_switch(monkeypatch):
    assert exclude_no_placement_params("sequential", Plain, None) == "sequential"
    assert exclude_no_placement_params({"": 0}, None, None) == {"": 0}
    monkeypatch.setenv("UNSLOTH_PLACE_NO_PLACEMENT_PARAMS", "1")
    assert exclude_no_placement_params("sequential", Model, None) == "sequential"


def test_multi_gpu_string_left_to_transformers(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert exclude_no_placement_params("sequential", Model, None) == "sequential"


def test_real_qwen4_exp_on_meta(monkeypatch):
    mod = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    cfg = Qwen4ExpTextConfig(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=16, indexer_compress_ratio=4,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, shared_expert_intermediate_size=16,
        vocab_size=128, hc_count=2, hc_lowrank=8, ple_layer_ids=[2], ngram_vocab_size_base=100, eos_token_id=1,
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    out = exclude_no_placement_params("sequential", mod.Qwen4ExpForCausalLM, cfg)
    assert isinstance(out, dict)
    assert not covered(out, "model.layers.1.ple.ple_embedding.ngram_embedding.weight")
    assert covered(out, "model.layers.1.ple.key_proj.weight") and covered(out, "model.layers.3.mlp.experts.down_proj")
