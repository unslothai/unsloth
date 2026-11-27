# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""Tensor parallel plan for Gemma 4 (gemma4_unified): per-layer head counts and values computed from the keys.

The text config is heterogeneous (sliding layers 2 key / value heads, the full attention layer 1 key / value head
of width 512 and attention_k_eq_v, so no v_proj). A global config.num_key_value_heads read raises transformers'
AmbiguousGlobalPerLayerAttributeError (not an AttributeError), which escaped derive_tp_plan as a crash; at tp = 2
the full layer's single key head is replicated, and its values come from k_proj, so k_proj alone is replicated.
CPU only, no process group."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_oracle import tpn  # noqa: E402

gemma4_unified = pytest.importorskip("transformers.models.gemma4_unified")


def _tiny():
    from transformers.models.gemma4_unified import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig

    config = Gemma4UnifiedTextConfig(
        vocab_size = 128,
        vocab_size_per_layer_input = 128,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 3,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        layer_types = ["sliding_attention", "sliding_attention", "full_attention"],
        per_layer_config = {"2": {"head_dim": 32, "num_key_value_heads": 1}},
        sliding_window = 8,
        attention_k_eq_v = True,
        hidden_size_per_layer_input = 0,
        num_kv_shared_layers = 0,
    )
    torch.manual_seed(0)
    return Gemma4UnifiedForCausalLM(config).eval()


def test_per_layer_config_read_does_not_escape():
    model = _tiny()
    text = model.config
    try:
        text.num_key_value_heads
    except AttributeError:
        pytest.skip("this transformers has no heterogeneous config guard")
    except Exception:
        pass  # the guard this test is about
    assert tpn._config_value(text, "num_key_value_heads") in (None, 2)


def test_gemma4_tp2_plan_replicates_the_one_global_key_head():
    model = _tiny()
    plan = tpn.derive_tp_plan(model, 2)
    attn = {r.path: r for r in plan.regions if r.kind == "attention"}
    assert len(attn) == 3
    full = next(r for p, r in attn.items() if p.endswith("layers.2.self_attn"))
    assert (full.heads, full.kv_heads, full.head_dim, full.kv_replicated) == (4, 1, 32, True)
    styles = {p.name: p.style for p in full.projs}
    assert styles == {
        "q_proj": tpn.COLWISE,
        "k_proj": tpn.KV_REPLICATED,
        "o_proj": tpn.ROWWISE,
    }, styles
    for path, region in attn.items():
        if path.endswith(("layers.0.self_attn", "layers.1.self_attn")):
            assert (region.heads, region.kv_heads, region.head_dim) == (4, 2, 16)
            assert not region.kv_replicated
            assert {p.name for p in region.projs} == {"q_proj", "k_proj", "v_proj", "o_proj"}


def test_gemma4_tp4_shards_sliding_kv_by_replication_too():
    # tp 4 > 2 sliding key heads: replicated there as well (separate k / v), and the full layer as at tp 2.
    plan = tpn.derive_tp_plan(_tiny(), 4)
    attn = [r for r in plan.regions if r.kind == "attention"]
    assert all(r.kv_replicated for r in attn)


def test_missing_value_projection_without_k_eq_v_is_still_refused():
    model = _tiny()
    attn = model.model.layers[0].self_attn
    del attn.v_proj  # neither a v_proj nor v_proj = None: values have no known source
    with pytest.raises(ValueError, match = "no separate key and value projections"):
        tpn.derive_tp_plan(model, 4)
