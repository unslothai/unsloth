# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Registry refuses tp for models tp_native cannot shard, with its reason, before weights load. CPU, meta models."""

import types

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import parallel_capabilities as caps
from unsloth.parallel_env import ParallelLayout


@pytest.fixture(autouse = True)
def clean_env(monkeypatch):
    for key in ("UNSLOTH_PARALLEL_STRICT", "UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising = False)


_SMALL = dict(
    vocab_size = 128,
    hidden_size = 64,
    intermediate_size = 32,
    num_hidden_layers = 1,
    num_attention_heads = 4,
    num_key_value_heads = 4,
    num_experts_per_tok = 2,
)


def _meta(family):
    from transformers import AutoModelForCausalLM, DeepseekV3Config, OlmoeConfig, Qwen3MoeConfig

    if family == "olmoe":
        config = OlmoeConfig(**_SMALL, num_experts = 4)
    elif family == "deepseek_v3":
        config = DeepseekV3Config(
            **_SMALL,
            n_routed_experts = 4,
            n_group = 1,
            topk_group = 1,
            first_k_dense_replace = 0,
            moe_intermediate_size = 32,
            kv_lora_rank = 16,
            q_lora_rank = 32,
            qk_rope_head_dim = 8,
            qk_nope_head_dim = 8,
            v_head_dim = 16,
        )
    else:
        config = Qwen3MoeConfig(**_SMALL, num_experts = 4, moe_intermediate_size = 32, head_dim = 16)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    return model.requires_grad_(False)


def _check(model, tp = 2):
    caps.check_layout(
        ParallelLayout(fsdp2 = False, tp = tp, dp = 1, world = tp),
        None,
        model,
        types.SimpleNamespace(packing = False),
    )


@pytest.mark.parametrize(
    "family, reason",
    [
        ("olmoe", "normalizes 64 features, not one head"),
        ("deepseek_v3", "multi-head latent attention"),
    ],
)
def test_models_tp_native_cannot_shard_are_refused_by_the_registry(family, reason):
    from unsloth.models import tp_native

    model = _meta(family)
    with pytest.raises(ValueError, match = reason):
        tp_native.derive_tp_plan(model, 2)
    query = caps.query_from(ParallelLayout(fsdp2 = False, tp = 2, dp = 1, world = 2), None, model)
    assert reason in query.tp_blocker, query
    with pytest.raises(NotImplementedError) as info:
        _check(model)
    message = str(info.value)
    assert message.startswith("Unsloth: tp x model: "), message
    assert reason in message and "Instead: drop tp" in message, message
    assert all(p.is_meta for p in model.parameters())  # nothing loaded or moved


def test_a_model_tp_native_shards_is_not_refused_and_no_tp_asks_nothing():
    model = _meta("qwen3_moe")  # per-head q / k norms: tp_native shards it
    _check(model)
    assert (
        caps.query_from(ParallelLayout(fsdp2 = False, tp = 2, dp = 1, world = 2), None, model).tp_blocker
        == ""
    )
    olmoe = _meta("olmoe")
    assert (
        caps.query_from(ParallelLayout(fsdp2 = False, tp = 1, dp = 2, world = 2), None, olmoe).tp_blocker
        == ""
    )


def test_cells_table_marks_olmoe_and_deepseek_v3_tp_refused():
    for family, why in (("olmoe", "not one head"), ("deepseek_v3", "MLA")):
        rows = [
            c
            for c in caps.CELLS
            if c.name.startswith(f"MoE {family} x ") and c.name.endswith(" x tp")
        ]
        assert rows, family
        for c in rows:
            assert c.missing() and "refused by the tp x model rule" in c.note and why in c.note, (
                c.name,
                c.note,
            )


def test_zoo_moe_matrix_families_are_registry_moe_families():
    for family in (
        "olmoe",
        "mixtral",
        "qwen2_moe",
        "qwen3_next",
        "qwen3_vl_moe",
        "llama4",
        "glm4_moe",
        "ernie4_5_moe",
        "nemotron_h",
        "jamba",
        "granitemoehybrid",
        "gemma4",
        "lfm2_moe",
    ):
        assert family in caps.MOE_FAMILIES, family
        assert any(c.name.startswith(f"MoE {family} x ") for c in caps.CELLS), family


def test_a_multimodal_text_config_names_the_family_without_text():
    text = types.SimpleNamespace(model_type = "qwen3_5_moe_text", layer_types = [])
    model = types.SimpleNamespace(
        config = types.SimpleNamespace(model_type = "qwen3_5_moe", text_config = text)
    )
    assert caps._model_traits(model)[0] == "qwen3_5_moe"
    assert (
        caps._model_traits(types.SimpleNamespace(config = types.SimpleNamespace(model_type = "olmoe")))[
            0
        ]
        == "olmoe"
    )
