# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor parallel on a model whose attention owns a parameter directly (gpt-oss attention sinks) is refused by
the capability registry, with tp_native's reason, before any weight is loaded or sharded. CPU only."""

import types

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import parallel_capabilities as caps
from unsloth.parallel_env import ParallelLayout

GPT_OSS = "trl-internal-testing/tiny-GptOssForCausalLM"
QWEN3 = "trl-internal-testing/tiny-Qwen3ForCausalLM"
QWEN3_MOE = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"


@pytest.fixture(autouse = True)
def clean_env(monkeypatch):
    for key in ("UNSLOTH_PARALLEL_STRICT", "UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising = False)


def _meta(repo):
    from transformers import AutoConfig, AutoModelForCausalLM
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(repo))
    return model.requires_grad_(
        False
    )  # a frozen base (LoRA), so the pp x mode rule does not answer first


def _check(model, **sizes):
    caps.check_layout(
        ParallelLayout(fsdp2 = False, **{"tp": 1, "dp": 1, "world": 2, **sizes}),
        None,
        model,
        types.SimpleNamespace(packing = False),
    )


@pytest.mark.parametrize("tp, pp", [(2, 1), (4, 1), (2, 2)])
def test_gpt_oss_under_tp_is_refused_by_the_registry_before_weights_load(tp, pp):
    model = _meta(GPT_OSS)
    assert all(p.is_meta for p in model.parameters())
    with pytest.raises(NotImplementedError) as info:
        _check(model, tp = tp, pp = pp, world = tp * pp)
    message = str(info.value)
    assert message.startswith("Unsloth: tp x model: "), message
    assert "own a parameter directly" in message and "Instead: drop tp" in message, message
    query = caps.query_from(None, None, model)
    assert query.family == "gpt_oss" and query.direct_params


def test_the_registry_and_tp_native_refuse_the_same_model():
    from transformers import AutoModelForCausalLM

    from unsloth.models import tp_native

    model = AutoModelForCausalLM.from_pretrained(GPT_OSS, dtype = torch.float32)
    with pytest.raises(NotImplementedError, match = r"^Unsloth: tp x model: "):
        _check(model, tp = 2)
    with pytest.raises(ValueError, match = "owns the parameter 'sinks'"):
        tp_native.derive_tp_plan(model, 2)


def test_gpt_oss_without_tp_is_not_refused_by_the_rule():
    model = _meta(GPT_OSS)
    _check(model, dp = 2)


@pytest.mark.parametrize("repo", [QWEN3, QWEN3_MOE])
def test_models_whose_blocks_own_no_parameter_keep_running_under_tp(repo):
    from unsloth.models import tp_native

    model = _meta(repo)
    _check(model, tp = 2)
    assert not getattr(caps.query_from(None, None, model), "direct_params", False)
    tp_native.derive_tp_plan(model, 2)


def test_cells_table_says_gpt_oss_tp_is_refused():
    rows = [c for c in caps.CELLS if c.name.startswith("MoE gpt_oss") and c.name.endswith(" x tp")]
    assert rows and all("refused by the tp x model rule" in c.note for c in rows), [
        c.note for c in rows
    ]
    others = [
        c for c in caps.CELLS if c.name.startswith("MoE qwen3_moe") and c.name.endswith(" x tp")
    ]
    assert others and all("tensor parallel MoE runs" in c.note for c in others)
    assert "| tp x model |" in caps.markdown_table()
