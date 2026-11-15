# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Native DeepSeek-V4.1 LoRA target mapping and get_peft_model routing on CPU."""

import os
import re
import sys

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_deepseek_v41_native import _tiny_model  # noqa: E402
from unsloth.models.deepseek_v41 import modeling as dsv41  # noqa: E402

remap = dsv41.DeepseekV41ForCausalLM.unsloth_lora_target_modules

ATTENTION_PARTS = {"self_attn", "compressor", "indexer", "engram"}
MLP_PARTS = {"mlp", "shared_experts", "experts"}
DEFAULT_LIST = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _is_attention(name):
    return bool(ATTENTION_PARTS.intersection(name.split(".")))


def _is_mlp(name):
    return (
        bool(MLP_PARTS.intersection(name.split(".")))
        or name.split(".")[-1]
        in (
            "gate_proj",
            "up_proj",
            "down_proj",
            "gate_up_proj",
        )
        and not _is_attention(name)
    )


def _wrapped(model):
    return {n.rsplit(".lora_A", 1)[0] for n, _ in model.named_parameters() if ".lora_A." in n}


def _kinds(names):
    return {re.sub(r"^.*?layers\.\d+\.", "", n) for n in names}


def test_explicit_list_without_attention_keeps_no_attention_target():
    mapped = remap(["q_proj", "gate_proj"], attention = False)
    assert not any(_is_attention(t) for t in mapped), mapped
    assert mapped == ["mlp.shared_experts.gate_proj"]


def test_explicit_list_without_mlp_keeps_no_mlp_target():
    mapped = remap(DEFAULT_LIST, mlp = False)
    assert mapped, mapped
    assert not any(_is_mlp(t) for t in mapped), mapped
    assert "self_attn.kv_proj" in mapped and "compressor.gate_proj" in mapped


@pytest.mark.parametrize("attention, mlp", [(True, True), (True, False), (False, True)])
def test_explicit_default_list_matches_the_automatic_list(attention, mlp):
    assert (
        set(remap(DEFAULT_LIST, attention = attention, mlp = mlp))
        == set(remap(None, attention = attention, mlp = mlp))
        == set(remap("all-linear", attention = attention, mlp = mlp))
    )


def test_bare_gate_proj_reaches_only_the_shared_expert():
    model = peft.get_peft_model(
        _tiny_model(), peft.LoraConfig(r = 4, target_modules = remap(["gate_proj"]))
    )
    kinds = _kinds(_wrapped(model))
    assert kinds == {"mlp.shared_experts.gate_proj"}, kinds


def test_passthrough_names_are_family_filtered():
    assert remap(["q_a_proj", "engram.wkv", "mlp.shared_experts.up_proj"], attention = False) == [
        "mlp.shared_experts.up_proj"
    ]
    assert remap(["q_a_proj", "engram.wkv", "gate_up_proj"], mlp = False) == [
        "q_a_proj",
        "engram.wkv",
    ]
    assert remap(r".*q_a_proj", attention = False) == r".*q_a_proj"


@pytest.fixture
def cpu_peft(monkeypatch):
    import unsloth  # noqa: F401
    import unsloth.models.vision as vision

    # Not under test here: the CPU cache clean and the trainer patching after PEFT wraps the model.
    monkeypatch.setattr(vision, "clean_gpu_cache", lambda: None)
    monkeypatch.setattr(
        vision.FastBaseModel, "post_patch_model", staticmethod(lambda model, **kwargs: model)
    )
    return vision.FastBaseModel


def _peft(fast, targets, **flags):
    model = _tiny_model()
    model.max_seq_length = 64
    kwargs = dict(
        r = 4,
        target_modules = targets,
        finetune_vision_layers = False,
        use_gradient_checkpointing = False,
    )
    kwargs.update(flags)
    out = fast.get_peft_model(model, **kwargs)
    return _kinds(_wrapped(out)), list(out.peft_config["default"].target_parameters or [])


@pytest.mark.parametrize("targets", [DEFAULT_LIST, ["q_proj", "gate_proj", "down_proj"]])
def test_get_peft_model_without_mlp_trains_no_expert(cpu_peft, targets):
    kinds, params = _peft(cpu_peft, targets, finetune_mlp_modules = False)
    assert kinds and not any(_is_mlp(k) for k in kinds), kinds
    assert not any("experts" in p for p in params), params


@pytest.mark.parametrize("targets", [DEFAULT_LIST, ["q_proj", "k_proj", "gate_proj"]])
def test_get_peft_model_without_attention_trains_no_attention(cpu_peft, targets):
    kinds, params = _peft(cpu_peft, targets, finetune_attention_modules = False)
    assert kinds and not any(_is_attention(k) for k in kinds), kinds
    assert any(p.endswith("experts.gate_up_proj") for p in params), params


def test_get_peft_model_default_still_reaches_every_family(cpu_peft):
    kinds, params = _peft(cpu_peft, None)
    for kind in (
        "self_attn.q_a_proj",
        "self_attn.kv_proj",
        "self_attn.o_a_proj",
        "self_attn.compressor.gate_proj",
        "engram.wkv",
        "mlp.shared_experts.gate_proj",
        "mlp.shared_experts.down_proj",
    ):
        assert kind in kinds, kind
    assert not any(".indexer." in k for k in kinds)
    assert {p.rsplit(".", 1)[-1] for p in params} == {"gate_up_proj", "down_proj"}, params


def test_get_peft_model_routed_experts_follow_the_caller_aliases(cpu_peft):
    _, params = _peft(cpu_peft, ["q_proj", "down_proj"])
    assert params and all(p.endswith("experts.down_proj") for p in params), params


def test_legacy_route_sends_remapping_models_to_fast_base(monkeypatch):
    import unsloth  # noqa: F401
    import unsloth.models.llama as llama

    calls = []

    def record(**kwargs):
        calls.append(kwargs)
        return kwargs["model"]

    monkeypatch.setenv("UNSLOTH_USE_NEW_MODEL", "0")
    monkeypatch.setattr(llama.FastBaseModel, "get_peft_model", staticmethod(record))
    model = _tiny_model()

    llama.FastLlamaModel.get_peft_model(model, r = 8, lora_dropout = 0.0, finetune_mlp_modules = False)
    assert len(calls) == 1
    assert (
        calls[0]["target_modules"] is None
    )  # the omitted causal default is the model's own default
    assert (
        calls[0]["finetune_mlp_modules"] is False and calls[0]["finetune_attention_modules"] is True
    )
    assert calls[0]["r"] == 8 and calls[0]["finetune_vision_layers"] is False

    explicit = ["q_proj", "gate_proj"]
    llama.FastLlamaModel.get_peft_model(model, target_modules = explicit)
    assert calls[1]["target_modules"] is explicit
    # An equal but caller-built list is not the omitted default.
    equal = list(llama._DEFAULT_TARGET_MODULES)
    llama.FastLlamaModel.get_peft_model(model, target_modules = equal)
    assert calls[2]["target_modules"] is equal


def test_legacy_route_end_to_end_remaps(cpu_peft, monkeypatch):
    from unsloth import FastLanguageModel

    monkeypatch.setenv("UNSLOTH_USE_NEW_MODEL", "0")
    model = _tiny_model()
    model.max_seq_length = 64
    out = FastLanguageModel.get_peft_model(model, r = 4, use_gradient_checkpointing = False)
    kinds = _kinds(_wrapped(out))
    assert {
        "self_attn.q_a_proj",
        "self_attn.kv_proj",
        "engram.wkv",
        "mlp.shared_experts.up_proj",
    } <= kinds, kinds
