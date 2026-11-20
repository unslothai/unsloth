# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL's own vLLM engine (fast_inference = False): frozen weights without an adapter are sent on
the first sync only. The engine must still end every sync holding exactly what a full sync gives,
and everything is sent again once the engine drops its weights."""

import types
from contextlib import nullcontext

import pytest
import torch

from tests.test_grpo_move_model_to_vllm_sync import (
    _patched_method,
    _trainer,
    _WeightRecorder,
)


class _Engine(_WeightRecorder):
    """Fake vLLM model: keeps the last tensor loaded under each name, counts bytes."""

    def __init__(self):
        super().__init__()
        self.state = {}
        self.bytes = 0

    def _take(self, name, tensor):
        self.state[name] = tensor.detach().clone()
        self.bytes += tensor.numel() * tensor.element_size()

    def load_weights(self, weights):
        weights = list(weights)
        super().load_weights(weights)
        for name, tensor in weights:
            self._take(name, tensor)

    def update_named_param(self, name, data):
        super().update_named_param(name, data)
        self._take(name, data)


class _LLM:
    """Fake vllm.LLM: sleep(level = 2) drops the weights, as vLLM does."""

    def __init__(self, engine):
        self.engine = engine
        self.llm_engine = types.SimpleNamespace(
            model_executor = types.SimpleNamespace(
                driver_worker = types.SimpleNamespace(
                    model_runner = types.SimpleNamespace(model = engine)
                )
            )
        )

    def sleep(self, level = 1):
        if level >= 2:
            self.engine.state.clear()

    def wake_up(self, tags = None):
        pass

    def collective_rpc(self, method, *args, **kwargs):
        if method == "reload_weights":
            self.engine.state.clear()

    def reset_prefix_cache(self):
        self.engine.resets += 1


def _tiny_llama_lora(target_modules = ("q_proj", "v_proj")):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        tie_word_embeddings = False,
    )
    model = transformers.LlamaForCausalLM(config).to(torch.bfloat16)
    lora = peft.LoraConfig(
        r = 4, lora_alpha = 8, target_modules = list(target_modules), init_lora_weights = False
    )
    return peft.get_peft_model(model, lora)


def _merged_trainer(model, vllm_mode = "colocate"):
    trainer, _ = _trainer(vllm_mode)
    engine = _Engine()
    trainer.model = model
    trainer.vllm_client = engine
    if vllm_mode == "colocate":
        trainer.llm = _LLM(engine)
    return trainer, engine


def _step(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.add_(0.25)


def _full_sync_state(model, vllm_mode):
    """What the engine holds after the same sync with the skip turned off."""
    import os

    trainer, engine = _merged_trainer(model, vllm_mode)
    os.environ["UNSLOTH_VLLM_SYNC_FROZEN_WEIGHTS"] = "1"
    try:
        _patched_method()(trainer)
    finally:
        del os.environ["UNSLOTH_VLLM_SYNC_FROZEN_WEIGHTS"]
    return engine.state


def _assert_same_state(got, want):
    assert sorted(got) == sorted(want)
    for name in want:
        assert torch.equal(got[name], want[name]), name


@pytest.mark.parametrize("vllm_mode", ["colocate", "server"])
def test_frozen_weights_sent_once_and_engine_matches_full_sync(vllm_mode):
    model = _tiny_llama_lora()
    trainer, engine = _merged_trainer(model, vllm_mode)
    method = _patched_method()
    sent_bytes = []
    for step in range(3):
        _step(model)
        before_bytes, before_names = engine.bytes, len(engine.loaded)
        method(trainer)
        sent_bytes.append(engine.bytes - before_bytes)
        names = engine.loaded[before_names:]
        if step > 0:
            # Only the LoRA-bearing projections: q_proj and v_proj in 2 layers.
            assert sorted(names) == sorted(
                f"model.layers.{i}.self_attn.{p}.weight"
                for i in range(2)
                for p in ("q_proj", "v_proj")
            )
        _assert_same_state(engine.state, _full_sync_state(model, vllm_mode))
    assert sent_bytes[1] == sent_bytes[2] < sent_bytes[0] / 5
    state = trainer._unsloth_vllm_frozen_sync
    assert state["syncs"] == 3 and state["skipped"] > 0
    assert state["skipped_bytes"] == 2 * (sent_bytes[0] - sent_bytes[1])


@pytest.mark.parametrize("level, resend", [(1, False), (2, True)])
def test_sleep_level_2_resends_everything(level, resend):
    model = _tiny_llama_lora()
    trainer, engine = _merged_trainer(model)
    method = _patched_method()
    method(trainer)
    full = len(engine.loaded)
    trainer.llm.sleep(level = level)
    before = len(engine.loaded)
    method(trainer)
    assert (len(engine.loaded) - before == full) is resend
    if resend:
        _assert_same_state(engine.state, _full_sync_state(model, "colocate"))
    # Positional level, as TRL < 0.28 online DPO calls it.
    trainer.llm.sleep(2)
    before = len(engine.loaded)
    method(trainer)
    assert len(engine.loaded) - before == full


def test_reload_weights_resends_everything():
    model = _tiny_llama_lora()
    trainer, engine = _merged_trainer(model)
    method = _patched_method()
    method(trainer)
    full = len(engine.loaded)
    trainer.llm.collective_rpc("reload_weights")
    before = len(engine.loaded)
    method(trainer)
    assert len(engine.loaded) - before == full
    _assert_same_state(engine.state, _full_sync_state(model, "colocate"))


def test_new_receiver_resends_everything():
    model = _tiny_llama_lora()
    trainer, engine = _merged_trainer(model, "server")
    method = _patched_method()
    method(trainer)
    full = len(engine.loaded)
    trainer.vllm_client = other = _Engine()
    method(trainer)
    assert len(other.loaded) == full


def test_unfrozen_or_edited_frozen_weight_is_sent():
    model = _tiny_llama_lora()
    trainer, engine = _merged_trainer(model)
    method = _patched_method()
    method(trainer)
    norm = model.base_model.model.model.norm.weight
    head = model.base_model.model.lm_head.weight
    # In place edit of a frozen weight bumps its version.
    with torch.no_grad():
        norm.mul_(2)
    head.requires_grad_(True)
    before = len(engine.loaded)
    method(trainer)
    names = engine.loaded[before:]
    assert "model.norm.weight" in names and "lm_head.weight" in names
    _assert_same_state(engine.state, _full_sync_state(model, "colocate"))
    # Trainable weights are sent every sync, even unchanged.
    before = len(engine.loaded)
    method(trainer)
    assert "lm_head.weight" in engine.loaded[before:]
    assert "model.norm.weight" not in engine.loaded[before:]


def test_opt_out_sends_everything(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_SYNC_FROZEN_WEIGHTS", "1")
    model = _tiny_llama_lora()
    trainer, engine = _merged_trainer(model)
    method = _patched_method()
    method(trainer)
    full = len(engine.loaded)
    method(trainer)
    assert len(engine.loaded) == 2 * full


def test_modules_to_save_toggle_resends_the_original_head():
    # The served copy (trainable) and original_module (frozen) share one vLLM name: whichever was
    # sent last decides, so switching back to the frozen original must send it again.
    peft = pytest.importorskip("peft")
    torch.manual_seed(0)
    base = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
    config = peft.LoraConfig(
        r = 2, target_modules = ["0"], modules_to_save = ["1"], init_lora_weights = False
    )
    model = peft.get_peft_model(base, config)
    with torch.no_grad():
        model.base_model.model[1].modules_to_save["default"].weight.add_(1.0)
    trainer, engine = _merged_trainer(model)
    method = _patched_method()
    head = model.base_model.model[1]
    for enabled in (False, True, False, False):
        if enabled:
            model.base_model.enable_adapter_layers()
        else:
            model.base_model.disable_adapter_layers()
        method(trainer)
        want = head.modules_to_save["default"] if enabled else head.original_module
        assert torch.equal(engine.state["1.weight"], want.weight)


# TRL >= 0.28: VLLMGeneration.sync_weights -> _unsloth_vllm_sync_colocate.


def _iter_named_params_like_trl(generation):
    # trl main generation/vllm_generation.py VLLMGeneration._iter_named_params, single process.
    model = generation.model
    if hasattr(model, "merge_adapter"):
        model.merge_adapter()
        for name, param in model.named_parameters():
            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            if model.prefix in name:
                continue
            if "original_module" in name:
                continue
            yield name.replace("modules_to_save.default.", ""), param.data
        model.unmerge_adapter()
    else:
        for name, param in model.named_parameters():
            yield name, param.data


def _generation(model, sharded = False):
    engine = _Engine()
    state = types.SimpleNamespace(deepspeed_plugin = object() if sharded else None, fsdp_plugin = None)
    generation = types.SimpleNamespace(
        model = model,
        llm = _LLM(engine),
        accelerator = types.SimpleNamespace(state = state),
    )
    # What a full sync gives, from the same pass (TRL's in-place merge drifts a bf16 base, so a
    # second pass would not reproduce it).
    generation.full = {}

    def _iter_named_params():
        for name, tensor in _iter_named_params_like_trl(generation):
            generation.full[name] = tensor.detach().clone()
            yield name, tensor

    generation._iter_named_params = _iter_named_params
    return generation, engine


@pytest.mark.parametrize("bucket_mb", ["1024", "0"])
def test_trl_028_colocate_dense_skips_frozen_tower(bucket_mb, monkeypatch):
    from unsloth.models.rl_replacements import _unsloth_vllm_sync_colocate

    monkeypatch.setenv("UNSLOTH_VLLM_WEIGHT_SYNC_BUCKET_MB", bucket_mb)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 16))
    model[0].requires_grad_(False)  # e.g. a frozen vision tower in full finetuning
    generation, engine = _generation(model)
    for step in range(3):
        _step(model)
        before = len(engine.loaded)
        _unsloth_vllm_sync_colocate(generation, engine)
        names = sorted(engine.loaded[before:])
        assert names == (["0.bias", "0.weight"] if step == 0 else []) + ["1.bias", "1.weight"]
        _assert_same_state(engine.state, generation.full)
    generation.llm.sleep(level = 2)
    before = len(engine.loaded)
    _unsloth_vllm_sync_colocate(generation, engine)
    assert len(engine.loaded) - before == 4


def test_trl_028_colocate_peft_in_place_merge():
    from unsloth.models.rl_replacements import _unsloth_vllm_sync_colocate

    model = _tiny_llama_lora()
    generation, engine = _generation(model)
    for step in range(3):
        _step(model)
        before = len(engine.loaded)
        _unsloth_vllm_sync_colocate(generation, engine)
        if step > 0:
            assert sorted(engine.loaded[before:]) == sorted(
                f"model.layers.{i}.self_attn.{p}.weight"
                for i in range(2)
                for p in ("q_proj", "v_proj")
            )
        _assert_same_state(engine.state, generation.full)


def test_trl_028_colocate_sharded_sends_everything():
    from unsloth.models.rl_replacements import _unsloth_vllm_sync_colocate

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    model[0].requires_grad_(False)
    generation, engine = _generation(model, sharded = True)
    for _ in range(2):
        _unsloth_vllm_sync_colocate(generation, engine)
    assert len(engine.loaded) == 8
