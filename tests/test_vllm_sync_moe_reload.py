# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL's own vLLM engine (fast_inference = False) with a MoE model: vLLM re-lays out unquantized
expert weights after load (FlashInfer TRTLLM / CUTLASS on Blackwell, ROCm AITER), so the sync must
go through vLLM's in-place reload, which re-runs that processing, and must send every weight.
Dense models, older vLLM and the kill switch keep the per-tensor load_weights path."""

import types

import pytest
import torch

from tests.test_grpo_move_model_to_vllm_sync import _patched_method, _trainer
from tests.test_vllm_sync_skips_frozen_weights import (
    _Engine,
    _LLM,
    _step,
    _tiny_llama_lora,
)


class _MoEEngine(torch.nn.Module):
    """Fake vLLM model: one FusedMoE-like module (w13_weight) plus a recorder."""

    def __init__(
        self,
        moe = True,
        backend = "FLASHINFER_TRTLLM",
    ):
        super().__init__()
        self.inner = _Engine()
        if moe:
            self.experts = torch.nn.Module()
            self.experts.w13_weight = torch.nn.Parameter(torch.zeros(2, 4, 4), requires_grad = False)
            self.experts.quant_method = types.SimpleNamespace(
                unquantized_backend = types.SimpleNamespace(name = backend)
            )

    def load_weights(self, weights):
        weights = list(weights)
        self.inner.load_weights(weights)
        return {name for name, _ in weights}


class _Runner:
    def __init__(
        self,
        model,
        new_api = True,
    ):
        self.model = model
        self.reloads = []
        if not new_api:
            self.reload_weights = self._reload_from_disk

    def reload_weights(
        self,
        weights_iterator = None,
        weights_path = None,
        is_checkpoint_format = True,
    ):
        self.reloads.append(is_checkpoint_format)
        self.model.load_weights(weights_iterator)

    def _reload_from_disk(self):
        raise AssertionError("vLLM 0.11 reload_weights() reloads from disk")


def _llm(model, new_api = True):
    runner = _Runner(model, new_api)
    llm = types.SimpleNamespace(
        llm_engine = types.SimpleNamespace(
            model_executor = types.SimpleNamespace(
                driver_worker = types.SimpleNamespace(model_runner = runner)
            )
        ),
        resets = 0,
    )

    def reset_prefix_cache():
        llm.resets += 1

    llm.reset_prefix_cache = reset_prefix_cache
    llm.sleep = lambda level = 1: None
    llm.wake_up = lambda tags = None: None
    llm.collective_rpc = lambda method, *a, **k: None
    return llm, runner


def _merged_trainer(
    model,
    engine_model,
    new_api = True,
):
    trainer, _ = _trainer("colocate")
    trainer.model = model
    trainer.llm, runner = _llm(engine_model, new_api)
    return trainer, runner


def _all_names(model):
    return sorted(
        name.removeprefix("base_model.model.").replace(".base_layer", "")
        for name, param in model.named_parameters()
        if "lora_" not in name
    )


@pytest.mark.parametrize(
    "moe, new_api, env, reloads",
    [
        (True, True, None, True),
        (True, True, "0", False),
        (True, False, None, False),
        (False, True, None, False),
        (False, True, "1", True),
    ],
)
def test_reload_runner_selection(moe, new_api, env, reloads, monkeypatch):
    from unsloth.models.rl_replacements import _unsloth_vllm_reload_runner

    if env is None:
        monkeypatch.delenv("UNSLOTH_VLLM_SYNC_RELOAD", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_VLLM_SYNC_RELOAD", env)
    engine_model = _MoEEngine(moe)
    llm, runner = _llm(engine_model, new_api)
    assert (_unsloth_vllm_reload_runner(llm, engine_model) is runner) == reloads


@pytest.mark.parametrize(
    "backend, reloads",
    [
        ("FLASHINFER_TRTLLM", True),
        ("FLASHINFER_CUTLASS", True),
        ("AITER", True),
        ("TRITON", False),
        ("BATCHED_TRITON", False),
    ],
)
def test_reload_only_for_backends_that_relay_out(backend, reloads, monkeypatch):
    from unsloth.models.rl_replacements import _unsloth_vllm_reload_runner

    monkeypatch.delenv("UNSLOTH_VLLM_SYNC_RELOAD", raising = False)
    engine_model = _MoEEngine(backend = backend)
    llm, runner = _llm(engine_model)
    assert (_unsloth_vllm_reload_runner(llm, engine_model) is runner) == reloads


def test_reload_for_quantized_or_old_vllm_moe():
    from unsloth.models.rl_replacements import _unsloth_vllm_moe_relaid_out

    class UnquantizedFusedMoEMethod:
        rocm_aiter_moe_enabled = False
        flashinfer_cutlass_moe_enabled = False

    old = types.SimpleNamespace(quant_method = UnquantizedFusedMoEMethod())
    assert not _unsloth_vllm_moe_relaid_out(old)
    old.quant_method.flashinfer_cutlass_moe_enabled = True
    assert _unsloth_vllm_moe_relaid_out(old)
    assert _unsloth_vllm_moe_relaid_out(types.SimpleNamespace(quant_method = object()))


def test_reload_runner_ignores_fake_engines_without_modules():
    from unsloth.models.rl_replacements import _unsloth_vllm_reload_runner
    engine = _Engine()
    assert _unsloth_vllm_reload_runner(_LLM(engine), engine) is None


def test_merged_sync_moe_reloads_every_weight_every_step(monkeypatch):
    monkeypatch.delenv("UNSLOTH_VLLM_SYNC_RELOAD", raising = False)
    model = _tiny_llama_lora()
    engine_model = _MoEEngine()
    trainer, runner = _merged_trainer(model, engine_model)
    method = _patched_method()
    for step in range(3):
        _step(model)
        before = len(engine_model.inner.loaded)
        method(trainer)
        # Frozen weights are not skipped: the reload re-lays out whole modules.
        assert sorted(engine_model.inner.loaded[before:]) == _all_names(model)
    assert runner.reloads == [True, True, True]
    assert trainer.llm.resets == 3


def test_merged_sync_dense_keeps_load_weights_and_frozen_skip(monkeypatch):
    monkeypatch.delenv("UNSLOTH_VLLM_SYNC_RELOAD", raising = False)
    model = _tiny_llama_lora()
    engine_model = _MoEEngine(moe = False)
    trainer, runner = _merged_trainer(model, engine_model)
    method = _patched_method()
    for step in range(2):
        _step(model)
        before = len(engine_model.inner.loaded)
        method(trainer)
        sent = engine_model.inner.loaded[before:]
    assert runner.reloads == []
    assert sorted(sent) == sorted(
        f"model.layers.{i}.self_attn.{p}.weight" for i in range(2) for p in ("q_proj", "v_proj")
    )


@pytest.mark.parametrize("sharded", [False, True])
def test_trl_028_colocate_moe_reloads(sharded, monkeypatch):
    from unsloth.models.rl_replacements import _unsloth_vllm_sync_colocate

    monkeypatch.delenv("UNSLOTH_VLLM_SYNC_RELOAD", raising = False)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    model[0].requires_grad_(False)
    engine_model = _MoEEngine()
    llm, runner = _llm(engine_model)
    live = dict(model.named_parameters())
    state = types.SimpleNamespace(deepspeed_plugin = object() if sharded else None, fsdp_plugin = None)
    generation = types.SimpleNamespace(
        model = model,
        llm = llm,
        accelerator = types.SimpleNamespace(state = state),
        _iter_named_params = lambda: iter(live.items()),
    )
    for step in range(2):
        _step(model)
        before = len(engine_model.inner.loaded)
        _unsloth_vllm_sync_colocate(generation, engine_model)
        assert sorted(engine_model.inner.loaded[before:]) == [
            "0.bias",
            "0.weight",
            "1.bias",
            "1.weight",
        ]
        for name, tensor in engine_model.inner.tensors[before:]:
            assert torch.equal(tensor, live[name].detach())
    assert runner.reloads == [True, True]


@pytest.mark.parametrize("moe", [True, False])
def test_online_dpo_state_dict_sync_reloads_moe(moe, monkeypatch):
    from unsloth.models.rl_replacements import _unsloth_vllm_load_state_dict

    monkeypatch.delenv("UNSLOTH_VLLM_SYNC_RELOAD", raising = False)
    model = torch.nn.Linear(4, 4)
    engine_model = _MoEEngine(moe)
    llm, runner = _llm(engine_model)
    trainer = types.SimpleNamespace(
        accelerator = types.SimpleNamespace(
            state = types.SimpleNamespace(deepspeed_plugin = None), unwrap_model = lambda m: m
        ),
        is_fsdp_enabled = False,
        llm = llm,
    )
    _unsloth_vllm_load_state_dict(trainer, model)
    assert sorted(engine_model.inner.loaded) == ["bias", "weight"]
    assert runner.reloads == ([True] if moe else [])
    assert llm.resets == 1


def test_reload_through_a_forwarding_runner(monkeypatch):
    """vLLM 0.29: the v2 runner's reload_weights(*args, **kwargs) forwards to the v1 runner's."""
    import sys

    from unsloth.models.rl_replacements import _unsloth_vllm_reload_runner

    monkeypatch.delenv("UNSLOTH_VLLM_SYNC_RELOAD", raising = False)

    class V1Runner:
        def reload_weights(
            self,
            weights_iterator = None,
            weights_path = None,
            is_checkpoint_format = True,
        ):
            pass

    names = ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu_model_runner")
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner = V1Runner
    engine_model = _MoEEngine()
    llm, runner = _llm(engine_model)
    runner.reload_weights = lambda *args, **kwargs: _Runner.reload_weights(runner, *args, **kwargs)
    assert _unsloth_vllm_reload_runner(llm, engine_model) is runner
    del V1Runner.reload_weights
    V1Runner.reload_weights = lambda self: None
    assert _unsloth_vllm_reload_runner(llm, engine_model) is None
