# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL >= 0.28 syncs TRL's own vLLM engine (fast_inference = False) through
VLLMGeneration._iter_named_params, whose PEFT branch runs merge_adapter() and unmerge_adapter() on
the training model. In bf16 that round trip rounds twice, so the frozen base weights drift a little
on every sync. Unsloth streams merged copies instead: the base keeps its exact bytes and vLLM gets
W + BA rounded once."""

import types
from contextlib import nullcontext

import pytest
import torch


def _tiny_lora(dtype = torch.bfloat16, target_modules = "all-linear"):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size = 64,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        tie_word_embeddings = False,
    )
    model = transformers.LlamaForCausalLM(config).to(dtype)
    lora = peft.LoraConfig(
        r = 8, lora_alpha = 16, target_modules = target_modules, init_lora_weights = False
    )
    return peft.get_peft_model(model, lora)


def _optimizer_step(model, step):
    generator = torch.Generator().manual_seed(step)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                noise = torch.randn(param.shape, generator = generator) * 0.01
                param.add_(noise.to(param.dtype))


def _base_weights(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}


def _reference(model):
    """W + scaling * B @ A in fp32, rounded once, under vLLM's names."""
    out = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            continue
        vllm_name = name.removeprefix("base_model.model.").replace(".base_layer", "")
        out[vllm_name] = param.detach().clone()
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or "default" not in module.lora_A:
            continue
        weight = module.base_layer.weight
        delta = module.lora_B["default"].weight.float() @ module.lora_A["default"].weight.float()
        merged = (weight.float() + delta * module.scaling["default"]).to(weight.dtype)
        vllm_name = name.removeprefix("base_model.model.") + ".weight"
        out[vllm_name] = merged
    return out


class _Engine:
    def __init__(self):
        self.state = {}
        self.calls = 0
        self.resets = 0

    def load_weights(self, weights):
        self.calls += 1
        for name, tensor in weights:
            self.state[name] = tensor.detach().clone()


class _LLM:
    def __init__(self, engine):
        self.llm_engine = types.SimpleNamespace(
            model_executor = types.SimpleNamespace(
                driver_worker = types.SimpleNamespace(
                    model_runner = types.SimpleNamespace(model = engine)
                )
            )
        )

    def sleep(self, level = 1):
        pass

    def wake_up(self, tags = None):
        pass

    def collective_rpc(self, method, *args, **kwargs):
        pass

    def reset_prefix_cache(self):
        pass


class _TrlGeneration:
    """TRL 1.15-1.18 VLLMGeneration._iter_named_params, unsharded PEFT branch, as shipped."""

    def __init__(
        self,
        model,
        sharded = False,
    ):
        self.model = model
        self.engine = _Engine()
        self.llm = _LLM(self.engine)
        plugin = types.SimpleNamespace(zero_stage = 3) if sharded else None
        self.accelerator = types.SimpleNamespace(
            state = types.SimpleNamespace(deepspeed_plugin = plugin, fsdp_plugin = None),
            is_main_process = True,
        )
        self._dist = types.SimpleNamespace(is_fsdp = False, gather_params = lambda p: nullcontext())

    def _fix_param_name_to_vllm(
        self,
        name,
        extra_prefixes = None,
    ):
        for prefix in ["_checkpoint_wrapped_module."] + (extra_prefixes or []):
            name = name.replace(prefix, "")
        return name

    def _iter_named_params(self):
        model = self.model
        if hasattr(model, "merge_adapter"):
            with self._dist.gather_params(list(model.parameters())):
                model.merge_adapter()
                for name, param in model.named_parameters():
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if model.prefix in name:
                        continue
                    if "original_module" in name:
                        continue
                    name = self._fix_param_name_to_vllm(
                        name, extra_prefixes = ["modules_to_save.default."]
                    )
                    yield name, param.data
                model.unmerge_adapter()
        else:
            for name, param in model.named_parameters():
                yield self._fix_param_name_to_vllm(name), param.data

    def sync_weights(self):
        for name, param in self._iter_named_params():
            self.engine.load_weights([(name, param)])


def _patched_class():
    from unsloth.models.rl_replacements import _unsloth_vllm_patch_iter_named_params

    cls = type("_PatchedTrlGeneration", (_TrlGeneration,), {})
    cls._iter_named_params = _TrlGeneration._iter_named_params
    assert _unsloth_vllm_patch_iter_named_params(cls)
    assert not _unsloth_vllm_patch_iter_named_params(cls)  # idempotent
    return cls


def _assert_no_drift_and_exact(
    generation,
    model,
    syncs = 8,
):
    base = _base_weights(model)
    for step in range(syncs):
        _optimizer_step(model, step)
        generation.sync_weights()
        for name, param in model.named_parameters():
            if name in base:
                assert torch.equal(param, base[name]), f"{name} drifted after sync {step + 1}"
        want = _reference(model)
        assert sorted(generation.engine.state) == sorted(want)
        for name, tensor in want.items():
            assert torch.equal(generation.engine.state[name], tensor), name


def test_trl_in_place_merge_drifts_bf16_base():
    """The defect: TRL's own branch moves the frozen base."""
    model = _tiny_lora()
    generation = _TrlGeneration(model)
    base = _base_weights(model)
    for step in range(8):
        _optimizer_step(model, step)
        generation.sync_weights()
    drift = max(
        (p.float() - base[n].float()).abs().max().item()
        for n, p in model.named_parameters()
        if n in base
    )
    assert drift > 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_out_of_place_merge_keeps_base_exact(dtype):
    model = _tiny_lora(dtype)
    generation = _patched_class()(model)
    _assert_no_drift_and_exact(generation, model)
    assert generation._unsloth_vllm_out_of_place_syncs == 8


def test_sharded_and_kill_switch_keep_trl_branch(monkeypatch):
    model = _tiny_lora()
    cls = _patched_class()
    sharded = cls(model, sharded = True)
    sharded.sync_weights()
    assert getattr(sharded, "_unsloth_vllm_out_of_place_syncs", 0) == 0
    monkeypatch.setenv("UNSLOTH_VLLM_INPLACE_MERGE", "1")
    opted_out = cls(model)
    opted_out.sync_weights()
    assert getattr(opted_out, "_unsloth_vllm_out_of_place_syncs", 0) == 0


def test_dense_model_keeps_trl_branch():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    generation = _patched_class()(model)
    generation.sync_weights()
    assert getattr(generation, "_unsloth_vllm_out_of_place_syncs", 0) == 0
    assert sorted(generation.engine.state) == ["0.bias", "0.weight", "1.bias", "1.weight"]


def test_frozen_weight_skip_still_engages():
    """unsloth #18609: frozen non-adapter weights are sent once, merged ones every sync."""
    from unsloth.models.rl_replacements import _unsloth_vllm_sync_colocate

    model = _tiny_lora(target_modules = ["q_proj", "v_proj"])
    generation = _patched_class()(model)
    engine = generation.engine
    for step in range(3):
        _optimizer_step(model, step)
        before = dict(engine.state)
        engine.state.clear()
        _unsloth_vllm_sync_colocate(generation, engine)
        sent = sorted(engine.state)
        engine.state = {**before, **engine.state}
        if step > 0:
            assert sent == sorted(
                f"model.layers.{i}.self_attn.{p}.weight"
                for i in range(2)
                for p in ("q_proj", "v_proj")
            )
        want = _reference(model)
        for name, tensor in want.items():
            assert torch.equal(engine.state[name], tensor), name
    assert generation._unsloth_vllm_frozen_sync["skipped"] > 0


def test_real_trl_vllm_generation_sync_weights_keeps_base_exact():
    """Against the installed TRL's VLLMGeneration (TRL >= 0.28), patched as Unsloth does at import."""
    vllm_generation = pytest.importorskip("trl.generation.vllm_generation")
    from unsloth.models.rl_replacements import vllm_generation_init_patch

    vllm_generation_init_patch()
    cls = vllm_generation.VLLMGeneration
    model = _tiny_lora()
    generation = cls.__new__(cls)
    fake = _TrlGeneration(model)
    generation.model = model
    generation.mode = "colocate"
    generation.enable_sleep_mode = False
    generation.accelerator = fake.accelerator
    generation._dist = fake._dist
    generation.llm = fake.llm
    generation.engine = fake.engine
    _assert_no_drift_and_exact(generation, model)
