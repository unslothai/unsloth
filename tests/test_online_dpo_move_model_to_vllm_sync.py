# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import textwrap
import types
from contextlib import nullcontext

import pytest
import torch


# trl 1.15.0 experimental/online_dpo/online_dpo_trainer.py, FSDP branches dropped.
_TRL_SOURCES = {
    "_move_model_to_vllm": """
    def _move_model_to_vllm(self):
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            with self.vllm_client.weight_update():
                self._move_model_to_vllm_inner()
        else:
            self._move_model_to_vllm_inner()

        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()
""",
    "_move_model_to_vllm_inner": """
    def _move_model_to_vllm_inner(self):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()
                for name, param in self.model.named_parameters():
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if self.model.prefix in name:
                        continue
                    if "original_module" in name:
                        continue
                    name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(name, param.data)])
                self.model.unmerge_adapter()
        else:
            for name, param in self.model.named_parameters():
                name = self._fix_param_name_to_vllm(name)
                with gather_if_zero3([param]):
                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(name, param.data)])
""",
    # trl 0.22.2 trainer/online_dpo_trainer.py: the sync is inline, before generation.
    "_generate_vllm": """
    def _generate_vllm(self, model, prompts):
        # Load the latest weights
        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(model.state_dict().items())
        return prompts
""",
}


class _Recorder:
    def __init__(self):
        self.loaded = []
        self.tensors = {}
        self.resets = 0

    def load_weights(self, weights):
        for name, tensor in weights:
            self.loaded.append(name)
            self.tensors[name] = tensor.clone()

    def reset_prefix_cache(self):
        self.resets += 1


class _Proj(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 16, dtype = torch.bfloat16)

    def forward(self, x):
        return self.proj(x)


def _lora_model():
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    return get_peft_model(
        _Proj(), LoraConfig(r = 4, lora_alpha = 8, target_modules = ["proj"], init_lora_weights = False)
    )


def _patched_trainer_class():
    from unsloth.models.rl import _strip_vllm_weight_loads
    from unsloth.models.rl_replacements import RL_FUNCTIONS

    namespace = {
        "nullcontext": nullcontext,
        "is_peft_model": lambda model: hasattr(model, "peft_config"),
    }
    for function, source in _TRL_SOURCES.items():
        for edit in RL_FUNCTIONS["online_dpo_trainer"]:
            source = edit(function, source)
        source = _strip_vllm_weight_loads(function, source, "online_dpo_trainer")
        exec(textwrap.dedent(source), namespace)
    return type("Trainer", (), {name: namespace[name] for name in _TRL_SOURCES})


def _trainer(model, shared_weights = False):
    rec = _Recorder()
    llm = types.SimpleNamespace(
        llm_engine = types.SimpleNamespace(
            model_executor = types.SimpleNamespace(
                driver_worker = types.SimpleNamespace(model_runner = types.SimpleNamespace(model = rec))
            )
        ),
        reset_prefix_cache = rec.reset_prefix_cache,
    )
    if shared_weights:
        llm.shared_weights = True
    trainer = _patched_trainer_class()()
    trainer.accelerator = types.SimpleNamespace(
        state = types.SimpleNamespace(deepspeed_plugin = None),
        is_main_process = True,
        unwrap_model = lambda m: getattr(m, "module", m),
    )
    trainer.model = model
    trainer.vllm_mode = "colocate"
    trainer.llm = llm
    trainer._fix_param_name_to_vllm = lambda name, extra_prefixes = None: name
    return trainer, rec


class _NoAdapterModel(torch.nn.Linear):
    # transformers PeftAdapterMixin.active_adapters on a model with no adapter loaded
    def active_adapters(self):
        raise ValueError("No adapter loaded. Please load an adapter first.")


def test_trl_colocated_vllm_receives_dense_weights():
    trainer, rec = _trainer(_NoAdapterModel(2, 2))
    trainer._move_model_to_vllm()
    assert rec.loaded == ["weight", "bias"]
    assert rec.resets == 1


def test_trl_colocated_vllm_receives_lora_merged_out_of_place():
    model = _lora_model()
    layer = model.base_model.model.proj
    base = layer.base_layer.weight.detach().clone()
    expected = (
        base.float()
        + (layer.lora_B["default"].weight.float() @ layer.lora_A["default"].weight.float())
        * layer.scaling["default"]
    ).to(base.dtype)
    trainer, rec = _trainer(model)
    for _ in range(3):
        trainer._move_model_to_vllm()
    assert rec.loaded[:2] == ["proj.weight", "proj.bias"]
    assert torch.equal(rec.tensors["proj.weight"], expected)
    assert torch.equal(layer.base_layer.weight, base)


def test_fast_inference_shared_weights_skips_sync():
    model = _lora_model()
    base = model.base_model.model.proj.base_layer.weight.detach().clone()
    trainer, rec = _trainer(model, shared_weights = True)
    assert trainer._move_model_to_vllm() is None
    assert trainer._generate_vllm(model, ["p"]) == ["p"]
    assert rec.loaded == [] and rec.resets == 0
    assert torch.equal(model.base_model.model.proj.base_layer.weight, base)


def test_trl_022_inline_sync_reaches_trl_engine():
    dense = torch.nn.Linear(2, 2)
    trainer, rec = _trainer(dense)
    assert trainer._generate_vllm(dense, ["p"]) == ["p"]
    assert rec.loaded == ["weight", "bias"]
    assert rec.resets == 1

    # DDP passes the wrapped model; vLLM rejects the "module." prefix.
    trainer, rec = _trainer(dense)
    trainer._generate_vllm(types.SimpleNamespace(module = dense), ["p"])
    assert rec.loaded == ["weight", "bias"]

    trainer, rec = _trainer(dense)
    trainer._generate_vllm(
        types.SimpleNamespace(module = types.SimpleNamespace(_orig_mod = dense)), ["p"]
    )
    assert rec.loaded == ["weight", "bias"]

    trainer, rec = _trainer(dense)
    trainer.accelerator.state.deepspeed_plugin = types.SimpleNamespace(zero_stage = 3)
    with pytest.raises(RuntimeError, match = "sharded"):
        trainer._generate_vllm(dense, ["p"])
    assert rec.loaded == []

    model = _lora_model()
    base = model.base_model.model.proj.base_layer.weight.detach().clone()
    trainer, rec = _trainer(model)
    trainer._generate_vllm(model, ["p"])
    assert rec.loaded[:2] == ["proj.weight", "proj.bias"]
    assert not torch.equal(rec.tensors["proj.weight"], base)
    assert torch.equal(model.base_model.model.proj.base_layer.weight, base)


# trl 1.15.0 experimental/online_dpo/online_dpo_trainer.py, generation trimmed.
_COLOCATE_SOURCE = """
    def _generate_vllm_colocate(self, prompts, images=None):
        if self.args.vllm_enable_sleep_mode:
            self.llm.wake_up(tags=["weights"])
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step
        if self.args.vllm_enable_sleep_mode:
            self.llm.sleep(level=2)
        return prompts
"""


def test_level2_sleep_reloads_weights_on_the_next_microbatch():
    from unsloth.models.rl_replacements import RL_FUNCTIONS

    source = _COLOCATE_SOURCE
    for edit in RL_FUNCTIONS["online_dpo_trainer"]:
        source = edit("_generate_vllm_colocate", source)
    namespace = {}
    exec(textwrap.dedent(source), namespace)
    syncs = []
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(vllm_enable_sleep_mode = True),
        llm = types.SimpleNamespace(wake_up = lambda tags: None, sleep = lambda level: None),
        state = types.SimpleNamespace(global_step = 0),
        _last_loaded_step = -1,
        _move_model_to_vllm = lambda: syncs.append(1),
    )
    for _ in range(2):
        namespace["_generate_vllm_colocate"](trainer, ["p"])
    assert len(syncs) == 2
