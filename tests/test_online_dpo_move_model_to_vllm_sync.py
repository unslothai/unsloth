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
}


class _Recorder:
    def __init__(self):
        self.loaded = []
        self.resets = 0

    def load_weights(self, weights):
        self.loaded.extend(name for name, _ in weights)

    def reset_prefix_cache(self):
        self.resets += 1


class _LoraModel(torch.nn.Module):
    prefix = "lora_"

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Module()
        self.proj.base_layer = torch.nn.Linear(2, 2)
        self.proj.lora_A = torch.nn.Parameter(torch.ones(1, 2))
        self.merges = 0

    def merge_adapter(self):
        self.merges += 1

    def unmerge_adapter(self):
        pass


def _patched_trainer_class():
    from unsloth.models.rl import _strip_vllm_weight_loads
    from unsloth.models.rl_replacements import RL_FUNCTIONS

    namespace = {
        "nullcontext": nullcontext,
        "is_peft_model": lambda model: hasattr(model, "merge_adapter"),
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
        state = types.SimpleNamespace(deepspeed_plugin = None), is_main_process = True
    )
    trainer.model = model
    trainer.vllm_mode = "colocate"
    trainer.llm = llm
    trainer._fix_param_name_to_vllm = lambda name, extra_prefixes = None: name
    return trainer, rec


@pytest.mark.parametrize(
    "model, names",
    [(torch.nn.Linear(2, 2), ["weight", "bias"]), (_LoraModel(), ["proj.weight", "proj.bias"])],
    ids = ["full", "lora"],
)
def test_trl_colocated_vllm_receives_trainer_weights(model, names):
    trainer, rec = _trainer(model)
    trainer._move_model_to_vllm()
    assert rec.loaded == names
    assert rec.resets == 1


def test_fast_inference_shared_weights_skips_sync():
    model = _LoraModel()
    trainer, rec = _trainer(model, shared_weights = True)
    assert trainer._move_model_to_vllm() is None
    assert rec.loaded == [] and rec.resets == 0 and model.merges == 0
