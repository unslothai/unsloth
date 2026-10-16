# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL < 0.28 GRPO: _move_model_to_vllm is a no-op only for fast_inference, TRL's own vLLM still gets weights."""

import types
from contextlib import nullcontext

import pytest
import torch


# trl 0.27.2 grpo_trainer.py GRPOTrainer._move_model_to_vllm, PEFT branch dropped.
_TRL_SOURCE = """
    def _move_model_to_vllm(self):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        gather_if_zero3 = nullcontext
        if self.is_fsdp_enabled:
            pass
        else:
            for name, param in self.model.named_parameters():
                name = self._fix_param_name_to_vllm(name)
                with gather_if_zero3([param]):
                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(name, param.data)])

        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()
"""


class _Recorder:
    def __init__(self):
        self.loaded = []
        self.resets = 0

    def load_weights(self, weights):
        self.loaded.extend(name for name, _ in weights)

    def update_named_param(self, name, data):
        self.loaded.append(name)

    def reset_prefix_cache(self):
        self.resets += 1


def _trainer(vllm_mode, shared_weights = False):
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
    trainer = types.SimpleNamespace(
        accelerator = types.SimpleNamespace(
            state = types.SimpleNamespace(deepspeed_plugin = None), is_main_process = True
        ),
        is_fsdp_enabled = False,
        model = torch.nn.Linear(2, 2),
        vllm_mode = vllm_mode,
        vllm_client = rec,
        _fix_param_name_to_vllm = lambda name: name,
    )
    if vllm_mode == "colocate":
        trainer.llm = llm
    return trainer, rec


def _patched_method():
    import textwrap

    from unsloth.models.rl_replacements import grpo_trainer__move_model_to_vllm

    source = grpo_trainer__move_model_to_vllm("_move_model_to_vllm", _TRL_SOURCE)
    try:
        from unsloth.models.rl import _strip_vllm_weight_loads
    except ImportError:
        _strip_vllm_weight_loads = None
    if _strip_vllm_weight_loads is not None:
        source = _strip_vllm_weight_loads("_move_model_to_vllm", source, "grpo_trainer")
    namespace = {"nullcontext": nullcontext}
    exec(textwrap.dedent(source), namespace)
    return namespace["_move_model_to_vllm"]


@pytest.mark.parametrize("vllm_mode", ["colocate", "server"])
def test_trl_vllm_receives_trainer_weights(vllm_mode):
    trainer, rec = _trainer(vllm_mode)
    _patched_method()(trainer)
    assert rec.loaded == ["weight", "bias"]
    assert rec.resets == 1


def test_fast_inference_shared_weights_skips_sync():
    trainer, rec = _trainer("colocate", shared_weights = True)
    assert _patched_method()(trainer) is None
    assert rec.loaded == [] and rec.resets == 0


def test_other_functions_and_trainers_still_stripped():
    from unsloth.models.rl import _strip_vllm_weight_loads

    line = "\n        llm_model.load_weights(state_dict.items())\n"
    for function, trainer_file in (
        ("_generate", "grpo_trainer"),
        ("_move_model_to_vllm", "rloo_trainer"),
    ):
        out = _strip_vllm_weight_loads(function, line, trainer_file)
        assert "load_weights" not in out and "pass" in out
    for function in (
        "_move_model_to_vllm",
        "_sync_fsdp_params_to_vllm",
        "_sync_fsdp2_params_to_vllm",
    ):
        assert _strip_vllm_weight_loads(function, line, "grpo_trainer") == line


class Params4bit(torch.nn.Parameter):
    pass


def test_bnb_quantized_base_keeps_no_op():
    # merge_adapter on a bitsandbytes base requantizes it from the fp32 LoRA delta and the next
    # fast_lora backward fails on dtype, so a quantized model must not enter TRL's sync body.
    trainer, rec = _trainer("colocate")
    trainer.model.weight = Params4bit(trainer.model.weight.data, requires_grad = False)
    method = _patched_method()
    assert method(trainer) is None and method(trainer) is None
    assert rec.loaded == [] and rec.resets == 0


def test_quantized_load_flag_keeps_no_op():
    # FSDP2 wraps Params4bit into a DTensor; the model's load flag still marks it quantized.
    trainer, rec = _trainer("colocate")
    trainer.model.is_loaded_in_4bit = True
    method = _patched_method()
    assert method(trainer) is None
    assert rec.loaded == [] and rec.resets == 0
