# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL 0.22 to 0.27 RLOO: _move_model_to_vllm must reach TRL's own vLLM engine (use_vllm = True,
fast_inference = False), and return early for the fast_inference engine that shares the weights.

The generic trainer patcher rewrote every `load_weights(...)` / `model_executor.driver_worker` line
to `pass`, exempting only GRPO and Online DPO, so RLOO with a colocated TRL engine sampled every
rollout from the initial checkpoint without an error. TRL 0.28+ RLOO syncs via VLLMGeneration."""

import textwrap
import types
from contextlib import nullcontext

import torch


# trl 0.27.2 trainer/rloo_trainer.py RLOOTrainer._move_model_to_vllm, verbatim (decorator dropped).
_TRL_SOURCE = """
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        self._sync_fsdp1_params_to_vllm(
                            self.model
                        )  # use memory-efficient post-order traversal for FSDP
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        # Skip PEFT layers: they don’t exist in vLLM, and they are merged already.
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(self.model)
            else:
                for name, param in self.model.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()
"""


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


class _NoAdapterModel(torch.nn.Linear):
    # transformers PeftAdapterMixin.active_adapters on a model with no adapter loaded
    def active_adapters(self):
        raise ValueError("No adapter loaded. Please load an adapter first.")


def _lora_model():
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    return get_peft_model(
        _Proj(), LoraConfig(r = 4, lora_alpha = 8, target_modules = ["proj"], init_lora_weights = False)
    )


def _patched_move_model_to_vllm():
    from unsloth.models.rl import _strip_vllm_weight_loads
    from unsloth.models.rl_replacements import RL_FUNCTIONS

    source = _TRL_SOURCE
    for edit in RL_FUNCTIONS["rloo_trainer"]:
        source = edit("_move_model_to_vllm", source)
    source = _strip_vllm_weight_loads("_move_model_to_vllm", source, "rloo_trainer")
    namespace = {
        "nullcontext": nullcontext,
        "is_peft_model": lambda model: hasattr(model, "peft_config"),
    }
    exec(textwrap.dedent(source), namespace)
    return namespace["_move_model_to_vllm"]


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
    trainer = type("Trainer", (), {"_move_model_to_vllm": _patched_move_model_to_vllm()})()
    trainer.accelerator = types.SimpleNamespace(
        state = types.SimpleNamespace(deepspeed_plugin = None),
        is_main_process = True,
        unwrap_model = lambda m: getattr(m, "module", m),
    )
    trainer.model = model
    trainer.is_fsdp_enabled = False
    trainer.vllm_mode = "colocate"
    trainer.llm = llm
    trainer._fix_param_name_to_vllm = lambda name, extra_prefixes = None: name
    return trainer, rec


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
    # TRL's merge_adapter / unmerge_adapter round trip would move the bf16 base every sync.
    assert torch.equal(layer.base_layer.weight, base)


def test_fast_inference_shared_weights_skips_sync():
    model = _lora_model()
    base = model.base_model.model.proj.base_layer.weight.detach().clone()
    merges = []
    model.merge_adapter = lambda *a, **k: merges.append(1)
    trainer, rec = _trainer(model, shared_weights = True)
    assert trainer._move_model_to_vllm() is None
    assert merges == [], "the shared fast_inference base must not be merged into"
    assert rec.loaded == [] and rec.resets == 0
    assert torch.equal(model.base_model.model.proj.base_layer.weight, base)


def test_rloo_generated_trainer_defines_grpo_update_sampling_params():
    # patch_functions rewrites TRL 0.22 to 0.27 RLOO's `SamplingParams(**generation_kwargs)` into a call
    # to grpo_update_SamplingParams; the generated RLOO trainer must define it (NameError at the first rollout).
    import inspect

    from unsloth.models.rl_replacements import RL_PRE_ITEMS, grpo_update_SamplingParams

    namespace = {"inspect": inspect}
    for item in RL_PRE_ITEMS["rloo_trainer"]:
        if "def grpo_update_SamplingParams" in item:
            exec(item, namespace)
    assert "grpo_update_SamplingParams" in namespace

    class SamplingParams:
        def __init__(
            self,
            n = 1,
            temperature = 1.0,
            max_tokens = 16,
            top_p = 1.0,
        ):
            pass

    kwargs = {"n": 1, "temperature": 0.7, "max_tokens": 8, "guided_decoding": None}
    assert namespace["grpo_update_SamplingParams"](SamplingParams, kwargs) == {
        "n": 1,
        "temperature": 0.7,
        "max_tokens": 8,
    }
    assert grpo_update_SamplingParams(SamplingParams, kwargs) == {
        "n": 1,
        "temperature": 0.7,
        "max_tokens": 8,
    }
