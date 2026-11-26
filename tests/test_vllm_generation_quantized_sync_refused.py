# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL 0.28 to 1.18 VLLMGeneration (use_vllm = True, fast_inference = False) merges the adapters into a
quantized base in place and sends param.data, the packed bitsandbytes storage. vLLM cannot load that:
a colocated 4-bit engine fails `assert param_data.shape == loaded_weight.shape` on the first sync
(Qwen3-0.6B QLoRA, TRL 1.18.2, vLLM 0.29). Unsloth raises the same clear error as on TRL < 0.28, or
skips the sync under UNSLOTH_VLLM_ALLOW_STALE_ROLLOUTS=1. TRL >= 1.19 sends dequantized merges."""

import types

import pytest
import torch


class _Model(torch.nn.Module):
    def __init__(self, in_4bit = True):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.peft_config = {"default": object()}
        self.prefix = "lora_"
        self.is_loaded_in_4bit = in_4bit
        self.merges = 0

    def merge_adapter(self):
        self.merges += 1

    def unmerge_adapter(self):
        pass


class _Generation:
    def __init__(self, model):
        self.model = model
        self.loaded = []
        self.accelerator = types.SimpleNamespace(
            state = types.SimpleNamespace(deepspeed_plugin = None, fsdp_plugin = None)
        )
        self._dist = types.SimpleNamespace(is_fsdp = False)

    def _iter_named_params(self):
        # TRL 1.18.2's unsharded PEFT branch: in-place merge, then param.data under vLLM's names.
        model = self.model
        model.merge_adapter()
        for name, param in model.named_parameters():
            yield name, param.data
        model.unmerge_adapter()

    def sync_weights(self):
        for name, param in self._iter_named_params():
            self.loaded.append(name)


class _DequantizingGeneration(_Generation):
    def _iter_named_params(self):
        # TRL >= 1.19 (#8269): bitsandbytes 4-bit weights are sent through bnb.functional.dequantize_4bit.
        dequantize_4bit = None  # noqa: F841
        for name, param in self.model.named_parameters():
            yield name, param.data


def _patched(base):
    from unsloth.models.rl_replacements import _unsloth_vllm_patch_iter_named_params

    cls = type("Patched" + base.__name__, (base,), {})
    cls._iter_named_params = base._iter_named_params
    assert _unsloth_vllm_patch_iter_named_params(cls)
    return cls


def test_quantized_in_place_sync_raises(monkeypatch):
    monkeypatch.delenv("UNSLOTH_VLLM_ALLOW_STALE_ROLLOUTS", raising = False)
    model = _Model()
    generation = _patched(_Generation)(model)
    with pytest.raises(RuntimeError, match = "fast_inference = True"):
        generation.sync_weights()
    # Refused before TRL merged into the packed base.
    assert model.merges == 0 and generation.loaded == []


def test_allow_stale_skips_the_sync(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_ALLOW_STALE_ROLLOUTS", "1")
    model = _Model()
    generation = _patched(_Generation)(model)
    generation.sync_weights()
    generation.sync_weights()
    assert model.merges == 0 and generation.loaded == []


def test_trl_that_dequantizes_keeps_its_sync(monkeypatch):
    monkeypatch.delenv("UNSLOTH_VLLM_ALLOW_STALE_ROLLOUTS", raising = False)
    generation = _patched(_DequantizingGeneration)(_Model())
    generation.sync_weights()
    assert generation.loaded == ["linear.weight", "linear.bias"]


def test_unquantized_peft_is_not_refused(monkeypatch):
    monkeypatch.delenv("UNSLOTH_VLLM_ALLOW_STALE_ROLLOUTS", raising = False)
    monkeypatch.setenv(
        "UNSLOTH_VLLM_INPLACE_MERGE", "1"
    )  # keep TRL's own branch for this fake model
    model = _Model(in_4bit = False)
    generation = _patched(_Generation)(model)
    generation.sync_weights()
    assert model.merges == 1 and generation.loaded == ["linear.weight", "linear.bias"]
