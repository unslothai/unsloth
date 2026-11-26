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


def _engine(quantization):
    # vLLM LLM: llm.llm_engine.model_config.quantization is the method the engine was built with.
    model_config = types.SimpleNamespace(quantization = quantization)
    return types.SimpleNamespace(llm_engine = types.SimpleNamespace(model_config = model_config))


def test_engine_quantization_is_read_from_the_llm():
    from unsloth.models.rl_replacements import _unsloth_vllm_engine_quantization

    assert _unsloth_vllm_engine_quantization(_engine("bitsandbytes")) == "bitsandbytes"
    assert _unsloth_vllm_engine_quantization(_engine(None)) is None
    assert _unsloth_vllm_engine_quantization(None) is None


def test_unquantized_engine_for_a_4bit_model_warns_once(monkeypatch, capsys):
    # TRL >= 1.19 builds its colocated engine without quantization = "bitsandbytes": a 16-bit vLLM copy.
    from unsloth.models.rl_replacements import _unsloth_trl_vllm_engine_check

    monkeypatch.delenv("UNSLOTH_VLLM_ALLOW_STALE_ROLLOUTS", raising = False)
    model = _Model()
    generation = _patched(_DequantizingGeneration)(model)
    generation.llm = _engine(None)
    _unsloth_trl_vllm_engine_check(generation, model)
    out = capsys.readouterr().out
    assert "builds its own vLLM engine unquantized" in out and "fast_inference = True" in out
    generation.sync_weights()
    assert generation.loaded == ["linear.weight", "linear.bias"]


def test_bitsandbytes_engine_refuses_dequantized_weights(monkeypatch, capsys):
    # A pre-quantized bnb-4bit checkpoint gives a bitsandbytes engine, which cannot load 16-bit weights.
    from unsloth.models.rl_replacements import _unsloth_trl_vllm_engine_check

    monkeypatch.delenv("UNSLOTH_VLLM_ALLOW_STALE_ROLLOUTS", raising = False)
    model = _Model()
    generation = _patched(_DequantizingGeneration)(model)
    generation.llm = _engine("bitsandbytes")
    _unsloth_trl_vllm_engine_check(generation, model)
    assert "unquantized" not in capsys.readouterr().out
    with pytest.raises(RuntimeError, match = "bitsandbytes-quantized"):
        generation.sync_weights()
    assert generation.loaded == []
    monkeypatch.setenv("UNSLOTH_VLLM_ALLOW_STALE_ROLLOUTS", "1")
    generation.sync_weights()
    assert generation.loaded == []


def test_16bit_model_gets_no_engine_warning(capsys):
    from unsloth.models.rl_replacements import _unsloth_trl_vllm_engine_check

    model = _Model(in_4bit = False)
    generation = _patched(_DequantizingGeneration)(model)
    generation.llm = _engine(None)
    _unsloth_trl_vllm_engine_check(generation, model)
    assert capsys.readouterr().out == ""


def test_installed_trl_init_vllm_runs_the_engine_check():
    # The patched VLLMGeneration._init_vllm calls the check right after building TRL's own engine.
    import inspect

    from packaging.version import Version

    trl_version = Version(pytest.importorskip("trl").__version__)
    if trl_version < Version("0.28.0"):
        pytest.skip("TRL < 0.28 has no VLLMGeneration")
    vllm_generation = pytest.importorskip("trl.generation.vllm_generation")
    from unsloth.models.rl_replacements import vllm_generation_init_patch

    vllm_generation_init_patch()
    src = inspect.getsource(vllm_generation.VLLMGeneration._init_vllm)
    assert "_unsloth_trl_vllm_engine_check(self, model)" in src
    assert callable(getattr(vllm_generation, "_unsloth_trl_vllm_engine_check", None))


def test_vllm_without_bitsandbytes_points_to_fast_inference():
    from unsloth.models.rl_replacements import _unsloth_trl_vllm_init_error

    vllm_error = ValueError(
        "1 validation error for ModelConfig\n  Value error, Unknown quantization method: bitsandbytes. "
        "Must be one of ['awq', 'auto_awq', 'fp8', 'mxfp8']."
    )
    replacement = _unsloth_trl_vllm_init_error(vllm_error)
    assert isinstance(replacement, RuntimeError)
    assert "fast_inference = True" in str(replacement) and "vllm-bnb-plugin" in str(replacement)
    assert _unsloth_trl_vllm_init_error(ValueError("CUDA out of memory")) is None
    assert _unsloth_trl_vllm_init_error(ValueError("Unknown quantization method: gguf")) is None


def test_installed_trl_init_vllm_maps_the_missing_bitsandbytes_error():
    import inspect

    from packaging.version import Version

    trl_version = Version(pytest.importorskip("trl").__version__)
    if trl_version < Version("0.28.0"):
        pytest.skip("TRL < 0.28 has no VLLMGeneration")
    vllm_generation = pytest.importorskip("trl.generation.vllm_generation")
    from unsloth.models.rl_replacements import vllm_generation_init_patch

    vllm_generation_init_patch()
    src = inspect.getsource(vllm_generation.VLLMGeneration._init_vllm)
    assert "_unsloth_trl_vllm_init_error(_unsloth_init_error)" in src
    assert "raise _unsloth_replacement from _unsloth_init_error" in src
    # The engine check still runs only after TRL's own LLM(...) returned.
    assert src.index("except Exception as _unsloth_init_error") < src.index(
        "_unsloth_trl_vllm_engine_check(self, model)"
    )
    assert callable(getattr(vllm_generation, "_unsloth_trl_vllm_init_error", None))
