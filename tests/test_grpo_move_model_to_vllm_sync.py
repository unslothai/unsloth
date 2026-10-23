# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL < 0.28 GRPO: _move_model_to_vllm is a no-op only for fast_inference, TRL's own vLLM still gets weights
(dense as is, LoRA merged out of place); a quantized base raises instead of training against frozen rollouts."""

import types
from contextlib import nullcontext

import pytest
import torch


# trl 0.27.2 grpo_trainer.py GRPOTrainer._move_model_to_vllm, FSDP branches dropped.
_TRL_SOURCE = """
    def _move_model_to_vllm(self):
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
        elif self.is_fsdp_enabled:
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


def _trl_fix_param_name_to_vllm(name, extra_prefixes = None):
    # trl 0.27.2 grpo_trainer.py GRPOTrainer._fix_param_name_to_vllm
    for prefix in ["_checkpoint_wrapped_module."] + (extra_prefixes or []):
        name = name.replace(prefix, "")
    return name


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
        _fix_param_name_to_vllm = _trl_fix_param_name_to_vllm,
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
    namespace = {
        "nullcontext": nullcontext,
        "is_peft_model": lambda model: hasattr(model, "merge_adapter"),
    }
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


_STALE_ENV = "UNSLOTH_VLLM_ALLOW_STALE_ROLLOUTS"


def test_bnb_quantized_base_raises(monkeypatch):
    # merge_adapter on a bitsandbytes base requantizes it from the fp32 LoRA delta and the next
    # fast_lora backward fails on dtype, and packed bytes are not vLLM's layout: rollouts would stay
    # at the initial weights, so refuse before the first rollout instead of training against them.
    monkeypatch.delenv(_STALE_ENV, raising = False)
    trainer, rec = _trainer("colocate")
    trainer.model.weight = Params4bit(trainer.model.weight.data, requires_grad = False)
    with pytest.raises(RuntimeError, match = "fast_inference = True"):
        _patched_method()(trainer)
    assert rec.loaded == [] and rec.resets == 0


def test_bnb_quantized_base_opt_in_keeps_no_op(monkeypatch):
    monkeypatch.setenv(_STALE_ENV, "1")
    trainer, rec = _trainer("colocate")
    trainer.model.weight = Params4bit(trainer.model.weight.data, requires_grad = False)
    method = _patched_method()
    assert method(trainer) is None and method(trainer) is None
    assert rec.loaded == [] and rec.resets == 0
    assert trainer._unsloth_vllm_sync_skip is True


@pytest.mark.parametrize("vllm_mode", ["colocate", "server"])
def test_quantized_load_flag_raises(vllm_mode, monkeypatch):
    # FSDP2 wraps Params4bit into a DTensor; the model's load flag still marks it quantized.
    monkeypatch.delenv(_STALE_ENV, raising = False)
    trainer, rec = _trainer(vllm_mode)
    trainer.model.is_loaded_in_4bit = True
    with pytest.raises(RuntimeError, match = "quantized"):
        _patched_method()(trainer)
    assert rec.loaded == [] and rec.resets == 0


class _WeightRecorder(_Recorder):
    def __init__(self):
        super().__init__()
        self.tensors = []

    def load_weights(self, weights):
        for name, tensor in weights:
            self.loaded.append(name)
            self.tensors.append((name, tensor.detach().clone()))

    def update_named_param(self, name, data):
        self.loaded.append(name)
        self.tensors.append((name, data.detach().clone()))


def _lora_model(dtype, **kwargs):
    peft = pytest.importorskip("peft")
    torch.manual_seed(1)
    base = torch.nn.Sequential(torch.nn.Linear(64, 64, bias = False), torch.nn.Linear(64, 8)).to(
        dtype
    )
    config = peft.LoraConfig(
        r = 4, lora_alpha = 8, target_modules = ["0"], init_lora_weights = False, **kwargs
    )
    return peft.get_peft_model(base, config)


def _lora_trainer(vllm_mode, dtype, **kwargs):
    model = _lora_model(dtype, **kwargs)
    trainer, _ = _trainer(vllm_mode)
    rec = _WeightRecorder()
    trainer.model = model
    trainer.vllm_client = rec
    if vllm_mode == "colocate":
        trainer.llm.llm_engine.model_executor.driver_worker.model_runner.model = rec
        trainer.llm.reset_prefix_cache = rec.reset_prefix_cache
    return trainer, rec, model


def _expected_merged(model):
    layer = model.base_model.model[0]
    weight = layer.base_layer.weight
    delta = layer.lora_B["default"].weight.float() @ layer.lora_A["default"].weight.float()
    return (weight.float() + delta * layer.scaling["default"]).to(weight.dtype)


@pytest.mark.parametrize("vllm_mode", ["colocate", "server"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lora_sync_pushes_merged_weights_every_step(vllm_mode, dtype):
    # TRL's own vLLM engine (fast_inference = False) must sample from the trained adapter: push
    # the merged weight under the base-layer name each step, as TRL's PEFT branch does.
    trainer, rec, model = _lora_trainer(vllm_mode, dtype)
    method = _patched_method()
    for step in range(2):
        with torch.no_grad():
            model.base_model.model[0].lora_B["default"].weight.add_(0.5)
        before = len(rec.tensors)
        assert method(trainer) is None
        pushed = dict(rec.tensors[before:])
        assert sorted(pushed) == ["0.weight", "1.bias", "1.weight"]
        assert torch.equal(pushed["0.weight"], _expected_merged(model))
        assert torch.equal(pushed["1.weight"], model.base_model.model[1].weight)
    assert rec.resets == 2
    assert trainer._unsloth_vllm_sync_skip is False


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lora_sync_never_touches_frozen_base(dtype):
    # merge_adapter + unmerge_adapter is a lossy round trip in fp16 / bf16, so the merged weights
    # are built out of place and the frozen base stays bit-identical across syncs.
    trainer, rec, model = _lora_trainer("colocate", dtype)
    layer = model.base_model.model[0]
    before = layer.base_layer.weight.detach().clone()
    method = _patched_method()
    for _ in range(3):
        assert method(trainer) is None
    assert torch.equal(layer.base_layer.weight, before)
    assert not layer.merged
    assert len(rec.tensors) == 9


def test_dora_uses_trl_peft_branch():
    # DoRA needs PEFT's own merge (magnitude vector): fall back to TRL's body, which still syncs.
    trainer, rec, model = _lora_trainer("colocate", torch.float32, use_dora = True)
    assert _patched_method()(trainer) is None
    assert "0.weight" in rec.loaded and not model.base_model.model[0].merged
    assert rec.resets == 1


def test_fp8_base_raises(monkeypatch):
    # merge_adapter refuses FP8 grouped-linear bases, and a 1-byte float base is not a dense sync.
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("no float8 dtype")
    monkeypatch.delenv(_STALE_ENV, raising = False)
    trainer, rec = _trainer("colocate")
    trainer.model.weight = torch.nn.Parameter(
        trainer.model.weight.data.to(torch.float8_e4m3fn), requires_grad = False
    )
    with pytest.raises(RuntimeError, match = "fast_inference = True"):
        _patched_method()(trainer)
    assert rec.loaded == [] and rec.resets == 0


# Codex review round 2: quantizers beyond bitsandbytes, non-LoRA PEFT, unmergeable LoRA variants,
# target_parameters, custom modules_to_save adapter names, trainable tokens.


class _PackedLinear(torch.nn.Module):
    # GPTQ QuantLinear / AWQ WQLinear keep int32 packed weights in a qweight buffer.
    def __init__(self):
        super().__init__()
        self.register_buffer("qweight", torch.zeros(2, 2, dtype = torch.int32))
        self.bias = torch.nn.Parameter(torch.zeros(2))


def _mark_hf_quantizer(model):
    model.hf_quantizer = object()


def _mark_quantization_method(model):
    model.quantization_method = "hqq"


def _mark_quantization_config(model):
    model.config = types.SimpleNamespace(quantization_config = {"quant_method": "awq", "bits": 4})


def _mark_packed_weight(model):
    model.add_module("packed", _PackedLinear())


@pytest.mark.parametrize(
    "mark",
    [_mark_hf_quantizer, _mark_quantization_method, _mark_quantization_config, _mark_packed_weight],
)
def test_gptq_awq_hqq_style_quantized_base_raises(mark, monkeypatch):
    monkeypatch.delenv(_STALE_ENV, raising = False)
    trainer, rec = _trainer("colocate")
    mark(trainer.model)
    with pytest.raises(RuntimeError, match = "quantized"):
        _patched_method()(trainer)
    assert rec.loaded == [] and rec.resets == 0


def test_dequantized_config_still_syncs():
    # A dequantize = True load leaves dense weights: it is a dense sync, not a quantized base.
    trainer, rec = _trainer("colocate")
    trainer.model.config = types.SimpleNamespace(
        quantization_config = {"quant_method": "fp8", "dequantize": True}
    )
    _patched_method()(trainer)
    assert rec.loaded == ["weight", "bias"]


def _peft_trainer(model, vllm_mode = "colocate"):
    trainer, _ = _trainer(vllm_mode)
    rec = _WeightRecorder()
    trainer.model = model
    trainer.vllm_client = rec
    if vllm_mode == "colocate":
        trainer.llm.llm_engine.model_executor.driver_worker.model_runner.model = rec
        trainer.llm.reset_prefix_cache = rec.reset_prefix_cache
    return trainer, rec


def _perturb_trainable(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.add_(torch.randn_like(param) * 0.3 + 0.2)


def _merged_state(model):
    import copy
    return {k: v.clone() for k, v in copy.deepcopy(model).merge_and_unload().state_dict().items()}


def _base_mlp():
    torch.manual_seed(1)
    return torch.nn.Sequential(torch.nn.Linear(16, 16, bias = False), torch.nn.Linear(16, 8))


def _non_lora_config(kind):
    peft = pytest.importorskip("peft")
    if kind == "loha":
        return peft.LoHaConfig(r = 2, target_modules = ["0"], init_weights = False)
    if kind == "lokr":
        return peft.LoKrConfig(r = 2, target_modules = ["0"], init_weights = False)
    if kind == "ia3":
        return peft.IA3Config(target_modules = ["0"], feedforward_modules = [])
    return peft.VeraConfig(r = 2, target_modules = ["0"])


@pytest.mark.parametrize("kind", ["loha", "lokr", "ia3", "vera"])
def test_non_lora_peft_syncs_merged_weights_via_trl(kind):
    # VeRA / LoHa / LoKr / IA3 have no lora_A: the out-of-place LoRA merge would push the frozen base
    # weight. TRL's PEFT branch merges them with PEFT's own merge_adapter, so it must be used.
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(_base_mlp(), _non_lora_config(kind))
    _perturb_trainable(model)
    expected = _merged_state(model)
    trainer, rec = _peft_trainer(model)
    assert _patched_method()(trainer) is None
    assert trainer._unsloth_vllm_sync_kind == "trl"
    pushed = dict(rec.tensors)
    assert sorted(pushed) == ["0.weight", "1.bias", "1.weight"]
    assert not torch.equal(pushed["0.weight"], model.base_model.model[0].base_layer.weight)
    torch.testing.assert_close(pushed["0.weight"], expected["0.weight"])
    assert rec.resets == 1


def test_alora_raises_instead_of_flattening(monkeypatch):
    # aLoRA only applies after its invocation tokens; PEFT refuses to merge it, and a static merge
    # would serve a different model than the one being trained.
    peft = pytest.importorskip("peft")
    if "alora_invocation_tokens" not in peft.LoraConfig.__dataclass_fields__:
        pytest.skip("this PEFT has no aLoRA")
    monkeypatch.delenv(_STALE_ENV, raising = False)
    config = peft.LoraConfig(
        r = 2, target_modules = ["0"], alora_invocation_tokens = [5], init_lora_weights = False
    )
    model = peft.get_peft_model(_base_mlp(), config)
    trainer, rec = _peft_trainer(model)
    with pytest.raises(RuntimeError, match = "aLoRA"):
        _patched_method()(trainer)
    assert rec.loaded == [] and rec.resets == 0


class _Experts(torch.nn.Module):
    # Fused MoE experts as in transformers 5 (gate_up_proj is a bare 3D parameter).
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.randn(4, 16, 8))

    def forward(self, x):
        return torch.einsum("bi,eio->bo", x, self.gate_up_proj)


def test_target_parameters_param_wrapper_syncs_merged_experts():
    # ParamWrapper has no base_layer.weight, so the out-of-place merge pushed the untrained experts.
    peft = pytest.importorskip("peft")
    if "target_parameters" not in peft.LoraConfig.__dataclass_fields__:
        pytest.skip("this PEFT has no target_parameters")
    torch.manual_seed(1)
    base = torch.nn.Sequential(torch.nn.Linear(16, 16, bias = False))
    base.add_module("experts", _Experts())
    config = peft.LoraConfig(
        r = 2,
        target_modules = ["0"],
        target_parameters = ["experts.gate_up_proj"],
        init_lora_weights = False,
    )
    model = peft.get_peft_model(base, config)
    _perturb_trainable(model)
    expected = _merged_state(model)
    trainer, rec = _peft_trainer(model)
    assert _patched_method()(trainer) is None
    assert trainer._unsloth_vllm_sync_kind == "trl"
    pushed = dict(rec.tensors)
    assert sorted(pushed) == ["0.weight", "experts.gate_up_proj"]
    torch.testing.assert_close(pushed["experts.gate_up_proj"], expected["experts.gate_up_proj"])
    torch.testing.assert_close(pushed["0.weight"], expected["0.weight"])


@pytest.mark.parametrize("second_adapter", [False, True])
def test_modules_to_save_custom_adapter_name(second_adapter):
    # TRL strips only "modules_to_save.default.": a custom name reached vLLM as is and the trained
    # head was never synced. Push the active adapter's copy under the base name, and nothing else.
    peft = pytest.importorskip("peft")
    config = peft.LoraConfig(
        r = 2, target_modules = ["0"], modules_to_save = ["1"], init_lora_weights = False
    )
    model = peft.get_peft_model(_base_mlp(), config, adapter_name = "mine")
    if second_adapter:
        model.add_adapter("other", config)
        model.set_adapter("mine")
    _perturb_trainable(model)
    head = model.base_model.model[1]
    trainer, rec = _peft_trainer(model)
    assert _patched_method()(trainer) is None
    assert trainer._unsloth_vllm_sync_kind == "peft_merged"
    assert sorted(rec.loaded) == ["0.weight", "1.bias", "1.weight"]
    pushed = dict(rec.tensors)
    assert torch.equal(pushed["1.weight"], head.modules_to_save["mine"].weight)
    assert not torch.equal(pushed["1.weight"], head.original_module.weight)
    layer = model.base_model.model[0]
    delta = layer.lora_B["mine"].weight.float() @ layer.lora_A["mine"].weight.float()
    expected = (layer.base_layer.weight.float() + delta * layer.scaling["mine"]).to(torch.float32)
    assert torch.equal(pushed["0.weight"], expected)


def test_trainable_token_indices_raises(monkeypatch):
    # The trained rows live in token_adapter.trainable_tokens_delta.<adapter>; neither TRL's names
    # nor a LoRA merge map that onto vLLM's embedding, so refuse instead of dropping the rows.
    peft = pytest.importorskip("peft")
    if "trainable_token_indices" not in peft.LoraConfig.__dataclass_fields__:
        pytest.skip("this PEFT has no trainable_token_indices")
    monkeypatch.delenv(_STALE_ENV, raising = False)
    torch.manual_seed(1)
    base = torch.nn.Sequential(torch.nn.Embedding(32, 16), torch.nn.Linear(16, 16))
    config = peft.LoraConfig(r = 2, target_modules = ["1"], trainable_token_indices = {"0": [1, 3]})
    model = peft.get_peft_model(base, config)
    trainer, rec = _peft_trainer(model)
    with pytest.raises(RuntimeError, match = "trainable"):
        _patched_method()(trainer)
    assert rec.loaded == [] and rec.resets == 0
