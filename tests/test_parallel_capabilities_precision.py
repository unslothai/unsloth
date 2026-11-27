# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The registry refuses float8_training under tp / pp / cp before the split. CPU only."""

import types

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import parallel_capabilities as caps
from unsloth.parallel_env import ParallelLayout

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

_ENV = (
    "UNSLOTH_PARALLEL",
    "UNSLOTH_TP_SIZE",
    "UNSLOTH_PP_SIZE",
    "UNSLOTH_CP_SIZE",
    "UNSLOTH_FSDP2",
    "UNSLOTH_PARALLEL_STRICT",
    "WORLD_SIZE",
)


@pytest.fixture(autouse = True)
def clean_env(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising = False)
    yield monkeypatch


def _config():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(TINY)
    # Tiny widths (8) are below the fp8 GEMM minimum (dims % 16): nothing would convert.
    config.hidden_size, config.intermediate_size, config.head_dim = 64, 128, 16
    return config


def _model():
    from transformers import AutoModelForCausalLM
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(_config(), dtype = torch.bfloat16)


def _lora(model):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(model, LoraConfig(r = 8, target_modules = TARGETS))


def _fp8_lora_attached():
    from unsloth.models import float8_training as f8

    model = _model()
    f8.mark_fp8_lora(model, f8.FP8_LORA_RECIPE)
    model = _lora(model)
    assert f8.enable_fp8_lora(model, f8.FP8_LORA_RECIPE, allow_cpu = True)
    assert any("_unsloth_fp8_lora" in m.__dict__ for m in model.modules())
    return model


def _fp8_lora_requested():
    from unsloth.models import float8_training as f8
    return f8.mark_fp8_lora(_model(), f8.FP8_LORA_RECIPE)


def _float8_full():
    from torchao.float8 import Float8LinearConfig

    from unsloth.models import float8_training as f8

    model = _model()
    converted = f8.convert_model_to_float8_training(
        model, Float8LinearConfig(emulate = True), size_filter = False, check_hardware = False
    )
    assert converted
    return model


def _float8_subclass():
    # torchao's own conversion (no Unsloth markers), then a renamed Float8Linear subclass.
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    from torchao.float8.float8_linear import Float8Linear

    model = convert_to_float8_training(_model(), config = Float8LinearConfig(emulate = True))
    renamed = type("MyFp8Linear", (Float8Linear,), {})
    swapped = 0
    for module in model.modules():
        if type(module) is Float8Linear:
            module.__class__ = renamed
            swapped += 1
    assert swapped
    return model


BUILDERS = {
    "fp8_lora_attached": _fp8_lora_attached,
    "fp8_lora_requested": _fp8_lora_requested,
    "float8_full": _float8_full,
    "float8_subclass": _float8_subclass,
}
LAYOUTS = {
    "cp2": (dict(tp = 1, dp = 1, world = 2, cp = 2), "cp x precision"),
    "tp2": (dict(tp = 2, dp = 1, world = 2), "tp / pp x precision"),
    "pp2": (dict(tp = 1, dp = 1, world = 2, pp = 2), "tp / pp x precision"),
    "cp2_tp2": (dict(tp = 2, dp = 1, world = 4, cp = 2), "cp x precision"),
}


def _snapshot(model):
    return {n: (p.data_ptr(), p.device, p.dtype) for n, p in model.named_parameters()}


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_low_precision_training_is_refused_under_tp_pp_cp_before_the_split(kind, layout):
    model = BUILDERS[kind]()
    assert caps.query_from(None, None, model).mode == "fp8"
    sizes, axis = LAYOUTS[layout]
    before = _snapshot(model)
    with pytest.raises(NotImplementedError) as info:
        caps.check_layout(
            ParallelLayout(fsdp2 = False, **sizes),
            None,
            model,
            types.SimpleNamespace(packing = False),
            cp_backend = "ulysses",
        )
    assert str(info.value).startswith(f"Unsloth: {axis}: "), str(info.value)
    assert _snapshot(model) == before, "a parameter moved before the refusal"


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_low_precision_training_alone_or_with_dp_is_not_refused(kind):
    model = BUILDERS[kind]()
    for fsdp2 in (False, True):
        caps.check_layout(
            ParallelLayout(1, 2, 2, fsdp2),
            None,
            model,
            types.SimpleNamespace(packing = False),
        )


def test_a_float8_request_that_converted_nothing_trains_in_16_bit_and_is_not_refused():
    # Tiny widths are below the fp8 minimum, so enable_fp8_lora converts nothing and training stays 16 bit.
    from transformers import AutoConfig, AutoModelForCausalLM

    from unsloth.models import float8_training as f8

    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(TINY), dtype = torch.bfloat16)
    f8.mark_fp8_lora(model, f8.FP8_LORA_RECIPE)
    model = _lora(model)
    f8.enable_fp8_lora(model, f8.FP8_LORA_RECIPE, allow_cpu = True)
    assert not any("_unsloth_fp8_lora" in m.__dict__ for m in model.modules())
    assert caps.query_from(None, None, model).mode == "lora"
    caps.check_layout(
        ParallelLayout(tp = 2, dp = 1, world = 2, fsdp2 = False),
        None,
        model,
        types.SimpleNamespace(packing = False),
    )


def test_models_without_a_low_precision_marker_keep_their_mode():
    from transformers import AutoModelForCausalLM

    full = AutoModelForCausalLM.from_pretrained(TINY, dtype = torch.bfloat16)
    assert caps.query_from(None, None, full).mode == "full"
    lora = _lora(full)
    assert caps.query_from(None, None, lora).mode == "lora"
    for sizes in (dict(tp = 2), dict(pp = 2), dict(cp = 2)):
        caps.check_layout(
            ParallelLayout(dp = 1, world = 2, fsdp2 = False, **{"tp": 1, **sizes}),
            None,
            lora,
            types.SimpleNamespace(packing = False),
            cp_backend = "ulysses",
        )
    frozen = AutoModelForCausalLM.from_pretrained(TINY, dtype = torch.bfloat16)
    frozen.requires_grad_(False)
    assert caps.query_from(None, None, frozen).mode == "lora"
    quantized = types.SimpleNamespace(
        config = types.SimpleNamespace(quantization_config = {"load_in_4bit": True}),
        is_loaded_in_4bit = True,
    )
    assert caps.query_from(None, None, quantized).mode == "qlora"
    assert caps.query_from(None, None, None).mode == "lora"
    assert caps.query_from(None, None, TINY).mode == "lora"


@pytest.mark.parametrize("env", ["UNSLOTH_PP_SIZE", "UNSLOTH_TP_SIZE"])
def test_the_trainer_init_check_refuses_before_the_split(clean_env, env):
    """models/_utils._parallel_capabilities_pre runs before the pp split and tp sharding."""
    from unsloth.models import _utils, tp_env

    class SFTTrainer:
        def __init__(
            self,
            model = None,
            args = None,
        ):
            self.model = model

    clean_env.setenv(env, "2")
    clean_env.setenv("WORLD_SIZE", "2")
    tp_env._reset_env_layout()
    try:
        model = _fp8_lora_attached()
        before = _snapshot(model)
        trainer = SFTTrainer.__new__(SFTTrainer)
        kwargs = {"model": model, "args": types.SimpleNamespace(packing = False)}
        with pytest.raises(NotImplementedError, match = r"^Unsloth: tp / pp x precision: "):
            _utils._parallel_capabilities_pre(trainer, SFTTrainer.__init__, (), kwargs, model)
        assert _snapshot(model) == before
    finally:
        tp_env._reset_env_layout()
