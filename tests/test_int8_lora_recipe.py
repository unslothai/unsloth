# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The opt-in rotated int8 LoRA base (UNSLOTH_LORA_BASE_INT8=1): registry entry, request resolution and refusals.
CPU only; the kernels themselves are covered by test_int8_lora_cuda.py."""

import pytest
import torch

from unsloth import precision as P
from unsloth.models import float8_training as f8t


def test_registry_entry():
    r = P.get_recipe("int8_lora_rotated")
    assert r.kind == "frozen_base" and r.modes == ("lora",) and r.compile == "opaque"
    for gpu in ("sm80", "sm89", "sm120"):  # measured faster than bf16 end to end
        assert r.gate(gpu).status == "explicit" and r.gate(gpu).evidence
    for gpu in ("sm75", "rocm"):
        assert r.gate(gpu).status == "refused" and r.gate(gpu).reason
    # opt-in only: never picked by name
    assert r.gate("sm120").status != "named"


def _linear_model(dtype = torch.bfloat16, quantized = False):
    model = torch.nn.Sequential(torch.nn.Linear(256, 256, bias = False, dtype = dtype))
    if quantized:
        model[0].weight.quant_state = object()  # what bitsandbytes Params4bit carries
    return model


@pytest.fixture
def env_on(monkeypatch):
    monkeypatch.setenv("UNSLOTH_LORA_BASE_INT8", "1")
    monkeypatch.setattr(f8t, "int8_lora_unavailable", lambda gpu = "current": None)
    monkeypatch.setattr(f8t, "_parallel_layout_requested", lambda: False)
    monkeypatch.setattr(P, "current_gpu_class", lambda: "sm120")


@pytest.mark.parametrize(
    "value, on", [("1", True), ("true", True), ("on", True), ("0", False), ("", False)]
)
def test_env_request(monkeypatch, env_on, value, on):
    monkeypatch.setenv("UNSLOTH_LORA_BASE_INT8", value)
    assert f8t.int8_lora_requested() is on
    assert f8t.int8_lora_from_env(_linear_model()) == ("int8_lora_rotated" if on else None)


def test_env_unset_changes_nothing(monkeypatch):
    monkeypatch.delenv("UNSLOTH_LORA_BASE_INT8", raising = False)
    assert f8t.int8_lora_from_env(_linear_model()) is None


@pytest.mark.parametrize(
    "kwargs, model, said",
    [
        (dict(full_finetuning = True), _linear_model(), "full finetuning"),
        (dict(qat_scheme = "int8-int4"), _linear_model(), "qat_scheme"),
        ({}, _linear_model(quantized = True), "quantized"),
        ({}, _linear_model(torch.float32), "float32"),
    ],
)
def test_env_opt_in_is_ignored_where_it_cannot_run(env_on, capsys, kwargs, model, said):
    """Decided on the LOADED model: a quantized / fp32 base, full finetuning or QAT keeps the load as is."""
    assert f8t.int8_lora_from_env(model, **kwargs) is None
    assert said in capsys.readouterr().out


def test_env_opt_in_on_a_16bit_load_whatever_the_flags(env_on):
    # A '-bf16' checkpoint loads 16 bit despite load_in_4bit's default: the loaded weights decide, not the flags.
    assert f8t.int8_lora_from_env(_linear_model(torch.bfloat16)) == "int8_lora_rotated"
    assert f8t.int8_lora_from_env(_linear_model(torch.float16)) == "int8_lora_rotated"


def test_env_opt_in_under_tp_or_ep_is_ignored(env_on, monkeypatch, capsys):
    monkeypatch.setattr(f8t, "_parallel_layout_requested", lambda: True)
    assert f8t.int8_lora_from_env(_linear_model()) is None
    assert "parallel" in capsys.readouterr().out


@pytest.mark.parametrize("gpu", ["sm75", "rocm", None])
def test_env_opt_in_on_unsupported_gpu_stays_16bit(monkeypatch, capsys, gpu):
    monkeypatch.setenv("UNSLOTH_LORA_BASE_INT8", "1")
    monkeypatch.setattr(f8t, "_parallel_layout_requested", lambda: False)
    monkeypatch.setattr(P, "current_gpu_class", lambda: gpu)
    assert f8t.int8_lora_unavailable() is not None
    assert f8t.int8_lora_from_env(_linear_model()) is None
    assert "ignored here" in capsys.readouterr().out


def test_explicit_request_still_refuses_fp32(monkeypatch):
    import unsloth.kernels.int8_lora as IL
    monkeypatch.setattr(IL, "int8_lora_supported", lambda device: None)
    with pytest.raises(P.PrecisionRefused, match = "fp32"):
        f8t.resolve_fp8_lora("int8_lora", dtype = torch.float32, gpu = "sm120")


@pytest.mark.parametrize(
    "kw",
    [
        dict(load_in_4bit = True),
        dict(load_in_8bit = True),
        dict(load_in_fp8 = True),
        dict(qat_scheme = "int4"),
    ],
)
def test_refuses_quantized_or_qat_bases(kw):
    args = dict(
        dtype = torch.bfloat16,
        load_in_4bit = False,
        load_in_8bit = False,
        load_in_fp8 = False,
        qat_scheme = None,
        gpu = "sm120",
    )
    args.update(kw)
    with pytest.raises(P.PrecisionRefused):
        f8t.resolve_fp8_lora("int8_lora", **args)


@pytest.mark.parametrize("gpu", ["sm75", "rocm"])
def test_refuses_where_measured_slower(monkeypatch, gpu):
    import unsloth.kernels.int8_lora as IL
    monkeypatch.setattr(IL, "int8_lora_supported", lambda device: None)
    with pytest.raises(P.PrecisionRefused, match = "refused"):
        f8t.resolve_fp8_lora("int8_lora", dtype = torch.bfloat16, gpu = gpu)


@pytest.mark.parametrize("gpu", ["sm80", "sm89", "sm120"])
def test_resolves_where_measured_faster(monkeypatch, gpu):
    import unsloth.kernels.int8_lora as IL

    monkeypatch.setattr(IL, "int8_lora_supported", lambda device: None)
    assert f8t.resolve_fp8_lora("int8_lora", dtype = torch.bfloat16, gpu = gpu) == "int8_lora_rotated"
    assert f8t.is_int8_lora_recipe("int8_lora_rotated")
    assert not f8t.is_int8_lora_recipe("fp8_lora_rowwise")


def test_refuses_without_triton(monkeypatch):
    import unsloth.kernels.int8_lora as IL
    monkeypatch.setattr(IL, "int8_lora_supported", lambda device: "Triton is not installed")
    with pytest.raises(P.PrecisionRefused, match = "Triton"):
        f8t.resolve_fp8_lora("int8_lora", dtype = torch.bfloat16, gpu = "sm120")


def test_eligibility_needs_rotation_block():
    import unsloth.kernels.int8_lora as IL

    ok = torch.nn.Linear(256, 64, bias = False, dtype = torch.bfloat16).requires_grad_(False)
    assert IL.eligible_base(ok) is None
    odd = torch.nn.Linear(272, 64, bias = False, dtype = torch.bfloat16).requires_grad_(False)
    assert "multiple of 128" in IL.eligible_base(odd)
    trainable = torch.nn.Linear(256, 64, bias = False, dtype = torch.bfloat16)
    assert IL.eligible_base(trainable) is not None


def test_weight_quantization_round_trip_cpu():
    import unsloth.kernels.int8_lora as IL

    w = torch.randn(64, 256) * 0.02
    wq, ws = IL.quantize_weight(w)
    assert wq.dtype == torch.int8 and ws.shape == (64,)
    back = IL.dequantize_weight(wq, ws, torch.float32)
    assert float((back - w).norm() / w.norm()) < 0.01


def _tiny_lora_model(hidden):
    from peft import LoraConfig, get_peft_model

    class Cfg:
        hidden_size = hidden

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()
            self.q_proj = torch.nn.Linear(hidden, hidden, bias = False, dtype = torch.bfloat16)

        def forward(self, x):
            return self.q_proj(x)

    base = Tiny().requires_grad_(False)
    return get_peft_model(base, LoraConfig(r = 4, target_modules = ["q_proj"]))


@pytest.mark.parametrize("hidden, converts", [(1024, False), (2048, True)])
def test_hidden_size_gate(monkeypatch, hidden, converts):
    model = _tiny_lora_model(hidden)
    monkeypatch.setattr(f8t, "_prepare_fp8_lora_loss", lambda m: None)
    monkeypatch.setattr(f8t, "_prepare_fp8_lora_operators", lambda m: None)
    converted = f8t.enable_fp8_lora(model, "int8_lora_rotated", allow_cpu = True)
    assert bool(converted) is converts
    states = [
        m.__dict__.get("_unsloth_fp8_lora")
        for m in model.modules()
        if "_unsloth_fp8_lora" in m.__dict__
    ]
    assert all(type(s).__name__ == "Int8LoRABaseState" for s in states) and bool(states) is converts


def test_fp16_activations_need_loss_scaling():
    assert P.grad_scaler_kind("int8_lora_rotated", torch.float16) == "GradScaler"
    assert P.grad_scaler_kind("int8_lora_rotated", torch.bfloat16) == "none"
