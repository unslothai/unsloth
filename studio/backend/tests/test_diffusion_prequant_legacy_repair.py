# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hosted pre-quant checkpoints built under an older contract load with the runtime's contract.

Real torchao (whatever this venv ships: 0.17 builds the v1 int8 wrapper, 0.18+ ``Int8Tensor``), CPU only:
- an fp8 weight built before ``activation_value_lb`` gets exactly the floor ``_make_quant_config`` sets, and with it
  an all-zero activation row gets a finite scale;
- an int8 weight the runtime would leave dense comes back as a plain dense tensor;
- Krea 2 pipeline picks seed their hosted denoiser like every other family.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")

from core.inference import diffusion_prequant as pq  # noqa: E402


def _float8_kwargs_cls():
    try:
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
            QuantizeTensorToFloat8Kwargs,
        )
    except Exception:  # noqa: BLE001
        pytest.skip("this torchao has no QuantizeTensorToFloat8Kwargs")
    return QuantizeTensorToFloat8Kwargs


def _legacy_fp8_weight():
    """A real Float8Tensor exactly as an artifact built before ``activation_value_lb`` holds it."""
    from torchao.quantization import PerRow
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor

    kwargs = _float8_kwargs_cls()(torch.float8_e4m3fn, PerRow())
    assert kwargs.hp_value_lb is None
    weight = torch.randn(64, 32, dtype = torch.bfloat16)
    return Float8Tensor.from_hp(weight, granularity = PerRow(), act_quant_kwargs = kwargs)


def test_restored_floor_is_the_runtime_config_floor():
    from core.inference.diffusion_transformer_quant import FP8_ACTIVATION_VALUE_LB, TQ_FP8, _make_quant_config

    config = _make_quant_config(TQ_FP8)
    if getattr(config, "activation_value_lb", None) is None:
        pytest.skip("this torchao has no activation_value_lb")
    assert config.activation_value_lb == FP8_ACTIVATION_VALUE_LB

    tensor = _legacy_fp8_weight()
    before = tensor.act_quant_kwargs
    qdata, scale = tensor.qdata.clone(), tensor.scale.clone()
    state_dict = {"blocks.0.attn.to_q.weight": tensor}
    assert pq._fp8_activation_floor_present(state_dict, None) is False
    assert pq._fp8_activation_floor_restorable(state_dict) is True
    assert pq._restore_fp8_activation_floor(state_dict) == 1
    after = tensor.act_quant_kwargs
    # Only the floor moves: every other activation kwarg and the weight bytes are untouched.
    assert after == dataclasses.replace(before, hp_value_lb = config.activation_value_lb)
    assert torch.equal(tensor.qdata.view(torch.uint8), qdata.view(torch.uint8))
    assert torch.equal(tensor.scale, scale)
    assert pq._fp8_activation_floor_present(state_dict, None) is True
    assert pq._restore_fp8_activation_floor(state_dict) == 0


def test_restored_floor_gives_a_zero_row_a_finite_scale():
    from torchao.quantization.quant_primitives import _choose_scale_float8

    tensor = _legacy_fp8_weight()
    zero_rows = torch.zeros(4, 32, dtype = torch.bfloat16)
    unfloored = _choose_scale_float8(
        zero_rows, [1, 32], torch.float8_e4m3fn, hp_value_lb = tensor.act_quant_kwargs.hp_value_lb
    )
    assert not bool((unfloored > 0).all())  # the black-frame failure: scale 0 for an all-zero row
    pq._restore_fp8_activation_floor({"w": tensor})
    floored = _choose_scale_float8(
        zero_rows, [1, 32], torch.float8_e4m3fn, hp_value_lb = tensor.act_quant_kwargs.hp_value_lb
    )
    assert bool((floored > 0).all()) and bool(torch.isfinite(floored).all())
    assert bool(torch.isfinite(zero_rows.float() / floored.reshape(-1, 1)).all())


def _real_int8_weight():
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    linear = torch.nn.Linear(64, 64, bias = False, dtype = torch.bfloat16)
    reference = linear.weight.detach().clone()
    quantize_(linear, Int8DynamicActivationInt8WeightConfig())
    return linear.weight, reference


def test_an_excluded_int8_weight_comes_back_dense():
    weight, reference = _real_int8_weight()
    assert pq._is_quantized_weight(weight)
    keep, _ = _real_int8_weight()
    bias = torch.zeros(64, dtype = torch.bfloat16)
    state_dict = {
        "transformer_blocks.0.txt_mlp.net.2.weight": weight,
        "transformer_blocks.0.img_mlp.net.2.weight": keep,
        "transformer_blocks.0.txt_mlp.net.2.bias": bias,
    }
    assert pq._densify_excluded_weights(state_dict, ("txt_mlp",)) == 1
    dense = state_dict["transformer_blocks.0.txt_mlp.net.2.weight"]
    assert type(dense) is torch.Tensor and dense.dtype == torch.bfloat16
    assert dense.shape == reference.shape and dense.is_contiguous()
    # per-row symmetric int8: within half a quantization step of the bf16 original
    step = reference.float().abs().amax(dim = 1, keepdim = True) / 127.0
    assert bool(((dense.float() - reference.float()).abs() <= step * 0.51 + 1e-3).all())
    assert state_dict["transformer_blocks.0.img_mlp.net.2.weight"] is keep
    assert state_dict["transformer_blocks.0.txt_mlp.net.2.bias"] is bias


def test_krea2_pipeline_picks_seed_their_hosted_denoiser():
    from core.inference.diffusion_denoiser_prequant import pipeline_seed_supported
    from core.inference.diffusion_families import IDEOGRAM4_FAMILY_NAME, detect_family

    krea = detect_family("krea/Krea-2-Turbo")
    assert krea is not None and dict(krea.prequant_repos)  # hosted int8 / fp8 artifacts exist
    assert pipeline_seed_supported(krea) is True
    ideogram = type("Fam", (), {"name": IDEOGRAM4_FAMILY_NAME})()
    assert pipeline_seed_supported(ideogram) is False


def _krea_pipeline_calls() -> list:
    source = (_BACKEND / "core" / "inference" / "diffusion.py").read_text()
    calls = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "load_krea2_pipeline":
            calls.append({kw.arg: ast.unparse(kw.value) for kw in node.keywords})
    return calls


def test_the_krea2_pipeline_assembly_takes_the_seeded_denoiser():
    # The pipeline-pick assembly must hand over the denoiser the seed put in pipe_kwargs (None when unseeded, which
    # makes load_krea2_pipeline read the dense shards as before), next to the pre-cast text encoder.
    calls = _krea_pipeline_calls()
    seeded = [c for c in calls if c.get("transformer") == "pipe_kwargs.get('transformer')"]
    assert len(seeded) == 1
    assert seeded[0].get("text_encoder") == "pipe_kwargs.get('text_encoder')"
    assert seeded[0].get("local_files_only") == "local_files_only"
