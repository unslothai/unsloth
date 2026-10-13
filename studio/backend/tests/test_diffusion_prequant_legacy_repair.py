# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Legacy hosted pre-quant checkpoints load with the runtime's contract (real torchao, CPU)."""

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
    from core.inference.diffusion_transformer_quant import (
        FP8_ACTIVATION_VALUE_LB,
        TQ_FP8,
        _make_quant_config,
    )

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


def test_krea2_pipeline_picks_seed_their_hosted_denoiser():
    from core.inference.diffusion_denoiser_prequant import pipeline_seed_supported
    from core.inference.diffusion_families import IDEOGRAM4_FAMILY_NAME, detect_family

    krea = detect_family("krea/Krea-2-Turbo")
    assert krea is not None and dict(krea.prequant_repos)  # hosted int8 / fp8 artifacts exist
    assert pipeline_seed_supported(krea) is True
    from core.inference.diffusion_families import family_prequant_repo

    for raw in ("krea/Krea-2-Raw", "unsloth/Krea-2-Raw"):
        assert family_prequant_repo(krea, "fp8", base_repo = raw) is None
        assert family_prequant_repo(krea, "int8", base_repo = raw) is None
    assert (
        family_prequant_repo(krea, "fp8", base_repo = "krea/Krea-2-Turbo")
        == "unsloth/Krea-2-Turbo-FP8"
    )
    ideogram = type("Fam", (), {"name": IDEOGRAM4_FAMILY_NAME})()
    assert pipeline_seed_supported(ideogram) is False


def _krea_pipeline_calls() -> list:
    source = (_BACKEND / "core" / "inference" / "diffusion.py").read_text(encoding = "utf-8")
    calls = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "load_krea2_pipeline":
            calls.append({kw.arg: ast.unparse(kw.value) for kw in node.keywords})
    return calls


def test_the_krea2_pipeline_assembly_takes_the_seeded_denoiser():
    calls = _krea_pipeline_calls()
    seeded = [c for c in calls if c.get("transformer") == "pipe_kwargs.get('transformer')"]
    assert len(seeded) == 1
    assert seeded[0].get("text_encoder") == "pipe_kwargs.get('text_encoder')"
    assert seeded[0].get("local_files_only") == "local_files_only"
