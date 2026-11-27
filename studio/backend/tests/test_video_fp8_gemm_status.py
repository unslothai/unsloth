# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The video status reports which fp8 GEMM an fp8 load runs (torchao row-wise, or the unit-scale fallback on torch
builds without the row-wise kernel, the Windows wheels), as the images status does. CPU only."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from core.inference import diffusion_fp8_unit_scale as us  # noqa: E402
from core.inference import diffusion_modelopt_fp8 as mo  # noqa: E402
from core.inference import video  # noqa: E402
from models.inference import VideoStatusResponse  # noqa: E402


class _Mod:
    def __init__(
        self,
        index,
        info = None,
    ):
        self.p = SimpleNamespace(device = SimpleNamespace(type = "cuda", index = index))
        if info is not None:
            setattr(self, mo.MODELOPT_ATTR, info)

    def parameters(self):
        return iter([self.p])

    def modules(self):
        return iter([self])


@pytest.fixture
def fallback_card(monkeypatch):
    """Card 1 runs fp8 on the unit-scale fallback (no row-wise kernel); card 0 is row-wise."""
    monkeypatch.setitem(us._STATE, "installed", True)
    monkeypatch.setitem(us._STATE, "fallback", frozenset({1}))
    monkeypatch.setitem(us._STATE, "fp32_acc", frozenset({1}))
    monkeypatch.setattr(us, "_ROWWISE", {0: True, 1: False})


def test_an_fp8_denoiser_reports_its_cards_gemm(fallback_card):
    pipe = SimpleNamespace(transformer = _Mod(1), text_encoder = _Mod(0))
    state = SimpleNamespace(transformer_quant = "fp8", text_encoder_quant = None, pipe = pipe)
    status = video._video_fp8_gemm_status(state)
    assert status["path"] == "unit_scale_fallback" and status["device_index"] == 1
    assert status["component"] == "transformer" and status["rowwise_kernel"] is False
    assert "modelopt" not in status


def test_an_fp8_text_encoder_reports_its_own_card(fallback_card):
    pipe = SimpleNamespace(transformer = _Mod(1), text_encoder = _Mod(0))
    # fp8_dynamic is the video encoder's compute-fp8 scheme
    state = SimpleNamespace(transformer_quant = "int8", text_encoder_quant = "fp8_dynamic", pipe = pipe)
    status = video._video_fp8_gemm_status(state)
    assert status["component"] == "text_encoder" and status["path"] == "torchao_rowwise"


def test_a_modelopt_denoiser_reports_its_w8a8_route(fallback_card, monkeypatch):
    monkeypatch.setattr(
        mo,
        "_GEMM_PROBE",
        {
            1: {
                "static": True,
                "dynamic": False,
                "dynamic_unit_scale": True,
                "unit_scale_fp32": True,
            }
        },
    )
    transformer = _Mod(1, {"runtime": "dynamic"})
    transformer.gemm_calls, transformer.unit_scale_calls, transformer.a16_calls = 7, 7, 2
    transformer.input_scale = transformer.weight_q = object()
    pipe = SimpleNamespace(transformer = transformer)
    # a hosted ModelOpt checkpoint does not set transformer_quant
    state = SimpleNamespace(transformer_quant = None, text_encoder_quant = None, pipe = pipe)
    status = video._video_fp8_gemm_status(state)
    assert status["component"] == "transformer"
    assert status["modelopt"]["runtime"] == "dynamic"
    assert status["modelopt"]["route"]["dynamic"] == "unit_scale_fallback"
    assert status["modelopt"]["calls"] == {"gemm": 7, "unit_scale": 7, "a16": 2}


def test_a_weight_only_modelopt_denoiser_runs_no_fp8_gemm(fallback_card):
    for runtime in ("stored", "dequant"):
        pipe = SimpleNamespace(transformer = _Mod(1, {"runtime": runtime}))
        state = SimpleNamespace(transformer_quant = None, text_encoder_quant = None, pipe = pipe)
        assert video._video_fp8_gemm_status(state) is None, runtime


def test_no_fp8_no_field_and_status_never_raises():
    state = SimpleNamespace(transformer_quant = "int8", text_encoder_quant = None, pipe = None)
    assert video._video_fp8_gemm_status(state) is None
    assert video._video_fp8_gemm_status(object()) is None


def test_the_status_dict_and_response_carry_the_field():
    payload = {"path": "unit_scale_fallback", "device_index": 0}
    assert VideoStatusResponse(loaded = True, fp8_gemm = payload).model_dump()["fp8_gemm"] == payload
    # the loaded-status builder emits the key (the dict literal is too entangled with a live load to call here)
    tree = ast.parse(Path(video.__file__).read_text(encoding = "utf-8"))
    status_dicts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and {"has_audio", "supports_cfg", "multi_gpu"}
        <= {k.value for k in node.keys if isinstance(k, ast.Constant)}
    ]
    assert status_dicts
    for node in status_dicts:
        values = {k.value: v for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant)}
        call = values.get("fp8_gemm")
        assert (
            isinstance(call, ast.Call)
            and getattr(call.func, "id", None) == "_video_fp8_gemm_status"
        )
