# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""float16 CUDA cards (T4) take Qwen-Image-2.1's hosted int8 ConvRot text encoder instead of the dense bf16 one."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_vl")
pytest.importorskip("accelerate")

import core.inference.diffusion_te_prequant as tpq  # noqa: E402
from core.inference.video_minimax_h3_te import _int8_convrot_linear_class  # noqa: E402

from .test_diffusion_te_int8_convrot import (  # noqa: E402, F401
    BASE,
    INT8_NAME,
    REPO,
    get_family,
    int8_file,
)

CUDA_FP16 = types.SimpleNamespace(device = "cuda", dtype = torch.float16)
CUDA_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
KILL_SWITCH = "UNSLOTH_TE_INT8_FP16"


@pytest.fixture(autouse = True)
def _default_env(monkeypatch):
    monkeypatch.delenv(KILL_SWITCH, raising = False)


def test_fp16_card_resolves_only_the_int8_file():
    fam = get_family("qwen-image-2.1")
    src = tpq.te_prequant_sources_for_base(fam, BASE, te_quant_mode = "int8", target = CUDA_FP16)[
        "text_encoder"
    ]
    assert (src.kind, src.location, src.filename) == ("repo", REPO, INT8_NAME)
    assert tpq.te_candidate_filenames(src) == (INT8_NAME,)


def test_fp16_kill_switch_and_other_cases_unchanged(monkeypatch):
    fam = get_family("qwen-image-2.1")
    assert tpq.te_prequant_sources(fam, te_quant_mode = "fp8", target = CUDA_FP16) == {}
    assert tpq.te_prequant_sources(get_family("sdxl"), te_quant_mode = "int8", target = CUDA_FP16) == {}
    bf16 = tpq.te_prequant_sources(fam, te_quant_mode = "int8", target = CUDA_BF16)["text_encoder"]
    assert len(tpq.te_candidate_filenames(bf16)) > 1
    monkeypatch.setenv(KILL_SWITCH, "0")
    assert tpq.te_prequant_sources(fam, te_quant_mode = "int8", target = CUDA_FP16) == {}


def test_fp16_scales_the_weight_before_the_matmul():
    from core.inference.diffusion_convrot import build_convrot_hadamard

    lin = _int8_convrot_linear_class()(256, 4, bias = False, group_size = 256)
    lin.weight = torch.full((4, 256), 127, dtype = torch.int8)
    lin.weight_scale = torch.full((4, 1), 1e-4, dtype = torch.float32)
    # Unscaled sum 640 * 127 = 81280 overflows float16; true output is 8.13.
    x = torch.full((1, 256), 40.0, dtype = torch.float16)
    h = build_convrot_hadamard(256, dtype = torch.float32)
    expected = torch.nn.functional.linear(x.float(), (lin.weight.float() * lin.weight_scale) @ h)
    assert not torch.isfinite(lin(x)).all()
    lin.scale_weight_first = True
    out = lin(x)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), expected, rtol = 2e-3, atol = 2e-3)


def test_fp16_load_flags_every_convrot_linear(int8_file):
    _, _, path = int8_file
    cls = _int8_convrot_linear_class()

    def load(dtype):
        enc = tpq.load_prequant_text_encoder(
            BASE,
            "text_encoder",
            tpq.TePrequantSource(kind = "path", location = str(path)),
            dtype = dtype,
            scheme = "int8",
            local_files_only = True,
            trim_lm_head = True,
        )
        return [getattr(m, "scale_weight_first", None) for m in enc.modules() if isinstance(m, cls)]

    assert set(load(torch.float16)) == {True}
    assert set(load(torch.bfloat16)) == {False}
