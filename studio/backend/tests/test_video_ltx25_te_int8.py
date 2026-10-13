# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5's Gemma 4 encoder runs int8 ConvRot by default, the precision ComfyUI's LTX-2.5 workflows load."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from core.inference import video_ltx25 as v25
from core.inference import video_ltx25_te_worker as worker
from core.inference.diffusion_convrot import build_convrot_hadamard, quantize_convrot_int8_weight


def _comfy_kitchen_reference(weight: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """comfy-kitchen's eager ``quantize_int8_convrot_weight``, in float32: kron powers of its H4, W @ H.T per group,
    per-row absmax / 127, round, clamp."""
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype = torch.float32)
    h = h4
    while h.shape[0] < group:
        h = torch.kron(h, h4)
    h = h / group**0.5
    out_f, in_f = weight.shape
    rotated = (weight.float().reshape(out_f, in_f // group, group) @ h.T).reshape(out_f, in_f)
    scale = (rotated.abs().amax(dim = -1, keepdim = True) / 127.0).clamp(min = 1e-30)
    return (rotated / scale).round().clamp(-128, 127).to(torch.int8), scale


def test_hadamard_is_comfy_kitchens_and_quantization_matches_it():
    torch.manual_seed(0)
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype = torch.float32)
    ref = torch.kron(torch.kron(torch.kron(h4, h4), h4), h4) / 16.0
    assert torch.equal(build_convrot_hadamard(256), ref)
    weight = (torch.randn(64, 512) * 0.02).to(torch.bfloat16)
    codes, scale = quantize_convrot_int8_weight(weight, 256)
    ref_codes, ref_scale = _comfy_kitchen_reference(weight, 256)
    assert codes.dtype == torch.int8 and scale.dtype == torch.float32 and scale.shape == (64, 1)
    assert torch.equal(codes, ref_codes)
    torch.testing.assert_close(scale, ref_scale, rtol = 1e-6, atol = 0)


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias = False)
        self.k_proj = nn.Linear(d, d // 2, bias = False)
        self.o_proj = nn.Linear(d, d, bias = False)
        self.q_norm = nn.LayerNorm(d)


class _Mlp(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate_proj = nn.Linear(d, 2 * d, bias = False)
        self.up_proj = nn.Linear(d, 2 * d, bias = False)
        self.down_proj = nn.Linear(2 * d, d, bias = False)


class _Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _Mlp(d)


class _TinyGemma(nn.Module):
    def __init__(self, d = 256, layers = 2):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(16, d)
        self.model.language_model.layers = nn.ModuleList(_Layer(d) for _ in range(layers))


def test_worker_swaps_every_decoder_projection_and_keeps_the_math():
    torch.manual_seed(0)
    model = _TinyGemma().to(torch.bfloat16)
    x = torch.randn(3, 256, dtype = torch.bfloat16)
    linear = model.model.language_model.layers[0].mlp.gate_proj
    want = linear(x).float()
    assert worker.quantize_int8_convrot_(model, "cpu") == 12  # 2 layers x (q, k, o, gate, up, down)
    swapped = model.model.language_model.layers[0].mlp.gate_proj
    assert type(swapped).__name__ == "Int8ConvRotLinear"
    assert swapped.weight.dtype == torch.int8 and swapped.weight_scale.dtype == torch.float32
    assert not any(isinstance(m, nn.Linear) for m in model.model.language_model.layers.modules())
    # Embeddings and norms are untouched.
    assert model.model.language_model.embed_tokens.weight.dtype == torch.bfloat16
    got = swapped(x).float()
    rel = (got - want).norm() / want.norm()
    assert rel < 0.02, rel
    # Buffers move with the module, so the worker's per-encode onload / offload carries the int8 weights.
    model.to("meta")
    assert swapped.weight.device.type == "meta" and swapped.weight.dtype == torch.int8


def test_worker_load_quantizes_only_when_asked(monkeypatch):
    import transformers

    built = []

    class _Cls:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            m = _TinyGemma()
            built.append(m)
            return m

    monkeypatch.setattr(transformers, "__version__", "5.10.2", raising = False)
    monkeypatch.setattr(transformers, "Gemma4UnifiedForConditionalGeneration", _Cls, raising = False)
    monkeypatch.setattr(
        transformers, "AutoTokenizer", SimpleNamespace(from_pretrained = lambda *a, **k: object())
    )
    base = {"repo": "Lightricks/LTX-2.5-Diffusers", "device": "cpu", "dtype": "bfloat16"}
    _tok, model = worker._load(dict(base, te_quant = "int8"))
    assert type(model.model.language_model.layers[1].self_attn.q_proj).__name__ == "Int8ConvRotLinear"
    _tok, model = worker._load(dict(base, te_quant = None))
    assert isinstance(model.model.language_model.layers[1].self_attn.q_proj, nn.Linear)
    with pytest.raises(RuntimeError, match = "unsupported text encoder precision"):
        worker._load(dict(base, te_quant = "fp8"))


def test_te_quant_tristate(monkeypatch):
    # int8 is opt-in: on the pack encoder its renders sat above the compile floor (LPIPS 0.310 vs 0.256).
    monkeypatch.delenv(v25.LTX25_TE_QUANT_ENV, raising = False)
    assert v25.ltx25_te_quant_scheme(None) is None
    assert v25.ltx25_te_quant_scheme("auto") is None
    assert v25.ltx25_te_quant_scheme("") is None
    assert v25.ltx25_te_quant_scheme("int8") == "int8"
    assert v25.ltx25_te_quant_scheme("none") is None
    assert v25.ltx25_te_quant_scheme("off") is None
    for scheme in ("fp8", "fp8_dynamic"):
        with pytest.raises(ValueError, match = "int8 .* or bf16"):
            v25.ltx25_te_quant_scheme(scheme)
    monkeypatch.setenv(v25.LTX25_TE_QUANT_ENV, "1")
    assert v25.ltx25_te_quant_scheme(None) == "int8"
    assert v25.ltx25_te_quant_scheme("off") is None


def test_budget_prices_the_int8_encoder(monkeypatch):
    from core.inference.diffusion_te_prequant import te_prequant_budget_scale
    from core.inference.video_families import detect_video_family

    monkeypatch.delenv(v25.LTX25_TE_QUANT_ENV, raising = False)
    fam = detect_video_family("Lightricks/LTX-2.5-Diffusers")
    bf16_gb = fam.bf16_components_gb[1]
    kw = {"target": None, "base": "Lightricks/LTX-2.5-Diffusers"}
    scale = te_prequant_budget_scale(fam, te_quant_mode = "int8", **kw)
    assert scale == pytest.approx(v25.LTX25_TE_INT8_GB / bf16_gb)
    assert scale < 0.6
    assert te_prequant_budget_scale(fam, te_quant_mode = "none", **kw) == 1.0
    assert te_prequant_budget_scale(fam, te_quant_mode = None, **kw) == 1.0
    monkeypatch.setenv(v25.LTX25_TE_QUANT_ENV, "1")
    assert te_prequant_budget_scale(fam, te_quant_mode = None, **kw) == scale


def test_client_carries_the_scheme_and_status_reports_it():
    kwargs = dict(
        repo = "Lightricks/LTX-2.5-Diffusers",
        hf_token = None,
        cache_dir = None,
        local_files_only = True,
        device = "cpu",
        dtype = "bfloat16",
        resident = False,
        venv_dir = "/nonexistent",
    )
    client = v25.Ltx25TextEncoderClient(te_quant = "int8", **kwargs)
    assert client.config["te_quant"] == "int8"
    outcome = v25.ltx25_te_outcome(client)
    assert outcome.mode == "int8" and "ConvRot" in outcome.reason
    assert v25.ltx25_te_outcome(v25.Ltx25TextEncoderClient(**kwargs)).mode is None


def test_validate_load_request_accepts_int8_and_off_and_refuses_other_schemes():
    from core.inference.video import get_video_backend

    backend = get_video_backend()
    for mode in (None, "auto", "int8", "none", "off"):
        assert (
            backend.validate_load_request(
                "Lightricks/LTX-2.5-Diffusers", model_kind = "pipeline", text_encoder_quant = mode
            ).name
            == "ltx-2.5"
        )
    with pytest.raises(ValueError, match = "int8 .* or bf16"):
        backend.validate_load_request(
            "Lightricks/LTX-2.5-Diffusers", model_kind = "pipeline", text_encoder_quant = "fp8"
        )
