# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""fp8 LoRA training seeded from a hosted unsloth/*-FP8 denoiser checkpoint (CPU only)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from core.training import diffusion_dit_trainer as dit
from core.training import diffusion_fp8_prequant as pq
from core.training.diffusion_frozen_fp8 import FrozenFp8Linear, _quantize_weight

_BASE = "Tongyi-MAI/Z-Image-Turbo"


def _meta(
    base = _BASE,
    scheme = "fp8",
    granularity = "per_row",
    fmt = pq._FORMAT,
):
    info = {"scheme": scheme, "base_model_id": base, "fp8_granularity": granularity}
    return {"unsloth_format": fmt, "unsloth_metadata": json.dumps(info)}


def _header(**kw):
    return {
        "__metadata__": _meta(**kw),
        "blk.lin._weight_qdata": {"dtype": "F8_E4M3", "shape": [32, 64]},
        "blk.lin._weight_scale": {"dtype": "F32", "shape": [32, 1]},
        "blk.norm.weight": {"dtype": "BF16", "shape": [64]},
    }


def test_checkpoint_problem_accepts_the_hosted_per_row_layout():
    assert pq.checkpoint_problem(_header(), _BASE) is None
    # a mirror / local spelling of the same base is the same base
    assert pq.checkpoint_problem(_header(), "/models/Z-Image-Turbo") is None


@pytest.mark.parametrize(
    "kw, base, why",
    [
        ({}, "Tongyi-MAI/Z-Image", "baked from"),
        (
            {"base": "black-forest-labs/FLUX.2-klein-4B"},
            "black-forest-labs/FLUX.2-klein-base-4B",
            "baked from",
        ),
        ({"scheme": "int8"}, _BASE, "scheme"),
        ({"granularity": "per_tensor"}, _BASE, "granularity"),
        ({"fmt": "comfy"}, _BASE, "format"),
    ],
)
def test_checkpoint_problem_refuses_other_bases_and_layouts(kw, base, why):
    assert why in pq.checkpoint_problem(_header(**kw), base)


@pytest.mark.parametrize(
    "scale",
    [
        {"dtype": "F32", "shape": [1]},  # per-tensor
        {"dtype": "F32", "shape": [32, 2]},  # block
        {"dtype": "BF16", "shape": [32, 1]},
    ],
)
def test_checkpoint_problem_refuses_non_row_scales(scale):
    h = _header()
    h["blk.lin._weight_scale"] = scale
    assert "scale" in pq.checkpoint_problem(h, _BASE)


class _Blk(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(64, 32, bias = True)
        self.norm = nn.LayerNorm(64)


class _Tiny(nn.Module):
    _keep_in_fp32_modules = ["norm"]

    def __init__(self):
        super().__init__()
        self.blk = _Blk()
        self.out = nn.Linear(32, 16, bias = False)
        self.pad = nn.Parameter(
            torch.randn(1, 64)
        )  # a root tensor, stored under unsloth_root:: (z-image)

    @classmethod
    def load_config(
        cls,
        repo,
        subfolder = None,
        token = None,
    ):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls()


def _write_ckpt(path, ref):
    from safetensors.torch import save_file

    q, s = _quantize_weight(ref.blk.lin.weight.detach())
    tensors = {
        "blk.lin._weight_qdata": q,
        "blk.lin._weight_scale": s,
        "blk.lin.bias": ref.blk.lin.bias.detach().bfloat16(),
        "blk.norm.weight": ref.blk.norm.weight.detach().bfloat16(),
        "blk.norm.bias": ref.blk.norm.bias.detach().bfloat16(),
        "out.weight": ref.out.weight.detach().bfloat16(),
        "unsloth_root::pad": ref.pad.detach().bfloat16(),
    }
    save_file(tensors, str(path), metadata = _meta())
    return q, s


def test_load_keeps_the_stored_codes_and_dequantizes_the_rest(tmp_path):
    torch.manual_seed(0)
    ref = _Tiny()
    path = tmp_path / "ck.safetensors"
    q, s = _write_ckpt(path, ref)
    assert pq.checkpoint_problem(pq._local_header(str(path)), _BASE) is None

    m = pq.load_fp8_transformer(_Tiny, _BASE, str(path), "cpu")
    lin = m.blk.lin
    # zero-stride placeholder: no bf16 copy of the fp8 weight is ever materialised
    assert lin.weight.shape == (32, 64) and lin.weight.stride() == (0, 0)
    assert lin.weight.dtype == torch.bfloat16 and not lin.weight.requires_grad
    assert m.blk.norm.weight.dtype == torch.float32  # _keep_in_fp32_modules, as diffusers loads it
    assert m.out.weight.dtype == torch.bfloat16
    assert torch.equal(m.pad, ref.pad.detach().bfloat16())

    fp8 = FrozenFp8Linear(lin)
    assert torch.equal(fp8.weight.view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(fp8.weight_scale, s.reshape(1, -1))
    assert torch.equal(fp8.bias, ref.blk.lin.bias.detach().bfloat16())

    assert pq.dequantize_placeholders(m) == 1
    assert lin.weight.stride() == (64, 1)
    assert torch.equal(lin.weight, (q.float() * s).bfloat16())
    assert pq.dequantize_placeholders(m) == 0  # idempotent


def test_dense_requantize_of_the_stored_weight_is_the_stored_codes(tmp_path):
    """The prequant layer and today's quantize-at-load agree: re-quantizing q * scale returns q and scale."""
    torch.manual_seed(1)
    w = torch.randn(48, 64) * torch.logspace(-3, 1, 48).unsqueeze(1)
    q, s = _quantize_weight(w)
    q2, s2 = _quantize_weight(q.float() * s)
    assert torch.equal(q2.view(torch.uint8), q.view(torch.uint8))
    assert torch.allclose(s2, s, rtol = 1e-6, atol = 0)


def test_load_refuses_a_checkpoint_for_another_architecture(tmp_path):
    from safetensors.torch import save_file
    save_file(
        {"other.weight": torch.zeros(2, 2)}, str(tmp_path / "x.safetensors"), metadata = _meta()
    )
    with pytest.raises(ValueError, match = "keys differ"):
        pq.load_fp8_transformer(_Tiny, _BASE, str(tmp_path / "x.safetensors"), "cpu")


def test_resolve_needs_a_mapping_and_honours_the_opt_out(monkeypatch):
    assert (
        pq.resolve_fp8_checkpoint("z-image", "Tongyi-MAI/Z-Image") is None
    )  # no hosted fp8 for the base
    assert pq.resolve_fp8_checkpoint("ltx-2", "Lightricks/LTX-2") is None  # not an image family
    assert pq.resolve_fp8_checkpoint(None, _BASE) is None
    monkeypatch.setenv(pq._DISABLE_ENV, "0")
    assert pq.resolve_fp8_checkpoint("z-image", _BASE) is None


def test_resolve_refuses_another_bases_checkpoint_before_downloading(monkeypatch):
    import core.inference.diffusion_prequant as dp

    seen = []
    monkeypatch.setattr(dp, "cached_checkpoint_path", lambda *a, **k: None)
    monkeypatch.setattr(
        pq,
        "_remote_header",
        lambda repo, name, token: seen.append(name) or _header(base = "x/FLUX.2-klein-4B"),
    )
    monkeypatch.setattr(dp, "_download_checkpoint_name", lambda *a, **k: pytest.fail("downloaded"))
    assert (
        pq.resolve_fp8_checkpoint("flux.2-klein", "black-forest-labs/FLUX.2-klein-base-4B") is None
    )
    assert seen == [
        "FLUX.2-klein-4B-FP8.safetensors"
    ]  # Studio's own safetensors only, no comfy / pickle


def test_resolve_downloads_the_inference_file_into_the_live_cache(monkeypatch, tmp_path):
    import core.inference.diffusion_prequant as dp

    path = tmp_path / "Z-Image-Turbo-FP8.safetensors"
    _write_ckpt(path, _Tiny())
    calls = []
    monkeypatch.setattr(dp, "cached_checkpoint_path", lambda *a, **k: None)
    monkeypatch.setattr(pq, "_remote_header", lambda repo, name, token: _header())
    monkeypatch.setattr(pq, "_hub_cache", lambda: "/live/cache")

    def _dl(src, name, token, cache_dir, **k):
        calls.append((src.location, name, cache_dir))
        return str(path)

    monkeypatch.setattr(dp, "_download_checkpoint_name", _dl)
    assert pq.resolve_fp8_checkpoint("z-image", _BASE) == str(path)
    assert calls == [("unsloth/Z-Image-Turbo-FP8", "Z-Image-Turbo-FP8.safetensors", "/live/cache")]


def test_resolve_never_raises(monkeypatch):
    import core.inference.diffusion_prequant as dp
    monkeypatch.setattr(dp, "resolve_prequant_source", lambda *a, **k: 1 / 0)
    assert pq.resolve_fp8_checkpoint("z-image", _BASE) is None


class _Cls:
    calls = []

    @classmethod
    def from_pretrained(cls, *a, **k):
        cls.calls.append(k.get("subfolder"))
        return nn.Linear(2, 2)


def _cfg():
    return SimpleNamespace(base_model = _BASE, hf_token = None, resolved_family = "z-image")


def test_fp8_load_falls_back_to_bf16_without_a_checkpoint(monkeypatch):
    _Cls.calls = []
    monkeypatch.setattr(pq, "resolve_fp8_checkpoint", lambda *a, **k: None)
    dit._load_dit_transformer(_Cls, _cfg(), "cpu", "fp8")
    assert _Cls.calls == ["transformer"]


def test_fp8_load_falls_back_to_bf16_when_the_checkpoint_fails(monkeypatch):
    _Cls.calls = []
    monkeypatch.setattr(pq, "resolve_fp8_checkpoint", lambda *a, **k: "/x.safetensors")
    monkeypatch.setattr(pq, "load_fp8_transformer", lambda *a, **k: 1 / 0)
    dit._load_dit_transformer(_Cls, _cfg(), "cpu", "fp8")
    assert _Cls.calls == ["transformer"]


@pytest.mark.parametrize("mode", ["bf16", "int8", "mxfp8"])
def test_other_modes_never_look_for_the_fp8_checkpoint(monkeypatch, mode):
    _Cls.calls = []
    monkeypatch.setattr(pq, "resolve_fp8_checkpoint", lambda *a, **k: pytest.fail("looked"))
    dit._load_dit_transformer(_Cls, _cfg(), "cpu", mode)
    assert _Cls.calls == ["transformer"]


def test_fp8_load_uses_the_checkpoint(monkeypatch):
    _Cls.calls = []
    sentinel = nn.Linear(2, 2)
    monkeypatch.setattr(pq, "resolve_fp8_checkpoint", lambda *a, **k: "/x.safetensors")
    monkeypatch.setattr(pq, "load_fp8_transformer", lambda *a, **k: sentinel)
    assert dit._load_dit_transformer(_Cls, _cfg(), "cpu", "fp8") is sentinel
    assert _Cls.calls == []


def test_failed_fp8_conversion_dequantizes_the_prequant_base(monkeypatch, tmp_path):
    path = tmp_path / "ck.safetensors"
    q, s = _write_ckpt(path, _Tiny())
    m = pq.load_fp8_transformer(_Tiny, _BASE, str(path), "cpu")
    monkeypatch.setattr(dit, "_convert_fp8", lambda *a, **k: 1 / 0)
    events = []
    assert dit._apply_fp8_training(m, events.append) is False
    assert [e["type"] for e in events] == ["warning"]
    assert torch.equal(m.blk.lin.weight, (q.float() * s).bfloat16())
    assert not hasattr(m.blk.lin, pq.PREQUANT_ATTR)
