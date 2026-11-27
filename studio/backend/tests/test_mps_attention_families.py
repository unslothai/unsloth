# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MPS SDPA vs a CPU reference at each image family's attention shape.

torch <2.13 MPSGraph SDPA is wrong past a size (pytorch#179352); 2.13+ uses a fused prefill kernel.
Large cells run on CI or with UNSLOTH_MPS_LARGE_ATTENTION=1 (0 disables).
"""

from __future__ import annotations

import math
import os
import warnings

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_attention as attention

# (heads, head_dim, text tokens) of each family's joint attention; patch 2 on an 8x VAE = 16 px per token.
FAMILIES = {
    "z-image": (30, 128, 128),
    "flux2-klein-4b": (24, 128, 512),
    "flux1": (24, 128, 512),
    "qwen-image": (24, 128, 256),
    "qwen-image-2.1": (32, 128, 256),
    "sd3.5-large": (38, 64, 333),
    "wan2.2-14b": (40, 128, 512),
    "ltx-video": (32, 64, 128),
}
LARGE = (
    os.environ.get("UNSLOTH_MPS_LARGE_ATTENTION")
    or ("1" if os.environ.get("GITHUB_ACTIONS") == "true" else "0")
) == "1"
# Elements Studio measured wrong on macOS 15 / torch 2.11 (diffusion_qwenimage21_math.MPS_SCORE_ELEMENTS doc).
STUDIO_WRONG_FROM = int(0.9 * 2**29)
# pytorch#179352: head h of a B=1 call is wrong once h * T * T reaches this.
UPSTREAM_WRONG_FROM = 2**32

mps = pytest.mark.skipif(
    not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()),
    reason = "needs MPS",
)


def _tokens(px: int, text: int) -> int:
    return (px // 16) ** 2 + text


def _qkv(
    heads,
    tokens,
    head_dim,
    dtype,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn((1, heads, tokens, head_dim), generator = g, dtype = torch.float32).to(dtype)
        for _ in range(3)
    ]


def _reference_rows(q, k, v, heads, rows):
    q = q[:, heads][:, :, rows].double()
    scores = q @ k[:, heads].double().transpose(-1, -2) / math.sqrt(q.shape[-1])
    return torch.softmax(scores, dim = -1) @ v[:, heads].double()


def _cos(a, b):
    return float(
        torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim = 0)
    )


DEVICE = "mps"


def _dev(*tensors):
    return [t.to(DEVICE) for t in tensors]


def _sync():
    if DEVICE == "mps":
        torch.mps.synchronize()


def _mps_attention(
    q,
    k,
    v,
    attn_mask = None,
):
    with torch.inference_mode():
        out = torch.nn.functional.scaled_dot_product_attention(*_dev(q, k, v), attn_mask = attn_mask)
        _sync()
    return out


def _driver_growth(fn):
    if DEVICE != "mps":
        return fn(), 0
    torch.mps.synchronize()
    torch.mps.empty_cache()
    base = torch.mps.driver_allocated_memory()
    out = fn()
    torch.mps.synchronize()
    return out, torch.mps.driver_allocated_memory() - base


def _record(
    family,
    px,
    dtype,
    tokens,
    heads,
    prefill,
    cos,
    grown = None,
):
    elements = heads * tokens * tokens
    line = (
        f"MPS-ATTN torch={torch.__version__} family={family} px={px} dtype={str(dtype).split('.')[-1]} "
        f"tokens={tokens} score_elements={elements} (2^{math.log2(elements):.2f}) prefill={prefill} "
        f"cos={cos:.6f}" + ("" if grown is None else f" grown_mib={grown / 2**20:.0f}")
    )
    print(line)
    warnings.warn(line, UserWarning, stacklevel = 2)


TOL = {torch.float32: 0.99999, torch.float16: 0.9995, torch.bfloat16: 0.998}


@mps
@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("px", (256, 512))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16), ids = str)
def test_small_sizes_match_cpu(family, px, dtype):
    heads, head_dim, text = FAMILIES[family]
    tokens = _tokens(px, text)
    q, k, v = _qkv(heads, tokens, head_dim, dtype)
    out = _mps_attention(q, k, v).cpu()
    prefill = attention.mps_prefill_attention(*_dev(q, k, v))
    rows = torch.arange(0, tokens, max(1, tokens // 256))
    picked = list(range(heads))
    ref = _reference_rows(q, k, v, picked, rows)
    cos = _cos(out[:, picked][:, :, rows], ref)
    _record(family, px, dtype, tokens, heads, prefill, cos)
    assert bool(torch.isfinite(out).all())
    assert cos >= TOL[dtype]


@mps
def test_key_padding_mask_matches_cpu():
    """Z-Image pads its caption and masks the padding with a bool key mask."""
    heads, head_dim, text = FAMILIES["z-image"]
    tokens = _tokens(512, text)
    q, k, v = _qkv(heads, tokens, head_dim, torch.bfloat16)
    mask = torch.ones((1, 1, 1, tokens), dtype = torch.bool)
    mask[..., -text // 2 :] = False
    out = _mps_attention(q, k, v, attn_mask = mask.to(DEVICE)).cpu()
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.double(), k.double(), v.double(), attn_mask = mask
    )
    cos = _cos(out, ref)
    _record("z-image+mask", 512, torch.bfloat16, tokens, heads, "n/a", cos)
    assert cos >= TOL[torch.bfloat16]


@mps
def test_prefill_engagement_follows_the_torch_version():
    heads, head_dim, text = FAMILIES["flux2-klein-4b"]
    tokens = _tokens(1024, text)
    q, k, v = _dev(*_qkv(heads, tokens, head_dim, torch.bfloat16))
    prefill = attention.mps_prefill_attention(q, k, v)
    assert prefill == (attention._version_tuple(torch.__version__)[:2] >= (2, 13))
    _, grown = _driver_growth(lambda: _mps_attention(q, k, v))
    scores = heads * tokens * tokens * 2
    _record("flux2-klein-4b", 1024, torch.bfloat16, tokens, heads, prefill, float("nan"), grown)
    if not prefill:
        assert grown >= scores // 2
    elif os.environ.get("UNSLOTH_MPS_EXPECT_PREFILL", "1") == "1":
        # Set UNSLOTH_MPS_EXPECT_PREFILL=0 on a GPU without the features the kernel needs, to record only.
        assert grown < scores // 4


def _large_cell(family, target_elements):
    heads, head_dim, _ = FAMILIES[family]
    tokens = int(math.ceil(math.sqrt(target_elements / heads)))
    q, k, v = _qkv(heads, tokens, head_dim, torch.bfloat16, seed = 1)
    out, grown = _driver_growth(lambda: _mps_attention(q, k, v))
    out = out.cpu()
    # The last heads sit at the largest offsets, where pytorch#179352 goes wrong first.
    picked = sorted({0, heads // 2, heads - 2, heads - 1})
    rows = torch.linspace(0, tokens - 1, 192).long()
    ref = _reference_rows(q, k, v, picked, rows)
    worst = min(_cos(out[:, [h]][:, :, rows], ref[:, [i]]) for i, h in enumerate(picked))
    prefill = attention.mps_prefill_attention(*_dev(q, k, v))
    _record(family, -1, torch.bfloat16, tokens, heads, prefill, worst, grown)
    return worst


@mps
@pytest.mark.skipif(
    not LARGE, reason = "UNSLOTH_MPS_LARGE_ATTENTION=1 or CI (GBs of scores on torch < 2.13)"
)
@pytest.mark.parametrize("family", ("z-image", "flux2-klein-4b", "qwen-image-2.1", "sd3.5-large"))
def test_past_the_studio_measured_threshold(family):
    """Just past where Studio saw wrong values on macOS 15 / torch 2.11: does every family hit it?"""
    assert _large_cell(family, int(1.15 * STUDIO_WRONG_FROM)) >= TOL[torch.bfloat16]


@mps
@pytest.mark.skipif(not LARGE, reason = "UNSLOTH_MPS_LARGE_ATTENTION=1 or CI")
def test_past_the_upstream_threshold_on_the_prefill_kernel():
    """Above 2^32 elements only the prefill kernel fits a CI runner: it never builds the scores."""
    heads, head_dim, _ = FAMILIES["qwen-image-2.1"]
    probe = _dev(*_qkv(heads, 16, head_dim, torch.bfloat16))
    if not attention.mps_prefill_attention(*probe):
        pytest.skip(
            f"torch {torch.__version__} has no MPS prefill kernel: the full matrix would be > 8 GB"
        )
    assert _large_cell("qwen-image-2.1", int(1.1 * UPSTREAM_WRONG_FROM)) >= TOL[torch.bfloat16]


@mps
@pytest.mark.skipif(
    not LARGE, reason = "UNSLOTH_MPS_LARGE_ATTENTION=1 or CI (GBs of scores on torch < 2.13)"
)
@pytest.mark.parametrize("family", ("z-image", "flux2-klein-4b", "qwen-image-2.1", "sd3.5-large"))
def test_the_guard_fixes_the_studio_measured_threshold(family, monkeypatch):
    """The same cells with Studio's MPS large-attention guard installed: right on every torch."""
    from core import _mps_sdpa_guard as guard

    monkeypatch.setattr(
        torch.nn.functional,
        "scaled_dot_product_attention",
        torch.nn.functional.scaled_dot_product_attention,
    )
    monkeypatch.setitem(guard.stats, "chunked_calls", 0)
    assert guard.install_mps_large_attention_guard() is True
    assert _large_cell(family, int(1.15 * STUDIO_WRONG_FROM)) >= TOL[torch.bfloat16]
    prefill = attention._version_tuple(torch.__version__)[:2] >= (2, 13)
    print(
        f"MPS-GUARD torch={torch.__version__} family={family} chunked_calls={guard.stats['chunked_calls']}"
    )
    # torch 2.13+ takes the prefill kernel, so nothing is split; below it the call must have been split.
    assert (guard.stats["chunked_calls"] == 0) == prefill
