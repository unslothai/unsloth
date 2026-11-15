# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 per-row AdaLN modulation / gated residual (video_minimax_h3_rowmod): structure match, CPU fallback,
kill switch, and on CUDA the kernels and whole forward bit-identical to the stock chain."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
for p in (_BACKEND, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")

from core.inference import diffusion_minimax_h3_image as h3i  # noqa: E402
from core.inference import video_minimax_h3_rowmod as R  # noqa: E402


def _model(dtype = torch.float32):
    import _minimax_h3_image_tiny as tiny

    cfg = h3i.transformer_config_from_native(tiny.TINY_TRANSFORMER_NATIVE)
    torch.manual_seed(0)
    model = h3i.transformer_class()(**cfg).eval()
    tiny.randomize(model, 0)
    return model.to(dtype)


def _inputs(
    model,
    dtype = torch.float32,
    n_text = 5,
    lh = 8,
    lw = 8,
    refs = (),
):
    g = torch.Generator().manual_seed(1)
    patch = tuple(model.config.patch_size)
    tags = torch.full((n_text,), h3i.H3_IMAGE_TEXT_TAG, dtype = torch.long)
    pos, tags, image_idx, text_idx, n_ref = h3i.build_packed_layout(tags, list(refs), lh, lw, patch)
    seq = int(image_idx.numel() + text_idx.numel())
    unique, index = h3i.row_timestep_plan(torch.tensor([0.75]), seq, image_idx[:n_ref])[0]
    rows = (lh // patch[1]) * (lw // patch[2]) + sum(
        (h // patch[1]) * (w // patch[2]) for h, w in refs
    )
    text_dim = int(model.context_embedder.in_features)
    in_dim = int(model.proj_in.in_features)
    return dict(
        hidden_states = torch.randn(1, rows, in_dim, generator = g).to(dtype),
        encoder_hidden_states = torch.randn(1, n_text, text_dim, generator = g).to(dtype),
        timestep = unique,
        timestep_indices = index,
        token_tags = tags,
        position_ids = pos,
        image_indices = image_idx,
        text_indices = text_idx,
        pdd_step_index = 0,
        return_dict = False,
    )


def _run(model, inputs):
    with torch.no_grad():
        return model(**inputs)[0]


def _to(inputs, device, dtype):
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if v.is_floating_point() and k in ("hidden_states", "encoder_hidden_states"):
                v = v.to(dtype)
        out[k] = v
    return out


def test_the_vendored_block_matches_the_stock_body():
    model = _model()
    assert all(R.matches_stock_block(b) for b in model.transformer_blocks)


def test_a_changed_block_forward_is_left_alone():
    model = _model()

    class Other(type(model.transformer_blocks[0])):
        def forward(
            self,
            hidden_states,
            temb,
            adaln_indices,
            rotary_emb,
            attention_mask = None,
        ):
            return hidden_states

    block = model.transformer_blocks[0]
    block.__class__ = Other
    assert not R.matches_stock_block(block)


def test_cpu_falls_back_to_the_stock_math_bit_for_bit(monkeypatch):
    monkeypatch.setattr(R, "install", R.install)
    model = _model(torch.bfloat16)
    inputs = _to(_inputs(model, refs = [(8, 8)]), "cpu", torch.bfloat16)
    ref = _run(model, inputs)
    for block in model.transformer_blocks:
        block.forward = R._row_mod_block_forward.__get__(block)
    before = R.STATS["stock"]
    got = _run(model, inputs)
    assert torch.equal(got, ref)
    assert R.STATS["stock"] > before


def test_install_is_off_without_cuda_and_under_the_switch(monkeypatch):
    model = _model()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert R.install(model) == 0
    monkeypatch.setenv(R.ROW_MOD_ENV, "0")
    assert R.install(model) == 0


def test_install_skips_blocks_with_an_instance_forward(monkeypatch):
    pytest.importorskip("triton")
    model = _model()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    marker = lambda *a, **k: None  # noqa: E731
    model.transformer_blocks[0].forward = marker
    n = R.install(model)
    assert n == len(model.transformer_blocks) - 1
    assert model.transformer_blocks[0].forward is marker
    assert R.uninstall(model) == n
    assert all("forward" not in b.__dict__ for b in model.transformer_blocks[1:])


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_kernels_match_the_eager_chain_bit_for_bit(dtype):
    pytest.importorskip("triton")
    v = R._check(torch.device("cuda"), dtype, 5376)
    assert v["exact"], v
    for seed in (1, 2):
        x, (s0, s1, s2, *_r), idx, out = R._probe("cuda", dtype, 5376, seed, rows = 1031)
        assert torch.equal(R._launch_modulate(x, s1, s0, idx), R.reference_modulate(x, s1, s0, idx))
        assert torch.equal(
            R._launch_gated(x, s2, out, idx), R.reference_gated_residual(x, s2, out, idx)
        )


@cuda
def test_the_row_forward_matches_the_stock_forward_on_cuda(monkeypatch):
    pytest.importorskip("triton")
    model = _model(torch.bfloat16).to("cuda")
    inputs = _to(_inputs(model, refs = [(8, 8)]), "cuda", torch.bfloat16)
    ref = _run(model, inputs)
    # the tiny model's 64-wide rows race too small to win: force the verdict so the kernels run
    for key, v in list(R._VERDICTS.items()):
        R._VERDICTS[key] = {**v, "use": v["exact"]}
    R.verdict(torch.empty(1, 1, 64, dtype = torch.bfloat16, device = "cuda"))
    for key, v in list(R._VERDICTS.items()):
        R._VERDICTS[key] = {**v, "use": v["exact"]}
    assert R.install(model) == len(model.transformer_blocks)
    before = R.STATS["fused"]
    got = _run(model, inputs)
    assert R.STATS["fused"] - before == 4 * len(model.transformer_blocks)
    assert torch.equal(got, ref)
    monkeypatch.setenv(R.ROW_MOD_ENV, "0")
    assert torch.equal(_run(model, inputs), ref)
