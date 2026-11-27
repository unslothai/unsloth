# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""DiffSynth-Studio H3 LoRAs carry per-head fused qkv rows [heads, 3, head_dim]; contiguous thirds would mix heads."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from core.inference import video_lora  # noqa: E402

H3 = "minimax-h3"
HEADS, HEAD_DIM, HIDDEN, RANK = 3, 128, 16, 4


def _fused_lora(seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(RANK, HIDDEN, generator = g)
    b = torch.randn(HEADS * 3 * HEAD_DIM, RANK, generator = g)
    return a, b


def _reference(a, b, scale):
    delta = (b @ a) * scale
    grouped = delta.reshape(HEADS, 3, HEAD_DIM, HIDDEN)
    return {
        name: grouped[:, i].reshape(HEADS * HEAD_DIM, HIDDEN)
        for i, name in enumerate(("to_q", "to_k", "to_v"))
    }


def _save(tmp_path, tensors):
    path = tmp_path / "diffsynth_h3.safetensors"
    save_file(tensors, str(path))
    return str(path)


@pytest.mark.parametrize(
    "module, target",
    [
        ("blocks.0.attn.qkv_proj", "transformer_blocks.0.attn"),
        ("blocks.7.attn.qkv_proj", "transformer_blocks.7.attn"),
        ("token_refiner.blocks.1.attn.qkv_proj", "token_refiner.refiner_blocks.1.attn"),
    ],
)
def test_diffsynth_fused_qkv_matches_the_raw_module(tmp_path, module, target):
    a, b = _fused_lora(len(module))
    path = _save(
        tmp_path,
        {f"{module}.lora_A.default.weight": a, f"{module}.lora_B.default.weight": b},
    )
    deltas = video_lora.read_deltas([(path, 0.75)], family = H3)
    expected = _reference(a, b, 0.75)
    for proj, ref in expected.items():
        ((got_a, got_b, scale),) = deltas[f"{target}.{proj}"]
        assert torch.equal((got_b @ got_a) * scale, ref), proj


def test_diffsynth_rows_match_diffsynths_own_layout(tmp_path):
    """DiffSynth's converter (utils/lora/minimax_h3.py) puts q / k / v rows at head * 384 + {0, 128, 256}."""
    g = torch.Generator().manual_seed(0)
    a = {p: torch.randn(RANK, HIDDEN, generator = g) for p in "qkv"}
    b = {p: torch.randn(HEADS * HEAD_DIM, RANK, generator = g) for p in "qkv"}
    out = HEADS * HEAD_DIM
    fused_b = torch.zeros(out * 3, RANK * 3)
    for modality, p in enumerate("qkv"):
        rows = (
            torch.arange(out).reshape(HEADS, 128)
            + modality * 128
            + torch.arange(HEADS)[:, None] * 256
        ).reshape(-1)
        fused_b[rows, modality * RANK : (modality + 1) * RANK] = b[p]
    fused_a = torch.cat([a[p] for p in "qkv"], dim = 0)
    path = _save(
        tmp_path,
        {
            "blocks.2.attn.qkv_proj.lora_A.default.weight": fused_a,
            "blocks.2.attn.qkv_proj.lora_B.default.weight": fused_b,
        },
    )
    deltas = video_lora.read_deltas([(path, 1.0)], family = H3)
    for p in "qkv":
        ((got_a, got_b, scale),) = deltas[f"transformer_blocks.2.attn.to_{p}"]
        assert scale == 1.0
        assert torch.allclose(got_b @ got_a, b[p] @ a[p], atol = 1e-5, rtol = 1e-5), p


def test_community_files_keep_the_contiguous_thirds(tmp_path):
    """Community files (#19818) store contiguous [q; k; v]."""
    a, b = _fused_lora(1)
    path = _save(
        tmp_path,
        {
            "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": a,
            "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": b,
        },
    )
    deltas = video_lora.read_deltas([(path, 1.0)], family = H3)
    for i, proj in enumerate(("to_q", "to_k", "to_v")):
        ((_a, got_b, _s),) = deltas[f"transformer_blocks.0.attn.{proj}"]
        third = HEADS * HEAD_DIM
        assert torch.equal(got_b, b[i * third : (i + 1) * third]), proj


def test_diffsynth_fc1_and_out_proj_are_unchanged(tmp_path):
    g = torch.Generator().manual_seed(3)
    tensors = {
        "blocks.0.mlp.fc1.lora_A.default.weight": torch.randn(RANK, HIDDEN, generator = g),
        "blocks.0.mlp.fc1.lora_B.default.weight": torch.randn(8, RANK, generator = g),
        "blocks.0.attn.out_proj.lora_A.default.weight": torch.randn(RANK, HIDDEN, generator = g),
        "blocks.0.attn.out_proj.lora_B.default.weight": torch.randn(6, RANK, generator = g),
    }
    deltas = video_lora.read_deltas([(_save(tmp_path, tensors), 1.0)], family = H3)
    fc1 = tensors["blocks.0.mlp.fc1.lora_B.default.weight"]
    ((_a, got, _s),) = deltas["transformer_blocks.0.ff.net.0.proj"]
    assert torch.equal(got, torch.cat([fc1[4:], fc1[:4]]))
    ((_a, got, _s),) = deltas["transformer_blocks.0.attn.to_out.0"]
    assert torch.equal(got, tensors["blocks.0.attn.out_proj.lora_B.default.weight"])


def test_per_head_qkv_needs_whole_heads():
    with pytest.raises(ValueError, match = "per-head"):
        video_lora.h3_community_key_map("blocks.0.attn.qkv_proj", 3 * 100, per_head_qkv = True)
