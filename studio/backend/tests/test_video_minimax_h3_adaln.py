# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic CPU tests for MiniMax-H3's pruned (curve-form) adaLN conversion.

No diffusers, no CUDA, no checkpoint: a tiny stand-in with the same module layout as
``MiniMaxH3Transformer3DModel`` (``transformer_blocks[i].adaln_proj.linear``, ``norm_out.linear``,
``time_embedder``, ``time_proj``) exercises every branch of the conversion, and the numerics are
checked against the reference formula written out longhand rather than against the implementation.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from core.inference.video_minimax_h3_adaln import (
    apply_h3_adaln_curve,
    is_curve_checkpoint,
)

HIDDEN = 4
CURVE_DIM = 3
CURVE_GRID = 5
MODALITIES = 3


def _curve_meta(**overrides):
    meta = {
        "adaln_form": "curve",
        "curve_dim": CURVE_DIM,
        "curve_grid": CURVE_GRID,
        "scheme": "int8",
        "family": "minimax-h3",
    }
    meta.update(overrides)
    return meta


class _Modulation(nn.Module):
    """Stands in for ``MiniMaxH3AdaLayerNormModulation`` (dense form: SiLU then project)."""

    def __init__(self, time_embed_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(time_embed_dim, 6 * hidden_size * MODALITIES)

    def forward(self, temb):
        temb = self.linear(nn.functional.silu(temb).to(self.linear.weight.dtype))
        return temb.view(-1, 6 * self.hidden_size).chunk(6, dim = -1)


class _NormOut(nn.Module):
    """Stands in for ``MiniMaxH3AdaLayerNormOut``."""

    def __init__(self, time_embed_dim: int, hidden_size: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps = 1e-5)
        self.linear = nn.Linear(time_embed_dim, 2 * hidden_size)

    def forward(self, hidden_states, temb, timestep_indices):
        shift, scale = self.linear(nn.functional.silu(temb).to(self.linear.weight.dtype)).chunk(
            2, dim = -1
        )
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )


class _Block(nn.Module):
    def __init__(self, time_embed_dim: int):
        super().__init__()
        self.adaln_proj = _Modulation(time_embed_dim, HIDDEN)


class _FourierTimeProj(nn.Module):
    """Stands in for ``Timesteps``: expands a scalar timestep into features.

    Deliberately NOT an identity, so a test asserting `time_proj` became a passthrough actually
    fails when the conversion forgets to replace it."""

    def forward(self, timestep):
        return torch.stack([timestep.sin(), timestep.cos()], dim = -1)


class _FakeH3(nn.Module):
    """The dense module layout the conversion rewrites."""

    def __init__(
        self,
        time_embed_dim: int = 7,
        num_layers: int = 2,
    ):
        super().__init__()
        self.time_proj = _FourierTimeProj()
        self.time_embedder = nn.Module()
        self.time_embedder.linear_1 = nn.Linear(2, time_embed_dim)
        self.time_embedder.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
        self.transformer_blocks = nn.ModuleList([_Block(time_embed_dim) for _ in range(num_layers)])
        self.norm_out = _NormOut(time_embed_dim, HIDDEN)


# ── is_curve_checkpoint ──────────────────────────────────────────────────────────
def test_is_curve_checkpoint_accepts_a_fully_specified_curve_artifact():
    assert is_curve_checkpoint(_curve_meta()) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"adaln_form": "dense"},
        {"adaln_form": None},
        {"curve_dim": 0},
        {"curve_grid": None},
    ],
)
def test_is_curve_checkpoint_rejects_incomplete_metadata(overrides):
    # A checkpoint that does not DECLARE the form, or omits either dimension, must not trigger a
    # reshape: the loader would then install differently-shaped weights and generate noise.
    assert is_curve_checkpoint(_curve_meta(**overrides)) is False


def test_is_curve_checkpoint_rejects_a_non_mapping():
    assert is_curve_checkpoint(None) is False
    assert is_curve_checkpoint("curve") is False


# ── apply_h3_adaln_curve ─────────────────────────────────────────────────────────
def test_dense_metadata_leaves_the_model_untouched():
    model = _FakeH3()
    before = model.transformer_blocks[0].adaln_proj.linear.in_features
    assert apply_h3_adaln_curve(model, {"adaln_form": "dense"}) is False
    assert model.transformer_blocks[0].adaln_proj.linear.in_features == before
    assert isinstance(model.time_proj, _FourierTimeProj)
    assert hasattr(model.time_embedder, "linear_1")


def test_curve_conversion_reshapes_every_projection():
    model = _FakeH3(num_layers = 3)
    assert apply_h3_adaln_curve(model, _curve_meta()) is True
    for block in model.transformer_blocks:
        assert block.adaln_proj.linear.in_features == CURVE_DIM
        # Output width is the modulation fan-out and must NOT change.
        assert block.adaln_proj.linear.out_features == 6 * HIDDEN * MODALITIES
    assert model.norm_out.linear.in_features == CURVE_DIM
    assert model.norm_out.linear.out_features == 2 * HIDDEN
    assert model.time_embedder.table.shape == (CURVE_GRID, CURVE_DIM)


def test_curve_conversion_rejects_a_model_without_the_h3_layout():
    with pytest.raises(ValueError, match = "MiniMaxH3Transformer3DModel"):
        apply_h3_adaln_curve(nn.Linear(2, 2), _curve_meta())


def test_curve_conversion_rejects_a_block_whose_projection_is_not_a_linear():
    model = _FakeH3()
    model.transformer_blocks[0].adaln_proj.linear = nn.Identity()
    with pytest.raises(ValueError, match = "expected a Linear"):
        apply_h3_adaln_curve(model, _curve_meta())


# ── the state dict must match the hosted checkpoint exactly ──────────────────────
def test_conversion_swaps_the_time_embedder_keys_for_the_table():
    model = _FakeH3()
    apply_h3_adaln_curve(model, _curve_meta())
    keys = set(model.state_dict())
    assert "time_embedder.table" in keys
    # The dense MLP keys must be GONE, or strict=True load of a curve checkpoint reports them missing.
    assert not any(k.startswith("time_embedder.linear_") for k in keys)


def test_the_dtype_shim_stays_out_of_the_state_dict():
    # The model's forward reads time_embedder.linear_1.weight.dtype, so the shim must exist as an
    # attribute but must NOT be a persistent buffer: an extra key breaks the strict load it exists to allow.
    model = _FakeH3()
    apply_h3_adaln_curve(model, _curve_meta())
    assert model.time_embedder.linear_1.weight.dtype == torch.float32
    assert "time_embedder.linear_1.weight" not in set(model.state_dict())


# ── numerics ─────────────────────────────────────────────────────────────────────
def _fill_table(model):
    with torch.no_grad():
        model.time_embedder.table.copy_(
            torch.arange(CURVE_GRID * CURVE_DIM, dtype = torch.float32).view(CURVE_GRID, CURVE_DIM)
        )


def test_time_embedder_interpolates_between_the_two_neighbouring_grid_rows():
    model = _FakeH3()
    apply_h3_adaln_curve(model, _curve_meta())
    _fill_table(model)
    table = model.time_embedder.table
    # Half-way between grid rows 0 and 1 (grid of 5 spans [0,1], so t=0.125 is row 0.5).
    got = model.time_embedder(torch.tensor([0.125]))
    expected = 0.5 * table[0] + 0.5 * table[1]
    assert torch.allclose(got[0], expected, atol = 1e-6)


def test_time_embedder_pins_the_grid_endpoints():
    model = _FakeH3()
    apply_h3_adaln_curve(model, _curve_meta())
    _fill_table(model)
    table = model.time_embedder.table
    got = model.time_embedder(torch.tensor([0.0, 1.0]))
    assert torch.equal(got[0], table[0])
    # t=1.0 must land exactly on the LAST row, not read past the table.
    assert torch.equal(got[1], table[-1])


def test_time_embedder_clamps_out_of_range_timesteps_to_the_curve_ends():
    model = _FakeH3()
    apply_h3_adaln_curve(model, _curve_meta())
    _fill_table(model)
    table = model.time_embedder.table
    got = model.time_embedder(torch.tensor([-3.0, 7.5]))
    assert torch.equal(got[0], table[0])
    assert torch.equal(got[1], table[-1])


def test_modulation_forward_drops_the_silu():
    # The tabulated curve is the dense path's POST-activation embedding projected onto the basis,
    # so re-applying SiLU would square the nonlinearity. This is the assertion that catches it.
    model = _FakeH3()
    apply_h3_adaln_curve(model, _curve_meta())
    proj = model.transformer_blocks[0].adaln_proj
    temb = torch.randn(2, CURVE_DIM)
    got = torch.cat(proj(temb), dim = -1)
    raw = proj.linear(temb).view(-1, 6 * HIDDEN)
    silu = proj.linear(nn.functional.silu(temb)).view(-1, 6 * HIDDEN)
    assert torch.allclose(got, raw, atol = 1e-6)
    assert not torch.allclose(got, silu, atol = 1e-4)


def test_modulation_casts_the_chunks_to_the_recorded_stream_dtype():
    # The pruned modulation is stored float32 while the block stack runs bfloat16, and the block's
    # forward multiplies without casting. Leaving the chunks float32 promotes the stack and the
    # first quantized matmul fails on mismatched dtypes, so this cast is what makes the model run.
    model = _FakeH3()
    apply_h3_adaln_curve(model, _curve_meta(adaln_out_dtype = "bfloat16"))
    proj = model.transformer_blocks[0].adaln_proj
    assert proj.linear.weight.dtype == torch.float32
    chunks = proj(torch.randn(2, CURVE_DIM))
    assert all(chunk.dtype == torch.bfloat16 for chunk in chunks)


def test_modulation_keeps_the_projection_dtype_when_none_was_recorded():
    # An unrecognised or absent value must not be guessed at: leave the chunks where they were.
    model = _FakeH3()
    apply_h3_adaln_curve(model, _curve_meta(adaln_out_dtype = "not_a_dtype"))
    chunks = model.transformer_blocks[0].adaln_proj(torch.randn(2, CURVE_DIM))
    assert all(chunk.dtype == torch.float32 for chunk in chunks)


def test_norm_out_is_not_cast_down():
    # Unlike the block modulation, the final layer's shift/scale stay at their own precision: the
    # result goes straight into the float32 output heads.
    model = _FakeH3()
    apply_h3_adaln_curve(model, _curve_meta(adaln_out_dtype = "bfloat16"))
    out = model.norm_out(torch.randn(2, HIDDEN), torch.randn(1, CURVE_DIM), torch.tensor([0, 0]))
    assert out.dtype == torch.float32


def test_the_out_dtype_marker_stays_out_of_the_state_dict():
    model = _FakeH3()
    apply_h3_adaln_curve(model, _curve_meta(adaln_out_dtype = "bfloat16"))
    assert not any("adaln_out_dtype" in k for k in model.state_dict())


def test_modulation_forward_keeps_the_dense_row_layout():
    # Rows are [t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]: the block's adaln_indices address that
    # layout, so a reshape that changed it would silently modulate the wrong modality.
    model = _FakeH3()
    apply_h3_adaln_curve(model, _curve_meta())
    chunks = model.transformer_blocks[0].adaln_proj(torch.randn(2, CURVE_DIM))
    assert len(chunks) == 6
    for chunk in chunks:
        assert chunk.shape == (2 * MODALITIES, HIDDEN)


def test_norm_out_forward_drops_the_silu_and_indexes_per_row():
    model = _FakeH3()
    apply_h3_adaln_curve(model, _curve_meta())
    norm_out = model.norm_out
    temb = torch.randn(2, CURVE_DIM)
    hidden = torch.randn(4, HIDDEN)
    indices = torch.tensor([0, 1, 1, 0])
    got = norm_out(hidden, temb, indices)
    shift, scale = norm_out.linear(temb).chunk(2, dim = -1)
    expected = norm_out.norm(hidden) * (1.0 + scale.index_select(0, indices)) + shift.index_select(
        0, indices
    )
    assert torch.allclose(got, expected, atol = 1e-6)
    silu_shift, silu_scale = norm_out.linear(nn.functional.silu(temb)).chunk(2, dim = -1)
    silu_expected = norm_out.norm(hidden) * (
        1.0 + silu_scale.index_select(0, indices)
    ) + silu_shift.index_select(0, indices)
    assert not torch.allclose(got, silu_expected, atol = 1e-4)


def test_time_proj_becomes_a_passthrough():
    # The curve table is indexed by the RAW timestep, not by time_proj's Fourier features.
    model = _FakeH3()
    apply_h3_adaln_curve(model, _curve_meta())
    timestep = torch.tensor([0.25, 0.75])
    assert torch.equal(model.time_proj(timestep), timestep)


def test_the_curve_forward_is_bound_per_instance_not_on_the_class():
    # A dense H3 load in the same process shares these classes; patching the class would corrupt it.
    converted = _FakeH3()
    dense = _FakeH3()
    apply_h3_adaln_curve(converted, _curve_meta())
    temb = torch.randn(2, dense.norm_out.linear.in_features)
    got = torch.cat(dense.transformer_blocks[0].adaln_proj(temb), dim = -1)
    expected = (
        dense.transformer_blocks[0].adaln_proj.linear(nn.functional.silu(temb)).view(-1, 6 * HIDDEN)
    )
    assert torch.allclose(got, expected, atol = 1e-6)
