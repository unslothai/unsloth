# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3's PRUNED (curve-form) adaptive layer norm, for hosted pre-quantized denoisers.

MiniMax-H3 spends roughly 40% of its parameters on modulation. Every block owns an
``adaln_proj.linear`` of shape ``(6 * hidden_size * 3, time_embed_dim)`` = ``(96768, 2688)``, and
with 50 blocks plus ``norm_out`` that is ~26 GB of the 66 GB denoiser, all of it recomputed from a
timestep embedding that only ever takes one of a few dozen distinct values in a sampling run.

The reference ComfyUI implementation removes that redundancy. ``silu(time_embedder(t))``, viewed as
a function of ``t`` over a fixed 1025-point grid on ``[0, 1]``, is a smooth curve living in a
rank-8 affine subspace, so it factors as ``u(t) ~= mean + B @ c(t)`` with ``B`` of shape
``(2688, 8)`` and the 8 coordinates ``c(t)`` tabulated per grid point. Folding ``B`` into the
projection collapses each block's modulation to ``(96768, 8)``:

    dense:  y = W_dense @ silu(temb) + b_dense
    curve:  y = W_curve @ c(t) + b_curve,  W_curve = W_dense @ B,  b_curve = b_dense + W_dense @ mean

The factorization is AFFINE, not a pure rank-8 product: the bias absorbs the curve's mean. A key
rename that inherits ``b_dense`` is wrong by an order of magnitude. Nothing here re-derives the
factorization; the hosted checkpoints ship ``W_curve`` / ``b_curve`` / the table already fitted, and
this module only rebuilds the module shapes and the forward that consume them.

``MiniMaxH3Transformer3DModel`` cannot load that form as shipped: against the dense config it is 4
keys missing (``time_embedder.linear_1/linear_2.{weight,bias}``), 1 unexpected
(``time_embedder.table``) and 51 shape mismatches (every ``adaln_proj.linear.weight`` plus
``norm_out.linear.weight``, ``(*, 8)`` against ``(*, 2688)``). So a hosted pre-quantized H3 denoiser
is unloadable by every route until the model is reshaped to match, which is what
``apply_h3_adaln_curve`` does, in place, between ``from_config`` and ``load_state_dict``.

Two behavioural differences from the dense path, both load-bearing:

* No SiLU. The tabulated curve is the activation's OWN output projected onto the basis, so applying
  SiLU again would square the nonlinearity.
* The table is indexed by the RAW timestep, not by the Fourier features. ``time_proj`` is therefore
  bypassed (it is parameter-free, so the state dict is unaffected).
"""

from __future__ import annotations

import types
from typing import Any, Optional

# Video, text and audio rows each get their own modulation row, so one projection emits three
# blocks of six chunks. Mirrors diffusers' MINIMAX_H3_MODALITY_NUM; hardcoded rather than imported
# so this module stays importable (and unit-testable) without diffusers.
MINIMAX_H3_MODALITY_NUM = 3

# Metadata keys the offline prequant builder bakes into a curve-form checkpoint.
ADALN_FORM_KEY = "adaln_form"
ADALN_CURVE_FORM = "curve"
CURVE_DIM_KEY = "curve_dim"
CURVE_GRID_KEY = "curve_grid"
# The dtype of the block stack the modulation feeds, recorded by the builder. See
# `_curve_modulation_forward` for why the chunks have to be cast to it.
ADALN_OUT_DTYPE_KEY = "adaln_out_dtype"


def _resolve_torch_dtype(name: Any) -> Any:
    """``"bfloat16"`` -> ``torch.bfloat16``; None for anything unrecognised.

    A missing or unknown value must NOT fall back to a guess: the cast it drives changes the block
    stack's precision, and silently picking the wrong one would be a quality regression no test
    would catch. None simply leaves the chunks at the projection's own dtype, which is the
    pre-existing behaviour."""
    if not isinstance(name, str) or not name:
        return None
    import torch

    dtype = getattr(torch, name.replace("torch.", ""), None)
    return dtype if isinstance(dtype, torch.dtype) else None


def is_curve_checkpoint(metadata: Any) -> bool:
    """True when ``metadata`` describes a pruned (curve-form) adaLN checkpoint.

    Keyed on the recorded form rather than on the presence of ``time_embedder.table``: a checkpoint
    that carries the table but does not declare the form is one this code has not been validated
    against, and silently reshaping the model for it would load mismatched weights under
    ``strict = True`` only to produce noise."""
    if not isinstance(metadata, dict):
        return False
    if metadata.get(ADALN_FORM_KEY) != ADALN_CURVE_FORM:
        return False
    # Both dimensions are required: without them the reshape would have to guess the basis rank.
    return bool(metadata.get(CURVE_DIM_KEY)) and bool(metadata.get(CURVE_GRID_KEY))


def _curve_modulation_forward(self: Any, temb: Any) -> tuple:
    """``MiniMaxH3AdaLayerNormModulation.forward`` for the curve form: no SiLU, then cast down.

    ``temb`` already holds the interpolated curve coordinates, i.e. the projection of the dense
    path's post-activation embedding onto the fitted basis. The view/chunk tail is byte-identical to
    the dense module, so the row layout the block's ``adaln_indices`` addresses is unchanged.

    The pruned modulation is stored FLOAT32 (the rank-8 curve is a small, precision-sensitive
    signal), while the dense checkpoint's projections are bfloat16. The block's forward multiplies
    the normed hidden states by these chunks WITHOUT casting, so leaving them float32 promotes the
    whole block stack to float32 and the very first quantized matmul dies with
    "expected mat1 and mat2 to have the same dtype". The reference casts modulation to the hidden
    stream's dtype at the point of use for exactly this reason; ``adaln_out_dtype`` is the dtype the
    offline builder recorded for that stream, so honour it here where the chunks are produced."""
    temb = self.linear(temb.to(self.linear.weight.dtype))
    out_dtype = getattr(self, "_unsloth_adaln_out_dtype", None)
    if out_dtype is not None:
        temb = temb.to(out_dtype)
    temb = temb.view(-1, 6 * self.hidden_size)
    return temb.chunk(6, dim = -1)


def _curve_norm_out_forward(self: Any, hidden_states: Any, temb: Any, timestep_indices: Any) -> Any:
    """``MiniMaxH3AdaLayerNormOut.forward`` for the curve form: no SiLU, same indexing.

    No cast down here, unlike the block modulation: the reference's final layer also consumes its
    shift/scale at their own precision, and the model's forward sends this result straight into the
    float32 output heads, so promoting is what the dense path effectively does too."""
    shift, scale = self.linear(temb.to(self.linear.weight.dtype)).chunk(2, dim = -1)
    hidden_states = self.norm(hidden_states)
    return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
        0, timestep_indices
    )


def _build_curve_time_embedder(curve_grid: int, curve_dim: int) -> Any:
    """The module replacing ``TimestepEmbedding`` on a curve-form model.

    Holds the fitted table under the checkpoint's own ``time_embedder.table`` key and turns a raw
    timestep into curve coordinates by linear interpolation between the two neighbouring grid rows,
    matching the reference exactly (clamp to ``[0, 1]``, then clamp the lower index to
    ``grid - 2`` so ``t == 1.0`` lands on the last interval instead of reading past the table)."""
    import torch
    from torch import nn

    class _MiniMaxH3CurveTimeEmbedder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Persistent: this IS the checkpoint's `time_embedder.table` entry, assigned by
            # load_state_dict. float32 like the reference; the curve is a smooth low-amplitude
            # signal whose differences drive every block's modulation.
            self.register_buffer("table", torch.empty(curve_grid, curve_dim, dtype = torch.float32))
            # The model's forward reads `self.time_embedder.linear_1.weight.dtype` to decide what to
            # cast the timestep to. The dense module has that Linear; this one does not, so expose a
            # NON-PERSISTENT stand-in carrying only the dtype. Non-persistent keeps it out of the
            # state dict, so `strict = True` still matches the checkpoint exactly.
            self.linear_1 = nn.Module()
            self.linear_1.register_buffer(
                "weight", torch.zeros(1, dtype = torch.float32), persistent = False
            )

        def forward(self, timestep: Any) -> Any:
            table = self.table
            grid = table.shape[0]
            # Out-of-range timesteps clamp to the curve's ends rather than extrapolating.
            pos = timestep.to(torch.float32).clamp(0.0, 1.0) * (grid - 1)
            i0 = pos.floor().long().clamp(max = grid - 2)
            return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))

    return _MiniMaxH3CurveTimeEmbedder()


def _passthrough_time_proj() -> Any:
    """Replaces ``Timesteps`` so the raw timestep reaches the curve embedder.

    The dense path feeds ``time_proj``'s Fourier features to the time embedder; the curve table is
    indexed by the timestep itself. ``Timesteps`` is parameter-free, so swapping it changes no
    state-dict key."""
    from torch import nn

    class _MiniMaxH3RawTimestep(nn.Module):
        def forward(self, timestep: Any) -> Any:
            return timestep

    return _MiniMaxH3RawTimestep()


def apply_h3_adaln_curve(
    transformer: Any,
    metadata: Any,
    logger: Any = None,
) -> bool:
    """Reshape a freshly built ``MiniMaxH3Transformer3DModel`` to the pruned adaLN form, in place.

    Call between ``from_config`` and ``load_state_dict``: it swaps ``time_proj`` / ``time_embedder``
    and re-shapes every ``adaln_proj.linear`` plus ``norm_out.linear`` from ``time_embed_dim`` inputs
    to ``curve_dim``, so the hosted checkpoint then loads under ``strict = True``.

    Returns True when the model was converted, False when ``metadata`` does not describe a
    curve-form checkpoint (a dense checkpoint must be left exactly as it was). Raises on a
    structurally unexpected model, so the prequant loader's caller falls back to dense rather than
    generating from a half-converted model."""
    if not is_curve_checkpoint(metadata):
        return False

    from torch import nn

    curve_dim = int(metadata[CURVE_DIM_KEY])
    curve_grid = int(metadata[CURVE_GRID_KEY])

    blocks = getattr(transformer, "transformer_blocks", None)
    norm_out = getattr(transformer, "norm_out", None)
    if blocks is None or norm_out is None:
        raise ValueError(
            "MiniMax-H3 curve conversion needs `transformer_blocks` and `norm_out`; this is not a "
            "MiniMaxH3Transformer3DModel."
        )

    def _reshape(linear: Any, where: str) -> Any:
        # Rebuild rather than resize: the replacement is a plain float32 Linear (the hosted
        # checkpoints store the pruned adaLN in float32) whose weights load_state_dict then
        # overwrites via assign=True. Built on the real device, never meta, so the module is
        # well-formed even if the checkpoint were to omit it and strict=True caught that instead.
        if not isinstance(linear, nn.Linear):
            raise ValueError(f"MiniMax-H3 curve conversion expected a Linear at {where}.")
        import torch
        return nn.Linear(curve_dim, linear.out_features, bias = linear.bias is not None).to(
            torch.float32
        )

    out_dtype = _resolve_torch_dtype(metadata.get(ADALN_OUT_DTYPE_KEY))

    converted = 0
    for index, block in enumerate(blocks):
        proj = getattr(block, "adaln_proj", None)
        if proj is None:
            raise ValueError(f"MiniMax-H3 curve conversion: block {index} has no `adaln_proj`.")
        proj.linear = _reshape(proj.linear, f"transformer_blocks.{index}.adaln_proj.linear")
        # Plain attribute, not a buffer: it must not reach the state dict, and `.to(device)` on the
        # module must not try to move it.
        proj._unsloth_adaln_out_dtype = out_dtype
        # Bind the SiLU-free forward per instance: the dense class is shared with dense loads in the
        # same process, so patching the CLASS would corrupt them.
        proj.forward = types.MethodType(_curve_modulation_forward, proj)
        converted += 1

    norm_out.linear = _reshape(norm_out.linear, "norm_out.linear")
    norm_out.forward = types.MethodType(_curve_norm_out_forward, norm_out)

    transformer.time_embedder = _build_curve_time_embedder(curve_grid, curve_dim)
    transformer.time_proj = _passthrough_time_proj()

    if logger is not None:
        logger.info(
            "video.h3_adaln_curve: converted %d block projections + norm_out to rank-%d "
            "(grid %d) pruned modulation, emitting %s",
            converted,
            curve_dim,
            curve_grid,
            out_dtype or "the projection dtype",
        )
    return True


def h3_prepare_prequant_model(logger: Any = None) -> Any:
    """A ``prepare_model`` callback for ``load_prequantized_transformer``.

    The loader builds the model from the base repo's DENSE transformer config, so a curve-form
    hosted checkpoint has to reshape it before ``load_state_dict``; this adapts
    ``apply_h3_adaln_curve`` to the callback's ``(transformer, metadata)`` shape."""

    def _prepare(transformer: Any, metadata: Optional[dict]) -> None:
        apply_h3_adaln_curve(transformer, metadata, logger = logger)

    return _prepare
