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

# Three modalities x six chunks; mirrors MINIMAX_H3_MODALITY_NUM, hardcoded to avoid diffusers.
MINIMAX_H3_MODALITY_NUM = 3

ADALN_FORM_KEY = "adaln_form"
ADALN_CURVE_FORM = "curve"
CURVE_DIM_KEY = "curve_dim"
CURVE_GRID_KEY = "curve_grid"
# Dtype of the block stack; see `_curve_modulation_forward` for why chunks are cast to it.
ADALN_OUT_DTYPE_KEY = "adaln_out_dtype"


def _resolve_torch_dtype(name: Any) -> Any:
    """Maps bfloat16 to torch.bfloat16; unknown names return None rather than guess a precision."""
    if not isinstance(name, str) or not name:
        return None
    import torch

    dtype = getattr(torch, name.replace("torch.", ""), None)
    return dtype if isinstance(dtype, torch.dtype) else None


def is_curve_checkpoint(metadata: Any) -> bool:
    """True only for a curve-form checkpoint that declares its form; a table alone is not enough."""
    if not isinstance(metadata, dict):
        return False
    if metadata.get(ADALN_FORM_KEY) != ADALN_CURVE_FORM:
        return False
    return bool(metadata.get(CURVE_DIM_KEY)) and bool(metadata.get(CURVE_GRID_KEY))


def _curve_modulation_forward(self: Any, temb: Any) -> tuple:
    """Curve-form adaLN, no SiLU; chunks cast to the stream dtype, as float32 breaks the first matmul."""
    temb = self.linear(temb.to(self.linear.weight.dtype))
    temb = temb.view(-1, MINIMAX_H3_MODALITY_NUM, 6 * self.hidden_size)
    out_dtype = getattr(self, "_unsloth_adaln_out_dtype", None)
    if out_dtype is not None:
        temb = temb.to(out_dtype)
    temb = temb.view(-1, 6 * self.hidden_size)
    return temb.chunk(6, dim = -1)


def _curve_norm_out_forward(self: Any, hidden_states: Any, temb: Any, timestep_indices: Any) -> Any:
    """Curve-form final adaLN, no SiLU; no cast down, as the float32 output heads expect promotion."""
    shift, scale = self.linear(temb.to(self.linear.weight.dtype)).chunk(2, dim = -1)
    hidden_states = self.norm(hidden_states)
    return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
        0, timestep_indices
    )


def _build_curve_time_embedder(curve_grid: int, curve_dim: int) -> Any:
    """Curve-form timestep embedder: interpolates the table, clamping so t=1.0 stays in bounds."""
    import torch
    from torch import nn

    class _MiniMaxH3CurveTimeEmbedder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Persistent: this IS the checkpoint's `time_embedder.table`; float32 like the reference.
            self.register_buffer("table", torch.empty(curve_grid, curve_dim, dtype = torch.float32))
            # Forward reads linear_1.weight.dtype; a non-persistent stand-in keeps strict=True exact.
            self.linear_1 = nn.Module()
            self.linear_1.register_buffer(
                "weight", torch.zeros(1, dtype = torch.float32), persistent = False
            )

        def forward(self, timestep: Any) -> Any:
            table = self.table
            grid = table.shape[0]
            pos = timestep.to(torch.float32).clamp(0.0, 1.0) * (grid - 1)
            i0 = pos.floor().long().clamp(max = grid - 2)
            return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))

    return _MiniMaxH3CurveTimeEmbedder()


def _passthrough_time_proj() -> Any:
    """Feeds the raw timestep to the curve embedder; Timesteps is parameter-free, so no state keys
    change."""
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
    """Must run between from_config and load_state_dict; an odd model raises rather than half-converting."""
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
        # Rebuild as a float32 Linear on the real device (not meta); assign=True then fills it.
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
        # Plain attribute, not a buffer: keep it out of the state dict and `.to(device)`.
        proj._unsloth_adaln_out_dtype = out_dtype
        # Bind per instance: the dense class is shared with dense loads in the same process.
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
    """Reshapes a curve-form model before load_state_dict; the loader builds it from the dense config."""

    def _prepare(transformer: Any, metadata: Optional[dict]) -> None:
        apply_h3_adaln_curve(transformer, metadata, logger = logger)

    return _prepare
