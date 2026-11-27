# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image batched forward: same-layout images run every block as ONE call over all their rows.

Same layout (canvas, text tokens and tags, reference grids, steps) means equal position ids, AdaLN indices, rotary
tables and timesteps, and no block op mixes rows, so each row computes as its own forward. The fp32 patch projection
and output head run per row at batch-1 GEMM shape (the 6-step sampler amplifies 1e-7 roundings), so batched images
are bit-identical to their own renders on CUDA.

Opt-in ``UNSLOTH_H3_IMAGE_BATCHED=1``; ``UNSLOTH_H3_IMAGE_BATCH_TOKENS`` (4608) caps packed rows per call.
Batch 4, int8: RTX PRO 6000 1024x1024 1.11x, 2048x2048 resident 0.88x; A100-80 1024x1024 resident 1.00x."""

from __future__ import annotations

import os
import types
from typing import Any, Optional, Sequence

BATCHED_ENV = "UNSLOTH_H3_IMAGE_BATCHED"
BATCH_TOKENS_ENV = "UNSLOTH_H3_IMAGE_BATCH_TOKENS"
BATCH_TOKENS = 4608
_SHARED_FIELDS = ("position_ids", "token_tags", "image_indices", "text_indices")


def batched_forward_enabled() -> bool:
    return (os.environ.get(BATCHED_ENV) or "").strip().lower() in ("1", "on", "true", "yes")


def layout_key(job: dict) -> tuple:
    """What two set-up images must share to run one block call; tensors are compared by ``same_layout``."""
    return (
        int(job["lh"]),
        int(job["lw"]),
        tuple(job["patch"]),
        int(job["seq"]),
        int(job["text_rows"]),
        int(job["n_ref"]),
        int(job["steps"]),
        int(job["rows"].shape[-1]),
    )


def same_layout(a: dict, b: dict) -> bool:
    import torch

    if layout_key(a) != layout_key(b):
        return False
    if a.get("dtype") != b.get("dtype") or sorted(a.get("extra") or {}) != sorted(
        b.get("extra") or {}
    ):
        return False
    if (a.get("bank") is None) != (b.get("bank") is None) or a.get("spans") != b.get("spans"):
        return False
    for name in _SHARED_FIELDS:
        x, y = a["fields"][name], b["fields"][name]
        if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y):
            return False
    if len(a["plan"]) != len(b["plan"]):
        return False
    for (ua, ia), (ub, ib) in zip(a["plan"], b["plan"]):
        if not (torch.equal(ua, ub) and torch.equal(ia, ib)):
            return False
    ta, tb = a["timesteps"], b["timesteps"]
    return ta.shape == tb.shape and torch.equal(ta.cpu(), tb.cpu())


def batch_groups(jobs: Sequence[dict], cap: int) -> list:
    """Job indices grouped by layout (first-seen order), cut into batches of at most ``cap``."""
    cap = max(1, int(cap))
    groups: list = []
    for index, job in enumerate(jobs):
        for group in groups:
            if same_layout(jobs[group[0]], job):
                group.append(index)
                break
        else:
            groups.append([index])
    out = []
    for group in groups:
        out.extend(group[i : i + cap] for i in range(0, len(group), cap))
    return out


def merged_static(calls: Sequence[dict]) -> dict:
    import torch

    out = {}
    for name in ("encoder_hidden_states", "text_embeds"):
        parts = [c.get(name) for c in calls]
        if all(p is not None for p in parts):
            out[name] = parts[0] if len(parts) == 1 else torch.cat(parts)
    return out


def merge_step_calls(calls: Sequence[dict], static: Optional[dict] = None) -> dict:
    import torch

    merged = dict(calls[0])
    if len(calls) == 1:
        return merged
    merged["hidden_states"] = torch.cat([c["hidden_states"] for c in calls])
    merged.update(static if static is not None else merged_static(calls))
    return merged


def split_rows(output: Any, count: int) -> list:
    if int(output.shape[0]) != count:
        raise ValueError(f"batched forward returned {int(output.shape[0])} rows for {count} images")
    return [output[b : b + 1] for b in range(count)]


def batched_row_gb(
    seq: int,
    hidden: int,
    ffn: int,
    heads_dim: Optional[int] = None,
    itemsize: int = 2,
) -> float:
    """GB one more batched row holds beyond its lockstep state; generous (every intermediate live at once)."""
    inner = int(heads_dim or hidden)
    per = seq * itemsize * (hidden * 4 + inner * 4 + ffn * 3)
    return per / 1e9


def batch_tokens() -> int:
    try:
        return max(1, int(os.environ.get(BATCH_TOKENS_ENV) or BATCH_TOKENS))
    except ValueError:
        return BATCH_TOKENS


def batch_cap(
    images: int,
    room: Optional[float],
    state_gb: float,
    row_gb: float,
    seq: int = 0,
) -> int:
    """Rows per block call that fit ``room`` and ``batch_tokens()``; 1 = one call per image."""
    if images < 2 or not batched_forward_enabled() or room is None:
        return 1
    tokens = images if seq <= 0 else max(1, batch_tokens() // int(seq))
    left = float(room) - (images - 1) * float(state_gb)
    if left <= 0:
        return 1
    fit = images if row_gb <= 0 else 1 + int(left // float(row_gb))
    return max(1, min(images, tokens, fit))


def row_layout_key(
    pipe: Any,
    prompt: str,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    image: Any = None,
    num_inference_steps: Optional[int] = None,
    min_generation_area: Optional[int] = None,
) -> tuple:
    """Coalescer key: equal keys share one batched block call (tokenizer only, no text encoder)."""
    from . import diffusion_minimax_h3_image as h3

    vae_refs, vision_refs = pipe.prepare_references(image)
    if not vae_refs:
        default_w = default_h = h3.H3_IMAGE_DEFAULT_SIZE
    elif h3.source_resize_enabled():
        first = image[0] if isinstance(image, (list, tuple)) else image
        default_w, default_h = h3.reference_edit_size(*h3.upright_size(first))
    else:
        default_w, default_h = h3.picture_size(vae_refs[0])
    width, height = int(width or default_w), int(height or default_h)
    if min_generation_area is None:
        canvas = (width, height)
    else:
        canvas_h, canvas_w = h3.canvas_size(height, width, min_generation_area)
        canvas = (canvas_w, canvas_h)
    tokens = len(pipe.tokenizer(prompt, add_special_tokens = False)["input_ids"]) if prompt else 0
    steps = h3.H3_IMAGE_STEPS if num_inference_steps is None else int(num_inference_steps)
    return (
        canvas,
        tokens,
        tuple(h3.picture_size(p) for p in vae_refs),
        tuple(h3.picture_size(p) for p in vision_refs),
        steps,
        (width, height),
    )


# --- the row-mod kernel at batch > 1 ---------------------------------------------------------------------------------


def _flat(x: Any) -> Any:
    b, s = int(x.shape[0]), int(x.shape[1])
    return x.reshape(1, b * s, x.shape[-1])


def batched_modulate(rowmod: Any, x: Any, scale: Any, shift: Any, idx: Any) -> Any:
    """``rowmod.modulate`` over ``(B, S, D)`` flattened to ``(1, B*S, D)`` (AdaLN index repeated) for the batch-1 kernel."""
    b = int(x.shape[0])
    if b == 1 or x.dim() != 3:
        return rowmod.modulate(x, scale, shift, idx)
    return rowmod.modulate(_flat(x), scale, shift, idx.repeat(b)).view(x.shape)


def batched_gated_residual(rowmod: Any, residual: Any, gate: Any, out: Any, idx: Any) -> Any:
    b = int(residual.shape[0])
    if b == 1 or residual.dim() != 3 or out.shape != residual.shape:
        return rowmod.gated_residual(residual, gate, out, idx)
    return rowmod.gated_residual(_flat(residual), gate, _flat(out), idx.repeat(b)).view(
        residual.shape
    )


def _batched_row_mod_block_forward(
    self: Any,
    hidden_states: Any,
    temb: Any,
    adaln_indices: Any,
    rotary_emb: Any,
    attention_mask: Any = None,
) -> Any:
    """``_row_mod_block_forward`` at any batch: batch 1 as is, else row ops over the flattened rows."""
    from . import video_minimax_h3_rowmod as rowmod

    if int(hidden_states.shape[0]) == 1:
        return rowmod._row_mod_block_forward(
            self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask
        )
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

    residual = hidden_states
    norm_hidden_states = batched_modulate(
        rowmod, self.norm1(hidden_states), scale_msa, shift_msa, adaln_indices
    )
    attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
    hidden_states = batched_gated_residual(rowmod, residual, gate_msa, attn_output, adaln_indices)

    residual = hidden_states
    norm_hidden_states = batched_modulate(
        rowmod, self.norm2(hidden_states), scale_mlp, shift_mlp, adaln_indices
    )
    ff_output = self.ff(norm_hidden_states)
    hidden_states = batched_gated_residual(rowmod, residual, gate_mlp, ff_output, adaln_indices)

    return hidden_states


def install_batched_row_mod(transformer: Any) -> int:
    from . import video_minimax_h3_rowmod as rowmod

    n = 0
    for block in getattr(transformer, "transformer_blocks", ()) or ():
        forward = getattr(block.__dict__.get("forward"), "__func__", None)
        if (
            not block.__dict__.get(rowmod._FORWARD_MARK)
            or forward is not rowmod._row_mod_block_forward
        ):
            continue
        block.forward = types.MethodType(_batched_row_mod_block_forward, block)
        n += 1
    return n
