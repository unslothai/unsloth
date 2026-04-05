# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Optional

import torch


def _read_flag(value, default = False):
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip().lower()
        return value in ("1", "true", "yes", "on")
    return bool(value)


def _read_int(value, default):
    try:
        parsed = int(value)
    except Exception:
        return default
    return max(1, parsed)


def _read_float(value, default):
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class _AttnResState:
    enabled: bool
    block_size: int
    alpha: float
    num_layers: int
    completed_block_summaries: list[torch.Tensor] = field(default_factory = list)
    current_block_states: list[torch.Tensor] = field(default_factory = list)


def _get_config_value(config, names, default = None):
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
    return default


def _build_state(model, *, use_cache = False):
    config = getattr(model, "config", None)
    if config is None:
        return None

    enabled = _read_flag(
        _get_config_value(
            config,
            (
                "attnres",
                "attn_residual",
                "attn_residuals",
                "use_attnres",
                "use_attn_residuals",
                "_attnres",
                "_use_attnres",
            ),
            default = None,
        ),
        default = False,
    )
    if not enabled:
        enabled = _read_flag(os.environ.get("UNSLOTH_ATTNRES"), default = False)
    if not enabled:
        enabled = _read_flag(os.environ.get("ATTNRES"), default = False)
    if not enabled:
        return None

    # Stateful accumulation can desync across recomputation passes.
    if (
        getattr(model, "training", False)
        and getattr(model, "gradient_checkpointing", False)
        and not use_cache
    ):
        return None

    block_size = _read_int(
        _get_config_value(
            config,
            ("attnres_block_size", "attn_residual_block_size", "attnres_block"),
            default = os.environ.get("UNSLOTH_ATTNRES_BLOCK_SIZE", 8),
        ),
        default = 8,
    )
    alpha = _read_float(
        _get_config_value(
            config,
            ("attnres_alpha", "attn_residual_alpha"),
            default = os.environ.get("UNSLOTH_ATTNRES_ALPHA", 1.0),
        ),
        default = 1.0,
    )
    num_layers = int(getattr(config, "num_hidden_layers", 0))
    if num_layers <= 0 and hasattr(model, "layers"):
        num_layers = len(model.layers)
    return _AttnResState(
        enabled = True,
        block_size = block_size,
        alpha = alpha,
        num_layers = num_layers,
    )


def begin_attnres_state(model, *, use_cache = False):
    return _build_state(model, use_cache = use_cache)


def attnres_init_forward_state(
    model,
    hidden_states = None,
    attention_mask = None,
    position_ids = None,
    past_key_values = None,
    use_cache = False,
    output_attentions = False,
    output_hidden_states = False,
):
    return begin_attnres_state(model, use_cache = use_cache)


def _compute_residual_mix(
    query: torch.Tensor,
    candidates: list[torch.Tensor],
) -> torch.Tensor:
    if len(candidates) == 0:
        return torch.zeros_like(query)

    # (bsz, seqlen, n_states, dim)
    stacked = torch.stack(candidates, dim = 2)
    dim = query.shape[-1]
    scale = 1.0 / math.sqrt(float(dim))
    logits = (query.unsqueeze(2) * stacked).sum(dim = -1) * scale
    weights = torch.softmax(logits, dim = -1)
    return (weights.unsqueeze(-1) * stacked).sum(dim = 2)


def attnres_transform_attention_output(
    attention_output: torch.Tensor,
    attnres_state: Optional[_AttnResState] = None,
    attnres_layer_idx: Optional[int] = None,
    residual: Optional[torch.Tensor] = None,
    attention_mask = None,
    causal_mask = None,
    position_ids = None,
):
    if attnres_state is None or not getattr(attnres_state, "enabled", False):
        return attention_output

    layer_idx = 0 if attnres_layer_idx is None else int(attnres_layer_idx)
    query = residual if residual is not None else attention_output

    candidates = list(attnres_state.completed_block_summaries)
    candidates.extend(attnres_state.current_block_states)
    if len(candidates) != 0:
        mixed = _compute_residual_mix(query, candidates)
        attention_output = attention_output + (attnres_state.alpha * mixed)

    # Keep current-layer information for future layers in this block.
    attnres_state.current_block_states.append(attention_output)

    # Finalize a block summary at boundaries.
    at_block_end = ((layer_idx + 1) % attnres_state.block_size) == 0
    at_model_end = (attnres_state.num_layers > 0) and (
        (layer_idx + 1) >= attnres_state.num_layers
    )
    if at_block_end or at_model_end:
        block_summary = torch.stack(attnres_state.current_block_states, dim = 0).sum(
            dim = 0
        )
        attnres_state.completed_block_summaries.append(block_summary)
        attnres_state.current_block_states.clear()

    return attention_output


__all__ = [
    "begin_attnres_state",
    "attnres_init_forward_state",
    "attnres_transform_attention_output",
]
