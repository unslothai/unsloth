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

"""SPLADE sparse pooling head for UEmbed (Qwen3.5) style embedding checkpoints.

UEmbed ships a `sparse_weights.pt` sidecar holding `num_eos_tokens` linear heads
(`sparse_lm_heads` + `sparse_bias`) that project hidden states onto vocabulary logits.
Two modes build the sparse vector (paper Eq. 3-4, arXiv:2608.02583):

    splade.last : head i reads `last_index - ((N - 1) - i)`, so head 0 pools the first
                  EOS slot and head N-1 the last; logits concatenated, `log1p(relu(.))`.
    splade.max  : head 0 runs over every position, `log1p(relu(.))`, padding masked out,
                  per-dimension max over the sequence.

The heads are trainable `nn.Parameter`s so the trainer can fine-tune them alongside the
LoRA adapter. Opt-in: constructed only when a SPLADE mode is requested, so dense
embedders are untouched. Torch-only, so it imports without an accelerator.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn


# Sidecar file UEmbed ships next to the weights, and the keys it stores.
SPARSE_WEIGHTS_FILENAME = "sparse_weights.pt"
SPARSE_LM_HEADS_KEY = "sparse_lm_heads"
SPARSE_BIAS_KEY = "sparse_bias"

# Pooling modes that select this module. Deliberately disjoint from the stock
# sentence-transformers modes so existing dense embedders never reach this code.
SPLADE_LAST = "splade.last"
SPLADE_MAX = "splade.max"
SPLADE_POOLING_MODES = frozenset({SPLADE_LAST, SPLADE_MAX})


def is_splade_pooling_mode(pooling_mode: Any) -> bool:
    """True when the caller explicitly asked for a SPLADE sparse pooling mode."""
    return isinstance(pooling_mode, str) and pooling_mode in SPLADE_POOLING_MODES


def _uembed_pooling():
    """Import the sibling pooling module, by package or (standalone) by file path.

    The file-path branch keeps this usable without importing all of `unsloth`.
    """
    try:
        from . import uembed_pooling  # noqa: PLC0415

        return uembed_pooling
    except ImportError:
        pass

    name = "unsloth_uembed_pooling_direct"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uembed_pooling.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SpladeHead(nn.Module):
    """Trainable SPLADE sparse pooling heads loaded from UEmbed's `sparse_weights.pt`.

    `sparse_lm_heads` are `(vocab_i, hidden)` matrices and `sparse_bias` the matching
    `(vocab_i,)` vectors, one pair per EOS slot. `num_eos_tokens` (from `sparse_info.json`)
    is how many heads `splade.last` consumes; 0 means no EOS block, so `splade.last` is
    unavailable. An empty head list is the "sidecar not loaded" state: construction
    succeeds so the module can be wired before weights arrive, but pooling then raises
    instead of returning a wrong-shaped vector.
    """

    def __init__(
        self,
        sparse_lm_heads: Sequence[torch.Tensor],
        sparse_bias: Sequence[torch.Tensor],
        num_eos_tokens: int = 0,
    ) -> None:
        super().__init__()
        heads, biases = list(sparse_lm_heads), list(sparse_bias)
        if len(heads) != len(biases):
            raise ValueError(
                f"Unsloth: `{SPARSE_LM_HEADS_KEY}` has {len(heads)} entries but "
                f"`{SPARSE_BIAS_KEY}` has {len(biases)}; they must match one-to-one."
            )
        for index, (weight, bias) in enumerate(zip(heads, biases)):
            if weight.dim() != 2 or bias.dim() != 1 or bias.shape[0] != weight.shape[0]:
                raise ValueError(
                    f"Unsloth: sparse head {index} must be a (vocab, hidden) matrix with a "
                    f"(vocab,) bias, got {tuple(weight.shape)} and {tuple(bias.shape)}."
                )
            if weight.shape[1] != heads[0].shape[1]:
                raise ValueError(
                    f"Unsloth: sparse head {index} expects hidden size {weight.shape[1]}, but "
                    f"head 0 expects {heads[0].shape[1]}."
                )
        is_integer = isinstance(num_eos_tokens, int) and not isinstance(num_eos_tokens, bool)
        if not is_integer or num_eos_tokens < 0:
            raise ValueError(
                f"Unsloth: num_eos_tokens must be a non-negative integer, got {num_eos_tokens!r}."
            )

        self.num_eos_tokens = int(num_eos_tokens)
        self.sparse_lm_heads = nn.ParameterList(
            [nn.Parameter(weight.detach().clone(), requires_grad=True) for weight in heads]
        )
        self.sparse_bias = nn.ParameterList(
            [nn.Parameter(bias.detach().clone(), requires_grad=True) for bias in biases]
        )

    @property
    def num_heads(self) -> int:
        return len(self.sparse_lm_heads)

    def _require_heads(self, mode: str) -> None:
        if self.num_heads == 0:
            raise ValueError(
                f"Unsloth: `{mode}` pooling needs the sparse heads, but none are loaded. "
                f"Load `{SPARSE_WEIGHTS_FILENAME}` via `SpladeHead.from_checkpoint(...)`."
            )

    def _masked_last_indices(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_state.dim() != 3:
            raise ValueError(
                f"Unsloth: SPLADE pooling expects a (batch, sequence, hidden) hidden state, "
                f"got shape {tuple(hidden_state.shape)}."
            )
        mask = attention_mask.to(device=hidden_state.device, dtype=torch.long)
        empty_rows = (mask.sum(dim=1) == 0).nonzero(as_tuple=False).flatten()
        if empty_rows.numel():
            raise ValueError(
                f"Unsloth: attention_mask has no unmasked position for batch row(s) "
                f"{empty_rows.tolist()}; SPLADE pooling has nothing to pool there."
            )
        # Last unmasked position: cumsum peaks at the final real token, mask breaks ties.
        return mask, (mask.cumsum(dim=1) * mask).argmax(dim=1)

    def splade_last(self, hidden_state: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Concatenate one head per EOS slot, then `log1p(relu(.))` (paper Eq. 3-4)."""
        self._require_heads(SPLADE_LAST)
        num_eos = self.num_eos_tokens
        if num_eos == 0:
            raise ValueError(
                f"Unsloth: `{SPLADE_LAST}` pooling needs num_eos_tokens > 0, but the "
                f"checkpoint reports 0 (no trailing EOS block). Use `{SPLADE_MAX}` instead."
            )
        if num_eos > self.num_heads:
            raise ValueError(
                f"Unsloth: `{SPLADE_LAST}` pooling needs {num_eos} sparse heads "
                f"(num_eos_tokens), but only {self.num_heads} were loaded."
            )

        _, last_indices = self._masked_last_indices(hidden_state, attn_mask)
        # Head 0 reaches furthest back, so it alone bounds how long a sequence must be.
        short_rows = (last_indices - (num_eos - 1) < 0).nonzero(as_tuple=False).flatten()
        if short_rows.numel():
            raise ValueError(
                f"Unsloth: batch row(s) {short_rows.tolist()} are shorter than the "
                f"{num_eos}-token EOS block (last unmasked index "
                f"{last_indices[short_rows].tolist()}); `{SPLADE_LAST}` cannot pool them."
            )

        batch_indices = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        logits = []
        for index in range(num_eos):
            offset = (num_eos - 1) - index
            pooled = hidden_state[batch_indices, last_indices - offset]
            logits.append(F.linear(pooled, self.sparse_lm_heads[index], self.sparse_bias[index]))
        return torch.log1p(torch.relu(torch.cat(logits, dim=-1)))

    def splade_max(self, hidden_state: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Head 0 over every position, `log1p(relu(.))`, max over the unmasked sequence."""
        self._require_heads(SPLADE_MAX)
        mask, _ = self._masked_last_indices(hidden_state, attn_mask)
        logits = F.linear(hidden_state, self.sparse_lm_heads[0], self.sparse_bias[0])
        weights = torch.log1p(torch.relu(logits))
        weights = weights.masked_fill(~mask.unsqueeze(-1).bool(), torch.finfo(weights.dtype).min)
        return weights.max(dim=1).values

    def forward(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor, mode: str = SPLADE_LAST
    ) -> torch.Tensor:
        if mode == SPLADE_LAST:
            return self.splade_last(hidden_state, attention_mask)
        if mode == SPLADE_MAX:
            return self.splade_max(hidden_state, attention_mask)
        raise ValueError(
            f"Unsloth: unknown SPLADE pooling mode {mode!r}; expected one of "
            f"{sorted(SPLADE_POOLING_MODES)}."
        )

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str,
        num_eos_tokens: int | None = None,
        token: str | bool | None = None,
        cache_dir: str | None = None,
        revision: str | None = None,
    ) -> SpladeHead:
        """Load `sparse_weights.pt`; `num_eos_tokens` defaults to `sparse_info.json`."""
        path = os.path.join(model_dir, SPARSE_WEIGHTS_FILENAME)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Unsloth: `{SPARSE_WEIGHTS_FILENAME}` not found in `{model_dir}`; this "
                f"checkpoint carries no SPLADE sparse heads."
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        keys = (SPARSE_LM_HEADS_KEY, SPARSE_BIAS_KEY)
        missing = [key for key in keys if not isinstance(state, dict) or key not in state]
        if missing:
            raise ValueError(
                f"Unsloth: `{path}` is missing the key(s) {missing}; expected a dict with "
                f"`{SPARSE_LM_HEADS_KEY}` and `{SPARSE_BIAS_KEY}`."
            )
        if num_eos_tokens is None:
            num_eos_tokens = _uembed_pooling().read_num_eos_tokens(
                model_dir, token=token, cache_dir=cache_dir, revision=revision
            )
        return cls(state[SPARSE_LM_HEADS_KEY], state[SPARSE_BIAS_KEY], num_eos_tokens)

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, num_eos_tokens={self.num_eos_tokens}"
