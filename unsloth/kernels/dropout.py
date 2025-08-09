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

import os
import torch

_UINT32_MAX_INV = 1.0 / 4294967295.0


def _mix32(x: torch.Tensor) -> torch.Tensor:
    x = (x + 0x9E3779B9) & 0xFFFFFFFF
    x ^= (x >> 16)
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= (x >> 13)
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= (x >> 16)
    return x


@torch.compiler.disable
def seeded_dropout(x: torch.Tensor, p: float, seed: int, scale: bool = True) -> torch.Tensor:
    if p <= 0.0 or not (x.requires_grad or x.training if hasattr(x, 'training') else True):
        return x
    device = x.device
    dtype = x.dtype
    bsz, seqlen, hidden = x.shape[0], x.shape[1], x.shape[-1]
    # Indices grids (broadcasted), keep memory modest by composing increments
    b_idx = torch.arange(bsz, device=device, dtype=torch.int64).view(bsz, 1, 1)
    t_idx = torch.arange(seqlen, device=device, dtype=torch.int64).view(1, seqlen, 1)
    c_idx = torch.arange(hidden, device=device, dtype=torch.int64).view(1, 1, hidden)

    # Large coprime-like multipliers for mixing
    mixed = (b_idx * 0x1F123BB5 + t_idx * 0x5DEECE66D + c_idx * 0xB5297A4D + (seed & 0xFFFFFFFF)) & 0xFFFFFFFF
    rnd = _mix32(mixed).to(torch.float32) * _UINT32_MAX_INV
    mask = (rnd >= p).to(dtype)
    if scale and p < 1.0:
        mask = mask / (1.0 - p)
    return x * mask


class DeterministicDropout(torch.nn.Module):
    def __init__(self, p: float, seed: int = 3407):
        super().__init__()
        self.p = float(p)
        # Allow override via env variable
        env_seed = os.environ.get("UNSLOTH_DROPOUT_SEED", None)
        self.seed = int(env_seed) if env_seed is not None else int(seed)
        self.register_buffer('_counter', torch.zeros((), dtype=torch.int64), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0:
            return x
        # Derive a new seed per call to decorrelate successive uses
        local = int(self._counter.item())
        self._counter.add_(1)
        derived_seed = (self.seed + 0x9E3779B9 * local) & 0xFFFFFFFF
        return seeded_dropout(x, self.p, derived_seed, scale=True)


