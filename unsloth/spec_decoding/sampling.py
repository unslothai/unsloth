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
"""Turn raw model logits into the probability distribution a sampler actually draws from.

Speculative-decoding acceptance is a property of the distributions the draft samples
from and the target verifies against -- *after* temperature / top-k / top-p are applied,
not of the raw softmax. To get an honest acceptance number we must apply the exact same
transform the deployment uses, consistently to both models.

All functions operate on the last dimension (vocab) and accept arbitrary leading dims.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen = True)
class SamplingParams:
    """The sampler configuration to evaluate acceptance under.

    ``temperature == 0`` means greedy (argmax); acceptance then reduces to top-1
    agreement between the two models.
    """

    temperature: float = 1.0
    top_k: int = 0  # 0 disables
    top_p: float = 1.0  # 1.0 disables

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0

    def describe(self) -> str:
        if self.is_greedy:
            return "greedy (temperature=0)"
        parts = [f"temperature={self.temperature:g}"]
        if self.top_k:
            parts.append(f"top_k={self.top_k}")
        if self.top_p < 1.0:
            parts.append(f"top_p={self.top_p:g}")
        return ", ".join(parts)


def sampling_distribution(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Map logits ``[..., V]`` to a normalised probability tensor ``[..., V]``.

    Applied in the order temperature -> top-k -> top-p -> renormalise, matching the
    common `transformers` / vLLM logits-processor pipeline.
    """
    if logits.dim() == 0:
        raise ValueError("logits must have a vocab dimension")
    logits = logits.float()

    if params.is_greedy:
        out = torch.zeros_like(logits)
        out.scatter_(-1, logits.argmax(dim = -1, keepdim = True), 1.0)
        return out

    probs = torch.softmax(logits / params.temperature, dim = -1)

    if params.top_k and params.top_k < probs.shape[-1]:
        kth = torch.topk(probs, params.top_k, dim = -1).values[..., -1, None]
        probs = torch.where(probs < kth, torch.zeros_like(probs), probs)
        probs = probs / probs.sum(dim = -1, keepdim = True)

    if params.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim = -1, descending = True)
        cumulative = torch.cumsum(sorted_probs, dim = -1)
        # keep every token up to and including the one that crosses top_p
        keep = cumulative - sorted_probs < params.top_p
        keep[..., 0] = True  # always keep the most likely token
        sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim = -1, keepdim = True)

    return probs


def total_variation(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """TV(p, q) = 1/2 * sum |p - q|, over the last dim."""
    return 0.5 * (p - q).abs().sum(dim = -1)


def align_vocab(
    target_logits: torch.Tensor, draft_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate both logit tensors to the smaller vocab size.

    Standard speculative decoding requires a shared tokenizer. Sizes still differ by a
    few tokens when one checkpoint pads the embedding matrix (e.g. 32000 vs 32001);
    truncating to the common prefix is exact in that case and approximate otherwise.
    The caller is responsible for warning the user.
    """
    vt, vd = target_logits.shape[-1], draft_logits.shape[-1]
    if vt == vd:
        return target_logits, draft_logits
    v = min(vt, vd)
    return target_logits[..., :v], draft_logits[..., :v]
