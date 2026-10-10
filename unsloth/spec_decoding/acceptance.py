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
"""Core acceptance-rate maths for speculative decoding.

For a single speculative step where the draft samples a token from ``q`` and the target
distribution is ``p``, the probability the token is accepted by the standard
Leviathan/Chen rejection rule is::

    alpha = E_{x ~ q}[ min(1, p(x) / q(x)) ] = sum_x min(p(x), q(x)) = 1 - TV(p, q)

This module computes that quantity from aligned probability tensors, and turns a mean
acceptance rate into the expected number of tokens emitted per verification step and a
rough wall-clock speedup estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def acceptance_prob(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> torch.Tensor:
    """Per-position acceptance probability ``sum min(p, q)``, over the last dim.

    ``target_probs`` and ``draft_probs`` must be normalised over a shared vocab and have
    matching shape ``[..., V]``. Returns ``[...]``.
    """
    if target_probs.shape != draft_probs.shape:
        raise ValueError(
            f"shape mismatch: {tuple(target_probs.shape)} vs {tuple(draft_probs.shape)}"
        )
    return torch.minimum(target_probs, draft_probs).sum(dim = -1)


def residual_distribution(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> torch.Tensor:
    """The normalised ``(p - q)_+`` distribution the target resamples from on rejection."""
    resid = torch.clamp(target_probs - draft_probs, min = 0.0)
    total = resid.sum(dim = -1, keepdim = True)
    # if p == q exactly the residual is all-zero; fall back to p to stay a valid dist
    return torch.where(total > 0, resid / total, target_probs)


@dataclass(frozen = True)
class AcceptedLength:
    """Expected outcome of one speculative cycle proposing ``gamma`` draft tokens."""

    gamma: int
    alpha: float
    accepted_draft_tokens: float  # E[length of accepted draft prefix], in [0, gamma]
    tokens_per_cycle: float  # accepted prefix + 1 always-present corrected/bonus token


def expected_accepted_length(alpha: float, gamma: int) -> AcceptedLength:
    """Closed form under the i.i.d.-acceptance approximation (constant ``alpha``).

    E[accepted prefix] = sum_{k=1..gamma} alpha^k ; one further token (a resample on the
    first rejection, or the target's bonus token if all gamma are accepted) is always
    emitted, so tokens_per_cycle = E[accepted prefix] + 1.
    """
    if not -1e-6 <= alpha <= 1.0 + 1e-6:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    alpha = min(1.0, max(0.0, alpha))  # tolerate fp rounding from the sum of minimums
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    if alpha == 1.0:
        accepted = float(gamma)
    else:
        accepted = alpha * (1.0 - alpha**gamma) / (1.0 - alpha)
    return AcceptedLength(
        gamma = gamma,
        alpha = alpha,
        accepted_draft_tokens = accepted,
        tokens_per_cycle = accepted + 1.0,
    )


def estimate_speedup(tokens_per_cycle: float, gamma: int, cost_ratio: float) -> float:
    """Rough wall-clock speedup vs plain autoregressive decoding.

    ``cost_ratio`` = (one draft forward) / (one target forward). One speculative cycle
    costs ~ ``gamma * cost_ratio`` (sequential draft steps) + ``1`` (batched target
    verification), and emits ``tokens_per_cycle`` tokens that would otherwise need
    ``tokens_per_cycle`` target forwards.

    Ignores draft KV-cache warmup, verification of gamma+1 positions being slightly more
    than one forward, batching, and memory-bandwidth effects. Treat as an order-of-
    magnitude indication, not a benchmark.
    """
    if not 0.0 <= cost_ratio:
        raise ValueError("cost_ratio must be >= 0")
    cycle_cost = gamma * cost_ratio + 1.0
    return tokens_per_cycle / cycle_cost
