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
"""Run the actual speculative-decoding loop and measure what really gets accepted.

The closed form in :mod:`unsloth.spec_decoding.acceptance` assumes acceptance is i.i.d. across draft
positions and that the draft always proposes from the reference context. Neither holds
in practice: acceptance drops along the draft, and once a token is accepted the draft
continues from *its own* output. This module runs the real rejection-sampling loop so
the reported accepted length -- and acceptance *per draft offset* -- reflect those effects.

Models are passed as callables ``ids -> logits`` (last-position or full-sequence logits),
so the loop is testable with stand-ins and has no `transformers` dependency itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

from .acceptance import residual_distribution
from .sampling import SamplingParams, sampling_distribution

# A LogitsFn takes a 1-D LongTensor of token ids and returns either the full
# [seq_len, vocab] logits or just the [vocab] logits for the next token.
LogitsFn = Callable[[torch.Tensor], torch.Tensor]


def _next_logits(fn: LogitsFn, ids: torch.Tensor) -> torch.Tensor:
    out = fn(ids)
    if out.dim() == 2:
        return out[-1]
    if out.dim() == 1:
        return out
    raise ValueError(f"logits fn returned shape {tuple(out.shape)}; expected [V] or [T, V]")


@dataclass
class SimulationResult:
    """Per-cycle record of a speculative-decoding run. Every statistic derives from
    ``verified_per_cycle`` / ``accepted_per_cycle`` so they cannot disagree.

    For cycle *c*, draft offsets ``0..verified[c]-1`` were examined by the target and
    offsets ``0..accepted[c]-1`` were accepted (always a prefix). ``verified == accepted``
    with ``verified < gamma`` means the cycle was cut short by ``max_new_tokens``/EOS
    before any rejection (a censored cycle).
    """

    gamma: int
    cycles: int = 0
    generated_tokens: int = 0
    verified_per_cycle: list[int] = field(default_factory = list)
    accepted_per_cycle: list[int] = field(default_factory = list)

    # ---- token-level -------------------------------------------------------------
    @property
    def proposed_draft_tokens(self) -> int:
        """Draft tokens the target actually examined."""
        return sum(self.verified_per_cycle)

    @property
    def accepted_draft_tokens(self) -> int:
        return sum(self.accepted_per_cycle)

    @property
    def empirical_alpha(self) -> float:
        """P(accept | examined) over all examined draft tokens -- comparable to the
        distributional alpha."""
        n = self.proposed_draft_tokens
        return self.accepted_draft_tokens / n if n else 0.0

    # ---- cycle-level -------------------------------------------------------------
    def _complete(self) -> list[int]:
        """Indices of cycles that ended naturally (a rejection, or all gamma accepted)."""
        return [
            i
            for i, (v, a) in enumerate(zip(self.verified_per_cycle, self.accepted_per_cycle))
            if v > a or v == self.gamma
        ]

    @property
    def accepted_lengths(self) -> list[int]:
        """Accepted-prefix length of each complete cycle."""
        return [self.accepted_per_cycle[i] for i in self._complete()]

    @property
    def mean_accepted_length(self) -> float:
        """Mean draft tokens accepted per complete cycle (excludes the bonus token)."""
        lens = self.accepted_lengths
        return sum(lens) / len(lens) if lens else 0.0

    @property
    def tokens_per_cycle(self) -> float:
        """Tokens emitted per complete cycle: the accepted prefix plus exactly one more
        (the residual resample on rejection, or the target's bonus on full accept)."""
        return self.mean_accepted_length + 1.0 if self.accepted_lengths else 0.0

    # ---- per draft offset --------------------------------------------------------
    @property
    def reached_by_offset(self) -> list[int]:
        """How many cycles examined draft offset j at all, j = 0..gamma-1.

        Reaching offset j requires offsets 0..j-1 to have been accepted, so this
        shrinks monotonically -- it is the cohort size behind each ``alpha_by_offset``.
        """
        return [sum(1 for v in self.verified_per_cycle if v > j) for j in range(self.gamma)]

    @property
    def alpha_by_offset(self) -> list[float]:
        """P(accept at draft offset j | offset j was reached), j = 0..gamma-1.

        **This is a conditional probability on a shrinking, self-selected cohort**, not a
        pure degradation curve. Two effects fight:

        - *drift* pushes it down: by offset j the draft is conditioning on j of its own
          tokens, so it has wandered further from the target.
        - *selection* pushes it up: only cycles where the draft already agreed j times
          reach offset j, and those are the easy, predictable contexts. On degenerate
          output (a greedy repetition loop, say) selection can win outright and the
          curve rises at the tail.

        Read it alongside :attr:`reached_by_offset`. NaN where an offset was never reached.
        """
        out: list[float] = []
        for j, reached in enumerate(self.reached_by_offset):
            accepted = sum(1 for a in self.accepted_per_cycle if a > j)
            out.append(accepted / reached if reached else math.nan)
        return out

    def extend(self, other: "SimulationResult") -> None:
        if other.gamma != self.gamma:
            raise ValueError("cannot merge simulations with different gamma")
        self.cycles += other.cycles
        self.generated_tokens += other.generated_tokens
        self.verified_per_cycle.extend(other.verified_per_cycle)
        self.accepted_per_cycle.extend(other.accepted_per_cycle)


@torch.no_grad()
def simulate_speculative(
    target: LogitsFn,
    draft: LogitsFn,
    prompt_ids: torch.Tensor,
    *,
    gamma: int = 4,
    max_new_tokens: int = 128,
    sampling: SamplingParams | None = None,
    generator: torch.Generator | None = None,
    eos_token_id: int | None = None,
    vocab_size: int | None = None,
) -> SimulationResult:
    """Speculative decoding with exact rejection sampling (Leviathan et al., 2023).

    ``vocab_size`` is the shared vocab both models are scored over; logits beyond it are
    dropped *before* the draft samples, so proposed tokens are always in range for the
    target. If ``None`` it is probed once as ``min(V_target, V_draft)``.

    Uses full-sequence forwards (no KV cache) -- fine for a measurement harness on small
    models, not meant to be fast.
    """
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    sampling = sampling or SamplingParams()
    ids = prompt_ids.clone().long().flatten()
    result = SimulationResult(gamma = gamma)

    if vocab_size is None:
        vocab_size = min(_next_logits(target, ids).shape[-1], _next_logits(draft, ids).shape[-1])
    V = vocab_size

    def _dist(logits: torch.Tensor) -> torch.Tensor:
        return sampling_distribution(logits[..., :V], sampling)

    def _sample(probs: torch.Tensor) -> int:
        if sampling.is_greedy:
            return int(probs.argmax())
        return int(torch.multinomial(probs, 1, generator = generator))

    def _emit(tok: int) -> None:
        nonlocal ids
        ids = torch.cat([ids, torch.tensor([tok], dtype = torch.long)])
        result.generated_tokens += 1

    while result.generated_tokens < max_new_tokens:
        result.cycles += 1
        base_len = ids.shape[0]

        # 1. draft proposes gamma tokens autoregressively, from its own outputs
        draft_tokens: list[int] = []
        draft_probs: list[torch.Tensor] = []
        for _ in range(gamma):
            ctx = (
                ids
                if not draft_tokens
                else torch.cat([ids, torch.tensor(draft_tokens, dtype = torch.long)])
            )
            q = _dist(_next_logits(draft, ctx))
            draft_tokens.append(_sample(q))
            draft_probs.append(q)

        # 2. target verifies all gamma positions + the bonus position in one forward
        cand = torch.cat([ids, torch.tensor(draft_tokens, dtype = torch.long)])
        t_logits = target(cand)
        if t_logits.dim() != 2:
            raise ValueError("target fn must return full [T, V] logits for verification")

        # 3. walk the proposed prefix, accept/reject each examined token
        n_accepted = 0
        n_verified = 0
        rejected = False
        hit_eos = False
        for j in range(gamma):
            if result.generated_tokens >= max_new_tokens:
                break
            n_verified += 1
            p_j = _dist(t_logits[base_len - 1 + j])
            q_j = draft_probs[j]
            tok = draft_tokens[j]

            if sampling.is_greedy:
                accept = tok == int(p_j.argmax())
            else:
                ratio = 1.0 if q_j[tok] <= 0 else min(1.0, float(p_j[tok] / q_j[tok]))
                accept = torch.rand(1, generator = generator).item() < ratio

            if accept:
                _emit(tok)
                n_accepted += 1
                if eos_token_id is not None and tok == eos_token_id:
                    hit_eos = True
                    break
            else:
                # resample the corrected token from the normalised residual (p - q)_+
                _emit(_sample(residual_distribution(p_j, q_j)))
                rejected = True
                break

        result.verified_per_cycle.append(n_verified)
        result.accepted_per_cycle.append(n_accepted)
        if hit_eos:
            break

        # 4. if the whole prefix was accepted, take the target's bonus token
        if not rejected and n_verified == gamma and result.generated_tokens < max_new_tokens:
            bonus = _sample(_dist(t_logits[base_len - 1 + gamma]))
            _emit(bonus)
            if eos_token_id is not None and bonus == eos_token_id:
                break

        if eos_token_id is not None and int(ids[-1]) == eos_token_id:
            break

    return result
