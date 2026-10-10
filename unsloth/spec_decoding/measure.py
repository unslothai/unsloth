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
"""High-level ``measure_acceptance`` entry point."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import torch

from .acceptance import acceptance_prob, estimate_speedup, expected_accepted_length
from .sampling import SamplingParams, align_vocab, sampling_distribution
from .simulate import SimulationResult, simulate_speculative


@dataclass
class AcceptanceReport:
    # distributional (teacher-forced) measurement
    alpha: float  # mean sum_v min(p, q) over all scored positions
    alpha_std: float  # std of the per-sequence means
    scored_positions: int
    n_sequences: int
    scored_span: str  # "prompt" | "target continuation"

    # derived, under the i.i.d.-acceptance approximation
    gamma: int
    accepted_draft_tokens: float
    tokens_per_cycle: float
    est_speedup: float
    cost_ratio: float

    sampling: str
    vocab_truncated_to: int | None = None

    # generative cross-check (only populated with simulate=True)
    simulated: bool = False
    empirical_alpha: float | None = None
    empirical_tokens_per_cycle: float | None = None
    empirical_mean_accepted_length: float | None = None
    empirical_est_speedup: float | None = None
    # P(accept at draft offset j | offset j reached), j = 1..gamma. Only the simulation
    # can measure this. Conditional on a shrinking cohort -- see reached_by_draft_offset.
    alpha_by_draft_offset: list[float] | None = None
    # cohort size behind each offset above: reaching offset j requires 0..j-1 accepted,
    # so this shrinks and the later alphas are self-selected toward easy contexts
    reached_by_draft_offset: list[int] | None = None

    notes: list[str] = field(default_factory = list)

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def _teacher_forced_acceptance(
    target_model,
    draft_model,
    sequences: Sequence[torch.Tensor],
    score_from: Sequence[int],
    sampling: SamplingParams,
    device: str,
) -> tuple[list[float], int, int | None]:
    """Mean acceptance per sequence, scoring positions ``score_from[i] .. end``.

    Position ``k`` scores the distribution that predicts token ``k+1``, so the last
    position of a sequence is never scored.
    """
    from .models import sequence_logits  # local import keeps torch-only tests light

    per_seq_alpha: list[float] = []
    total_scored = 0
    truncated_to: int | None = None

    for ids, start in zip(sequences, score_from):
        t_logits = sequence_logits(target_model, ids, device)  # [L, V]
        d_logits = sequence_logits(draft_model, ids, device)  # [L, V]
        if t_logits.shape[-1] != d_logits.shape[-1]:
            truncated_to = min(t_logits.shape[-1], d_logits.shape[-1])
        t_logits, d_logits = align_vocab(t_logits, d_logits)

        # score [start, L-1): every position predicting a real next token in the span
        lo = max(0, min(start, t_logits.shape[0] - 1))
        p = sampling_distribution(t_logits[lo:-1], sampling)
        q = sampling_distribution(d_logits[lo:-1], sampling)
        if p.shape[0] == 0:
            continue
        a = acceptance_prob(p, q).clamp_(0.0, 1.0)
        per_seq_alpha.append(float(a.mean()))
        total_scored += a.numel()

    return per_seq_alpha, total_scored, truncated_to


def measure_acceptance(
    target_model,
    draft_model,
    sequences: Sequence[torch.Tensor],
    *,
    sampling: SamplingParams | None = None,
    gamma: int = 4,
    cost_ratio: float = 0.15,
    device: str = "cpu",
    continuation_tokens: int = 0,
    simulate: bool = False,
    simulate_max_new_tokens: int = 96,
    seed: int = 0,
    eos_token_id: int | None = None,
) -> AcceptanceReport:
    """Measure speculative-decoding acceptance for a target/draft pair.

    ``sequences`` are 1-D token-id tensors sharing the models' tokenizer.

    The headline number is the *distributional* acceptance
    ``alpha = mean_k sum_v min(p(v|x_<k), q(v|x_<k))`` measured teacher-forced at the
    given ``sampling`` setting.

    ``continuation_tokens > 0`` first extends each prompt with a sample from the **target**
    and scores only those generated positions. Prefer this: at serving time the draft is
    predicting the target's own output, not the human-written prompt, and acceptance on
    the two differs a lot. With ``0`` the prompt tokens themselves are scored.

    ``simulate=True`` additionally runs the real rejection-sampling loop, which is the only
    way to get acceptance *per draft offset* and an accepted length that accounts for the
    draft conditioning on its own proposals.
    """
    sampling = sampling or SamplingParams()
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    notes: list[str] = []

    seqs = list(sequences)
    score_from = [0] * len(seqs)
    span = "prompt"
    if continuation_tokens > 0:
        from .models import generate_continuation

        grown, starts = [], []
        for i, ids in enumerate(seqs):
            # position k scores the prediction of token k+1, so to score the first
            # generated token (index L) the span must start at L-1
            starts.append(max(0, int(ids.numel()) - 1))
            grown.append(
                generate_continuation(
                    target_model,
                    ids,
                    continuation_tokens,
                    sampling,
                    device,
                    seed = seed + i,
                    eos_token_id = eos_token_id,
                )
            )
        seqs, score_from, span = grown, starts, "target continuation"

    per_seq_alpha, scored, truncated = _teacher_forced_acceptance(
        target_model, draft_model, seqs, score_from, sampling, device
    )
    if not per_seq_alpha:
        raise ValueError(
            "no positions were scored; sequences may be too short "
            "(or the target generated nothing)"
        )

    alpha_t = torch.tensor(per_seq_alpha).clamp_(0.0, 1.0)
    alpha = float(alpha_t.mean())
    alpha_std = float(alpha_t.std(unbiased = False))

    length = expected_accepted_length(alpha, gamma)
    speedup = estimate_speedup(length.tokens_per_cycle, gamma, cost_ratio)

    if truncated is not None:
        notes.append(
            f"vocab sizes differed; logits truncated to {truncated} tokens "
            "(exact only if the surplus tokens are padding)"
        )
    if sampling.is_greedy:
        notes.append(
            "greedy: alpha is exact top-1 agreement. Deployments that sample at T>0 will "
            "see a lower acceptance rate -- re-measure at the serving temperature."
        )
    if span == "prompt":
        notes.append(
            "scored on prompt tokens. At serving time the draft predicts the target's "
            "own output; pass continuation_tokens>0 (--continuation-tokens) for a "
            "representative number."
        )

    report = AcceptanceReport(
        alpha = alpha,
        alpha_std = alpha_std,
        scored_positions = scored,
        n_sequences = len(per_seq_alpha),
        scored_span = span,
        gamma = gamma,
        accepted_draft_tokens = length.accepted_draft_tokens,
        tokens_per_cycle = length.tokens_per_cycle,
        est_speedup = speedup,
        cost_ratio = cost_ratio,
        sampling = sampling.describe(),
        vocab_truncated_to = truncated,
        notes = notes,
    )

    if simulate:
        agg = _run_simulation(
            target_model,
            draft_model,
            sequences,
            sampling,
            gamma,
            device,
            simulate_max_new_tokens,
            seed,
            eos_token_id,
        )
        report.simulated = True
        report.empirical_alpha = agg.empirical_alpha
        report.empirical_tokens_per_cycle = agg.tokens_per_cycle
        report.empirical_mean_accepted_length = agg.mean_accepted_length
        report.alpha_by_draft_offset = agg.alpha_by_offset
        report.reached_by_draft_offset = agg.reached_by_offset
        report.empirical_est_speedup = estimate_speedup(agg.tokens_per_cycle, gamma, cost_ratio)
        report.notes.append(
            f"simulation: {agg.cycles} cycles over {len(sequences)} prompts, "
            f"{agg.generated_tokens} tokens generated"
        )

    return report


@torch.no_grad()
def _run_simulation(
    target_model,
    draft_model,
    sequences,
    sampling,
    gamma,
    device,
    max_new_tokens,
    seed,
    eos_token_id,
) -> SimulationResult:
    from .models import sequence_logits

    def target_fn(ids: torch.Tensor) -> torch.Tensor:
        return sequence_logits(target_model, ids, device)

    def draft_fn(ids: torch.Tensor) -> torch.Tensor:
        return sequence_logits(draft_model, ids, device)

    # probe the shared vocab once so the draft never proposes a token the target lacks
    probe = sequences[0]
    vocab = min(target_fn(probe).shape[-1], draft_fn(probe).shape[-1])

    total = SimulationResult(gamma = gamma)
    for i, ids in enumerate(sequences):
        total.extend(
            simulate_speculative(
                target_fn,
                draft_fn,
                ids,
                gamma = gamma,
                max_new_tokens = max_new_tokens,
                sampling = sampling,
                generator = torch.Generator().manual_seed(seed + i),  # reproducible per prompt
                eos_token_id = eos_token_id,
                vocab_size = vocab,
            )
        )
    return total
