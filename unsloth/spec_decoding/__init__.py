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
"""Measure speculative-decoding acceptance rate and expected speedup.

The acceptance rate is the objective function every draft-training method optimises:
``alpha = 1 - TV(p_target, p_draft)``. This package measures it for a given target/draft
pair so you can tell a working draft from a broken one before investing in training.

Quick start::

    from unsloth.spec_decoding import measure_acceptance, SamplingParams
    from unsloth.spec_decoding.models import load_model_and_tokenizer
    from unsloth.spec_decoding.data import build_sequences, load_texts

    target, tok, dev = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B")
    draft, _, _ = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B", device=dev)
    seqs = build_sequences(load_texts("builtin"), tok, max_samples=16)
    report = measure_acceptance(
        target, draft, seqs,
        sampling=SamplingParams(temperature=0.7), device=dev,
        continuation_tokens=64,   # score on the target's own output, not the prompt
        simulate=True,            # the number to trust
    )
    print(report.alpha, report.empirical_alpha, report.alpha_by_draft_offset)
"""

from __future__ import annotations

from .acceptance import (
    AcceptedLength,
    acceptance_prob,
    estimate_speedup,
    expected_accepted_length,
    residual_distribution,
)
from .measure import AcceptanceReport, measure_acceptance
from .sampling import SamplingParams, sampling_distribution, total_variation
from .simulate import SimulationResult, simulate_speculative

__version__ = "0.1.0"

__all__ = [
    "measure_acceptance",
    "AcceptanceReport",
    "SamplingParams",
    "sampling_distribution",
    "total_variation",
    "acceptance_prob",
    "residual_distribution",
    "expected_accepted_length",
    "estimate_speedup",
    "AcceptedLength",
    "simulate_speculative",
    "SimulationResult",
    "__version__",
]
