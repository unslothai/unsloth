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
"""Render an :class:`~unsloth.spec_decoding.measure.AcceptanceReport` for a terminal or as JSON."""

from __future__ import annotations

import json
import math

from .measure import AcceptanceReport

_RULE = "  " + "─" * 62


def to_json(report: AcceptanceReport, indent: int = 2) -> str:
    return json.dumps(report.to_dict(), indent = indent)


def to_text(
    report: AcceptanceReport,
    target: str = "target",
    draft: str = "draft",
) -> str:
    r = report
    lines = [
        "",
        f"  speculative-decoding acceptance  —  {draft}  ⇒  {target}",
        f"  sampling: {r.sampling}    gamma (draft block): {r.gamma}",
        _RULE,
        f"  distributional (teacher-forced on {r.scored_span})",
        f"    acceptance rate  alpha       {r.alpha:6.3f}   ± {r.alpha_std:.3f} (per-seq)",
        f"    scored positions             {r.scored_positions:,} over {r.n_sequences} sequences",
        f"    E[accepted draft tokens]     {r.accepted_draft_tokens:6.3f}  of {r.gamma}",
        f"    E[tokens per cycle]          {r.tokens_per_cycle:6.3f}",
        f"    est. speedup                 {r.est_speedup:6.2f}x   (cost_ratio={r.cost_ratio:g}, rough)",
    ]

    if r.simulated:
        lines += [
            _RULE,
            "  generative (real rejection sampling — the number to trust)",
            f"    acceptance rate  alpha       {r.empirical_alpha:6.3f}",
            f"    mean accepted length         {r.empirical_mean_accepted_length:6.3f}  of {r.gamma}",
            f"    tokens per cycle             {r.empirical_tokens_per_cycle:6.3f}",
            f"    est. speedup                 {r.empirical_est_speedup:6.2f}x   (cost_ratio={r.cost_ratio:g}, rough)",
        ]
        if r.alpha_by_draft_offset:
            by_pos = "  ".join(
                f"{x:.2f}" if not math.isnan(x) else "  - " for x in r.alpha_by_draft_offset
            )
            lines.append(f"    alpha by draft offset 1..{r.gamma}  {by_pos}")
            if r.reached_by_draft_offset:
                counts = "  ".join(f"{n:>4d}" for n in r.reached_by_draft_offset)
                lines.append(f"      cycles reaching it        {counts}")
            lines.append("      (conditional: reaching offset j needs 0..j-1 accepted, so")
            lines.append("       later offsets are self-selected toward easy contexts)")

    if r.notes:
        lines.append(_RULE)
        for n in r.notes:
            lines.append(f"  note: {n}")
    lines.append("")
    return "\n".join(lines)
