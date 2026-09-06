# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The 31 planned cells published in issue #9861, as ground-truth labels.

These are somebody else's measurements on hardware we do not have (i5-12400F,
6 cores, RTX 6000 Ada 48 GB + RTX 3090 24 GB, n_ctx 32768 across 4 slots, flash
attention on, ``--load-mode none``). They are worth keeping because they are the
only side-by-side numbers against the placement Studio actually falls back to
when the planner abstains, and because a cost gate tuned on our own Colab matrix
would otherwise be tuned on the regime that produced the bug.

Arm naming follows the issue, which is the REVERSE of our bench notebooks:

    A = the planner's placement
    B = llama.cpp's own ``--fit on``

so ``pp_planner`` under ``pp_fit`` means the planner prefills slower.

Workload the numbers were taken at, needed to turn throughputs back into a
verdict: a unique 2311-2362 token prompt per request so the prompt cache cannot
serve it, and 128 generated tokens at temperature 0. ``WORKLOAD_PROMPT_TOKENS``
takes the midpoint.

The reporter has since attached a caveat to the generation column: n_ctx was
allocated at 32768 but only about 2.2K tokens were ever live, so generation was
measured where cache residency is worth least. That caveat is real and it is why
``breakeven_generated_tokens`` below is computed rather than asserted -- but note
it cannot rescue a cell whose generation is ALSO slower, and 27 of these 31 are.
"""

from __future__ import annotations

from dataclasses import dataclass

WORKLOAD_PROMPT_TOKENS = 2336.0
WORKLOAD_GENERATED_TOKENS = 128.0


@dataclass(frozen = True)
class ReportedCell:
    """One published row: what the planner did, and what it cost."""

    model: str
    gpu: str
    free_mib: int
    blocks_spilled: int
    blocks_total: int
    pp_planner: float
    pp_fit: float
    tg_planner: float
    tg_fit: float

    @property
    def label(self) -> str:
        return f"{self.model} @ {self.gpu} {self.free_mib} MiB"

    def total_seconds(
        self,
        n_prompt: float = WORKLOAD_PROMPT_TOKENS,
        n_generated: float = WORKLOAD_GENERATED_TOKENS,
    ) -> tuple[float, float]:
        """(planner, fit) wall seconds for one request of this shape.

        A single throughput column cannot say which arm a user would rather
        have; prefill and generation trade against each other and the exchange
        rate is the request shape. Both columns are folded into one number here
        so the label below is not an artefact of picking a favourite column.
        """
        planner = n_prompt / self.pp_planner + n_generated / self.tg_planner
        fit = n_prompt / self.pp_fit + n_generated / self.tg_fit
        return planner, fit

    def speedup(
        self,
        n_prompt: float = WORKLOAD_PROMPT_TOKENS,
        n_generated: float = WORKLOAD_GENERATED_TOKENS,
    ) -> float:
        """Above 1.0 means planning this cell beat letting ``--fit on`` place it."""
        planner, fit = self.total_seconds(n_prompt, n_generated)
        return fit / planner

    @property
    def breakeven_generated_tokens(self) -> float | None:
        """Generated tokens needed before the generation win repays the prefill loss.

        ``None`` when there is no such length: the planner is slower per output
        token as well, so decoding for longer only widens the gap. That is the
        case the short-sequence caveat on the issue cannot argue away, and it
        covers 27 of these 31 cells.
        """
        per_token_saved = 1.0 / self.tg_planner - 1.0 / self.tg_fit
        if per_token_saved >= 0.0:
            return None
        prefill_given_up = WORKLOAD_PROMPT_TOKENS * (1.0 / self.pp_fit - 1.0 / self.pp_planner)
        n = prefill_given_up / per_token_saved
        return n if n > 0.0 else 0.0


# Ordered as published: descending generation ratio, so the few cells the
# planner wins sit at the top and the collapse at the bottom is visible.
REPORTED_CELLS: tuple[ReportedCell, ...] = (
    ReportedCell("Llama-3.3-70B Q4", "Ada", 38016, 38, 80, 267.3, 309.1, 15.27, 7.68),
    ReportedCell("Qwen3.6-35B-A3B Q4", "3090", 9344, 37, 40, 118.1, 132.5, 15.66, 12.31),
    ReportedCell("Llama-3.3-70B Q4", "3090", 24448, 73, 80, 46.7, 57.4, 6.56, 5.96),
    ReportedCell("Qwen3.6-35B-A3B Q4", "Ada", 16768, 23, 40, 662.2, 839.2, 66.55, 65.01),
    ReportedCell("Qwen3.8-27B Q4", "3090", 14848, 64, 64, 108.0, 163.3, 4.35, 4.64),
    ReportedCell("Llama-3.3-70B Q4", "Ada", 47360, 17, 80, 445.6, 507.7, 34.28, 38.65),
    ReportedCell("Qwen3.8-27B Q4", "Ada", 14848, 64, 64, 338.2, 476.6, 3.96, 4.67),
    ReportedCell("Qwen3-32B Q4", "3090", 24320, 26, 64, 200.0, 300.4, 24.11, 28.61),
    ReportedCell("Qwen3.8-27B Q4", "3090", 18560, 38, 64, 159.3, 319.7, 8.00, 9.57),
    ReportedCell("Qwen3-32B Q4", "3090", 24064, 27, 64, 193.8, 301.0, 23.62, 29.77),
    ReportedCell("GLM-4.7-Flash Q4", "3090", 11008, 39, 46, 132.2, 178.0, 20.88, 26.60),
    ReportedCell("Qwen3.6-35B-A3B Q4", "Ada", 9344, 38, 40, 410.8, 475.8, 44.34, 56.53),
    ReportedCell("Qwen3-32B Q4", "Ada", 18560, 55, 64, 371.8, 525.2, 20.60, 26.72),
    ReportedCell("GLM-4.7-Flash Q4", "Ada", 11008, 40, 46, 470.4, 645.1, 63.33, 83.41),
    ReportedCell("Qwen3.8-27B Q4", "Ada", 18560, 41, 64, 469.5, 823.6, 7.26, 9.72),
    ReportedCell("Qwen3.6-35B-A3B Q4", "3090", 16768, 21, 40, 216.4, 239.4, 25.28, 34.49),
    ReportedCell("Qwen3.6-35B-A3B Q4", "Ada", 23296, 8, 40, 1536.6, 3142.7, 121.26, 165.49),
    ReportedCell("Qwen3.6-35B-A3B Q2", "3090", 7168, 39, 40, 197.2, 237.5, 19.76, 28.24),
    ReportedCell("Llama-3.3-70B Q4", "Ada", 25984, 70, 80, 169.5, 204.1, 2.88, 4.28),
    # "full" in the published table: every block spilled.
    ReportedCell("Qwen3.6-35B-A3B Q2", "Ada", 7168, 40, 40, 650.9, 816.1, 53.09, 84.43),
    ReportedCell("Qwen3-32B Q4", "3090", 18560, 53, 64, 111.0, 166.0, 11.92, 20.99),
    ReportedCell("GLM-4.7-Flash Q4", "Ada", 16768, 23, 46, 781.1, 1466.5, 97.41, 172.17),
    ReportedCell("Qwen3-32B Q4", "Ada", 24064, 30, 64, 587.8, 826.0, 45.01, 82.96),
    ReportedCell("GLM-4.7-Flash Q4", "3090", 16768, 21, 46, 240.3, 444.4, 34.55, 65.40),
    ReportedCell("Qwen3.6-35B-A3B Q2", "3090", 11136, 23, 40, 335.4, 442.9, 25.12, 58.57),
    ReportedCell("Qwen3.6-35B-A3B Q4", "3090", 23296, 7, 40, 526.8, 1257.3, 54.04, 133.04),
    ReportedCell("Qwen3.6-35B-A3B Q2", "Ada", 11136, 25, 40, 1028.0, 1409.6, 48.80, 132.94),
    ReportedCell("Qwen3-32B Q4", "Ada", 29440, 7, 64, 1255.2, 1743.5, 94.57, 318.62),
    ReportedCell("Qwen3.6-35B-A3B Q2", "Ada", 14592, 11, 40, 1695.0, 4055.8, 73.30, 305.52),
    ReportedCell("Qwen3-8B Q4", "3090", 11008, 22, 36, 662.2, 3348.3, 84.28, 447.67),
    ReportedCell("Qwen3-8B Q4", "Ada", 11008, 29, 36, 1818.2, 6194.0, 101.81, 858.68),
)

# The two cells a gate must not throw away. Everything else in the table is a
# loss at the published workload, so a gate that declined all 31 would score
# well on the losses and still be wrong.
WINNING_CELLS: tuple[ReportedCell, ...] = tuple(
    cell for cell in REPORTED_CELLS if cell.speedup() > 1.0
)
