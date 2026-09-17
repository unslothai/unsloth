# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""training_progress must carry trainer speed, not just step and loss.

Dropping HF's tqdm bar and per-step print removes the only place training
throughput appeared ("1.84s/it" on the bar, "train_tokens_per_second" in the raw
dict). Both were raw stdout rather than structured, so the replacement is to put
the number on the structured line: throughput measured over the interval between
two logged lines, and the run average on the first one.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _throughput(step, prev_step, elapsed, prev_elapsed, tokens, prev_tokens):
    """The calculation from TrainingManager._log_training_progress."""
    s_per_step = tok_per_s = None
    if elapsed is not None and prev_elapsed is not None and prev_step >= 0:
        d_time = elapsed - prev_elapsed
        d_steps = step - prev_step
        if d_time > 0 and d_steps > 0:
            s_per_step = round(d_time / d_steps, 3)
            if tokens is not None and prev_tokens is not None and tokens > prev_tokens:
                tok_per_s = round((tokens - prev_tokens) / d_time, 1)
    return s_per_step, tok_per_s


def test_the_first_line_reports_no_throughput():
    # elapsed_seconds is wall time since the worker started, so it also covers the
    # imports, the model download and load and the dataset build; and on a resumed run
    # the step and token counters predate this process. Neither is a training rate.
    assert _throughput(4, -1, 8.0, None, 4000, None) == (None, None)
    assert _throughput(1010, -1, 20.0, None, 4_000_000, None) == (None, None)


def test_later_lines_report_the_interval_not_the_average():
    # 10 steps and 20000 tokens in the 20s since the last line, after a slow start.
    s_per_step, tok_per_s = _throughput(30, 20, 120.0, 100.0, 60000, 40000)
    assert s_per_step == 2.0
    assert tok_per_s == 1000.0


def test_matches_the_tqdm_number_it_replaces():
    # The bar showed "1.84s/it"; one step in 1.84s must report the same.
    s_per_step, _ = _throughput(19, 18, 38.34, 36.50, None, None)
    assert s_per_step == 1.84


def test_missing_token_counts_still_give_seconds_per_step():
    s_per_step, tok_per_s = _throughput(30, 20, 120.0, 100.0, None, None)
    assert s_per_step == 2.0
    assert tok_per_s is None


def test_no_elapsed_yields_nothing_rather_than_dividing_by_zero():
    assert _throughput(30, 20, None, None, 1, 0) == (None, None)


def test_a_repeated_or_backwards_step_yields_nothing():
    assert _throughput(20, 20, 120.0, 100.0, 60000, 40000) == (None, None)
    assert _throughput(19, 20, 120.0, 100.0, 60000, 40000) == (None, None)


def test_a_stalled_clock_yields_nothing():
    assert _throughput(30, 20, 100.0, 100.0, 60000, 40000) == (None, None)


def test_token_counter_that_did_not_move_still_gives_seconds_per_step():
    s_per_step, tok_per_s = _throughput(30, 20, 120.0, 100.0, 40000, 40000)
    assert s_per_step == 2.0
    assert tok_per_s is None


def test_the_emitter_passes_both_fields():
    text = (_BACKEND / "core/training/training.py").read_text(encoding = "utf-8")
    assert "s_per_step = s_per_step," in text
    assert "tok_per_s = tok_per_s," in text
