# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prompt/generation speed for the safetensors path.

Transformers reports no timings, so the chat UI showed a prompt and generation speed
for GGUF and MLX but nothing for safetensors. These tests pin the measurement: the
prefill boundary is stamped once (at the first logits-processor call, not at every
decode step), the emitted object matches llama-server's ``timings`` shape, and an
unmeasurable rate is omitted rather than reported as zero.
"""

import pytest
import torch

from core.inference.generation_timing import (
    GenerationTimer,
    build_generation_timings,
    with_prefill_boundary_processor,
)
from core.inference.presence_penalty import _make_presence_penalty_processor

try:
    # core.inference.inference imports unsloth at module scope, which requires
    # unsloth_zoo. The dependency-light backend CI matrix job does not install it,
    # so the _record_generation_stats check runs only when the stack is importable.
    from core.inference.inference import InferenceBackend
except ImportError:
    InferenceBackend = None


def test_windows_are_none_until_generation_is_measured():
    timer = GenerationTimer()
    assert timer.prompt_ms is None
    assert timer.predicted_ms is None

    timer.start()
    assert timer.prompt_ms is None  # prefill has not produced logits yet

    timer.mark_prefill_end()
    assert timer.prompt_ms is not None
    assert timer.predicted_ms is None  # generation has not returned yet

    timer.finish()
    assert timer.predicted_ms is not None


def test_prefill_boundary_stamps_once_so_decode_steps_do_not_move_it():
    timer = GenerationTimer()
    timer.start()
    timer.mark_prefill_end()
    boundary = timer.prefill_ended_at
    for _ in range(5):
        timer.mark_prefill_end()
    assert timer.prefill_ended_at == boundary


def test_a_run_that_never_reached_prefill_reports_no_prompt_window():
    timer = GenerationTimer()
    timer.start()
    timer.finish()  # generate() raised before producing logits
    assert timer.prompt_ms is None
    assert timer.predicted_ms is None


def test_timings_carry_llama_server_field_names_and_rates():
    timings = build_generation_timings(
        prompt_n = 400,
        predicted_n = 50,
        prompt_ms = 200.0,
        predicted_ms = 2000.0,
    )
    assert timings["prompt_n"] == 400
    assert timings["prompt_ms"] == pytest.approx(200.0)
    assert timings["prompt_per_second"] == pytest.approx(2000.0)
    assert timings["prompt_per_token_ms"] == pytest.approx(0.5)
    assert timings["predicted_n"] == 50
    assert timings["predicted_ms"] == pytest.approx(2000.0)
    assert timings["predicted_per_second"] == pytest.approx(25.0)
    assert timings["predicted_per_token_ms"] == pytest.approx(40.0)
    assert timings["cache_n"] == 0


def test_unmeasured_split_reports_no_timings_at_all():
    assert (
        build_generation_timings(
            prompt_n = 10,
            predicted_n = 5,
            prompt_ms = None,
            predicted_ms = 12.0,
        )
        is None
    )


@pytest.mark.parametrize(
    "prompt_n, prompt_ms",
    [
        (0, 30.0),  # a run whose prompt length was never measured
        (10, 0.0),  # a window too short to have a rate
    ],
)
def test_unratable_prompt_window_omits_the_rate_instead_of_reporting_zero(prompt_n, prompt_ms):
    timings = build_generation_timings(
        prompt_n = prompt_n,
        predicted_n = 5,
        prompt_ms = prompt_ms,
        predicted_ms = 100.0,
    )
    assert "prompt_per_second" not in timings
    assert "prompt_per_token_ms" not in timings
    assert timings["predicted_per_second"] == pytest.approx(50.0)


def test_processor_stamps_the_boundary_and_keeps_the_penalty_processor():
    timer = GenerationTimer()
    timer.start()
    penalty = _make_presence_penalty_processor(1.0, prompt_len = 2)
    processors = with_prefill_boundary_processor(penalty, timer)
    assert len(processors) == 2

    input_ids = torch.tensor([[0, 1, 3]])
    scores = processors(input_ids, torch.zeros(1, 5))
    assert timer.prompt_ms is not None
    # the wrapped penalty still ran: the one distinct completion token lost 1.0
    assert scores[0, 3].item() == pytest.approx(-1.0)


def test_boundary_lands_after_the_prompt_forward_pass_in_a_real_generate():
    """The whole split rests on transformers calling the processor once per step,
    the first time with the prompt still unextended. Pinned against a real
    ``generate`` so a change in that contract fails here, not in a wrong tok/s."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size = 64, n_positions = 128, n_embd = 32, n_layer = 2, n_head = 2)
    ).eval()

    seen_lengths = []

    class _RecordingProcessor:
        def __call__(self, input_ids, scores):
            seen_lengths.append(int(input_ids.shape[1]))
            return scores

    prompt_len = 20
    timer = GenerationTimer()
    processors = with_prefill_boundary_processor(None, timer)
    processors.append(_RecordingProcessor())

    timer.start()
    outputs = model.generate(
        input_ids = torch.randint(0, 64, (1, prompt_len)),
        max_new_tokens = 8,
        do_sample = False,
        logits_processor = processors,
        pad_token_id = 0,
    )
    timer.finish()

    assert seen_lengths[0] == prompt_len  # first call sees the prompt alone: prefill
    assert seen_lengths == list(range(prompt_len, prompt_len + 8))
    timings = build_generation_timings(
        prompt_n = prompt_len,
        predicted_n = int(outputs.shape[1]) - prompt_len,
        prompt_ms = timer.prompt_ms,
        predicted_ms = timer.predicted_ms,
    )
    assert timings["predicted_n"] == 8
    assert timings["prompt_per_second"] > 0
    assert timings["predicted_per_second"] > 0


def test_processor_wraps_a_zero_penalty_run_that_has_no_processor_of_its_own():
    timer = GenerationTimer()
    timer.start()
    processors = with_prefill_boundary_processor(None, timer)
    scores = torch.zeros(1, 5)
    out = processors(torch.tensor([[0, 1, 3]]), scores)
    assert timer.prompt_ms is not None
    assert torch.equal(out, scores)


@pytest.mark.skipif(InferenceBackend is None, reason = "unsloth stack not installed")
def test_recorded_stats_carry_timings_only_when_a_run_was_timed():
    timer = GenerationTimer()
    timer.start()
    timer.mark_prefill_end()
    timer.finish()

    backend = InferenceBackend.__new__(InferenceBackend)
    InferenceBackend._record_generation_stats(
        backend,
        prompt_tokens = 64,
        completion_tokens = 16,
        max_new_tokens = 256,
        timer = timer,
    )
    stats = backend.last_generation_stats
    assert stats["usage"] == {"prompt_tokens": 64, "completion_tokens": 16, "total_tokens": 80}
    assert stats["timings"]["prompt_n"] == 64
    assert stats["timings"]["predicted_n"] == 16

    InferenceBackend._record_generation_stats(
        backend,
        prompt_tokens = 64,
        completion_tokens = 16,
        max_new_tokens = 256,
    )
    assert "timings" not in backend.last_generation_stats
