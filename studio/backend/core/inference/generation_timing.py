# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prefill/decode timing for the safetensors generate paths.

Transformers reports no timings of its own, so the prompt and generation speeds the
chat UI reads off llama-server's ``timings`` object have to be measured here. The
split point is the first logits-processor call, which transformers makes once the
prefill forward pass has produced its logits, before the first token is sampled.

Reaching that callback only means the kernels were queued, so each stamp waits for the
device first. Without the wait a 2048-token prefill on an RTX 3080 reads as 28 ms
instead of 77 ms, inflating prompt throughput 2.7x and charging the rest to decode.

Kept in a dependency-light leaf module (torch + transformers only, no unsloth / peft)
so the arithmetic can be unit-tested without loading a model, matching
``core.inference.presence_penalty``.
"""

import time

import torch


def _wait_for_device(device):
    """Drain queued work so a wall-clock stamp reflects finished compute, not dispatch."""
    if device is None or device.type == "cpu":
        return
    synchronize = getattr(getattr(torch, device.type, None), "synchronize", None)
    if synchronize is None:
        return
    try:
        synchronize(device)
    except TypeError:  # torch.mps.synchronize takes no device argument
        synchronize()


class GenerationTimer:
    """Monotonic prefill/decode split around one ``model.generate()`` call."""

    def __init__(self):
        self.started_at = None
        self.prefill_ended_at = None
        self.ended_at = None
        self._device = None

    def start(self):
        self.started_at = time.monotonic()

    def mark_prefill_end(self, device = None):
        """Stamp the end of prefill; later decode steps must not move the boundary."""
        if self.started_at is None or self.prefill_ended_at is not None:
            return
        _wait_for_device(device)
        # latched for finish(), which has no tensor of its own to read a device off
        self._device = device
        self.prefill_ended_at = time.monotonic()

    def finish(self):
        if self.started_at is None or self.ended_at is not None:
            return
        _wait_for_device(self._device)
        self.ended_at = time.monotonic()

    @property
    def prompt_ms(self):
        """Prefill wall time, or None when generation never reached its first logits."""
        if self.started_at is None or self.prefill_ended_at is None:
            return None
        return max(0.0, (self.prefill_ended_at - self.started_at) * 1000.0)

    @property
    def predicted_ms(self):
        """Decode wall time, or None when the prefill boundary or the end is unknown."""
        if self.prefill_ended_at is None or self.ended_at is None:
            return None
        return max(0.0, (self.ended_at - self.prefill_ended_at) * 1000.0)


def with_prefill_boundary_processor(logits_processor, timer):
    """Prepend a prefill-boundary stamp to ``logits_processor`` (which may be None).

    The stamp runs first so an expensive penalty processor cannot be charged to prefill.
    """
    from transformers import LogitsProcessor, LogitsProcessorList

    class _PrefillBoundaryLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            # scores is the prefill output, so its device is the one to wait on
            timer.mark_prefill_end(scores.device)
            return scores

    processors = LogitsProcessorList([_PrefillBoundaryLogitsProcessor()])
    if logits_processor:
        processors.extend(logits_processor)
    return processors


def build_generation_timings(
    *,
    prompt_n,
    predicted_n,
    prompt_ms,
    predicted_ms,
    cached_n = 0,
):
    """Map a measured prefill/decode split onto the timings shape llama-server emits.

    Returns None when the split was never measured. A rate is omitted rather than
    reported as zero when its window or token count is empty, so the UI falls back to
    its client-side metrics instead of showing an invented speed.
    """
    if prompt_ms is None or predicted_ms is None:
        return None
    prompt_n = int(prompt_n or 0)
    predicted_n = int(predicted_n or 0)
    prompt_ms = float(prompt_ms)
    predicted_ms = float(predicted_ms)
    timings = {
        "prompt_n": prompt_n,
        "prompt_ms": prompt_ms,
        "predicted_n": predicted_n,
        "predicted_ms": predicted_ms,
        "cache_n": int(cached_n or 0),
    }
    if prompt_n > 0 and prompt_ms > 0:
        timings["prompt_per_token_ms"] = prompt_ms / prompt_n
        timings["prompt_per_second"] = prompt_n / (prompt_ms / 1000.0)
    if predicted_n > 0 and predicted_ms > 0:
        timings["predicted_per_token_ms"] = predicted_ms / predicted_n
        timings["predicted_per_second"] = predicted_n / (predicted_ms / 1000.0)
    return timings
