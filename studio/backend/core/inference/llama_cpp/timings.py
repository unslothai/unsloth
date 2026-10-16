# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Usage and timing events from llama-server."""

import math

from .log import logger


def _backfill_usage_from_timings(usage, timings):
    """Synthesize ``usage`` from llama-server's ``timings`` when the
    OpenAI-style usage block is missing or reports zero tokens.

    The Unsloth chat UI computes generation t/s from
    ``meta.usage.completion_tokens / totalStreamTime``. llama-server always
    populates ``timings.predicted_n`` (true decoded count) and
    ``timings.prompt_n``, but the final SSE chunk's ``usage`` can be absent
    or zero on some server builds / streaming configs, making the UI fall
    back to wall-clock t/s and dilute speculative-decoding speedups.
    """
    if not timings:
        return usage
    if usage and usage.get("completion_tokens"):
        return usage
    predicted_n = timings.get("predicted_n")
    prompt_n = timings.get("prompt_n")
    if predicted_n is None and prompt_n is None:
        return usage
    out = dict(usage or {})
    if not out.get("completion_tokens") and predicted_n is not None:
        out["completion_tokens"] = predicted_n
    if not out.get("prompt_tokens") and prompt_n is not None:
        out["prompt_tokens"] = prompt_n
    out["total_tokens"] = int(out.get("prompt_tokens") or 0) + int(
        out.get("completion_tokens") or 0
    )
    return out


def _context_usage_event(usage, timings):
    """UI frame pricing the context one finished tool-loop pass leaves behind, or None.

    ``timings.prompt_n`` excludes the KV-cache hit (``cache_n``) while ``usage.prompt_tokens``
    includes it, so a prompt read from timings adds the hit back.
    """
    usage = usage or {}
    timings = timings or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or timings.get(
        "cache_n"
    )
    prompt = usage.get("prompt_tokens")
    if not prompt and timings.get("prompt_n") is not None:
        prompt = int(timings.get("prompt_n") or 0) + int(timings.get("cache_n") or 0)
    completion = usage.get("completion_tokens") or timings.get("predicted_n")
    if not prompt:
        return None
    prompt, completion = int(prompt), int(completion or 0)
    return {
        "type": "context_usage",
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "context_tokens": prompt + completion,
        "cached_tokens": int(cached or 0),
    }


def _llama_chunk_has_generated_output(data: dict) -> bool:
    if data.get("type") == "diffusion_frame":
        return True
    choices = data.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict) and any(
            value not in (None, "", []) for key, value in delta.items() if key != "role"
        ):
            return True
        if choice.get("text") not in (None, ""):
            return True
    return False


def _perf_callback_wants_timings(callback) -> bool:
    """A callback that only tracks prefill progress opts out of per-token timings."""
    return callback is not None and getattr(callback, "wants_timings", True)


def _report_live_llama_timings(callback, chunk) -> None:
    """Report request-scoped llama.cpp progress without altering the public stream."""
    if callback is None or not isinstance(chunk, dict):
        return
    timings = chunk.get("timings")
    sample = dict(timings) if isinstance(timings, dict) else {}
    # Live samples are not TTFT; terminal metadata retains prompt_ms for the fallback.
    sample.pop("prompt_ms", None)
    progress = chunk.get("prompt_progress")
    if isinstance(progress, dict):
        sample["prompt_progress"] = dict(progress)
        try:
            processed = max(0.0, float(progress.get("processed", 0)))
            cached = max(0.0, float(progress.get("cache", 0)))
            elapsed_ms = float(progress.get("time_ms", 0))
            prompt_n = max(0.0, processed - cached)
            if math.isfinite(elapsed_ms) and elapsed_ms > 0 and math.isfinite(prompt_n):
                sample.update(
                    prompt_n = prompt_n,
                    prompt_per_second = prompt_n / (elapsed_ms / 1000.0),
                )
        except (TypeError, ValueError, OverflowError):
            pass
    if getattr(callback, "needs_phase", True) and _llama_chunk_has_generated_output(chunk):
        sample["running_phase"] = "token_generation"
    if not sample:
        return
    try:
        callback(sample)
    except Exception:
        logger.debug("Ignoring API monitor throughput callback failure", exc_info = True)
