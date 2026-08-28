# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Several MLX replies through one decode, on a real model."""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    _METAL = mx.metal.is_available()
except Exception:
    pytest.skip("requires mlx", allow_module_level = True)

metal_only = pytest.mark.skipif(not _METAL, reason = "requires Apple Silicon Metal")

MODEL = "mlx-community/SmolLM2-135M-Instruct"
PROMPT = [{"role": "user", "content": "Name a colour and explain why."}]


@pytest.fixture(scope = "module")
def backend():
    from core.inference.mlx_inference import MLXInferenceBackend

    loaded = MLXInferenceBackend()
    loaded.load_model(MODEL)
    return loaded


def _request(**overrides):
    base = dict(messages = PROMPT, max_new_tokens = 24, temperature = 0.9)
    base.update(overrides)
    return base


def _drain(backend, requests, **kwargs):
    """Cumulative snapshots per row, plus the per-row stats latched at the end."""
    replies, _events, stats = _drain_events(backend, requests, **kwargs)
    return replies, stats


def _drain_events(backend, requests, **kwargs):
    """The same, keeping the interleaving: ``(row, None)`` marks a reply finished."""
    replies: dict[int, list[str]] = {row: [] for row in range(len(requests))}
    events = []
    for row, snapshot in backend.generate_chat_batch(requests, **kwargs):
        events.append((row, snapshot))
        if snapshot is not None:
            replies[row].append(snapshot)
    finished = [row for row, snapshot in events if snapshot is None]
    assert sorted(finished) == sorted(replies), "every row must report completion once"
    return replies, events, list(backend.last_batch_generation_stats)


@metal_only
def test_a_batch_costs_one_forward_per_token_not_one_per_reply(backend):
    calls = {"n": 0}
    model = backend._model
    original = type(model).__call__

    def counted(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    type(model).__call__ = counted
    try:
        calls["n"] = 0
        replies, stats = _drain(backend, [_request(seed = 200 + i) for i in range(4)])
        batched_forwards = calls["n"]
    finally:
        type(model).__call__ = original

    generated = sum(entry["usage"]["completion_tokens"] for entry in stats)
    assert generated >= 4 * 20
    assert (
        batched_forwards < generated / 2
    ), f"{batched_forwards} forwards for {generated} tokens across 4 replies"


@metal_only
def test_greedy_replies_match_what_each_would_have_produced_alone(backend):
    """The wiring check: same prompt, same cache, no sampling."""
    request = dict(
        messages = [{"role": "user", "content": "Say hello."}], temperature = 0.0, max_new_tokens = 24
    )
    alone = list(backend.generate_chat_response(**request))
    alone_stats = backend.last_generation_stats
    replies, stats = _drain(backend, [dict(request) for _ in range(2)])
    assert replies[0] == alone
    assert replies[1] == alone
    for entry in stats:
        assert entry["usage"]["completion_tokens"] == alone_stats["usage"]["completion_tokens"]
        assert entry["finish_reason"] == alone_stats["finish_reason"]


@metal_only
def test_an_out_of_range_logit_bias_id_cannot_corrupt_a_batch(backend):
    """MLX does no bounds checking on the gather these processors do."""
    plain = _request(seed = 21)
    stray = _request(seed = 21, logit_bias = {10**9: 5.0, -4: 3.0})
    _first, _ = _drain(backend, [plain, plain])
    opener = backend._tokenizer.encode(_first[0][-1])[0]
    real = _request(seed = 21, logit_bias = {int(opener): -100.0})

    replies, stats = _drain(backend, [plain, stray, stray, real])
    assert all(entry["usage"]["completion_tokens"] > 0 for entry in stats)
    assert replies[1] == replies[0]
    assert replies[2] == replies[0]
    assert replies[3] != replies[0]


def test_a_repeated_token_is_charged_the_penalty_it_earned():
    """The frequency processor counts occurrences in float32 and scales once."""
    import mlx.core as mx

    from core.inference.mlx_inference import _make_mlx_frequency_penalty_processor

    processor = _make_mlx_frequency_penalty_processor(0.3)
    logits = mx.zeros((1, 8), dtype = mx.float16)
    processor(mx.array([0, 1]), logits)  # latches the prompt length
    tokens = mx.array([0, 1] + [5] * 1000)
    charged = processor(tokens, logits)
    assert abs(float(charged[0, 5]) + 300.0) < 0.5, float(charged[0, 5])
    assert float(charged[0, 4]) == 0.0

