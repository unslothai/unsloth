# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A transcript many windows long still compacts when counting it whole would time out (#15800)."""

from __future__ import annotations

import httpx
import pytest

from core.inference.checkpoint import fit_checkpoint_context
from core.inference.context_window import bounded_prompt_count, fit_rolling_context

CONTEXT = 16384
# Any single count past this fails, as llama-server's render and tokenize did past the 10 s timeout.
COUNT_LIMIT = 200_000


def _transcript(turns: int) -> list[dict]:
    messages = [{"role": "system", "content": "you are helpful"}]
    for index in range(turns):
        messages.append({"role": "user", "content": f"question {index} " + "word " * 400})
        messages.append({"role": "assistant", "content": f"answer {index} " + "word " * 2000})
    messages.append({"role": "user", "content": "What is 2+2?"})
    return messages


class _Counter:
    """A tokenizer at `chars_per_token`; the estimator assumes 4, so this sets its error."""

    def __init__(self, chars_per_token: float):
        self.chars_per_token = chars_per_token
        self.counted: list[int] = []

    def __call__(self, messages: list[dict]) -> int:
        tokens = int(
            sum(len(str(m.get("content") or "")) for m in messages) / self.chars_per_token
        ) + 4 * len(messages)
        self.counted.append(tokens)
        if tokens > COUNT_LIMIT:
            raise RuntimeError("llama-server timed out counting the prompt")
        return tokens


# Estimate overshoots 1.5x, undershoots 2x, overshoots 3x.
DENSITIES = (6.0, 2.0, 12.0)


@pytest.mark.parametrize("chars_per_token", DENSITIES)
def test_rolling_fits_a_transcript_too_long_to_count_whole(chars_per_token):
    count = _Counter(chars_per_token)
    fitted, truncation = fit_rolling_context(
        _transcript(400), context_length = CONTEXT, max_tokens = 64, count_tokens = count
    )
    assert truncation["fits"] is True
    assert truncation["dropped_messages"] > 0
    assert fitted[-1]["content"] == "What is 2+2?"
    assert truncation["prompt_tokens_before"] > CONTEXT
    assert max(count.counted) <= COUNT_LIMIT


@pytest.mark.parametrize("chars_per_token", DENSITIES)
@pytest.mark.parametrize("force_reset", (False, True))
def test_checkpoint_resets_a_transcript_too_long_to_count_whole(chars_per_token, force_reset):
    count = _Counter(chars_per_token)
    fitted, truncation = fit_checkpoint_context(
        _transcript(400),
        context_length = CONTEXT,
        max_tokens = 64,
        count_tokens = count,
        can_reset = True,
        searchable = True,
        force_reset = force_reset,
    )
    assert truncation["fits"] is True
    assert truncation["checkpoint_started"] is True
    assert fitted[-1]["content"] == "What is 2+2?"
    assert max(count.counted) <= COUNT_LIMIT


def test_a_transcript_a_few_windows_long_is_still_counted_exactly():
    count = _Counter(4.0)
    messages = _transcript(12)
    exact = count(messages)
    assert CONTEXT < exact < 4 * CONTEXT
    count.counted.clear()
    _, truncation = fit_rolling_context(
        messages, context_length = CONTEXT, max_tokens = 64, count_tokens = count
    )
    assert truncation["prompt_tokens_before"] == exact
    assert count.counted[0] == exact


def test_a_transcript_that_fits_despite_its_estimate_gets_its_exact_count():
    # The estimate puts this far over, but at 24 chars a token it fits: no slice proves otherwise.
    count = _Counter(24.0)
    messages = _transcript(8)
    exact = count(messages)
    assert bounded_prompt_count(messages, count, exact) == exact


def test_an_over_proof_compares_like_the_exact_count():
    count = _Counter(6.0)
    messages = _transcript(400)
    bounded = bounded_prompt_count(messages, count, CONTEXT)
    assert bounded > CONTEXT
    assert max(count.counted) <= COUNT_LIMIT


def _timing_out_client():
    class _Resp:
        status_code = 200

        def json(self):
            return {"prompt": "rendered"}

    class _Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def post(self, url, **_kwargs):
            if url.endswith("/apply-template"):
                return _Resp()
            raise httpx.ReadTimeout("timed out")

    return _Client


@pytest.mark.parametrize("strict", (True, False))
def test_a_count_timeout_is_not_reported_as_a_template_refusal(monkeypatch, strict):
    import core.inference.llama_cpp as llama_cpp

    class _Backend(llama_cpp.LlamaCppBackend):
        is_loaded = True
        base_url = "http://127.0.0.1:8080"
        _auth_headers: dict = {}

    monkeypatch.setattr(llama_cpp.httpx, "Client", _timing_out_client())
    backend = _Backend.__new__(_Backend)
    messages = [{"role": "user", "content": "hello"}]
    if strict:
        with pytest.raises(RuntimeError, match = "timed out counting the prompt"):
            backend.count_chat_tokens(messages, strict = True)
    else:
        assert backend.count_chat_tokens(messages) == 0
