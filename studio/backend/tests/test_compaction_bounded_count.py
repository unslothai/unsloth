# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A transcript many windows long still compacts when counting it whole would time out (#15800)."""

from __future__ import annotations

import json

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
        chars = sum(
            len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or []))
            for m in messages
        )
        tokens = int(chars / self.chars_per_token) + 4 * len(messages)
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


def test_a_transcript_proven_over_but_small_gets_its_exact_count():
    # Old whitespace the estimate overstates 24x, then a dense question: a slice proves it over,
    # but the whole is under twice the threshold, so it is counted exactly, since a refusal's
    # diagnosis subtracts exact counts from it.
    messages = [
        SYSTEM,
        {"role": "user", "content": "\n" * 60_000},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "dense " * 10_000},
    ]
    counted: list[int] = []

    def count(fitted):
        tokens = sum(
            len(m["content"]) // (4 if m["content"].startswith("dense") else 48) for m in fitted
        )
        counted.append(tokens)
        return tokens

    exact = count(messages)
    counted.clear()
    assert bounded_prompt_count(messages, count, 10_000) == exact
    assert len(counted) == 2  # the slice, then the whole


def test_an_over_proof_compares_like_the_exact_count():
    count = _Counter(6.0)
    messages = _transcript(400)
    bounded = bounded_prompt_count(messages, count, CONTEXT)
    assert bounded > CONTEXT
    assert max(count.counted) <= COUNT_LIMIT


def _tool_exchanges(count: int, result_chars: int) -> list[dict]:
    messages = []
    for index in range(count):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call{index}",
                        "type": "function",
                        "function": {"name": "python", "arguments": "{}"},
                    }
                ],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": f"call{index}", "content": "r" * result_chars}
        )
    return messages


QUESTION = {"role": "user", "content": "What is 2+2?"}
SYSTEM = {"role": "system", "content": "you are helpful"}
# Uneven turns: almost all of each transcript sits in one evictable turn.
UNEVEN = {
    "one huge old message": [
        SYSTEM,
        {"role": "user", "content": "x" * 1_200_000},
        {"role": "assistant", "content": "ok"},
        QUESTION,
    ],
    "one old turn of many tool calls": [
        SYSTEM,
        {"role": "user", "content": "run the job"},
        *_tool_exchanges(100, 12_000),
        {"role": "assistant", "content": "done"},
        QUESTION,
    ],
    "one old tool call with huge arguments": [
        SYSTEM,
        {"role": "user", "content": "write the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "write",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps({"text": "y" * 1_200_000}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "write", "content": "written"},
        {"role": "assistant", "content": "done"},
        QUESTION,
    ],
}


@pytest.mark.parametrize("shape", sorted(UNEVEN))
def test_rolling_fits_when_one_old_turn_holds_most_of_the_transcript(shape):
    count = _Counter(4.0)
    fitted, truncation = fit_rolling_context(
        UNEVEN[shape], context_length = CONTEXT, max_tokens = 64, count_tokens = count
    )
    assert truncation["fits"] is True
    assert fitted[-1] is QUESTION
    assert max(count.counted) <= COUNT_LIMIT


@pytest.mark.parametrize("shape", sorted(UNEVEN))
@pytest.mark.parametrize("force_reset", (False, True))
def test_checkpoint_resets_when_one_old_turn_holds_most_of_the_transcript(shape, force_reset):
    count = _Counter(4.0)
    fitted, truncation = fit_checkpoint_context(
        UNEVEN[shape],
        context_length = CONTEXT,
        max_tokens = 64,
        count_tokens = count,
        can_reset = True,
        searchable = True,
        force_reset = force_reset,
    )
    assert truncation["fits"] is True
    assert truncation["checkpoint_started"] is True
    assert fitted[-1] is QUESTION
    assert max(count.counted) <= COUNT_LIMIT


def test_checkpoint_reset_releases_a_long_turn_in_progress_without_counting_it_whole():
    # A reset first holds the whole turn in progress; counting that hold must not time out.
    history = [SYSTEM]
    for index in range(20):
        history.append({"role": "user", "content": f"q{index} " + "w " * 500})
        history.append({"role": "assistant", "content": "a " * 500})
    task = {"role": "user", "content": "Do the task."}
    messages = [*history, task, *_tool_exchanges(100, 12_000)]
    count = _Counter(4.0)
    _, truncation = fit_checkpoint_context(
        messages,
        context_length = CONTEXT,
        max_tokens = 64,
        count_tokens = count,
        can_reset = True,
        searchable = True,
    )
    assert truncation["fits"] is True
    assert truncation["checkpoint_started"] is True
    assert max(count.counted) <= COUNT_LIMIT


def test_a_newest_message_far_over_the_window_is_refused_not_raised():
    messages = [
        SYSTEM,
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "x" * 1_200_000},
    ]
    _, truncation = fit_rolling_context(
        messages, context_length = CONTEXT, max_tokens = 64, count_tokens = _Counter(4.0)
    )
    assert truncation["fits"] is False


def test_a_slice_the_template_refuses_falls_back_to_the_whole_count():
    messages = _transcript(40)
    whole = len(messages)

    def count(fitted):
        if len(fitted) != whole:
            raise RuntimeError("llama-server could not render the chat template")
        return 10**6

    assert bounded_prompt_count(messages, count, CONTEXT) == 10**6


def test_a_slice_opens_on_a_user_turn_and_only_removes_text():
    from core.inference.context_window import _newest_slice, estimate_message_tokens

    for messages in UNEVEN.values():
        sizes = [estimate_message_tokens(message) for message in messages]
        newest = _newest_slice(messages, sizes, 2 * CONTEXT)
        assert newest[0] is SYSTEM
        assert newest[1]["role"] == "user"
        assert newest[-1] is QUESTION
        assert sum(len(str(m.get("content"))) for m in newest) < sum(
            len(str(m.get("content"))) for m in messages
        )


def _template_counter(name: str):
    """Renders with a template from unsloth/chat_templates.py; 4 rendered characters a token."""
    import ast
    from pathlib import Path

    sandbox = pytest.importorskip("jinja2.sandbox")
    source = Path(__file__).resolve().parents[3] / "unsloth" / "chat_templates.py"
    template = next(
        ast.literal_eval(node.value)
        for node in ast.parse(source.read_text(encoding = "utf-8")).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name
    )
    env = sandbox.ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True)
    compiled = env.from_string(template)

    def count(messages):
        rendered = compiled.render(
            messages = messages, add_generation_prompt = True, bos_token = ""
        )
        return len(rendered) // 4

    return count


# Each template strips an old turn's reasoning between its own markers: Qwen3 splits on the
# closing </think>, Gemma-4 drops a section holding the opening <|channel>. A cut that lost either
# marker would render that reasoning and "prove" a 40-token chat over the window.
REASONING_TURNS = {
    "qwen3_template": "<think>" + "pondering " * 100_000 + "</think>Hello!",
    "gemma4_template": "<|channel>thought\n" + "pondering " * 100_000 + "<channel|>Hello!",
}


@pytest.mark.parametrize("template", sorted(REASONING_TURNS))
def test_a_slice_keeps_the_markers_a_template_strips_old_reasoning_by(template):
    count = _template_counter(template)
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": REASONING_TURNS[template]},
        QUESTION,
    ]
    exact = count(messages)
    assert exact < 100
    assert bounded_prompt_count(messages, count, CONTEXT) == exact
    fitted, truncation = fit_rolling_context(
        messages, context_length = CONTEXT, max_tokens = 64, count_tokens = count
    )
    assert truncation is None
    assert fitted == messages
    fitted, truncation = fit_checkpoint_context(
        messages,
        context_length = CONTEXT,
        max_tokens = 64,
        count_tokens = count,
        can_reset = True,
        searchable = True,
    )
    assert truncation is None
    assert fitted == messages


@pytest.mark.parametrize(
    "arguments",
    (
        # Numbers are not strings, so cutting only strings left this whole.
        {"data": list(range(200_000))},
        # Empty containers hold no string or number, so only charging their syntax bounds them.
        {"data": [{}] * 400_000},
    ),
    ids = ("numbers", "empty objects"),
)
def test_large_json_arguments_are_cut_to_a_size_not_just_their_strings(arguments):
    messages = [
        SYSTEM,
        {"role": "user", "content": "store the data"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "store",
                    "type": "function",
                    "function": {
                        "name": "store",
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "store", "content": "stored"},
        {"role": "assistant", "content": "done"},
        QUESTION,
    ]
    for fit in ("rolling", "checkpoint", "compact"):
        count = _Counter(4.0)
        if fit == "rolling":
            _, truncation = fit_rolling_context(
                messages, context_length = CONTEXT, max_tokens = 64, count_tokens = count
            )
        else:
            _, truncation = fit_checkpoint_context(
                messages,
                context_length = CONTEXT,
                max_tokens = 64,
                count_tokens = count,
                can_reset = True,
                searchable = True,
                force_reset = fit == "compact",
            )
        assert truncation["fits"] is True, fit
        assert max(count.counted) <= COUNT_LIMIT, fit


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
