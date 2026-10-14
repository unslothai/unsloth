# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prompt cache hits reach the clients that read them, on the Messages and Responses APIs."""

from __future__ import annotations

import json

from core.inference import anthropic_compat
from core.inference.anthropic_compat import anthropic_input_split, cached_prompt_tokens
from core.inference.engine_adapters import launch_arguments
from models.inference import ResponsesUsage

LLAMA_USAGE = {
    "prompt_tokens": 3568,
    "completion_tokens": 9,
    "prompt_tokens_details": {"cached_tokens": 3254},
}


def test_cache_reads_are_reported_apart_from_input_and_still_sum_to_the_prompt():
    assert anthropic_input_split(LLAMA_USAGE) == (314, 3254)
    delta = anthropic_compat._message_delta_usage(LLAMA_USAGE)
    assert delta["cache_read_input_tokens"] == 3254
    assert delta["input_tokens"] + delta["cache_read_input_tokens"] == 3568
    assert delta["cache_creation_input_tokens"] == 0


def test_a_server_that_says_nothing_about_its_cache_reports_as_before():
    usage = {"prompt_tokens": 100, "completion_tokens": 5}

    assert cached_prompt_tokens(usage) is None
    assert anthropic_compat._message_delta_usage(usage) == {
        "input_tokens": 100,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 5,
    }
    assert anthropic_compat._message_delta_usage(None)["input_tokens"] == 0


def test_a_cached_count_above_the_prompt_is_clamped():
    usage = {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 50}}

    assert anthropic_input_split(usage) == (0, 10)


def test_responses_usage_carries_cached_tokens_only_when_known():
    known = json.loads(
        ResponsesUsage(
            input_tokens = 3568, input_tokens_details = {"cached_tokens": 3254}
        ).model_dump_json()
    )
    assert known["input_tokens_details"] == {"cached_tokens": 3254}
    assert (
        json.loads(ResponsesUsage(input_tokens = 1).model_dump_json())["input_tokens_details"] is None
    )


def test_managed_engines_are_asked_to_report_their_cache_hits():
    for name, flag in (
        ("vllm", "--enable-prompt-tokens-details"),
        ("sglang", "--enable-cache-report"),
    ):
        args = launch_arguments(name, "python", "org/model", 8000, "key", 4096)
        assert flag in args, name
