# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The API page's example request reads its sampling values from /sampling-defaults, so they have to
be the ones a /v1 request that omits them would get."""

import asyncio
import os
import sys

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from routes import inference as inference_routes
from utils.inference import inference_config


def test_route_answers_with_the_request_path_resolution():
    model = "unsloth/gemma-3-270m-it-GGUF"
    expected = inference_config.resolve_effective_sampling(model, {})
    got = asyncio.run(inference_routes.get_sampling_defaults(model = model, current_subject = "owner"))
    assert got == expected
    assert set(got) == set(inference_config.SAMPLING_FIELD_NAMES)


def test_an_operator_pin_wins_over_the_recommendation(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", "0.25")
    got = asyncio.run(
        inference_routes.get_sampling_defaults(
            model = "unsloth/Qwen3-4B-GGUF", current_subject = "owner"
        )
    )
    assert got["temperature"] == 0.25


class _LoadedQwen:
    is_loaded = True
    model_identifier = "unsloth/Qwen3-4B-GGUF"
    hf_variant = "Q4_K_M"
    supports_reasoning = True


def test_the_loaded_model_resolves_with_its_thinking_mode(monkeypatch):
    monkeypatch.setattr(inference_routes, "get_llama_cpp_backend", lambda: _LoadedQwen())
    seen = []
    monkeypatch.setattr(
        inference_routes,
        "_sampling_thinking_mode",
        lambda backend, payload: seen.append(payload) or False,
    )
    expected = inference_config.resolve_effective_sampling(
        "unsloth/Qwen3-4B-GGUF", {}, thinking = False
    )
    assert expected != inference_config.resolve_effective_sampling("unsloth/Qwen3-4B-GGUF", {})
    for name in ("unsloth/Qwen3-4B-GGUF", "unsloth/Qwen3-4B-GGUF:Q4_K_M"):
        got = asyncio.run(
            inference_routes.get_sampling_defaults(model = name, current_subject = "owner")
        )
        assert got == expected
    assert all(p.enable_thinking is None and p.reasoning_effort is None for p in seen)

    other = asyncio.run(
        inference_routes.get_sampling_defaults(
            model = "unsloth/gemma-3-270m-it-GGUF", current_subject = "owner"
        )
    )
    assert other == inference_config.resolve_effective_sampling("unsloth/gemma-3-270m-it-GGUF", {})
