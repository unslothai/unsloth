# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LLM nodes and Chat's saved per-model settings: which value wins, what reaches the request, and that a flow saved
before the setting existed runs exactly as it did."""

import asyncio

from core.flows import engine as engine_mod
from core.flows import model_settings
from core.flows.engine import param_values
from core.flows.nodes import _common
from core.flows.specs import NODE_SPECS

GENERATE = next(s for s in NODE_SPECS if s["type"] == "llm.generate")
VISION = next(s for s in NODE_SPECS if s["type"] == "llm.vision")
REPO = "unsloth/Qwen3-4B-GGUF"
SAVED = {
    "temperature": 0.3,
    "topP": 0.7,
    "topK": 40,
    "minP": 0.05,
    "minPMode": "custom",
    "repetitionPenalty": 1.1,
    "presencePenalty": 0.5,
    "maxTokens": 2048,
    "systemPrompt": "Be brief.",
    "seed": 7,
}


def test_saved_settings_win_over_the_card_when_on():
    card = param_values(
        GENERATE, {"params": {"temperature": 1.5, "system": "card", "use_model_settings": True}}
    )
    out = model_settings.overlay(card, SAVED, GENERATE)
    assert out["temperature"] == 0.3 and out["system"] == "Be brief."
    assert (out["top_p"], out["top_k"], out["min_p"]) == (0.7, 40, 0.05)
    assert (out["repetition_penalty"], out["presence_penalty"], out["max_tokens"]) == (
        1.1,
        0.5,
        2048,
    )
    # The flow's seed stays the card's: Re-roll and the flow seed own it.
    assert out["seed"] is None


def test_unsaved_keys_keep_the_card_values():
    card = param_values(GENERATE, {"params": {"temperature": 1.5, "top_p": 0.9}})
    out = model_settings.overlay(card, {"temperature": 0.2}, GENERATE)
    assert out["temperature"] == 0.2 and out["top_p"] == 0.9 and out["max_tokens"] == 1024
    assert model_settings.overlay(card, None, GENERATE) is card


def test_server_min_p_and_max_output_are_not_copied():
    card = param_values(GENERATE, {"params": {"max_tokens": 900}})
    saved = {"minP": 0.2, "minPMode": "server-default", "maxTokens": 2**31 - 1}
    out = model_settings.overlay(card, saved, GENERATE)
    assert out["min_p"] is None and out["max_tokens"] == 900


def test_server_min_p_clears_the_cards_own():
    card = param_values(GENERATE, {"params": {"min_p": 0.3}})
    out = model_settings.overlay(card, {"minPMode": "server-default"}, GENERATE)
    assert out["min_p"] is None and "min_p" not in model_settings.sampling(out)
    assert model_settings.overlay(card, {"minPMode": "server-default"}, GENERATE, {"min_p"})["min_p"] == 0.3


def test_saved_values_are_clamped_to_the_node():
    out = model_settings.overlay(param_values(VISION, {"params": {}}), {"maxTokens": 20000}, VISION)
    assert out["max_tokens"] == 8192


def test_a_connected_input_beats_the_saved_value():
    card = param_values(GENERATE, {"params": {"temperature": 1.2}})
    out = model_settings.overlay(card, SAVED, GENERATE, {"temperature"})
    assert out["temperature"] == 1.2 and out["top_p"] == 0.7


def test_saved_entry_matches_the_model_id_case_insensitively():
    by_model = {"Unsloth/Qwen3-4B-GGUF": {"temperature": 0.1}, "other": {"temperature": 0.9}}
    assert model_settings.find_saved(by_model, REPO) == {"temperature": 0.1}
    assert model_settings.find_saved(by_model, "missing") is None
    assert model_settings.find_saved(None, REPO) is None


def test_an_old_flow_runs_unchanged():
    # Saved before these params existed: the toggle is off and no sampling key is sent.
    old = param_values(GENERATE, {"params": {"system": "s", "temperature": 0.2, "max_tokens": 300}})
    assert old["use_model_settings"] is False
    assert model_settings.sampling(old) == {}
    assert (old["system"], old["temperature"], old["max_tokens"]) == ("s", 0.2, 300)
    vision = param_values(VISION, {"params": {"max_tokens": 120}})
    assert (
        vision["temperature"] is None
        and vision["system"] == ""
        and not vision["use_model_settings"]
    )


class _Ctx:
    chat_model = None
    resident: dict = {}

    def on_cancel(self, fn):
        pass


def test_sampling_reaches_the_request_body():
    body = asyncio.run(
        _common._chat_body(
            _Ctx(),
            system = "",
            prompt = "p",
            temperature = 0.5,
            max_tokens = 8,
            images = None,
            thinking = False,
            seed = None,
            stream = False,
            sampling = model_settings.sampling({"top_p": 0.8, "top_k": 0, "min_p": None}),
        )
    )
    assert body["top_p"] == 0.8 and body["top_k"] == 0 and "min_p" not in body


def test_the_engine_reads_the_loaded_model_when_the_card_names_none(monkeypatch):
    seen = []

    async def fake_resident(ctx, kind):
        return REPO

    def fake_read(model_id):
        seen.append(model_id)
        return SAVED

    monkeypatch.setattr(_common, "resident_name", fake_resident)
    monkeypatch.setattr(model_settings, "read_saved", fake_read)
    params = param_values(GENERATE, {"params": {"use_model_settings": True}})
    asyncio.run(engine_mod._apply_saved_settings(_Ctx(), GENERATE, params, set()))
    assert seen == [REPO] and params["temperature"] == 0.3

    seen.clear()
    params = param_values(
        GENERATE,
        {"params": {"use_model_settings": True, "model": {"repo": "a/b", "variant": None}}},
    )
    asyncio.run(engine_mod._apply_saved_settings(_Ctx(), GENERATE, params, set()))
    assert seen == ["a/b"]
