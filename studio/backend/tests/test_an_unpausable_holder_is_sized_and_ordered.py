# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Holders the controller cannot pause: how big they are counted, and when they count as measured.

A raw holder measured before any residency reading must still be ordered against a probe
already in flight, a batch of short completion prompts is not nothing, and a GGUF speech call
occupies the same cache as the chats beside it.
"""

from __future__ import annotations

import asyncio
import contextlib

from core.inference.llama_preemption import ParticipantState, PreemptionController
import routes.inference as inference
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


def _controller() -> PreemptionController:
    made = PreemptionController("http://127.0.0.1:65081")
    made.configure(budget = 16384, kv_unified = True, slots = 4)
    return made


class TestTheFirstMeasurementIsOrdered:
    def test_a_probe_in_flight_before_any_reading_does_not_promote_it(self):
        controller = _controller()
        controller.register("raw", tokens = 6000, state = ParticipantState.STREAMING_RAW)
        in_flight = controller.residency_epoch()
        controller.note_measured("raw")
        # The probe that left before the prompt landed, reporting another holder's cells.
        controller.note_resident(9000, started_at_seq = in_flight)
        assert controller.participant("raw").measured is False, (
            "the reading predates the prompt; promoting on it folds the charge into cells "
            "the cache does not hold yet"
        )
        assert controller.snapshot().committed == 15000
        controller.note_resident(15000, started_at_seq = controller.residency_epoch())
        assert controller.participant("raw").measured is True
        assert controller.snapshot().committed == 15000


class TestABatchOfShortPromptsCounts:
    def test_each_string_rounds_up(self):
        assert inference._prompt_kv_tokens(["a", "b", "c"]) == 3
        assert inference._prompt_kv_tokens("abcd") == 1
        assert inference._prompt_kv_tokens("abcde") == 2
        assert inference._prompt_kv_tokens("") == 0
        assert inference._prompt_kv_tokens([[1, 2, 3], "ab"]) == 4


class TestAGgufSpeechCallIsCounted:
    def test_the_route_states_its_footprint(self, monkeypatch):
        seen = {}

        class _Llama:
            is_loaded = True
            _is_audio = True
            _audio_type = "snac"
            model_identifier = "some/tts-gguf"

            def generate_audio_response(self, **kwargs):
                seen["max_new_tokens"] = kwargs["max_new_tokens"]
                return b"RIFFfake", 24000

        @contextlib.contextmanager
        def _direct(counted = True, *, kv_tokens = 0):
            seen["counted"], seen["kv_tokens"] = counted, kv_tokens
            yield

        async def _noop_switch(*_args, **_kwargs):
            return None

        from models.inference import ChatCompletionRequest

        monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: _Llama())
        monkeypatch.setattr(inference, "_maybe_auto_switch_model", _noop_switch)
        monkeypatch.setattr(inference, "_direct_llama_request", _direct)
        monkeypatch.setattr(inference, "_llama_public_model_id", lambda backend: "some/tts-gguf")
        payload = ChatCompletionRequest(
            model = "some/tts-gguf", messages = [{"role": "user", "content": "hello there"}]
        )
        try:
            asyncio.run(
                inference._generate_tts_wav(
                    "hello there", payload, request = None, current_subject = "t"
                )
            )
        except Exception:
            pass
        assert seen.get("counted") is True
        assert seen["max_new_tokens"] > 0, "the speech call never ran"
        assert (
            seen["kv_tokens"] == inference._prompt_kv_tokens("hello there") + seen["max_new_tokens"]
        )
