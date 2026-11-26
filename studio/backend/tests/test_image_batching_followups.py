# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-family batch grouping through the route, with requests arriving interleaved: members that encode the same
way per call share one call even when they are not adjacent in the queue, and members that do not never share."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import httpx
import pytest

from core.inference import image_request_batching as batching

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_image_concurrency import (  # noqa: E402,F401  (fixture)
    _batched_calls,
    _openai,
    _post_while_rendering,
    studio,
)


def _words(text, **_kwargs):
    return {"input_ids": list(range(len(text.split())))}


class _ClipWords:
    """A CLIP-shaped tokenizer: one id per word, bos / eos."""

    bos_token_id, eos_token_id, pad_token_id = 0, 1, None

    def get_vocab(self):
        return {",</w>": 2}

    def __call__(self, text, **_kwargs):
        return {"input_ids": [10 + len(word) for word in text.split()]}


def _plan(backend, family, pipe):
    """The real engine's coalesce_plan over a stand-in load state."""
    from core.inference.diffusion import DiffusionBackend

    backend._state = types.SimpleNamespace(family = family, pipe = pipe)
    backend.coalesce_plan = types.MethodType(DiffusionBackend.coalesce_plan, backend)


def _qwen21_pipe():
    transformer = type("QwenImage21Transformer2DModel", (), {})()
    return types.SimpleNamespace(
        transformer = transformer, prompt_template_t2i = "{}", tokenizer = _words
    )


def _groups(backend) -> list[list[str]]:
    return sorted(sorted(call["prompts"]) for call in _batched_calls(backend))


def _alone(backend) -> list[str]:
    return [call["prompt"] for call in backend.calls if call["prompts"] is None]


def _interleaved(*prompts):
    accounts = ("alice", "bob", "carol", "dave", "erin", "frank")
    return [
        ("/v1/images/generations", account, _openai(prompt))
        for account, prompt in zip(accounts, prompts)
    ]


def test_qwen_image_21_never_puts_two_lengths_in_one_call(studio):
    app, backend, _saved = studio
    _plan(backend, types.SimpleNamespace(name = "qwen-image-2.1"), _qwen21_pipe())
    prompts = ("a fox", "a fox in snow", "a car", "a car on roads", "an owl", "an owl at night")
    responses = _post_while_rendering(app, backend, _interleaved(*prompts), waiting = 6)
    assert all(r.status_code == 200 for r in responses)
    for call in _batched_calls(backend):
        assert len({len(p.split()) for p in call["prompts"]}) == 1, call["prompts"]
    assert sum(len(call["prompts"]) for call in _batched_calls(backend)) == 6


def test_minimax_h3_image_renders_each_request_at_once_without_opening_a_batch(studio):
    """Its pipeline loops one image at a time, so a batch gains nothing: no request waits to coalesce."""
    app, backend, _saved = studio
    _plan(backend, types.SimpleNamespace(name = "minimax-h3-image"), None)
    opened: list[int] = []

    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            backend.rendering = True
            out = []
            for path, account, body in _interleaved("a fox", "a cat", "an owl"):
                # Finishes while another render still holds the slot: it never waited to join a batch.
                response = await asyncio.wait_for(
                    client.post(path, json = body, headers = {"x-test-account": account}),
                    timeout = 5,
                )
                opened.append(len(batching._OPEN))
                out.append(response)
            backend.rendering = False
            return out

    responses = asyncio.run(run())
    assert [r.status_code for r in responses] == [200, 200, 200]
    assert opened == [0, 0, 0]
    assert _batched_calls(backend) == []
    assert _alone(backend) == ["a fox", "a cat", "an owl"]


def test_flux1_members_share_a_call_only_within_one_t5_bucket_when_interleaved(studio):
    """A solo short prompt pads T5 to 256 and a long one to 512, so a short member next to a long one would
    not match its solo render. Interleaved short / long requests group by bucket."""
    from core.inference.diffusion_text_length import flux_t5_sequence_length

    app, backend, _saved = studio
    pipe = types.SimpleNamespace(tokenizer_2 = _words)
    _plan(backend, types.SimpleNamespace(name = "flux.1"), pipe)
    short1, short2 = "a red fox", "a blue car on a road"
    long1, long2 = " ".join(["fox"] * 300), " ".join(["car"] * 400)
    assert (
        flux_t5_sequence_length(_words, [short1])
        == flux_t5_sequence_length(_words, [short2])
        == 256
    )
    assert (
        flux_t5_sequence_length(_words, [long1]) == flux_t5_sequence_length(_words, [long2]) == 512
    )
    responses = _post_while_rendering(
        app, backend, _interleaved(short1, long1, short2, long2), waiting = 4
    )
    assert all(r.status_code == 200 for r in responses)
    assert _groups(backend) == [sorted([short1, short2]), sorted([long1, long2])]
    assert _alone(backend) == []


def test_sdxl_a_long_prompt_never_shares_a_call_with_short_ones(studio, monkeypatch):
    """Over 75 tokens SDXL encodes two CLIP chunks and the batch would pad every member to two: a long member
    renders with other long ones (or alone), never beside a short one."""
    import core.inference.diffusion_embeddings as embeddings

    monkeypatch.setattr(embeddings, "list_embeddings", lambda kind = None: [])
    app, backend, _saved = studio
    pipe = types.SimpleNamespace(tokenizer = _ClipWords(), text_encoder = object())
    _plan(backend, types.SimpleNamespace(name = "sdxl", textual_inversion = "sdxl"), pipe)
    long_prompt = " ".join(["word"] * 80)
    short1, short2, short3 = "a red fox", "a blue car", "an owl at night"
    responses = _post_while_rendering(
        app, backend, _interleaved(short1, long_prompt, short2, short3), waiting = 4
    )
    assert all(r.status_code == 200 for r in responses)
    assert _groups(backend) == [sorted([short1, short2, short3])]
    assert _alone(backend) == [long_prompt]


@pytest.mark.parametrize("words", [76, 150])
def test_sdxl_two_long_prompts_with_the_same_chunk_count_still_batch(studio, monkeypatch, words):
    import core.inference.diffusion_embeddings as embeddings

    monkeypatch.setattr(embeddings, "list_embeddings", lambda kind = None: [])
    app, backend, _saved = studio
    pipe = types.SimpleNamespace(tokenizer = _ClipWords(), text_encoder = object())
    _plan(backend, types.SimpleNamespace(name = "sdxl", textual_inversion = "sdxl"), pipe)
    long1, long2 = " ".join(["fox"] * words), " ".join(["car"] * words)
    responses = _post_while_rendering(
        app, backend, _interleaved(long1, "a red fox", long2), waiting = 3
    )
    assert all(r.status_code == 200 for r in responses)
    assert _groups(backend) == [sorted([long1, long2])]
    assert _alone(backend) == ["a red fox"]
