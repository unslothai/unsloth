# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The generated clip keeps its complete spoken text in the API reply."""

from studio.backend.tests.test_openai_audio_speech_route import _make_client


def test_audio_generate_answers_with_the_text_the_clip_speaks(monkeypatch):
    """The chat renders choices[0].message.content under the player and keeps it in
    history, so it has to be the spoken text itself, whole, not a status label cut at
    100 characters."""
    cli, _calls, _saved = _make_client(monkeypatch)
    text = (
        "This sentence is deliberately longer than one hundred characters so that a "
        "truncated label would show it. "
    ) * 2
    resp = cli.post(
        "/v1/audio/generate",
        json = {"model": "default", "messages": [{"role": "user", "content": text}]},
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == text
