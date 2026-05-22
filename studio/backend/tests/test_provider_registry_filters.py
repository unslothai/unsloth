# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Provider registry model-id filter regression tests.

The OpenAI ``model_id_allowlist`` previously hardcoded the gpt-5.3/4/5
families plus gpt-4.5 / o3 -- silently dropping every future family
OpenAI shipped. Anthropic's ``model_id_denylist`` previously stripped
every dated id, hiding the canonical names of every pre-4.6 model
(Opus 4.5, Sonnet 4.5, Haiku 4.5, Opus 4.1, the 4.0 family).

These tests pin the new non-chat denylist (OpenAI) and the empty
denylist (Anthropic) by walking realistic ``/v1/models`` listings
through ``PROVIDER_REGISTRY`` and asserting the surviving set.
"""

from core.inference.providers import PROVIDER_REGISTRY


def _apply(provider_type: str, candidate_ids: list[str]) -> list[str]:
    """Mirror the filter logic in ``routes/providers.list_models``."""
    info = PROVIDER_REGISTRY[provider_type]
    out = list(candidate_ids)
    allow = info.get("model_id_allowlist")
    if allow is not None:
        out = [m for m in out if allow.match(m)]
    deny = info.get("model_id_denylist")
    if deny is not None:
        out = [m for m in out if not deny.search(m)]
    return out


# ── OpenAI: non-chat denylist drops only non-chat ids ──────────────


def test_openai_keeps_every_known_chat_family():
    live = [
        # Current generation (must survive).
        "gpt-5.5", "gpt-5.5-pro",
        "gpt-5.4", "gpt-5.4-pro", "gpt-5.4-mini", "gpt-5.4-nano",
        "gpt-5.3-codex", "gpt-5.3-chat-latest",
        "o3", "o3-pro", "o3-mini", "o3-deep-research",
        # Hypothetical future families that the old allowlist would have
        # silently dropped -- they MUST surface under the new denylist.
        "gpt-5.6", "gpt-5.6-mini", "gpt-6", "gpt-6-pro",
        "o4", "o4-pro", "o5",
    ]
    surviving = _apply("openai", live)
    assert surviving == live, surviving


def test_openai_drops_non_chat_ids():
    noise = [
        # Embeddings / TTS / image / moderation / whisper / audio etc.
        "text-embedding-3-small", "text-embedding-3-large",
        "text-embedding-ada-002",
        "text-moderation-latest", "text-moderation-stable",
        "tts-1", "tts-1-hd", "gpt-4o-tts",
        "whisper-1",
        "dall-e-2", "dall-e-3",
        "gpt-image-1", "gpt-image-2", "gpt-image-1-mini",
        "chatgpt-image-latest",
        "gpt-audio-1.5", "gpt-realtime-2", "gpt-4o-realtime-preview",
        "gpt-4o-transcribe", "gpt-4o-search-preview",
        "gpt-4o-mini-search-preview", "gpt-4o-mini-transcribe",
        "gpt-4o-mini-tts",
        "omni-moderation-latest",
        # Video generation.
        "sora-2", "sora-2-pro",
        # Computer-use is an agentic harness, not a chat id.
        "computer-use-preview",
        # Legacy bases.
        "babbage-002", "davinci-002", "text-davinci-003",
        "text-curie-001", "text-ada-001",
        # Fine-tunes.
        "ft:gpt-4o-mini:acme:abc:xyz",
        # Dated snapshots are still hidden.
        "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18",
        "gpt-5.5-2026-04-23",
    ]
    surviving = _apply("openai", noise)
    assert surviving == [], surviving


# ── Anthropic: empty denylist; dated ids ARE canonical ───────────────


def test_anthropic_surfaces_every_live_model_including_dated_ids():
    # The full set of ids /v1/models returns today.
    live = [
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
    ]
    surviving = _apply("anthropic", live)
    assert surviving == live, surviving


def test_anthropic_default_models_match_filter():
    info = PROVIDER_REGISTRY["anthropic"]
    surviving = _apply("anthropic", list(info["default_models"]))
    assert surviving == list(info["default_models"]), surviving
