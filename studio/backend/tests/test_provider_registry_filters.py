# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Provider registry model-id filter regression tests.

The OpenAI ``model_id_allowlist`` previously hardcoded the gpt-5.3/4/5
families plus gpt-4.5 / o3 -- silently dropping every future family
OpenAI shipped. Anthropic's ``model_id_denylist`` previously stripped
every dated id, hiding the canonical names of every pre-4.6 model
(Opus 4.5, Sonnet 4.5, Haiku 4.5, Opus 4.1, the 4.0 family).

These tests pin the new denylist (OpenAI) and the empty denylist
(Anthropic) by walking realistic ``/v1/models`` listings through
``PROVIDER_REGISTRY`` and asserting the surviving set. The OpenAI bar is
"servable on /v1/responses with stream: true", not merely "chat model" --
Studio has no other OpenAI transport.
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


# ── OpenAI: denylist drops what /v1/responses cannot serve ─────────


def test_openai_keeps_every_known_chat_family():
    live = [
        # Current generation (must survive).
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.4-pro",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.3-codex",
        "gpt-5.3-chat-latest",
        "o3",
        "o3-pro",
        "o3-mini",
        # Hypothetical future families that the old allowlist would have
        # silently dropped -- they MUST surface under the new denylist.
        "gpt-5.6",
        "gpt-5.6-mini",
        "gpt-6",
        "gpt-6-pro",
        "o4",
        "o4-pro",
        "o5",
    ]
    surviving = _apply("openai", live)
    assert surviving == live, surviving


def test_openai_drops_families_that_are_not_on_the_responses_endpoint():
    # Studio serves every OpenAI turn from /v1/responses, so a model whose
    # page marks that endpoint Not supported is unusable however
    # chat-capable it is over /v1/chat/completions.
    dropped = _apply(
        "openai",
        [
            "gpt-audio",
            "gpt-audio-1.5",
            "gpt-audio-mini",
            "gpt-4o-audio-preview",
            "gpt-4o-mini-audio-preview",
            "gpt-realtime",
            "gpt-realtime-mini",
            "gpt-realtime-1.5",
            "gpt-4o-realtime-preview",
            "gpt-4o-mini-realtime-preview",
            "gpt-4o-search-preview",
            "gpt-4o-mini-search-preview",
            "o1-mini",
            "o1-preview",
            "gpt-4o-transcribe",
            "gpt-4o-mini-transcribe",
        ],
    )
    assert dropped == [], dropped


def test_openai_drops_deep_research_models():
    # Deep research rejects a request with no data source; a default Studio
    # turn sends no tools.
    dropped = _apply(
        "openai",
        ["o3-deep-research", "o4-mini-deep-research", "gpt-6-deep-research"],
    )
    assert dropped == [], dropped


def test_openai_drops_non_chat_ids():
    noise = [
        # Embeddings / TTS / image / moderation / whisper / etc.
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
        "text-moderation-latest",
        "text-moderation-stable",
        "tts-1",
        "tts-1-hd",
        "gpt-4o-tts",
        "whisper-1",
        "dall-e-2",
        "dall-e-3",
        "gpt-image-1",
        "gpt-image-2",
        "gpt-image-1-mini",
        "chatgpt-image-latest",
        "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe",
        "gpt-4o-mini-tts",
        "omni-moderation-latest",
        # Standalone search API endpoint.
        "gpt-5-search-api",
        "gpt-5-search-api-2025-10-14",
        # Video generation.
        "sora-2",
        "sora-2-pro",
        # Computer-use is an agentic harness, not a chat id.
        "computer-use-preview",
        # Legacy bases and the first-generation embedding / search /
        # similarity line.
        "babbage-002",
        "davinci-002",
        "text-davinci-003",
        "text-curie-001",
        "text-ada-001",
        "text-similarity-ada-001",
        "text-search-ada-doc-001",
        "text-search-curie-query-001",
        "code-search-ada-code-001",
        "code-davinci-002",
        "code-cushman-001",
        # Fine-tunes.
        "ft:gpt-4o-mini:acme:abc:xyz",
        # Dated snapshots are still hidden.
        "gpt-4o-2024-08-06",
        "gpt-4o-mini-2024-07-18",
        "gpt-5.5-2026-04-23",
    ]
    surviving = _apply("openai", noise)
    assert surviving == [], surviving


def test_openai_realtime_translate_variants_are_dropped():
    dropped = _apply(
        "openai",
        [
            "gpt-realtime-translate",
            "gpt-realtime-translate-mini",
            "gpt-audio-translate",
            "gpt-4o-realtime-translate-preview",
        ],
    )
    assert dropped == [], dropped


def test_openai_legacy_instruct_completion_ids_are_dropped():
    """Legacy `*-instruct` completion-only ids (gpt-3.5-turbo-instruct
    and friends) speak /v1/completions, not chat/responses. Our OpenAI
    bridge only knows the chat/responses transport, so admitting them
    into the picker would 4xx every selection."""
    dropped = _apply(
        "openai",
        [
            "gpt-3.5-turbo-instruct",
            "gpt-3.5-turbo-instruct-0914",
            "davinci-002-instruct",
        ],
    )
    assert dropped == [], dropped


def test_openai_legacy_compact_snapshot_suffixes_are_dropped():
    """Legacy `-MMDD` snapshot suffixes (gpt-3.5-turbo-0125,
    gpt-4-0613, gpt-4-1106-preview, etc.) hide behind the canonical
    id which the listing also returns; surface only the canonical so
    users do not pick a deprecated snapshot by accident. The
    `-\\d{4}(?:-preview)?$` rule must not catch canonical ids whose
    minor version happens to be a year-like number (e.g. gpt-4.5,
    o3) -- those are tested as KEEP below."""
    dropped = _apply(
        "openai",
        [
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0613",
            "gpt-4-0314",
            "gpt-4-32k-0613",
            "gpt-4-1106-preview",
            "gpt-4-0125-preview",
        ],
    )
    assert dropped == [], dropped

    # Canonical chat ids that share a digit-heavy tail must survive.
    kept = _apply(
        "openai",
        [
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-4.5",
            "gpt-5.5",
            "gpt-5.5-mini",
            "gpt-5.5-pro",
            "o3",
        ],
    )
    assert set(kept) >= {
        "gpt-3.5-turbo",
        "gpt-4o",
        "gpt-4.5",
        "gpt-5.5",
        "gpt-5.5-mini",
        "gpt-5.5-pro",
        "o3",
    }, kept


def test_openai_legacy_completion_names_only_match_at_id_start():
    # Hypothetical future chat ids that happen to contain a legacy
    # completion-family name mid-string must NOT be dropped. The
    # `^(?:babbage|davinci|ada|curie)\b` anchor is what makes this safe.
    kept = _apply(
        "openai",
        [
            "gpt-7-davinci-edition",
            "gpt-7-ada-chat",
            "gpt-7-curie-pro",
            "gpt-7-babbage-mini",
            # `^(?:text|code)-` needs the hyphen, so the codex family stays.
            "codex-mini-latest",
        ],
    )
    assert kept == [
        "gpt-7-davinci-edition",
        "gpt-7-ada-chat",
        "gpt-7-curie-pro",
        "gpt-7-babbage-mini",
        "codex-mini-latest",
    ], kept
    # ...but the actual legacy-base ids stay dropped.
    dropped = _apply(
        "openai",
        ["babbage-002", "davinci-002", "text-davinci-003"],
    )
    assert dropped == [], dropped


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
