# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Adversarial edge cases for ``calculate_cost`` / ``_lookup``: prefix
boundary, negative tokens, chat vs raw parity, long-context crossover
on billable count, and malformed sub-objects."""

import math

from core.inference.pricing import (
    ANTHROPIC_CACHE_5M_WRITE_MULT,
    ANTHROPIC_CACHE_READ_MULT,
    ANTHROPIC_PRICING,
    OPENAI_CACHE_READ_MULT,
    OPENAI_PRICING,
    _lookup,
    calculate_cost,
)


def _isclose(
    a,
    b,
    tol = 1e-6,
):
    return math.isclose(a, b, rel_tol = tol, abs_tol = tol)


# ── prefix-match boundary checks ────────────────────────────────────


def test_prefix_match_requires_dash_boundary_opus_variant():
    # `claude-opus-4-15` must not inherit `claude-opus-4-1` pricing; the next
    # char must be `-` or end-of-string.
    assert _lookup("anthropic", "claude-opus-4-15") is None
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-15",
        {"input_tokens": 1_000_000, "output_tokens": 0},
    )
    assert out["priced"] is False
    assert out["total_usd"] == 0.0


def test_prefix_match_requires_dash_boundary_gpt_variant():
    # Same dash-boundary invariant for OpenAI ids.
    assert _lookup("openai", "gpt-5.55") is None
    assert _lookup("openai", "gpt-5.55-2026-04-23") is None
    out = calculate_cost(
        "openai",
        "gpt-5.55-2026-04-23",
        {"input_tokens": 1_000_000, "output_tokens": 0},
    )
    assert out["priced"] is False


def test_prefix_match_requires_dash_boundary_pro_lookalike():
    # `gpt-5.5-prod` must fall through `gpt-5.5-pro` (6x overcharge) and land on
    # the canonical `gpt-5.5` row.
    prices = _lookup("openai", "gpt-5.5-prod")
    assert prices is not None
    assert (
        prices["input_per_mtok"] == OPENAI_PRICING["gpt-5.5"]["input_per_mtok"]
    ), "expected fallback to gpt-5.5 base ($5), not gpt-5.5-pro ($30)"
    out = calculate_cost(
        "openai",
        "gpt-5.5-prod",
        {"input_tokens": 100_000, "output_tokens": 0},
    )
    assert out["priced"] is True
    assert _isclose(out["input_usd"], 100_000 / 1_000_000.0 * 5.0)


def test_prefix_match_still_resolves_legit_dated_snapshots():
    # Boundary fix must not regress legit dated snapshots.
    out = calculate_cost(
        "openai",
        "gpt-5.4-mini-2026-04-23",
        {"input_tokens": 1_000_000, "output_tokens": 0},
    )
    assert out["priced"] is True
    assert _isclose(out["input_usd"], 0.75)

    # Anthropic dated snapshot still resolves to the canonical row.
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-7-20260414",
        {"input_tokens": 1_000_000, "output_tokens": 0},
    )
    assert out["priced"] is True
    assert _isclose(out["input_usd"], 5.0)


# ── precedence: input_tokens wins over prompt_tokens (and 0 is real) ──


def test_explicit_zero_input_tokens_wins_over_stale_prompt_tokens():
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {
            "input_tokens": 0,
            "prompt_tokens": 1_000_000,  # stale chat-style mirror
            "output_tokens": 100,
        },
    )
    assert out["billable_input_tokens"] == 0
    assert out["input_usd"] == 0.0


def test_none_input_tokens_falls_through_to_prompt_tokens():
    # `None` means "key present but unset"; chat-style mirror wins.
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {
            "input_tokens": None,
            "prompt_tokens": 200_000,
            "output_tokens": None,
            "completion_tokens": 5_000,
        },
    )
    assert out["billable_input_tokens"] == 200_000
    assert out["billable_output_tokens"] == 5_000
    assert _isclose(out["input_usd"], 200_000 / 1_000_000.0 * 5.0)
    assert _isclose(out["output_usd"], 5_000 / 1_000_000.0 * 30.0)


# ── negative / corrupted upstream values clamp to zero ──────────────


def test_negative_tokens_clamp_to_zero_no_negative_bill():
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {"input_tokens": -100, "output_tokens": -50},
    )
    assert out["billable_input_tokens"] == 0
    assert out["billable_output_tokens"] == 0
    assert out["input_usd"] == 0.0
    assert out["output_usd"] == 0.0
    assert out["total_usd"] == 0.0


def test_negative_cache_buckets_clamp_to_zero():
    # Negative cache_read on Anthropic would otherwise refund the bill.
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-7",
        {
            "input_tokens": 1_000,
            "output_tokens": 0,
            "cache_creation_input_tokens": -500,
            "cache_read_input_tokens": -1_000,
        },
    )
    assert out["cache_write_usd"] == 0.0
    assert out["cache_read_usd"] == 0.0
    assert out["billable_input_tokens"] == 1_000
    assert out["total_usd"] >= 0.0


def test_negative_prompt_tokens_chat_style_clamp():
    out = calculate_cost(
        "openai",
        "gpt-5.4-mini",
        {"prompt_tokens": -100, "completion_tokens": -50},
    )
    assert out["billable_input_tokens"] == 0
    assert out["billable_output_tokens"] == 0
    assert out["total_usd"] == 0.0


# ── cache_read > prompt_tokens corruption: no negative billable ─────


def test_anthropic_chat_cache_read_exceeds_prompt_no_negative_billable():
    # cache_read > prompt_tokens clamps uncached_input at 0; billable still
    # reflects cache buckets (we charge for what we got).
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-7",
        {
            "prompt_tokens": 100,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 500,
            "completion_tokens": 0,
        },
    )
    assert out["input_usd"] == 0.0  # uncached clamped to 0
    assert out["billable_input_tokens"] == 500  # 0 uncached + 500 cache_read
    # cache_read still priced at the discount rate.
    base = ANTHROPIC_PRICING["claude-opus-4-7"]["input_per_mtok"]
    assert _isclose(
        out["cache_read_usd"], 500 / 1_000_000.0 * base * ANTHROPIC_CACHE_READ_MULT
    )


def test_openai_raw_cached_tokens_exceeds_input_clamp_non_cached():
    # OpenAI variant: cached > input must not produce negative input_usd.
    base = OPENAI_PRICING["gpt-5.5"]["input_per_mtok"]
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {
            "input_tokens": 100,
            "output_tokens": 0,
            "input_tokens_details": {"cached_tokens": 500},
        },
    )
    assert out["input_usd"] == 0.0
    # Cache read still priced (the 0.1x bucket).
    assert _isclose(
        out["cache_read_usd"], 500 / 1_000_000.0 * base * OPENAI_CACHE_READ_MULT
    )


# ── long-context tier crosses on billable, including cache_creation ──


def test_openai_long_context_triggers_on_cache_creation_inflated_billable():
    # cache_creation pushes billable past 272k -> long-context tier must fire to
    # avoid undercounting.
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {
            "input_tokens": 250_000,
            "cache_creation_input_tokens": 50_000,
            "output_tokens": 1_000,
        },
    )
    assert out["billable_input_tokens"] == 300_000
    assert "long-context" in out["model_priced"]
    assert _isclose(out["input_usd"], 250_000 / 1_000_000.0 * 10.0)
    assert _isclose(out["output_usd"], 1_000 / 1_000_000.0 * 45.0)


def test_openai_long_context_threshold_boundary_inclusive():
    # Threshold is inclusive (>=).
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {"input_tokens": 272_000, "output_tokens": 1_000},
    )
    assert "long-context" in out["model_priced"]
    out_lo = calculate_cost(
        "openai",
        "gpt-5.5",
        {"input_tokens": 271_999, "output_tokens": 1_000},
    )
    assert "long-context" not in out_lo["model_priced"]


# ── chat-style vs raw envelope parity at OpenAI long-context tier ──


def test_openai_chat_envelope_long_context_parity_with_raw():
    raw = calculate_cost(
        "openai",
        "gpt-5.5",
        {"input_tokens": 300_000, "output_tokens": 10_000},
    )
    chat = calculate_cost(
        "openai",
        "gpt-5.5",
        {"prompt_tokens": 300_000, "completion_tokens": 10_000},
    )
    assert _isclose(chat["total_usd"], raw["total_usd"])
    assert "long-context" in chat["model_priced"]
    assert "long-context" in raw["model_priced"]


# ── malformed sub-objects: no crash, no false bill ──────────────────


def test_cache_creation_as_int_does_not_crash():
    # Proxies sometimes fold cache_creation to an int; tolerate it and fall back
    # to the 5m default.
    base = ANTHROPIC_PRICING["claude-opus-4-7"]["input_per_mtok"]
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-7",
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 1_000_000,
            "cache_creation": 12345,  # malformed; must not raise
        },
    )
    # Falls back to 5m default for the whole bucket.
    assert _isclose(
        out["cache_write_usd"],
        1_000_000 / 1_000_000.0 * base * ANTHROPIC_CACHE_5M_WRITE_MULT,
    )


def test_non_dict_server_tool_use_is_ignored():
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-7",
        {"input_tokens": 100, "output_tokens": 100, "server_tool_use": "garbage"},
    )
    assert out["server_tools_usd"] == 0.0

    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {"input_tokens": 100, "output_tokens": 100, "openai_tool_use": [1, 2, 3]},
    )
    assert out["server_tools_usd"] == 0.0


def test_non_dict_input_tokens_details_is_ignored():
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {
            "input_tokens": 100,
            "output_tokens": 0,
            "input_tokens_details": "nope",
            "prompt_tokens_details": [1, 2, 3],
        },
    )
    # No cached_tokens recovered -> no discount.
    assert out["cache_read_usd"] == 0.0


# ── unknown provider degrades gracefully ────────────────────────────


def test_unknown_provider_priced_false_zero_bill():
    out = calculate_cost(
        "gemini",
        "gemini-pro",
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
    )
    assert out["priced"] is False
    assert out["total_usd"] == 0.0
    # Tokens still report for the UI.
    assert out["billable_input_tokens"] == 1_000_000
    assert out["billable_output_tokens"] == 1_000_000


def test_anthropic_provider_with_openai_model_priced_false():
    # OpenAI id against Anthropic table must not falsely match.
    out = calculate_cost(
        "anthropic",
        "gpt-5.5",
        {"input_tokens": 1_000_000, "output_tokens": 0},
    )
    assert out["priced"] is False


# ── all-zero / empty usage stays at zero ────────────────────────────


def test_empty_usage_dict_zero_bill():
    out = calculate_cost("openai", "gpt-5.5", {})
    assert out["priced"] is True  # model is in the table
    assert out["billable_input_tokens"] == 0
    assert out["total_usd"] == 0.0


# ── Defense-in-depth: Anthropic prompt_tokens_details.cached_tokens ──


def test_anthropic_prompt_tokens_details_fallback_when_native_key_missing():
    """Chat-style envelope without `cache_read_input_tokens` but with mirrored
    `prompt_tokens_details.cached_tokens` should still apply the cache_read
    discount."""
    r = calculate_cost(
        provider = "anthropic",
        model = "claude-opus-4-7",
        usage = {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 0,
            # Mirrored shape only (no native key).
            "prompt_tokens_details": {"cached_tokens": 1_000_000},
            "cache_creation_input_tokens": 0,
        },
    )
    assert r["billable_input_tokens"] == 1_000_000, r
    # 1M cached at 0.1x of $5 (opus 4.7) = $0.50
    assert math.isclose(r["cache_read_usd"], 0.5, rel_tol = 1e-3), r


def test_anthropic_native_key_takes_precedence_over_mirrored():
    """When both native and mirrored cache-read fields are present, the native
    Anthropic field wins (mirror is fallback-only)."""
    r = calculate_cost(
        provider = "anthropic",
        model = "claude-opus-4-7",
        usage = {
            "prompt_tokens": 1_000_000,
            "cache_read_input_tokens": 800_000,
            "prompt_tokens_details": {"cached_tokens": 1_000_000},
            "cache_creation_input_tokens": 0,
        },
    )
    # billable = uncached_input + cache_creation + cache_read
    #         = (1M - 0 - 800k) + 0 + 800k = 1M
    assert r["billable_input_tokens"] == 1_000_000, r
    # cache_read uses 800k (native), not 1M (mirrored).
    assert math.isclose(r["cache_read_usd"], 0.4, rel_tol = 1e-3), r


def test_anthropic_native_zero_takes_precedence_over_mirrored():
    """Explicit `cache_read_input_tokens: 0` is authoritative; a stale mirrored
    block from a proxy must not inflate cache_read past it."""
    r = calculate_cost(
        provider = "anthropic",
        model = "claude-opus-4-7",
        usage = {
            "input_tokens": 1_000_000,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            # Stale mirror from a proxy; must be ignored (native present).
            "prompt_tokens_details": {"cached_tokens": 1_000_000},
        },
    )
    # Native is 0 -> cache_read stays 0.
    assert r["cache_read_usd"] == 0.0, r
    # billable = input + cache_creation + cache_read = 1M + 0 + 0
    assert r["billable_input_tokens"] == 1_000_000, r
    # 1M uncached at $5/M (no discount).
    assert math.isclose(r["input_usd"], 5.0, rel_tol = 1e-3), r
    assert math.isclose(r["total_usd"], 5.0, rel_tol = 1e-3), r


# ── _build_usage_chunk preserves cache_creation breakdown ──


def test_build_usage_chunk_forwards_anthropic_cache_creation_breakdown():
    """Chat-style envelope must carry the 5m/1h cache-write breakdown so
    downstream cost calc applies the 2x 1h premium."""
    import json
    from core.inference.external_provider import _build_usage_chunk

    chunk = _build_usage_chunk(
        completion_id = "cmpl-x",
        provider = "anthropic",
        last_usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 1_000_000,
            "cache_read_input_tokens": 0,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 250_000,
                "ephemeral_1h_input_tokens": 750_000,
            },
        },
    )
    assert chunk is not None
    payload = json.loads(chunk.split("data: ", 1)[1])
    cc = payload["usage"]["cache_creation"]
    assert cc["ephemeral_1h_input_tokens"] == 750_000, cc
    assert cc["ephemeral_5m_input_tokens"] == 250_000, cc


def test_calculate_cost_uses_forwarded_cache_creation_for_1h_premium():
    """Re-emitted chat envelope must price 1h cache writes at 2x base."""
    r = calculate_cost(
        provider = "anthropic",
        model = "claude-opus-4-7",
        usage = {
            "prompt_tokens": 1_000_010,
            "completion_tokens": 0,
            "cache_creation_input_tokens": 1_000_000,
            "cache_read_input_tokens": 0,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 0,
                "ephemeral_1h_input_tokens": 1_000_000,
            },
        },
    )
    # 1M at 1h-premium (2x of $5 = $10); 5m baseline would be $6.25.
    assert math.isclose(r["cache_write_usd"], 10.0, rel_tol = 1e-2), r
