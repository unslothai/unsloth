# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the per-session cost calculator.

Pricing inputs are baked into ``core/inference/pricing.py``; this
test verifies the math (with multipliers from the prompt-caching
docs) and that unknown models / empty usage degrade gracefully.
"""

import math

from core.inference.pricing import (
    ANTHROPIC_CACHE_5M_WRITE_MULT,
    ANTHROPIC_CACHE_1H_WRITE_MULT,
    ANTHROPIC_CACHE_READ_MULT,
    ANTHROPIC_PRICING,
    OPENAI_CACHE_READ_MULT,
    OPENAI_PRICING,
    calculate_cost,
    pricing_snapshot,
)


def _isclose(a, b, tol = 1e-6):
    return math.isclose(a, b, rel_tol = tol, abs_tol = tol)


# ── unknown model -> priced=False, totals zero, tokens still report ──


def test_unknown_model_priced_false():
    out = calculate_cost(
        "anthropic",
        "made-up-model-9000",
        {"input_tokens": 100, "output_tokens": 50},
    )
    assert out["priced"] is False
    assert out["total_usd"] == 0.0
    assert out["billable_input_tokens"] == 100
    assert out["billable_output_tokens"] == 50


# ── Anthropic base math (Opus 4.7: 5/25 per MTok) ────────────────────


def test_anthropic_opus_4_7_input_and_output_math():
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-7",
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
    )
    assert _isclose(out["input_usd"], 5.0)
    assert _isclose(out["output_usd"], 25.0)
    assert _isclose(out["total_usd"], 30.0)


# ── Anthropic cache write 5m + read multipliers ──────────────────────


def test_anthropic_cache_5m_and_read_use_correct_multipliers():
    base = ANTHROPIC_PRICING["claude-opus-4-7"]["input_per_mtok"]
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-7",
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 1_000_000,
            "cache_read_input_tokens": 1_000_000,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 1_000_000,
                "ephemeral_1h_input_tokens": 0,
            },
        },
    )
    assert _isclose(out["cache_write_usd"], base * ANTHROPIC_CACHE_5M_WRITE_MULT)
    assert _isclose(out["cache_read_usd"], base * ANTHROPIC_CACHE_READ_MULT)
    # billable_input_tokens = input + cache_create + cache_read
    assert out["billable_input_tokens"] == 2_000_000


def test_anthropic_cache_1h_write_uses_2x_multiplier():
    base = ANTHROPIC_PRICING["claude-opus-4-7"]["input_per_mtok"]
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-7",
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 1_000_000,
            "cache_read_input_tokens": 0,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 0,
                "ephemeral_1h_input_tokens": 1_000_000,
            },
        },
    )
    assert _isclose(out["cache_write_usd"], base * ANTHROPIC_CACHE_1H_WRITE_MULT)


def test_anthropic_cache_5m_default_when_no_breakdown():
    # When the docs/response doesn't surface the 5m/1h split, treat
    # the full cache_creation bucket as 5m (the upstream default pool).
    base = ANTHROPIC_PRICING["claude-opus-4-7"]["input_per_mtok"]
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-7",
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 500_000,
        },
    )
    expected = 0.5 * base * ANTHROPIC_CACHE_5M_WRITE_MULT
    assert _isclose(out["cache_write_usd"], expected)


# ── Anthropic server-tool surcharges ────────────────────────────────


def test_anthropic_web_search_charged_per_thousand():
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-7",
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "server_tool_use": {"web_search_requests": 250},
        },
    )
    assert _isclose(out["server_tools_usd"], 2.5)  # $10/1000 * 250


def test_anthropic_code_exec_charged_per_hour():
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-7",
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "server_tool_use": {"code_execution_hours": 2.0},
        },
    )
    assert _isclose(out["server_tools_usd"], 0.10)  # $0.05/hr * 2


def test_anthropic_dated_id_falls_back_to_canonical_prefix():
    # Hypothetical dated snapshot of claude-opus-4-7 should still
    # inherit the canonical-id pricing via the prefix-match fallback.
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-7-20260712",
        {"input_tokens": 1_000_000, "output_tokens": 0},
    )
    assert out["priced"] is True
    assert _isclose(out["input_usd"], 5.0)


# ── OpenAI base math (gpt-5.5: 5/30 per MTok) ────────────────────────


def test_openai_gpt55_input_output_math():
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
    )
    assert _isclose(out["input_usd"], 5.0)
    assert _isclose(out["output_usd"], 30.0)
    assert _isclose(out["total_usd"], 35.0)


def test_openai_cache_read_subtracted_from_input_at_discount():
    # OpenAI folds cached tokens into input_tokens, unlike Anthropic.
    # The calculator must subtract cached_tokens from the "full price"
    # bucket and re-bill them at 0.1x.
    base = OPENAI_PRICING["gpt-5.5"]["input_per_mtok"]
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {
            "input_tokens": 1_000_000,
            "output_tokens": 0,
            "input_tokens_details": {"cached_tokens": 800_000},
        },
    )
    # 200k charged at full price, 800k charged at 0.1x
    assert _isclose(out["input_usd"], 0.2 * base)
    assert _isclose(out["cache_read_usd"], 0.8 * base * OPENAI_CACHE_READ_MULT)


def test_openai_billable_input_tokens_does_not_double_count_cache_read():
    # OpenAI's input_tokens already includes cached_tokens, so the
    # billable counter must NOT add cache_read on top -- otherwise the
    # tooltip says 1.8M input when the bill is for 1.0M.
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {
            "input_tokens": 1_000_000,
            "output_tokens": 0,
            "input_tokens_details": {"cached_tokens": 800_000},
        },
    )
    assert out["billable_input_tokens"] == 1_000_000


def test_openai_dated_snapshot_inherits_canonical_pricing():
    out = calculate_cost(
        "openai",
        "gpt-5.5-2026-04-23",
        {"input_tokens": 1_000_000, "output_tokens": 0},
    )
    assert out["priced"] is True
    assert _isclose(out["input_usd"], 5.0)


def test_openai_gpt54_family_uses_verified_prices():
    # Spot-check the lower-tier rows that previously underbilled.
    cases = {
        "gpt-5.4":      (2.5,  15.0),
        "gpt-5.4-mini": (0.75, 4.5),
        "gpt-5.4-nano": (0.20, 1.25),
        "gpt-5.3-codex": (1.75, 14.0),
    }
    for model, (inp, outp) in cases.items():
        out = calculate_cost(
            "openai",
            model,
            {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        )
        assert out["priced"] is True, model
        assert _isclose(out["input_usd"], inp), model
        assert _isclose(out["output_usd"], outp), model


def test_openai_unlisted_model_priced_false_not_zero_default():
    # o-series / gpt-4.5 are no longer on the pricing page, so we
    # intentionally drop them rather than silently underbill at $0.
    for model in ("o3", "o4-mini", "gpt-4.5", "gpt-4.5-preview"):
        out = calculate_cost(
            "openai",
            model,
            {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        )
        assert out["priced"] is False, model
        assert out["total_usd"] == 0.0, model
        # Token counts still report so the UI can render usage.
        assert out["billable_input_tokens"] == 1_000_000, model
        assert out["billable_output_tokens"] == 1_000_000, model


# ── snapshot endpoint includes the multipliers ───────────────────────


def test_snapshot_contains_provider_buckets_and_multipliers():
    snap = pricing_snapshot()
    assert set(snap.keys()) == {"anthropic", "openai"}
    a = snap["anthropic"]
    o = snap["openai"]
    assert "models" in a and "claude-opus-4-7" in a["models"]
    assert a["cache_5m_write_mult"] == ANTHROPIC_CACHE_5M_WRITE_MULT
    assert a["cache_1h_write_mult"] == ANTHROPIC_CACHE_1H_WRITE_MULT
    assert a["cache_read_mult"] == ANTHROPIC_CACHE_READ_MULT
    assert "web_search_usd_per_1k" in a
    assert "code_execution_usd_per_hour" in a
    assert "models" in o and "gpt-5.5" in o["models"]
    assert o["cache_read_mult"] == OPENAI_CACHE_READ_MULT
