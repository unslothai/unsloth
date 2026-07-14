# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the per-session cost calculator: math against
``core/inference/pricing.py`` plus graceful degradation."""

import math

from core.inference.pricing import (
    ANTHROPIC_CACHE_5M_WRITE_MULT,
    ANTHROPIC_CACHE_1H_WRITE_MULT,
    ANTHROPIC_CACHE_READ_MULT,
    ANTHROPIC_FAST_MODE_MULT,
    ANTHROPIC_PRICING,
    OPENAI_CACHE_READ_MULT,
    OPENAI_CONTAINER_USD_PER_HOUR,
    OPENAI_PRICING,
    OPENAI_WEB_SEARCH_USD_PER_1K,
    calculate_cost,
    pricing_snapshot,
)


def _isclose(
    a,
    b,
    tol = 1e-6,
):
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


# ── Anthropic fast-mode 6x multiplier (Opus 4.6 / 4.7 only) ─────────


def test_anthropic_fast_mode_charges_6x_standard_opus():
    """6x on input + output when ``usage.speed == "fast"``.
    https://platform.claude.com/docs/en/build-with-claude/fast-mode"""
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-7",
        {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "speed": "fast",
        },
    )
    assert _isclose(out["input_usd"], 5.0 * ANTHROPIC_FAST_MODE_MULT)
    assert _isclose(out["output_usd"], 25.0 * ANTHROPIC_FAST_MODE_MULT)
    assert _isclose(out["total_usd"], 30.0 * ANTHROPIC_FAST_MODE_MULT)
    assert "(fast)" in out["model_priced"], out["model_priced"]


def test_anthropic_fast_mode_does_not_affect_standard_speed():
    """``speed: "standard"`` (or missing) keeps the base rates."""
    out_standard = calculate_cost(
        "anthropic",
        "claude-opus-4-7",
        {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "speed": "standard",
        },
    )
    out_missing = calculate_cost(
        "anthropic",
        "claude-opus-4-7",
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
    )
    assert _isclose(out_standard["total_usd"], out_missing["total_usd"])
    assert _isclose(out_standard["total_usd"], 30.0)


def test_anthropic_fast_mode_stacks_with_cache_read_multiplier():
    """Cache multipliers apply on top of fast-mode (per docs)."""
    base = ANTHROPIC_PRICING["claude-opus-4-7"]["input_per_mtok"]
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-7",
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 1_000_000,
            "speed": "fast",
        },
    )
    expected = base * ANTHROPIC_FAST_MODE_MULT * ANTHROPIC_CACHE_READ_MULT
    assert _isclose(out["cache_read_usd"], expected)


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
    # No 5m/1h split surfaced -> assume the default 5m pool.
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
    # Dated snapshot inherits canonical pricing via prefix-match.
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-7-20260712",
        {"input_tokens": 1_000_000, "output_tokens": 0},
    )
    assert out["priced"] is True
    assert _isclose(out["input_usd"], 5.0)


# ── OpenAI base math (gpt-5.5: 5/30 per MTok) ────────────────────────


def test_openai_gpt55_input_output_math():
    # Sub-272k stays in short-context tier ($5/$30).
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {"input_tokens": 200_000, "output_tokens": 50_000},
    )
    assert _isclose(out["input_usd"], 200_000 / 1_000_000.0 * 5.0)
    assert _isclose(out["output_usd"], 50_000 / 1_000_000.0 * 30.0)
    assert _isclose(out["total_usd"], 1.0 + 1.5)


def test_openai_cache_read_subtracted_from_input_at_discount():
    # OpenAI folds cached into input_tokens; subtract and re-bill at 0.1x.
    base = OPENAI_PRICING["gpt-5.5"]["input_per_mtok"]
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {
            "input_tokens": 100_000,
            "output_tokens": 0,
            "input_tokens_details": {"cached_tokens": 80_000},
        },
    )
    # 20k charged at full price, 80k charged at 0.1x
    assert _isclose(out["input_usd"], 20_000 / 1_000_000.0 * base)
    assert _isclose(out["cache_read_usd"], 80_000 / 1_000_000.0 * base * OPENAI_CACHE_READ_MULT)


def test_openai_billable_input_tokens_does_not_double_count_cache_read():
    # input_tokens already includes cached; don't double-count.
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {
            "input_tokens": 100_000,
            "output_tokens": 0,
            "input_tokens_details": {"cached_tokens": 80_000},
        },
    )
    assert out["billable_input_tokens"] == 100_000


def test_openai_dated_snapshot_inherits_canonical_pricing():
    # Dated snapshot inherits gpt-5.5 pricing via prefix-match.
    out = calculate_cost(
        "openai",
        "gpt-5.5-2026-04-23",
        {"input_tokens": 200_000, "output_tokens": 0},
    )
    assert out["priced"] is True
    assert _isclose(out["input_usd"], 200_000 / 1_000_000.0 * 5.0)


def test_openai_gpt54_family_uses_verified_prices():
    # Spot-check lower-tier rows that previously underbilled.
    cases = {
        # (input_tokens, expected_input_usd, expected_output_usd)
        "gpt-5.4": (200_000, 200_000 / 1_000_000.0 * 2.5, 200_000 / 1_000_000.0 * 15.0),
        "gpt-5.4-mini": (1_000_000, 0.75, 4.5),
        "gpt-5.4-nano": (1_000_000, 0.20, 1.25),
        "gpt-5.3-codex": (1_000_000, 1.75, 14.0),
    }
    for model, (in_tokens, exp_in, exp_out) in cases.items():
        out = calculate_cost(
            "openai",
            model,
            {"input_tokens": in_tokens, "output_tokens": in_tokens},
        )
        assert out["priced"] is True, model
        assert _isclose(out["input_usd"], exp_in), model
        assert _isclose(out["output_usd"], exp_out), model


def test_openai_unlisted_model_priced_false_not_zero_default():
    # o-series / gpt-4.5 are off the pricing page; drop rather than $0.
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


# ── canonical Anthropic 4.5 ids now resolve to a price ─────────────


def test_anthropic_canonical_4_5_ids_are_priced():
    # Pin the bare-id aliases (backend defaults reference these).
    cases = {
        "claude-opus-4-5": (5.0, 25.0),
        "claude-sonnet-4-5": (3.0, 15.0),
        "claude-haiku-4-5": (1.0, 5.0),
        # Opus 4.1 has the same problem.
        "claude-opus-4-1": (15.0, 75.0),
    }
    for model, (inp, outp) in cases.items():
        out = calculate_cost(
            "anthropic",
            model,
            {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        )
        assert out["priced"] is True, model
        assert _isclose(out["input_usd"], inp), model
        assert _isclose(out["output_usd"], outp), model


# ── OpenAI long-context tier crossover ──────────────────────────────


def test_openai_gpt55_short_context_under_272k_uses_base_rates():
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {"input_tokens": 100_000, "output_tokens": 5_000},
    )
    assert _isclose(out["input_usd"], 100_000 / 1_000_000.0 * 5.0)
    assert _isclose(out["output_usd"], 5_000 / 1_000_000.0 * 30.0)
    # No long-context marker on the model id when we stayed under.
    assert "long-context" not in out["model_priced"], out["model_priced"]


def test_openai_gpt55_long_context_crossover_uses_higher_rates():
    # >272k billable -> long-context tier on the whole turn.
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {"input_tokens": 300_000, "output_tokens": 10_000},
    )
    assert _isclose(out["input_usd"], 300_000 / 1_000_000.0 * 10.0)
    assert _isclose(out["output_usd"], 10_000 / 1_000_000.0 * 45.0)
    assert "long-context" in out["model_priced"], out["model_priced"]


def test_openai_gpt54_long_context_crossover():
    out = calculate_cost(
        "openai",
        "gpt-5.4",
        {"input_tokens": 500_000, "output_tokens": 20_000},
    )
    assert _isclose(out["input_usd"], 500_000 / 1_000_000.0 * 5.0)
    assert _isclose(out["output_usd"], 20_000 / 1_000_000.0 * 22.5)


def test_openai_gpt54_mini_has_no_long_context_tier():
    # Mini/nano/codex have no long-context tier; base rate always applies.
    out = calculate_cost(
        "openai",
        "gpt-5.4-mini",
        {"input_tokens": 500_000, "output_tokens": 0},
    )
    assert _isclose(out["input_usd"], 500_000 / 1_000_000.0 * 0.75)
    assert "long-context" not in out["model_priced"], out["model_priced"]


# ── OpenAI server-tool surcharges ──────────────────────────────────


def test_openai_web_search_charged_per_thousand():
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "openai_tool_use": {"web_search_requests": 250},
        },
    )
    assert _isclose(out["server_tools_usd"], 250 / 1_000.0 * OPENAI_WEB_SEARCH_USD_PER_1K)
    assert _isclose(out["total_usd"], 250 / 1_000.0 * OPENAI_WEB_SEARCH_USD_PER_1K)


def test_openai_container_hours_charged():
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "openai_tool_use": {"container_hours": 1.5},
        },
    )
    assert _isclose(out["server_tools_usd"], 1.5 * OPENAI_CONTAINER_USD_PER_HOUR)


def test_openai_tool_surcharges_added_to_total():
    # End-to-end: total must sum input + output + web_search + container.
    out = calculate_cost(
        "openai",
        "gpt-5.5",
        {
            "input_tokens": 100_000,
            "output_tokens": 5_000,
            "openai_tool_use": {
                "web_search_requests": 3,
                "container_hours": 0.25,
            },
        },
    )
    expected_input = 100_000 / 1_000_000.0 * 5.0
    expected_output = 5_000 / 1_000_000.0 * 30.0
    expected_tools = (
        3 / 1_000.0 * OPENAI_WEB_SEARCH_USD_PER_1K + 0.25 * OPENAI_CONTAINER_USD_PER_HOUR
    )
    assert _isclose(
        out["total_usd"],
        round(expected_input + expected_output + expected_tools, 6),
    )


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
    assert a["fast_mode_mult"] == ANTHROPIC_FAST_MODE_MULT
    assert "web_search_usd_per_1k" in a
    assert "code_execution_usd_per_hour" in a
    assert "models" in o and "gpt-5.5" in o["models"]
    assert o["cache_read_mult"] == OPENAI_CACHE_READ_MULT
    # OpenAI tool surcharge constants are exposed for the frontend.
    assert o["web_search_usd_per_1k"] == OPENAI_WEB_SEARCH_USD_PER_1K
    assert o["container_usd_per_hour"] == OPENAI_CONTAINER_USD_PER_HOUR
    # Long-context tier metadata travels with the model row.
    gpt55 = o["models"]["gpt-5.5"]
    assert gpt55["long_context_threshold"] == 272_000
    assert gpt55["long_context_input_per_mtok"] == 10.0
    assert gpt55["long_context_output_per_mtok"] == 45.0


# ── longest-prefix match: dated mini variant must not collide with the
#    shorter family prefix ──


def test_longest_prefix_match_wins_for_dated_mini_snapshot():
    """`gpt-5.4-mini-2026-...` inherits the mini rate, not the shorter
    `gpt-5.4` rate (longest prefix wins)."""
    out = calculate_cost(
        "openai",
        "gpt-5.4-mini-2026-04-23",
        {"input_tokens": 1_000_000, "output_tokens": 0},
    )
    assert out["priced"] is True
    # mini = 0.75/MTok, shorter gpt-5.4 = 2.5/MTok (>3x overcharge).
    assert _isclose(out["input_usd"], 0.75), out


def test_longest_prefix_match_wins_for_dated_pro_snapshot():
    out = calculate_cost(
        "openai",
        "gpt-5.5-pro-2026-04-23",
        {"input_tokens": 1_000_000, "output_tokens": 0},
    )
    assert out["priced"] is True
    # gpt-5.5-pro = 30/MTok vs gpt-5.5 = 5/MTok; longest wins.
    assert _isclose(out["input_usd"], 30.0), out


# ── accept both chat-style and Responses envelope shapes. ──


def test_openai_chat_style_usage_keys_priced_correctly():
    """Chat-style envelope (`prompt_tokens`/`completion_tokens`) must
    produce a non-zero cost (previously silently zeroed)."""
    out = calculate_cost(
        "openai",
        "gpt-5.4-mini",
        {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
    )
    # gpt-5.4-mini: 0.75 input + 4.5 output per MTok.
    assert _isclose(out["input_usd"], 0.75), out
    assert _isclose(out["output_usd"], 4.5), out


def test_input_tokens_preferred_when_both_keys_present():
    """Raw key wins when both envelope shapes are present."""
    out = calculate_cost(
        "openai",
        "gpt-5.4-mini",
        {
            "input_tokens": 2_000_000,
            "prompt_tokens": 5_000_000,
            "output_tokens": 0,
        },
    )
    # input_tokens=2M wins -> 2 * 0.75 = 1.50.
    assert _isclose(out["input_usd"], 1.50), out


def test_anthropic_chat_style_prompt_tokens_dedupes_cache_buckets():
    """Anthropic chat-style prompt_tokens already folds cache buckets;
    don't double-count billable input."""
    # 1M uncached + 200K cache_creation + 500K cache_read -> 1.7M folded.
    raw = calculate_cost(
        "anthropic",
        "claude-opus-4-7",
        {
            "input_tokens": 1_000_000,
            "cache_creation_input_tokens": 200_000,
            "cache_read_input_tokens": 500_000,
            "output_tokens": 0,
        },
    )
    chat = calculate_cost(
        "anthropic",
        "claude-opus-4-7",
        {
            "prompt_tokens": 1_700_000,
            "cache_creation_input_tokens": 200_000,
            "cache_read_input_tokens": 500_000,
            "completion_tokens": 0,
        },
    )
    # Both envelopes must price the same.
    assert _isclose(chat["input_usd"], raw["input_usd"]), (chat, raw)
    assert _isclose(chat["cache_write_usd"], raw["cache_write_usd"]), (chat, raw)
    assert _isclose(chat["cache_read_usd"], raw["cache_read_usd"]), (chat, raw)
    assert _isclose(chat["total_usd"], raw["total_usd"]), (chat, raw)
    assert chat["billable_input_tokens"] == raw["billable_input_tokens"], (chat, raw)


def test_openai_chat_style_prompt_tokens_keeps_cache_read_semantics():
    """OpenAI prompt_tokens includes cache_read like raw input_tokens."""
    raw = calculate_cost(
        "openai",
        "gpt-5.5",
        {
            "input_tokens": 1_000_000,
            "input_tokens_details": {"cached_tokens": 200_000},
            "output_tokens": 100_000,
        },
    )
    chat = calculate_cost(
        "openai",
        "gpt-5.5",
        {
            "prompt_tokens": 1_000_000,
            "cache_read_input_tokens": 200_000,
            "completion_tokens": 100_000,
        },
    )
    assert _isclose(chat["total_usd"], raw["total_usd"]), (chat, raw)


def test_openai_chat_style_envelope_reads_cache_from_prompt_tokens_details():
    """Chat-style envelope ships cached under prompt_tokens_details;
    calculator must honour both that and input_tokens_details."""
    base = OPENAI_PRICING["gpt-5.5"]["input_per_mtok"]
    raw = calculate_cost(
        "openai",
        "gpt-5.5",
        {
            "input_tokens": 100_000,
            "input_tokens_details": {"cached_tokens": 80_000},
            "output_tokens": 0,
        },
    )
    chat_style = calculate_cost(
        "openai",
        "gpt-5.5",
        {
            "prompt_tokens": 100_000,
            "prompt_tokens_details": {"cached_tokens": 80_000},
            "completion_tokens": 0,
        },
    )
    # Both envelopes must price identically.
    assert _isclose(chat_style["input_usd"], raw["input_usd"]), (chat_style, raw)
    assert _isclose(chat_style["cache_read_usd"], raw["cache_read_usd"]), (chat_style, raw)
    # 80k at 0.1x base, 20k at full.
    assert _isclose(
        chat_style["cache_read_usd"],
        80_000 / 1_000_000.0 * base * OPENAI_CACHE_READ_MULT,
    )


def test_explicit_zero_output_tokens_wins_over_stale_completion_tokens():
    """Explicit ``output_tokens: 0`` beats a stale ``completion_tokens``;
    the previous `or` fallback treated 0 as missing."""
    out = calculate_cost(
        "openai",
        "gpt-4o-mini",
        {
            "input_tokens": 100,
            "output_tokens": 0,
            # Stale chat-style mirror; must not bill against it.
            "completion_tokens": 50,
        },
    )
    assert out["billable_output_tokens"] == 0, out
    assert out["output_usd"] == 0.0, out
