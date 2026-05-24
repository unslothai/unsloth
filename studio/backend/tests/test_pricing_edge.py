# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Adversarial edge cases for ``calculate_cost`` / ``_lookup``.

The companion ``test_pricing.py`` module pins the happy-path math.
This module pins the corner cases reviewers asked about on the
pricing follow-up:

- prefix matching must land on a token boundary so a future
  ``claude-opus-4-15`` can't silently inherit ``claude-opus-4-1``
  pricing, and a hypothetical ``gpt-5.5-prod`` can't bill as the
  much more expensive ``gpt-5.5-pro``;
- negative / non-int token values from a corrupted upstream payload
  must never produce a negative bill that masks real spend in the
  running session total;
- chat-style and raw envelopes must produce identical totals on the
  cache-heavy boundary cases (zero cache_read, cache_read > prompt,
  cache_creation > prompt) so the cost ledger doesn't flip per
  envelope choice;
- the long-context tier crossover has to fire on the ``billable``
  input count (which folds cache_creation for OpenAI), not the raw
  ``input_tokens`` only, otherwise cache-heavy turns dodge the
  long-context premium they should pay;
- malformed sub-objects (cache_creation as int, server_tool_use as
  str) must not raise -- the calculator surfaces zero cost for the
  malformed bucket and keeps pricing the rest of the turn.
"""

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


def _isclose(a, b, tol = 1e-6):
    return math.isclose(a, b, rel_tol = tol, abs_tol = tol)


# ── prefix-match boundary checks ────────────────────────────────────


def test_prefix_match_requires_dash_boundary_opus_variant():
    # Hypothetical future ``claude-opus-4-15`` must NOT inherit the
    # ``claude-opus-4-1`` ($15 / $75) row by way of a naive substring
    # ``startswith`` -- the next char has to be ``-`` or end-of-string.
    assert _lookup("anthropic", "claude-opus-4-15") is None
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-15",
        {"input_tokens": 1_000_000, "output_tokens": 0},
    )
    assert out["priced"] is False
    assert out["total_usd"] == 0.0


def test_prefix_match_requires_dash_boundary_gpt_variant():
    # ``gpt-5.55`` is hypothetical but defends the same invariant:
    # a different model in the same family tree can't piggy-back on the
    # ``gpt-5.5`` row just because the canonical id is a string prefix.
    assert _lookup("openai", "gpt-5.55") is None
    assert _lookup("openai", "gpt-5.55-2026-04-23") is None
    out = calculate_cost(
        "openai",
        "gpt-5.55-2026-04-23",
        {"input_tokens": 1_000_000, "output_tokens": 0},
    )
    assert out["priced"] is False


def test_prefix_match_requires_dash_boundary_pro_lookalike():
    # ``gpt-5.5-pro`` and ``gpt-5.5-prod`` share an undelimited "prod"
    # / "pro" component. A naive prefix match would land
    # ``gpt-5.5-prod`` on the ``gpt-5.5-pro`` row ($30 / MTok) -- a 6x
    # overcharge against the canonical ``gpt-5.5`` row ($5 / MTok). The
    # dash-boundary check forces the longest matching key to end on
    # ``-``, so ``gpt-5.5-prod`` falls through ``gpt-5.5-pro`` and
    # lands on the canonical ``gpt-5.5`` row instead.
    prices = _lookup("openai", "gpt-5.5-prod")
    assert prices is not None
    assert prices["input_per_mtok"] == OPENAI_PRICING["gpt-5.5"]["input_per_mtok"], (
        "expected fallback to gpt-5.5 base ($5), not gpt-5.5-pro ($30)"
    )
    out = calculate_cost(
        "openai",
        "gpt-5.5-prod",
        {"input_tokens": 100_000, "output_tokens": 0},
    )
    assert out["priced"] is True
    assert _isclose(out["input_usd"], 100_000 / 1_000_000.0 * 5.0)


def test_prefix_match_still_resolves_legit_dated_snapshots():
    # The boundary fix must NOT regress the real win the PR is for.
    # gpt-5.4-mini-2026-04-23 inherits the mini row, NOT gpt-5.4's.
    out = calculate_cost(
        "openai",
        "gpt-5.4-mini-2026-04-23",
        {"input_tokens": 1_000_000, "output_tokens": 0},
    )
    assert out["priced"] is True
    assert _isclose(out["input_usd"], 0.75)

    # And an Anthropic dated snapshot still resolves to its canonical row.
    out = calculate_cost(
        "anthropic",
        "claude-opus-4-7-20260414",
        {"input_tokens": 1_000_000, "output_tokens": 0},
    )
    assert out["priced"] is True
    assert _isclose(out["input_usd"], 5.0)


# ── precedence: input_tokens wins over prompt_tokens (and 0 is real) ──


def test_explicit_zero_input_tokens_wins_over_stale_prompt_tokens():
    # Mirror of ``test_explicit_zero_output_tokens_wins_over_stale_completion_tokens``
    # for the input side -- a turn that genuinely consumed zero prompt
    # tokens must not silently bill against the chat-style mirror.
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
    # ``None`` is the "key present but unset" case -- treated as
    # missing so the chat-style mirror still wins.
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
    # If cache_read claims more tokens than prompt_tokens reports,
    # the uncached_input should clamp at 0 -- but the billable counter
    # still reflects the cache buckets (we charge for what we got).
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
    # OpenAI variant of the same defense: cached > input shouldn't
    # produce a negative "non_cached_input" or a negative input_usd.
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
    # 250k raw input + 50k cache_creation pushes billable to 300k, which
    # crosses the 272k threshold. The whole turn should price at the
    # long-context tier so we don't undercount.
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
    # 272_000 exactly should land in the long-context tier (>=).
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
    # Some upstream proxies fold cache_creation down to a single int.
    # The calculator must tolerate that and fall back to the 5m default.
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
    # Falls back to the 5m default for the whole cache_creation bucket.
    assert _isclose(
        out["cache_write_usd"], 1_000_000 / 1_000_000.0 * base * ANTHROPIC_CACHE_5M_WRITE_MULT
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
    # No cached_tokens recovered, so no cache_read_usd discount.
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
    # Tokens still report for the UI counter.
    assert out["billable_input_tokens"] == 1_000_000
    assert out["billable_output_tokens"] == 1_000_000


def test_anthropic_provider_with_openai_model_priced_false():
    # Provider routing mistake: handing an OpenAI id to the Anthropic
    # table must not falsely match a similar-looking Anthropic key.
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
