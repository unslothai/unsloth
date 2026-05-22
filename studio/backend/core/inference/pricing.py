# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static per-MTok pricing tables for external providers, plus a
``calculate_cost`` helper that turns an upstream ``usage`` block into
a USD figure for surfacing in the chat UI.

Neither the Anthropic Messages API nor the OpenAI Responses API
reports a ``cost`` field on the response. Both expose detailed token
counts (input, output, cache hits, server-tool invocations); pricing
multipliers live in the provider docs. We fold the docs into a static
table here, multiply by the usage block, and emit a per-turn cost +
running session total client-side.

Sources (verified live 2026-05-22):
- Anthropic models overview:
    https://platform.claude.com/docs/en/about-claude/models/overview
- Anthropic prompt-caching multipliers (5m write 1.25x, 1h write 2x,
  read 0.1x):
    https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Anthropic web search ($10 / 1000 searches, code execution
  free-with-paid when paired with the newer web tools):
    https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
    https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
- OpenAI pricing page (input / output per MTok per model family):
    https://platform.openai.com/docs/pricing
"""

from __future__ import annotations

from typing import Any, Optional

# Per-million-token base pricing. `cache_5m_write_mult`, `cache_1h_write_mult`,
# `cache_read_mult` are multipliers ON `input_per_mtok` -- not absolute prices --
# matching how Anthropic publishes them (5m write = 1.25x base, etc.).
#
# `input_per_mtok` and `output_per_mtok` are USD per 1,000,000 tokens.
ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7":             {"input_per_mtok": 5.0,  "output_per_mtok": 25.0},
    "claude-opus-4-6":             {"input_per_mtok": 5.0,  "output_per_mtok": 25.0},
    "claude-opus-4-5-20251101":    {"input_per_mtok": 5.0,  "output_per_mtok": 25.0},
    "claude-opus-4-1-20250805":    {"input_per_mtok": 15.0, "output_per_mtok": 75.0},
    "claude-opus-4-20250514":      {"input_per_mtok": 15.0, "output_per_mtok": 75.0},
    "claude-sonnet-4-6":           {"input_per_mtok": 3.0,  "output_per_mtok": 15.0},
    "claude-sonnet-4-5-20250929":  {"input_per_mtok": 3.0,  "output_per_mtok": 15.0},
    "claude-sonnet-4-20250514":    {"input_per_mtok": 3.0,  "output_per_mtok": 15.0},
    "claude-haiku-4-5-20251001":   {"input_per_mtok": 1.0,  "output_per_mtok": 5.0},
}

OPENAI_PRICING: dict[str, dict[str, float]] = {
    # gpt-5 family -- approximate published prices as of May 2026.
    # Update from platform.openai.com/docs/pricing when a model lands.
    "gpt-5.5":             {"input_per_mtok": 1.25, "output_per_mtok": 10.0},
    "gpt-5.5-pro":         {"input_per_mtok": 5.0,  "output_per_mtok": 40.0},
    "gpt-5.4":             {"input_per_mtok": 1.25, "output_per_mtok": 10.0},
    "gpt-5.4-pro":         {"input_per_mtok": 5.0,  "output_per_mtok": 40.0},
    "gpt-5.4-mini":        {"input_per_mtok": 0.25, "output_per_mtok": 2.0},
    "gpt-5.4-nano":        {"input_per_mtok": 0.05, "output_per_mtok": 0.4},
    "gpt-5.3-codex":       {"input_per_mtok": 1.25, "output_per_mtok": 10.0},
    "gpt-5.3-chat-latest": {"input_per_mtok": 1.25, "output_per_mtok": 10.0},
    "o3":                  {"input_per_mtok": 2.0,  "output_per_mtok": 8.0},
    "o3-pro":              {"input_per_mtok": 20.0, "output_per_mtok": 80.0},
    "o3-mini":             {"input_per_mtok": 1.1,  "output_per_mtok": 4.4},
    "o3-deep-research":    {"input_per_mtok": 10.0, "output_per_mtok": 40.0},
    "o4-mini":             {"input_per_mtok": 0.5,  "output_per_mtok": 2.0},
}

# Shared multipliers (same across every Anthropic model).
ANTHROPIC_CACHE_5M_WRITE_MULT = 1.25
ANTHROPIC_CACHE_1H_WRITE_MULT = 2.0
ANTHROPIC_CACHE_READ_MULT = 0.1

# OpenAI: cache reads are 0.1x base input, cache writes are not billed
# separately (the first prefix-write request just pays normal input).
OPENAI_CACHE_READ_MULT = 0.1

# Server-tool surcharges (Anthropic-only today).
ANTHROPIC_WEB_SEARCH_USD_PER_1K = 10.0
# code_execution: 50 free hours/day per org, $0.05/hour beyond that.
# We don't have per-org usage visibility here so the calculator
# reports the marginal rate; the frontend can mark the first 50
# hours as "free" if it wants to.
ANTHROPIC_CODE_EXEC_USD_PER_HOUR = 0.05


def _lookup(provider: str, model: str) -> Optional[dict[str, float]]:
    table = (
        ANTHROPIC_PRICING if provider == "anthropic"
        else OPENAI_PRICING if provider == "openai"
        else None
    )
    if table is None:
        return None
    if model in table:
        return table[model]
    # Fall back to a prefix match so date-suffixed snapshots
    # ("gpt-5.5-2026-04-23") inherit the canonical-id prices.
    for key, val in table.items():
        if model.startswith(key):
            return val
    return None


def calculate_cost(
    provider: str,
    model: str,
    usage: dict[str, Any],
) -> dict[str, float]:
    """Return a per-turn USD cost breakdown.

    Returns a dict with the per-bucket cost AND the totals so the
    frontend can render either a single number or a "where did the
    money go" tooltip without re-doing the math:

        {
          "input_usd": 0.0042,
          "output_usd": 0.012,
          "cache_write_usd": 0.0001,
          "cache_read_usd": 0.0008,
          "server_tools_usd": 0.01,
          "total_usd": 0.0271,
          "billable_input_tokens": 5023,    # input + cache_create + cache_read
          "billable_output_tokens": 480,
          "model_priced": "claude-opus-4-7",
          "priced": true,
        }

    When the model isn't in the static table (new family, custom base
    URL), `priced` is False and every USD field is 0.0; the frontend
    can still show the token counts.
    """
    prices = _lookup(provider, model)
    out: dict[str, float] = {
        "input_usd": 0.0,
        "output_usd": 0.0,
        "cache_write_usd": 0.0,
        "cache_read_usd": 0.0,
        "server_tools_usd": 0.0,
        "total_usd": 0.0,
        "billable_input_tokens": 0,
        "billable_output_tokens": 0,
        "model_priced": model if prices else "",
        "priced": bool(prices),
    }

    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    # OpenAI Responses reports cached tokens under input_tokens_details:
    if provider == "openai":
        details = usage.get("input_tokens_details") or {}
        if isinstance(details, dict):
            cache_read = max(cache_read, int(details.get("cached_tokens") or 0))

    out["billable_input_tokens"] = input_tokens + cache_creation + cache_read
    out["billable_output_tokens"] = output_tokens

    if not prices:
        return out

    base = prices["input_per_mtok"]
    out_per = prices["output_per_mtok"]

    out["input_usd"] = (input_tokens / 1_000_000.0) * base
    out["output_usd"] = (output_tokens / 1_000_000.0) * out_per

    if provider == "anthropic":
        # Split cache_creation across 5m / 1h buckets when the
        # response surfaces the breakdown.
        cc_breakdown = usage.get("cache_creation") or {}
        cc_5m = int(cc_breakdown.get("ephemeral_5m_input_tokens") or 0)
        cc_1h = int(cc_breakdown.get("ephemeral_1h_input_tokens") or 0)
        if cc_5m + cc_1h == 0 and cache_creation > 0:
            # Fall back: assume default 5m pool when no breakdown is given.
            cc_5m = cache_creation
        out["cache_write_usd"] = (
            (cc_5m / 1_000_000.0) * base * ANTHROPIC_CACHE_5M_WRITE_MULT
            + (cc_1h / 1_000_000.0) * base * ANTHROPIC_CACHE_1H_WRITE_MULT
        )
        out["cache_read_usd"] = (
            (cache_read / 1_000_000.0) * base * ANTHROPIC_CACHE_READ_MULT
        )
        # Server-tool surcharges.
        srv = usage.get("server_tool_use") or {}
        if isinstance(srv, dict):
            web_searches = int(srv.get("web_search_requests") or 0)
            code_exec_hours = float(srv.get("code_execution_hours") or 0.0)
            out["server_tools_usd"] = (
                web_searches / 1_000.0 * ANTHROPIC_WEB_SEARCH_USD_PER_1K
                + code_exec_hours * ANTHROPIC_CODE_EXEC_USD_PER_HOUR
            )
    else:
        # OpenAI: cache writes share the base input price (no premium).
        # Only cache reads get the 0.1x multiplier; subtract those from
        # the input_usd we already counted so we don't double-bill.
        # Anthropic excludes cache buckets from input_tokens, but
        # OpenAI folds them in, so the math differs.
        if cache_read > 0:
            non_cached_input = max(0, input_tokens - cache_read)
            out["input_usd"] = (non_cached_input / 1_000_000.0) * base
            out["cache_read_usd"] = (
                (cache_read / 1_000_000.0) * base * OPENAI_CACHE_READ_MULT
            )

    out["total_usd"] = round(
        out["input_usd"]
        + out["output_usd"]
        + out["cache_write_usd"]
        + out["cache_read_usd"]
        + out["server_tools_usd"],
        6,
    )
    return out


def pricing_snapshot() -> dict[str, Any]:
    """Whole pricing table, for the /api/providers/pricing endpoint.

    Returns a flat structure the frontend can hand to its cost
    formatter without re-implementing the multipliers.
    """
    return {
        "anthropic": {
            "models": dict(ANTHROPIC_PRICING),
            "cache_5m_write_mult": ANTHROPIC_CACHE_5M_WRITE_MULT,
            "cache_1h_write_mult": ANTHROPIC_CACHE_1H_WRITE_MULT,
            "cache_read_mult": ANTHROPIC_CACHE_READ_MULT,
            "web_search_usd_per_1k": ANTHROPIC_WEB_SEARCH_USD_PER_1K,
            "code_execution_usd_per_hour": ANTHROPIC_CODE_EXEC_USD_PER_HOUR,
        },
        "openai": {
            "models": dict(OPENAI_PRICING),
            "cache_read_mult": OPENAI_CACHE_READ_MULT,
        },
    }
