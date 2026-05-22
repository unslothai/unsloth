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
    "claude-opus-4-7": {"input_per_mtok": 5.0, "output_per_mtok": 25.0},
    "claude-opus-4-6": {"input_per_mtok": 5.0, "output_per_mtok": 25.0},
    # Canonical 4.5 ids are referenced from backend defaults (e.g.
    # PROVIDER_REGISTRY['anthropic'].default_models) without the date
    # suffix. The dated ids ARE the canonical names per Anthropic's
    # models overview, but lookups for the bare id ("claude-opus-4-5")
    # don't prefix-match the dated key the other way around, so we
    # alias both forms here. Otherwise calculate_cost returns
    # priced=False + zero cost for the common ids.
    "claude-opus-4-5": {"input_per_mtok": 5.0, "output_per_mtok": 25.0},
    "claude-opus-4-5-20251101": {"input_per_mtok": 5.0, "output_per_mtok": 25.0},
    "claude-opus-4-1": {"input_per_mtok": 15.0, "output_per_mtok": 75.0},
    "claude-opus-4-1-20250805": {"input_per_mtok": 15.0, "output_per_mtok": 75.0},
    "claude-opus-4-20250514": {"input_per_mtok": 15.0, "output_per_mtok": 75.0},
    "claude-sonnet-4-6": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
    "claude-sonnet-4-5": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
    "claude-sonnet-4-5-20250929": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
    "claude-sonnet-4-20250514": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
    "claude-haiku-4-5": {"input_per_mtok": 1.0, "output_per_mtok": 5.0},
    "claude-haiku-4-5-20251001": {"input_per_mtok": 1.0, "output_per_mtok": 5.0},
}

OPENAI_PRICING: dict[str, dict[str, float]] = {
    # All values verified against developers.openai.com/api/docs/pricing
    # 2026-05-22. Update against the live pricing page on every model launch.
    # Initial commit underbilled every gpt-5.x family 2-6x -- fixed here
    # after PR review caught it via doc cross-check.
    #
    # `long_context_input_per_mtok` / `long_context_output_per_mtok` /
    # `long_context_threshold` are populated when OpenAI publishes a
    # second pricing tier for prompts above N input tokens. gpt-5.5 and
    # gpt-5.4 cross over at 272k input tokens; the long-context rates
    # are double the headline input price (and ~1.5x on output). Other
    # families currently ship with a single rate (no `long_context_*`
    # keys = no tier crossover). Reference:
    #   https://developers.openai.com/api/docs/pricing
    "gpt-5.5": {
        "input_per_mtok": 5.0,
        "output_per_mtok": 30.0,
        "long_context_threshold": 272_000,
        "long_context_input_per_mtok": 10.0,
        "long_context_output_per_mtok": 45.0,
    },
    "gpt-5.5-pro": {"input_per_mtok": 30.0, "output_per_mtok": 180.0},
    "gpt-5.4": {
        "input_per_mtok": 2.5,
        "output_per_mtok": 15.0,
        "long_context_threshold": 272_000,
        "long_context_input_per_mtok": 5.0,
        "long_context_output_per_mtok": 22.5,
    },
    "gpt-5.4-pro": {"input_per_mtok": 30.0, "output_per_mtok": 180.0},
    "gpt-5.4-mini": {"input_per_mtok": 0.75, "output_per_mtok": 4.5},
    "gpt-5.4-nano": {"input_per_mtok": 0.20, "output_per_mtok": 1.25},
    "gpt-5.3-codex": {"input_per_mtok": 1.75, "output_per_mtok": 14.0},
    # chat-latest / gpt-5.3-chat-latest is an alias for the current
    # ChatGPT model; same price as gpt-5.5.
    "gpt-5.3-chat-latest": {"input_per_mtok": 5.0, "output_per_mtok": 30.0},
    "chat-latest": {"input_per_mtok": 5.0, "output_per_mtok": 30.0},
    # o-series and gpt-4.5: NOT currently listed on the pricing page.
    # Removed to avoid silent-underbilling drift. Returning priced=False
    # is honest; the UI can still render token counts. Restore with
    # verified per-MTok rates if/when the page lists them again.
}

# Shared multipliers (same across every Anthropic model).
ANTHROPIC_CACHE_5M_WRITE_MULT = 1.25
ANTHROPIC_CACHE_1H_WRITE_MULT = 2.0
ANTHROPIC_CACHE_READ_MULT = 0.1

# OpenAI: cache reads are 0.1x base input, cache writes are not billed
# separately (the first prefix-write request just pays normal input).
OPENAI_CACHE_READ_MULT = 0.1

# Server-tool surcharges.
# Anthropic: $10 / 1000 web searches; code_execution is $0.05/hr after
# 50 free hours/day per org (no per-org visibility here, so the
# calculator reports the marginal rate).
ANTHROPIC_WEB_SEARCH_USD_PER_1K = 10.0
ANTHROPIC_CODE_EXEC_USD_PER_HOUR = 0.05

# OpenAI: web_search is billed at $10/1000 calls plus the model's
# token rate for the returned search content (already captured under
# input/output_tokens). The hosted shell tool bills per 20-minute
# session per container memory tier (1g/4g/16g/64g at
# $0.03/$0.12/$0.48/$1.92). Since Studio doesn't surface the memory
# tier in the cost ledger and most users land on the default 1g, we
# bill the 1g rate ($0.09/hour) and let the user inspect the OpenAI
# dashboard for the exact figure on heavier configs.
# Source: developers.openai.com/api/docs/pricing 2026-05-22.
OPENAI_WEB_SEARCH_USD_PER_1K = 10.0
OPENAI_CONTAINER_USD_PER_HOUR = 0.09  # 1g default tier; 3 x $0.03 / 60min


def _lookup(provider: str, model: str) -> Optional[dict[str, float]]:
    table = (
        ANTHROPIC_PRICING
        if provider == "anthropic"
        else OPENAI_PRICING
        if provider == "openai"
        else None
    )
    if table is None:
        return None
    if model in table:
        return table[model]
    # Fall back to a longest-prefix match so dated snapshots
    # ("gpt-5.5-2026-04-23") inherit the canonical-id prices AND ids
    # like "gpt-5.4-mini-2026-..." match "gpt-5.4-mini" before they
    # collide with the shorter "gpt-5.4" entry. Sorting keys by
    # length descending picks the most specific table row first.
    for key in sorted(table, key = len, reverse = True):
        if model.startswith(key):
            return table[key]
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

    # Accept both shapes: raw Anthropic / OpenAI Responses usage
    # carries ``input_tokens`` / ``output_tokens``; the OpenAI-Chat-
    # style envelope Studio re-emits (``_build_usage_chunk``) uses
    # ``prompt_tokens`` / ``completion_tokens``. Normalise to a single
    # ``uncached_input`` view because the two envelopes treat the
    # cache buckets differently:
    #
    #   raw Anthropic:    input_tokens EXCLUDES cache_creation + cache_read
    #   raw OpenAI:       input_tokens INCLUDES cache_read (no cache_create)
    #   Studio Anthropic: prompt_tokens INCLUDES cache_creation + cache_read
    #   Studio OpenAI:    prompt_tokens == raw input_tokens (includes cache_read)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    has_input_tokens = "input_tokens" in usage and usage.get("input_tokens") is not None
    if has_input_tokens:
        # Raw upstream envelope.
        input_tokens = int(usage.get("input_tokens") or 0)
    else:
        # Studio chat-style envelope: prompt_tokens already folds the
        # cache buckets for Anthropic, so peel them off to recover the
        # raw uncached prompt count and keep downstream math symmetric.
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        if provider == "anthropic":
            input_tokens = max(0, prompt_tokens - cache_creation - cache_read)
        else:
            input_tokens = prompt_tokens
    # Prefer the raw upstream key when present, even when its value is
    # explicitly 0 -- the ``or`` fallback would mistakenly pick a stale
    # ``completion_tokens`` for an empty completion. Mirrors the
    # has_input_tokens precedence above.
    if "output_tokens" in usage and usage.get("output_tokens") is not None:
        output_tokens = int(usage.get("output_tokens") or 0)
    else:
        output_tokens = int(usage.get("completion_tokens") or 0)
    if provider == "openai":
        # Cached prompt tokens live on different sub-objects depending on
        # which envelope landed here. Raw OpenAI Responses usage uses
        # ``input_tokens_details.cached_tokens``; the OpenAI Chat
        # Completions envelope Studio re-emits via ``_build_usage_chunk``
        # uses ``prompt_tokens_details.cached_tokens``. Check both so
        # cache-heavy chat-style turns get the 0.1x cache_read_mult
        # discount instead of full input pricing.
        for key in ("input_tokens_details", "prompt_tokens_details"):
            details = usage.get(key) or {}
            if isinstance(details, dict):
                cache_read = max(cache_read, int(details.get("cached_tokens") or 0))
        # OpenAI: cache_read already counted inside input_tokens.
        out["billable_input_tokens"] = input_tokens + cache_creation
    else:
        # Anthropic: input_tokens (post-normalisation) excludes cache
        # buckets, so add them all back.
        out["billable_input_tokens"] = input_tokens + cache_creation + cache_read
    out["billable_output_tokens"] = output_tokens

    if not prices:
        return out

    # Long-context tier crossover (gpt-5.5 / gpt-5.4 today). OpenAI
    # bills the whole turn at the long-context rate once the prompt
    # crosses the threshold, NOT a per-token blend, so we pick a
    # single (base, out_per) pair for this turn based on
    # billable_input_tokens.
    lc_thresh = prices.get("long_context_threshold")
    in_long_context_tier = (
        lc_thresh is not None
        and out["billable_input_tokens"] >= int(lc_thresh)
        and "long_context_input_per_mtok" in prices
        and "long_context_output_per_mtok" in prices
    )
    if in_long_context_tier:
        base = prices["long_context_input_per_mtok"]
        out_per = prices["long_context_output_per_mtok"]
        out["model_priced"] = f"{model} (long-context >{lc_thresh})"
    else:
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
            cc_5m / 1_000_000.0
        ) * base * ANTHROPIC_CACHE_5M_WRITE_MULT + (
            cc_1h / 1_000_000.0
        ) * base * ANTHROPIC_CACHE_1H_WRITE_MULT
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
        # Server-tool surcharges. OpenAI doesn't include these on its
        # `usage` object directly -- web_search invocations are counted
        # from `ResponseFunctionWebSearch` items in the output array,
        # and container hours come from the SSE translator's shell-tool
        # accounting. Studio surfaces both under a normalised
        # `openai_tool_use` key on the usage dict the SSE finaliser
        # hands to this calculator.
        srv = usage.get("openai_tool_use") or {}
        if isinstance(srv, dict):
            web_searches = int(srv.get("web_search_requests") or 0)
            container_hours = float(srv.get("container_hours") or 0.0)
            out["server_tools_usd"] = (
                web_searches / 1_000.0 * OPENAI_WEB_SEARCH_USD_PER_1K
                + container_hours * OPENAI_CONTAINER_USD_PER_HOUR
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
            "web_search_usd_per_1k": OPENAI_WEB_SEARCH_USD_PER_1K,
            "container_usd_per_hour": OPENAI_CONTAINER_USD_PER_HOUR,
        },
    }
