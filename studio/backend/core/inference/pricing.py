# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-MTok pricing tables and ``calculate_cost`` (usage block -> USD).

Sources: Anthropic prompt-caching docs (5m write 1.25x, 1h write 2x,
read 0.1x), web search ($10/1000), code execution; OpenAI pricing page.
"""

from __future__ import annotations

from typing import Any, Optional

# Per-MTok base USD. Cache multipliers apply to `input_per_mtok`
# (not absolute prices), per Anthropic docs.
ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {"input_per_mtok": 5.0, "output_per_mtok": 25.0},
    "claude-opus-4-6": {"input_per_mtok": 5.0, "output_per_mtok": 25.0},
    # Alias bare + dated id: backend defaults use the bare form, which
    # won't prefix-match the dated key.
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
    # Verified against developers.openai.com/api/docs/pricing.
    # `long_context_*` keys apply past the threshold (gpt-5.5/5.4: 272k);
    # families without them ship a single rate.
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
    # chat-latest aliases gpt-5.5.
    "gpt-5.3-chat-latest": {"input_per_mtok": 5.0, "output_per_mtok": 30.0},
    "chat-latest": {"input_per_mtok": 5.0, "output_per_mtok": 30.0},
    # o-series / gpt-4.5 left off the pricing page: omit so calculate_cost
    # returns priced=False instead of silently $0.
}

# Shared multipliers (all Anthropic models).
ANTHROPIC_CACHE_5M_WRITE_MULT = 1.25
ANTHROPIC_CACHE_1H_WRITE_MULT = 2.0
ANTHROPIC_CACHE_READ_MULT = 0.1
# Anthropic fast-mode (Opus 4.6/4.7 only): 6x on input + output.
# https://platform.claude.com/docs/en/build-with-claude/fast-mode#pricing
ANTHROPIC_FAST_MODE_MULT = 6.0

# OpenAI: cache reads 0.1x; cache writes pay input price.
OPENAI_CACHE_READ_MULT = 0.1

# Server-tool surcharges. Anthropic code_exec: $0.05/hr marginal
# (50 free hours/day per org, not shown here).
ANTHROPIC_WEB_SEARCH_USD_PER_1K = 10.0
ANTHROPIC_CODE_EXEC_USD_PER_HOUR = 0.05

# OpenAI container bills per memory tier; report the 1g default
# ($0.09/hr) since the tier isn't surfaced to the ledger.
OPENAI_WEB_SEARCH_USD_PER_1K = 10.0
OPENAI_CONTAINER_USD_PER_HOUR = 0.09  # 1g default tier


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
    # Longest-prefix match on a dash boundary: dated snapshots inherit
    # canonical prices, but "claude-opus-4-15" won't match "claude-opus-4-1".
    for key in sorted(table, key = len, reverse = True):
        if model.startswith(key) and (len(model) == len(key) or model[len(key)] == "-"):
            return table[key]
    return None


def calculate_cost(provider: str, model: str, usage: dict[str, Any]) -> dict[str, float]:
    """Return a per-turn USD cost breakdown (per-bucket + total).

    Unknown model -> ``priced`` False and USD fields 0.0 (token counts still report).
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

    # Accept raw (input_tokens/output_tokens) and Unsloth chat-style
    # (prompt_tokens/completion_tokens) envelopes. Cache buckets differ:
    #   raw Anthropic:    input_tokens EXCLUDES cache buckets
    #   raw OpenAI:       input_tokens INCLUDES cache_read
    #   Unsloth Anthropic: prompt_tokens INCLUDES cache_creation + cache_read
    #   Unsloth OpenAI:    prompt_tokens == raw input_tokens
    # Clamp >=0 so corrupted payloads can't produce a negative bill.
    cache_creation = max(0, int(usage.get("cache_creation_input_tokens") or 0))
    cache_read_native_present = (
        "cache_read_input_tokens" in usage and usage.get("cache_read_input_tokens") is not None
    )
    cache_read = max(0, int(usage.get("cache_read_input_tokens") or 0))
    # Fall back to mirrored prompt_tokens_details only when native
    # cache_read_input_tokens is absent; an explicit native 0 is
    # authoritative, so a stale proxy mirror can't inflate cache_read.
    if not cache_read_native_present:
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            cache_read = max(0, int(details.get("cached_tokens") or 0))
    has_input_tokens = "input_tokens" in usage and usage.get("input_tokens") is not None
    if has_input_tokens:
        input_tokens = max(0, int(usage.get("input_tokens") or 0))
    else:
        # Chat-style: peel cache buckets back out for Anthropic to get
        # the raw uncached prompt count.
        prompt_tokens = max(0, int(usage.get("prompt_tokens") or 0))
        if provider == "anthropic":
            input_tokens = max(0, prompt_tokens - cache_creation - cache_read)
        else:
            input_tokens = prompt_tokens
    # Prefer raw output_tokens even when 0 (an `or` would pick a stale
    # completion_tokens).
    if "output_tokens" in usage and usage.get("output_tokens") is not None:
        output_tokens = max(0, int(usage.get("output_tokens") or 0))
    else:
        output_tokens = max(0, int(usage.get("completion_tokens") or 0))
    if provider == "openai":
        # Cached tokens land on input_tokens_details (raw Responses) or
        # prompt_tokens_details (Unsloth chat-style).
        for key in ("input_tokens_details", "prompt_tokens_details"):
            details = usage.get(key) or {}
            if isinstance(details, dict):
                cache_read = max(cache_read, int(details.get("cached_tokens") or 0))
        # OpenAI input_tokens already counts cache_read.
        out["billable_input_tokens"] = input_tokens + cache_creation
    else:
        # Anthropic input_tokens excludes cache buckets; add them back.
        out["billable_input_tokens"] = input_tokens + cache_creation + cache_read
    out["billable_output_tokens"] = output_tokens

    if not prices:
        return out

    # Long-context tier: whole-turn flip (not per-token blend) once
    # billable_input_tokens crosses the threshold.
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

    # Anthropic fast-mode: 6x on input + output. Cache multipliers stack
    # on top, so applying once to (base, out_per) flows into the
    # cache_*_usd buckets below.
    if provider == "anthropic" and usage.get("speed") == "fast":
        base *= ANTHROPIC_FAST_MODE_MULT
        out_per *= ANTHROPIC_FAST_MODE_MULT
        if out["model_priced"]:
            out["model_priced"] = f"{out['model_priced']} (fast)"

    out["input_usd"] = (input_tokens / 1_000_000.0) * base
    out["output_usd"] = (output_tokens / 1_000_000.0) * out_per

    if provider == "anthropic":
        # Split cache_creation into 5m / 1h buckets when surfaced.
        # Tolerate non-dict (some proxies fold to an int total).
        cc_raw = usage.get("cache_creation")
        cc_breakdown = cc_raw if isinstance(cc_raw, dict) else {}
        cc_5m = max(0, int(cc_breakdown.get("ephemeral_5m_input_tokens") or 0))
        cc_1h = max(0, int(cc_breakdown.get("ephemeral_1h_input_tokens") or 0))
        if cc_5m + cc_1h == 0 and cache_creation > 0:
            # No breakdown -- assume default 5m pool.
            cc_5m = cache_creation
        out["cache_write_usd"] = (cc_5m / 1_000_000.0) * base * ANTHROPIC_CACHE_5M_WRITE_MULT + (
            cc_1h / 1_000_000.0
        ) * base * ANTHROPIC_CACHE_1H_WRITE_MULT
        out["cache_read_usd"] = (cache_read / 1_000_000.0) * base * ANTHROPIC_CACHE_READ_MULT
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
        # OpenAI: cache writes pay base input, only reads get 0.1x.
        # Subtract cached from already-counted input_usd to avoid
        # double-billing (OpenAI folds cache into input_tokens).
        if cache_read > 0:
            non_cached_input = max(0, input_tokens - cache_read)
            out["input_usd"] = (non_cached_input / 1_000_000.0) * base
            out["cache_read_usd"] = (cache_read / 1_000_000.0) * base * OPENAI_CACHE_READ_MULT
        # OpenAI server-tool surcharges arrive under `openai_tool_use`
        # (normalised by the SSE finaliser from output items).
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
    """Whole pricing table for the /api/providers/pricing endpoint."""
    return {
        "anthropic": {
            "models": dict(ANTHROPIC_PRICING),
            "cache_5m_write_mult": ANTHROPIC_CACHE_5M_WRITE_MULT,
            "cache_1h_write_mult": ANTHROPIC_CACHE_1H_WRITE_MULT,
            "cache_read_mult": ANTHROPIC_CACHE_READ_MULT,
            "fast_mode_mult": ANTHROPIC_FAST_MODE_MULT,
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
