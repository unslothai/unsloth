// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface PublishedPricing {
  rates: Record<string, string>;
  overrides?: Record<string, unknown>[];
}

/** Reject coercions (empty strings, booleans, hex), negative and nonfinite rates. */
export function nonnegativeDecimal(value: unknown): number | null {
  if (typeof value !== "number" && typeof value !== "string") return null;
  if (
    typeof value === "string" &&
    !/^(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(value)
  )
    return null;
  const n = Number(value);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

export function formatUsd(value: number | null): string {
  if (value == null || !Number.isFinite(value) || value < 0)
    return "Unavailable";
  if (value === 0) return "$0";
  if (value < 0.000001) return `$${value.toExponential(2)}`;
  return `$${new Intl.NumberFormat("en-US", { maximumFractionDigits: 8 }).format(value)}`;
}

export function tokenRate(value: unknown): string {
  const rate = nonnegativeDecimal(value);
  return formatUsd(rate == null ? null : rate * 1_000_000);
}
