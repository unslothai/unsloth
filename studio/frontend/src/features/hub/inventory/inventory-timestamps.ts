// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** normalize inventory timestamps to epoch milliseconds. */
export function normalizeTimestamp(value?: number | null): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  return value < 10_000_000_000 ? value * 1000 : value;
}

/** convert a normalized inventory timestamp back to the backend dto unit. */
export function epochMillisecondsToSeconds(
  value?: number | null,
): number | undefined {
  return value == null ? undefined : value / 1000;
}
