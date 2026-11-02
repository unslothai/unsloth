// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const MONITOR_REFRESH_OPTIONS_MS = [1000, 2000, 5000, 10000] as const;
export const DEFAULT_MONITOR_REFRESH_MS = 5000;

/** A persisted value outside the options (older build, hand edit) falls back to the default. */
export function monitorRefreshMs(value: unknown): number {
  return MONITOR_REFRESH_OPTIONS_MS.includes(value as never)
    ? (value as number)
    : DEFAULT_MONITOR_REFRESH_MS;
}
