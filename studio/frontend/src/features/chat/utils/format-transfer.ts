// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** True when `value` still reads below 1024 once rounded to `digits` decimals. Picking the unit on
 *  the raw value let 1023.6 B/s print as "1024 B/s" and 1048575 B/s as "1024.0 KB/s". */
function staysBelowNextUnit(value: number, digits: number): boolean {
  return Number(value.toFixed(digits)) < 1024;
}

export function formatRate(bytesPerSecond: number): string {
  if (!Number.isFinite(bytesPerSecond) || bytesPerSecond <= 0) return "--";
  const bps = bytesPerSecond;
  const kb = bps / 1024;
  const mb = bps / 1024 ** 2;
  if (staysBelowNextUnit(bps, 0)) return `${bps.toFixed(0)} B/s`;
  if (staysBelowNextUnit(kb, 1)) return `${kb.toFixed(1)} KB/s`;
  if (staysBelowNextUnit(mb, 1)) return `${mb.toFixed(1)} MB/s`;
  return `${(bps / 1024 ** 3).toFixed(2)} GB/s`;
}

/** Clamp beyond a day, matching the hub formatter. */
const MAX_DISPLAYABLE_ETA_SECONDS = 24 * 60 * 60;

export function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "--";
  const s = Math.round(seconds);
  if (s >= MAX_DISPLAYABLE_ETA_SECONDS) return "> 24h";
  if (s < 60) return `${s}s`;
  if (s < 3600) {
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return rem > 0 ? `${m}m ${rem}s` : `${m}m`;
  }
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}
