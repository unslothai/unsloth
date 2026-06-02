// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Format a byte-per-second rate as a human-readable string.
 *
 *   512            → "512 B/s"
 *   1_234_567      → "1.2 MB/s"
 *   1_234_567_890  → "1.15 GB/s"
 *
 * Returns `"--"` for non-finite or non-positive inputs so callers can
 * render the label safely before the first stable sample arrives.
 */
export function formatRate(bytesPerSecond: number): string {
  if (!Number.isFinite(bytesPerSecond) || bytesPerSecond <= 0) return "--";
  const bps = bytesPerSecond;
  if (bps < 1024) return `${bps.toFixed(0)} B/s`;
  if (bps < 1024 ** 2) return `${(bps / 1024).toFixed(1)} KB/s`;
  if (bps < 1024 ** 3) return `${(bps / 1024 ** 2).toFixed(1)} MB/s`;
  return `${(bps / 1024 ** 3).toFixed(2)} GB/s`;
}

/**
 * Format an ETA (in seconds) as a short human-readable string.
 *
 *   47       → "47s"
 *   125      → "2m 5s"
 *   3725     → "1h 2m"
 *
 * Returns `"--"` for non-finite or non-positive inputs.
 */
export function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "--";
  const s = Math.round(seconds);
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
