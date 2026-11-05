// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Memory units are in the function names: `formatGiB` takes gibibytes, `formatBytesGiB` bytes.
 * Mixing them up is off by 1024^3 and still typechecks. Every figure is a binary divide.
 * No `@/` alias imports: tests under node --experimental-strip-types cannot resolve them.
 */

const BYTES_PER_GIB = 1024 ** 3;

/** Clamped: negative or NaN wire values would otherwise read as real measurements. */
export function formatGiB(gib: number): string {
  if (!Number.isFinite(gib) || gib <= 0) return "0 GiB";
  return `${gib < 10 ? gib.toFixed(1) : Math.round(gib)} GiB`;
}

/** Fixed two decimals for itemized columns. */
export function formatBytesGiB(bytes: number): string {
  const safe = Number.isFinite(bytes) && bytes > 0 ? bytes : 0;
  return `${(safe / BYTES_PER_GIB).toFixed(2)} GiB`;
}

export function formatKvRate(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 KiB";
  const kib = bytes / 1024;
  if (kib < 1024) return `${kib < 10 ? kib.toFixed(1) : Math.round(kib)} KiB`;
  const mib = kib / 1024;
  return `${mib < 10 ? mib.toFixed(1) : Math.round(mib)} MiB`;
}
