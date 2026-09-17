// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * How Studio prints a memory figure. One implementation, for every surface.
 *
 * There used to be two, and they were both called `formatMemoryGb`:
 *
 * - `model-config/memory-fit.ts` took BYTES and printed `"24.00 GB"`
 * - `lib/model-memory.ts` took GIGABYTES and printed `"24 GiB"`
 *
 * Same name, same `(number) => string` signature, different input unit and
 * different output label. Importing the wrong one is off by 1024^3 and
 * typechecks cleanly, and the panel's variant was labelling a binary divide as
 * "GB" across seven figures, which is the defect #9570 fixed elsewhere.
 *
 * So the unit is in the NAME here. `formatGiB` takes gibibytes, `formatBytesGiB`
 * takes bytes, and neither can be confused for the other at a call site.
 *
 * Everything Studio measures in memory is a binary divide: weights and KV come
 * from `bytes / 1024**3`, and the GPU total arrives from the backend as
 * `props.total_memory / 1024**3` with nothing subtracting a budget on the way.
 * Calling any of them GB overstates each by 7.4%.
 *
 * No `@/` alias imports, deliberately. `tests/` runs under `node --experimental-strip-types`,
 * which does not resolve the `@/` alias, and the modules this replaces documented
 * the same constraint. A single import here makes the whole chain untestable.
 */

const BYTES_PER_GIB = 1024 ** 3;

/**
 * Adaptive precision, for a one-line readout: `"7.2 GiB"`, `"24 GiB"`.
 *
 * Clamped rather than trusted. Every figure here comes off the wire, and
 * `"-3.0 GiB"` or `"NaN GiB"` both read as a measurement rather than as the
 * missing reading they are.
 */
export function formatGiB(gib: number): string {
  if (!Number.isFinite(gib) || gib <= 0) return "0 GiB";
  return `${gib < 10 ? gib.toFixed(1) : Math.round(gib)} GiB`;
}

/**
 * Fixed two decimals, for an itemized column where the figures line up:
 * `"24.00 GiB"`.
 *
 * A separate function rather than a flag, because the choice is about the
 * SURFACE and not about the number. A readout that says "about 7 GiB" and a
 * table row that has to add up want different things, and threading a boolean
 * through every call site is how they end up inconsistent again.
 */
export function formatBytesGiB(bytes: number): string {
  const safe = Number.isFinite(bytes) && bytes > 0 ? bytes : 0;
  return `${(safe / BYTES_PER_GIB).toFixed(2)} GiB`;
}

/**
 * A per-token KV rate, which lands in KiB or MiB.
 *
 * Binary throughout, and labelled so. The previous version divided by 1024 and
 * printed "KB", the same mislabel as above one scale down.
 */
export function formatKvRate(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 KiB";
  const kib = bytes / 1024;
  if (kib < 1024) return `${kib < 10 ? kib.toFixed(1) : Math.round(kib)} KiB`;
  const mib = kib / 1024;
  return `${mib < 10 ? mib.toFixed(1) : Math.round(mib)} MiB`;
}
