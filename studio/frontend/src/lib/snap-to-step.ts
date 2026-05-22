// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Decimal places implied by a step, handling scientific notation: String(1e-7)
// is "1e-7", whose naive ".split('.')" yields 0 and rounds the result to an
// integer.
function stepDecimals(step: number): number {
  const s = String(step);
  const e = s.indexOf("e");
  if (e !== -1) {
    const mantissa = s.slice(0, e);
    const mantissaDecimals = mantissa.includes(".")
      ? mantissa.split(".")[1].length
      : 0;
    return Math.max(0, mantissaDecimals - Number(s.slice(e + 1)));
  }
  return s.includes(".") ? s.split(".")[1].length : 0;
}

/** Clamp to [min, max] and snap to the nearest step, anchored at min. */
export function snapToStep(
  value: number,
  step: number,
  min?: number,
  max?: number,
): number {
  const lo = min ?? Number.NEGATIVE_INFINITY;
  const hi = max ?? Number.POSITIVE_INFINITY;
  const clamped = Math.min(Math.max(value, lo), hi);
  // A non-positive or non-finite step has no grid to snap to; clamp only.
  if (!Number.isFinite(step) || step <= 0) return clamped;
  const base = Number.isFinite(lo) ? lo : 0;
  const snapped = base + Math.round((clamped - base) / step) * step;
  return Number(
    Math.min(Math.max(snapped, lo), hi).toFixed(stepDecimals(step)),
  );
}
