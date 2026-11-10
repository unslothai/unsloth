// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const ABSENT = Symbol("absent");

function plain(value: unknown): unknown {
  if (
    value === undefined ||
    typeof value === "function" ||
    typeof value === "symbol"
  )
    return ABSENT;
  if (typeof value === "number" && !Number.isFinite(value)) return null;
  return value;
}

function special(value: unknown): boolean {
  return (
    typeof value === "bigint" ||
    (typeof value === "object" &&
      value !== null &&
      (typeof (value as { toJSON?: unknown }).toJSON === "function" ||
        value instanceof Number ||
        value instanceof String ||
        value instanceof Boolean))
  );
}

function equal(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (special(a) || special(b)) return JSON.stringify(a) === JSON.stringify(b);
  if (typeof a !== "object" || typeof b !== "object" || !a || !b) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i += 1) {
      const x = plain(a[i]);
      const y = plain(b[i]);
      if (!equal(x === ABSENT ? null : x, y === ABSENT ? null : y)) return false;
    }
    return true;
  }
  const ka = Object.keys(a).filter((k) => plain((a as Record<string, unknown>)[k]) !== ABSENT);
  const kb = Object.keys(b).filter((k) => plain((b as Record<string, unknown>)[k]) !== ABSENT);
  if (ka.length !== kb.length) return false;
  for (let i = 0; i < ka.length; i += 1) {
    if (ka[i] !== kb[i]) return false;
    const x = plain((a as Record<string, unknown>)[ka[i]]);
    const y = plain((b as Record<string, unknown>)[kb[i]]);
    if (!equal(x, y)) return false;
  }
  return true;
}

/** `JSON.stringify(a) === JSON.stringify(b)` without building the strings: key order counts, and
 * what JSON drops or nulls (undefined, functions, NaN) is dropped or nulled here too. */
export function jsonEqual(a: unknown, b: unknown): boolean {
  return equal(plain(a), plain(b));
}
