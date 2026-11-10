// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Deep equality over plain objects and arrays, for reusing a value rebuilt with the same content. */
export function sameValue(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (!(a && b && typeof a === "object" && typeof b === "object")) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a);
  if (ka.length !== Object.keys(b).length) return false;
  for (const key of ka)
    if (
      !sameValue(
        (a as Record<string, unknown>)[key],
        (b as Record<string, unknown>)[key],
      )
    )
      return false;
  return true;
}
