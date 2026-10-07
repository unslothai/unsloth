// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Older backends send only `dense_quant_supported`.

/** Absent or malformed is [], never a guess like "fp8". */
export function normalizeDenseQuantSchemes(
  schemes: readonly unknown[] | undefined | null,
): readonly string[] {
  if (!Array.isArray(schemes)) return [];
  return schemes
    .filter((scheme): scheme is string => typeof scheme === "string")
    .map((scheme) => scheme.trim().toLowerCase())
    .filter((scheme) => scheme.length > 0);
}
