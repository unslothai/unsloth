// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `/api/system.dense_quant_schemes` sits beside the older `dense_quant_supported` boolean, so a
// reader must survive a backend sending only the boolean.

/** The reported list, lower-cased and emptied of blanks. An absent or malformed field is [], never a
 *  guess: inventing "fp8" would name a precision an Ampere host never runs. */
export function normalizeDenseQuantSchemes(
  schemes: readonly unknown[] | undefined | null,
): readonly string[] {
  if (!Array.isArray(schemes)) return [];
  return schemes
    .filter((scheme): scheme is string => typeof scheme === "string")
    .map((scheme) => scheme.trim().toLowerCase())
    .filter((scheme) => scheme.length > 0);
}
