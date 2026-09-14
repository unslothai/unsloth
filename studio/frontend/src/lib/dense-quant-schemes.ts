// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `/api/system.dense_quant_schemes`: which dense torchao schemes this host can actually run, best
// first ("fp8" on Ada / Hopper / Blackwell, "int8" on Ampere, empty where the path is
// unsupported). It sits beside the older `dense_quant_supported` boolean, so the reader has to
// survive a backend that sends only the boolean. Its own module because the GPU hook reads the
// field and the media picker consumes it, and neither should have to import the other.

/** The reported list, lower-cased and emptied of blanks. An absent or malformed field is []
 *  rather than a guessed default: inventing "fp8" for an older backend would put a precision on a
 *  row that an Ampere host never runs. Order is the backend's preference and is preserved. */
export function normalizeDenseQuantSchemes(
  schemes: readonly unknown[] | undefined | null,
): readonly string[] {
  if (!Array.isArray(schemes)) return [];
  return schemes
    .filter((scheme): scheme is string => typeof scheme === "string")
    .map((scheme) => scheme.trim().toLowerCase())
    .filter((scheme) => scheme.length > 0);
}
