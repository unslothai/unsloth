// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Read by property, not instanceof: DOMException does not inherit from Error in older WebKit, so
// `err instanceof Error` misses exactly the aborts this needs to classify.
function propString(err: unknown, key: "name" | "message"): string {
  if (typeof err !== "object" || err === null) return "";
  const value = (err as Record<string, unknown>)[key];
  return typeof value === "string" ? value : "";
}

/** Why a repo's quant listing did not arrive. A raw timeout reads as "The operation timed out.",
 *  which names neither the Hub nor the retry that usually clears it. Engines without
 *  `signal.reason` report a plain AbortError for a timed-out fetch, so that counts as a timeout
 *  too; a deliberate abort never reaches here, since the expander drops its result first. */
export function describeVariantListingError(err: unknown): string {
  const name = propString(err, "name");
  if (name === "TimeoutError" || name === "AbortError") {
    return "Timed out listing quantizations. Check your connection to Hugging Face, then retry.";
  }
  return propString(err, "message") || "Failed to load variants";
}
