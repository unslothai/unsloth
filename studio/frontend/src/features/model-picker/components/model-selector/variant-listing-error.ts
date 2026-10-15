// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Read by property: DOMException does not inherit from Error in older WebKit.
function propString(err: unknown, key: "name" | "message"): string {
  if (typeof err !== "object" || err === null) return "";
  const value = (err as Record<string, unknown>)[key];
  return typeof value === "string" ? value : "";
}

/** Engines without `signal.reason` report a timed-out fetch as AbortError, so it counts as timeout. */
export function describeVariantListingError(err: unknown): string {
  const name = propString(err, "name");
  if (name === "TimeoutError" || name === "AbortError") {
    return "Timed out listing quantizations. Check your connection to Hugging Face, then retry.";
  }
  return propString(err, "message") || "Failed to load variants";
}
