// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure state rules for the floating quote pill.

export type QuotePillMode = "hidden" | "single" | "multiple";

export function quotePillMode(count: number): QuotePillMode {
  if (count <= 0) {
    return "hidden";
  }
  return count === 1 ? "single" : "multiple";
}

/** Removing the last quote closes the dropdown; otherwise its state survives. */
export function quotePillExpandedAfterCountChange(
  previousCount: number,
  nextCount: number,
  expanded: boolean,
): boolean {
  if (nextCount <= 0) {
    return false;
  }
  if (previousCount <= 0) {
    return false;
  }
  return expanded;
}
