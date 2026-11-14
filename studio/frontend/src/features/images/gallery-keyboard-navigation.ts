// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Return the visible gallery id immediately before or after the focused tile. */
export function adjacentGalleryId(
  ids: readonly string[],
  currentId: string,
  direction: "previous" | "next",
): string | null {
  const current = ids.indexOf(currentId);
  if (current < 0) return null;
  const target = current + (direction === "previous" ? -1 : 1);
  return ids[target] ?? null;
}
