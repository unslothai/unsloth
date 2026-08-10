// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Optimistic list maths for the Images and Video galleries, kept pure so both pages share one
 * implementation and it can be tested without rendering either.
 *
 * The backend is the source of truth for order (pinned first, then newest first). These helpers
 * only reproduce that ordering locally so a click lands instantly instead of waiting on a refetch.
 */

/** The two fields these helpers care about; both gallery record types structurally satisfy it. */
export interface FlaggableItem {
  id: string;
  pinned?: boolean;
  archived?: boolean;
  created_at: number | string;
}

/** Newest first, matching the backend's mtime ordering closely enough for an optimistic reorder. */
function newestFirst<T extends FlaggableItem>(a: T, b: T): number {
  const at = typeof a.created_at === "number" ? a.created_at : Date.parse(a.created_at);
  const bt = typeof b.created_at === "number" ? b.created_at : Date.parse(b.created_at);
  return bt - at;
}

/**
 * Re-sort one shelf: pinned items lead, then everything else newest first.
 *
 * Within the pinned group the backend sorts by pin time, which the client does not know, so a
 * freshly pinned item is moved to the very front -- the same "most recently pinned first" rule,
 * and what the user just asked for visually. A reload reconciles with the server.
 */
export function sortGalleryItems<T extends FlaggableItem>(items: T[], justPinnedId?: string): T[] {
  const pinned = items.filter((i) => i.pinned);
  const rest = items.filter((i) => !i.pinned);
  pinned.sort(newestFirst);
  rest.sort(newestFirst);
  if (justPinnedId) {
    const at = pinned.findIndex((i) => i.id === justPinnedId);
    if (at > 0) pinned.unshift(...pinned.splice(at, 1));
  }
  return [...pinned, ...rest];
}

/**
 * Apply a pin toggle in place and re-sort. Archiving is NOT handled here: an archived item leaves
 * this shelf entirely, which is `removeGalleryItem`.
 */
export function applyPin<T extends FlaggableItem>(items: T[], id: string, pinned: boolean): T[] {
  const next = items.map((i) => (i.id === id ? { ...i, pinned } : i));
  return sortGalleryItems(next, pinned ? id : undefined);
}

/** Drop an item that was archived or deleted; order among the rest is untouched. */
export function removeGalleryItem<T extends FlaggableItem>(items: T[], id: string): T[] {
  return items.filter((i) => i.id !== id);
}

/**
 * The selection after `removedId` left the strip: keep the current pick unless it was the one that
 * left, in which case fall to its neighbour so the preview never blanks out with items still on
 * screen. `remaining` must already have the item removed.
 */
export function nextSelectedId<T extends FlaggableItem>(
  remaining: T[],
  removedId: string,
  selectedId: string | null,
  removedIndex: number,
): string | null {
  if (selectedId !== removedId) return selectedId;
  if (remaining.length === 0) return null;
  return remaining[Math.min(Math.max(removedIndex, 0), remaining.length - 1)].id;
}
