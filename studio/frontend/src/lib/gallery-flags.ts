// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Optimistic list maths for the Images and Video galleries, kept pure so both pages share one
 * implementation and it can be tested without rendering either.
 *
 * The backend is the source of truth for order (pinned first, then newest first). These helpers
 * only reproduce that ordering locally so a click lands instantly instead of waiting on a refetch.
 */

export const GALLERY_CHANGED_EVENT = "unsloth:gallery-changed";

/** Which gallery a change landed in. */
export type GalleryKind = "images" | "videos";

/**
 * Announce that a gallery changed from OUTSIDE the page that owns it.
 *
 * The Images and Video pages are mounted persistently by `__root.tsx` (so an in-flight batch
 * survives leaving the tab) and their `loadGallery` effects run only on mount. Restoring or
 * deleting from the Settings archive therefore left the strip stale until a full reload, since
 * closing Settings never remounts the page. In lib/, not a feature: the emitter (Settings) and
 * the listeners (Images, Video) are both features.
 */
export function notifyGalleryChanged(kind: GalleryKind): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(GALLERY_CHANGED_EVENT, { detail: { kind } }));
}

/** Calls back when `kind` changed elsewhere. Returns an unsubscriber. */
export function subscribeGalleryChanged(kind: GalleryKind, onChanged: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  const handler = (event: Event) => {
    const detail = (event as CustomEvent<{ kind?: string }>).detail;
    if (detail?.kind === kind) onChanged();
  };
  window.addEventListener(GALLERY_CHANGED_EVENT, handler);
  return () => window.removeEventListener(GALLERY_CHANGED_EVENT, handler);
}

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

/** Pages to walk looking for the newest unpinned record, so an absurd pin count cannot turn a
 * recovery probe into a full gallery scan. One page is normally enough. */
export const NEW_RECORD_PROBE_MAX_PAGES = 5;

/**
 * Whether the gallery holds a record absent from `knownIds`, used to prove that a generation whose
 * POST response was lost did in fact reach the server.
 *
 * A saved record is always unpinned and newest, so it is the FIRST unpinned row of the listing.
 * This walks the pinned prefix to reach that row instead of reading row 0: pinned records sort
 * ahead of everything, so with any pin present row 0 is a record we already knew, and the probe
 * would report a finished generation as never submitted.
 */
export async function hasUnknownRecord<T extends FlaggableItem>(
  knownIds: ReadonlySet<string>,
  fetchPage: (offset: number) => Promise<{ items: T[]; hasMore: boolean }>,
  pageSize: number,
  maxPages: number = NEW_RECORD_PROBE_MAX_PAGES,
): Promise<boolean> {
  for (let page = 0; page < maxPages; page += 1) {
    const { items, hasMore } = await fetchPage(page * pageSize);
    for (const record of items) {
      if (!knownIds.has(record.id)) return true;
      // The first unpinned row was already known, so nothing new was saved.
      if (!record.pinned) return false;
    }
    if (!hasMore || items.length === 0) return false;
  }
  return false;
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
