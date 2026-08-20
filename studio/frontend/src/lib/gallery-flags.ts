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
 * Announce that a gallery changed from OUTSIDE the page that owns it. Both pages are mounted
 * persistently and load only on mount, so a restore from Settings would otherwise leave the strip
 * stale until a full reload. In lib/ because emitter and listeners are both features.
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
 * Re-sort one shelf: pinned lead, then newest first.
 *
 * The pinned group keeps ARRIVAL order, which is the server's. It sorts pins by pin time, which the
 * client never learns, so re-sorting them by `created_at` would rearrange them on any unrelated
 * merge. `justPinnedId` is the exception: a fresh pin leads, matching the backend.
 */
export function sortGalleryItems<T extends FlaggableItem>(items: T[], justPinnedId?: string): T[] {
  const pinned = items.filter((i) => i.pinned);
  const rest = items.filter((i) => !i.pinned);
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

/** The pinned ids in their current order, to hand back to `restorePinOrder` on a failed unpin. */
export function pinnedOrder<T extends FlaggableItem>(items: T[]): string[] {
  return items.filter((i) => i.pinned).map((i) => i.id);
}

/**
 * Undo a failed unpin, back to where the item WAS. `applyPin(..., true)` means "freshly pinned" and
 * would promote it to the head instead, disagreeing with a server order that never changed.
 *
 * `order` is the pinned ids from before the click. An id missing from it was pinned mid-request, and
 * `indexOf` scoring those -1 is exactly right: the newest pin leads.
 */
export function restorePinOrder<T extends FlaggableItem>(
  items: T[],
  id: string,
  order: readonly string[],
): T[] {
  const next = items.map((i) => (i.id === id ? { ...i, pinned: true } : i));
  const pinned = next.filter((i) => i.pinned);
  const rest = next.filter((i) => !i.pinned);
  rest.sort(newestFirst);
  pinned.sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));
  return [...pinned, ...rest];
}

/** Prepend a finished run's records, dropping any a concurrent load already brought in. */
export function mergeGenerated<T extends FlaggableItem>(items: T[], fresh: T[]): T[] {
  const known = new Set(items.map((i) => i.id));
  return sortGalleryItems([...fresh.filter((i) => !known.has(i.id)), ...items]);
}

/** Pages to walk looking for the newest unpinned record, so an absurd pin count cannot turn a
 * recovery probe into a full gallery scan. One page is normally enough. */
export const NEW_RECORD_PROBE_MAX_PAGES = 5;

/**
 * Whether the gallery holds a record absent from `knownIds`, proving a generation whose POST
 * response was lost did reach the server. A saved record is unpinned and newest, so it is the first
 * UNPINNED row; reading row 0 instead would find a pin we already knew and report the run as never
 * submitted.
 */
export interface NewRecordProbeBaseline {
  /** Every gallery id the client had loaded when the POST went out. */
  knownIds: ReadonlySet<string>;
  /**
   * Whether the window can judge an unpinned row: it held one already, or it was the whole gallery.
   * An all-pinned window with more pages behind it cannot, since every unpinned row is unfamiliar
   * for never having been loaded, so "unknown" stops meaning "new".
   */
  canJudgeUnpinned: boolean;
}

/** Build the baseline from what the client currently has loaded. */
export function newRecordProbeBaseline<T extends FlaggableItem>(
  loaded: T[],
  hasMore: boolean,
  knownIds: ReadonlySet<string>,
): NewRecordProbeBaseline {
  return { knownIds, canJudgeUnpinned: loaded.some((i) => !i.pinned) || !hasMore };
}

export async function hasUnknownRecord<T extends FlaggableItem>(
  baseline: NewRecordProbeBaseline,
  fetchPage: (offset: number) => Promise<{ items: T[]; hasMore: boolean }>,
  pageSize: number,
  maxPages: number = NEW_RECORD_PROBE_MAX_PAGES,
): Promise<boolean> {
  // Nothing the listing can show would be conclusive, so refuse to claim proof. The caller then
  // reports the submission error, which is the loud failure rather than a silent false success.
  if (!baseline.canJudgeUnpinned) return false;
  for (let page = 0; page < maxPages; page += 1) {
    const { items, hasMore } = await fetchPage(page * pageSize);
    for (const record of items) {
      // A saved record is never pinned, so a pinned row is never the proof -- not even an unknown
      // one. With more pins than the client had loaded, treating an unfamiliar pin as evidence
      // would report a generation that never reached the server as finished.
      if (record.pinned) continue;
      // The first unpinned row is where a new record would be, so it alone decides.
      return !baseline.knownIds.has(record.id);
    }
    if (!hasMore || items.length === 0) return false;
  }
  return false;
}

/** In-flight tail of each key's chain, dropped once that key goes idle so the map stays small. */
const queues = new Map<string, Promise<unknown>>();

/**
 * Run `task` after everything already queued under `key`, so a burst of clicks reaches the server in
 * click order. Different keys stay parallel. A rejection does not break the chain, and is still
 * delivered to whoever awaited it.
 */
export function serializeById<T>(key: string, task: () => Promise<T>): Promise<T> {
  const previous = queues.get(key);
  const run = previous ? previous.then(task, task) : task();
  const settled = run.then(
    () => {},
    () => {},
  );
  queues.set(key, settled);
  void settled.then(() => {
    // Only the last link clears the key; an earlier one finishing must not drop a live chain.
    if (queues.get(key) === settled) queues.delete(key);
  });
  return run;
}

/** Drop an item that was archived or deleted; order among the rest is untouched. */
export function removeGalleryItem<T extends FlaggableItem>(items: T[], id: string): T[] {
  return items.filter((i) => i.id !== id);
}

/** Attempts a page fetch may make before giving up, so a burst of actions cannot spin it. */
export const PAGE_MAX_ATTEMPTS = 4;

/**
 * Run `fetch`, and use its result only if `token` held still. For a GET whose response REPLACES the
 * strip: the backend can snapshot flags before a pin or archive and answer after it, reverting an
 * action the user was told had succeeded, with nothing scheduled to correct it. Both pages render
 * from a module cache while this runs, so their tiles are actionable throughout.
 *
 * It retries rather than fails, since the server now agrees with the local change. Null after
 * `maxAttempts` leaves the caller its optimistic state, which is what the server just confirmed.
 */
export async function fetchWhileStable<T>(
  token: () => number,
  fetch: () => Promise<T>,
  maxAttempts: number = PAGE_MAX_ATTEMPTS,
): Promise<T | null> {
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const before = token();
    const result = await fetch();
    if (token() === before) return result;
  }
  return null;
}

/**
 * Fetch the next page at an offset that is still true when the response lands.
 *
 * Archiving or deleting shortens the server's shelf under an in-flight request: with 50 loaded,
 * losing one moves the record at index 50 down to 49, so a page starting at 50 begins after it and
 * no page ever returns it. A short final page can then set `hasMore` false, so nothing looks again.
 *
 * All three guards are needed, because each covers a different part of the round trip.
 * `count()` catches the row being dropped DURING the fetch. `token()`, bumped when the request
 * starts, catches a mutation beginning during it. `pending()` catches the gap between those two,
 * where a page begins after the bump and ends before the drop and sees both hold still across a
 * shelf the server has already shortened.
 */
export async function fetchNextPage<T>(
  count: () => number,
  token: () => number,
  pending: () => number,
  fetchPage: (offset: number) => Promise<T>,
  maxAttempts: number = PAGE_MAX_ATTEMPTS,
): Promise<{ page: T; offset: number } | null> {
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const offset = count();
    const before = token();
    const page = await fetchPage(offset);
    if (pending() === 0 && count() === offset && token() === before) {
      return { page, offset };
    }
  }
  return null;
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
