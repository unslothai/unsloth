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
 * The pinned group keeps the order it arrived in. The backend sorts it by PIN time, which the
 * client never learns, so re-sorting it by `created_at` would silently rearrange the pins whenever
 * an unrelated merge ran -- wrong whenever an older item was pinned more recently than a newer one.
 * Arriving order is the server's order, so leaving it alone is what keeps the two agreeing.
 *
 * `justPinnedId` is the one exception: a freshly pinned item goes to the very front, which is the
 * same "most recently pinned first" rule the backend will apply.
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
 * Undo a failed unpin, putting the item back where it WAS rather than at the front.
 *
 * `applyPin(..., true)` is the wrong rollback: it means "freshly pinned", so it promotes the item
 * to the head of the pinned group, and an item that had been third by pin time comes back first.
 * The server's order never changed, so the strip would sit wrong until a refetch.
 *
 * `order` is the pinned ids as they stood before the click. An id missing from it was pinned while
 * this request was in flight; `indexOf` scores those -1, which is exactly right, since the newest
 * pin leads. Ties keep their arrival order, which is the server's.
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
 * Whether the gallery holds a record absent from `knownIds`, used to prove that a generation whose
 * POST response was lost did in fact reach the server.
 *
 * A saved record is always unpinned and newest, so it is the FIRST unpinned row of the listing.
 * This walks the pinned prefix to reach that row instead of reading row 0: pinned records sort
 * ahead of everything, so with any pin present row 0 is a record we already knew, and the probe
 * would report a finished generation as never submitted.
 */
export interface NewRecordProbeBaseline {
  /** Every gallery id the client had loaded when the POST went out. */
  knownIds: ReadonlySet<string>;
  /**
   * Whether that loaded window is a sound basis for calling an unpinned row new.
   *
   * It is when the window already held an unpinned record, or when it was the entire gallery.
   * It is NOT when the window was all pinned with more pages behind it: every unpinned row is
   * then unfamiliar simply because it was never loaded, so "unknown" stops meaning "new".
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
 * Run `task` after every task already queued under `key`, so a burst of clicks reaches the server
 * in click order rather than whichever request happens to arrive first. Different keys stay
 * parallel, so one gallery's writes never wait on another's.
 *
 * A rejected task does not break the chain; the next one still runs, and the rejection is still
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
 * Run `fetch` and hand back its result only if `token` did not move while it ran.
 *
 * For a GET whose response REPLACES the strip. The backend can snapshot the flags before a pin or
 * archive and answer after it, so applying that response unconditionally reverts an action the
 * user was already told had succeeded: a pin reads as unpinned, an archived image comes back onto
 * the strip, and nothing refreshes it again. Both galleries render from a module cache while their
 * first-page load runs, so the tiles are actionable for the whole of that window.
 *
 * Retrying is right rather than failing: the local change is already applied and the server now
 * agrees, so a later read is the one worth having. Giving up after `maxAttempts` returns null and
 * the caller keeps what it has, which is the optimistic state the server just confirmed.
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
 * The offset is how many records are loaded, and archiving or deleting one shortens the server's
 * shelf under an in-flight request: with 50 loaded, archiving one mid-flight moves the record that
 * was at index 50 down to 49, so a page starting at 50 begins AFTER it and that record is returned
 * by no page at all. It then stays missing until a reload, and a short final page can set
 * `hasMore` false so nothing ever goes looking for it again.
 *
 * `count()` is read again after the response, from live state rather than a captured number, and
 * the fetch is retried at the corrected offset when it moved. Same shape as the archived list's
 * `showMore`, which had this bug first.
 *
 * `token()` is the second half, and the count alone is not enough without it. The server shelf
 * shortens when it PROCESSES the archive, while the count only moves when that response gets back
 * and the row is dropped locally. A page read inside that gap sees the shortened shelf at the old
 * offset and the count agrees with itself, so the boundary record is skipped with nothing to
 * notice. Callers bump the token when the request STARTS, which covers the whole round trip.
 */
export async function fetchNextPage<T>(
  count: () => number,
  token: () => number,
  fetchPage: (offset: number) => Promise<T>,
  maxAttempts: number = PAGE_MAX_ATTEMPTS,
): Promise<{ page: T; offset: number } | null> {
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const offset = count();
    const before = token();
    const page = await fetchPage(offset);
    if (count() === offset && token() === before) return { page, offset };
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
