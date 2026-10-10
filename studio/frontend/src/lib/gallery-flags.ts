// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Pure, shared by Images and Video. The backend owns order (pinned, then newest or manual key);
 * this only reproduces it so a click lands instantly. */

export const GALLERY_CHANGED_EVENT = "unsloth:gallery-changed";

export type GalleryKind = "images" | "videos" | "audio";

/** Both pages stay mounted and load only on mount, so a restore from Settings needs this. */
export function notifyGalleryChanged(kind: GalleryKind): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(GALLERY_CHANGED_EVENT, { detail: { kind } }));
}

/** Returns an unsubscriber. */
export function subscribeGalleryChanged(kind: GalleryKind, onChanged: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  const handler = (event: Event) => {
    const detail = (event as CustomEvent<{ kind?: string }>).detail;
    if (detail?.kind === kind) onChanged();
  };
  window.addEventListener(GALLERY_CHANGED_EVENT, handler);
  return () => window.removeEventListener(GALLERY_CHANGED_EVENT, handler);
}

export interface FlaggableItem {
  id: string;
  pinned?: boolean;
  archived?: boolean;
  /** Epoch seconds (drag key, else file mtime). */
  order_at?: number | null;
  created_at: number | string;
}

/** Images store seconds, videos an ISO string. */
function orderKey(item: FlaggableItem): number {
  if (typeof item.order_at === "number") return item.order_at;
  return typeof item.created_at === "number"
    ? item.created_at
    : Date.parse(item.created_at) / 1000;
}

function newestFirst<T extends FlaggableItem>(a: T, b: T): number {
  return orderKey(b) - orderKey(a);
}

/** Pins keep ARRIVAL order (the server sorts by pin time, which the client never learns), except
 * `justPinnedId`, which leads. */
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

/** Archiving is `removeGalleryItem`, not this. */
export function applyPin<T extends FlaggableItem>(items: T[], id: string, pinned: boolean): T[] {
  const next = items.map((i) => (i.id === id ? { ...i, pinned } : i));
  return sortGalleryItems(next, pinned ? id : undefined);
}

/** Backend pin rule: between pins pins it, between unpinned unpins it, on the boundary keeps it.
 * Returns `items` unchanged for a no-op. */
export function moveGalleryItem<T extends FlaggableItem>(
  items: T[],
  id: string,
  afterId: string | null,
): T[] {
  const from = items.findIndex((i) => i.id === id);
  if (from < 0 || afterId === id) return items;
  const rest = items.filter((i) => i.id !== id);
  const at = afterId === null ? 0 : rest.findIndex((i) => i.id === afterId) + 1;
  if (afterId !== null && at === 0) return items;
  const moved = items[from];
  const above = rest[at - 1];
  const below = rest[at];
  const pinned =
    above && below
      ? Boolean(above.pinned) === Boolean(below.pinned)
        ? Boolean(above.pinned)
        : Boolean(moved.pinned)
      : Boolean((above ?? below ?? moved).pinned);
  if (at === from && pinned === Boolean(moved.pinned)) return items;
  return [...rest.slice(0, at), { ...moved, pinned }, ...rest.slice(at)];
}

/** In a filtered view hidden neighbours would decide the pin, so keep the shown state. */
export function scopedMoveAfterId<T extends FlaggableItem>(
  items: T[],
  inView: (item: T) => boolean,
  id: string,
  afterId: string | null,
): string | null {
  const view = items.filter(inView);
  const intended = moveGalleryItem(view, id, afterId);
  if (intended === view) return afterId;
  const at = intended.findIndex((i) => i.id === id);
  const wantPinned = Boolean(intended[at]?.pinned);
  const above = intended[at - 1];
  const below = intended[at + 1];
  const rest = items.filter((i) => i.id !== id);
  const first = above ? rest.findIndex((i) => i.id === above.id) : -1;
  const last = below ? rest.findIndex((i) => i.id === below.id) : rest.length;
  for (let gap = first; gap < last; gap++) {
    const candidate = gap < 0 ? null : rest[gap].id;
    const moved = moveGalleryItem(items, id, candidate).find((i) => i.id === id);
    if (Boolean(moved?.pinned) === wantPinned) return candidate;
  }
  return afterId;
}

export function pinnedOrder<T extends FlaggableItem>(items: T[]): string[] {
  return items.filter((i) => i.pinned).map((i) => i.id);
}

/** `applyPin(..., true)` would promote it to the head. Ids absent from `order` were pinned
 * mid-request and score -1, so the newest pin leads. */
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

/** Drops any a concurrent load already brought in. */
export function mergeGenerated<T extends FlaggableItem>(items: T[], fresh: T[]): T[] {
  const known = new Set(items.map((i) => i.id));
  return sortGalleryItems([...fresh.filter((i) => !known.has(i.id)), ...items]);
}

/** Bounds a recovery probe so a huge pin count cannot scan the gallery. */
export const NEW_RECORD_PROBE_MAX_PAGES = 5;

/** Proves a generation whose POST response was lost reached the server. A saved record is the
 * first UNPINNED row, not row 0. */
export interface NewRecordProbeBaseline {
  knownIds: ReadonlySet<string>;
  /** An all-pinned window with more pages behind it cannot judge, since every unpinned row is
   * unfamiliar. */
  canJudgeUnpinned: boolean;
}

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
  // Refuse to claim proof; the caller reports the submission error loudly.
  if (!baseline.canJudgeUnpinned) return false;
  for (let page = 0; page < maxPages; page += 1) {
    const { items, hasMore } = await fetchPage(page * pageSize);
    for (const record of items) {
      // A saved record is never pinned, so an unfamiliar pin is not evidence.
      if (record.pinned) continue;
      return !baseline.knownIds.has(record.id);
    }
    if (!hasMore || items.length === 0) return false;
  }
  return false;
}

/** Dropped once a key goes idle so the map stays small. */
const queues = new Map<string, Promise<unknown>>();

/** Click order per key; keys stay parallel. A rejection does not break the chain. */
export function serializeById<T>(key: string, task: () => Promise<T>): Promise<T> {
  const previous = queues.get(key);
  const run = previous ? previous.then(task, task) : task();
  const settled = run.then(
    () => {},
    () => {},
  );
  queues.set(key, settled);
  void settled.then(() => {
    // Only the last link clears the key.
    if (queues.get(key) === settled) queues.delete(key);
  });
  return run;
}

export function removeGalleryItem<T extends FlaggableItem>(items: T[], id: string): T[] {
  return items.filter((i) => i.id !== id);
}

/** So a burst of actions cannot spin it. */
export const PAGE_MAX_ATTEMPTS = 4;

/** For a GET that REPLACES the strip: a response snapshotted before a pin or archive would revert
 * it. Retries rather than fails; null after `maxAttempts` keeps the optimistic state. */
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
 * Archiving shortens the server's shelf mid-request, so a page at the old offset skips a record.
 * `count()` catches a drop during the fetch, `token()` a mutation starting during it, and
 * `pending()` the gap between those two.
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

/** Falls to a neighbour so the preview never blanks with items on screen; `remaining` excludes
 * the removed item. */
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
