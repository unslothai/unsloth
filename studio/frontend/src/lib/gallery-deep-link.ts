// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Enough for thousands of items; past it the item is left unselected rather than paging forever.
const MAX_PAGES = 100;
// A page already loading makes loadMore a no-op, so wait for it instead of spinning.
const BUSY_WAIT_MS = 200;
// Attempts that were not blocked by another load yet brought nothing: the gallery is failing (or a
// guarded page was dropped), so stop rather than retrying for every remaining page.
const MAX_FRUITLESS = 3;

/**
 * Page a gallery back until an item is loaded, for a link from the Library. Resolves true once it
 * is, false when the gallery runs out first (archived or deleted) or the link was superseded.
 */
export async function loadGalleryUntil({
  has,
  count,
  hasMore,
  refresh,
  loadMore,
  busy,
  cancelled,
}: {
  has: () => boolean;
  count: () => number;
  hasMore: () => boolean;
  refresh: () => Promise<unknown>;
  loadMore: () => Promise<unknown>;
  /** A page is already loading, so loadMore would return without trying. */
  busy: () => boolean;
  cancelled: () => boolean;
}): Promise<boolean> {
  // The first page, fresh: the item may be newer than what this page last loaded.
  await refresh();
  let fruitless = 0;
  for (let page = 0; page < MAX_PAGES; page++) {
    if (cancelled()) return false;
    if (has()) return true;
    if (!hasMore()) return false;
    const before = count();
    const waited = busy();
    await loadMore();
    if (count() !== before) {
      fruitless = 0;
      continue;
    }
    if (!waited && ++fruitless >= MAX_FRUITLESS) return false;
    await new Promise((resolve) => setTimeout(resolve, BUSY_WAIT_MS));
  }
  return !cancelled() && has();
}
