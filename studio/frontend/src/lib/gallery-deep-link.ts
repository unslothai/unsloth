// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Enough for thousands of items; past it the item is left unselected rather than paging forever.
const MAX_PAGES = 100;
// A page already loading makes loadMore a no-op, so wait for it instead of spinning.
const BUSY_WAIT_MS = 200;

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
  cancelled,
}: {
  has: () => boolean;
  count: () => number;
  hasMore: () => boolean;
  refresh: () => Promise<unknown>;
  loadMore: () => Promise<unknown>;
  cancelled: () => boolean;
}): Promise<boolean> {
  // The first page, fresh: the item may be newer than what this page last loaded.
  await refresh();
  for (let page = 0; page < MAX_PAGES; page++) {
    if (cancelled()) return false;
    if (has()) return true;
    if (!hasMore()) return false;
    const before = count();
    await loadMore();
    if (count() === before) await new Promise((resolve) => setTimeout(resolve, BUSY_WAIT_MS));
  }
  return !cancelled() && has();
}
