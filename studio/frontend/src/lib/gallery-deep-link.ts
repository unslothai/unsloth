// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const MAX_PAGES = 100;
const BUSY_WAIT_MS = 200;
const MAX_FRUITLESS = 3;

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
  busy: () => boolean;
  cancelled: () => boolean;
}): Promise<boolean> {
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
