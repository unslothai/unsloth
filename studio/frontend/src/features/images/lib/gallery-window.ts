// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GalleryImage, GalleryPage } from "../api.ts";

/** Pages the gallery from the top until it holds `count` records and the selected record. New records land at the
 *  front and push the selection past the old count, so it is looked for one page further, and no further: a
 *  selection deleted elsewhere must not page through the whole gallery. `selectedId` is read again after every
 *  fetch, and the last read follows the last fetch, so a pick made while a page is in flight is kept too. */
export async function collectGalleryWindow(
  fetchPage: (offset: number, limit: number) => Promise<GalleryPage>,
  count: number,
  pageSize: number,
  selectedId: () => string | null,
): Promise<{ images: GalleryImage[]; hasMore: boolean }> {
  const images: GalleryImage[] = [];
  let hasMore = false;
  let ended = false;
  while (!ended) {
    const keep = selectedId();
    const reach = keep ? count + pageSize : count;
    if (images.length >= reach) break;
    if (images.length >= count && (!keep || images.some((image) => image.id === keep))) break;
    const page = await fetchPage(images.length, Math.min(pageSize, reach - images.length));
    images.push(...page.images);
    hasMore = page.has_more;
    ended = !page.has_more || page.images.length === 0;
  }
  return { images, hasMore };
}
