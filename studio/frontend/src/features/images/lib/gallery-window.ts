// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GalleryImage, GalleryPage } from "../api.ts";

/** Pages from the top to `count` records, and up to one page further while a selection `loaded` held is missing (new
 *  records push it down). One it never held was deleted elsewhere: looking for it would widen every refresh by a page.
 *  `selectedId` is re-read per fetch, keeping mid-request picks. */
export async function collectGalleryWindow(
  fetchPage: (offset: number, limit: number) => Promise<GalleryPage>,
  loaded: readonly GalleryImage[],
  count: number,
  pageSize: number,
  selectedId: () => string | null,
): Promise<{ images: GalleryImage[]; hasMore: boolean }> {
  const held = new Set(loaded.map((image) => image.id));
  const images: GalleryImage[] = [];
  let hasMore = false;
  let ended = false;
  while (!ended) {
    const id = selectedId();
    const keep = id && held.has(id) ? id : null;
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
