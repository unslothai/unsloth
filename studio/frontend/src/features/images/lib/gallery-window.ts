// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GalleryImage, GalleryPage } from "../api.ts";

/** Pages the gallery from the top until it holds `count` records and, when given, the record `keepId`. New
 *  records land at the front and push the selection past the old count, so it is looked for one page further.
 *  No further: a selection deleted elsewhere must not page through the whole gallery. */
export async function collectGalleryWindow(
  fetchPage: (offset: number, limit: number) => Promise<GalleryPage>,
  count: number,
  pageSize: number,
  keepId: string | null,
): Promise<{ images: GalleryImage[]; hasMore: boolean }> {
  const reach = keepId ? count + pageSize : count;
  const images: GalleryImage[] = [];
  let hasMore = false;
  while (images.length < reach) {
    if (images.length >= count && (!keepId || images.some((image) => image.id === keepId))) break;
    const page = await fetchPage(images.length, Math.min(pageSize, reach - images.length));
    images.push(...page.images);
    hasMore = page.has_more;
    if (!page.has_more || page.images.length === 0) break;
  }
  return { images, hasMore };
}
