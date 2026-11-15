// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GalleryImage, GalleryPage } from "../api.ts";

/** Pages from the top to `count` records, plus up to one page for a selection `loaded` held (new records push it down;
 *  one it never held was deleted elsewhere). Re-reads `selectedId` per fetch; null once `stillFresh` fails. */
export async function collectGalleryWindow(
  fetchPage: (offset: number, limit: number) => Promise<GalleryPage>,
  loaded: readonly GalleryImage[],
  count: number,
  pageSize: number,
  selectedId: () => string | null,
  stillFresh: () => boolean = () => true,
): Promise<{ images: GalleryImage[]; hasMore: boolean } | null> {
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
    if (!stillFresh()) return null;
    images.push(...page.images);
    hasMore = page.has_more;
    ended = !page.has_more || page.images.length === 0;
  }
  return { images, hasMore };
}
