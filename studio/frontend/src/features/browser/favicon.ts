// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fetchBrowserPage } from "./api";

const MAX_ICON_BYTES = 256 * 1024;
const MAX_ICONS = 64;

// Favicon URL to a blob URL, oldest first.
const icons = new Map<string, Promise<string | null>>();

async function load(url: string): Promise<string | null> {
  try {
    const page = await fetchBrowserPage({ url }, new AbortController().signal);
    if (page.kind !== "raw" || !page.contentType.startsWith("image/") || page.blob.size > MAX_ICON_BYTES) {
      return null;
    }
    return URL.createObjectURL(new Blob([page.blob], { type: page.contentType }));
  } catch {
    return null;
  }
}

/**
 * A page's favicon as a blob URL, fetched through the guarded proxy. Loading it
 * directly would let a page point Studio at a local or LAN address.
 */
export function proxiedFavicon(url: string): Promise<string | null> {
  if (/^data:image\//i.test(url)) return Promise.resolve(url);
  let icon = icons.get(url);
  if (!icon) {
    icon = load(url);
    icons.set(url, icon);
    if (icons.size > MAX_ICONS) {
      const [oldest, evicted] = icons.entries().next().value!;
      icons.delete(oldest);
      void evicted.then((blobUrl) => blobUrl?.startsWith("blob:") && URL.revokeObjectURL(blobUrl));
    }
  }
  return icon;
}
