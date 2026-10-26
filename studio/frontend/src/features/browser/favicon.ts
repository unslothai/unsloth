// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fetchBrowserPage } from "./api";

const MAX_ICON_BYTES = 256 * 1024;
const MAX_ICONS = 64;

const icons = new Map<string, Promise<string | null>>();

function dataUrl(blob: Blob): Promise<string | null> {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : null);
    reader.onerror = () => resolve(null);
    reader.readAsDataURL(blob);
  });
}

async function load(url: string): Promise<string | null> {
  try {
    const page = await fetchBrowserPage({ url, maxBytes: MAX_ICON_BYTES }, new AbortController().signal);
    if (page.kind !== "raw" || !page.contentType.startsWith("image/") || page.blob.size > MAX_ICON_BYTES) {
      return null;
    }
    // data:, not blob:, so eviction can't blank a tab and an SVG can't run on Studio's origin.
    return dataUrl(new Blob([page.blob], { type: page.contentType }));
  } catch {
    return null;
  }
}

/** A page's favicon as a data: URL via the guarded proxy; loaded directly, a page could aim Studio at a LAN address. */
export function proxiedFavicon(url: string): Promise<string | null> {
  if (/^data:image\//i.test(url)) return Promise.resolve(url);
  let icon = icons.get(url);
  if (!icon) {
    icon = load(url);
    icons.set(url, icon);
    if (icons.size > MAX_ICONS) {
      const oldest = icons.keys().next().value;
      if (oldest !== undefined) icons.delete(oldest);
    }
  }
  return icon;
}
