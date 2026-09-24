// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type RefObject, useEffect, useState } from "react";
import { type LibraryItem, fetchLibraryBlob } from "./api";
import { hasImagePreview } from "./file-kind";

// Thumbnails are auth-fetched blobs, so the browser cache cannot hold them. Keep the most recent
// ones as object URLs; a revisit of the page then paints instantly instead of refetching.
const MAX_CACHED_URLS = 300;
const objectUrls = new Map<string, Promise<string>>();

function objectUrlFor(item: LibraryItem): Promise<string> {
  const key = `${item.id}@${item.updatedAt}`;
  let url = objectUrls.get(key);
  if (!url) {
    url = fetchLibraryBlob(item).then((blob) => URL.createObjectURL(blob));
    url.catch(() => objectUrls.delete(key));
    objectUrls.set(key, url);
    if (objectUrls.size > MAX_CACHED_URLS) {
      const [oldestKey, oldest] = objectUrls.entries().next().value!;
      objectUrls.delete(oldestKey);
      void oldest.then((stale) => URL.revokeObjectURL(stale), () => {});
    }
  }
  return url;
}

/**
 * Object URL for an item's bytes once `enabled`; null until then or on failure. Only images share
 * the cache: a preview of a large clip or PDF is released as soon as it closes.
 */
export function useLibraryObjectUrl(
  item: LibraryItem,
  enabled: boolean,
): string | null {
  const key = `${item.id}@${item.updatedAt}`;
  const [state, setState] = useState<{ key: string; url: string | null } | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const cached = hasImagePreview(item);
    const next = cached
      ? objectUrlFor(item)
      : fetchLibraryBlob(item).then((blob) => URL.createObjectURL(blob));
    next.then(
      (url) => !cancelled && setState({ key, url }),
      () => !cancelled && setState({ key, url: null }),
    );
    return () => {
      cancelled = true;
      if (!cached) void next.then((url) => URL.revokeObjectURL(url), () => {});
    };
    // `key` carries the item's identity and version; the object itself changes on every refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled]);
  return enabled && state?.key === key ? state.url : null;
}

/** True once the element has come within a screen of the viewport; never flips back. */
export function useSeen(ref: RefObject<Element | null>): boolean {
  const [seen, setSeen] = useState(() => typeof IntersectionObserver === "undefined");
  useEffect(() => {
    const element = ref.current;
    if (!element || seen) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setSeen(true);
          observer.disconnect();
        }
      },
      { rootMargin: "400px" },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref, seen]);
  return seen;
}

/** Masonry column count for the container's width. */
export function useColumnCount(
  ref: RefObject<HTMLElement | null>,
  minColumnWidth = 200,
  maxColumns = 5,
): number {
  const [columns, setColumns] = useState(4);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const measure = () => {
      const width = element.clientWidth;
      setColumns(
        Math.max(2, Math.min(maxColumns, Math.floor(width / minColumnWidth))),
      );
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref, minColumnWidth, maxColumns]);
  return columns;
}
