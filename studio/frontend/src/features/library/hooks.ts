// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type RefObject, useEffect, useState } from "react";
import { type LibraryItem, fetchLibraryBlob, fetchLibraryThumbnail } from "./api";
import { hasImagePreview } from "./file-kind";

// Thumbnails are auth-fetched blobs, so the browser cache cannot hold them. Keep the most recent
// ones as object URLs; a revisit of the page then paints instantly instead of refetching.
const MAX_CACHED_URLS = 300;
const objectUrls = new Map<string, Promise<string>>();

/** Drop every cached URL, so a sign-out leaves nothing of the last account's files. */
export function clearCachedObjectUrls(): void {
  for (const url of objectUrls.values()) void url.then((stale) => URL.revokeObjectURL(stale), () => {});
  objectUrls.clear();
}

function cachedObjectUrl(key: string, load: () => Promise<Blob>): Promise<string> {
  let url = objectUrls.get(key);
  if (!url) {
    url = load().then((blob) => URL.createObjectURL(blob));
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
 * Object URL for an item's bytes once `enabled`, null until then; `error` once it cannot load. Only
 * images share the cache: a preview of a large clip or PDF is released as soon as it closes.
 */
export function useLibraryObjectUrl(
  item: LibraryItem,
  enabled: boolean,
): { url: string | null; error: string | null } {
  const key = `${item.id}@${item.updatedAt}`;
  const [state, setState] = useState<{
    key: string;
    url: string | null;
    error?: string;
  } | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const cached = hasImagePreview(item);
    const next = cached
      ? cachedObjectUrl(key, () => fetchLibraryBlob(item))
      : fetchLibraryBlob(item).then((blob) => URL.createObjectURL(blob));
    next.then(
      (url) => !cancelled && setState({ key, url }),
      (err: unknown) =>
        !cancelled &&
        setState({ key, url: null, error: err instanceof Error ? err.message : String(err) }),
    );
    return () => {
      cancelled = true;
      if (!cached) void next.then((url) => URL.revokeObjectURL(url), () => {});
    };
    // `key` carries the item's identity and version; the object itself changes on every refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled]);
  const current = enabled && state?.key === key ? state : null;
  return { url: current?.url ?? null, error: current?.error ?? null };
}

/** A card's picture once `enabled`: the image itself, or a video's first frame. Cached like images;
 *  `failed` once it cannot load, so the card can fall back to the type icon. */
export function useLibraryThumbnail(
  item: LibraryItem,
  enabled: boolean,
): { url: string | null; failed: boolean } {
  const key = `${item.id}@${item.updatedAt}`;
  const [state, setState] = useState<{ key: string; url: string | null } | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const next = hasImagePreview(item)
      ? cachedObjectUrl(key, () => fetchLibraryBlob(item))
      : cachedObjectUrl(`thumbnail:${key}`, () => fetchLibraryThumbnail(item));
    next.then(
      (url) => !cancelled && setState({ key, url }),
      () => !cancelled && setState({ key, url: null }),
    );
    return () => {
      cancelled = true;
    };
    // `key` carries the item's identity and version; the object itself changes on every refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled]);
  const current = enabled && state?.key === key ? state : null;
  return { url: current?.url ?? null, failed: current !== null && current.url === null };
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
