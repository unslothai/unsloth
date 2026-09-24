// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type RefObject, useEffect, useState } from "react";
import { type LibraryItem, fetchLibraryBlob, fetchLibraryThumbnail } from "./api";
import { hasImagePreview } from "./file-kind";

// Thumbnails are auth-fetched blobs, so the browser cache cannot hold them. Keep the most recent
// ones as object URLs, within a count and a byte budget; a revisit of the page then paints
// instantly instead of refetching.
const MAX_CACHED_URLS = 300;
const MAX_CACHED_BYTES = 128 * 1024 * 1024;
// A card loads an image this small as it is, animation and all; a larger one gets a thumbnail.
const MAX_ORIGINAL_THUMB_BYTES = 2 * 1024 * 1024;
const objectUrls = new Map<string, { url: Promise<string>; bytes: number }>();
let cachedBytes = 0;

/** Drop every cached URL, so a sign-out leaves nothing of the last account's files. */
export function clearCachedObjectUrls(): void {
  for (const { url } of objectUrls.values()) {
    void url.then((stale) => URL.revokeObjectURL(stale), () => {});
  }
  objectUrls.clear();
  cachedBytes = 0;
}

function evict(key: string): void {
  const entry = objectUrls.get(key);
  if (!entry) return;
  objectUrls.delete(key);
  cachedBytes -= entry.bytes;
  void entry.url.then((stale) => URL.revokeObjectURL(stale), () => {});
}

function cachedObjectUrl(key: string, load: () => Promise<Blob>): Promise<string> {
  const hit = objectUrls.get(key);
  if (hit) {
    // Most recently used goes last, so eviction takes what has gone unseen longest.
    objectUrls.delete(key);
    objectUrls.set(key, hit);
    return hit.url;
  }
  const entry = { url: Promise.resolve(""), bytes: 0 };
  entry.url = load().then((blob) => {
    if (objectUrls.get(key) === entry) {
      entry.bytes = blob.size;
      cachedBytes += blob.size;
      for (const oldest of objectUrls.keys()) {
        if (cachedBytes <= MAX_CACHED_BYTES || oldest === key) break;
        evict(oldest);
      }
    }
    return URL.createObjectURL(blob);
  });
  entry.url.catch(() => {
    if (objectUrls.get(key) === entry) evict(key);
  });
  objectUrls.set(key, entry);
  if (objectUrls.size > MAX_CACHED_URLS) evict(objectUrls.keys().next().value!);
  return entry.url;
}

/** Whether a card shows the image itself rather than a thumbnail the server makes. */
function showsOriginal(item: LibraryItem): boolean {
  return (
    hasImagePreview(item) && item.sizeBytes !== null && item.sizeBytes <= MAX_ORIGINAL_THUMB_BYTES
  );
}

/**
 * Object URL for an item's bytes once `enabled`, null until then; `error` once it cannot load. Only
 * small images share the cache: a preview of anything larger is released as soon as it closes.
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
    // Only a small image is shared with the cards; a large one is released once the preview closes.
    const cached = showsOriginal(item);
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

/** A card's picture once `enabled`: a small image itself, else a bounded thumbnail of the image or
 *  a video's first frame. Cached; `failed` once it cannot load, so the card can fall back to the
 *  type icon. */
export function useLibraryThumbnail(
  item: LibraryItem,
  enabled: boolean,
): { url: string | null; failed: boolean } {
  const key = `${item.id}@${item.updatedAt}`;
  const [state, setState] = useState<{ key: string; url: string | null } | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const next = showsOriginal(item)
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
