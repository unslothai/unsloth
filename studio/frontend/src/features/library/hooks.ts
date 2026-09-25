// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { translate } from "@/i18n";
import { type RefObject, useEffect, useState } from "react";
import {
  type LibraryItem,
  errorMessage,
  fetchLibraryBlob,
  fetchLibraryStreamUrl,
  fetchLibraryThumbnail,
} from "./api";
import { type EmbeddedBody, embeddedBlobType, streamsPreview } from "./file-name";
import { acquireObjectUrl } from "./object-url-cache";

// Past this a preview is not buffered into memory as a blob: the file is a download away.
const MAX_BUFFERED_PREVIEW_BYTES = 256 * 1024 * 1024;

/**
 * Where the preview element loads the item from. Audio and video with a file of their own stream
 * from a short-lived signed link, whatever their size. The rest, and those too where an older
 * server mints no link, come as a typed blob (see embeddedBlobType), refused past a size cap.
 */
async function previewSource(
  item: LibraryItem,
  body: EmbeddedBody,
): Promise<{ url: string; streamed: boolean }> {
  if (streamsPreview(item.id, body)) {
    try {
      return { url: await fetchLibraryStreamUrl(item), streamed: true };
    } catch {
      // An older server without the route: the blob below still plays it.
    }
  }
  if (item.sizeBytes !== null && item.sizeBytes > MAX_BUFFERED_PREVIEW_BYTES) {
    throw new Error(translate("library.preview.tooLargeToPreview"));
  }
  const blob = await fetchLibraryBlob(item, embeddedBlobType(body, item.contentType));
  return { url: URL.createObjectURL(blob), streamed: false };
}

/**
 * A URL the preview's `body` element can load the item from once `enabled`, null until then;
 * `error` once it cannot load. Not cached: a preview's file is released as soon as it closes.
 * `retry` is for the element's error: a streamed link gets minted afresh once (it expired, or the
 * backend restarted), and it says whether it did; otherwise the error stands.
 */
export function useLibraryPreviewUrl(
  item: LibraryItem,
  body: EmbeddedBody | null,
): { url: string | null; error: string | null; retry: () => boolean } {
  const baseKey = `${item.id}@${item.updatedAt}:${body ?? ""}`;
  // The preview whose link was minted again, so another file (or version) starts afresh.
  const [reminted, setReminted] = useState<string | null>(null);
  const key = `${baseKey}#${reminted === baseKey ? 1 : 0}`;
  const [state, setState] = useState<{
    key: string;
    url: string | null;
    streamed?: boolean;
    error?: string;
  } | null>(null);
  useEffect(() => {
    if (!body) return;
    let cancelled = false;
    const next = previewSource(item, body);
    next.then(
      ({ url, streamed }) => !cancelled && setState({ key, url, streamed }),
      (err: unknown) => !cancelled && setState({ key, url: null, error: errorMessage(err) }),
    );
    return () => {
      cancelled = true;
      void next.then(({ url, streamed }) => !streamed && URL.revokeObjectURL(url), () => {});
    };
    // `key` carries the item's identity and version; the object itself changes on every refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  const current = body && state?.key === key ? state : null;
  const retry = () => {
    // A signed link expires, or dies with a backend restart: one fresh link per opening.
    if (!current?.url || !current.streamed || reminted === baseKey) return false;
    setReminted(baseKey);
    return true;
  };
  return { url: current?.url ?? null, error: current?.error ?? null, retry };
}

/** A card's picture once `enabled`: a bounded thumbnail of the image, or a video's first frame,
 *  never the original, whose decoded size has no limit. Cached, and held while the card shows it;
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
    const { url: next, release } = acquireObjectUrl(key, () => fetchLibraryThumbnail(item));
    next.then(
      (url) => !cancelled && setState({ key, url }),
      () => !cancelled && setState({ key, url: null }),
    );
    return () => {
      cancelled = true;
      release();
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
