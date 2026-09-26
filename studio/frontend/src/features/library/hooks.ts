// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MAX_DOCUMENT_PREVIEW_BYTES } from "@/components/file-viewer";
import { translate } from "@/i18n";
import { type RefObject, useEffect, useState } from "react";
import {
  LibraryFileTooLarge,
  type LibraryItem,
  errorMessage,
  fetchLibraryBlob,
  fetchLibraryStreamUrl,
  fetchLibraryThumbnail,
} from "./api";
import { type EmbeddedBody, embeddedBlobType, itemVersion, streamsPreview } from "./file-name";
import { acquireObjectUrl } from "./object-url-cache";

const MAX_BUFFERED_PREVIEW_BYTES = 256 * 1024 * 1024;

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
  const tooLarge = () => new Error(translate("library.preview.tooLargeToPreview"));
  if (item.sizeBytes !== null && item.sizeBytes > MAX_BUFFERED_PREVIEW_BYTES) throw tooLarge();
  const type = embeddedBlobType(body, item.contentType);
  const blob = await fetchLibraryBlob(item, type, MAX_BUFFERED_PREVIEW_BYTES).catch((error) => {
    throw error instanceof LibraryFileTooLarge ? tooLarge() : error;
  });
  return { url: URL.createObjectURL(blob), streamed: false };
}

export function useLibraryPreviewUrl(
  item: LibraryItem,
  body: EmbeddedBody | null,
): { url: string | null; error: string | null; retry: () => boolean } {
  const baseKey = `${itemVersion(item)}:${body ?? ""}`;
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
    if (!current?.url || !current.streamed || reminted === baseKey) return false;
    setReminted(baseKey);
    return true;
  };
  return { url: current?.url ?? null, error: current?.error ?? null, retry };
}

/** The item's bytes for a document viewer once `enabled`, or why they cannot be shown. */
export function useLibraryDocument(
  item: LibraryItem,
  enabled: boolean,
): { file: Blob | null; error: string | null } {
  const key = itemVersion(item);
  const [state, setState] = useState<{ key: string; file?: Blob; error?: string } | null>(null);
  const tooLarge = item.sizeBytes !== null && item.sizeBytes > MAX_DOCUMENT_PREVIEW_BYTES;
  useEffect(() => {
    if (!enabled || tooLarge) return;
    let cancelled = false;
    fetchLibraryBlob(item, item.contentType, MAX_DOCUMENT_PREVIEW_BYTES).then(
      (file) => !cancelled && setState({ key, file }),
      (err: unknown) =>
        !cancelled &&
        setState({
          key,
          error:
            err instanceof LibraryFileTooLarge
              ? translate("library.preview.tooLargeToPreview")
              : errorMessage(err),
        }),
    );
    return () => {
      cancelled = true;
    };
    // `key` carries the item's identity and version; the object itself changes on every refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled]);
  if (enabled && tooLarge) return { file: null, error: translate("library.preview.tooLargeToPreview") };
  const current = enabled && state?.key === key ? state : null;
  return { file: current?.file ?? null, error: current?.error ?? null };
}

/** A card's picture once `enabled`: a bounded thumbnail of the image, or a video's first frame,
 *  never the original, whose decoded size has no limit. Cached, and held while the card shows it;
 *  `failed` once it cannot load, so the card can fall back to the type icon. */
export function useLibraryThumbnail(
  item: LibraryItem,
  enabled: boolean,
): { url: string | null; failed: boolean } {
  const key = itemVersion(item);
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
      // Each column but the last also takes a gap.
      const gap = parseFloat(getComputedStyle(element).columnGap) || 0;
      const fit = Math.floor((element.clientWidth + gap) / (minColumnWidth + gap));
      setColumns(Math.max(2, Math.min(maxColumns, fit)));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref, minColumnWidth, maxColumns]);
  return columns;
}
