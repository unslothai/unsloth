// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { type ViewerImage, openImageViewer } from "@/components/image-viewer";
import { authFetch } from "@/features/auth";
import { type SearchImageEntry, searchImagePath } from "@/features/chat";
import { openLink } from "@/lib/open-link";
import { cn } from "@/lib/utils";
import {
  createContext,
  memo,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

export const SearchImagesContext = createContext<
  ReadonlyMap<string, SearchImageEntry>
>(new Map());

type LoadState =
  | { status: "idle" }
  | { status: "loaded"; url: string }
  | { status: "failed" };

const IDLE: LoadState = { status: "idle" };

function useSearchThumbnail(id: string, nearViewport: boolean): LoadState {
  // Keyed by id so a reused element reads idle for a new image without resetting state in the effect.
  const [state, setState] = useState<{ id: string; load: LoadState }>({
    id,
    load: IDLE,
  });

  useEffect(() => {
    if (!nearViewport) return;
    const controller = new AbortController();
    let objectUrl: string | null = null;

    authFetch(searchImagePath(id), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          // Guard like the success path, or a stale id's state is written back and the skeleton never resolves.
          if (controller.signal.aborted) return;
          setState({ id, load: { status: "failed" } });
          return;
        }
        const blob = await response.blob();
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        setState({ id, load: { status: "loaded", url: objectUrl } });
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setState({ id, load: { status: "failed" } });
        }
      });

    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [id, nearViewport]);

  return state.id === id ? state.load : IDLE;
}

function viewerImage(entry: SearchImageEntry): ViewerImage {
  const title = entry.title || entry.domain || "Image";
  return {
    key: entry.id,
    title,
    fileName: `${
      title
        .replace(/[\\/:*?"<>|]+/g, " ")
        .trim()
        .slice(0, 80) || "image"
    }.jpg`,
    source: entry.source,
    load: async () => {
      const response = await authFetch(searchImagePath(entry.id));
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.blob();
    },
  };
}

function useNearViewport<T extends Element>() {
  const ref = useRef<T>(null);
  const [near, setNear] = useState(
    () => typeof IntersectionObserver === "undefined",
  );
  useEffect(() => {
    if (near) return;
    const element = ref.current;
    if (!element) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setNear(true);
          observer.disconnect();
        }
      },
      { rootMargin: "200px" },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [near]);
  return [ref, near] as const;
}

export function SearchImageThumb({
  entry,
  className,
  size = "card",
}: {
  entry: SearchImageEntry;
  className?: string;
  size?: "card" | "strip";
}) {
  const gallery = useContext(SearchImagesContext);
  const [ref, nearViewport] = useNearViewport<HTMLAnchorElement>();
  const image = useSearchThumbnail(entry.id, nearViewport);
  const href = image.status === "loaded" ? entry.source : undefined;
  const label = entry.title || entry.domain || "Image";

  if (image.status === "failed") return null;

  return (
    <a
      ref={ref}
      href={href}
      rel="noopener noreferrer"
      title={entry.domain ? `${label} · ${entry.domain}` : label}
      aria-label={label}
      onClick={(event) => {
        if (!href) return;
        event.preventDefault();
        if (event.metaKey || event.ctrlKey || event.shiftKey) {
          openLink(href);
          return;
        }
        const entries = [
          ...new Map(
            [...gallery.values()].map((item) => [item.id, item]),
          ).values(),
        ];
        const images =
          entries.length > 0 ? entries.map(viewerImage) : [viewerImage(entry)];
        openImageViewer(
          images,
          Math.max(
            0,
            images.findIndex((image) => image.key === entry.id),
          ),
        );
      }}
      className={cn(
        "group inline-flex shrink-0 flex-col overflow-hidden rounded-lg border border-border bg-muted/40 text-left no-underline transition-colors hover:border-primary/50",
        size === "card" ? "max-w-[320px]" : "size-16",
        className,
      )}
    >
      <span
        className={cn(
          "block overflow-hidden bg-muted",
          size === "card" ? "h-40 w-full min-w-[160px]" : "size-full",
        )}
      >
        {image.status === "loaded" ? (
          <img
            src={image.url}
            alt={label}
            loading="lazy"
            decoding="async"
            className="size-full object-cover"
          />
        ) : (
          <span className="block size-full animate-pulse bg-muted" />
        )}
      </span>
      {size === "card" && (
        <span className="flex min-w-0 flex-col gap-0.5 px-2 py-1.5">
          {entry.title && (
            <span className="truncate text-xs font-medium text-foreground">
              {entry.title}
            </span>
          )}
          {entry.domain && (
            <span className="truncate text-ui-10 text-muted-foreground">
              {entry.domain}
            </span>
          )}
        </span>
      )}
    </a>
  );
}

// A token the message cannot resolve renders nothing, so an invented one never shows.
export const SearchImageElement = memo(function SearchImageElement(props: {
  token?: string;
}) {
  const images = useContext(SearchImagesContext);
  const entry = props.token ? images.get(props.token) : undefined;
  if (!entry) return null;
  // Explicitly block: list items make paragraphs inline, so text would wrap around the card.
  return (
    <span
      className="my-2 flex flex-wrap gap-2 empty:hidden"
      data-search-image={entry.id}
    >
      <SearchImageThumb entry={entry} />
    </span>
  );
});
