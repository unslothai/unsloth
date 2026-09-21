// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

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

// Provided once per message part by MarkdownText, so every chip reads the same map.
export const SearchImagesContext = createContext<
  ReadonlyMap<string, SearchImageEntry>
>(new Map());

type LoadState =
  | { status: "idle" }
  | { status: "loaded"; url: string }
  | { status: "failed" };

const IDLE: LoadState = { status: "idle" };

function useSearchThumbnail(id: string, nearViewport: boolean): LoadState {
  // Keyed by id so a re-used element for another image reads idle, not the
  // previous image's blob, without resetting state inside the effect.
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
          // Guarded like the success path below: a non-ok response for the id this
          // element used to hold would otherwise write that id's state back, and the
          // render below falls through to idle for it -- a skeleton that never resolves
          // because the effect has no reason to run again.
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
  const [ref, nearViewport] = useNearViewport<HTMLAnchorElement>();
  const image = useSearchThumbnail(entry.id, nearViewport);
  // Only a card the backend actually served gets a live link.
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
        if (href && openLink(href)) event.preventDefault();
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

// Rendered by Streamdown for <search-image token="…">; a token the message cannot
// resolve renders nothing, so an invented one never shows as text or markup.
export const SearchImageElement = memo(function SearchImageElement(props: {
  token?: string;
}) {
  const images = useContext(SearchImagesContext);
  const entry = props.token ? images.get(props.token) : undefined;
  if (!entry) return null;
  // Explicitly block: a list item styles its paragraphs `[&>p]:inline`, so an
  // inline card would flow into the sentence and the text would wrap around it.
  // `empty:hidden` keeps a thumb that failed to load from leaving a gap.
  return (
    <span
      className="my-2 flex flex-wrap gap-2 empty:hidden"
      data-search-image={entry.id}
    >
      <SearchImageThumb entry={entry} />
    </span>
  );
});
