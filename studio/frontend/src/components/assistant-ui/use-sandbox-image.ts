// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { authFetch } from "@/features/auth";
import { type RefObject, useEffect, useRef, useState } from "react";

export type SandboxImageState =
  | { status: "idle" }
  | { status: "loaded"; url: string }
  | { status: "failed" };

const IDLE: SandboxImageState = { status: "idle" };

/**
 * Fetches one auth-protected sandbox file into an object URL.
 *
 * The route answers on the Authorization header (`_authenticate_header_or_query`), so a bare
 * `<img src>` hitting it straight gets a 401 and the renderer's "Image not available" placeholder.
 * Pass the URL from `sandboxFilePath()`/`markdownSandboxImageSrc()` and render what comes back.
 *
 * Keyed by url: an element that moves to another file reads idle rather than showing the previous
 * file's blob, and a stale response can never write state for a url it was not fetched for.
 */
export function useSandboxImage(url: string | null): {
  ref: RefObject<HTMLImageElement | null>;
  state: SandboxImageState;
} {
  const ref = useRef<HTMLImageElement>(null);
  const [state, setState] = useState<{ url: string | null; load: SandboxImageState }>({
    url,
    load: IDLE,
  });
  const [nearViewport, setNearViewport] = useState(
    () => typeof IntersectionObserver === "undefined",
  );

  useEffect(() => {
    if (nearViewport || !url) return;
    const element = ref.current;
    if (!element) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setNearViewport(true);
          observer.disconnect();
        }
      },
      { rootMargin: "200px" },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [nearViewport, url]);

  useEffect(() => {
    if (!url || !nearViewport) return;
    const controller = new AbortController();
    let objectUrl: string | null = null;

    authFetch(url, { signal: controller.signal })
      .then(async (response) => {
        // Guarded on both arms: a late write for the url this element used to hold must not land.
        if (!response.ok) {
          if (!controller.signal.aborted) setState({ url, load: { status: "failed" } });
          return;
        }
        const blob = await response.blob();
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        setState({ url, load: { status: "loaded", url: objectUrl } });
      })
      .catch(() => {
        if (!controller.signal.aborted) setState({ url, load: { status: "failed" } });
      });

    return () => {
      controller.abort();
      // The URL is ours to give up: an aborted fetch that had already made one would otherwise
      // pin the bytes for the rest of the session.
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [url, nearViewport]);

  return { ref, state: state.url === url ? state.load : IDLE };
}
