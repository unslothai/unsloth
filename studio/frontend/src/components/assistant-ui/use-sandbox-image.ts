// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { authFetch } from "@/features/auth";
import { type RefObject, useEffect, useRef, useState } from "react";

export type SandboxImageState =
  | { status: "idle" }
  | { status: "loaded"; url: string; blob: Blob }
  | { status: "failed" };

const IDLE: SandboxImageState = { status: "idle" };

/** The route needs the Authorization header, so a bare <img src> 401s. Keyed by url. */
export function useSandboxImage(url: string | null): {
  ref: RefObject<HTMLImageElement | null>;
  state: SandboxImageState;
} {
  const ref = useRef<HTMLImageElement>(null);
  const [state, setState] = useState<{
    url: string | null;
    load: SandboxImageState;
    signal?: AbortSignal;
  }>({
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
          if (!controller.signal.aborted)
            setState({
              url,
              load: { status: "failed" },
              signal: controller.signal,
            });
          return;
        }
        const blob = await response.blob();
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        setState({
          url,
          load: { status: "loaded", url: objectUrl, blob },
          signal: controller.signal,
        });
      })
      .catch(() => {
        if (!controller.signal.aborted)
          setState({
            url,
            load: { status: "failed" },
            signal: controller.signal,
          });
      });

    return () => {
      controller.abort();
      // Revoke even if aborted, or the bytes stay pinned for the session.
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [url, nearViewport]);

  // Returning to the same URL must not reuse a result whose fetch was cleaned up.
  return {
    ref,
    state: state.url === url && !state.signal?.aborted ? state.load : IDLE,
  };
}
