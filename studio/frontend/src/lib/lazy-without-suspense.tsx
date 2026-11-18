// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ComponentType, useEffect, useReducer, useState } from "react";
import { onceLoader } from "./once-loader";

export type LazyWithoutSuspense<P extends object> = ComponentType<P> & {
  preload: () => Promise<ComponentType<P>>;
};

/**
 * Skips the React 19 300 ms Suspense reveal hold: null until the module loads, then renders at
 * once; a failed import throws from render and retries on next mount
 */
export function lazyWithoutSuspense<P extends object>(
  load: () => Promise<ComponentType<P>>,
): LazyWithoutSuspense<P> {
  const loader = onceLoader(load);

  function Deferred(props: P) {
    const [, rerender] = useReducer((count: number) => count + 1, 0);
    const [failure, setFailure] = useState<{ error: unknown } | null>(null);
    const Component = loader.current();
    const renderedLoaded = Component !== null;
    useEffect(() => {
      // A load that finished between this render and the effect still re-renders: load() resolves at once.
      if (renderedLoaded) return;
      let live = true;
      loader.load().then(
        () => {
          if (live) rerender();
        },
        (error: unknown) => {
          if (live) setFailure({ error });
        },
      );
      return () => {
        live = false;
      };
    }, [renderedLoaded]);
    if (failure) throw failure.error;
    return Component ? <Component {...props} /> : null;
  }

  return Object.assign(Deferred, { preload: loader.load });
}

type IdleWindow = Window & {
  requestIdleCallback?: (callback: () => void, options?: { timeout: number }) => number;
  cancelIdleCallback?: (handle: number) => void;
};

/** Starts `preload` once the page is idle (a timer where idle callbacks are missing, as in Safari); returns a cancel. */
export function preloadWhenIdle(preload: () => Promise<unknown>, fallbackDelayMs = 1500): () => void {
  const warm = () => {
    // A failed warm-up is retried by the next mount; the panel itself reports the error.
    preload().catch(() => {});
  };
  const idle = window as IdleWindow;
  if (typeof idle.requestIdleCallback === "function") {
    const handle = idle.requestIdleCallback(warm, { timeout: 4000 });
    return () => idle.cancelIdleCallback?.(handle);
  }
  const handle = window.setTimeout(warm, fallbackDelayMs);
  return () => window.clearTimeout(handle);
}
