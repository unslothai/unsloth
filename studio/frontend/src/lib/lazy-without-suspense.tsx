// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ComponentType, useEffect, useReducer, useState } from "react";
import { onceLoader } from "./once-loader";

export type LazyWithoutSuspense<P extends object> = ComponentType<P> & {
  preload: () => Promise<ComponentType<P>>;
};

/**
 * A code-split component that never suspends. React 19 holds a Suspense reveal for up to
 * 300 ms after its fallback commits, so a `lazy()` panel waits that long on its first open
 * even when its chunk is already loaded. This renders nothing until the module resolves,
 * then the component itself; once loaded it renders synchronously. A failed import is
 * thrown from render (as `lazy()` would) and retried on the next mount.
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
