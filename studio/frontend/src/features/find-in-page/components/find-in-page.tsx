// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isSurfaceBackgrounded, useShortcut } from "@/features/settings";
import { Suspense, lazy, useEffect } from "react";
import { FIND_SCOPE_ATTRIBUTE } from "../lib/find-attributes.ts";
import { useFindInPageStore } from "../stores/find-in-page-store.ts";

// Lazy, so the index, the observer and the highlights are not on the first screen. Nobody has
// searched yet when the shell mounts, and the chord is the only thing that has to be ready.
const FindBar = lazy(() => import("./find-bar.tsx"));

/**
 * The find bar, and the chord that raises it.
 *
 * Split in two: this component is always mounted and holds nothing but the shortcut, while
 * `FindBar` owns the index, the observer and the highlights and exists only while the bar is open.
 * A Studio nobody is searching runs one keydown listener and no engine.
 */
export function FindInPage({ enabled = true }: { enabled?: boolean }) {
  const open = useFindInPageStore((state) => state.open);
  const requestFocus = useFindInPageStore((state) => state.requestFocus);
  const reset = useFindInPageStore((state) => state.reset);

  // Fetch the bar's chunk once the shell is idle. Lazy keeps it off the first screen, which is the
  // point, but a reader who does press the chord should not wait for a round trip to see a search
  // box. Idle, so it never competes with the first paint, and the import cache makes it free after.
  useEffect(() => {
    const warm = () => void import("./find-bar.tsx");
    const idle = globalThis as {
      requestIdleCallback?: (cb: () => void) => number;
      cancelIdleCallback?: (handle: number) => void;
    };
    if (typeof idle.requestIdleCallback === "function") {
      const handle = idle.requestIdleCallback(warm);
      return () => idle.cancelIdleCallback?.(handle);
    }
    // Safari below 18.4 has no idle callback; a timer after the first paint is close enough.
    const timer = setTimeout(warm, 1000);
    return () => clearTimeout(timer);
  }, []);

  // Leaving the shell for good, which on the web means signing out: the store is module-global and
  // keeps the query across a close, so without this the next person to sign in in the same tab is
  // handed the last one's search. Unmount, not `enabled`, which a dialog also turns off.
  useEffect(() => reset, [reset]);

  // Not `skipInTextFields`: the chord has to work from the composer, and pressing it inside the
  // find field is how a find bar is asked to start over.
  useShortcut("findInPage", requestFocus, {
    enabled,
    // Every modal, not just Settings: Radix marks the shell `aria-hidden`/`inert` while one is up,
    // so a bar behind it is unreachable. As `claims`, not a return from the handler, which runs
    // after the event is prevented: declining there would leave the chord dead and native find
    // suppressed.
    claims: () => !isSurfaceBackgrounded(`[${FIND_SCOPE_ATTRIBUTE}]`),
  });

  if (!enabled || !open) return null;
  // No fallback: the bar appearing a frame later is the whole cost, and a spinner where a search
  // box is about to be would be worse than nothing.
  return (
    <Suspense fallback={null}>
      <FindBar />
    </Suspense>
  );
}
