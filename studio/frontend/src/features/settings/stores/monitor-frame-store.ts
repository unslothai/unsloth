// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Where the Live monitor currently sits, in viewport coordinates. It is
// draggable and resizable and defaults to the bottom-right corner, which is
// also where the overlay stack lives, so the stack needs its real box to keep
// clear of it. Null whenever the monitor is closed.

import { useEffect, useState } from "react";
import { create } from "zustand";

export type MonitorFrame = {
  left: number;
  top: number;
  right: number;
  bottom: number;
};

interface MonitorFrameState {
  frame: MonitorFrame | null;
  setFrame: (frame: MonitorFrame | null) => void;
}

function sameFrame(a: MonitorFrame | null, b: MonitorFrame | null): boolean {
  if (a === null || b === null) return a === b;
  return (
    a.left === b.left &&
    a.top === b.top &&
    a.right === b.right &&
    a.bottom === b.bottom
  );
}

export const useMonitorFrameStore = create<MonitorFrameState>((set) => ({
  frame: null,
  // Written from a layout effect on every reconcile, so no-op writes must not
  // notify: the overlay stack re-renders on this.
  setFrame: (frame) =>
    set((state) => (sameFrame(state.frame, frame) ? state : { frame })),
}));

// The corner stack's own inset, and the gap left between it and the monitor.
const STACK_INSET = 16;
const STACK_GAP = 8;
// Widest overlay the stack holds (the download panel and update banners).
const STACK_WIDTH = 400;
// Never lift so far that the stack itself is pushed off the top.
const MIN_STACK_ROOM = 120;

/**
 * How far above the bottom edge the overlay stack must sit to clear the Live
 * monitor. The monitor defaults to the same corner, so without this the
 * update banners, the download panel and the loaded models card all land
 * underneath it.
 */
export function stackBottomInset(
  frame: MonitorFrame | null,
  viewportWidth: number,
  viewportHeight: number,
): number {
  if (!frame) return STACK_INSET;
  // Only dodge a monitor that is actually in the stack's column and low
  // enough to be in its way; one dragged elsewhere leaves the corner free.
  const columnLeft = viewportWidth - STACK_INSET - STACK_WIDTH;
  const inColumn =
    frame.right > columnLeft && frame.left < viewportWidth - STACK_INSET;
  const lowEnough = frame.bottom > viewportHeight / 2;
  if (!(inColumn && lowEnough)) return STACK_INSET;
  const lifted = viewportHeight - frame.top + STACK_GAP;
  return Math.max(
    STACK_INSET,
    Math.min(lifted, viewportHeight - MIN_STACK_ROOM),
  );
}

/** `stackBottomInset` in px, recomputed as the monitor moves or resizes. */
export function useStackBottomInset(): number {
  const frame = useMonitorFrameStore((state) => state.frame);
  const [viewport, setViewport] = useState(() => ({
    width: typeof window === "undefined" ? 0 : window.innerWidth,
    height: typeof window === "undefined" ? 0 : window.innerHeight,
  }));
  useEffect(() => {
    const onResize = () =>
      setViewport({ width: window.innerWidth, height: window.innerHeight });
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  return stackBottomInset(frame, viewport.width, viewport.height);
}
