// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the bottom-right overlay stack has to keep clear of, in viewport
// coordinates. The Live monitor is draggable and resizable and defaults to that
// same corner; the chat composer docks to the bottom of the same column once a
// thread has turns. Each publishes its box here while it is mounted.

import { useEffect, useMemo, useState } from "react";
import { create } from "zustand";

export type MonitorFrame = {
  left: number;
  top: number;
  right: number;
  bottom: number;
};

/**
 * Whose box this is. Identity matters twice over: reopening the monitor during
 * its exit animation leaves two panels mounted at once and the one on its way
 * out unmounts last, so its cleanup must clear only its own; and the monitor
 * and the composer are published side by side.
 */
export type MonitorFramePublisher = object;

interface MonitorFrameState {
  /** Every published box, kept apart. Deliberately not merged into one: the
   *  rectangle around a left-hand monitor and a bottom-right composer spans
   *  the empty space between them, and reading that as a single obstacle
   *  lifted the stack to its cap for a monitor nowhere near its column.
   *  `stackGeometry` folds them one at a time. */
  frames: ReadonlyMap<MonitorFramePublisher, MonitorFrame>;
  setFrame: (publisher: MonitorFramePublisher, frame: MonitorFrame) => void;
  /** Drops only this publisher's box; the others still count. */
  clearFrame: (publisher: MonitorFramePublisher) => void;
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
  frames: new Map(),
  // Written from a layout effect on every reconcile, so no-op writes must not
  // notify: the overlay stack re-renders on this.
  setFrame: (publisher, frame) =>
    set((state) => {
      if (sameFrame(state.frames.get(publisher) ?? null, frame)) return state;
      return { frames: new Map(state.frames).set(publisher, frame) };
    }),
  clearFrame: (publisher) =>
    set((state) => {
      if (!state.frames.has(publisher)) return state;
      const frames = new Map(state.frames);
      frames.delete(publisher);
      return { frames };
    }),
}));

// The corner stack's own inset, and the gap left between it and the monitor.
const STACK_INSET = 16;
const STACK_GAP = 8;
// Widest overlay the stack holds: the update banners, at max-w-[448px].
const STACK_WIDTH = 448;
// Never lift so far that the stack itself is pushed off the top.
const MIN_STACK_ROOM = 120;

/**
 * How far above the bottom edge the overlay stack must sit to clear the Live
 * monitor. The monitor defaults to the same corner, so without this the
 * update banners, the download panel and the loaded models card all land
 * underneath it.
 */
/** Whether the monitor overlaps the column the stack's overlays occupy. */
function inStackColumn(frame: MonitorFrame, viewportWidth: number): boolean {
  const columnLeft = viewportWidth - STACK_INSET - STACK_WIDTH;
  return (
    frame.right > columnLeft && frame.left < viewportWidth - STACK_INSET
  );
}

export function stackBottomInset(
  frame: MonitorFrame | null,
  viewportWidth: number,
  viewportHeight: number,
): number {
  if (!frame) return STACK_INSET;
  // Only dodge a monitor that is actually in the stack's column and low
  // enough to be in its way; one dragged elsewhere leaves the corner free.
  const lowEnough = frame.bottom > viewportHeight / 2;
  if (!(inStackColumn(frame, viewportWidth) && lowEnough)) return STACK_INSET;
  const lifted = viewportHeight - frame.top + STACK_GAP;
  return Math.max(
    STACK_INSET,
    Math.min(lifted, viewportHeight - MIN_STACK_ROOM),
  );
}

/**
 * How tall the stack may grow while sitting on `bottomInset`, keeping its own
 * margin at the top. Lifting the stack over the monitor shortens it by the same
 * amount, or a long download list plus the card runs off the top of the screen.
 *
 * A monitor parked high in the same column is not lifted over, because the free
 * space is underneath it. It still has to be dodged: the stack grows upwards
 * from the bottom, and a full download list plus the card is easily tall enough
 * to reach it. Cap the height at the gap below it instead.
 */
export function stackMaxHeight(
  frame: MonitorFrame | null,
  viewportWidth: number,
  viewportHeight: number,
  bottomInset: number,
): number {
  const ownMargin = viewportHeight - bottomInset - STACK_INSET;
  if (!frame || !inStackColumn(frame, viewportWidth)) return ownMargin;
  if (frame.bottom > viewportHeight / 2) return ownMargin;
  const belowMonitor = viewportHeight - bottomInset - frame.bottom - STACK_GAP;
  return Math.max(MIN_STACK_ROOM, Math.min(ownMargin, belowMonitor));
}

export type StackGeometry = { bottom: number; maxHeight: number };

/**
 * Where the overlay stack sits and how tall it may be, given everything it has
 * to keep clear of.
 *
 * Folded per box, never over their union: a tall monitor and the wide docked
 * composer share almost no area, and the rectangle around the pair covers most
 * of the viewport. Reading that as one obstacle pinned the stack to the top of
 * the screen and put it back over the monitor it was dodging. Each box asks for
 * the lift it needs; the stack takes the largest, and the shortest height.
 */
export function stackGeometry(
  frames: MonitorFrame | null | readonly MonitorFrame[],
  viewportWidth: number,
  viewportHeight: number,
): StackGeometry {
  const list = frames === null ? [] : Array.isArray(frames) ? frames : [frames];
  if (list.length === 0) {
    const bottom = stackBottomInset(null, viewportWidth, viewportHeight);
    return {
      bottom,
      maxHeight: stackMaxHeight(null, viewportWidth, viewportHeight, bottom),
    };
  }
  const bottom = Math.max(
    ...list.map((f) => stackBottomInset(f, viewportWidth, viewportHeight)),
  );
  return {
    bottom,
    maxHeight: Math.min(
      ...list.map((f) =>
        stackMaxHeight(f, viewportWidth, viewportHeight, bottom),
      ),
    ),
  };
}

/** `stackGeometry` in px, recomputed as the monitor moves or resizes. */
export function useStackGeometry(): StackGeometry {
  const frames = useMonitorFrameStore((state) => state.frames);
  // Every published box, not their union: see stackGeometry.
  const published = useMemo(() => [...frames.values()], [frames]);
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
  return stackGeometry(published, viewport.width, viewport.height);
}
