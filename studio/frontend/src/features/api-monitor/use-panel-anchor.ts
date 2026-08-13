// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Keeps the API monitor panel clear of the other floating panels, and publishes
// its own box so the notification stack keeps clear of it in turn. The geometry
// is in panel-placement.ts; this is the wiring.

import { useMonitorFrameStore } from "@/features/settings";
import { useCallback, useLayoutEffect, useMemo, useState } from "react";
import {
  clampPanelToViewport,
  isFullyCovered,
  type PanelAnchor,
  type PanelRect,
  type PanelSize,
  placeFloatingPanel,
} from "./panel-placement";

function sameAnchor(a: PanelAnchor | null, b: PanelAnchor): boolean {
  return a !== null && a.left === b.left && a.top === b.top;
}

export interface PanelPlacement {
  /** Where the panel should sit, or null until it has been measured. */
  anchor: PanelAnchor | null;
  /** Nothing of the panel is showing, so the layer has to give instead. */
  covered: boolean;
  /** Commit a position the user dragged the panel to, and report where it
   *  actually landed once clamped to the viewport. */
  place: (anchor: PanelAnchor) => PanelAnchor;
}

/**
 * Place the panel, and say whether it ended up hidden anyway.
 *
 * `frozen` is the user's placement winning. Once the panel has been dragged it
 * is where the user put it, overlapping or not: motion holds a drag as a
 * transform on top of this anchor, so moving the anchor afterwards would both
 * fight the user and shift the panel by the drag distance a second time.
 */
export function usePanelAnchor(
  element: HTMLElement | null,
  frozen: boolean,
  republishToken: unknown,
): PanelPlacement {
  // This panel's claim on the frame store, stable for the life of the component.
  const publisher = useMemo(() => ({}), []);
  const frames = useMonitorFrameStore((state) => state.frames);
  const [size, setSize] = useState<PanelSize | null>(null);
  const [viewport, setViewport] = useState(() => ({
    width: typeof window === "undefined" ? 0 : window.innerWidth,
    height: typeof window === "undefined" ? 0 : window.innerHeight,
  }));
  const [anchor, setAnchor] = useState<PanelAnchor | null>(null);

  // Everything published except this panel. Reading its own box back would make
  // it dodge itself, one step per frame, off the edge of the screen.
  const obstacles = useMemo<PanelRect[]>(() => {
    const others: PanelRect[] = [];
    for (const [owner, frame] of frames) {
      if (owner !== publisher) {
        others.push(frame);
      }
    }
    return others;
  }, [frames, publisher]);

  useLayoutEffect(() => {
    const onResize = () =>
      setViewport({ width: window.innerWidth, height: window.innerHeight });
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // The panel is content-sized and natively resizable, so its box is not known
  // ahead of time and does not stay put.
  useLayoutEffect(() => {
    if (!element) {
      setSize(null);
      return;
    }
    // offsetWidth/offsetHeight, not the bounding rect: the panel animates in
    // from scale 0.94, and a rect measured through that transform is 6% short.
    // Placed against it, the panel then overhangs the right edge of the screen
    // once the animation lands.
    const measure = () => {
      const width = element.offsetWidth;
      const height = element.offsetHeight;
      setSize((current) =>
        current && current.width === width && current.height === height
          ? current
          : { width, height },
      );
    };
    measure();
    const observer =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(measure);
    observer?.observe(element);
    return () => observer?.disconnect();
  }, [element]);

  const commit = useCallback((next: PanelAnchor) => {
    setAnchor((current) => (sameAnchor(current, next) ? current : next));
  }, []);

  useLayoutEffect(() => {
    if (!size || viewport.width === 0) {
      return;
    }
    // Placed by hand: keep it there, but keep it on screen. Shrinking the
    // window is the case -- the panel remembers nothing across reloads, so one
    // pushed off the edge is gone until the panel is closed and reopened.
    if (frozen) {
      setAnchor((current) =>
        current === null
          ? current
          : clampPanelToViewport(current, size, viewport),
      );
      return;
    }
    commit(placeFloatingPanel(size, obstacles, viewport));
  }, [frozen, size, obstacles, viewport, commit]);

  const place = useCallback(
    (next: PanelAnchor) => {
      const landed =
        !size || viewport.width === 0
          ? next
          : clampPanelToViewport(next, size, viewport);
      commit(landed);
      return landed;
    },
    [commit, size, viewport],
  );

  // Publish where it actually landed, so the notification stack steps over this
  // panel the same way it steps over the resource monitor. Measured rather than
  // derived from the anchor, because before the first anchor is committed the
  // panel is still sitting on its CSS corner.
  useLayoutEffect(() => {
    const { setFrame, clearFrame } = useMonitorFrameStore.getState();
    if (!element) {
      clearFrame(publisher);
      return;
    }
    const box = element.getBoundingClientRect();
    if (box.width === 0 && box.height === 0) {
      clearFrame(publisher);
      return;
    }
    setFrame(publisher, {
      left: box.left,
      top: box.top,
      right: box.right,
      bottom: box.bottom,
    });
  }, [element, anchor, size, publisher, republishToken]);

  // The panel unmounts on close and on every route change to the full page, so
  // a frame left behind would hold the stack up over an empty corner forever.
  useLayoutEffect(
    () => () => useMonitorFrameStore.getState().clearFrame(publisher),
    [publisher],
  );

  return {
    anchor,
    // Asked of the placed box rather than the measured one, so the answer does
    // not flicker while the panel animates in.
    covered:
      anchor !== null &&
      size !== null &&
      isFullyCovered(anchor, size, obstacles),
    place,
  };
}
