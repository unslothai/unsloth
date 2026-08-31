// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Publish an element's box to the overlay-frame store, so the API monitor
// panel opens somewhere it is not. The Live monitor does this by hand inside
// its own layout effect; anything else uses this.

import { useEffect, useMemo } from "react";

import { useMonitorFrameStore } from "../stores/monitor-frame-store";

/**
 * Keep the floating panels off `element` for as long as it is mounted.
 *
 * The chat composer is the case this exists for: it docks to the bottom of the
 * viewport once a thread has turns, in the same corner the API monitor panel
 * opens in, and a panel landing on it covers Send.
 *
 * Re-measured on resize and through a ResizeObserver, because the composer
 * grows with its input and moves from centred to docked with no resize event.
 */
export function usePublishedFrame(element: HTMLElement | null): void {
  // This caller's claim on the store, stable for the life of the component.
  const publisher = useMemo(() => ({}), []);
  const setFrame = useMonitorFrameStore((state) => state.setFrame);
  const clearFrame = useMonitorFrameStore((state) => state.clearFrame);

  useEffect(() => {
    if (!element) {
      clearFrame(publisher);
      return;
    }
    const measure = () => {
      const box = element.getBoundingClientRect();
      // A hidden composer measures 0x0; publishing that would push the panels
      // out of the top-left corner for an obstacle that is not there.
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
    };
    measure();
    window.addEventListener("resize", measure);
    const observer =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(measure);
    observer?.observe(element);
    return () => {
      window.removeEventListener("resize", measure);
      observer?.disconnect();
      clearFrame(publisher);
    };
  }, [element, publisher, setFrame, clearFrame]);
}
