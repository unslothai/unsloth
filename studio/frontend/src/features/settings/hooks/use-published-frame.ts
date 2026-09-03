// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Publish an element's box to the overlay-frame store, so the bottom-right
// stack keeps clear of it. The Live monitor does this by hand inside its own
// layout effect; anything else that must not be covered uses this.

import { useEffect, useMemo } from "react";

import { useMonitorFrameStore } from "../stores/monitor-frame-store";

/**
 * Keep the corner stack off `element` for as long as it is mounted.
 *
 * The chat composer is the case this exists for: it docks to the bottom of the
 * viewport once a thread has turns, in the same column the stack occupies, and
 * the loaded models card is the first overlay there that is persistent rather
 * than transient. It sat over the Send button and swallowed the click.
 *
 * Re-measured on resize and through a ResizeObserver, because the composer
 * grows with its input and moves from centred to docked with no resize event.
 *
 * `coverable` says the stack may paint over this box rather than clip itself,
 * for the windows too short to do both. Off by default: the Live monitor's
 * controls have to stay clickable, and anything else that publishes here should
 * decide deliberately rather than inherit the composer's answer.
 */
export function usePublishedFrame(
  element: HTMLElement | null,
  { coverable = false }: { coverable?: boolean } = {},
): void {
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
      // A hidden composer measures 0x0; publishing that would pin the stack to
      // the top-left corner rather than leave it where it belongs.
      if (box.width === 0 && box.height === 0) {
        clearFrame(publisher);
        return;
      }
      setFrame(publisher, {
        left: box.left,
        top: box.top,
        right: box.right,
        bottom: box.bottom,
        coverable,
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
  }, [element, publisher, coverable, setFrame, clearFrame]);
}
