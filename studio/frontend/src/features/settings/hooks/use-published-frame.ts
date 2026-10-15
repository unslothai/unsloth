// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useMemo } from "react";

import { useMonitorFrameStore } from "../stores/monitor-frame-store";

/** Keep floating panels off `element` (the docked composer); ResizeObserver catches moves
 * that fire no resize event. */
export function usePublishedFrame(element: HTMLElement | null): void {
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
      // A hidden element measures 0x0 and must not push panels away.
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
