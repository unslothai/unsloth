// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useRef } from "react";

/**
 * Callback ref for scroll containers inside modal scroll locks. The lock may
 * cancel native wheel scrolling, so apply the delta before it reaches the
 * portaled dialog boundary.
 */
export function useWheelScrollRef<T extends HTMLElement>() {
  const detachRef = useRef<(() => void) | null>(null);

  return useCallback((node: T | null) => {
    detachRef.current?.();
    detachRef.current = null;
    if (!node) {
      return;
    }

    const onWheel = (event: WheelEvent) => {
      if (event.ctrlKey || node.scrollHeight <= node.clientHeight) {
        return;
      }
      const multiplier =
        event.deltaMode === WheelEvent.DOM_DELTA_LINE
          ? 16
          : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
            ? node.clientHeight
            : 1;
      const delta = event.deltaY * multiplier;
      if (delta === 0) {
        return;
      }
      node.scrollTop += delta;
      event.preventDefault();
      event.stopPropagation();
    };

    node.addEventListener("wheel", onWheel, { passive: false });
    detachRef.current = () => node.removeEventListener("wheel", onWheel);
  }, []);
}
