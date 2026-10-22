// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCornerCardFrameStore } from "@/features/settings/stores/monitor-frame-store";
import { subscribeCornerInsets } from "@/lib/toast-offset";
import { useEffect } from "react";

export function usePublishedCornerFrame(element: HTMLElement | null): void {
  useEffect(() => {
    const { setFrame } = useCornerCardFrameStore.getState();
    if (!element) {
      setFrame(null);
      return;
    }
    const measure = () => {
      const box = element.getBoundingClientRect();
      setFrame(
        box.width > 0 && box.height > 0
          ? {
              left: box.left,
              top: box.top,
              right: box.right,
              bottom: box.bottom,
            }
          : null,
      );
    };
    measure();
    window.addEventListener("resize", measure);
    // Run settings and the composer lift move the rail by its padding, which resizes nothing.
    const unsubscribe = subscribeCornerInsets(measure);
    const observer =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(measure);
    observer?.observe(element);
    return () => {
      window.removeEventListener("resize", measure);
      unsubscribe();
      observer?.disconnect();
      setFrame(null);
    };
  }, [element]);
}
