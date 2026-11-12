// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";

const RESIZE_MS = 220;
const RESIZE_EASE = "cubic-bezier(0.22, 1, 0.36, 1)";

/**
 * Glides a list between its old and new height when its rows change, as Zeron's picker does, so a
 * panel opened above its trigger does not jump its top edge on every tab or search.
 */
export function useAnimatedListHeight(el: HTMLElement | null): void {
  useEffect(() => {
    if (!el) {
      return;
    }
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    // The observer fires after the new rows are in, so at rest the old height must be remembered.
    let shown = el.getBoundingClientRect().height;
    const release = () => {
      el.style.height = "";
      el.style.transition = "";
      shown = el.getBoundingClientRect().height;
    };
    const glide = () => {
      // Mid-glide the pinned height is the one on screen, so a second change continues from it.
      const from = el.style.height ? el.getBoundingClientRect().height : shown;
      el.style.transition = "none";
      el.style.height = "";
      const to = el.getBoundingClientRect().height;
      // The OS setting, or Studio's own Reduce motion (Settings > Appearance).
      const still =
        reduce?.matches ||
        document.documentElement.classList.contains("force-reduced-motion");
      if (still || Math.abs(to - from) < 1) {
        release();
        return;
      }
      el.style.height = `${from}px`;
      el.getBoundingClientRect();
      el.style.transition = `height ${RESIZE_MS}ms ${RESIZE_EASE}`;
      el.style.height = `${to}px`;
      shown = to;
    };
    // A window resize moves the resting height without a mutation.
    const resize = new ResizeObserver(() => {
      if (!el.style.height) {
        shown = el.getBoundingClientRect().height;
      }
    });
    resize.observe(el);
    const onEnd = (event: TransitionEvent) => {
      if (event.target === el && event.propertyName === "height") {
        release();
      }
    };
    // Runs before paint, so the new rows never show at the old height for a frame.
    const observer = new MutationObserver(glide);
    observer.observe(el, { childList: true, subtree: true });
    el.addEventListener("transitionend", onEnd);
    return () => {
      observer.disconnect();
      resize.disconnect();
      el.removeEventListener("transitionend", onEnd);
      release();
    };
  }, [el]);
}
