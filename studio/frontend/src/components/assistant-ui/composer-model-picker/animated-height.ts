// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";

const RESIZE_MS = 220;
const RESIZE_EASE = "cubic-bezier(0.22, 1, 0.36, 1)";

/** Glides a list to its new height when its rows change, so a panel opened upward does not jump. */
export function useAnimatedListHeight(el: HTMLElement | null): void {
  useEffect(() => {
    if (!el) {
      return;
    }
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    // The observer fires after the change, so the resting height is remembered.
    let shown = el.getBoundingClientRect().height;
    const release = () => {
      el.style.height = "";
      el.style.transition = "";
      shown = el.getBoundingClientRect().height;
    };
    const glide = () => {
      const from = el.style.height ? el.getBoundingClientRect().height : shown;
      el.style.transition = "none";
      el.style.height = "";
      const to = el.getBoundingClientRect().height;
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
    // A microtask, before paint: no frame shows the new rows at the old height.
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
