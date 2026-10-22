// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

const FIT_GAP = 24;

/** Caps an element's height to what's left of its scroll area below its top, so a card
 * ends above the window's edge and scrolls inside instead. Re-measures on scroll, resize
 * and layout shifts above it; pair with a max-height transition for the grow and shrink. */
export function useFitToViewport(
  min = 320,
): [(el: HTMLElement | null) => void, number | null] {
  const [el, setEl] = useState<HTMLElement | null>(null);
  const [height, setHeight] = useState<number | null>(null);
  useEffect(() => {
    if (!el) return;
    let scroller: HTMLElement | null = el.parentElement;
    while (scroller) {
      const oy = getComputedStyle(scroller).overflowY;
      if (oy === "auto" || oy === "scroll") break;
      scroller = scroller.parentElement;
    }
    let frame = 0;
    const measure = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const bottom = scroller
          ? scroller.getBoundingClientRect().bottom
          : window.innerHeight;
        const top = Math.max(el.getBoundingClientRect().top, 0);
        setHeight(Math.max(min, Math.floor(bottom - top - FIT_GAP)));
      });
    };
    measure();
    const target: HTMLElement | Window = scroller ?? window;
    target.addEventListener("scroll", measure, { passive: true });
    window.addEventListener("resize", measure);
    // Cards appearing above (the live run card, a notice) move this one's top.
    const shifts = new ResizeObserver(measure);
    shifts.observe(scroller?.firstElementChild ?? document.body);
    return () => {
      cancelAnimationFrame(frame);
      target.removeEventListener("scroll", measure);
      window.removeEventListener("resize", measure);
      shifts.disconnect();
    };
  }, [el, min]);
  return [setEl, height];
}
