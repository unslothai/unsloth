// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type RefObject, useLayoutEffect, useState } from "react";

function scrollParent(el: HTMLElement): HTMLElement | null {
  for (let node = el.parentElement; node; node = node.parentElement) {
    const { overflowY } = getComputedStyle(node);
    if (overflowY === "auto" || overflowY === "scroll") return node;
  }
  return null;
}

/**
 * Height that leaves room for `trailing` (and `gap`) above the bottom of the visible area.
 * It grows as the element scrolls up, until it reaches the top gap. Null below `query`.
 */
export function useFitToViewport(
  ref: RefObject<HTMLElement | null>,
  {
    trailing,
    gap = 24,
    min = 360,
    query = "(min-width: 1024px)",
  }: {
    trailing?: RefObject<HTMLElement | null>;
    gap?: number;
    min?: number;
    query?: string;
  } = {},
): number | null {
  const [height, setHeight] = useState<number | null>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const media = window.matchMedia(query);
    const scroller = scrollParent(el);
    let frame = 0;

    const measure = () => {
      frame = 0;
      if (!media.matches) {
        setHeight(null);
        return;
      }
      const view = scroller?.getBoundingClientRect() ?? {
        top: 0,
        bottom: window.innerHeight,
      };
      const top = Math.max(el.getBoundingClientRect().top, view.top + gap);
      const after = trailing?.current?.offsetHeight ?? 0;
      const reserved = after > 0 ? after + gap : 0;
      setHeight(Math.round(Math.max(min, view.bottom - top - gap - reserved)));
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(measure);
    };

    measure();
    const target: HTMLElement | Window = scroller ?? window;
    target.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    media.addEventListener("change", schedule);
    // Content above (cards loading, an error banner) moves the element.
    const observer = new ResizeObserver(schedule);
    if (el.parentElement) observer.observe(el.parentElement);
    if (trailing?.current) observer.observe(trailing.current);
    return () => {
      if (frame) cancelAnimationFrame(frame);
      target.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
      media.removeEventListener("change", schedule);
      observer.disconnect();
    };
  }, [ref, trailing, gap, min, query]);

  return height;
}
