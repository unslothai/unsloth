// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { type ReactNode, useLayoutEffect, useRef, useState } from "react";

/** Set on the header grid when space is short: the pill and link labels go compact. */
export const MEDIA_HEADER_TIGHT_ATTR = "data-media-header-tight";

// The loosest step that fits; an unmeasured step is tried, then measured.
function pickStep(need: (number | undefined)[], room: number): 0 | 1 | 2 {
  for (const step of [0, 1] as const) {
    const width = need[step];
    if (width === undefined || width <= room) {
      return step;
    }
  }
  return 2;
}

/**
 * The media header's right column. The grid gives it at least the links' own width, so a
 * tighter header moves the Create / Train pill left of centre rather than clipping them.
 * Tighter still, the pill and the link labels go compact; the links hide only when even
 * that cannot fit, and always on a phone or tablet.
 */
export function MediaHeaderActions({
  className,
  groupClassName,
  children,
}: {
  className?: string;
  groupClassName?: string;
  children: ReactNode;
}) {
  const slotRef = useRef<HTMLDivElement>(null);
  const groupRef = useRef<HTMLDivElement>(null);
  // Width the pill and links need at each step (full, compact), measured while shown.
  const need = useRef<(number | undefined)[]>([]);
  const [step, setStep] = useState<0 | 1 | 2>(0);
  const stepRef = useRef(step);
  stepRef.current = step;

  useLayoutEffect(() => {
    const slot = slotRef.current;
    const group = groupRef.current;
    const grid = slot?.parentElement;
    if (!(slot && group && grid)) {
      return;
    }
    grid.toggleAttribute(MEDIA_HEADER_TIGHT_ATTR, step > 0);
  }, [step]);

  useLayoutEffect(() => {
    const slot = slotRef.current;
    const group = groupRef.current;
    const grid = slot?.parentElement;
    if (!(slot && group && grid)) {
      return;
    }
    const centre = grid.querySelector<HTMLElement>(
      "[data-media-header-centre]",
    );
    const check = () => {
      const current = stepRef.current;
      if (current < 2 && group.offsetWidth > 0) {
        const style = getComputedStyle(slot);
        need.current[current] =
          (centre?.offsetWidth ?? 0) +
          group.scrollWidth +
          Number.parseFloat(style.paddingLeft) +
          Number.parseFloat(style.paddingRight);
      }
      const next = pickStep(need.current, grid.clientWidth);
      if (next !== current) {
        setStep(next);
      }
    };
    check();
    const observer = new ResizeObserver(check);
    observer.observe(grid);
    observer.observe(group);
    if (centre) {
      observer.observe(centre);
    }
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={slotRef}
      className={cn(
        "pointer-events-none col-start-3 row-start-1 flex min-w-0 items-start justify-end overflow-hidden pl-3 pt-[var(--studio-chat-header-padding-top,11px)] @max-[40rem]:hidden",
        className,
      )}
    >
      <div
        ref={groupRef}
        data-fits={step < 2}
        className={cn(
          "pointer-events-auto flex shrink-0 items-center gap-1",
          groupClassName,
          step === 2 && "hidden",
        )}
      >
        {children}
      </div>
    </div>
  );
}
