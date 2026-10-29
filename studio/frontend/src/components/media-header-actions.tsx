// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { type ReactNode, useLayoutEffect, useRef, useState } from "react";

/**
 * The media header's right column. Its links show only when they fit beside the centred
 * Create / Train pill; otherwise they hide as a group, so nothing clips or crowds the pill.
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
  const [fits, setFits] = useState(true);

  useLayoutEffect(() => {
    const slot = slotRef.current;
    const group = groupRef.current;
    if (!(slot && group)) {
      return;
    }
    const check = () => {
      const style = getComputedStyle(slot);
      const room =
        slot.clientWidth -
        Number.parseFloat(style.paddingLeft) -
        Number.parseFloat(style.paddingRight);
      setFits(group.scrollWidth <= room);
    };
    check();
    const observer = new ResizeObserver(check);
    observer.observe(slot);
    observer.observe(group);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={slotRef}
      className={cn(
        // A phone keeps only the pill, even when the links would squeeze in.
        "pointer-events-none col-start-3 row-start-1 flex min-w-0 items-start justify-end overflow-hidden pt-[var(--studio-chat-header-padding-top,11px)] @max-[30rem]:hidden",
        className,
      )}
    >
      <div
        ref={groupRef}
        data-fits={fits}
        inert={!fits || undefined}
        className={cn(
          "pointer-events-auto flex shrink-0 items-center gap-1",
          groupClassName,
          !fits && "invisible",
        )}
      >
        {children}
      </div>
    </div>
  );
}
