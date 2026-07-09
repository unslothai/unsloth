// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { type ReactElement, useLayoutEffect, useRef, useState } from "react";

type InlineCategoryBadgesProps = {
  values: string[];
};

export function InlineCategoryBadges({
  values,
}: InlineCategoryBadgesProps): ReactElement {
  const containerRef = useRef<HTMLDivElement>(null);
  const [visibleCount, setVisibleCount] = useState(values.length);

  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const badges = Array.from(container.children) as HTMLElement[];
    if (badges.length === 0) {
      const id = requestAnimationFrame(() => setVisibleCount(0));
      return () => cancelAnimationFrame(id);
    }

    const containerWidth = container.clientWidth;
    // Reserve space for the "+N" badge (~36px)
    const overflowBadgeWidth = 36;
    let count = 0;
    let usedWidth = 0;

    for (const badge of badges) {
      const badgeWidth = badge.scrollWidth + 4; // 4px for gap
      if (usedWidth + badgeWidth > containerWidth - overflowBadgeWidth && count < badges.length - 1) {
        break;
      }
      if (usedWidth + badgeWidth > containerWidth) {
        break;
      }
      usedWidth += badgeWidth;
      count++;
    }

    const id = requestAnimationFrame(() => setVisibleCount(count || 1));
    return () => cancelAnimationFrame(id);
  }, [values]);

  if (values.length === 0) {
    return <p className="text-xs text-muted-foreground">No values</p>;
  }

  const overflow = values.length - visibleCount;

  return (
    <div className="relative">
      {/* Hidden measurer */}
      <div
        ref={containerRef}
        className="pointer-events-none invisible absolute inset-x-0 top-0 flex flex-nowrap gap-1"
        aria-hidden
      >
        {values.map((v, i) => (
          <Badge
            key={`m-${v}-${i}`}
            variant="secondary"
            className="corner-squircle h-4 shrink-0 px-1.5 text-[10px]"
          >
            {v}
          </Badge>
        ))}
      </div>
      {/* Visible badges */}
      <div className="flex flex-wrap gap-1">
        {values.slice(0, visibleCount).map((v, i) => (
          <Badge
            key={`${v}-${i}`}
            variant="secondary"
            className="corner-squircle h-4 px-1.5 text-[10px]"
          >
            {v}
          </Badge>
        ))}
        {overflow > 0 && (
          <Badge variant="outline" className="corner-squircle h-4 px-1.5 text-[10px]">
            +{overflow}
          </Badge>
        )}
      </div>
    </div>
  );
}
