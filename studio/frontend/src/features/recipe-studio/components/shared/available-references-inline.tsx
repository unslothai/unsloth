// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { type ReactElement, useLayoutEffect, useRef, useState } from "react";
import type { AvailableVariableEntry } from "../../utils/variables";

type AvailableReferencesInlineProps = {
  entries: AvailableVariableEntry[];
};

const MAX_ROWS = 2;

export function AvailableReferencesInline({
  entries,
}: AvailableReferencesInlineProps): ReactElement | null {
  const [expanded, setExpanded] = useState(false);
  const [collapsedCount, setCollapsedCount] = useState(entries.length);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const measureRefs = useRef<Array<HTMLSpanElement | null>>([]);

  useLayoutEffect(() => {
    if (expanded) {
      return;
    }
    const wrapper = wrapperRef.current;
    const items = measureRefs.current.filter(
      (node): node is HTMLSpanElement => Boolean(node),
    );
    if (!(wrapper && items.length > 0)) {
      setCollapsedCount(entries.length);
      return;
    }

    const compute = () => {
      const rowTops: number[] = [];
      let cutoff = items.length;
      for (let i = 0; i < items.length; i += 1) {
        const top = items[i].offsetTop;
        if (!rowTops.some((value) => Math.abs(value - top) <= 1)) {
          rowTops.push(top);
        }
        if (rowTops.length > MAX_ROWS) {
          cutoff = i;
          break;
        }
      }
      if (cutoff < items.length) {
        cutoff = Math.max(0, cutoff - 1);
      }
      setCollapsedCount(cutoff);
    };

    compute();
    const observer = new ResizeObserver(compute);
    observer.observe(wrapper);
    return () => observer.disconnect();
  }, [entries.length, expanded]);

  if (entries.length === 0) {
    return null;
  }

  const shown = expanded ? entries : entries.slice(0, collapsedCount);
  const hiddenCount = Math.max(0, entries.length - shown.length);

  return (
    <div className="space-y-1">
      <p className="text-[10px] font-medium text-muted-foreground">
        Available references
      </p>
      <div ref={wrapperRef} className="relative">
        {!expanded && (
          <div className="invisible pointer-events-none absolute inset-0 -z-10">
            <div className="flex flex-wrap gap-1">
              {entries.map((entry, index) => (
                <Badge
                  // biome-ignore lint/suspicious/noArrayIndexKey: static measurement mirror
                  key={`${entry.source}:${entry.name}:${index}`}
                  ref={(node) => {
                    measureRefs.current[index] = node;
                  }}
                  variant="secondary"
                  className={
                    entry.source === "seed"
                      ? "corner-squircle h-4 border-blue-500/25 bg-blue-500/10 px-1.5 font-mono text-[10px] text-blue-700 dark:text-blue-300"
                      : "corner-squircle h-4 px-1.5 font-mono text-[10px]"
                  }
                >
                  {entry.name}
                </Badge>
              ))}
            </div>
          </div>
        )}
        <div className="flex flex-wrap gap-1">
          {shown.map((entry) => (
            <Badge
              key={`${entry.source}:${entry.name}`}
              variant="secondary"
              className={
                entry.source === "seed"
                  ? "corner-squircle h-4 border-blue-500/25 bg-blue-500/10 px-1.5 font-mono text-[10px] text-blue-700 dark:text-blue-300"
                  : "corner-squircle h-4 px-1.5 font-mono text-[10px]"
              }
            >
              {entry.name}
            </Badge>
          ))}
          {!expanded && hiddenCount > 0 && (
            <button
              type="button"
              className="corner-squircle h-4 px-1.5 text-[10px] text-muted-foreground hover:text-foreground"
              onClick={() => setExpanded(true)}
            >
              +{hiddenCount} more
            </button>
          )}
          {expanded && collapsedCount < entries.length && (
            <button
              type="button"
              className="corner-squircle h-4 px-1.5 text-[10px] text-muted-foreground hover:text-foreground"
              onClick={() => setExpanded(false)}
            >
              Show less
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
