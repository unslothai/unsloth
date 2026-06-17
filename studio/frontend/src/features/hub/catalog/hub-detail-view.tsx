// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { cn } from "@/lib/utils";
import { type ComponentProps, useEffect, useRef, useState } from "react";
import { ModelInspector } from "./model-inspector";

type InspectorProps = ComponentProps<typeof ModelInspector>;

export function HubDetailView({
  onBack,
  compact = false,
  ...inspectorProps
}: InspectorProps & { onBack: () => void; compact?: boolean }) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [scrolled, setScrolled] = useState(false);
  // The split master-detail pane is much narrower than the full-page overlay, so
  // the readme/inspector reads better with a tighter measure and less gutter.
  const measure = compact
    ? "mx-auto w-full max-w-[860px] px-5 sm:px-5"
    : "mx-auto w-full max-w-[1100px] px-5 sm:px-8";

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => {
      const next = el.scrollTop > 0;
      setScrolled((current) => (current === next ? current : next));
    };
    onScroll();
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      {/* Same top scroll fade as the left column, so scrolling the readme reads
          consistently. The sticky back-bar (when shown) sits above and hides it. */}
      <div
        aria-hidden="true"
        data-scrolled={scrolled || undefined}
        className="hub-scroll-fade pointer-events-none absolute inset-x-0 top-0 z-10 h-7"
      />
      <div
        ref={scrollRef}
        data-hub-scroll="true"
        // Slight right margin nudges the scrollbar in from the pane's edge so it
        // sits a touch closer to the readme content instead of hugging the far edge.
        className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto [overflow-anchor:none] mr-2 [scrollbar-gutter:stable] [scrollbar-width:thin]"
      >
        <div
          className={cn(
            "hub-detail-bar sticky top-0 z-20",
            // In split view on large screens the master list sits alongside, so
            // "Back to Hub" is redundant. Keep it only for the overlay (small
            // screens / full-page mode) where the list is hidden.
            compact && "lg:hidden",
          )}
          data-scrolled={scrolled || undefined}
        >
          <div className={`${measure} py-3`}>
            <button
              type="button"
              onClick={onBack}
              className="-ml-1.5 inline-flex h-8 cursor-pointer select-none items-center gap-1.5 rounded-full pl-1.5 pr-2.5 text-[12.5px] font-medium text-muted-foreground transition-colors hover:bg-foreground/[0.05] hover:text-foreground dark:hover:bg-white/[0.06]"
            >
              <HugeiconsIcon
                icon={ArrowLeft01Icon}
                strokeWidth={1.75}
                className="size-3.5"
              />
              Back to Hub
            </button>
          </div>
        </div>
        <div className={cn(measure, "pb-20", compact && "lg:pt-4")}>
          <ModelInspector {...inspectorProps} />
        </div>
      </div>
    </div>
  );
}
