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
  // Split pane is narrower than the full-page overlay; tighter measure reads better.
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
      {/* Same top scroll fade as the left column. The sticky back-bar, when
          shown, sits above and hides it. */}
      <div
        aria-hidden="true"
        data-scrolled={scrolled || undefined}
        className="hub-scroll-fade pointer-events-none absolute inset-x-0 top-0 z-10 h-7"
      />
      <div
        ref={scrollRef}
        data-hub-scroll="true"
        // Mirror the catalog's gutter strategy so the centered column lines up
        // with the top bar: the full overlay reserves an equal both-edges gutter
        // to stay symmetric; split pins a narrow pane so it nudges the scrollbar
        // in from the edge with a right gutter only.
        className={cn(
          "min-h-0 flex-1 overflow-x-hidden overflow-y-auto [overflow-anchor:none] [scrollbar-width:thin]",
          compact
            ? "mr-2 [scrollbar-gutter:stable]"
            : "[scrollbar-gutter:stable_both-edges]",
        )}
      >
        <div
          className={cn(
            "hub-detail-bar sticky top-0 z-20",
            // In split view on large screens the list sits alongside, so "Back
            // to Hub" is redundant; keep it only for the overlay where it's hidden.
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
