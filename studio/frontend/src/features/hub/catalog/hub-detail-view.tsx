// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ComponentProps, useEffect, useRef, useState } from "react";
import { ModelInspector } from "./model-inspector";

type InspectorProps = ComponentProps<typeof ModelInspector>;

export function HubDetailView({
  onBack,
  ...inspectorProps
}: InspectorProps & { onBack: () => void }) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [scrolled, setScrolled] = useState(false);

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
    <div className="flex min-h-0 flex-1 flex-col">
      <div
        ref={scrollRef}
        data-hub-scroll="true"
        className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto [overflow-anchor:none] [scrollbar-gutter:stable] [scrollbar-width:thin]"
      >
        <div
          className="hub-detail-bar sticky top-0 z-20"
          data-scrolled={scrolled || undefined}
        >
          <div className="mx-auto w-full max-w-[1100px] px-5 py-3 sm:px-8">
            <button
              type="button"
              onClick={onBack}
              className="-ml-2.5 inline-flex h-8 cursor-pointer select-none items-center gap-1.5 rounded-full px-2.5 text-[12.5px] font-medium text-muted-foreground transition-colors hover:bg-foreground/[0.05] hover:text-foreground dark:hover:bg-white/[0.06]"
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
        <div className="mx-auto w-full max-w-[1100px] px-5 pb-20 sm:px-8">
          <ModelInspector {...inspectorProps} />
        </div>
      </div>
    </div>
  );
}
