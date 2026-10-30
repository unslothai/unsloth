// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactNode, useState } from "react";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

/** Small i-icon tooltip trigger, the standard inline help control across the settings UI. */
export function InfoHint({
  children,
  side = "top",
  hoverOnly = false,
}: {
  children: ReactNode;
  side?: "top" | "right" | "bottom" | "left";
  /** Hover only, so a click cannot leave it open; pair it with sr-only text. */
  hoverOnly?: boolean;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <Tooltip
      {...(hoverOnly ? { open: hovered, onOpenChange: () => {} } : {})}
    >
      <TooltipTrigger asChild>
        <button
          onPointerEnter={hoverOnly ? () => setHovered(true) : undefined}
          onPointerLeave={hoverOnly ? () => setHovered(false) : undefined}
          type="button"
          data-slot="info-hint"
          aria-label="More info"
          className="inline-flex size-3 shrink-0 cursor-help items-center justify-center rounded-full text-muted-foreground/70 transition-colors hover:text-[#383835] dark:hover:text-[#e8e8e8] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          <HugeiconsIcon
            icon={InformationCircleIcon}
            strokeWidth={1.75}
            className="size-2.5"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent
        side={side}
        align="center"
        sideOffset={6}
        collisionPadding={12}
        className="[&_span>svg]:hidden! duration-0 max-w-[calc(240px*var(--ui-space-scale,1))] text-left"
      >
        {children}
      </TooltipContent>
    </Tooltip>
  );
}
