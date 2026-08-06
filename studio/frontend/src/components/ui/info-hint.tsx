


import type { ReactNode } from "react";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

/** Small "i" affordance that reveals a styled tooltip on hover/focus. The
 *  standard inline help control across the settings UI. */
export function InfoHint({ children }: { children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
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
        side="top"
        align="center"
        sideOffset={6}
        collisionPadding={12}
        className="[&_span>svg]:hidden! duration-0 max-w-[240px] text-left"
      >
        {children}
      </TooltipContent>
    </Tooltip>
  );
}
