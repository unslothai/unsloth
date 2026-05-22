// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactNode } from "react";

const DEFAULT_TRIGGER_CLASS =
  "inline-flex size-4 shrink-0 cursor-help items-center justify-center rounded-full text-muted-foreground/60 transition-colors hover:text-foreground";
const DEFAULT_CONTENT_CLASS = "tooltip-compact max-w-64";

/** Small "?" trigger that reveals `children` in a left-side tooltip. */
export function InfoHint({
  children,
  triggerClassName,
  contentClassName,
}: {
  children: ReactNode;
  triggerClassName?: string;
  contentClassName?: string;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label="More info"
          className={triggerClassName ?? DEFAULT_TRIGGER_CLASS}
        >
          <HugeiconsIcon
            icon={InformationCircleIcon}
            strokeWidth={1.75}
            className="size-3.5"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="left"
        sideOffset={8}
        className={contentClassName ?? DEFAULT_CONTENT_CLASS}
      >
        {children}
      </TooltipContent>
    </Tooltip>
  );
}
