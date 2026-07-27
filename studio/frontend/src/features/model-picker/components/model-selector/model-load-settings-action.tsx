// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Settings02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

export function ModelLoadSettingsAction({
  ariaLabel,
  onConfigure,
  className,
}: {
  ariaLabel: string;
  onConfigure: () => void;
  className?: string;
}) {
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onConfigure();
          }}
          aria-label={ariaLabel}
          className={cn(
            "shrink-0 rounded-md p-1 text-muted-foreground/60 transition-colors hover:bg-black/5 hover:text-foreground dark:hover:bg-white/10",
            className,
          )}
        >
          <HugeiconsIcon
            icon={Settings02Icon}
            strokeWidth={1.75}
            className="size-3"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        Configure run settings before loading model
      </TooltipContent>
    </Tooltip>
  );
}
