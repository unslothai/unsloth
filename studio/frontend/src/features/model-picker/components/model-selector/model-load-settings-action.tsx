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
import { useHasSavedRunSettings } from "../../model-config/saved-run-settings";
import type { ModelPickTarget } from "./types";

export function ModelLoadSettingsAction({
  ariaLabel,
  onConfigure,
  className,
  tooltip = "Configure run settings before loading model",
  savedFor,
}: {
  ariaLabel: string;
  onConfigure: () => void;
  className?: string;
  /** Hover copy. The default describes a local load, so Connected rows override it. */
  tooltip?: string;
  savedFor?: ModelPickTarget;
}) {
  const saved = useHasSavedRunSettings(savedFor ?? null);
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
          data-run-settings-saved={saved || undefined}
          className={cn(
            "relative",
            // Fixed box, not padding around the glyph, so this and the dots menu hover as one size. Callers
            // can still size it up.
            "flex size-5 shrink-0 items-center justify-center rounded-md text-muted-foreground/80 transition-colors hover:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] hover:text-foreground dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]",
            className,
          )}
        >
          {/* A size down from the dots: the gear fills its whole box. */}
          <HugeiconsIcon
            icon={Settings02Icon}
            strokeWidth={1.75}
            className="size-3"
          />
          {saved && (
            <span
              aria-hidden={true}
              className="absolute top-0.5 right-0.5 size-1.5 rounded-full bg-primary"
            />
          )}
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        {saved ? "Edit saved run settings" : tooltip}
      </TooltipContent>
    </Tooltip>
  );
}
