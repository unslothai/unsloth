// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { Globe02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useBrowserStore } from "./store";

/** The chat header's button for showing and hiding the browser panel. */
export function BrowserToggleButton() {
  const t = useT();
  const open = useBrowserStore((state) => state.open);
  const togglePanel = useBrowserStore((state) => state.togglePanel);
  const label = open ? t("browser.hide") : t("browser.show");
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          onClick={togglePanel}
          aria-label={label}
          aria-pressed={open}
          className={cn(
            "flex size-[calc(30px*var(--ui-space-scale,1))] cursor-pointer items-center justify-center rounded-[10px] transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
            open
              ? "bg-primary/10 text-primary hover:bg-primary/15"
              : "text-nav-fg hover:bg-nav-surface-hover hover:text-black dark:hover:text-white",
          )}
        >
          <HugeiconsIcon icon={Globe02Icon} strokeWidth={1.75} className="size-icon" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={6} className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}
