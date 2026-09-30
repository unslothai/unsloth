// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useT } from "@/i18n";
import { InternetIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useBrowserStore } from "./store";

/** The chat header's button for opening the browser panel. Hidden in a new chat, and while the panel
 *  is open, since the panel closes itself. */
export function BrowserToggleButton() {
  const t = useT();
  const open = useBrowserStore((state) => state.open);
  const chatHasMessages = useBrowserStore((state) => state.chatHasMessages);
  const openPanel = useBrowserStore((state) => state.openPanel);
  if (open || !chatHasMessages) return null;
  const label = t("browser.show");
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          onClick={openPanel}
          aria-label={label}
          className="flex size-[calc(30px*var(--ui-space-scale,1))] cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:hover:text-white"
        >
          <HugeiconsIcon icon={InternetIcon} strokeWidth={1.75} className="size-icon" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={6} className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}
