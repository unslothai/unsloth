// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

/** Info icon shown beside a field label. */
export function FieldHint({ text, label }: { text: string; label: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label={label}
          className="text-foreground/70 hover:text-foreground"
        >
          <HugeiconsIcon icon={InformationCircleIcon} className="size-3" />
        </button>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{text}</TooltipContent>
    </Tooltip>
  );
}
