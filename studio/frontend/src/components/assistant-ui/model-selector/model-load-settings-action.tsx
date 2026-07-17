// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { cn } from "@/lib/utils";
import { Settings02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

/** Gear button on a downloaded quant row. Stages the model into the Run
 * settings sidebar (always, regardless of the Load-on-selection toggle) so the
 * user can set load options, then click Load model. */
export function ModelLoadSettingsAction({
  ariaLabel,
  repoId,
  quant,
  maxContext,
}: {
  ariaLabel: string;
  repoId: string;
  quant: string;
  maxContext?: number | null;
}) {
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            useChatRuntimeStore.getState().stageModel({
              id: repoId,
              ggufVariant: quant,
              isDownloaded: true,
              contextLength: maxContext ?? null,
            });
          }}
          aria-label={ariaLabel}
          className={cn(
            "shrink-0 rounded-md p-1 text-muted-foreground/60 transition-colors hover:bg-muted hover:text-foreground",
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
