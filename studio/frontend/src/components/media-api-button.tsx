// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ApiIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { publicModelId } from "@/features/hub";
import { useSettingsDialogStore } from "@/features/settings";
import { useT } from "@/i18n";

export function MediaApiButton({
  kind,
  model,
}: {
  kind: "images" | "video";
  model: string | null;
}) {
  const t = useT();
  const label = t("settings.apiKeys.audioApi.useViaApi");
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label={label}
          onClick={() =>
            useSettingsDialogStore.getState().openMediaApi({
              kind,
              // /v1 names a model by its public id.
              model: model ? publicModelId(model) : null,
            })
          }
          className="flex h-[calc(34px*var(--ui-space-scale,1))] w-[calc(34px*var(--ui-space-scale,1))] shrink-0 items-center justify-center gap-1.5 rounded-full text-ui-13 @[84rem]:w-auto @[84rem]:px-2.5 font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <HugeiconsIcon icon={ApiIcon} className="size-4 shrink-0" />
          <span className="hidden whitespace-nowrap @[84rem]:inline">
            {label}
          </span>
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={6} className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}
