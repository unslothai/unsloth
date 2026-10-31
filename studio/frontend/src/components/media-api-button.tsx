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
import { HEADER_ICON_BUTTON } from "@/components/media-page-link";
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
          className={HEADER_ICON_BUTTON}
        >
          <HugeiconsIcon
            icon={ApiIcon}
            strokeWidth={1.75}
            className="size-icon"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={6} className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}
