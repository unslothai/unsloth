// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HEADER_ICON_BUTTON } from "@/components/media-page-link";
import { datasetDisplayName } from "@/components/resource-picker/dataset-display-name";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  trainingApiRequestFromPayload,
  useSettingsDialogStore,
} from "@/features/settings";
import { useTrainingConfigStore } from "@/features/training";
import { buildTrainingStartPayload } from "@/features/training/api/mappers";
import { useT } from "@/i18n";
import { ApiIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";

/** Opens Settings > API on the Training example, filled from the form as Start would send it. */
export function TrainingApiButton(): ReactElement {
  const t = useT();
  const label = t("settings.apiKeys.audioApi.useViaApi");
  const openTrainingApi = () => {
    const config = useTrainingConfigStore.getState();
    const payload = buildTrainingStartPayload(config, null);
    const uploaded = config.datasetSource === "upload" ? config.uploadedFile : null;
    useSettingsDialogStore
      .getState()
      .openTrainingApi(
        trainingApiRequestFromPayload(
          payload,
          uploaded ? datasetDisplayName(uploaded) : null,
        ),
      );
  };
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label={label}
          onClick={openTrainingApi}
          className={HEADER_ICON_BUTTON}
        >
          <HugeiconsIcon icon={ApiIcon} strokeWidth={1.75} className="size-icon" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={6} className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}
