// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DatasetSelector } from "@/features/dataset-picker";
import { useTrainingConfigStore } from "@/features/training";
import { useT } from "@/i18n";
import { type ReactNode, useEffect } from "react";
import { useShallow } from "zustand/react/shallow";
import { DatasetAdvancedSettingsSection } from "./dataset-advanced-settings-section";
import { DatasetSelectionSection } from "./dataset-selection";
import { DatasetUploadField } from "./dataset-upload";
import { FieldHint } from "./field-hint";
import { S3ConfigForm } from "./s3-config-form";
import { useDatasetUploads } from "./use-dataset-uploads";
import { useLocalDatasetInventory } from "./use-local-dataset-inventory";

function FieldLabel({
  children,
  hint,
  hintLabel,
}: {
  children: ReactNode;
  hint?: string;
  hintLabel?: string;
}) {
  return (
    <span className="flex items-center gap-1.5 text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
      {children}
      {hint ? <FieldHint text={hint} label={hintLabel ?? hint} /> : null}
    </span>
  );
}

export function DatasetPanel() {
  const t = useT();
  const {
    datasetSource,
    restoreBrowseDatasetSource,
    isVisionModel,
    isAudioModel,
    modelType,
  } = useTrainingConfigStore(
    useShallow((state) => ({
      datasetSource: state.datasetSource,
      restoreBrowseDatasetSource: state.restoreBrowseDatasetSource,
      isVisionModel: state.isVisionModel,
      isAudioModel: state.isAudioModel,
      modelType: state.modelType,
    })),
  );
  const localDatasetInventory = useLocalDatasetInventory(datasetSource);
  const uploads = useDatasetUploads();
  const effectiveModelType = modelType ?? "text";
  const isMultimodalModel =
    effectiveModelType === "vision" ||
    effectiveModelType === "audio" ||
    isVisionModel ||
    isAudioModel;

  useEffect(() => {
    if (datasetSource === "s3" && isMultimodalModel) {
      restoreBrowseDatasetSource();
    }
  }, [datasetSource, isMultimodalModel, restoreBrowseDatasetSource]);

  return (
    <div className="flex min-w-0 flex-col gap-5">
      {datasetSource === "s3" && <S3ConfigForm />}

      {datasetSource !== "s3" && (
        <div className="grid grid-cols-1 items-start gap-4 @xl/train-section:grid-cols-2 @xl/train-section:gap-5">
          <div className="flex flex-col gap-2">
            <FieldLabel
              hint={t("studio.wizard.datasetTooltip")}
              hintLabel={t("studio.wizard.datasetLabel")}
            >
              {t("studio.wizard.datasetLabel")}
            </FieldLabel>
            <DatasetSelector />
          </div>
          <DatasetUploadField uploads={uploads} />
        </div>
      )}

      <DatasetSelectionSection
        localDatasets={localDatasetInventory.rows}
        localInventorySettled={localDatasetInventory.settled}
        uploads={uploads}
      />
      <DatasetAdvancedSettingsSection />
    </div>
  );
}
