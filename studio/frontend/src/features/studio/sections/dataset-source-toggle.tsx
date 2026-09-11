// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SegmentedControl } from "@/components/segmented-control";
import { useTrainingConfigStore } from "@/features/training";
import { useT } from "@/i18n";
import type { DatasetSource } from "@/types/training";
import { useShallow } from "zustand/react/shallow";

type DatasetSourceMode = "browse" | "s3";

export function DatasetSourceToggle({
  datasetSource,
  isVisionModel,
  restoreBrowseDatasetSource,
  selectS3Source,
}: {
  datasetSource: DatasetSource;
  isVisionModel: boolean;
  restoreBrowseDatasetSource: () => void;
  selectS3Source: () => void;
}) {
  const t = useT();

  if (isVisionModel) {
    return null;
  }

  const value: DatasetSourceMode = datasetSource === "s3" ? "s3" : "browse";
  const options = [
    { value: "browse", label: t("studio.wizard.sourceBrowse") },
    { value: "s3", label: "Amazon S3" },
  ] as const;

  return (
    <SegmentedControl
      value={value}
      options={options}
      onValueChange={(nextValue) => {
        if (nextValue === "s3") {
          selectS3Source();
          return;
        }
        restoreBrowseDatasetSource();
      }}
      ariaLabel={t("studio.dataset.sourceAriaLabel")}
      size="compact"
      className="@md/train-card:w-[200px]"
    />
  );
}

/** Store-connected variant for the Dataset section header. */
export function DatasetSourceToggleAction() {
  const {
    datasetSource,
    selectS3Source,
    restoreBrowseDatasetSource,
    isVisionModel,
    modelType,
  } = useTrainingConfigStore(
    useShallow((state) => ({
      datasetSource: state.datasetSource,
      selectS3Source: state.selectS3Source,
      restoreBrowseDatasetSource: state.restoreBrowseDatasetSource,
      isVisionModel: state.isVisionModel,
      modelType: state.modelType,
    })),
  );
  const effectiveModelType = modelType ?? "text";

  return (
    <DatasetSourceToggle
      datasetSource={datasetSource}
      // Vision only: audio S3 datasets load since #4539 (audio downloaded beside its manifest).
      isVisionModel={effectiveModelType === "vision" || isVisionModel}
      restoreBrowseDatasetSource={restoreBrowseDatasetSource}
      selectS3Source={selectS3Source}
    />
  );
}
