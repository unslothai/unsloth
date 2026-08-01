// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SegmentedControl } from "@/components/segmented-control";
import { useT } from "@/i18n";
import type { DatasetSource } from "@/types/training";

type DatasetSourceMode = "browse" | "s3";

export function DatasetSourceToggle({
  datasetSource,
  isMultimodalModel,
  restoreBrowseDatasetSource,
  selectS3Source,
}: {
  datasetSource: DatasetSource;
  isMultimodalModel: boolean;
  restoreBrowseDatasetSource: () => void;
  selectS3Source: () => void;
}) {
  const t = useT();

  if (isMultimodalModel) {
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
    />
  );
}
