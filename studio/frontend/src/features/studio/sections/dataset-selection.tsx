// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { datasetDisplayName } from "@/features/dataset-picker";
import {
  type LocalDatasetInfo,
  hfApiToken,
  useHfTokenStore,
  useOnlineStatus,
} from "@/features/hub";
import {
  HfDatasetSubsetSplitSelectors,
  cacheLocalPathMatchesSelection,
  isHuggingFaceDatasetSelected,
  shouldClearMissingLocalDatasetSelection,
  useDatasetPreviewDialogStore,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { FileAttachmentIcon, ViewIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { formatUpdatedDate } from "./dataset-panel-helpers";
import { EvaluationDatasetUpload } from "./dataset-upload";
import type { DatasetUploads } from "./use-dataset-uploads";

function DatasetSubsetConfiguration({
  datasetName,
  enabled,
}: {
  datasetName: string | null;
  enabled: boolean;
}) {
  const hfToken = useHfTokenStore((state) => state.token);
  const online = useOnlineStatus();
  const settings = useTrainingConfigStore(
    useShallow((state) => ({
      datasetEvalSplit: state.datasetEvalSplit,
      datasetKnownCached: state.datasetKnownCached,
      datasetLocalPath: state.datasetLocalPath,
      datasetStreaming: state.datasetStreaming,
      datasetSplit: state.datasetSplit,
      datasetSubset: state.datasetSubset,
      setDatasetEvalSplit: state.setDatasetEvalSplit,
      setManualDatasetOptionsValid: state.setManualDatasetOptionsValid,
      markManualDatasetOptionsEdited: state.markManualDatasetOptionsEdited,
      setDatasetSplit: state.setDatasetSplit,
      setDatasetSubset: state.setDatasetSubset,
    })),
  );
  return (
    <HfDatasetSubsetSplitSelectors
      variant="studio"
      enabled={enabled}
      datasetName={datasetName}
      accessToken={hfApiToken(hfToken)}
      localPath={settings.datasetLocalPath}
      online={online}
      preferLocalCache={
        settings.datasetKnownCached && !settings.datasetStreaming
      }
      datasetSubset={settings.datasetSubset}
      setDatasetSubset={settings.setDatasetSubset}
      datasetSplit={settings.datasetSplit}
      setDatasetSplit={settings.setDatasetSplit}
      datasetEvalSplit={settings.datasetEvalSplit}
      setDatasetEvalSplit={settings.setDatasetEvalSplit}
      datasetStreaming={settings.datasetStreaming}
      setManualDatasetOptionsValid={settings.setManualDatasetOptionsValid}
      markManualDatasetOptionsEdited={settings.markManualDatasetOptionsEdited}
    />
  );
}

function MetadataRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-2 rounded-md bg-background/60 px-2 py-1.5">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium text-foreground">{value}</span>
    </div>
  );
}

function LocalDatasetMetadata({ dataset }: { dataset: LocalDatasetInfo }) {
  const t = useT();
  const metadata = dataset.metadata ?? null;
  const columns = metadata?.columns ?? [];
  const rows = dataset.rows ?? metadata?.actual_num_records ?? null;
  return (
    <div className="rounded-lg border bg-muted/20 px-3.5 py-3">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <p className="text-xs font-medium text-muted-foreground">
            {t("studio.dataset.localDatasetMetadata")}
          </p>
          <p className="text-ui-10 text-muted-foreground/80">
            {t("studio.dataset.dataRecipeOutput")}
          </p>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
        <MetadataRow
          label={t("studio.dataset.rows")}
          value={typeof rows === "number" ? rows.toLocaleString() : "--"}
        />
        <MetadataRow
          label={t("studio.dataset.columns")}
          value={columns.length > 0 ? String(columns.length) : "--"}
        />
        <MetadataRow
          label={t("studio.dataset.batches")}
          value={
            typeof metadata?.num_completed_batches === "number" &&
            typeof metadata?.total_num_batches === "number"
              ? `${metadata.num_completed_batches}/${metadata.total_num_batches}`
              : "--"
          }
        />
        <MetadataRow
          label={t("studio.dataset.updated")}
          value={formatUpdatedDate(dataset.updated_at ?? null)}
        />
      </div>
    </div>
  );
}

function DatasetConfiguration({
  selectedLocalDataset,
}: {
  selectedLocalDataset: LocalDatasetInfo | null;
}) {
  const datasetSource = useTrainingConfigStore((state) => state.datasetSource);
  const dataset = useTrainingConfigStore((state) => state.dataset);
  const uploadedFile = useTrainingConfigStore((state) => state.uploadedFile);

  if (datasetSource === "s3") {
    return null;
  }
  if (isHuggingFaceDatasetSelected(datasetSource, dataset)) {
    return <DatasetSubsetConfiguration enabled={true} datasetName={dataset} />;
  }
  const selectedDatasetName =
    datasetSource === "upload" ? uploadedFile : dataset;
  if (!selectedDatasetName) {
    return <DatasetSubsetConfiguration enabled={false} datasetName={null} />;
  }
  if (datasetSource !== "upload" || selectedLocalDataset?.source !== "recipe") {
    return null;
  }
  return <LocalDatasetMetadata dataset={selectedLocalDataset} />;
}

function SelectedDatasetCard({
  selectedLocalDataset,
}: {
  selectedLocalDataset: LocalDatasetInfo | null;
}) {
  const t = useT();
  const openPreview = useDatasetPreviewDialogStore(
    (state) => state.openPreview,
  );
  const settings = useTrainingConfigStore(
    useShallow((state) => ({
      dataset: state.dataset,
      datasetSource: state.datasetSource,
      datasetSplit: state.datasetSplit,
      datasetSubset: state.datasetSubset,
      selectHfDataset: state.selectHfDataset,
      selectLocalDataset: state.selectLocalDataset,
      uploadedFile: state.uploadedFile,
    })),
  );
  const selectedDatasetName =
    settings.datasetSource === "upload"
      ? settings.uploadedFile
      : settings.dataset;
  if (settings.datasetSource === "s3" || !selectedDatasetName) {
    return null;
  }
  const selectedLocalRows =
    selectedLocalDataset?.rows ??
    selectedLocalDataset?.metadata?.actual_num_records ??
    null;
  const clearSelection =
    settings.datasetSource === "upload"
      ? settings.selectLocalDataset
      : settings.selectHfDataset;

  return (
    <div className="flex flex-col items-stretch gap-3 rounded-lg border bg-muted/40 px-3.5 py-3 @md/train-section:flex-row @md/train-section:items-center">
      <div className="flex min-w-0 items-center gap-3">
        <div className="shrink-0 rounded-md bg-indigo-500/10 p-1.5">
          <HugeiconsIcon
            icon={FileAttachmentIcon}
            className="size-4 text-indigo-500"
          />
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate font-mono text-sm font-medium">
            {settings.datasetSource === "upload"
              ? (selectedLocalDataset?.label ??
                datasetDisplayName(selectedDatasetName))
              : selectedDatasetName}
          </p>
          <p className="text-ui-10 text-muted-foreground">
            {settings.datasetSource === "upload" ? (
              <>
                {t("studio.dataset.localDataset")}
                {selectedLocalRows != null
                  ? t("studio.dataset.localDatasetRows", {
                      count: selectedLocalRows.toLocaleString(),
                    })
                  : ""}
              </>
            ) : (
              <>
                {t("studio.dataset.huggingFaceDataset")}
                {settings.datasetSubset && ` / ${settings.datasetSubset}`}
                {settings.datasetSplit && ` / ${settings.datasetSplit}`}
              </>
            )}
          </p>
        </div>
      </div>
      <div className="flex flex-wrap items-center justify-end gap-1 @md/train-section:ml-auto @md/train-section:flex-nowrap">
        <Button
          variant="ghost"
          size="sm"
          className="shrink-0 text-xs"
          onClick={() => openPreview()}
        >
          <HugeiconsIcon icon={ViewIcon} className="size-3.5" />
          {t("studio.dataset.viewDataset")}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="shrink-0 text-xs"
          onClick={() => clearSelection(null)}
        >
          {t("studio.dataset.clear")}
        </Button>
      </div>
    </div>
  );
}

export function DatasetSelectionSection({
  localDatasets,
  localInventorySettled,
  uploads,
}: {
  localDatasets: LocalDatasetInfo[];
  localInventorySettled: boolean;
  uploads: DatasetUploads;
}) {
  const { datasetSource, selectLocalDataset, uploadedFile } =
    useTrainingConfigStore(
      useShallow((state) => ({
        datasetSource: state.datasetSource,
        selectLocalDataset: state.selectLocalDataset,
        uploadedFile: state.uploadedFile,
      })),
    );
  const selectedLocalDataset = useMemo(() => {
    if (!uploadedFile) {
      return null;
    }
    return (
      localDatasets.find((item) =>
        cacheLocalPathMatchesSelection(item.path, uploadedFile),
      ) ?? null
    );
  }, [localDatasets, uploadedFile]);

  useEffect(() => {
    if (
      shouldClearMissingLocalDatasetSelection({
        source: datasetSource,
        selectedPath: uploadedFile,
        inventorySettled: localInventorySettled,
        inventoryMatchFound: selectedLocalDataset !== null,
      })
    ) {
      selectLocalDataset(null);
    }
  }, [
    datasetSource,
    localInventorySettled,
    selectedLocalDataset,
    selectLocalDataset,
    uploadedFile,
  ]);

  return (
    <>
      <SelectedDatasetCard selectedLocalDataset={selectedLocalDataset} />
      <DatasetConfiguration selectedLocalDataset={selectedLocalDataset} />
      <EvaluationDatasetUpload uploads={uploads} />
    </>
  );
}
