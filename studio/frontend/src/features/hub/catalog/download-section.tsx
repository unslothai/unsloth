// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { AssetRuntime } from "./use-required-assets-download";
import type { MediaStudioPage } from "../hooks/use-media-companion-bytes";
import type { ModelInventoryFormat } from "../inventory";
import type { HubModelRunSelection } from "../lib/model-run-selection";
import { GgufDownloadCard } from "./gguf-download-card";
import { MlxPrecisionCard } from "./mlx-precision-card";
import { SafetensorsDownloadCard } from "./safetensors-download-card";

export function DownloadSection({
  repoId,
  isGguf,
  isDownloaded,
  isPartial = false,
  partialTransport = null,
  partialResumable = false,
  companionPrefetch = false,
  modelFormat,
  isActive,
  activeQuant,
  preferredGgufFile = null,

  preferredGgufFileIntent = 0,
  isLoadingThisModel,
  gpuGb,
  gpuCount,
  systemRamGb,
  unifiedLoadBudgetGb,
  cachePath,
  knownBytes,
  onRun,
  runPending = false,
  onChange,
  showMemoryBar = true,
  mediaPage,
  assetRuntime,
  isMlx = false,
  onRunPrecision,
}: {
  repoId: string;
  isGguf: boolean;
  isDownloaded: boolean;
  isPartial?: boolean;
  partialTransport?: string | null;
  partialResumable?: boolean;
  companionPrefetch?: boolean;
  modelFormat?: ModelInventoryFormat | null;
  isActive: boolean;
  activeQuant: string | null;
  preferredGgufFile?: string | null;

  preferredGgufFileIntent?: number;
  isLoadingThisModel: boolean;
  gpuGb?: number;
  gpuCount?: number;
  systemRamGb?: number;
  unifiedLoadBudgetGb?: number | null;
  cachePath?: string | null;
  knownBytes?: number | null;
  onRun?: (selection: HubModelRunSelection) => void;
  runPending?: boolean;
  onChange?: () => void;
  /** False for diffusion/audio/video GGUFs, which do not load through llama.cpp. */
  showMemoryBar?: boolean;
  mediaPage?: MediaStudioPage;
  assetRuntime?: AssetRuntime;
  isMlx?: boolean;
  onRunPrecision?: (selection: HubModelRunSelection) => void;
}) {
  if (isGguf || preferredGgufFile) {
    return (
      <GgufDownloadCard
        repoId={repoId}
        isActive={isActive}
        activeQuant={activeQuant}
        preferredFile={preferredGgufFile}
        preferredFileIntent={preferredGgufFileIntent}
        isLoadingThisModel={isLoadingThisModel}
        gpuGb={gpuGb}
        gpuCount={gpuCount}
        systemRamGb={systemRamGb}
        unifiedLoadBudgetGb={unifiedLoadBudgetGb}
        cachePath={cachePath}
        isPartial={isPartial}
        onRun={onRun}
        runPending={runPending}
        onChange={onChange}
        showMemoryBar={showMemoryBar}
        mediaPage={mediaPage}
      />
    );
  }
  const repoCard = (
    <SafetensorsDownloadCard
      assetRuntime={assetRuntime}
      repoId={repoId}
      isDownloaded={isDownloaded}
      isPartial={isPartial}
      partialTransport={partialTransport}
      partialResumable={partialResumable}
      companionPrefetch={companionPrefetch}
      modelFormat={modelFormat}
      isActive={isActive}
      isLoadingThisModel={isLoadingThisModel}
      cachePath={cachePath}
      knownBytes={knownBytes}
      onRun={onRun ? () => onRun({}) : undefined}
      runPending={runPending}
      onChange={onChange}
    />
  );
  if (!isMlx) return repoCard;
  return (
    <div className="flex w-full flex-col gap-2">
      <MlxPrecisionCard
        repoId={repoId}
        onRun={onRunPrecision}
        runPending={runPending}
      />
      {repoCard}
    </div>
  );
}
