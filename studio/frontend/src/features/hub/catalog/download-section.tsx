// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelInventoryFormat } from "../inventory";
import { GgufDownloadCard } from "./gguf-download-card";
import { SafetensorsDownloadCard } from "./safetensors-download-card";

export function DownloadSection({
  repoId,
  isGguf,
  isDownloaded,
  isPartial = false,
  partialTransport = null,
  partialResumable = false,
  modelFormat,
  isActive,
  activeQuant,
  preferredGgufFile = null,

  preferredGgufFileIntent = 0,
  isLoadingThisModel,
  gpuGb,
  systemRamGb,
  cachePath,
  knownBytes,
  onChange,
  showMemoryBar = true,
}: {
  repoId: string;
  isGguf: boolean;
  isDownloaded: boolean;
  isPartial?: boolean;
  partialTransport?: string | null;
  partialResumable?: boolean;
  modelFormat?: ModelInventoryFormat | null;
  isActive: boolean;
  activeQuant: string | null;
  preferredGgufFile?: string | null;

  preferredGgufFileIntent?: number;
  isLoadingThisModel: boolean;
  gpuGb?: number;
  systemRamGb?: number;
  cachePath?: string | null;
  knownBytes?: number | null;
  onChange?: () => void;
  /** False for diffusion / audio / video GGUFs, which do not load through
   *  llama.cpp and so have nothing the KV estimator can say about them. */
  showMemoryBar?: boolean;
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
        systemRamGb={systemRamGb}
        cachePath={cachePath}
        isPartial={isPartial}
        onChange={onChange}
        showMemoryBar={showMemoryBar}
      />
    );
  }
  return (
    <SafetensorsDownloadCard
      repoId={repoId}
      isDownloaded={isDownloaded}
      isPartial={isPartial}
      partialTransport={partialTransport}
      partialResumable={partialResumable}
      modelFormat={modelFormat}
      isActive={isActive}
      isLoadingThisModel={isLoadingThisModel}
      cachePath={cachePath}
      knownBytes={knownBytes}
      onChange={onChange}
    />
  );
}
