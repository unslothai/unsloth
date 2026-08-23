// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CachedInventoryCopy, ModelInventoryFormat } from "../inventory";
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
  preferredGgufFile = null,
  preferredGgufFileIntent = 0,
  gpuGb,
  systemRamGb,
  cachePath,
  activeCache,
  cacheCopies,
  knownBytes,
  onChange,
}: {
  repoId: string;
  isGguf: boolean;
  isDownloaded: boolean;
  isPartial?: boolean;
  partialTransport?: string | null;
  partialResumable?: boolean;
  modelFormat?: ModelInventoryFormat | null;
  preferredGgufFile?: string | null;
  preferredGgufFileIntent?: number;
  gpuGb?: number;
  systemRamGb?: number;
  cachePath?: string | null;
  activeCache?: boolean | null;
  cacheCopies?: CachedInventoryCopy[];
  knownBytes?: number | null;
  onChange?: () => void;
}) {
  if (isGguf || preferredGgufFile) {
    return (
      <GgufDownloadCard
        repoId={repoId}
        preferredFile={preferredGgufFile}
        preferredFileIntent={preferredGgufFileIntent}
        gpuGb={gpuGb}
        systemRamGb={systemRamGb}
        cachePath={cachePath}
        activeCache={activeCache}
        cacheCopies={cacheCopies}
        isPartial={isPartial}
        onChange={onChange}
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
      cachePath={cachePath}
      activeCache={activeCache}
      cacheCopies={cacheCopies}
      knownBytes={knownBytes}
      onChange={onChange}
    />
  );
}
