// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { GgufDownloadCard } from "./gguf-download-card";
import { SafetensorsDownloadCard } from "./safetensors-download-card";
import type { InventoryHint } from "./download-types";

export type { InventoryHint } from "./download-types";

export function DownloadSection({
  repoId,
  isGguf,
  isDownloaded,
  isPartial = false,
  isActive,
  activeQuant,
  isLoadingThisModel,
  gpuGb,
  systemRamGb,
  cachePath,
  onLoad,
  onUseInChat,
  onTrain,
  onChange,
}: {
  repoId: string;
  isGguf: boolean;
  isDownloaded: boolean;
  isPartial?: boolean;
  isActive: boolean;
  activeQuant: string | null;
  isLoadingThisModel: boolean;
  gpuGb?: number;
  systemRamGb?: number;
  cachePath?: string | null;
  onLoad: (opts: { ggufVariant?: string; expectedBytes?: number }) => void;
  onUseInChat?: () => void;
  onTrain?: () => void;
  onChange?: (hint?: InventoryHint) => void;
}) {
  if (isGguf) {
    return (
      <GgufDownloadCard
        repoId={repoId}
        isActive={isActive}
        activeQuant={activeQuant}
        isLoadingThisModel={isLoadingThisModel}
        gpuGb={gpuGb}
        systemRamGb={systemRamGb}
        cachePath={cachePath}
        onLoad={onLoad}
        onUseInChat={onUseInChat}
        onChange={onChange}
      />
    );
  }
  return (
    <SafetensorsDownloadCard
      repoId={repoId}
      isDownloaded={isDownloaded}
      isPartial={isPartial}
      isActive={isActive}
      isLoadingThisModel={isLoadingThisModel}
      cachePath={cachePath}
      onLoad={onLoad}
      onUseInChat={onUseInChat}
      onTrain={onTrain}
      onChange={onChange}
    />
  );
}
