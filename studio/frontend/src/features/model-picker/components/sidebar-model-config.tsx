// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMemo } from "react";
import type { PerModelConfig } from "../model-config/per-model-config";
import { ModelConfigPage } from "./model-config-page";
import type { ModelPickTarget } from "./model-selector/types";

interface SidebarModelConfigProps {
  modelId: string;
  ggufVariant: string | null;
  isGguf: boolean;
  loadedContextLength: number | null;
  loadedConfig: PerModelConfig;
  onReload: (config: PerModelConfig) => void;
}

const TRAILING_SEPARATORS = /[\\/]+$/;

function leafName(id: string): string {
  const trimmed = id.replace(TRAILING_SEPARATORS, "");
  const separator = Math.max(
    trimmed.lastIndexOf("/"),
    trimmed.lastIndexOf("\\"),
  );
  return separator >= 0 ? trimmed.slice(separator + 1) : trimmed;
}

function configSignature(config: PerModelConfig): string {
  return [
    config.customContextLength ?? "",
    config.maxSeqLength ?? "",
    config.kvCacheDtype ?? "",
    config.speculativeType ?? "",
    config.specDraftNMax ?? "",
    config.tensorParallel ? "1" : "0",
    config.chatTemplateOverride?.length ?? "",
  ].join("|");
}

export function SidebarModelConfig({
  modelId,
  ggufVariant,
  isGguf,
  loadedContextLength,
  loadedConfig,
  onReload,
}: SidebarModelConfigProps) {
  const target = useMemo<ModelPickTarget>(() => {
    const leaf = leafName(modelId);
    return {
      id: modelId,
      displayName: ggufVariant ? `${leaf} · ${ggufVariant}` : leaf,
      ggufVariant,
      isGguf,
      meta: {
        source: "local",
        isLora: false,
        ggufVariant: ggufVariant ?? undefined,
        isGguf,
        isDownloaded: true,
        contextLength: null,
      },
    };
  }, [modelId, ggufVariant, isGguf]);

  return (
    <ModelConfigPage
      key={`${modelId}::${ggufVariant ?? ""}::${configSignature(loadedConfig)}`}
      target={target}
      onRun={onReload}
      loadedConfig={loadedConfig}
      loadedContextLength={loadedContextLength}
      variant="sidebar"
    />
  );
}
