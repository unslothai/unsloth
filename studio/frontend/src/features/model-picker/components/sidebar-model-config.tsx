// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMemo } from "react";
import { modelConfigInstanceKey } from "../model-config/config-signature";
import {
  isOllamaLinkPath,
  isStandaloneGgufPath,
} from "../model-config/model-identity";
import type { PerModelConfig } from "../model-config/per-model-config";
import { ModelConfigPage } from "./model-config-page";
import type { ModelPickTarget } from "./model-selector/types";

interface SidebarModelConfigProps {
  modelId: string;
  ggufVariant: string | null;
  isGguf: boolean;
  nativeContextLength: number | null;
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

export function SidebarModelConfig({
  modelId,
  ggufVariant,
  isGguf,
  nativeContextLength,
  loadedContextLength,
  loadedConfig,
  onReload,
}: SidebarModelConfigProps) {
  // A standalone .gguf has no quant to choose between, but the loader labels it
  // from its filename (llama_cpp falls back to _extract_quant_label when the load
  // named no variant) and /status echoes that as gguf_variant. Keying settings by
  // it would write "<path>:Q4_K_M" while the Hub row, the picker and the backfill
  // all use the bare path, so the same file would carry two configs. The
  // auto-switch lookup reads the bare path first, so the sidebar's entry would
  // never be the one an API load applies. Same rule as settingsGgufVariantForRow.
  const settingsGgufVariant = isStandaloneGgufPath(modelId) ? null : ggufVariant;
  const target = useMemo<ModelPickTarget>(() => {
    const leaf = leafName(modelId);
    return {
      id: modelId,
      displayName: ggufVariant ? `${leaf} · ${ggufVariant}` : leaf,
      ggufVariant: settingsGgufVariant,
      isGguf,
      // An Ollama blob loads through a link dir the auto-switch resolver skips,
      // so its settings must not be mirrored as if the API could load it.
      apiLoadable: isGguf && !isOllamaLinkPath(modelId),
      meta: {
        source: "local",
        isLora: false,
        ggufVariant: settingsGgufVariant ?? undefined,
        isGguf,
        isDownloaded: true,
        contextLength: nativeContextLength,
      },
    };
  }, [modelId, ggufVariant, settingsGgufVariant, isGguf, nativeContextLength]);

  return (
    <ModelConfigPage
      key={modelConfigInstanceKey(modelId, settingsGgufVariant, loadedConfig)}
      target={target}
      onRun={onReload}
      loadedConfig={loadedConfig}
      loadedContextLength={loadedContextLength}
      variant="sidebar"
    />
  );
}
